import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FDU(nn.Module):
	def __init__(self, channel, reduction=16):
		super(FDU, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel*2, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c*2, 1, 1)

		y1, y2 = torch.chunk(y, 2, dim=1)

		return x * y1 + y2


class PFDB(nn.Module):
	def __init__(self, planes, reduction=16):
		super(PFDB, self).__init__()
		self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

		self.fdu = FDU(planes, reduction)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.relu(out)
		out = self.conv2(out)

		out = self.fdu(out)

		out += residual

		return out


class PFDN(nn.Module):
	def __init__(self):
		super(PFDN, self).__init__()
		self.blocks = 9

		self.encoder = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.ReLU(True),
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.ReLU(True),
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
			nn.ReLU(True)
		)

		pfdbs = []
		for _ in range(self.blocks):
			pfdbs += [PFDB(256)]

		self.pfdbs = nn.Sequential(*pfdbs)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 2, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 64, 2, stride=2),
			nn.ReLU(True),
			nn.Conv2d(64, 3, kernel_size=3, padding=1),
			nn.Tanh()
		)

	def forward(self, x):
		_, _, H, W = x.size()

		# poor network design
		padding_h = math.ceil(H / 16) * 16 - H
		padding_w = math.ceil(W / 16) * 16 - W

		x_pad = F.pad(x, (0, padding_w, 0, padding_h), mode='reflect')

		out  = self.encoder(x_pad)
		out = self.pfdbs(out)
		out = self.decoder(out)[:, :, :H ,:W]

		return out
