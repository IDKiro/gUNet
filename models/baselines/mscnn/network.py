import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCNN(nn.Module):
	def __init__(self):
		super(MSCNN, self).__init__()
		self.CS = nn.Sequential(
			nn.Conv2d(3, 5, kernel_size=11, padding=5),
			nn.MaxPool2d(2, stride=2),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(5, 5, kernel_size=9, padding=4),
			nn.MaxPool2d(2, stride=2),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(5, 10, kernel_size=7, padding=3),
			nn.MaxPool2d(2, stride=2),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(10, 1, kernel_size=1),
			nn.Sigmoid()
		)

		self.FS1 = nn.Sequential(
			nn.Conv2d(3, 4, kernel_size=7, padding=3),
			nn.MaxPool2d(2, stride=2),
			nn.Upsample(scale_factor=2)
		)

		self.FS2 = nn.Sequential(
			nn.Conv2d(5, 5, kernel_size=5, padding=2),
			nn.MaxPool2d(2, stride=2),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(5, 10, kernel_size=3, padding=1),
			nn.MaxPool2d(2, stride=2),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(10, 1, kernel_size=1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = x * 0.5 + 0.5   # to [0, 1]

		N, _, H, W = x.size()

		# poor network design
		padding_h = math.ceil(H / 32) * 32 - H
		padding_w = math.ceil(W / 32) * 32 - W

		x_pad = F.pad(x, (0, padding_w, 0, padding_h), mode='reflect')

		out1 = self.CS(x_pad)
		out2 = self.FS1(x_pad)

		out = torch.cat([out1, out2], dim=1)
		t = self.FS2(out)[:, :, :H ,:W]

		## unstable processing
		# k = int(H * W * 1e-3)
		# t0 = torch.topk(t.flatten(2), k, dim=2)[0][:, :, -1].view(N, 1, 1, 1)
		# filtered = torch.ge(t, t0).float() * x
		# a = torch.max(filtered.flatten(2), dim=2)[0].view(N, 3, 1, 1)
		# J = (x - a.detach()) / t + a.detach()

		J = (x - 1) / t + 1

		output = J * 2 - 1     # to [-1, 1]

		return output
