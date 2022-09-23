import torch
import torch.nn as nn
import torch.nn.functional as F


class Maxout(nn.Hardtanh):
	def __init__(self, groups):
		super(Maxout, self).__init__()
		self.groups = groups

	def forward(self, x):
		x = x.view(x.shape[0], self.groups, x.shape[1]//self.groups, x.shape[2], x.shape[3])
		x = torch.max(x, dim=2, keepdim=True)[0]
		out = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
		return out
		

class BRelu(nn.Hardtanh):
	def __init__(self):
		super(BRelu, self).__init__()
		
	def forward(self, x):
		x = torch.max(x, torch.zeros_like(x))
		x = torch.min(x, torch.ones_like(x))
		return x


class DehazeNet(nn.Module):
	def __init__(self):
		super(DehazeNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, padding_mode='reflect')
		self.maxout = Maxout(4)
		self.conv2 = nn.Conv2d(4, 16, kernel_size=3, padding=1, padding_mode='reflect')
		self.conv3 = nn.Conv2d(4, 16, kernel_size=5, padding=2, padding_mode='reflect')
		self.conv4 = nn.Conv2d(4, 16, kernel_size=7, padding=3, padding_mode='reflect')
		self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
		self.conv5 = nn.Conv2d(48, 1, kernel_size=7, padding=3, padding_mode='reflect')		# 6->7
		self.sigmoid = nn.Sigmoid()															# brelu->sigmoid
		for _, m in self.named_modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight, mean=0, std=0.001)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = x * 0.5 + 0.5   # to [0, 1]

		N, _, H, W = x.size()

		out = self.conv1(x)
		out = self.maxout(out)
		out1 = self.conv2(out)
		out2 = self.conv3(out)
		out3 = self.conv4(out)
		out = torch.cat((out1, out2, out3), dim=1)
		out = self.maxpool(out)
		out = self.conv5(out)
		t = self.sigmoid(out)		

		## unstable processing
		# k = int(H * W * 1e-3)
		# t0 = torch.topk(t.flatten(2), k, dim=2)[0][:, :, -1].view(N, 1, 1, 1)
		# filtered = torch.ge(t, t0).float() * x
		# a = torch.max(filtered.flatten(2), dim=2)[0].view(N, 3, 1, 1)
		# J = (x - a.detach()) / t + a.detach()

		J = (x - 1) / t + 1

		output = J * 2 - 1     # to [-1, 1]

		return output
