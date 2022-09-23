
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import dim


class double_conv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.AvgPool2d(2),
			double_conv(in_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


class up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(up, self).__init__()
		self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
						diffY // 2, diffY - diffY//2))

		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1)
			
	def forward(self, x):
		x = self.conv(x)
		return x


class ED(nn.Module):
	def __init__(self, in_dim=12, out_dim=3, first_k=3, dilation=1):
		super(ED, self).__init__()
		self.e1 = nn.Sequential(
			nn.Conv2d(in_dim, 32, kernel_size=first_k, dilation=dilation, padding=(first_k-1)//2*dilation),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, dilation=dilation, padding=dilation),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, dilation=dilation, padding=dilation),
			nn.LeakyReLU(0.1, True)
		)

		self.e2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, dilation=dilation, padding=dilation),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True)
		)

		self.e3 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True)
		)

		self.d1 = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),	# Conv is same
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True)
		)

		self.d2 = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True)
		)

		self.d3_1 = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
		)

		self.d3_2 = nn.Sequential(
			nn.Conv2d(192, out_dim, kernel_size=3, padding=1),
			nn.LeakyReLU(0.1, True),
		)

	def forward(self, x):
		x1 = self.e1(x)
		x2 = self.e2(x1)
		x3 = self.e3(x2)

		x4 = self.d1(x3)
		x5 = self.d2(x4)
		x6 = self.d3_1(x5)

		x_cat = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

		out = self.d3_2(x_cat)

		return out
		

class GFN(nn.Module):
	def __init__(self):
		super(GFN, self).__init__()
		self.pool4x = nn.MaxPool2d(4)		# why max pool
		self.pool2x = nn.MaxPool2d(2)

		self.net_s1 = ED(12, 3, 3, 1)
		self.up1_2 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)

		self.net_s2 = ED(15, 3, 5, 2)
		self.up2_3 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)

		self.up2x = nn.Upsample(scale_factor=2)
		self.up4x = nn.Upsample(scale_factor=4)

		self.net_s3 = ED(15, 3, 7, 3)

	def WB(self, x):
		rgb = F.adaptive_avg_pool2d(x, 1)
		gray = torch.mean(rgb, dim=1, keepdim=True)
		scale = gray / (rgb + 1e-3)
		x = x * scale
		x = torch.clamp(x, 0, 1)
		return x

	def CE(self, x):
		x_a = torch.mean(x, dim=(1,2,3), keepdim=True)
		x = 2 * (0.5 + x_a) * (x - x_a)
		return x

	def GC(self, x):
		x = x * x	# torch.pow(x, 2.5) lead to nan
		return x

	def forward(self, x):
		x = x * 0.5 + 0.5   # to [0, 1]

		N, _, H, W = x.size()
		padding_h = math.ceil(H / 4) * 4 - H
		padding_w = math.ceil(W / 4) * 4 - W

		x_pad = F.pad(x, (0, padding_w, 0, padding_h), mode='reflect')

		x_wb = self.WB(x_pad)
		x_ce = self.CE(x_pad)
		x_gc = self.GC(x_pad)

		x_in_1x = torch.cat([x_pad, x_wb, x_ce, x_gc], dim=1)
		x_in_2x = self.pool2x(x_in_1x)
		x_in_4x = self.pool4x(x_in_1x)

		w1 = self.net_s1(x_in_4x)
		out1 = w1[:, [0], :, :] * x_in_4x[:, 3:6, :, :] + \
			   w1[:, [1], :, :] * x_in_4x[:, 6:9, :, :] + \
			   w1[:, [2], :, :] * x_in_4x[:, 9:12, :, :]

		out1_up = self.up1_2(out1)

		w2 = self.net_s2(torch.cat([out1_up, x_in_2x], dim=1))
		out2 = w2[:, [0], :, :] * x_in_2x[:, 3:6, :, :] + \
			   w2[:, [1], :, :] * x_in_2x[:, 6:9, :, :] + \
			   w2[:, [2], :, :] * x_in_2x[:, 9:12, :, :]

		out2_up = self.up2_3(out2)

		w3 = self.net_s3(torch.cat([out2_up, x_in_1x], dim=1))
		out3 = w3[:, [0], :, :] * x_in_1x[:, 3:6, :, :] + \
			   w3[:, [1], :, :] * x_in_1x[:, 6:9, :, :] + \
			   w3[:, [2], :, :] * x_in_1x[:, 9:12, :, :]

		out = out3[:, :, :H ,:W]	# deep supervised is not implemented

		out = out * 2 - 1     # to [-1, 1]

		return out
