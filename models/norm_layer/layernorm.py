import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_


class LayerNorm(nn.Module):
	def __init__(self, dim, eps=1e-5):
		super(LayerNorm, self).__init__()
		self.eps = eps

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

	def forward(self, input):
		mean = torch.mean(input, dim=1, keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=1, keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		out = normalized_input * self.weight + self.bias
		return out


class LayerNormP(nn.Module):
	def __init__(self, dim, eps=1e-5):
		super(LayerNormP, self).__init__()
		self.eps = eps

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

	def forward(self, input):
		# mean = torch.mean(input, dim=(1,2,3), keepdim=True)
		# std = torch.sqrt((input - mean).pow(2).mean(dim=(1,2,3), keepdim=True) + self.eps)

		normalized_input = F.layer_norm(input, input.size()[1:], eps=self.eps)

		out = normalized_input * self.weight + self.bias

		return out


class LayerNormR(nn.Module):
	r"""Rescaling LayerNorm"""
	def __init__(self, dim, eps=1e-5, detach_grad=False):
		super(LayerNormR, self).__init__()
		self.eps = eps
		self.detach_grad = detach_grad

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

		self.meta1 = nn.Conv2d(1, dim, 1)
		self.meta2 = nn.Conv2d(1, dim, 1)

		trunc_normal_(self.meta1.weight, std=.02)
		nn.init.constant_(self.meta1.bias, 1)

		trunc_normal_(self.meta2.weight, std=.02)
		nn.init.constant_(self.meta2.bias, 0)

	def forward(self, input):
		mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		if self.detach_grad:
			rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		else:
			rescale, rebias = self.meta1(std), self.meta2(mean)

		out = normalized_input * self.weight + self.bias
		return out, rescale, rebias

