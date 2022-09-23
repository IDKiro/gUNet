import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['PSNRLoss', 'CharbonnierLoss']


class PSNRLoss(nn.Module):
	def __init__(self, eps=1e-8):
		super(PSNRLoss, self).__init__()
		self.eps = eps

	def forward(self, output, target):
		diff = output - target
		mse = (diff * diff).mean(dim=(1, 2, 3))
		loss = torch.log(mse + self.eps).mean()
		return loss


class CharbonnierLoss(nn.Module):
	def __init__(self, eps=1e-3):
		super(CharbonnierLoss, self).__init__()
		self.eps2 = eps ** 2

	def forward(self, output, target):
		diff = output - target
		loss = torch.mean(torch.sqrt((diff * diff) + self.eps2))
		return loss
