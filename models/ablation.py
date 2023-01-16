import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from .norm_layer import *
from .gunet import gUNet, SKFusion, ConvLayer


class SumFusion(nn.Module):
	def __init__(self, dim):
		super(SumFusion, self).__init__()

	def forward(self, in_feats):
		return sum(in_feats)


class CatFusion(nn.Module):
	def __init__(self, dim, height=2):
		super(CatFusion, self).__init__()
		self.height = height
		self.conv = nn.Conv2d(dim * height, dim, kernel_size=1)

	def forward(self, in_feats):
		in_feats = torch.cat(in_feats, dim=1)
		out = self.conv(in_feats)
		return out


class CALayer(nn.Module):
	def __init__(self, channel, reduction=8):
		super(CALayer, self).__init__()
		d = max(channel // reduction, 4)

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ca = nn.Sequential(
				nn.Conv2d(channel, d, 1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(d, channel, 1, bias=False),
				nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.ca(y)
		return x * y


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ConvLayerSE(ConvLayer):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__(net_depth, dim, kernel_size, gate_act)
		self.se = CALayer(dim)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		out = self.se(out)
		return out


class ConvLayerRS(ConvLayer):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__(net_depth, dim, kernel_size, gate_act)
		self.scale = nn.Parameter(torch.ones(1))

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		out = out * self.scale
		return out


class ConvLayerECA(ConvLayer):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__(net_depth, dim, kernel_size, gate_act)
		self.ca = ECALayer(dim)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		out = self.ca(out)
		return out


class ConvLayerWithoutGating(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.proj1 = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect'),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh, nn.GELU] else gate_act(inplace=True),
			nn.Conv2d(dim, dim, 1)
		)

		self.proj2 = nn.Conv2d(dim, dim, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.proj1(X) + self.proj2(X)
		return out


def GhostBN_1(dim):
	return GhostBatchNorm(dim, 1)

def GhostBN_2(dim):
	return GhostBatchNorm(dim, 2)

def GhostBN_4(dim):
	return GhostBatchNorm(dim, 4)

def GhostBN_8(dim):
	return GhostBatchNorm(dim, 8)

def GhostBN_16(dim):
	return GhostBatchNorm(dim, 16)


__all__ = ['gunet_k3_t', 'gunet_k7_t', 'gunet_s5_t', 'gunet_s9_t', 'gunet_ln_t', 'gunet_in_t', 'gunet_nn_t', 'gunet_hsig_t', 'gunet_tanh_t', 'gunet_relu_t', 'gunet_gelu_t', 
		   'gunet_idt_t', 'gunet_sum_t', 'gunet_cat_t', 'gunet_se_t', 'gunet_eca_t', 'gunet_rs_t', 'gunet_d2x_t', 'gunet_w2x_t',
		   'gunet_nb1_t', 'gunet_nb2_t', 'gunet_nb4_t', 'gunet_nb8_t', 'gunet_nb16_t', 'gunet_nb32_t', 'gunet_nb64_t', 'gunet_nb128_t',
		   'gunet_0wd_t', 'gunet_cwd_t', 'gunet_nf_t', 'gunet_nw_t', 'gunet_ni_t', 'gunet_nmp_t', 'gunet_t_0', 'gunet_t_1', 'gunet_t_2', 'gunet_t_3', 'gunet_t_4']

def gunet_k3_t():
	return gUNet(kernel_size=3, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_k7_t():
	return gUNet(kernel_size=7, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_s5_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 4, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_s9_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 2, 4, 2, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_ln_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=LayerNorm, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_in_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.InstanceNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nn_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.Identity, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_hsig_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Hardsigmoid, fusion_layer=SKFusion)

def gunet_tanh_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Tanh, fusion_layer=SKFusion)

def gunet_relu_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayerWithoutGating, norm_layer=nn.BatchNorm2d, gate_act=nn.ReLU, fusion_layer=SKFusion)

def gunet_gelu_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayerWithoutGating, norm_layer=nn.BatchNorm2d, gate_act=nn.GELU, fusion_layer=SKFusion)

def gunet_idt_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Identity, fusion_layer=SKFusion)

def gunet_sum_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SumFusion)

def gunet_cat_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=CatFusion)

def gunet_se_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayerSE, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_rs_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayerRS, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_eca_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayerECA, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_d2x_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[4, 4, 4, 8, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_w2x_t():
	return gUNet(kernel_size=5, base_dim=32, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb1_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=GhostBN_1, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb2_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=GhostBN_2, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb4_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=GhostBN_4, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb8_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=GhostBN_8, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb16_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=GhostBN_16, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb32_t():	# default
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb64_t():	# 2 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nb128_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.SyncBatchNorm, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

# For the ablation of some other training strategies, the training code needs to be modified
def gunet_0wd_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_cwd_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nf_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nw_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_ni_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_nmp_t():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_t_0():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_t_1():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_t_2():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_t_3():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_t_4():
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
