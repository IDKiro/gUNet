import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
from mmcv.ops import DeformConv2dPack
import math


def isqrt_newton_schulz_autograd(A, numIters):
	dim = A.shape[0]
	normA=A.norm()
	Y = A.div(normA)
	I = torch.eye(dim,dtype=A.dtype,device=A.device)
	Z = torch.eye(dim,dtype=A.dtype,device=A.device)

	for i in range(numIters):
		T = 0.5*(3.0*I - Z@Y)
		Y = Y@T
		Z = T@Z
	#A_sqrt = Y*torch.sqrt(normA)
	A_isqrt = Z / torch.sqrt(normA)
	return A_isqrt


def isqrt_newton_schulz_autograd_batch(A, numIters):
	batchSize,dim,_ = A.shape
	normA=A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
	Y = A.div(normA)
	I = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)
	Z = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)

	for i in range(numIters):
		T = 0.5*(3.0*I - Z.bmm(Y))
		Y = Y.bmm(T)
		Z = T.bmm(Z)
	#A_sqrt = Y*torch.sqrt(normA)
	A_isqrt = Z / torch.sqrt(normA)

	return A_isqrt


class FastDeconv(conv._ConvNd):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100):
		self.momentum = momentum
		self.n_iter = n_iter
		self.eps = eps
		self.counter=0
		self.track_running_stats=True
		super(FastDeconv, self).__init__(
			in_channels, out_channels,  _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
			False, _pair(0), groups, bias, padding_mode='zeros')

		if block > in_channels:
			block = in_channels
		else:
			if in_channels%block!=0:
				block=math.gcd(block,in_channels)

		if groups>1:
			#grouped conv
			block=in_channels//groups

		self.block=block

		self.num_features = kernel_size**2 *block
		if groups==1:
			self.register_buffer('running_mean', torch.zeros(self.num_features))
			self.register_buffer('running_deconv', torch.eye(self.num_features))
		else:
			self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
			self.register_buffer('running_deconv', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

		self.sampling_stride=sampling_stride*stride
		self.counter=0
		self.freeze_iter=freeze_iter
		self.freeze=freeze

	def forward(self, x):
		N, C, H, W = x.shape
		B = self.block
		frozen=self.freeze and (self.counter>self.freeze_iter)
		if self.training and self.track_running_stats:
			self.counter+=1
			self.counter %= (self.freeze_iter * 10)

		if self.training and (not frozen):

			# 1. im2col: N x cols x pixels -> N*pixles x cols
			if self.kernel_size[0]>1:
				X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
			else:
				#channel wise
				X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

			if self.groups==1:
				# (C//B*N*pixels,k*k*B)
				X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
			else:
				X=X.view(-1,X.shape[-1])

			# 2. subtract mean
			X_mean = X.mean(0)
			X = X - X_mean.unsqueeze(0)

			# 3. calculate COV, COV^(-0.5), then deconv
			if self.groups==1:
				#Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
				Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
				Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
				deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
			else:
				X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
				Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
				Cov = torch.baddbmm(self.eps, Id, 1. / X.shape[1], X.transpose(1, 2), X)

				deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

			if self.track_running_stats:
				self.running_mean.mul_(1 - self.momentum)
				self.running_mean.add_(X_mean.detach() * self.momentum)
				# track stats for evaluation
				self.running_deconv.mul_(1 - self.momentum)
				self.running_deconv.add_(deconv.detach() * self.momentum)

		else:
			X_mean = self.running_mean
			deconv = self.running_deconv

		#4. X * deconv * conv = X * (deconv * conv)
		if self.groups==1:
			w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ deconv
			b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
			w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
		else:
			w = self.weight.view(C//B, -1,self.num_features)@deconv
			b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

		w = w.view(self.weight.shape)
		x= F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

		return x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
	return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
	def __init__(self, channel):
		super(PALayer, self).__init__()
		self.pa = nn.Sequential(
			nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.pa(x)
		return x * y


class CALayer(nn.Module):
	def __init__(self, channel):
		super(CALayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ca = nn.Sequential(
			nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.ca(y)
		return x * y


class DehazeBlock(nn.Module):
	def __init__(self, conv, dim, kernel_size, ):
		super(DehazeBlock, self).__init__()
		self.conv1 = conv(dim, dim, kernel_size, bias=True)
		self.act1 = nn.ReLU(inplace=True)
		self.conv2 = conv(dim, dim, kernel_size, bias=True)
		self.calayer = CALayer(dim)
		self.palayer = PALayer(dim)

	def forward(self, x):
		res = self.act1(self.conv1(x))
		res = res + x
		res = self.conv2(res)
		res = self.calayer(res)
		res = self.palayer(res)
		res += x
		return res


class Mix(nn.Module):
	def __init__(self, m=-0.80):
		super(Mix, self).__init__()
		w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
		w = torch.nn.Parameter(w, requires_grad=True)
		self.w = w
		self.mix_block = nn.Sigmoid()

	def forward(self, fea1, fea2):
		mix_factor = self.mix_block(self.w)
		out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
		return out


class AECRNet(nn.Module):
	def __init__(self, input_nc=3, output_nc=3, ngf=64):
		super(AECRNet, self).__init__()

		# NOTE: share parameters: self.block, self.dcn_block ?

		###### downsample
		self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
								   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
								   nn.ReLU(True))
		self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
								   nn.ReLU(True))
		self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
								   nn.ReLU(True))

		###### FFA blocks
		self.block = DehazeBlock(default_conv, ngf * 4, 3)

		###### upsample
		self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
								 nn.ReLU(True))
		self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
								 nn.ReLU(True))
		self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
								 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
								 nn.Tanh())


		self.dcn_block = DeformConv2dPack(256, 256, 3, padding=1)

		self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

		self.mix1 = Mix(m=-1)
		self.mix2 = Mix(m=-0.6)

	def forward(self, input):
		_, _, H, W = input.size()

		# poor network design
		padding_h = math.ceil(H / 16) * 16 - H
		padding_w = math.ceil(W / 16) * 16 - W

		input = F.pad(input, (0, padding_w, 0, padding_h), mode='reflect')

		x_deconv = self.deconv(input) # preprocess

		x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
		x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
		x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]

		x1 = self.block(x_down3)
		x2 = self.block(x1)
		x3 = self.block(x2)
		x4 = self.block(x3)
		x5 = self.block(x4)
		x6 = self.block(x5)

		x_dcn1 = self.dcn_block(x6)
		x_dcn2 = self.dcn_block(x_dcn1)

		x_out_mix = self.mix1(x_down3, x_dcn2)
		x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
		x_up1_mix = self.mix2(x_down2, x_up1)
		x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256] 
		out = self.up3(x_up2) # [bs,  3, 256, 256]

		return out[:, :, :H ,:W]