import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from collections import OrderedDict

from utils import AverageMeter, pad_img
from datasets import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_t', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs for QAT finetune')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--train_set', default='ITS', type=str, help='train dataset name')
parser.add_argument('--val_set', default='SOTS-IN', type=str, help='valid dataset name')
parser.add_argument('--exp', default='reside-in', type=str, help='experiment setting')
args = parser.parse_args()


# training config
with open(os.path.join('configs', args.exp, 'base.json'), 'r') as f:
	b_setup = json.load(f)

with open(os.path.join('configs', args.exp, 'model_'+args.model.split('_')[-1]+'.json'), 'r') as f:
	m_setup = json.load(f)


def single(state_dict):
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def train(train_loader, network, criterion, optimizer):
	network.train()

	losses = AverageMeter()

	for batch in tqdm(train_loader, desc='Train'):
		source_img = batch['source']
		target_img = batch['target']

		output = network(source_img)
		loss = criterion(output, target_img)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.update(loss.item())

	return losses.avg


def valid(val_loader, network):
	network.eval()

	PSNR = AverageMeter()

	for batch in tqdm(val_loader, desc='Valid'):
		source_img = batch['source']
		target_img = batch['target']

		with torch.no_grad():
			H, W = source_img.shape[2:]
			source_img = pad_img(source_img, network.patch_size if hasattr(network, 'patch_size') else 16)
			output = network(source_img).clamp_(-1, 1)
			output = output[:, :, :H, :W]

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()

		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


def main():
	network = eval(args.model)()
	criterion = nn.L1Loss()
	optimizer = torch.optim.AdamW(network.parameters(), lr=m_setup['lr'] * 0.1, weight_decay=b_setup['weight_decay'] * 0.1)

	# load saved model
	save_dir = os.path.join(args.save_dir, args.exp)
	model_info = torch.load(os.path.join(save_dir, args.model+'.pth'), map_location='cpu')
	network.load_state_dict(single(model_info['state_dict']))

	network = torch.quantization.QuantWrapper(network)
	network.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
	torch.quantization.prepare_qat(network, inplace=True)

	# define dataset
	train_dataset = PairLoader(os.path.join(args.data_dir, args.train_set), 'train', 
							   b_setup['t_patch_size'], 
							   b_setup['edge_decay'], 
							   b_setup['data_augment'], 
							   b_setup['cache_memory'])
	train_loader = DataLoader(train_dataset,
							  batch_size=32,
							  sampler=RandomSampler(train_dataset, num_samples=b_setup['num_iter']),
							  num_workers=args.num_workers,
							  pin_memory=True,
							  drop_last=True,
							  persistent_workers=True)

	val_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), 'test', 
							 b_setup['v_patch_size'])
	val_loader = DataLoader(val_dataset,
							batch_size=1,
							num_workers=args.num_workers,
							pin_memory=True)

	best_psnr = valid(val_loader, network)

	for epoch in range(args.epochs):
		loss = train(train_loader, network, criterion, optimizer)

		avg_psnr = valid(val_loader, network)
		if avg_psnr > best_psnr:
			best_psnr = avg_psnr
			quantized_network = torch.quantization.convert(network.eval(), inplace=False)
			torch.save({'state_dict': quantized_network.state_dict()},
						os.path.join(save_dir, args.model+'_qat.pth'))
						
		print('Test: [{0}]\t Best PSNR: {1:.02f} \t Current PSNR: {2:.02f}'
				.format(epoch, best_psnr, avg_psnr))

if __name__ == '__main__':
	main()
