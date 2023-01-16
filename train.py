import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter, CosineScheduler, pad_img
from datasets import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_t', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--use_mp', action='store_true', default=False, help='use Mixed Precision')
parser.add_argument('--use_ddp', action='store_true', default=False, help='use Distributed Data Parallel')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--train_set', default='ITS', type=str, help='train dataset name')
parser.add_argument('--val_set', default='SOTS-IN', type=str, help='valid dataset name')
parser.add_argument('--exp', default='reside-in', type=str, help='experiment setting')
args = parser.parse_args()


# training environment
if args.use_ddp:
	torch.distributed.init_process_group(backend='nccl', init_method='env://')
	world_size = dist.get_world_size()
	local_rank = dist.get_rank()
	torch.cuda.set_device(local_rank)
	if local_rank == 0: print('==> Using DDP.')
else:
	world_size = 1


# training config
with open(os.path.join('configs', args.exp, 'base.json'), 'r') as f:
	b_setup = json.load(f)

variant = args.model.split('_')[-1]
config_name = 'model_'+variant+'.json' if variant in ['t', 's', 'b', 'd'] else 'default.json'	# default.json as baselines' configuration file 
with open(os.path.join('configs', args.exp, config_name), 'r') as f:
	m_setup = json.load(f)


def reduce_mean(tensor, nprocs):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= nprocs
	return rt


def train(train_loader, network, criterion, optimizer, scaler, frozen_bn=False):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.eval() if frozen_bn else network.train()	# simplified implementation that other modules may be affected

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.use_mp):
			output = network(source_img)
			loss = criterion(output, target_img)

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		if args.use_ddp: loss = reduce_mean(loss, dist.get_world_size())
		losses.update(loss.item())

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():
			H, W = source_img.shape[2:]
			source_img = pad_img(source_img, network.module.patch_size if hasattr(network.module, 'patch_size') else 16)
			output = network(source_img).clamp_(-1, 1)
			output = output[:, :, :H, :W]

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		# if args.use_ddp: psnr = reduce_mean(psnr, dist.get_world_size())		# comment this line for more accurate validation
		
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


def main():
	# define network, and use DDP for faster training
	network = eval(args.model)()
	network.cuda()

	if args.use_ddp:
		network = DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
		if m_setup['batch_size'] // world_size < 16:
			if local_rank == 0: print('==> Using SyncBN because of too small norm-batch-size.')
			nn.SyncBatchNorm.convert_sync_batchnorm(network)
	else:
		network = DataParallel(network)
		if m_setup['batch_size'] // torch.cuda.device_count() < 16:
			print('==> Using SyncBN because of too small norm-batch-size.')
			convert_model(network)

	# define loss function
	criterion = nn.L1Loss()
	
	# define optimizer
	optimizer = torch.optim.AdamW(network.parameters(), lr=m_setup['lr'], weight_decay=b_setup['weight_decay'])
	lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=b_setup['epochs'], value_min=m_setup['lr'] * 1e-2, 
								   warmup_t=b_setup['warmup_epochs'], const_t=b_setup['const_epochs'])
	wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=b_setup['epochs'])	# seems not to work
	scaler = GradScaler()

	# load saved model
	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)
	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		best_psnr = 0
		cur_epoch = 0
	else:
		if not args.use_ddp or local_rank == 0: print('==> Loaded existing trained model.')
		model_info = torch.load(os.path.join(save_dir, args.model+'.pth'), map_location='cpu')
		network.load_state_dict(model_info['state_dict'])
		optimizer.load_state_dict(model_info['optimizer'])
		lr_scheduler.load_state_dict(model_info['lr_scheduler'])
		wd_scheduler.load_state_dict(model_info['wd_scheduler'])
		scaler.load_state_dict(model_info['scaler'])
		cur_epoch = model_info['cur_epoch']
		best_psnr = model_info['best_psnr']

	# define dataset
	train_dataset = PairLoader(os.path.join(args.data_dir, args.train_set), 'train', 
							   b_setup['t_patch_size'], 
							   b_setup['edge_decay'], 
							   b_setup['data_augment'], 
							   b_setup['cache_memory'])
	train_loader = DataLoader(train_dataset,
							  batch_size=m_setup['batch_size'] // world_size,
							  sampler=RandomSampler(train_dataset, num_samples=b_setup['num_iter'] // world_size),
							  num_workers=args.num_workers // world_size,
							  pin_memory=True,
							  drop_last=True,
							  persistent_workers=True)	# comment this line for cache_memory

	val_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'], 
							 b_setup['v_patch_size'])
	val_loader = DataLoader(val_dataset,
							batch_size=max(int(m_setup['batch_size'] * b_setup['v_batch_ratio'] // world_size), 1),
							# sampler=DistributedSampler(val_dataset, shuffle=False),		# comment this line for more accurate validation
							num_workers=args.num_workers // world_size,
							pin_memory=True)

	# start training
	if not args.use_ddp or local_rank == 0:
		print('==> Start training, current model name: ' + args.model)
		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

	for epoch in tqdm(range(cur_epoch, b_setup['epochs'] + 1)):
		frozen_bn = epoch > (b_setup['epochs'] - b_setup['frozen_epochs']) 
		
		loss = train(train_loader, network, criterion, optimizer, scaler, frozen_bn)
		lr_scheduler.step(epoch + 1)
		wd_scheduler.step(epoch + 1)

		if not args.use_ddp or local_rank == 0:
			writer.add_scalar('train_loss', loss, epoch)

		if epoch % b_setup['eval_freq'] == 0:
			avg_psnr = valid(val_loader, network)
			
			if not args.use_ddp or local_rank == 0:
				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'cur_epoch': epoch + 1,
								'best_psnr': best_psnr,
								'state_dict': network.state_dict(),
								'optimizer' : optimizer.state_dict(),
								'lr_scheduler' : lr_scheduler.state_dict(),
								'wd_scheduler' : wd_scheduler.state_dict(),
								'scaler' : scaler.state_dict()},
								os.path.join(save_dir, args.model+'.pth'))
				
				writer.add_scalar('valid_psnr', avg_psnr, epoch)
				writer.add_scalar('best_psnr', best_psnr, epoch)
		
			if args.use_ddp: dist.barrier()
		

if __name__ == '__main__':
	main()
