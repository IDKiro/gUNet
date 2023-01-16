import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., data_augment=True):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	if data_augment:
		# horizontal flip
		if random.randint(0, 1) == 1:
			for i in range(len(imgs)):
				imgs[i] = np.flip(imgs[i], axis=1)

		# bad data augmentations for outdoor dehazing
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, root_dir, mode, size=256, edge_decay=0, data_augment=True, cache_memory=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.data_augment = data_augment

		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

		self.cache_memory = cache_memory
		self.source_files = {}
		self.target_files = {}

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# select a image pair
		img_name = self.img_names[idx]

		# read images
		if img_name not in self.source_files:
			source_img = read_img(os.path.join(self.root_dir, 'IN', img_name), to_float=False)
			target_img = read_img(os.path.join(self.root_dir, 'GT', img_name), to_float=False)

			# cache in memory if specific (uint8 to save memory), need num_workers=0
			if self.cache_memory:
				self.source_files[img_name] = source_img
				self.target_files[img_name] = target_img
		else:
			# load cached images
			source_img = self.source_files[img_name]
			target_img = self.target_files[img_name]

		# [0, 1] to [-1, 1]
		source_img = source_img.astype('float32') / 255.0 * 2 - 1
		target_img = target_img.astype('float32') / 255.0 * 2 - 1
		
		# data augmentation
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.data_augment)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
