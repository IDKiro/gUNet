import time
import argparse
import torch

from torchprofile import profile_macs
from thop import profile
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_t', type=str, help='model name')
parser.add_argument('--device', default='cuda', type=str, help='test device')
parser.add_argument('--profiler', action='store_true', default=False, help='use profiler')
args = parser.parse_args()


if __name__ == '__main__':
	test_w = [256]
	test_h = [256]
	test_iter = [100]
	test_epoch = 5

	network = eval(args.model)()
	network.to(args.device)
	network.eval()

	macs = profile_macs(network, torch.rand([1, 3, 256, 256]).to(args.device))
	macs_G = macs / (1024**3)

	_, params = profile(network, inputs=(torch.rand([1, 3, 256, 256]).to(args.device), ))
	params_M = params / (1024 ** 2)

	with torch.no_grad():
		for (h, w, it) in zip(test_w, test_h, test_iter):
			rand_img = torch.rand([1, 3, h, w]).to(args.device)
			trace_network = torch.jit.trace(network, [rand_img])

			if args.profiler:							# torch.profiler slows down the model
				with torch.profiler.profile(
					activities=[
						torch.profiler.ProfilerActivity.CPU,
						torch.profiler.ProfilerActivity.CUDA,
					]
				) as p:

					for _ in range(it):
						output = trace_network(rand_img)

				print(p.key_averages().table(
					sort_by="self_cuda_time_total", row_limit=-1))

			fps_list = []
			for i in range(test_epoch):

				torch.cuda.synchronize()
				t1 = time.time()
				
				for _ in range(it):
					output = trace_network(rand_img)

				torch.cuda.synchronize()
				t2 = time.time()

				fps = it/(t2-t1)
				fps_list.append(fps)

			fps_list = sorted(fps_list)
			avg_fps = fps_list[test_epoch//2]

			print('Input Shape: {0:s}\nParams (M): {1:.3f}\nMACs (G): {2:.3f}\nRuntime (ms): {3:.2f}'
				  .format(str((1, 3, h, w)), params_M, macs_G, 1e3 / avg_fps))
