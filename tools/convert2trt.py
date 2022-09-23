import os
import time
import argparse
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
from onnxsim import simplify

from utils import chw_to_hwc, hwc_to_chw
from models import *
from test import single


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_t', type=str, help='model name')
parser.add_argument('--load_dir', default='./saved_models/', type=str, help='path to models loading')
parser.add_argument('--save_dir', default='./runtime_models/', type=str, help='path to models saving')
parser.add_argument('--exp', default='reside-in', type=str, help='experiment setting')
args = parser.parse_args()


def check_speed(trt_model_path, H=256, W=256):
	# whether FP16
	USE_FP16 = True 
	target_dtype = np.float16 if USE_FP16 else np.float32

	# init engine
	f = open(trt_model_path, "rb")
	runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
	engine = runtime.deserialize_cuda_engine(f.read())
	context = engine.create_execution_context()

	# allocate memory
	input_batch = np.random.randn(1, H, W, 3).astype(target_dtype)
	output = np.empty([1, H, W, 3], dtype=target_dtype)
	d_input = cuda.mem_alloc(1 * input_batch.nbytes)
	d_output = cuda.mem_alloc(1 * output.nbytes)
	bindings = [int(d_input), int(d_output)]

	stream = cuda.Stream()
	def predict(batch):
		cuda.memcpy_htod_async(d_input, batch, stream)
		context.execute_async_v2(bindings, stream.handle, None)
		cuda.memcpy_dtoh_async(output, d_output, stream)
		stream.synchronize()
		return output

	preprocessed_inputs = np.array([hwc_to_chw(input) for input in input_batch])  # (BATCH_SIZE,224,224,3)——>(BATCH_SIZE,3,224,224)
	print("Warming up...")
	pred = predict(preprocessed_inputs)
	print("Done warming up!")

	t0 = time.time()
	for i in range(1000):
		pred = predict(preprocessed_inputs)
	t = time.time() - t0
	print("{:.4f}s for 1000 samples".format(t))


def main():
	network = eval(args.model)()
	network.cuda()
	saved_model_dir = os.path.join(args.load_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start converting, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	network.eval()

	input = torch.randn(1, 3, 256, 256).cuda()

	input_names = ['input']
	output_names = ['output']

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	# convert model to onnx
	torch.onnx.export(
		network,
		input,
		os.path.join(args.save_dir, args.exp, args.model+'.onnx'),
		verbose=True,
		input_names=input_names,
		output_names=output_names,
		dynamic_axes={'input':[2,3], 'output':[2,3]},
		opset_version=11
	)

	# use simplify
	model = onnx.load(os.path.join(args.save_dir, args.exp, args.model+'.onnx'))
	model_simp, check = simplify(model)
	assert check
	onnx.save(model_simp, os.path.join(args.save_dir, args.exp, args.model+'.onnx'))

	# convert model to tensorrt
	os.system(
		'trtexec --onnx={0:s} --saveEngine={1:s} --shapes=input:1x3x256x256 --int8'.format(
			os.path.join(args.save_dir, args.exp, args.model+'.onnx'),
			os.path.join(args.save_dir, args.exp, args.model+'.trt')
		)
	)

	# validate converted model
	check_speed(os.path.join(args.save_dir, args.exp, args.model+'.trt'))


if __name__ == "__main__":
	main()
