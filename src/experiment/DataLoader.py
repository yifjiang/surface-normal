import os
import h5py
import math
import random
import torch
import array
from common.NYU_params import *
from DataPointer import DataPointer
from torchvision import transforms
from PIL import Image
from common.NYU_params import g_input_height, g_input_width, g_large_input_width, g_large_input_height


# _batch_target_relative_depth_gpu = {}
# for i in range(0,g_args.bs):#g_args is from main.py
# 	_batch_target_relative_depth_gpu[i] = {}
# 	_batch_target_relative_depth_gpu[i]['y_A'] = torch.Tensor().cuda()
# 	_batch_target_relative_depth_gpu[i]['x_A'] = torch.Tensor().cuda()
# 	_batch_target_relative_depth_gpu[i]['y_B'] = torch.Tensor().cuda()
# 	_batch_target_relative_depth_gpu[i]['x_B'] = torch.Tensor().cuda()
# 	_batch_target_relative_depth_gpu[i]['ordianl_relation'] = torch.Tensor().cuda()

class DataLoader(object):
	"""docstring for DataLoader"""
	def __init__(self, relative_depth_filename, normal_filename, n_max_depth = None, n_max_normal = None):
		super(DataLoader, self).__init__()
		print(">>>>>>>>>>>>>>>>> Using DataLoader")

		self.n_max_depth = n_max_depth
		self.n_max_normal = n_max_normal

		print("n_max_depth =", n_max_depth, "n_max_normal =", n_max_normal)

		self.parse_depth_and_normal(relative_depth_filename,normal_filename)
		self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
		self.data_ptr_normal = DataPointer(self.n_normal_sample)
		print("DataLoader init: \n \t{} relative depth samples \n \t{} normal samples".format(self.n_relative_depth_sample, self.n_normal_sample))



	def parse_relative_depth_line(self, line, n_max_depth):
		splits = line.split(',')
		sample = {}
		sample['img_filename'] = splits[0]
		sample['n_point'] = int(splits[2])
		if n_max_depth is not None:
			sample['n_point'] = min(int(n_max_depth), int(sample['n_point']))
		return sample

	def parse_normal_line(self, line, n_max_normal):
		splits = line.split(',')
		sample = {}
		sample['img_filename'] = splits[0]
		sample['n_point'] = int(splits[1])

		if n_max_normal is not None:
			# print(int(n_max_normal))
			# print(int(sample['n_point']))
			sample['n_point'] = min(int(n_max_normal), int(sample['n_point']))

		if len(splits) > 2:
			sample['focal_length'] = float(splits[2])
		return sample

	def parse_csv(self, filename, parsing_func, n_max_point):
		_handle = {}

		if filename == None:
			return _handle

		_n_lines = 0
		f = open(filename, 'r')
		for l in f:
			_n_lines+=1
		f.close()

		csv_file_handle = open(filename, 'r')
		_sample_idx = 0
		print(_n_lines)
		while _sample_idx < _n_lines:
			this_line = csv_file_handle.readline()
			if this_line != '':
				_handle[_sample_idx] = parsing_func(this_line, n_max_point)
				_sample_idx+=1
			else:
				_n_lines-=1
				print('empty')

		csv_file_handle.close()

		return _handle

	def parse_depth_and_normal(self, relative_depth_filename, normal_filename):
		if relative_depth_filename is not None:
			_simplified_relative_depth_filename = relative_depth_filename.replace('.csv', '_name.csv')
			if os.path.isfile(_simplified_relative_depth_filename):
				print(_simplified_relative_depth_filename+" already exists.")
			else:
				command = "grep '.png' "+ relative_depth_filename + " > " + _simplified_relative_depth_filename
				print("executing:{}".format(command))
				os.system(command)

			self.relative_depth_handle = self.parse_csv(_simplified_relative_depth_filename, self.parse_relative_depth_line, self.n_max_depth)

			hdf5_filename = relative_depth_filename.replace('.csv', '.h5')
			self.relative_depth_handle['hdf5_handle'] = h5py.File(hdf5_filename, 'r')

		else:
			self.relative_depth_handle = {}

		if normal_filename is not None:
			self.normal_handle = self.parse_csv(normal_filename, self.parse_normal_line, self.n_max_normal)
		else:
			self.normal_handle = {}

		self.n_relative_depth_sample = len(self.relative_depth_handle)-1
		self.n_normal_sample = len(self.normal_handle)#check this!

	def close():
		pass

	def mixed_sample_strategy1(self, batch_size):
		n_depth = random.randint(0,batch_size-1)
		return n_depth, batch_size - n_depth

	def mixed_sample_strategy2(self, batch_size):
		n_depth = floor(batch_size/2)
		return n_depth, batch_size - n_depth #careful about the index


	def load_indices(self, depth_indices, normal_indices, b_load_gtz = False):
		if depth_indices is not None and self.n_relative_depth_sample>0:
			n_depth = depth_indices.size()[0]
		else:
			n_depth = 0

		if normal_indices is not None and self.n_normal_sample > 0:
			n_normal = normal_indices.size()[0]
		else:
			n_normal = 0

		if n_depth == 0 and n_normal == 0:
			print("---->>>> Warning: Both n_depth and n_normal equal 0 in DataLoader:load_indices().")
			assert(false)

		batch_size = n_depth + n_normal
		color = torch.Tensor(batch_size, 3, g_input_height, g_input_width) # now it's a Tensor, remember to make it a Variable

		_batch_target_relative_depth_gpu = {}
		_batch_target_relative_depth_gpu['n_sample'] = n_depth

		_batch_target_normal_gpu = {}
		_batch_target_normal_gpu['n_sample'] = n_normal

		if b_load_gtz and n_depth > 0:
			_batch_target_relative_depth_gpu['gt_depth'] = torch.Tensor(batch_size, g_large_input_height, g_large_input_width)# now it's a Tensor, remember to make it a Variable
		if b_load_gtz and n_normal > 0:
			_batch_target_normal_gpu['gt_depth'] = torch.Tensor(batch_size, g_large_input_height, g_large_input_width)# now it's a Tensor, remember to make it a Variable



		loader = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # may not need this
			])
		# loader = transforms.ToTensor()

		for i in range(0,n_depth):
			idx = depth_indices[i]
			_batch_target_relative_depth_gpu[i] = {}
			img_name = self.relative_depth_handle[idx]['img_filename']
			# print(img_name)
			n_point = self.relative_depth_handle[idx]['n_point']
			
			image = Image.open(img_name)
			image = loader(image).float()
			# print(image)
			# print(image.size())
			# image = Variable(image, require_grad=True)
			color[i,:,:,:].copy_(image)

			if b_load_gtz and n_depth:
				gt_z_filename = img_name.replace('.png','_gt_depth.h5')
				if os.path.isfile(gt_z_filename):
					gtz_h5_handle = h5py.File(gt_z_filename, 'r')
					_batch_target_relative_depth_gpu['gt_depth'][i,:,:].copy_(torch.Tensor(gtz_h5_handle['/gt_depth']))
					gtz_h5_handle.close()
				else:
					print("File not found:", gt_z_filename)


			_hdf5_offset = int(5*idx) #zero-indexed
			# print(self.relative_depth_handle)
			# print(n_point)
			# print(_hdf5_offset)
			_this_sample_hdf5 = self.relative_depth_handle['hdf5_handle']['/data'][_hdf5_offset:_hdf5_offset+5,0:n_point]#todo:check this
			# print(_this_sample_hdf5)
			# print(type(_this_sample_hdf5))
			# print(_this_sample_hdf5.size)

			assert(_this_sample_hdf5.shape[0] == 5)
			assert(_this_sample_hdf5.shape[1] == n_point)

			_batch_target_relative_depth_gpu[i]['y_A']= torch.autograd.Variable(torch.from_numpy(_this_sample_hdf5[0]-1)).cuda()
			_batch_target_relative_depth_gpu[i]['x_A']= torch.autograd.Variable(torch.from_numpy(_this_sample_hdf5[1]-1)).cuda()
			_batch_target_relative_depth_gpu[i]['y_B']= torch.autograd.Variable(torch.from_numpy(_this_sample_hdf5[2]-1)).cuda()
			_batch_target_relative_depth_gpu[i]['x_B']= torch.autograd.Variable(torch.from_numpy(_this_sample_hdf5[3]-1)).cuda()			
			_batch_target_relative_depth_gpu[i]['ordianl_relation']= torch.autograd.Variable(torch.from_numpy(_this_sample_hdf5[4])).cuda()
			_batch_target_relative_depth_gpu[i]['n_point'] = n_point

		if b_load_gtz and n_depth:
			_batch_target_relative_depth_gpu['gt_depth'] = torch.autograd.Variable(_batch_target_relative_depth_gpu['gt_depth'].cuda())

		_batch_target_normal_gpu['focal_length'] = torch.Tensor(n_normal,1)#remember to make this a Variable!

		for i in range(n_depth,batch_size):
			idx = normal_indices[i - n_depth]
			img_name = self.normal_handle[idx]['img_filename']
			n_point = self.normal_handle[idx]['n_point']

			_batch_target_normal_gpu['focal_length'][i - n_depth, 0] = self.normal_handle[idx]['focal_length']

			color[i,:,:,:].copy_(loader(Image.open(img_name)).float())

			if b_load_gtz and n_normal:
				gt_z_filename = img_name.replace('.png','_gt_depth.h5')
				if os.path.isfile(gt_z_filename):
					gtz_h5_handle = h5py.File(gt_z_filename, 'r')
					_batch_target_normal_gpu['gt_depth'][i - n_depth,:,:].copy_(torch.Tensor(gtz_h5_handle['/gt_depth']))
					gtz_h5_handle.close()
				else:
					print("File not found:", gt_z_filename)

			normal_name = img_name.replace('.png','_normal.bin')
			file = open(normal_name,'rb')
			normal = array.array('d')
			# print(file)
			normal.fromfile(file,5*n_point)
			# print(normal)
			file.close()
			normal = torch.Tensor(normal.tolist()).view(n_point, 5).transpose(0,1)#check this!
			# print(normal)
			# print(normal)

			# print(i-n_depth)
			_batch_target_normal_gpu[i - n_depth] = {}
			_batch_target_normal_gpu[i - n_depth]['x'] = torch.autograd.Variable(normal[0].int().cuda())
			_batch_target_normal_gpu[i - n_depth]['y'] = torch.autograd.Variable(normal[1].int().cuda())
			_batch_target_normal_gpu[i - n_depth]['normal'] = torch.autograd.Variable(normal[2:5].cuda())
			_batch_target_normal_gpu[i - n_depth]['n_point'] = n_point

		if b_load_gtz  and n_normal:
			_batch_target_normal_gpu['gt_depth'] = torch.autograd.Variable(_batch_target_normal_gpu['gt_depth'].cuda())
		_batch_target_normal_gpu['focal_length'] = torch.autograd.Variable(_batch_target_normal_gpu['focal_length'].cuda())



		return torch.autograd.Variable(color.cuda()), [_batch_target_relative_depth_gpu, _batch_target_normal_gpu]

	def load_next_batch(self, batch_size):

		if self.n_normal_sample>0 and self.n_relative_depth_sample>0:
			n_depth, n_normal = self.mixed_sample_strategy1(batch_size)
		elif self.n_normal_sample > 0:
			n_normal = batch_size
			n_depth = 0
		elif self.n_relative_depth_sample > 0:
			n_normal = 0
			n_depth = batch_size
		else:
			n_normal = 0
			n_depth = 0
			print(">>>>>>>>>    Error: No depth sample or normal sample!")
			assert(False)

		depth_indices = self.data_ptr_relative_depth.load_next_batch(n_depth)
		normal_indices = self.data_ptr_normal.load_next_batch(n_normal)


		return self.load_indices(depth_indices, normal_indices)

	def reset(self):
		self.current_pos = 1