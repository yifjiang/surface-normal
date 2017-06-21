import torch
from torch import nn
from torch.autograd import Variable

from .get_theoretical_depth_from_normal import get_theoretical_depth, get_shift_mask
from .scale_inv_depth_loss import scale_inv_depth_loss
from .relative_depth_margin import relative_depth_crit

import h5py
from PIL import Image

def load_hdf5_z(h5_filename, field_name):
	myFile = h5py.File(h5_filename, 'r')
	return myFile

def save(img, filename):
	img.save(filename)
	print('Done saving to', filename)

def visualize_mask(mask, filename):
	_mask_img = mask.clone()
	t_back = transforms.ToPILImage()
	_mask_img = t_back(_mask_img)
	save(_mask_img, filename)

def visualize_depth(z, filename):
	_z_img = z.clone()
	_z_img-=torch.min(_z_img)
	_z_img/=torch.max(_z_img)
	t_back = transforms.ToPILImage()
	_z_img = t_back(_z_img)
	save(_z_img,filename)

def visualize_normal(normal,  filename):
	_normal_img = normal.clone()
	_normal_img = _normal_img + 1
	_normal_img = _normal_img*0.5
	t_back = transforms.ToPILImage()
	_normal_img = t_back(_normal_img)
	save(_normal_img,filename)


class relative_depth_margin_log_normal_depth(nn.Module):
	"""docstring for relative_depth_margin_log_normal_depth"""
	def __init__(self, w_normal, margin, camera):
		super(relative_depth_margin_log_normal_depth, self).__init__()
		self.depth_crit = relative_depth_crit(margin).cuda()
		self.normal_crit = scale_inv_depth_loss().cuda()
		self.normal_to_depth = get_theoretical_depth(camera).cuda()
		self.shift_mask = get_shift_mask().cuda()

		self.w_normal = Variable(torch.Tensor([w_normal])).cuda() #TODO:tensor or Variable?

	def forward(self, input, target):
		n_depth = target[0]['n_sample']
		n_normal = target[1]['n_sample']

		output = 0
		self.loss_relative_depth = 0
		self.loss_normal = 0

		if n_depth > 0:
			self.loss_relative_depth = self.depth_crit(torch.log(input[0:n_depth]), target[0])
			output =  self.loss_relative_depth
		if n_normal > 0:
			gt_normal_map = Variable(torch.zeros(n_normal, 3, input.size(2), input.size(3))).cuda()
			_gt_normal_mask = Variable(torch.zeros(n_normal, 3, input.size(2), input.size(3))).cuda()
			for batch_idx in range(0, n_normal):
				x_arr = target[1][batch_idx]['x']
				y_arr = target[1][batch_idx]['y']
				unsqueeze = torch.unsqueeze(target[1][batch_idx]['normal'],1).cuda()
				p2 = Variable(torch.zeros(3,input.size(2), target[1][batch_idx]['n_point'])).cuda()
				p2.scatter_(1,y_arr.view(1,-1).repeat(3,1).view(3,1,-1).long(),unsqueeze)
				gt_normal_map[batch_idx,:,:,:] = gt_normal_map[batch_idx,:,:,:].clone().index_add_(2,x_arr.long(),p2)#Check this

				unsqueeze.data.fill_(1.0)
				p2.scatter_(1,y_arr.view(1,-1).repeat(3,1).view(3,1,-1).long(),unsqueeze)
				_gt_normal_mask[batch_idx,:,:,:] = _gt_normal_mask[batch_idx,:,:,:].clone().index_add_(2,x_arr.long(),p2)

			gt_normal_mask = self.shift_mask(_gt_normal_mask[:,0:1,:,:])#Check this!

			normal_to_depth_output = self.normal_to_depth(depth_input=input[n_depth:],normal_input = gt_normal_map)
			# print(normal_to_depth_output)
			# replicated_depth = Variable(torch.zeros(n_normal, 4*input.size(1), input.size(2), input.size(3))).cuda()
			replicated_depth = input[n_depth:].repeat(1,4,1,1)#Check this!
			# print(replicated_depth)

			self.loss_normal = self.w_normal * self.normal_crit([normal_to_depth_output, gt_normal_mask], replicated_depth)
			output = output+self.loss_normal

		return output