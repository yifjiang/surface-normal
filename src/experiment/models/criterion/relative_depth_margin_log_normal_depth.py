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

		self.w_normal = Variable(torch.Tensor([w_normal])) #TODO:tensor or Variable?

	def forward(self, input, target):
		