import torch
from torch import nn
from torch.autograd import Variable

from .get_theoretical_depth_from_normal import get_theoretical_depth
from .scale_inv_depth_loss import scale_inv_depth_loss
from .relative_depth_margin import relative_depth_crit

import h5py
from PIL import Image

def load_hdf5_z(h5_filename, field_name):
	myFile = h5py.File(h5_filename, 'r')
	return myFile



class relative_depth_margin_log_normal_depth(nn.Module):
	"""docstring for relative_depth_margin_log_normal_depth"""
	def __init__(self, arg):
		super(relative_depth_margin_log_normal_depth, self).__init__()
		self.arg = arg
