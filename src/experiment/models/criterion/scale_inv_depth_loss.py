import torch
from torch import nn
from torch.autograd import Variable

class scale_inv_depth_loss(nn.Module):
	"""docstring for scale_inv_depth_loss"""
	def __init__(self):
		super(scale_inv_depth_loss, self).__init__()
	
	def forward(self, input, target):
		denominator = torch.pow(input[0]-target,2)
		nominator = torch.pow(input[0]+target,2)

		zero_mask = (nominator==0)
		nominator[zero_mask] = 1e-7
		denominator = denominator/nominator
		denominator = denominator*input[1]
		output = torch.sum(denominator)
		return output/torch.sum(input[1])