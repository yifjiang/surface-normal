import torch
from torch import nn
from torch.autograd import Variable

class normal_negative_cos_crit(nn.Module):
	"""docstring for normal_negative_cos_crit"""
	def __init__(self):
		super(normal_negative_cos_crit, self).__init__()

	def forward(self, input, target):
		output = 0
		n_points = 0
		for batch_idx in range(0,input.size()[0]):
			n_points += target[batch_idx]['n_point']
			x_arr = target[batch_idx]['x']
			y_arr = target[batch_idx]['y']
			batch_input = input[batch_idx]
			normal_arr =batch_input.index_select(2,x_arr.long()).gather(1, y_arr.view(1,-1).long().repeat(3,1).view(3,1,-1)).squeeze()
			ground_truth_arr = target[batch_idx]['normal']
			output-=torch.sum(normal_arr*ground_truth_arr)

		return output/n_points

