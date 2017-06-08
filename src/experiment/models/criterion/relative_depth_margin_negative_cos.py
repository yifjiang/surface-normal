import torch
from torch import nn
from torch.autograd import Variable

from .relative_depth_margin import relative_depth_crit
from .normal_negative_cos import normal_negative_cos_crit

from ..img_coord_to_world_coord import img_coord_to_world_coord
from ..world_coord_to_normal import world_coord_to_normal

class relative_depth_negative_cos(nn.Module):
	"""docstring for relative_depth_negative_cos"""
	def __init__(self, w_normal, margin,camera):
		super(relative_depth_negative_cos, self).__init__()
		self.depth_crit = relative_depth_crit(margin)
		self.normal_crit = normal_negative_cos_crit()
		self.depth_to_normal = nn.Sequential(
			img_coord_to_world_coord(camera),
			world_coord_to_normal()
			)
		self.w_normal = w_normal

	def forward(input, target):
		n_depth = target[0]['n_sample']
		n_normal = target[1]['n_sample']

		loss_normal = 0
		loss_relative_depth = 0
		output = 0

		if n_depth > 0:
			loss_relative_depth = self.depth_crit(input[0:n_depth], target[0])
			output += loss_relative_depth
		if n_normal > 0:
			normal = self.depth_to_normal(input[n_depth:])
			loss_normal = self.w_normal*self.normal_crit(normal, target[1])
			output += loss_normal

		return output

if __name__ == '__main__':
	#test
	camera = {}
	camera['input_width'] = 5
	camera['input_height'] = 5
	camera['cx'] = 2
	camera['cy'] = 3
	camera['fx'] = 6
	camera['fy'] = -4

	crit = relative_depth_negative_cos(1,3,camera).cuda()

	x = torch.zeros(1,1,6,6).cuda()
	x[0,0,1,1] = 2
	x[0,0,2,2] = 5
	x[0,0,3,3] = -4
	x[0,0,4,4] = -4
	x[0,0,5,5] = 4
	x = x.repeat(2,1,1,1);
	x = Variable(x.cuda(),requires_grad = True)

	target = {}
	target[0] = {}
	target[0]['x_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['y_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['x_B'] = Variable(torch.Tensor([0,0,0,0,0,0])).cuda()
	target[0]['y_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	target[0]['ordianl_relation'] = Variable(torch.Tensor([-1,0,1,1,-1,-1])).cuda()
	target[0]['n_point'] = 6
	target['n_sample'] = 1

	target_n = {}
	target_n[0] = {}
	target_n[0]['x'] = Variable(torch.Tensor([0,1])).cuda()
	target_n[0]['y'] = Variable(torch.Tensor([0,0])).cuda()
	target_n[0]['normal'] = Variable(torch.Tensor([[1,0],[0,1],[0,0]])).cuda()
	target_n[0]['n_point'] = 2
	target_n['n_sample'] = 1

	target_both = {}
	target_both[0] = target
	target_both[1] = target_n

	loss = crit.forward(x,target_both)
	print(loss)
	loss.backward()
	print(x.grad)