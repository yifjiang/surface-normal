import torch
from torch import nn
from torch.autograd import Variable

class vectorize(nn.Module):
	"""docstring for vectorize"""
	def __init__(self, kernel):
		super(vectorize, self).__init__()
		self.model = nn.Conv2d(3,3,(kernel.size()[0],kernel.size()[1]), 1,1)
		self.model.weight.data=kernel
		self.model.bias.data.zero_()

	def forward(self,input):
		output = self.model(input)
		return output 

class spatial_normalization(nn.Module):
	"""docstring for spatial_normalization"""
	def __init__(self):
		super(spatial_normalization, self).__init__()
	
	def forward(self,input):
		# input is N x 3 x H x W
		square_sum = input[:,0,:,:]*input[:,0,:,:]+input[:,1,:,:]*input[:,1,:,:]+input[:,2,:,:]*input[:,2,:,:]
		square_sum = torch.sqrt(square_sum)
		square_sum += (square_sum==0).float()*0.0000000001
		output = input.clone()
		output[:,0,:,:]/=square_sum
		output[:,1,:,:]/=square_sum
		output[:,2,:,:]/=square_sum
		return output
		

class world_coord_to_normal(nn.Module):
	"""docstring for world_coord_to_normal"""
	def __init__(self):
		super(world_coord_to_normal, self).__init__()
		kernel_up = torch.zeros(3,3,3,3)
		kernel_up[0,0] = torch.Tensor([[[[0,-1,0],[0,0,0],[0,1,0]]]])
		kernel_up[1,1] = torch.Tensor([[[[0,-1,0],[0,0,0],[0,1,0]]]])
		kernel_up[2,2] = torch.Tensor([[[[0,-1,0],[0,0,0],[0,1,0]]]])
		kernel_left = torch.zeros(3,3,3,3)
		kernel_left[0,0] = torch.Tensor([[[[0,0,0],[-1,0,1],[0,0,0]]]])
		kernel_left[1,1] = torch.Tensor([[[[0,0,0],[-1,0,1],[0,0,0]]]])
		kernel_left[2,2] = torch.Tensor([[[[0,0,0],[-1,0,1],[0,0,0]]]])
		self.v_up = vectorize(kernel_up)
		self.v_left = vectorize(kernel_left)
		self.normalize = spatial_normalization()

	
	def forward(self,input):
		v_up = self.v_up(input)
		v_left = self.v_left(input)
		v_norm = torch.cross(v_up,v_left, dim = 1)
		return self.normalize(v_norm)

if __name__ == '__main__':
	# test
	input = torch.zeros(1,3,5,5)
	input[0,0,1,0] = -1
	input[0,0,1,2] = 2
	input[0,1,0,1] = -1
	input[0,1,2,1] = 1
	input = Variable(input.cuda())
	wtn = world_coord_to_normal().cuda()
	print(input)
	print(wtn(input))
