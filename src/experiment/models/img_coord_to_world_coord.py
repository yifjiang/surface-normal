import torch
from torch import nn
from torch.autograd import Variable

class img_coord_to_world_coord(nn.Module):
	"""docstring for img_coord_to_world_coord"""
	def __init__(self, camera):
		super(img_coord_to_world_coord, self).__init__()
		self.camera = camera
		self.x_factor = Variable(torch.range(0,camera['input_width']).view(1,-1).repeat(camera['input_height'],1).cuda())
		self.y_factor = Variable(torch.range(0,camera['input_height']).view(-1,1).repeat(1,camera['input_width']).cuda())
		self.x_factor = (self.x_factor-camera['cx'])/camera['fx']
		self.y_factor = (self.y_factor-camera['cy'])/camera['fy']
		self.z_factor = Variable(torch.ones(camera['input_height'],camera['input_width']).cuda())
		self.per_batch_factor = Variable(torch.Tensor([self.x_factor,self.y_factor, self.z_factor]).cuda())

	def forward(input):
		output = Variable(torch.Tensor(input.size()[0],3,input.size()[2],input.size()[3]).cuda())
		output[:,0,:,:] = input
		output[:,1,:,:] = input
		output[:,2,:,:] = input
		factor = self.per_batch_factor.view(1,self.camera['input_height'], self.camera['input_width']).repeat(input.size()[0],1,1,1)
		return output*factor

if __name__ == '__main__':
	x = Variable(torch.ones(1,1,5,5).cuda())
	camera = {}
	camera['input_width'] = 5
	camera['input_height'] = 5
	camera['cx'] = 2
	camera['cy'] = 3
	camera['fx'] = 6
	camera['fy'] = -4
	img_to_world = img_coord_to_world_coord(camera)
	print(img_to_world(x))