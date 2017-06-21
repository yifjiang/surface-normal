import torch
from torch import nn
from torch.autograd import Variable

class ABC_sum(nn.Module):
	"""docstring for ABC_sum"""
	def __init__(self, mode, camera):
		super(ABC_sum, self).__init__()

		self.constant_X_Y_1 = Variable(torch.Tensor(3,camera['input_height'], camera['input_width']).cuda())

		_offset_x = 0
		_offset_y = 0
		if mode == 'center':
			pass
		elif mode == 'left':
			_offset_x = -1
		elif mode == 'right':
			_offset_x = 1
		elif mode == 'up':
			_offset_y = -1
		elif mode == 'down':
			_offset_y = 1

		for y in range(0, camera['input_height']):
			for x in range(0, camera['input_width']):
				self.constant_X_Y_1[0,y,x] = (x+_offset_x - camera['cx'])/camera['fx']
				self.constant_X_Y_1[1,y,x] = (y+_offset_y - camera['cy'])/camera['fy']

		self.constant_X_Y_1[2].data.fill_(1.0)

	def forward(self, input):
		output = Variable(torch.Tensor(input.size()[0],1,input.size()[2],input.size()[3]).cuda())
		for batch_idx in range(0,input.size()[0]):
			output[batch_idx] = torch.sum(self.constant_X_Y_1*input[batch_idx], 0)
		
		return output

class elementwise_div(nn.Module):
	"""docstring for elementwise_div"""
	def __init__(self):
		super(elementwise_div, self).__init__()
	
	def forward(self, input):
		output = input[0]/input[1]
		zero_mask = (input[1]==0)
		output[zero_mask] = 0
		return output

class elementwise_mul(nn.Module):
	"""docstring for elementwise_mul"""
	def __init__(self):
		super(elementwise_mul, self).__init__()

	def forward(self, input):
		output = input[0]*input[1]
		return output
		
class elementwise_shift(nn.Module):
	"""docstring for elementwise_shift"""
	def __init__(self, mode):
		super(elementwise_shift, self).__init__()
		self.mode = mode

	def forward(self, input):
		output = Variable(torch.zeros(input.size()[0],1,input.size()[2],input.size()[3]).cuda())
		width = input.size()[3]
		height = input.size()[2]
		if self.mode == 'left':
			output[:,:,:,0:(width-1)] = input[:,:,:,1:width]
		elif self.mode == 'right':
			output[:,:,:,1:width] = input[:,:,:,0:(width-1)]
		elif self.mode == 'up':
			output[:,:,0:(height-1),:] = input[:,:,1:height,:]
		elif self.mode == 'down':
			output[:,:,1:height,:] = input[:,:,0:(height-1),:]

		return output

class elementwise_concat(nn.Module):
	"""docstring for elementwise_concat"""
	def __init__(self):
		super(elementwise_concat, self).__init__()
	
	def forward(self, input):
		output = Variable(torch.zeros(input[0].size()[0], len(input) * input[0].size(1),input[0].size()[2],input[0].size()[3])).cuda()
		for i in range(0,len(input)):
			output[:,(input[0].size()[1]*i):(input[0].size(1)*(i+1)),:,:] = input[i]

		return output

class Model(nn.Module):
	"""docstring for Model"""
	def __init__(self, camera):
		super(Model, self).__init__()
		self.ABC_right = ABC_sum('right',camera)
		self.ABC_left = ABC_sum('left',camera)
		self.ABC_down = ABC_sum('down',camera)
		self.ABC_up = ABC_sum('up', camera)

		self.z_i_left = elementwise_shift('right')
		self.z_i_right = elementwise_shift('left')
		self.z_i_up = elementwise_shift('down')
		self.z_i_down = elementwise_shift('up')

		self.div = elementwise_div()
		self.mul = elementwise_mul()
		self.z_o_right = elementwise_shift('right')

		self.z_o_left = elementwise_shift('left')

		self.z_o_down = elementwise_shift('down')

		self.z_o_up = elementwise_shift('up')

		self.z_concat = elementwise_concat()

	def forward(self, depth_input, normal_input):
		ABC_right = self.ABC_right(normal_input)
		ABC_left = self.ABC_left(normal_input)
		ABC_up = self.ABC_up(normal_input)
		ABC_down = self.ABC_down(normal_input)

		z_i_left = self.z_i_left(depth_input)
		z_i_right = self.z_i_right(depth_input)
		z_i_up = self.z_i_up(depth_input)
		z_i_down = self.z_i_down(depth_input)

		left_div_right = self.div([ABC_left, ABC_right])
		D_right = self.mul([left_div_right, z_i_left])
		z_o_right = self.z_o_right(D_right)

		right_div_left = self.div([ABC_right, ABC_left])
		D_left = self.mul([right_div_left, z_i_right])
		z_o_left = self.z_o_left(D_left)

		up_div_down = self.div([ABC_up, ABC_down])
		D_down = self.mul([up_div_down, z_i_up])
		z_o_down = self.z_o_down(D_down)
		
		down_div_up = self.div([ABC_down, ABC_up])
		D_up = self.mul([down_div_up, z_i_down])
		z_o_up = self.z_o_up(D_up)

		z_concat = self.z_concat([z_o_up, z_o_down, z_o_left, z_o_right])

		return z_concat
		
def get_theoretical_depth(camera):
	print("get_theoretical_depth_v2")
	return Model(camera)

class Mask(nn.Module):
	"""docstring for Mask"""
	def __init__(self):
		super(Mask, self).__init__()
		self.z_o_right = elementwise_shift('right')
		self.z_o_left = elementwise_shift('left')
		self.z_o_down = elementwise_shift('down')
		self.z_o_up = elementwise_shift('up')
		self.concat = elementwise_concat()

	def forward(self, input):
		mask_o_left = self.z_o_left(input)
		mask_o_right = self.z_o_right(input)
		mask_o_up = self.z_o_up(input)
		mask_o_down = self.z_o_down(input)
		mask_concat = self.concat([mask_o_up, mask_o_down, mask_o_left, mask_o_right])

		return mask_concat

def get_shift_mask():
	return Mask()

if __name__ == '__main__':
	camera = {
		'input_width' : 4,
		'input_height' : 5,
		'cx' : 1,
		'cy' : -2,
		'fx' : 2,
		'fy' : 3
		}
	normal_to_depth = get_theoretical_depth(camera).cuda()
	input = Variable(torch.rand(1,1,5,4)).cuda()
	normal = Variable(torch.rand(1,3,5,4)).cuda()
	print(normal_to_depth(input,normal))

	shift_mask = get_shift_mask()
	