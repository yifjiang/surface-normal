import hourglass

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.seq = nn.Sequential(
			hourglass.Model(),
			nn.Softplus(beta = 1, threshold=20)
			)

	def forward(self,x):
		return self.seq(x)

def get_model():
	return Model().cuda()

from .criterion.relative_depth import relative_depth_crit
def get_criterion():
	return relative_depth_crit()

def f_depth_from_model_output():
	print(">>>>>>>>>>>>>>>>>>>>>>>>>    depth = model_output")
	return ____get_depth_from_model_output

def ____get_depth_from_model_output(model_output):
	return model_output

		
if __name__ == '__main__':
	from torch import optim
	test = Model().cuda()
	print(test)
	x = Variable(torch.rand(1,3,320,320).cuda())
	print(x)
	print(test(x))
	target = {}
	target[0] = {}
	target[0]['x_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['y_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['x_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	target[0]['y_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	target[0]['ordianl_relation'] = Variable(torch.Tensor([0,1,-1,0,-1,1])).cuda()
	target[0]['n_point'] = 1
	c = get_criterion()
	loss = c(test(x),target)
	print(loss)
	o = optim.Adam(test.parameters())
	for i in range(0,30):
		o.zero_grad()
		loss.backward()
		o.step()
		loss = c(test(x),target)
		print(loss)