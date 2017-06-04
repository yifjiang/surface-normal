import torch
from torch import nn
from torch.autograd import Variable

class relative_depth_crit(nn.Module):

	def __loss_func_arr(self, z_A, z_B, ground_truth):
		mask = torch.abs(ground_truth)#ground_truth[i] is either -1,0 or 1
		z_A = z_A[0]#may not need this
		z_B = z_B[0]
		z_A_z_B = z_A-z_B
		return mask*torch.log(1+torch.exp(-torch.min(ground_truth*z_A_z_B, margin)))+(1-mask)*torch.max(z_A_z_B*z_A_z_B, margin*margin)

	def __init__(self, margin):
		super(relative_depth_crit, self).__init__()
		self.margin = margin

	def forward(self, input, target):
		self.output = Variable(torch.Tensor([0])).cuda()
		n_point_total = 0
		cpu_input = input
		for batch_idx in range(0,cpu_input.size()[0]):
			n_point_total+=target[batch_idx]['n_point']

			x_A_arr = target[batch_idx]['x_A']
			y_A_arr = target[batch_idx]['y_A']
			x_B_arr = target[batch_idx]['x_B']
			y_B_arr = target[batch_idx]['y_B']

			batch_input = cpu_input[batch_idx, 0]
			z_A_arr = batch_input.index_select(1, x_A_arr.long()).gather(0, y_A_arr.view(1,-1).long())
			z_B_arr = batch_input.index_select(1, x_B_arr.long()).gather(0, y_B_arr.view(1,-1).long())

			ground_truth_arr = target[batch_idx]['ordianl_relation']
			self.output += torch.sum(self.__loss_func_arr(z_A_arr, z_B_arr, ground_truth_arr))

		return self.output/n_point_total#n_point_total should be of type int or IntTensor

if __name__ == '__main__':
	crit = relative_depth_crit()
	print(crit)
	x = Variable(torch.zeros(1,1,6,6).cuda(), requires_grad = True)
	target = {}
	target[0] = {}
	target[0]['x_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['y_A'] = Variable(torch.Tensor([0,1,2,3,4,5])).cuda()
	target[0]['x_B'] = Variable(torch.Tensor([0,0,0,0,0,0])).cuda()
	target[0]['y_B'] = Variable(torch.Tensor([5,4,3,2,1,0])).cuda()
	target[0]['ordianl_relation'] = Variable(torch.Tensor([-1,0,1,1,-1,-1])).cuda()
	target[0]['n_point'] = 6
	loss = crit.forward(x,target)
	print(loss)
	loss.backward()
	print(x.grad)
