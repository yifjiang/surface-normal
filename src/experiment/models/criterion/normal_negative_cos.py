import torch
from torch import nn
from torch.autograd import Variable

class normal_negative_cos_crit(nn.Module):
	"""docstring for normal_negative_cos_crit"""
	def __init__(self):
		super(normal_negative_cos_crit, self).__init__()

	def forward(self, input, target):
		"""input shape: [batch_size, 3, y, x]; normal shape: [3,n_point]"""
		output = Variable(torch.Tensor([0])).cuda()
		n_points = 0
		for batch_idx in range(0,input.size()[0]):
			n_points += target[batch_idx]['n_point']
			x_arr = target[batch_idx]['x']
			y_arr = target[batch_idx]['y']
			batch_input = input[batch_idx]
			normal_arr =batch_input.index_select(2,x_arr.long()).gather(1, y_arr.view(1,-1).long().repeat(3,1).view(3,1,-1)).squeeze()
			ground_truth_arr = target[batch_idx]['normal']
			# print(normal_arr)
			# print(ground_truth_arr)
			# assert(normal_arr.size() == ground_truth_arr.size())
			# print(normal_arr*ground_truth_arr)
			output-=torch.sum(normal_arr*ground_truth_arr)

		# print(n_points)
		# print((output/n_points).data[0])
		return output/n_points

if __name__ == '__main__':
	#test
	crit = normal_negative_cos_crit()
	input = Variable(torch.ones(1,3,5,5).cuda(), requires_grad = True)
	target = {}
	target[0] = {}
	target[0]['x'] = Variable(torch.Tensor([0,1])).cuda()
	target[0]['y'] = Variable(torch.Tensor([0,0])).cuda()
	target[0]['normal'] = Variable(torch.Tensor([[1,0],[0,1],[0,0]])).cuda()
	target[0]['n_point'] = 2
	loss = crit(input, target)
	print(loss)
	loss.backward()
	print(input.grad)
