import torch
from torch import nn
import gc
from PIL import Image
from torchvision import transforms
from common import *
from math import sqrt

def angle_diff(input, target):
	x_arr = target['x'].data
	y_arr = target['y'].data

	normal_arr = input.index_select(2,x_arr.long()).gather(1, y_arr.view(1,-1).long().repeat(3,1).view(3,1,-1)).squeeze()
	ground_truth_arr = target['normal']

	cos = torch.sum(normal_arr*ground_truth_arr.data, dim=0)

	mask_gt1 = (cos>1).float()
	cos = cos*(1-mask_gt1)+mask_gt1
	mask_lt_1 = (cos<-1).float()
	cos = cos*(1-mask_lt_1)-mask_lt_1

	acos = torch.acos(cos)

	if torch.sum(mask_gt1) > 0:
		print(">>>>>>>> Greater than 1 cos!")
	if torch.sum(mask_lt_1) > 0:
		print(">>>>>>>> Less than -1 cos!")

	return torch.sum(acos)


def _classify(z_A,z_B,ground_truth,thresh):
	if z_A - z_B > thresh:#z_A, z_B should be primitive type
		_classify_res = 1
	elif z_A - z_B < -thresh:
		_classify_res = -1
	elif z_A - z_B <= thresh and z_A - z_B >= -thresh: #this may be unecessary
		_classify_res = 0
	else:
		print("z_A - z_B exception")
		assert(False)
	return (_classify_res == ground_truth)#note the type of the return value!

def _count_correct(output, target, record):
	for point_idx in range(0,target['n_point']):
		x_A = target['x_A'][point_idx]
		y_A = target['y_A'][point_idx]
		x_B = target['x_B'][point_idx]
		y_B = target['y_B'][point_idx]

		z_A = output[0,0, y_A.data.int()[0], x_A.data.int()[0]] #zero-indexed
		z_B = output[0,0, y_B.data.int()[0], x_B.data.int()[0]]

		assert(torch.sum(x_A != x_B).data[0] or torch.sum(y_A != y_B).data[0])

		ground_truth = target['ordianl_relation'][point_idx].data[0]

		for tau_idx in range(0,record['n_thresh']):
			if _classify(z_A, z_B, ground_truth, record['thresh'][tau_idx]):
				if ground_truth == 0:
					record['eq_correct_count'][tau_idx] += 1
				elif ground_truth == 1 or ground_truth == -1: #this may be unnecessary
					record['not_eq_correct_count'][tau_idx] += 1

		if ground_truth == 0:
			# print(ground_truth)
			record['eq_count'] += 1
		else:
			record['not_eq_count'] += 1

def normalize_with_mean_std(input, mean, std):
	normed_input = input.clone()
	normed_input -= torch.mean(normed_input)
	normed_input /= torch.std(normed_input)
	normed_input *= std
	normed_input += mean

	if torch.sum(normed_input<=0)>0:
		normed_input[normed_input<=0] = torch.min(normed_input[normed_input>0])+0.00001

	return normed_input

def visulize_depth(z, filename):
	_z_img = z.clone()
	_z_img-=torch.min(_z_img)
	_z_img/=torch.max(_z_img)
	t_back = transforms.ToPILImage()
	_z_img = t_back(_z_img)
	new_img = Image.new('RGB', _z_img.size()[1], _z_img.size()[0])
	new_img.paste(_z_img, (0,0))
	new_img.save(filename)

def metric_error(gt_z, z):
	scale = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Scale((640, 480)),
		transforms.ToTensor()
		])
	z_min = torch.min(z)
	z_range = (torch.max(z) - z_min)
	# print(z.size())
	resize_z = (z - z_min)/z_range
	resize_z = scale(resize_z.cpu())
	resize_z = (resize_z*z_range)+z_min
	resize_z = resize_z.squeeze().cuda()

	std_of_NYU_training = 0.6148231626
	mean_of_NYU_training = 2.8424594402

	_crop = 16
	# print(gt_z.size())
	gt_z = gt_z[_crop:(480 - _crop), _crop:(640 - _crop)].data
	# print(resize_z.size())
	resize_z = resize_z[_crop:(480 - _crop), _crop:(640 - _crop)]

	normed_NYU_z = normalize_with_mean_std(resize_z, mean_of_NYU_training, std_of_NYU_training)

	# print(gt_z,normed_NYU_z)
	fmse = torch.mean(torch.pow(gt_z - normed_NYU_z, 2))
	flsi = torch.mean(torch.pow(torch.log(normed_NYU_z) - torch.log(gt_z) + torch.mean(torch.log(normed_NYU_z) - torch.log(gt_z)), 2))

	gt_mean = torch.mean(gt_z)
	gt_std = torch.std(gt_z-gt_mean)
	normed_gt_z = normalize_with_mean_std(resize_z, gt_mean, gt_std)
	fmse_si = torch.mean(torch.pow(gt_z - normed_gt_z, 2))

	return fmse, flsi, fmse_si


_eval_record = {}
_eval_record['n_thresh'] = 140
_eval_record['eq_correct_count'] = torch.Tensor(_eval_record['n_thresh'])
_eval_record['not_eq_correct_count'] = torch.Tensor(_eval_record['n_thresh'])
_eval_record['not_eq_count'] = 0
_eval_record['eq_count'] = 0
_eval_record['thresh'] = torch.Tensor(_eval_record['n_thresh'])
_eval_record['WKDR'] = torch.Tensor(_eval_record['n_thresh'], 4)
WKDR_step = 0.01
for i in range(0,_eval_record['n_thresh']):
	_eval_record['thresh'][i] = float(i)*WKDR_step+0.1

print('>>>>>> Validation: margin = {}, WKDR Step = {}'.format(g_args.margin, WKDR_step))
# print(_eval_record['thresh'])


def reset_record(record):
	record['eq_correct_count'].fill_(0)
	record['not_eq_correct_count'].fill_(0)
	record['WKDR'].fill_(0)
	record['not_eq_count'] = 0
	record['eq_count'] = 0

def evaluate(data_loader, model, criterion, max_n_sample):
	print('>>>>>>>>>>>>>>>>>>>>>>>>> Valid Crit Threshed: Evaluating on validation set...')
	print('Evaluate() Switch On!!!')
	# model.evaluate()
	reset_record(_eval_record)

	total_depth_validation_loss = 0
	total_normal_validation_loss = 0
	total_angle_difference = 0
	n_depth_iters = min(data_loader.n_relative_depth_sample, max_n_sample)
	n_normal_iters = min(data_loader.n_normal_sample, max_n_sample)
	n_total_depth_point_pair = 0
	n_total_normal_point = 0

	fmse = torch.zeros(n_depth_iters)
	fmse_si = torch.zeros(n_depth_iters)
	flsi = torch.zeros(n_depth_iters)

	print("Number of relative depth samples we are going to examine: {}".format(n_depth_iters))
	print("Number of normal samples we are going to examine: {}".format(n_normal_iters))

	for i in range(0,n_depth_iters):
		print(i)
		batch_input, batch_target = data_loader.load_indices(torch.Tensor([i]), None, True) 

		relative_depth_target = batch_target[0][0]

		# print(i)
		batch_output = model.forward(batch_input)
		batch_loss = criterion.forward(batch_output, batch_target).data

		# output_depth = get_depth_from_model_output(batch_output) #get_depth_from_model_output is from main.py
		output_depth = batch_output.data #temporary solution

		_count_correct(output_depth, relative_depth_target, _eval_record)

		total_depth_validation_loss += (batch_loss * relative_depth_target['n_point'])[0]

		n_total_depth_point_pair += relative_depth_target['n_point']

		fmse[i], flsi[i], fmse_si[i] = metric_error(batch_target[0]['gt_depth'][0], output_depth[0])

		gc.collect()
	
	print('Evaluate() depth Switch Off!!!')
	# model.training()

	from models.criterion.img_coord_to_world_coord import img_coord_to_world_coord
	from models.criterion.world_coord_to_normal import world_coord_to_normal

	camera = {
	'input_width' : g_input_width,
	'input_height' : g_input_height,
	'cx' : g_cx_rgb,
	'cy' : g_cy_rgb,
	'fx' : g_fx_rgb,
	'fy' : g_fy_rgb
	}
	_depth_to_normal = nn.Sequential(
		img_coord_to_world_coord(camera),
		world_coord_to_normal()
		).cuda()

	for iter in range(0,n_normal_iters):
		print(iter)
		batch_input, batch_target = data_loader.load_indices(None, torch.Tensor([iter]), True)

		if g_args.n_scale == 1:
			normal_target = batch_target[1][0]
		else:
			normal_target = batch_target[1][0][0]

		batch_output = model.forward(batch_input)

		batch_loss = criterion.forward(batch_output, batch_target).data
		normal = _depth_to_normal(batch_output).data

		sum_angle_diff = angle_diff(normal[0], normal_target)
		total_angle_difference += sum_angle_diff

		total_normal_validation_loss += batch_loss*normal_target['n_point']

		n_total_normal_point += normal_target['n_point']

		if n_depth_iters == 0:
			fmse[iter], flsi[iter], fmse_si[iter] = metric_error(batch_target[1].gt_depth[0], batch_output[0])

	print("Evaluate() Switch Off!!!")

	max_min = 0
	max_min_i = 1
	for tau_idx in range(0,_eval_record['n_thresh']):
		_eval_record['WKDR'][tau_idx, 0] = _eval_record['thresh'][tau_idx]
		_eval_record['WKDR'][tau_idx, 1] = float(_eval_record['eq_correct_count'][tau_idx]+_eval_record['not_eq_correct_count'][tau_idx])/float(_eval_record['eq_count'] + _eval_record['not_eq_count'])
		_eval_record['WKDR'][tau_idx, 2] = float(_eval_record['eq_correct_count'][tau_idx])/float( _eval_record['eq_count'])
		# print(_eval_record['eq_count'], _eval_record['not_eq_count'])
		_eval_record['WKDR'][tau_idx, 3] = float(_eval_record['not_eq_correct_count'][tau_idx])/float(_eval_record['not_eq_count'])

		if min(_eval_record['WKDR'][tau_idx,2], _eval_record['WKDR'][tau_idx, 3])>max_min:
			max_min = min(_eval_record['WKDR'][tau_idx,2], _eval_record['WKDR'][tau_idx, 3])
			max_min_i = tau_idx

	print(_eval_record['WKDR'])
	print(_eval_record['WKDR'][max_min_i])
	print('\tEvaluation Completed. Average Relative Depth Loss = {}, WKDR = {}'.format(total_depth_validation_loss/n_total_depth_point_pair, 1 - max_min))
	print('Evaluation Completed. Average Normal Loss = {}, Average Angular Difference = {}'.format(total_normal_validation_loss/n_total_normal_point, total_angle_difference/n_total_normal_point))
	rmse = sqrt(torch.mean(fmse))
	rmse_si = sqrt(torch.mean(fmse_si))
	lsi = sqrt(torch.mean(flsi))
	print('\trmse:\t{}'.format(rmse))
	print('\trmse_si:\t{}'.format(rmse_si))
	print('\tlsi:\t{}'.format(lsi))
	# print("\tEvaluation Completed. Loss = {}, WKDR = {}".format(total_validation_loss, 1 - max_min))

	return float(total_depth_validation_loss) / float(n_total_depth_point_pair), 1 - max_min, total_normal_validation_loss/n_total_normal_point, total_angle_difference/n_total_normal_point, rmse, rmse_si, lsi
