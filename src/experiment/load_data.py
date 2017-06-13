import sys
train_depth_path = None
valid_depth_path = None

train_normal_path = None
valid_normal_path = None

folderpath = '../../data/'

if g_args.t_depth_file != '':
	train_depth_path = folderpath + g_args.t_depth_file

if g_args.v_depth_file != '':
	valid_depth_path = folderpath + g_args.v_depth_file

if g_args.t_normal_file != '';
	train_normal_path = folderpath + g_args.t_normal_file

if g_args.v_normal_file != '':
	valid_normal_path = folderpath + g_args.v_normal_file


if train_depth_path is None:
	print("Error: Missing training file for depth!")
	sys.exit(1)

if valid_depth_path is None:
	print("Error: Missing validation file for depth!")
	sys.exit(1)

if train_normal_path is None and train_depth_path is None:
	print("Error: No training files at all.")
	sys.exit(1)

if (train_normal_path is None and valid_normal_path is not None) or (train_normal_path is not None and valid_normal_path is not None):
	print("Error: Either train_normal_path or valid_normal_path is not valid")
	sys.exit(1)


def TrainDataLoader():
	_train_depth_path = train_depth_path
	_train_normal_path = train_normal_path
	return DataLoader(train_depth_path)

def ValidDataLoader():
	return DataLoader(valid_depth_path)