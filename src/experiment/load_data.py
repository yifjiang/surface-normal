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

if g_args.t_normal_file != '':
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

if (train_normal_path is None and valid_normal_path is not None) or (train_normal_path is not None and valid_normal_path is None):
	print(train_normal_path, valid_normal_path)
	print("Error: Either train_normal_path or valid_normal_path is not valid")
	assert(False)


def TrainDataLoader():
	_train_depth_path = train_depth_path
	_train_normal_path = train_normal_path
	if g_args.n_max_depth == 0:
		_train_depth_path = None
		print("\t\t>>>>>>>>>>>>Warning: No depth training data specified!")

	if g_args.n_max_normal == 0:
		_train_normal_path = None
		print("\t\t>>>>>>>>>>>>Warning: No normal training data specified!")

	if train_depth_path is None and train_normal_path is None:
		print(">>>>>>>>>	Error: Both normal data and depth data are nil!")
		assert(False)

	return DataLoader(_train_depth_path, _train_normal_path, g_args.n_max_depth, g_args.n_max_normal)

def ValidDataLoader():
	return DataLoader(valid_depth_path, valid_normal_path)

def Train_During_Valid_DataLoader():
	_n_max_depth = g_args.n_max_depth
	_n_max_normal = g_args.n_max_normal
	if g_args.n_max_depth == 0:
		_n_max_depth = 800
	if g_args.n_max_normal == 0:
		_n_max_normal = 5000

	return DataLoader(train_depth_path, train_normal_path, _n_max_depth, _n_max_normal)