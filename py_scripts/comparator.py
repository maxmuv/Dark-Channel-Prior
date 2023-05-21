from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import argparse
import pathlib 

def recursive_dir_traversal(input_dir):
	# Returns all files in directory and subdirectories
	input_dir = pathlib.Path(input_dir)
	if not input_dir.exists():
		raise RuntimeError("input_dir doesn't exist")
	if not input_dir.is_dir():
		raise RuntimeError("input_dir isn't a directory")
	result = []
	for p in input_dir.iterdir():
		if p.is_dir():
			to_merge = recursive_dir_traversal(p)
			result.extend(to_merge)
		elif p.name.find("_tr") != -1 or p.name.find("_dc") != -1:
			continue
		else:
			result.append(p)
	return result

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="Comparator",
		description="Program calculate ssim and mse")
	parser.add_argument("lhs_dir")
	parser.add_argument("rhs_dir")
	args = parser.parse_args()
	lhs_dir = pathlib.Path(args.lhs_dir)
	rhs_dir = pathlib.Path(args.rhs_dir)
	lhs_ims = recursive_dir_traversal(lhs_dir)
	rhs_ims = recursive_dir_traversal(rhs_dir)
	rhs_ims_names = [right.name for right in rhs_ims]
	pairs = [(left, rhs_ims[rhs_ims_names.index(left.name)]) for left in lhs_ims]
	im_num = 0
	sum_ssim = 0
	sum_mse = 0
	for p in pairs:
		lhs = imread(str(p[0]))
		rhs = imread(str(p[1]))
		ssim_measure = ssim(lhs, rhs, data_range=255, channel_axis=2)
		mse_measure = mean_squared_error(lhs, rhs)
		print(p[0].name, ":",ssim_measure, mse_measure)
		im_num += 1
		sum_ssim += ssim_measure
		sum_mse += mse_measure
	print("Average measures: ssim ", sum_ssim / im_num, " ,mse " , sum_mse /im_num)
