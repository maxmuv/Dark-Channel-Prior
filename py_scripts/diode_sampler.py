import pathlib 
import shutil
import numpy as np
import argparse
from PIL import Image

def find_all_images(files_list):
	# Gets a list of files were found recursively and
	# selects all images from it
	return filter(lambda f: (f.suffix == ".png"), files_list)

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
		else:
			result.append(p)
	return result

def convert_numpy_map_to_img(p):
	# Gets path to image, opens image, map and mask and creates image of depth map
	im_stem = p.stem
	de_path = p.with_name(f"{im_stem}_depth.npy")
	de_mask_path = p.with_name(f"{im_stem}_depth_mask.npy")

	de = np.load(str(de_path)).squeeze()
	de_mask = np.load(str(de_mask_path))

	validity_mask = de_mask > 0
	MIN_DEPTH = 0.5
	MAX_DEPTH = min(300, np.percentile(de, 99))
	de = np.clip(de, MIN_DEPTH, MAX_DEPTH)
	filled_de = np.ma.array(de, mask=~validity_mask)
	filled_de = filled_de.filled(2*MAX_DEPTH)
	de_int = np.interp(filled_de, [filled_de.min(), filled_de.max()], [0, 255])
	return Image.fromarray(np.uint8(de_int), 'L')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="diode_sampler",
		description="Program creates dataset of 2 dirs with images and depth maps")
	parser.add_argument('outdoor_input_dir')
	parser.add_argument('output_dir')
	parser.add_argument('num_of_samples')
	args = parser.parse_args()
	outdoor_input_dir = pathlib.Path(args.outdoor_input_dir)
	output_dir = pathlib.Path(args.output_dir)
	num_of_samples = int(args.num_of_samples)
	if num_of_samples < 0:
		raise RuntimeError("num_of_samples has to be positive integer")
	if not output_dir.exists():
		raise RuntimeError("output_dir doesn't exist")
	if not output_dir.is_dir():
		raise RuntimeError("output_dir isn't a directory")
	if len(list(output_dir.iterdir())) > 0:
		raise RuntimeError("output_dir isn't empty")
	image_dir_path = output_dir / "images"
	maps_dir_path = output_dir / "maps"
	image_dir_path.mkdir()
	maps_dir_path.mkdir()
	files_list = recursive_dir_traversal(outdoor_input_dir)
	images_paths_list = find_all_images(files_list)
	idx = 0
	for ip in images_paths_list:
		if (idx >= num_of_samples):
			break
		dp = convert_numpy_map_to_img(ip)
		dp.save(str(maps_dir_path / ip.name))
		shutil.copy(str(ip), str(image_dir_path))
		idx += 1

