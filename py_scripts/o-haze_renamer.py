import argparse
import pathlib 
import os

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="O-HAZE-renamer",
		description="program renames files in o-haze dir and subdir")
	parser.add_argument("o_haze_dir")
	args = parser.parse_args()
	o_haze_dir = pathlib.Path(args.o_haze_dir)
	files_dirs_list = recursive_dir_traversal(o_haze_dir)
	for f in files_dirs_list:
		if f.is_dir():
			continue
		old_name = f.name
		new_name = "".join(old_name.split("_")[:-1]) + f.suffix
		new_f = f
		new_f = new_f.with_name(new_name)
		os.rename(str(f), str(new_f))