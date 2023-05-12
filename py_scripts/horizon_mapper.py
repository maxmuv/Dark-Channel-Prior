import pathlib
import argparse
import cv2

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

def result_map(m):
	max_height = []
	result = m.copy()
	last_idx = len(m) - 1 
	for c in range(len(m[0])):
		for r in range(len(m)):
			if result[last_idx - r, c] == 255 and len(max_height) == c:
				max_height.append(r)
			if len(max_height) == c+1:
				result[last_idx - r, c] = 255
		if len(max_height) == c:
			max_height.append(last_idx)
	for c in range(len(m[0])):
		for r in range(len(m)):
			if result[last_idx - r, c] != 255:
				result[last_idx - r, c] = int(255*r/max_height[c])
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="horizon_mapper",
		description="Program finds horizon on images and create depth maps")
	parser.add_argument('horizon_input_dir')
	parser.add_argument('output_dir')
	args = parser.parse_args()
	horizon_input_dir = pathlib.Path(args.horizon_input_dir)
	output_dir = pathlib.Path(args.output_dir)
	input_dir_files = recursive_dir_traversal(horizon_input_dir)
	for f in input_dir_files:
		grayscale_image = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
		blur = cv2.GaussianBlur(grayscale_image,(51,51),0)
		ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th , 4 , cv2.CV_32S)
		max_area = 0
		idx = -1 
		for i in range(1, num_labels):
			area = stats[i, cv2.CC_STAT_AREA]
			if area > max_area:
				max_area = area
				idx = i
		mask = (labels == idx).astype("uint8")*255
		comp = cv2.bitwise_and(th, th, mask=mask)
		res = result_map(comp)
		cv2.imwrite(str(output_dir / f.name), res)