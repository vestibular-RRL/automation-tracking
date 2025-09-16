import argparse
import os
import sys
from folder_processor import list_files_in_directory
from video_crop_utils import crop_video

def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Print all files directly inside the specified folder",
	)
	parser.add_argument(
		"folder",
		help="Path to the folder whose files will be listed",
	)
	return parser.parse_args(argv)


def main(argv: list[str]) -> int:
	args = parse_args(argv)
	files = list_files_in_directory(args.folder)
	if isinstance(files, list):
		for file in files:
			print(file)
		return 0
	else:
		# Error case - files is an exit code
		return files

if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))


