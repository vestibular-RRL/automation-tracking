# Replace main.py content with:
import argparse
import os
import sys
from process_video import process_video_pipeline

def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Process all .mp4 videos in a folder: crop, split, track, and combine."
	)
	parser.add_argument("folder", nargs="?", help="Folder containing .mp4 files")
	parser.add_argument("--folders", help="Text file with list of folders to process (one per line)")
	parser.add_argument("--model", required=True, help="Path to YOLO model (e.g., model\\best copy.pt)")
	parser.add_argument("--out", default=".", help="Output directory (default: current working dir)")
	parser.add_argument("--csv", help="Path to annotation CSV file with Frame# and Annotation columns")
	parser.add_argument("--test", action="store_true", help="Process only the first video and print outputs")
	return parser.parse_args(argv)

def process_folder(folder, args):
	if not os.path.isdir(folder):
		print(f"Error: Not a directory: {folder}", file=sys.stderr)
		return 0

	os.makedirs(args.out, exist_ok=True)

	mp4s = [
		n for n in sorted(os.listdir(folder))
		if n.lower().endswith('.mp4')
		and '_cropped' not in n
		and '_combined_traced' not in n
		and '_left' not in n
		and '_right' not in n
	]
	if not mp4s:
		print(f"No .mp4 files found in the provided folder: {folder}")
		return 0

	processed = 0
	for idx, name in enumerate(mp4s):
		in_path = os.path.join(folder, name)
		base_name = os.path.splitext(name)[0]
		csv_path = os.path.join(folder, base_name + ".csv")
		annotation_csv = csv_path if os.path.exists(csv_path) else args.csv
		try:
			results = process_video_pipeline(
				in_path, args.model, output_dir=args.out, annotation_csv_path=annotation_csv
			)
			print(f"[✓] Done: {name}")
			print(f" - left_csv: {results['left_csv']}")
			print(f" - right_csv: {results['right_csv']}")
			processed += 1
			if args.test:
				print("[test mode] Only processed the first video.")
				break
		except Exception as e:
			print(f"[x] Failed: {name} -> {e}", file=sys.stderr)
	return processed

def main(argv: list[str]) -> int:
	args = parse_args(argv)
	total_processed = 0

	if args.folders:
		if not os.path.isfile(args.folders):
			print(f"Error: Not a file: {args.folders}", file=sys.stderr)
			return 1
		with open(args.folders, "r", encoding="utf-8") as f:
			folders = [line.strip() for line in f if line.strip()]
		for folder in folders:
			print(f"\n[Batch] Processing folder: {folder}")
			total_processed += process_folder(folder, args)
			if args.test:
				break
	else:
		if not args.folder:
			print("Error: Must specify either a folder or --folders", file=sys.stderr)
			return 1
		total_processed += process_folder(args.folder, args)

	return 0

if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))