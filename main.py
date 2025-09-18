# Replace main.py content with:
import argparse
import os
import sys
from process_video import process_video_pipeline

def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Process all .mp4 videos in a folder: crop, split, track, and combine."
	)
	parser.add_argument("folder", help="Folder containing .mp4 files")
	parser.add_argument("--model", required=True, help="Path to YOLO model (e.g., model\\best copy.pt)")
	parser.add_argument("--out", default=".", help="Output directory (default: current working dir)")
	parser.add_argument("--csv", help="Path to annotation CSV file with Frame# and Annotation columns")
	return parser.parse_args(argv)

def main(argv: list[str]) -> int:
	args = parse_args(argv)
	if not os.path.isdir(args.folder):
		print(f"Error: Not a directory: {args.folder}", file=sys.stderr)
		return 1

	os.makedirs(args.out, exist_ok=True)

	mp4s = [n for n in sorted(os.listdir(args.folder)) if n.lower().endswith('.mp4')]
	if not mp4s:
		print("No .mp4 files found in the provided folder.")
		return 0

	processed = 0
	for name in mp4s:
		in_path = os.path.join(args.folder, name)
		# Auto-detect CSV in same directory with same base name
		base_name = os.path.splitext(name)[0]
		csv_path = os.path.join(args.folder, base_name + ".csv")
		annotation_csv = csv_path if os.path.exists(csv_path) else args.csv
		try:
			results = process_video_pipeline(in_path, args.model, output_dir=args.out, annotation_csv_path=annotation_csv)
			print(f"[✓] Done: {name}")
			for k, v in results.items():
				print(f" - {k}: {v}")
			processed += 1
		except Exception as e:
			print(f"[x] Failed: {name} -> {e}", file=sys.stderr)

	return 0
	return 0

if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))