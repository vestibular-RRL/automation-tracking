import os
import sys


def list_files_in_directory(directory_path: str) -> int:
	"""Prints all files directly inside the given directory. Returns exit code."""
	if not os.path.exists(directory_path):
		print(f"Error: Path does not exist: {directory_path}", file=sys.stderr)
		return 1

	if not os.path.isdir(directory_path):
		print(f"Error: Path is not a directory: {directory_path}", file=sys.stderr)
		return 1
                                                                                       
	try:
		entries = os.listdir(directory_path)
	except OSError as exc:
		print(f"Error: Unable to list directory '{directory_path}': {exc}", file=sys.stderr)
		return 1

	# Filter to files only (non-recursive) and sort for stable output
	files_only = [name for name in entries if os.path.isfile(os.path.join(directory_path, name))]
	return files_only