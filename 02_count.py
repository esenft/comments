from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Count extracted comments in a JSON file.",
	)
	parser.add_argument(
		"-i",
		"--input",
		default="comments.json",
		help="Path to comments JSON file (default: comments.json).",
	)
	parser.add_argument(
		"-e",
		"--expected",
		type=int,
		default=11193,
		help="Expected unique comment count (default: 11193).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = Path(args.input)

	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	data = json.loads(input_path.read_text(encoding="utf-8"))
	if not isinstance(data, list):
		raise ValueError("Expected a JSON array of comments.")

	total_entries = len(data)
	unique_comment_ids = {
		item.get("cid") if isinstance(item, dict) else None
		for item in data
	}
	unique_comment_ids.discard(None)
	unique_count = len(unique_comment_ids)

	print(f"Total comments (unique entries): {total_entries}")
	print(f"Unique comment IDs (cid): {unique_count}")
	print(f"Duplicate cid entries: {total_entries - unique_count}")

	if total_entries == args.expected:
		print(f"Matches expected comment count: {args.expected}")
	else:
		print(f"Does not match expected comment count: {args.expected}")


if __name__ == "__main__":
	main()
