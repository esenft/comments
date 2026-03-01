from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
	from youtube_comment_downloader import YoutubeCommentDownloader
except ModuleNotFoundError as error:
	if error.name == "youtube_comment_downloader":
		raise SystemExit(
			"Missing dependency 'youtube-comment-downloader'.\n"
			"Install dependencies with:\n"
			f"  {sys.executable} -m pip install -r requirements.txt"
		) from error
	raise


def download_comments(
	url: str,
	limit: int | None = None,
	progress_every: int = 1000,
) -> list[dict]:
	downloader = YoutubeCommentDownloader()
	comments: list[dict] = []

	for index, comment in enumerate(downloader.get_comments_from_url(url), start=1):
		comments.append(comment)
		if progress_every > 0 and index % progress_every == 0:
			print(f"Downloaded {index} comments...", flush=True)
		if limit is not None and index >= limit:
			break

	return comments


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download comments from a YouTube video and save them as JSON.",
	)
	parser.add_argument(
		"url",
		nargs="?",
		default="https://www.youtube.com/watch?v=L_Guz73e6fw",
		help="YouTube video URL.",
	)
	parser.add_argument(
		"-o",
		"--output",
		default="comments.json",
		help="Path to output JSON file (default: comments.json).",
	)
	parser.add_argument(
		"-n",
		"--limit",
		type=int,
		default=1000,
		help="Maximum number of comments to download (default: 1000).",
	)
	parser.add_argument(
		"--progress-every",
		type=int,
		default=1000,
		help="Print progress every N comments (default: 1000, 0 to disable).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.limit is not None and args.limit <= 0:
		raise ValueError("--limit must be a positive integer.")
	if args.progress_every < 0:
		raise ValueError("--progress-every must be 0 or a positive integer.")

	comments = download_comments(
		url=args.url,
		limit=args.limit,
		progress_every=args.progress_every,
	)

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(comments, ensure_ascii=False, indent=2), encoding="utf-8")

	print(f"Downloaded {len(comments)} comments to {output_path}")


if __name__ == "__main__":
	main()
