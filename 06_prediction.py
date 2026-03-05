from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Classify first N comments with Ollama and update SQLite fields.",
	)
	parser.add_argument("--database", default="comments.db", help="SQLite database path.")
	parser.add_argument("--table", default="comments", help="Table name (default: comments).")
	parser.add_argument(
		"--prompt-file",
		default="prompt.txt",
		help="Prompt template path with {{COMMENT_TEXT}} placeholder.",
	)
	parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
	parser.add_argument("--model", default="llama3.2", help="Ollama model name.")
	parser.add_argument("--limit", type=int, default=100, help="Number of first comments to classify.")
	parser.add_argument(
		"--request-timeout",
		type=int,
		default=60,
		help="Timeout per Ollama request in seconds (default: 60).",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=2,
		help="Retries per comment if a request fails (default: 2).",
	)
	parser.add_argument(
		"--progress-every",
		type=int,
		default=5,
		help="Print progress every N comments (default: 5).",
	)
	return parser.parse_args()


def quote_identifier(identifier: str) -> str:
	escaped = identifier.replace('"', '""')
	return f'"{escaped}"'


def post_json(host: str, path: str, payload: dict, timeout_seconds: int) -> dict:
	request = urllib.request.Request(
		url=f"{host.rstrip('/')}{path}",
		data=json.dumps(payload).encode("utf-8"),
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
		return json.loads(response.read().decode("utf-8"))


def ask_ollama_for_labels(host: str, model: str, prompt: str, request_timeout: int) -> dict:
	try:
		body = post_json(
			host,
			"/api/chat",
			{
				"model": model,
				"messages": [{"role": "user", "content": prompt}],
				"stream": False,
				"format": "json",
				"options": {"temperature": 0, "num_predict": 64},
			},
			request_timeout,
		)
		text = body.get("message", {}).get("content", "").strip()
		if text:
			return parse_labels(text)
	except urllib.error.HTTPError as error:
		if error.code != 404:
			error_body = error.read().decode("utf-8", errors="replace")
			raise RuntimeError(f"Ollama request failed ({error.code}): {error_body}") from error
	except urllib.error.URLError as error:
		raise RuntimeError("Could not reach Ollama. Start it with: ollama serve") from error

	try:
		body = post_json(
			host,
			"/v1/chat/completions",
			{
				"model": model,
				"messages": [{"role": "user", "content": prompt}],
				"response_format": {"type": "json_object"},
				"temperature": 0,
				"max_tokens": 64,
			},
			request_timeout,
		)
		choices = body.get("choices", [])
		if choices:
			text = choices[0].get("message", {}).get("content", "").strip()
			if text:
				return parse_labels(text)
	except urllib.error.HTTPError as error:
		error_body = error.read().decode("utf-8", errors="replace")
		raise RuntimeError(
			f"Ollama request failed ({error.code}): {error_body}. "
			f"If needed, run: ollama pull {model}"
		) from error

	raise ValueError("Model returned empty classification output.")


def parse_labels(raw_text: str) -> dict:
	# Some models may add explanation text; extract the first JSON object.
	start = raw_text.find("{")
	end = raw_text.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError(f"Could not parse JSON from model output: {raw_text}")

	data = json.loads(raw_text[start : end + 1])
	if not isinstance(data, dict):
		raise ValueError("Model output is not a JSON object.")

	required = ("angry", "negative", "response", "spam")
	labels: dict[str, bool] = {}
	for key in required:
		if key not in data:
			raise ValueError(f"Missing '{key}' in model output: {data}")
		value = data[key]
		if isinstance(value, bool):
			labels[key] = value
			continue
		if isinstance(value, (int, float)):
			labels[key] = bool(value)
			continue
		if isinstance(value, str):
			normalized = value.strip().lower()
			if normalized in {"true", "yes", "1"}:
				labels[key] = True
				continue
			if normalized in {"false", "no", "0"}:
				labels[key] = False
				continue
		raise ValueError(f"Invalid value for '{key}': {value}")

	return labels


def main() -> None:
	args = parse_args()

	if args.limit <= 0:
		raise ValueError("--limit must be a positive integer.")
	if args.request_timeout <= 0:
		raise ValueError("--request-timeout must be a positive integer.")
	if args.max_retries < 0:
		raise ValueError("--max-retries must be 0 or a positive integer.")
	if args.progress_every <= 0:
		raise ValueError("--progress-every must be a positive integer.")

	prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
	if "{{COMMENT_TEXT}}" not in prompt_template:
		raise ValueError("Prompt file must include the placeholder '{{COMMENT_TEXT}}'.")

	db_path = Path(args.database)
	if not db_path.exists():
		raise FileNotFoundError(f"Database not found: {db_path}")

	table_name = quote_identifier(args.table)

	with sqlite3.connect(db_path) as connection:
		cursor = connection.cursor()
		cursor.execute(
			f"SELECT rowid, text, response FROM {table_name} ORDER BY rowid ASC LIMIT ?",
			(args.limit,),
		)
		all_rows = cursor.fetchall()
		comments = [
			(rowid, text)
			for rowid, text, response in all_rows
			if str(response).strip().lower() not in {"0", "1", "true", "false"}
		]

		if not all_rows:
			print("No comments found to classify.")
			return

		if not comments:
			print(f"All first {len(all_rows)} comments are already classified.")
			return

		target_count = len(comments)
		updated_count = 0
		for index, (rowid, comment_text) in enumerate(comments, start=1):
			text_value = comment_text or ""
			prompt = prompt_template.replace("{{COMMENT_TEXT}}", text_value)

			labels: dict | None = None
			for attempt in range(args.max_retries + 1):
				try:
					labels = ask_ollama_for_labels(
						args.host,
						args.model,
						prompt,
						args.request_timeout,
					)
					break
				except Exception as error:  # noqa: BLE001
					if attempt >= args.max_retries:
						print(
							f"Skipping rowid={rowid} after {attempt + 1} failed attempt(s): {error}",
							flush=True,
						)
						labels = None
						break

			if labels is None:
				continue

			cursor.execute(
				f"UPDATE {table_name} "
				f"SET negative = ?, angry = ?, spam = ?, response = ? "
				f"WHERE rowid = ?",
				(
					int(labels["negative"]),
					int(labels["angry"]),
					int(labels["spam"]),
					int(labels["response"]),
					rowid,
				),
			)
			updated_count += 1

			if index % args.progress_every == 0 or index == target_count:
				print(f"Classified {index}/{target_count} comments...", flush=True)

		connection.commit()

	skipped_count = target_count - updated_count
	print(
		f"Classification complete. Updated {updated_count}/{target_count} comments "
		f"in '{args.database}' table '{args.table}' (skipped: {skipped_count})."
	)


if __name__ == "__main__":
	main()
