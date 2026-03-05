from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_PROMPT_TEMPLATE = (
	"You are helping moderate a YouTube channel. "
	"Write one concise, polite creator reply to the following comment. "
	"Keep it under 60 words, avoid promises you cannot verify, and do not mention moderation labels. "
	"Return only the reply text.\n\n"
	"Comment:\n{{COMMENT_TEXT}}"
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate creator responses for comments marked as requiring a response.",
	)
	parser.add_argument("--database", default="comments.db", help="SQLite database path.")
	parser.add_argument("--table", default="comments", help="Table name (default: comments).")
	parser.add_argument(
		"--prompt-file",
		default="",
		help="Optional prompt template path with {{COMMENT_TEXT}} placeholder.",
	)
	parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
	parser.add_argument("--model", default="llama3.2", help="Ollama model name.")
	parser.add_argument(
		"--limit",
		type=int,
		default=0,
		help="Maximum number of rows to process (0 means all).",
	)
	parser.add_argument(
		"--request-timeout",
		type=int,
		default=90,
		help="Timeout per Ollama request in seconds.",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=2,
		help="Retries per comment if request fails.",
	)
	parser.add_argument(
		"--progress-every",
		type=int,
		default=5,
		help="Print progress every N comments.",
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


def ask_ollama_for_response(host: str, model: str, prompt: str, request_timeout: int) -> str:
	try:
		body = post_json(
			host,
			"/api/chat",
			{
				"model": model,
				"messages": [{"role": "user", "content": prompt}],
				"stream": False,
				"options": {"temperature": 0.3, "num_predict": 160},
			},
			request_timeout,
		)
		text = body.get("message", {}).get("content", "").strip()
		if text:
			return text
	except urllib.error.HTTPError as error:
		if error.code != 404:
			error_body = error.read().decode("utf-8", errors="replace")
			raise SystemExit(f"Ollama request failed ({error.code}): {error_body}") from error
	except urllib.error.URLError as error:
		raise SystemExit("Could not reach Ollama. Start it with: ollama serve") from error

	try:
		body = post_json(
			host,
			"/v1/chat/completions",
			{
				"model": model,
				"messages": [{"role": "user", "content": prompt}],
				"temperature": 0.3,
				"max_tokens": 160,
			},
			request_timeout,
		)
		choices = body.get("choices", [])
		if choices:
			text = choices[0].get("message", {}).get("content", "").strip()
			if text:
				return text
	except urllib.error.HTTPError as error:
		error_body = error.read().decode("utf-8", errors="replace")
		raise SystemExit(
			f"Ollama request failed ({error.code}): {error_body}. "
			f"If needed, run: ollama pull {model}"
		) from error

	raise ValueError("Model returned an empty response.")


def load_prompt_template(prompt_file: str) -> str:
	if not prompt_file:
		return DEFAULT_PROMPT_TEMPLATE

	template = Path(prompt_file).read_text(encoding="utf-8")
	if "{{COMMENT_TEXT}}" not in template:
		raise ValueError("Prompt file must include the placeholder '{{COMMENT_TEXT}}'.")
	return template


def column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
	cursor.execute(f"PRAGMA table_info({quote_identifier(table)})")
	columns = cursor.fetchall()
	return any(str(row[1]) == column for row in columns)


def ensure_responses_column(cursor: sqlite3.Cursor, table: str) -> None:
	if not column_exists(cursor, table, "responses"):
		cursor.execute(f"ALTER TABLE {quote_identifier(table)} ADD COLUMN responses TEXT")


def main() -> None:
	args = parse_args()

	if args.limit < 0:
		raise ValueError("--limit must be 0 or a positive integer.")
	if args.request_timeout <= 0:
		raise ValueError("--request-timeout must be a positive integer.")
	if args.max_retries < 0:
		raise ValueError("--max-retries must be 0 or a positive integer.")
	if args.progress_every <= 0:
		raise ValueError("--progress-every must be a positive integer.")

	prompt_template = load_prompt_template(args.prompt_file)
	db_path = Path(args.database)
	if not db_path.exists():
		raise FileNotFoundError(f"Database not found: {db_path}")

	with sqlite3.connect(db_path) as connection:
		cursor = connection.cursor()
		ensure_responses_column(cursor, args.table)

		limit_clause = "" if args.limit == 0 else " LIMIT ?"
		params: tuple[int, ...] = () if args.limit == 0 else (args.limit,)

		cursor.execute(
			f"SELECT rowid, text FROM {quote_identifier(args.table)} "
			"WHERE lower(trim(CAST(response AS TEXT))) IN ('1', 'true') "
			"AND (responses IS NULL OR trim(responses) = '') "
			"ORDER BY rowid ASC"
			+ limit_clause,
			params,
		)
		rows = cursor.fetchall()

		if not rows:
			print("No comments need generated responses.")
			connection.commit()
			return

		total = len(rows)
		for index, (rowid, comment_text) in enumerate(rows, start=1):
			prompt = prompt_template.replace("{{COMMENT_TEXT}}", comment_text or "")

			result: str | None = None
			for attempt in range(args.max_retries + 1):
				try:
					result = ask_ollama_for_response(
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
						result = None

			if result is None:
				continue

			cursor.execute(
				f"UPDATE {quote_identifier(args.table)} SET responses = ? WHERE rowid = ?",
				(result, rowid),
			)

			if index % args.progress_every == 0 or index == total:
				print(f"Generated {index}/{total} responses...", flush=True)

		connection.commit()

	print(f"Stored generated responses for up to {len(rows)} comments.")


if __name__ == "__main__":
	main()
