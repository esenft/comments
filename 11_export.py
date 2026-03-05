# File structure: each entry is labeled with the row id, comment author, channel, comment id, heart status, photo URL, number of replies, reply status, text, time, parsed time, votes, negative label, angry label, spam label, response label, and generated responses (if applicable)

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


FIELD_DESCRIPTIONS: dict[str, str] = {
	"rowid": "SQLite row identifier used as stable export order and record id.",
	"author": "YouTube comment author handle.",
	"channel": "YouTube channel id of the commenter.",
	"cid": "Unique YouTube comment id.",
	"heart": "Whether the channel hearted the comment (boolean).",
	"photo": "Commenter profile photo URL.",
	"replies": "Number of replies on the comment (integer).",
	"reply": "Whether this row is a reply (boolean).",
	"text": "Comment body text.",
	"time": "Original relative time string from source (for example '2 years ago').",
	"time_parsed": "Unix timestamp (seconds) parsed from source metadata.",
	"votes": "Like/upvote count normalized to integer.",
	"negative": "Model label: negative sentiment (boolean).",
	"angry": "Model label: angry/hostile tone (boolean).",
	"spam": "Model label: spam indicator (boolean).",
	"response": "Model label: comment requires response (boolean).",
	"responses": "Generated creator response text, if available.",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export and clean comments database into JSON dataset.",
	)
	parser.add_argument("--database", default="comments.db", help="SQLite database path.")
	parser.add_argument("--table", default="comments", help="Table name (default: comments).")
	parser.add_argument(
		"--output",
		default="clean_dataset.json",
		help="Output JSON dataset path (default: clean_dataset.json).",
	)
	parser.add_argument(
		"--schema-out",
		default="clean_dataset_schema.json",
		help="Output schema/documentation JSON path.",
	)
	return parser.parse_args()


def quote_identifier(identifier: str) -> str:
	escaped = identifier.replace('"', '""')
	return f'"{escaped}"'


def to_bool(value: Any) -> bool:
	text = str(value or "").strip().lower()
	return text in {"1", "true", "yes"}


def to_int(value: Any, default: int = 0) -> int:
	if value is None:
		return default
	text = str(value).strip()
	if not text:
		return default
	try:
		return int(float(text))
	except ValueError:
		return default


def parse_votes(value: Any) -> int:
	if value is None:
		return 0
	text = str(value).strip().upper().replace(",", "")
	if not text:
		return 0

	match = re.match(r"^([0-9]+(?:\.[0-9]+)?)([KMB])?$", text)
	if not match:
		return 0

	number = float(match.group(1))
	suffix = match.group(2)
	multiplier = {None: 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
	return int(number * multiplier)


def clean_text(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text if text else None


def clean_record(raw: dict[str, Any]) -> dict[str, Any]:
	return {
		"rowid": to_int(raw.get("rowid"), default=0),
		"author": clean_text(raw.get("author")),
		"channel": clean_text(raw.get("channel")),
		"cid": clean_text(raw.get("cid")),
		"heart": to_bool(raw.get("heart")),
		"photo": clean_text(raw.get("photo")),
		"replies": to_int(raw.get("replies"), default=0),
		"reply": to_bool(raw.get("reply")),
		"text": clean_text(raw.get("text")),
		"time": clean_text(raw.get("time")),
		"time_parsed": float(raw.get("time_parsed")) if raw.get("time_parsed") not in (None, "") else None,
		"votes": parse_votes(raw.get("votes")),
		"negative": to_bool(raw.get("negative")),
		"angry": to_bool(raw.get("angry")),
		"spam": to_bool(raw.get("spam")),
		"response": to_bool(raw.get("response")),
		"responses": clean_text(raw.get("responses")),
	}


def export_schema(schema_path: Path) -> None:
	columns = [
		{"name": "rowid", "type": "integer", "description": FIELD_DESCRIPTIONS["rowid"]},
		{"name": "author", "type": "string|null", "description": FIELD_DESCRIPTIONS["author"]},
		{"name": "channel", "type": "string|null", "description": FIELD_DESCRIPTIONS["channel"]},
		{"name": "cid", "type": "string|null", "description": FIELD_DESCRIPTIONS["cid"]},
		{"name": "heart", "type": "boolean", "description": FIELD_DESCRIPTIONS["heart"]},
		{"name": "photo", "type": "string|null", "description": FIELD_DESCRIPTIONS["photo"]},
		{"name": "replies", "type": "integer", "description": FIELD_DESCRIPTIONS["replies"]},
		{"name": "reply", "type": "boolean", "description": FIELD_DESCRIPTIONS["reply"]},
		{"name": "text", "type": "string|null", "description": FIELD_DESCRIPTIONS["text"]},
		{"name": "time", "type": "string|null", "description": FIELD_DESCRIPTIONS["time"]},
		{"name": "time_parsed", "type": "number|null", "description": FIELD_DESCRIPTIONS["time_parsed"]},
		{"name": "votes", "type": "integer", "description": FIELD_DESCRIPTIONS["votes"]},
		{"name": "negative", "type": "boolean", "description": FIELD_DESCRIPTIONS["negative"]},
		{"name": "angry", "type": "boolean", "description": FIELD_DESCRIPTIONS["angry"]},
		{"name": "spam", "type": "boolean", "description": FIELD_DESCRIPTIONS["spam"]},
		{"name": "response", "type": "boolean", "description": FIELD_DESCRIPTIONS["response"]},
		{"name": "responses", "type": "string|null", "description": FIELD_DESCRIPTIONS["responses"]},
	]

	schema = {
		"dataset": "clean_dataset.json",
		"record_format": "array of objects",
		"columns": columns,
		"assumptions_and_decisions": [
			"Rows are exported in ascending SQLite rowid order.",
			"Boolean-like values (0/1/true/false/yes) are normalized to booleans.",
			"Blank strings are normalized to null for text-like fields.",
			"Votes are normalized to integers; K/M/B suffixes are expanded (for example 1.2K -> 1200).",
			"Missing numeric-like fields are set to 0 for replies and votes.",
		],
	}
	schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main() -> None:
	args = parse_args()
	db_path = Path(args.database)
	if not db_path.exists():
		raise FileNotFoundError(f"Database not found: {db_path}")

	table_name = quote_identifier(args.table)
	query = (
		f"SELECT rowid, author, channel, cid, heart, photo, replies, reply, text, "
		f"time, time_parsed, votes, negative, angry, spam, response, responses "
		f"FROM {table_name} ORDER BY rowid ASC"
	)

	with sqlite3.connect(db_path) as connection:
		connection.row_factory = sqlite3.Row
		rows = connection.execute(query).fetchall()

	cleaned = [clean_record(dict(row)) for row in rows]

	output_path = Path(args.output)
	output_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

	schema_path = Path(args.schema_out)
	export_schema(schema_path)

	print(f"Exported {len(cleaned)} records to {output_path}")
	print(f"Wrote dataset documentation to {schema_path}")


if __name__ == "__main__":
	main()
