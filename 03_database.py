from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create a SQLite database from comments JSON.",
	)
	parser.add_argument(
		"-i",
		"--input",
		default="comments.json",
		help="Path to input comments JSON file (default: comments.json).",
	)
	parser.add_argument(
		"-t",
		"--table",
		default="comments",
		help="Table name to create/populate (default: comments).",
	)
	return parser.parse_args()


def quote_identifier(identifier: str) -> str:
	return f'"{identifier.replace("\"", "\"\"")}"'


def normalize_value(value: Any) -> Any:
	if isinstance(value, (dict, list)):
		return json.dumps(value, ensure_ascii=False)
	return value


def main() -> None:
	args = parse_args()
	json_path = Path(args.input)
	db_path = Path("comments.db")

	if not json_path.exists():
		raise FileNotFoundError(f"Input JSON file not found: {json_path}")

	data = json.loads(json_path.read_text(encoding="utf-8"))
	if not isinstance(data, list):
		raise ValueError("Input JSON must be an array of comment objects.")
	if not all(isinstance(item, dict) for item in data):
		raise ValueError("Each comment entry must be a JSON object.")

	json_columns = sorted({key for item in data for key in item.keys()})
	extra_columns = ["negative", "angry", "spam", "response"]
	all_columns = json_columns + extra_columns

	table_name = quote_identifier(args.table)
	column_definitions = [f"{quote_identifier(column)} TEXT" for column in json_columns]
	column_definitions.extend(
		[
			f'{quote_identifier("negative")} INTEGER DEFAULT 0',
			f'{quote_identifier("angry")} INTEGER DEFAULT 0',
			f'{quote_identifier("spam")} INTEGER DEFAULT 0',
			f'{quote_identifier("response")} TEXT',
		]
	)

	create_table_sql = (
		f"CREATE TABLE {table_name} ("
		+ ", ".join(column_definitions)
		+ ")"
	)

	insert_columns_sql = ", ".join(quote_identifier(column) for column in all_columns)
	placeholders_sql = ", ".join("?" for _ in all_columns)
	insert_sql = f"INSERT INTO {table_name} ({insert_columns_sql}) VALUES ({placeholders_sql})"

	rows = []
	for item in data:
		row = [normalize_value(item.get(column)) for column in json_columns]
		row.extend([0, 0, 0, None])
		rows.append(row)

	with sqlite3.connect(db_path) as connection:
		cursor = connection.cursor()
		cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
		cursor.execute(create_table_sql)
		cursor.executemany(insert_sql, rows)
		connection.commit()

	print(f"Created {db_path} with table '{args.table}' and inserted {len(rows)} rows.")


if __name__ == "__main__":
	main()
