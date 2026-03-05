# Categories present from initial pipeline: angry, negative, spam, needs_response
# Higher-level themes and categories: ethics/safety/governance, questions/clarification, praise/support, criticism/complaint
# Ethical subcategories: alignment, censorship/limits, morality/ethics, bias/fairness, privacy/surveillance, safety/harm

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


THEMATIC_KEYWORDS: dict[str, tuple[str, ...]] = {
	"ethics_safety_governance": (
		"ethic",
		"moral",
		"bias",
		"fair",
		"privacy",
		"alignment",
		"align",
		"safe",
		"safety",
		"censor",
	),
	"questions_clarification": (
		"?",
		"how ",
		"what ",
		"why ",
		"can you",
		"could you",
	),
	"praise_support": (
		"thank",
		"love",
		"great",
		"amazing",
		"awesome",
		"brilliant",
	),
	"criticism_complaint": (
		"bad",
		"wrong",
		"terrible",
		"stupid",
		"hate",
		"problem",
		"disappoint",
	),
}


ETHICS_SUBCATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
	"alignment": ("align", "alignment"),
	"censorship_limits": ("censor", "limit", "restriction"),
	"morality_ethics": ("moral", "ethic"),
	"bias_fairness": ("bias", "fair"),
	"privacy_surveillance": ("privacy", "surveillance"),
	"safety_harm": ("safe", "safety", "harm"),
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate comment category summaries from the SQLite database.",
	)
	parser.add_argument("--database", default="comments.db", help="SQLite database path.")
	parser.add_argument("--table", default="comments", help="Table name (default: comments).")
	parser.add_argument(
		"--json-out",
		default="category_summary.json",
		help="Output JSON report path (default: category_summary.json).",
	)
	parser.add_argument(
		"--csv-out",
		default="category_summary.csv",
		help="Output CSV report path (default: category_summary.csv).",
	)
	return parser.parse_args()


def quote_identifier(identifier: str) -> str:
	escaped = identifier.replace('"', '""')
	return f'"{escaped}"'


def is_truthy(value: object) -> bool:
	text = str(value or "").strip().lower()
	return text in {"1", "true"}


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
	return any(keyword in text for keyword in keywords)


def main() -> None:
	args = parse_args()
	db_path = Path(args.database)
	if not db_path.exists():
		raise FileNotFoundError(f"Database not found: {db_path}")

	table_name = quote_identifier(args.table)

	with sqlite3.connect(db_path) as connection:
		cursor = connection.cursor()
		cursor.execute(
			f"SELECT text, angry, negative, spam, response FROM {table_name} ORDER BY rowid ASC"
		)
		rows = cursor.fetchall()

	if not rows:
		raise ValueError("No comments found in the selected table.")

	total = len(rows)

	label_counts = {
		"angry": 0,
		"negative": 0,
		"spam": 0,
		"needs_response": 0,
	}
	thematic_counts = {key: 0 for key in THEMATIC_KEYWORDS}
	ethics_subcategory_counts = {key: 0 for key in ETHICS_SUBCATEGORY_KEYWORDS}

	for text, angry, negative, spam, response in rows:
		text_normalized = str(text or "").lower()

		if is_truthy(angry):
			label_counts["angry"] += 1
		if is_truthy(negative):
			label_counts["negative"] += 1
		if is_truthy(spam):
			label_counts["spam"] += 1
		if is_truthy(response):
			label_counts["needs_response"] += 1

		for name, keywords in THEMATIC_KEYWORDS.items():
			if contains_any(text_normalized, keywords):
				thematic_counts[name] += 1

		for name, keywords in ETHICS_SUBCATEGORY_KEYWORDS.items():
			if contains_any(text_normalized, keywords):
				ethics_subcategory_counts[name] += 1

	report = {
		"total_comments": total,
		"label_categories": label_counts,
		"thematic_categories_non_exclusive": thematic_counts,
		"ethics_subcategories_non_exclusive": ethics_subcategory_counts,
	}

	json_path = Path(args.json_out)
	json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

	csv_path = Path(args.csv_out)
	with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["group", "category", "count", "percentage_of_total"])

		for category, count in label_counts.items():
			writer.writerow(["label", category, count, f"{(count / total) * 100:.2f}"])

		for category, count in thematic_counts.items():
			writer.writerow(["thematic", category, count, f"{(count / total) * 100:.2f}"])

		for category, count in ethics_subcategory_counts.items():
			writer.writerow(["ethics_subcategory", category, count, f"{(count / total) * 100:.2f}"])

	print(f"Analyzed {total} comments from '{args.database}' table '{args.table}'.")
	print(f"Wrote JSON report: {json_path}")
	print(f"Wrote CSV report: {csv_path}")


if __name__ == "__main__":
	main()
