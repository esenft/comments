# TO RUN: Need to install ollama in the environment
# In terminal: run "ollama serve" to start the server, then run "ollama pull llama3.2" to pull the model
# Then, you can run python 05_ping_openai.py to generate the result

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


QUESTION = "How many states are there in the United States of America?"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Send a test question to an Ollama model and print JSON output.",
	)
	parser.add_argument(
		"--model",
		default="llama3.2",
		help="Ollama model name (default: llama3.2).",
	)
	parser.add_argument(
		"--host",
		default="http://localhost:11434",
		help="Ollama host URL (default: http://localhost:11434).",
	)
	return parser.parse_args()


def ask_ollama(host: str, model: str, question: str) -> str:
	prompt = (
		"Answer the question using JSON only. "
		"Return exactly one object in this format: {\"answer\": <value>}.\n"
		f"Question: {question}"
	)

	def post_json(path: str, payload: dict) -> dict:
		request = urllib.request.Request(
			url=f"{host.rstrip('/')}{path}",
			data=json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=120) as response:
			return json.loads(response.read().decode("utf-8"))

	try:
		# Preferred endpoint for recent Ollama versions.
		body = post_json(
			"/api/chat",
			{
				"model": model,
				"messages": [{"role": "user", "content": prompt}],
				"stream": False,
				"format": "json",
			},
		)
		model_text = body.get("message", {}).get("content", "").strip()
		if model_text:
			return model_text
	except urllib.error.HTTPError as error:
		if error.code != 404:
			error_body = error.read().decode("utf-8", errors="replace")
			raise SystemExit(f"Ollama request failed ({error.code}): {error_body}") from error
		# If endpoint is unavailable, try OpenAI-compatible endpoint below.
	except urllib.error.URLError as error:
		raise SystemExit(
			"Could not reach Ollama. Ensure it is installed and running, e.g. 'ollama serve'."
		) from error

	try:
		body = post_json(
			"/v1/chat/completions",
			{
				"model": model,
				"messages": [{"role": "user", "content": prompt}],
				"response_format": {"type": "json_object"},
			},
		)
		choices = body.get("choices", [])
		if choices:
			model_text = choices[0].get("message", {}).get("content", "").strip()
			if model_text:
				return model_text
	except urllib.error.HTTPError as error:
		error_body = error.read().decode("utf-8", errors="replace")
		raise SystemExit(
			f"Ollama request failed ({error.code}): {error_body}. "
			f"If needed, pull the model first: ollama pull {model}"
		) from error

	raise SystemExit(
		f"Ollama returned an empty response. If the model is missing, run: ollama pull {model}"
	)


def main() -> None:
	args = parse_args()
	model_json = ask_ollama(args.host, args.model, QUESTION)

	# Validate/normalize to the required key-value JSON object.
	parsed = json.loads(model_json)
	if not isinstance(parsed, dict):
		raise SystemExit("Model response is not a JSON object.")

	if "answer" not in parsed:
		parsed = {"answer": parsed}

	print(json.dumps(parsed, ensure_ascii=False))


if __name__ == "__main__":
	main()
