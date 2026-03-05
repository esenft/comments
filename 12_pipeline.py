from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all existing project scripts in order from one command.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument(
        "--video-url",
        default="https://www.youtube.com/watch?v=L_Guz73e6fw",
        help="YouTube URL passed to 01_extract.py.",
    )
    parser.add_argument(
        "--extract-limit",
        type=int,
        default=11193,
        help="Comment download limit passed to 01_extract.py (default: 11193).",
    )
    parser.add_argument(
        "--classify-limit",
        type=int,
        default=100,
        help="Comment analysis limit passed to 06_prediction.py (default: 100).",
    )
    parser.add_argument("--database", default="comments.db", help="Database path.")
    parser.add_argument("--table", default="comments", help="Database table name.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name.")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip running setup_requirements.py.",
    )
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise PipelineError(f"Step failed: {name} (exit code {result.returncode})")


def require_file(path: Path, reason: str) -> None:
    if not path.exists():
        raise PipelineError(f"Required file missing for {reason}: {path}")


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    python_exe = args.python
    comments_json = root / "comments.json"
    sample_json = root / "sample_data.json"
    prompt_file = root / "prompt.txt"
    database_path = root / args.database

    if args.extract_limit <= 0:
        raise PipelineError("--extract-limit must be positive.")
    if args.classify_limit <= 0:
        raise PipelineError("--classify-limit must be positive.")

    try:
        if not args.skip_install:
            run_step("Install requirements", [python_exe, str(root / "setup_requirements.py")])

        run_step(
            "Extract data",
            [
                python_exe,
                str(root / "01_extract.py"),
                args.video_url,
                "--output",
                str(comments_json),
                "--limit",
                str(args.extract_limit),
            ],
        )
        require_file(comments_json, "extract data")

        run_step(
            "Count comments",
            [python_exe, str(root / "02_count.py"), "--input", str(comments_json)],
        )

        run_step(
            "Create database",
            [
                python_exe,
                str(root / "03_database.py"),
                "--input",
                str(comments_json),
                "--table",
                args.table,
            ],
        )
        require_file(database_path, "create database")

        # These are existing project artifacts used by downstream scripts.
        require_file(sample_json, "sample data artifact")
        require_file(prompt_file, "comment classification prompt")

        run_step(
            "OpenAI API/Ollama ping",
            [
                python_exe,
                str(root / "05_ping_openai.py"),
                "--host",
                args.host,
                "--model",
                args.model,
            ],
        )

        run_step(
            "Comment analysis",
            [
                python_exe,
                str(root / "06_prediction.py"),
                "--database",
                str(database_path),
                "--table",
                args.table,
                "--prompt-file",
                str(prompt_file),
                "--host",
                args.host,
                "--model",
                args.model,
                "--limit",
                str(args.classify_limit),
            ],
        )

        run_step(
            "Create responses",
            [
                python_exe,
                str(root / "07_create_responses.py"),
                "--database",
                str(database_path),
                "--table",
                args.table,
                "--host",
                args.host,
                "--model",
                args.model,
            ],
        )

        run_step(
            "Extract categories",
            [
                python_exe,
                str(root / "08_categories.py"),
                "--database",
                str(database_path),
                "--table",
                args.table,
            ],
        )

        run_step(
            "Create visualization",
            [
                python_exe,
                str(root / "09_visualization.py"),
                "--database",
                str(database_path),
                "--table",
                args.table,
            ],
        )

        run_step(
            "Final dataset export",
            [
                python_exe,
                str(root / "11_export.py"),
                "--database",
                str(database_path),
                "--table",
                args.table,
                "--output",
                str(root / "clean_dataset.json"),
                "--schema-out",
                str(root / "clean_dataset_schema.json"),
            ],
        )

        print("\nPipeline completed successfully.", flush=True)
    except PipelineError as error:
        print(f"\nPipeline failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
