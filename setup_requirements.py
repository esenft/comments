from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
	requirements_path = Path("requirements.txt")
	if not requirements_path.exists():
		raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

	command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
	print(f"Installing dependencies from {requirements_path}...")
	subprocess.run(command, check=True)
	print("Dependency installation complete.")


if __name__ == "__main__":
	main()
