#!/bin/bash
# Creates .venv in this folder with a uv-managed Python 3.13 and installs the pinned packages
# (requirements.txt names them, constraints.txt pins every version). Works in Git Bash on Windows,
# macOS and Linux. Safe to run again: the venv is rebuilt from scratch each time.
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null 2>&1 || { echo "uv not found - install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
uv venv --python 3.13 --managed-python --clear .venv
uv pip install -r requirements.txt -c constraints.txt
uv pip install -e . -c constraints.txt   # makes "from load import load" work in every folder
echo
echo "Done. Activate the environment with:"
echo "  macOS / Linux:  source .venv/bin/activate"
echo "  Git Bash:       source .venv/Scripts/activate"
echo "  PowerShell:     .venv\\Scripts\\Activate.ps1"
echo "Then run the tests with:  python -m pytest -m \"not data and not slow\""
echo "and check the data files with:  python load.py --verify"
