#!/bin/bash
# Creates .venv in this folder and installs the packages the scripts need.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Done. Activate it with:  source .venv/bin/activate"
