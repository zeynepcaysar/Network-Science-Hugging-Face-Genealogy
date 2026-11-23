#!/bin/bash
# Creates .venv in this folder and installs the packages the scripts need.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                  # makes "from load import load" work in every folder
echo
echo "Done. Activate it with:  source .venv/bin/activate"
