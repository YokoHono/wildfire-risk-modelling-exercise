#!/usr/bin/env bash
# Set up a local Python virtual environment and install all dependencies
# required to run the Wildfire Risk Modelling Exercise notebooks.
#
# Usage:   bash install_packages.sh
# Then:    source .venv/bin/activate
#          jupyter trust Forest_submission.ipynb Prairie_submission.ipynb
#          jupyter notebook   # (or open the notebooks in VS Code / JupyterLab)

set -euo pipefail

if [[ ! -d ".venv" ]]; then
    echo "[install] creating virtual environment in .venv/"
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[install] upgrading pip"
python -m pip install --upgrade pip > install.log 2>&1

echo "[install] installing requirements (see install.log for full output)"
python -m pip install -r requirements.txt >> install.log 2>&1

echo "[install] done. Activate with:  source .venv/bin/activate"
