#!/usr/bin/env bash
# Step 1: Environment Setup
# Key versions: Python 3.10+, PyTorch >= 2.0.
# No Flower/federated framework needed -- the paper simulates FL in a single process.

python -m venv mmo-fl
source mmo-fl/bin/activate
pip install -r requirements.txt
