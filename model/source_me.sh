#!/bin/bash

virtualenv venv

source venv/bin/activate

pip3 install -r requirements-pytorch.txt

./setup-scripts/check_cuda.py
