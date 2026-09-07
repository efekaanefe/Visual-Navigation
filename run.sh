#!/bin/bash

set -e
mkdir -p build
cd build
cmake ..
make
./main
cd ..
python3 vis.py
