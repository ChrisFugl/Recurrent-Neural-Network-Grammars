#!/bin/bash

cd tools/percyliang_browncluster
make
cd ../..
pip install -r requirements.txt
