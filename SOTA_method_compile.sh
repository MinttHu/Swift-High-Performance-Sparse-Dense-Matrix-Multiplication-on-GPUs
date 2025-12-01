#!/bin/bash

cd RoDe
mkdir build
cd build
cmake ..
make
cp -r SOTA_script RoDe/