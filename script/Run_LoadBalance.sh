#!/bin/bash
#./Run_LoadBalance.sh
rm ../DataProcess/LoadBalance/*.txt

cd ../test/LoadBalance
./test -d 0 ../../MTX_samples/JP.mtx 32
./test -d 0 ../../MTX_samples/JP.mtx 64
./test -d 0 ../../MTX_samples/JP.mtx 128
./test -d 0 ../../MTX_samples/JP.mtx 256
./test -d 0 ../../MTX_samples/JP.mtx 512
./test -d 0 ../../MTX_samples/JP.mtx 1024
./test -d 0 ../../MTX_samples/JP.mtx 2048

cp ./data/*.txt ../../DataProcess/LoadBalance/