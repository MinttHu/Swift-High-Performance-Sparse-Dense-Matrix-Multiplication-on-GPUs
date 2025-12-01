#!/bin/bash

cd ../src
./test -d 0 ../MTX_samples/bundle1.mtx
./test -d 0 ../MTX_samples/c-48.mtx
./test -d 0 ../MTX_samples/HEP-th.mtx
./test -d 0 ../MTX_samples/rajat22.mtx
./test -d 0 ../MTX_samples/RFdevice.mtx

echo "--Width = 32--"

echo "--Coalesced memory access test--"

cd ../test/k-32/CMA-compare

./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--Float precision test--"

cd ../FP32
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--double precision test--"

cd ../FP64
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--Ideal Coalesced memory access test--"

cd ../idealCMA
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--Irregular part optimization test--"

cd ../irr_opt
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--Width = 128--"

echo "--Coalesced memory access test--"

cd ../../k-128/CMA-compare-128

./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--Float precision test--"

cd ../FP32-128
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--double precision test--"
cd ../FP64-128
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--Ideal Coalesced memory access test--"
cd ../idealCMA-128
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx


echo "--Irregular part optimization test--"
cd ../irr_opt-128
./test -d 0 ../../../MTX_samples/bundle1.mtx
./test -d 0 ../../../MTX_samples/c-48.mtx
./test -d 0 ../../../MTX_samples/HEP-th.mtx
./test -d 0 ../../../MTX_samples/rajat22.mtx
./test -d 0 ../../../MTX_samples/RFdevice.mtx

echo "--Irregular part ratio test--"
cd ../../irr_regular_ratio
./test -d 0 ../../MTX_samples/bundle1.mtx
./test -d 0 ../../MTX_samples/c-48.mtx
./test -d 0 ../../MTX_samples/HEP-th.mtx
./test -d 0 ../../MTX_samples/rajat22.mtx
./test -d 0 ../../MTX_samples/RFdevice.mtx


echo "--Preprocess time test--"
cd ../preprocess
./test -d 0 ../../MTX_samples/bundle1.mtx
./test -d 0 ../../MTX_samples/c-48.mtx
./test -d 0 ../../MTX_samples/HEP-th.mtx
./test -d 0 ../../MTX_samples/rajat22.mtx
./test -d 0 ../../MTX_samples/RFdevice.mtx

echo "--MTX information test--"
cd ../MTX_info
./test -d 0 ../../MTX_samples/bundle1.mtx
./test -d 0 ../../MTX_samples/c-48.mtx
./test -d 0 ../../MTX_samples/HEP-th.mtx
./test -d 0 ../../MTX_samples/rajat22.mtx
./test -d 0 ../../MTX_samples/RFdevice.mtx

echo "--Load balance test--"
cd ../LoadBalance
./test -d 0 ../../MTX_samples/JP.mtx

cd ../ncu
./test -d 0 ../../MTX_samples/bundle1.mtx
./test -d 0 ../../MTX_samples/c-48.mtx
./test -d 0 ../../MTX_samples/HEP-th.mtx
./test -d 0 ../../MTX_samples/rajat22.mtx
./test -d 0 ../../MTX_samples/RFdevice.mtx


