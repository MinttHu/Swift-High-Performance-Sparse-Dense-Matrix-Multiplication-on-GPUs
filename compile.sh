#!/bin/bash

rm src/Makefile
rm test/irr_regular_ratio/Makefile
rm test/preprocess/Makefile
rm test/LoadBalance/Makefile
rm test/MTX_info/Makefile 
rm test/ncu/Makefile

rm test/k-32/CMA-compare/Makefile 
rm test/k-32/FP32/Makefile
rm test/k-32/FP64/Makefile
rm test/k-32/idealCMA/Makefile
rm test/k-32/irr_opt/Makefile   

rm test/k-128/CMA-compare-128/Makefile    
rm test/k-128/FP32-128/Makefile
rm test/k-128/FP64-128/Makefile  
rm test/k-128/idealCMA-128/Makefile
rm test/k-128/irr_opt-128/Makefile    

rm test/irrgular_K/FP32/Makefile
rm test/irrgular_K/FP64/Makefile

cp Makefile src/

cp Makefile test/irr_regular_ratio/
cp Makefile test/preprocess/
cp Makefile test/LoadBalance/
cp Makefile test/MTX_info/Makefile
cp Makefile test/ncu/

cp Makefile test/k-32/CMA-compare/
cp Makefile test/k-32/FP32/
cp Makefile test/k-32/FP64/
cp Makefile test/k-32/idealCMA/
cp Makefile test/k-32/irr_opt/

cp Makefile test/k-128/CMA-compare-128/
cp Makefile test/k-128/FP32-128/
cp Makefile test/k-128/FP64-128/
cp Makefile test/k-128/idealCMA-128/
cp Makefile test/k-128/irr_opt-128/

cp Makefile test/irregular_K/FP32/
cp Makefile test/irregular_K/FP64/

cd src
make


cd ../test/irr_regular_ratio
make

cd ../preprocess
make

cd ../LoadBalance
make

cd ../MTX_info
make

cd ../ncu
make

cd ../irregular_K/FP32
make
cd ../FP64
make

cd ../../k-32/CMA-compare
make

cd ../FP32
make

cd ../FP64
make

cd ../idealCMA
make

cd ../irr_opt
make


#k=128
cd ../../k-128/CMA-compare-128
make

cd ../FP32-128
make

cd ../FP64-128
make

cd ../idealCMA-128
make

cd ../irr_opt-128
make







