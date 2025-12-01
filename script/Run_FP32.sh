#!/bin/bash

#./Run_FP32.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt

file_path="$1"
MTX_LIST="$2"


rm ../DataProcess/FP32/results_Swift_FP32_32.txt
rm ../DataProcess/FP32/results_Swift_FP32_128.txt

cp eval_spmm.sh ../test/k-32/FP32
cp eval_spmm.sh ../test/k-128/FP32-128

cd ../test/k-32/FP32
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../../DataProcess/FP32

cd ../../k-128/FP32-128
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh
cp ./data/*.txt ../../../DataProcess/FP32
cp ./data/results_Swift_FP32_128.txt ../../../DataProcess/MTX_info