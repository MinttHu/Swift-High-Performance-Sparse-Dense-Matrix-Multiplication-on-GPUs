#!/bin/bash

#./Run_FP64.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt

file_path="$1"
MTX_LIST="$2"

rm ../DataProcess/FP64/results_Swift_FP64_32.txt
rm ../DataProcess/FP64/results_Swift_FP64_128.txt

cp eval_spmm.sh ../test/k-32/FP64
cp eval_spmm.sh ../test/k-128/FP64-128

cd ../test/k-32/FP64
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../../DataProcess/FP64

cd ../../k-128/FP64-128
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../../DataProcess/FP64