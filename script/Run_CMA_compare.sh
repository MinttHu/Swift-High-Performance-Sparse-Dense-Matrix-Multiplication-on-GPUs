#!/bin/bash

#./Run_CMA_compare.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt
file_path="$1"
MTX_LIST="$2"
rm ../DataProcess/CMA-compare/*.txt
cp eval_spmm.sh ../test/k-32/CMA-compare
cp eval_spmm.sh ../test/k-128/CMA-compare-128

cd ../test/k-32/CMA-compare
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh
cp ./data/*.txt ../../../DataProcess/CMA-compare

cd ../../k-128/CMA-compare-128
./eval_spmm.sh $file_path $MTX_LIST

rm eval_spmm.sh
cp ./data/*.txt ../../../DataProcess/CMA-compare