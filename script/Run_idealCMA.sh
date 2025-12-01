#!/bin/bash
#./Run_idealCMA.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt
file_path="$1"
MTX_LIST="$2"

rm ../DataProcess/idealCMA/*.txt
cp eval_spmm.sh ../test/k-32/idealCMA
cp eval_spmm.sh ../test/k-128/idealCMA-128

cd ../test/k-32/idealCMA
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh
cp ./data/*.txt ../../../DataProcess/idealCMA

cd ../../k-128/idealCMA-128
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../../DataProcess/idealCMA