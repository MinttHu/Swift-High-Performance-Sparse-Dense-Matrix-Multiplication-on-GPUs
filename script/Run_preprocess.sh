#!/bin/bash
#./Run_preprocess.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt

file_path="$1"
MTX_LIST="$2"
rm ../DataProcess/preprocess/*.txt

cp eval_spmm.sh ../test/preprocess
cd ../test/preprocess
./eval_spmm.sh $file_path $MTX_LIST

rm eval_spmm.sh
cp ./data/*.txt ../../DataProcess/preprocess