#!/bin/bash

#./Run_MTX_info.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt

file_path="$1"
MTX_LIST="$2"

rm ../DataProcess/MTX_info/*.txt


cp eval_spmm.sh ../test/MTX_info
cd ../test/MTX_info

./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../DataProcess/MTX_info

