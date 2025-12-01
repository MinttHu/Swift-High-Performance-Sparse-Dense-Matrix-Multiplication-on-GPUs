#!/bin/bash
#./Run_irr_regular_ratio.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt

file_path="$1"
MTX_LIST="$2"
rm ../DataProcess/irr_regular_ratio/*.txt

cp eval_spmm.sh ../test/irr_regular_ratio
cd ../test/irr_regular_ratio
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../DataProcess/irr_regular_ratio

