#!/bin/bash
#./Run_irr_opt.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt

file_path="$1"
MTX_LIST="$2"
rm ../DataProcess/irr_opt/*.txt
cp eval_spmm.sh ../test/k-32/irr_opt
cp eval_spmm.sh ../test/k-128/irr_opt-128

cd ../test/k-32/irr_opt
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../../DataProcess/irr_opt

cd ../../k-128/irr_opt-128
./eval_spmm.sh $file_path $MTX_LIST
rm eval_spmm.sh

cp ./data/*.txt ../../../DataProcess/irr_opt