#!/bin/bash
#./Run_SOTA_preprocess.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

DATA_PATH1=$1
MTX_LIST=$2
cd ../RoDe/SOTA_script

./ASpT_preprocess.sh $1 $2
./_eval_preprocess.sh $1 $2