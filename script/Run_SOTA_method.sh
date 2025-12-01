#!/bin/bash
#./Run_SOTA_method.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

DATA_PATH1=$1
MTX_LIST=$2
cd ../RoDe/SOTA_script

./ASpT_eval_64.sh $1 $2
./ASpT_eval.sh $1 $2
./_eval_64.sh $1 $2
./_eval.sh $1 $2
