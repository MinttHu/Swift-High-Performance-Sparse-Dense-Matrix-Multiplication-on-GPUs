DATA_PATH=$1
MTX_LIST=$2

cp eval_spmm_*  ../build/eval
cp eval_sddmm_* ../build/eval
cp $MTX_LIST ../build/eval

cd ../build/eval


echo "evaluating spmm_preprocess..."                                             #include preprocessing time
./eval_spmm_preprocess.sh eval_spmm_f64_n32  $DATA_PATH $MTX_LIST > pre_result__spmm.csv  #include preprocessing time

mv pre_result_* ../../../DataProcess/preprocess/
rm *.sh
rm $MTX_LIST