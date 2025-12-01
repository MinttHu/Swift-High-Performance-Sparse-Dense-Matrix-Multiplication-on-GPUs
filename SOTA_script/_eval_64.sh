DATA_PATH=$1
MTX_LIST=$2
cp eval_spmm_*  ../build/eval
cp eval_sddmm_* ../build/eval
cp $MTX_LIST ../build/eval

cd ../build/eval

echo "evaluating spmm_f64_n32..."
./eval_spmm_call.sh eval_spmm_f64_n32 $DATA_PATH $MTX_LIST > result__spmm_f64_n32.csv

echo "evaluating spmm_f64_n128..."
./eval_spmm_f64_n128.sh                $DATA_PATH $MTX_LIST > result__spmm_f64_n128.csv


cp result_*  ../../../DataProcess/FP64/
mv result_*  ../../../DataProcess/MTX_info/
rm *.sh
rm $MTX_LIST