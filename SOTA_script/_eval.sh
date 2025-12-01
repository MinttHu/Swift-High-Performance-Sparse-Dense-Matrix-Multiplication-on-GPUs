DATA_PATH=$1
MTX_LIST=$2

cp eval_spmm_*  ../build/eval
cp eval_sddmm_* ../build/eval
cp $MTX_LIST ../build/eval

cd ../build/eval

echo "evaluating spmm_f32_n32..."
./eval_spmm_call.sh eval_spmm_f32_n32 $DATA_PATH $MTX_LIST > result__spmm_f32_n32.csv

echo "evaluating spmm_f32_n128..."
./eval_spmm_call.sh eval_spmm_f32_n128 $DATA_PATH $MTX_LIST > result__spmm_f32_n128.csv


mv result_*  ../../../DataProcess/FP32/
rm *.sh
rm $MTX_LIST