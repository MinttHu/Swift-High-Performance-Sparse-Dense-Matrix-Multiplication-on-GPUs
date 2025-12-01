DATA_PATH=$1
MTX_LIST=$2

cp eval_ASpT_spmm_*  ../build/ASpT_SpMM_GPU/
cp $MTX_LIST ../build/ASpT_SpMM_GPU/

cd ../build/ASpT_SpMM_GPU

echo "evaluating ASpT_spmm_f32_n32..."
./eval_ASpT_spmm_f32_n32.sh $DATA_PATH $MTX_LIST > result_ASpT_spmm_f32_n32.csv

echo "evaluating ASpT_spmm_f32_n128..."
./eval_ASpT_spmm_f32_n128.sh $DATA_PATH $MTX_LIST > result_ASpT_spmm_f32_n128.csv


cp result_*  ../../../DataProcess/FP32/
mv result_*  ../../../DataProcess/MTX_info/
rm eval_*
rm $MTX_LIST
