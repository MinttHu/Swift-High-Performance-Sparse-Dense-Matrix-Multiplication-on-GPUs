DATA_PATH=$1
MTX_LIST=$2

cp eval_ASpT_spmm_*  ../build/ASpT_SpMM_GPU/
cp eval_ASpT_preprocess.sh*  ../build/ASpT_SpMM_GPU/
cp $MTX_LIST ../build/ASpT_SpMM_GPU/

cd ../build/ASpT_SpMM_GPU

echo "evaluating ASpT_preprocess..."
./eval_ASpT_preprocess.sh $DATA_PATH $MTX_LIST > pre_result_ASpT.csv  #include preprocessing time


mv pre_result_* ../../../DataProcess/preprocess/
rm eval_*
rm $MTX_LIST