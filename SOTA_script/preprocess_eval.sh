#!/bin/bash
base_path=$1

cp mtxList.txt ../build
cd ../build

echo "Dataset,m,n,nnz,ASpT,Sputnik,RoDe" >> result_preprocess.csv
#base_path=$1
#for i in $base_path/*
#do 
#ii=$(basename "$i")
#fpath="${i}/${ii}.mtx"
#echo -n $ii >> result_preprocess.csv
#./ASpT_SpMM_GPU/pure_preprocess $fpath >> result_preprocess.csv
#./Preprocess_opt/preprocess $fpath >> result_preprocess.csv
# echo ">>>>>>>>>>"
#done
#mv result_preprocess.csv ../result/

filenames=$(cat mtxList.txt)

for filename in $filenames; do
    echo -n ${filename} >> result_preprocess.csv
    echo "$(./eval/get_matrix_info $base_path/$filename.mtx) $(./ASpT_SpMM_GPU/pure_preprocess $base_path/$filename.mtx) $(./Preprocess_opt/preprocess $base_path/$filename.mtx)" >> result_preprocess.csv
    #./ASpT_SpMM_GPU/pure_preprocess $base_path/$filename.mtx >> result_preprocess.csv
    #echo "$(./Preprocess_opt/preprocess $base_path/$filename.mtx)" > result_preprocess.csv
    #./Preprocess_opt/preprocess $base_path/$filename.mtx >> result_preprocess.csv
done

mv result_preprocess.csv ../preprocess_result/
rm mtxList.txt