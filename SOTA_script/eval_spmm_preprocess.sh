#!/bin/bash

base_path=$2
MTX_LIST=$3
echo "Dataset,nr,nc,nnz,Sputnikpre,RoDepre,Sputnik_time,Sputnik_gflops,cuSPARSE_time,cuSPARSE_gflops,ours_time,ours_gflops"

filenames=$(cat $MTX_LIST)

for filename in $filenames; do
    echo -n ${filename}
    echo "$(./get_matrix_info $base_path/$filename.mtx)$(../Preprocess_opt/preprocess $base_path/$filename.mtx)$(./$1 $base_path/$filename.mtx)" 
done