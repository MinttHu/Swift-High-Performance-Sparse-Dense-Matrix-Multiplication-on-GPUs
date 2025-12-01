base_path=$1
MTX_LIST=$2

echo "Dataset,nr,nc,nnz,Sputnik_time,Sputnik_gflops,cuSPARSE_time,cuSPARSE_gflops,ours_time,ours_gflops"
filenames=$(cat $MTX_LIST)

for filename in $filenames; do
    echo -n ${filename}
    echo "$(./get_matrix_info $base_path/$filename.mtx)$(./eval_spmm_f64_n128_p1 $base_path/$filename.mtx ) $(./eval_spmm_f64_n128_p2 $base_path/$filename.mtx)" | sed -Ee "s/CUDA Error: an illegal memory access was encountered//g"
done
