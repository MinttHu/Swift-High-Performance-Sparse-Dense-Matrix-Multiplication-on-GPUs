echo "Dataset,nr,nc,nnz,ASpT_time,ASpT_gflops"
base_path=$1
MTX_LIST=$2

filenames=$(cat $MTX_LIST)

for filename in $filenames; do
    echo -n ${filename}
    echo "$(./../eval/get_matrix_info $base_path/$filename.mtx) $(./ASpT_spmm_f32_n32 $base_path/$filename.mtx 32)"     
done