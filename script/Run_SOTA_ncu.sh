
#!/bin/bash

#./Run_SOTA_ncu.sh /Direct/to/Swift/MTX_samples /Direct/to/Swift/DataProcess/SOTA_ncu test_mtx_list.txt ncu_mtxlist.txt
file_path="$1"
storage_path="$2"
MTX_LIST="$3"
rm ../DataProcess/SOTA_ncu/*.ncu-rep
#aspt

cp ../RoDe/SOTA_script/$MTX_LIST ../RoDe/build/ASpT_SpMM_GPU
cp ../RoDe/SOTA_script/$MTX_LIST ../RoDe/build/eval
cd ../RoDe/build/ASpT_SpMM_GPU

filenames=$(cat $MTX_LIST)
for filename in $filenames; do
    ncu --set full -o $storage_path/ASpT_${filename} ./ASpT_spmm_f64_n32 $file_path/$filename.mtx 32
done

rm $MTX_LIST

cd ../eval

#rode #cusparse  #spuntiki
filenames=$(cat $MTX_LIST)
for filename in $filenames; do
    ncu --set full -o $storage_path/Spuntiki_${filename} ./eval_spmm_f64_n32 $file_path/$filename.mtx
done

rm $MTX_LIST





