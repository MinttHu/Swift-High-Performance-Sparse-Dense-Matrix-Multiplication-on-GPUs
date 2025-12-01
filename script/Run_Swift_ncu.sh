#!/bin/bash
#./Run_Swift_ncu.sh /Direct/to/Swift/MTX_samples /Direct/to/Swift/DataProcess/ncu test_mtx_list.txt or mtxlist.txt
file_path="$1" 
storage_path="$2"
MTX_LIST="$3"
rm ../DataProcess/ncu/*.ncu-rep

cd ../test/ncu
filenames=$(cat $MTX_LIST)
for filename in $filenames; do
    ncu --set full -o $storage_path/Swift_${filename} ./test -d 0 $file_path/$filename.mtx
done