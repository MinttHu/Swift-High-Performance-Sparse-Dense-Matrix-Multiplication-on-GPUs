#!/bin/bash

#./Run_irregular_K.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt or mtxlist.txt
file_path="$1"
MTX_LIST="$2"
rm ../DataProcess/irregular_K/*.txt

cd ../test/irregular_K/FP32

filenames=$(cat $MTX_LIST)

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 24
done
mv ./data/Swift_irregular_K_FP32.txt ./data/FP32_K24.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 48
done
mv ./data/Swift_irregular_K_FP32.txt ./data/FP32_K48.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 96
done
mv ./data/Swift_irregular_K_FP32.txt ./data/FP32_K96.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 192
done
mv ./data/Swift_irregular_K_FP32.txt ./data/FP32_K192.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 384
done
mv ./data/Swift_irregular_K_FP32.txt ./data/FP32_K384.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 768
done
mv ./data/Swift_irregular_K_FP32.txt ./data/FP32_K768.txt

cp ./data/*.txt ../../../DataProcess/irregular_K/

cd ../FP64
#filenames=$(cat mtxlist.txt)
filenames=$(cat test_mtx_list.txt)

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 24
done
mv ./data/Swift_irregular_K_FP64.txt ./data/FP64_K24.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 48
done
mv ./data/Swift_irregular_K_FP64.txt ./data/FP64_K48.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 96
done
mv ./data/Swift_irregular_K_FP64.txt ./data/FP64_K96.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 192
done
mv ./data/Swift_irregular_K_FP64.txt ./data/FP64_K192.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 384
done
mv ./data/Swift_irregular_K_FP64.txt ./data/FP64_K384.txt

for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx 768
done
mv ./data/Swift_irregular_K_FP64.txt ./data/FP64_K768.txt

cp ./data/*.txt ../../../DataProcess/irregular_K/
