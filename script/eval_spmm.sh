#!/bin/bash
file_path="$1"
MTX_LIST="$2"


filenames=$(cat $MTX_LIST)


for filename in $filenames; do
    ./test -d 0 $file_path/$filename.mtx
done