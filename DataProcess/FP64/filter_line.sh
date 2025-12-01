#!/bin/bash
#./filter_lines.sh A.txt B.txt 

input_file="$1"
output_file="$2"
min_columns="$3"
tmp_file="${input_file}_tmp"

if [ ! -f "$input_file" ]; then
    echo "error: file $input_file no exist。"
    exit 1
fi


awk -v n="$min_columns" '{
    if (NF < n)
        print > out_file;     
    else
        print > tmp_file;     
}' out_file="$output_file" tmp_file="$tmp_file" "$input_file"


mv "$tmp_file" "$input_file"
