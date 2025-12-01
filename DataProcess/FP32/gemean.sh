#!/bin/bash

file_a="result__spmm_f32_n32.txt"
file_b="result_ASpT_spmm_f32_n32.txt"
file_c="results_Swift_FP32_32.txt"
output_file="gemean_fp32_n32.txt"

output_file1="tmp1.txt"
output_file2="tmp2.txt"
output_file3="tmp3.txt"
output_file4="tmp4.txt"
output_file5="tmp5.txt"
output_file6="tmp6.txt"
output_file7="tmp7.txt"
output_file8="tmp8.txt"

awk 'NR==FNR { a[$1]; next } $1 in a'  $file_a $file_b > $output_file1

awk 'NR==FNR { a[$1]; next } $1 in a'  $file_b $file_a > $output_file2


if [ ! -f "$output_file1" ]; then
  echo "$output_file1 no exist"
  exit 1
fi

if [ ! -f "$file_a" ]; then
  echo "$file_a no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$file_a"; then
        success=false
        break
    fi
done < "$output_file1"

if [ "$success" = true ]; then
    echo "ASpT check 1 success"
else
    echo "ASpT check1 failed"
fi

if [ ! -f "$output_file2" ]; then
  echo "$output_file2 no exist"
  exit 1
fi

if [ ! -f "$file_b" ]; then
  echo "$file_b no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$file_b"; then
        success=false
        break
    fi
done < "$output_file2"

if [ "$success" = true ]; then
    echo "Sputik check 2 success"
else
    echo "Sputik check2 failed"
fi


if [ ! -f "$output_file1" ]; then
  echo "$output_file1 文件不存在"
  exit 1
fi

if [ ! -f "$output_file2" ]; then
  echo "$output_file2 no exist"
  exit 1
fi

while IFS= read -r line_a && IFS= read -r line_b <&3; do
    line_number=$((line_number + 1))
    column_a=$(echo "$line_a" | awk '{print $1}')
    column_b=$(echo "$line_b" | awk '{print $1}')

    if [ "$column_a" != "$column_b" ]; then
         echo "ASpT-Sputiki check 3 failed at line $line_number"   
        exit 1
    fi
done < "$output_file2" 3< "$output_file1"

echo "ASpT-Sputiki check 3 success"


while read lineA <&3 && read lineB <&4; do
    col2=$(echo "$lineA" | awk '{print $1}') #name
    col7=$(echo "$lineA" | awk '{print $4}') #nnz
    col8=$(echo "$lineA" | awk '{print $5}') #Spuntik time
    col9=$(echo "$lineA" | awk '{print $6}') #Sputik gflops
    col10=$(echo "$lineA" | awk '{print $7}') #cusparse time
    col11=$(echo "$lineA" | awk '{print $8}') #cusparse gflops
    col12=$(echo "$lineA" | awk '{print $9}') #Rode time
    col13=$(echo "$lineA" | awk '{print $10}') #Rode gflops

    col5B=$(echo "$lineB" | awk '{print $5}') #ASpT time
    col6B=$(echo "$lineB" | awk '{print $6}') #ASpT gflops
    echo "$col2 $col7 $col8 $col9 $col10 $col11 $col12 $col13 $col5B $col6B" >> $output_file3
done 3< $output_file2 4< $output_file1


echo "Sputik cusparse rode ASpT finish"



awk 'NR==FNR { a[$1]; next } $1 in a'  $output_file3 $file_c > $output_file6

awk 'NR==FNR { a[$1]; next } $1 in a'  $file_c $output_file3 > $output_file7


if [ ! -f "$output_file6" ]; then
  echo "$output_file6 no exist"
  exit 1
fi

if [ ! -f "$output_file3" ]; then
  echo "$output_file3 no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$output_file3"; then
        success=false
        break
    fi
done < "$output_file6"

if [ "$success" = true ]; then
    echo "Swift-SOTA check 1 success"
else
    echo "Swift-SOTA check 1 failed"
fi


if [ ! -f "$output_file7" ]; then
  echo "$output_file7 no exist"
  exit 1
fi

if [ ! -f "$file_c" ]; then
  echo "$file_c no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$file_c"; then
        success=false
        break
    fi
done < "$output_file7"

if [ "$success" = true ]; then
    echo "Swift-SOTA check 2 success"
else
    echo "Swift-SOTA check 2 failed"
fi


if [ ! -f "$output_file6" ]; then
  echo "$output_file6 no exist"
  exit 1
fi

if [ ! -f "$output_file7" ]; then
  echo "$output_file7 no exist"
  exit 1
fi

while IFS= read -r line_a && IFS= read -r line_b <&3; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    column_b=$(echo "$line_b" | awk '{print $1}')

    if [ "$column_a" != "$column_b" ]; then
        echo "Swift-SOTA check 3 failed"
        exit 1
    fi
done < "$output_file6" 3< "$output_file7"

echo "Swift-SOTA check 3 success"

while read lineA <&3 && read lineB <&4; do
    col1=$(echo "$lineA" | awk '{print $1}') #matrix name
    col2=$(echo "$lineA" | awk '{print $2}') #nnz
    col3=$(echo "$lineA" | awk '{print $3}') #Sputiki time
    col4=$(echo "$lineA" | awk '{print $4}') #Sputiki flops
    col5=$(echo "$lineA" | awk '{print $5}') #cusparse time 
    col6=$(echo "$lineA" | awk '{print $6}') #cusparse flops
    col7=$(echo "$lineA" | awk '{print $7}') #rode time
    col8=$(echo "$lineA" | awk '{print $8}') #rode flops
    col9=$(echo "$lineA" | awk '{print $9}') #ASpT time
    col10=$(echo "$lineA" | awk '{print $10}') #ASpT flops


    col9B=$(echo "$lineB" | awk '{print $13}') #Swift time

    echo "$col1 $col2 $col3 $col4 $col5 $col6 $col7 $col8 $col9 $col10 $col9B" >> $output_file8
done 3< $output_file7 4< $output_file6

awk '{print $0, NR}' "$output_file8" > "$output_file"

rm $output_file1
rm $output_file2
rm $output_file3

rm $output_file6
rm $output_file7
rm $output_file8

echo "-------------------------------------"

file_a="result__spmm_f32_n128.txt"
file_b="result_ASpT_spmm_f32_n128.txt"
file_c="results_Swift_FP32_128.txt"
output_file="gemean_fp32_n128.txt"

output_file1="tmp1.txt"
output_file2="tmp2.txt"
output_file3="tmp3.txt"
output_file4="tmp4.txt"
output_file5="tmp5.txt"
output_file6="tmp6.txt"
output_file7="tmp7.txt"
output_file8="tmp8.txt"


awk 'NR==FNR { a[$1]; next } $1 in a'  $file_a $file_b > $output_file1

awk 'NR==FNR { a[$1]; next } $1 in a'  $file_b $file_a > $output_file2


if [ ! -f "$output_file1" ]; then
  echo "$output_file1 no exist"
  exit 1
fi

if [ ! -f "$file_a" ]; then
  echo "$file_a no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$file_a"; then
        success=false
        break
    fi
done < "$output_file1"

if [ "$success" = true ]; then
    echo "ASpT check 1 success"
else
    echo "ASpT check1 failed"
fi

if [ ! -f "$output_file2" ]; then
  echo "$output_file2 no exist"
  exit 1
fi

if [ ! -f "$file_b" ]; then
  echo "$file_b no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$file_b"; then
        success=false
        break
    fi
done < "$output_file2"

if [ "$success" = true ]; then
    echo "Sputik check 2 success"
else
    echo "Sputik check2 failed"
fi


if [ ! -f "$output_file1" ]; then
  echo "$output_file1 文件不存在"
  exit 1
fi

if [ ! -f "$output_file2" ]; then
  echo "$output_file2 no exist"
  exit 1
fi

while IFS= read -r line_a && IFS= read -r line_b <&3; do
    line_number=$((line_number + 1))
    column_a=$(echo "$line_a" | awk '{print $1}')
    column_b=$(echo "$line_b" | awk '{print $1}')

    if [ "$column_a" != "$column_b" ]; then
         echo "ASpT-Sputiki check 3 failed at line $line_number"   
        exit 1
    fi
done < "$output_file2" 3< "$output_file1"

echo "ASpT-Sputiki check 3 success"


while read lineA <&3 && read lineB <&4; do
    col2=$(echo "$lineA" | awk '{print $1}') #name
    col7=$(echo "$lineA" | awk '{print $4}') #nnz
    col8=$(echo "$lineA" | awk '{print $5}') #Spuntik time
    col9=$(echo "$lineA" | awk '{print $6}') #Sputik gflops
    col10=$(echo "$lineA" | awk '{print $7}') #cusparse time
    col11=$(echo "$lineA" | awk '{print $8}') #cusparse gflops
    col12=$(echo "$lineA" | awk '{print $9}') #Rode time
    col13=$(echo "$lineA" | awk '{print $10}') #Rode gflops

    col5B=$(echo "$lineB" | awk '{print $5}') #ASpT time
    col6B=$(echo "$lineB" | awk '{print $6}') #ASpT gflops
    echo "$col2 $col7 $col8 $col9 $col10 $col11 $col12 $col13 $col5B $col6B" >> $output_file3
done 3< $output_file2 4< $output_file1


echo "Sputik cusparse rode ASpT finish"



awk 'NR==FNR { a[$1]; next } $1 in a'  $output_file3 $file_c > $output_file6

awk 'NR==FNR { a[$1]; next } $1 in a'  $file_c $output_file3 > $output_file7


if [ ! -f "$output_file6" ]; then
  echo "$output_file6 no exist"
  exit 1
fi

if [ ! -f "$output_file3" ]; then
  echo "$output_file3 no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$output_file3"; then
        success=false
        break
    fi
done < "$output_file6"

if [ "$success" = true ]; then
    echo "Swift-SOTA check 1 success"
else
    echo "Swift-SOTA check 1 failed"
fi


if [ ! -f "$output_file7" ]; then
  echo "$output_file7 no exist"
  exit 1
fi

if [ ! -f "$file_c" ]; then
  echo "$file_c no exist"
  exit 1
fi
success=true

while IFS= read -r line_a; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    if ! grep -q -w "$column_a" "$file_c"; then
        success=false
        break
    fi
done < "$output_file7"

if [ "$success" = true ]; then
    echo "Swift-SOTA check 2 success"
else
    echo "Swift-SOTA check 2 failed"
fi


if [ ! -f "$output_file6" ]; then
  echo "$output_file6 no exist"
  exit 1
fi

if [ ! -f "$output_file7" ]; then
  echo "$output_file7 no exist"
  exit 1
fi

while IFS= read -r line_a && IFS= read -r line_b <&3; do
    column_a=$(echo "$line_a" | awk '{print $1}')
    column_b=$(echo "$line_b" | awk '{print $1}')

    if [ "$column_a" != "$column_b" ]; then
        echo "Swift-SOTA check 3 failed"
        exit 1
    fi
done < "$output_file6" 3< "$output_file7"

echo "Swift-SOTA check 3 success"

while read lineA <&3 && read lineB <&4; do
    col1=$(echo "$lineA" | awk '{print $1}') #matrix name
    col2=$(echo "$lineA" | awk '{print $2}') #nnz
    col3=$(echo "$lineA" | awk '{print $3}') #Sputiki time
    col4=$(echo "$lineA" | awk '{print $4}') #Sputiki flops
    col5=$(echo "$lineA" | awk '{print $5}') #cusparse time 
    col6=$(echo "$lineA" | awk '{print $6}') #cusparse flops
    col7=$(echo "$lineA" | awk '{print $7}') #rode time
    col8=$(echo "$lineA" | awk '{print $8}') #rode flops
    col9=$(echo "$lineA" | awk '{print $9}') #ASpT time
    col10=$(echo "$lineA" | awk '{print $10}') #ASpT flops


    col9B=$(echo "$lineB" | awk '{print $13}') #Swift time

    echo "$col1 $col2 $col3 $col4 $col5 $col6 $col7 $col8 $col9 $col10 $col9B" >> $output_file8
done 3< $output_file7 4< $output_file6

awk '{print $0, NR}' "$output_file8" > "$output_file"

rm $output_file1
rm $output_file2
rm $output_file3

rm $output_file6
rm $output_file7
rm $output_file8