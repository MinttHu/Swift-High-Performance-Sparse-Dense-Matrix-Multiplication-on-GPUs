#!/bin/bash

rm result__spmm_f64_n32.csv
mv result__spmm_f64_n128.csv result__spmm_f64_n128.txt

sed 's/ //g' result__spmm_f64_n128.txt > result__spmm_f64_n128_1.txt
sed 's/,/ /g' result__spmm_f64_n128_1.txt > result__spmm_f64_n128_2.txt

./filter_line.sh result__spmm_f64_n128_2.txt spmm_128.txt 10


rm result__spmm_f64_n128.txt
rm result__spmm_f64_n128_1.txt
rm spmm_128.txt

mv result__spmm_f64_n128_2.txt result__spmm_f64_n128.txt

rm result_ASpT_spmm_f64_n32.csv
mv result_ASpT_spmm_f64_n128.csv result_ASpT_spmm_f64_n128.txt

sed 's/ //g' result_ASpT_spmm_f64_n128.txt > result_ASpT_spmm_f64_n128_1.txt
sed 's/,/ /g' result_ASpT_spmm_f64_n128_1.txt > result_ASpT_spmm_f64_n128_2.txt

./filter_line.sh result_ASpT_spmm_f64_n128_2.txt ASpT_128.txt 6

rm result_ASpT_spmm_f64_n128.txt
rm result_ASpT_spmm_f64_n128_1.txt
rm ASpT_128.txt


mv result_ASpT_spmm_f64_n128_2.txt result_ASpT_spmm_f64_n128.txt


sed -i '1d' result_ASpT_spmm_f64_n128.txt
sed -i '1d' result__spmm_f64_n128.txt

./filter_line.sh MTX_info.txt Swift_32.txt 17
rm Swift_32.txt



rm result_ASpT_spmm_f32_n32.csv 
mv result_ASpT_spmm_f32_n128.csv result_ASpT_spmm_f32_n128.txt

sed 's/ //g' result_ASpT_spmm_f32_n128.txt > result_ASpT_spmm_f32_n128_1.txt
sed 's/,/ /g' result_ASpT_spmm_f32_n128_1.txt > result_ASpT_spmm_f32_n128_2.txt

./filter_line.sh result_ASpT_spmm_f32_n128_2.txt ASpT_128.txt 6

rm result_ASpT_spmm_f32_n128.txt
rm result_ASpT_spmm_f32_n128_1.txt
rm ASpT_128.txt

mv result_ASpT_spmm_f32_n128_2.txt result_ASpT_spmm_f32_n128.txt
sed -i '1d' result_ASpT_spmm_f32_n128.txt
./filter_line.sh results_Swift_FP32_128.txt Swift_128.txt 13
rm Swift_128.txt

#check 0.000000  inf