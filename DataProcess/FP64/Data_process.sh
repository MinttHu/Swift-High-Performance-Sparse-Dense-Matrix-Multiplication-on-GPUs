#!/bin/bash

mv result__spmm_f64_n32.csv result__spmm_f64_n32.txt
mv result__spmm_f64_n128.csv result__spmm_f64_n128.txt

sed 's/ //g' result__spmm_f64_n32.txt > result__spmm_f64_n32_1.txt
sed 's/ //g' result__spmm_f64_n128.txt > result__spmm_f64_n128_1.txt
sed 's/,/ /g' result__spmm_f64_n32_1.txt > result__spmm_f64_n32_2.txt
sed 's/,/ /g' result__spmm_f64_n128_1.txt > result__spmm_f64_n128_2.txt

./filter_line.sh result__spmm_f64_n32_2.txt spmm_32.txt 10
./filter_line.sh result__spmm_f64_n128_2.txt spmm_128.txt 10

rm result__spmm_f64_n32.txt
rm result__spmm_f64_n128.txt
rm result__spmm_f64_n32_1.txt
rm result__spmm_f64_n128_1.txt
rm spmm_32.txt
rm spmm_128.txt

mv result__spmm_f64_n32_2.txt result__spmm_f64_n32.txt
mv result__spmm_f64_n128_2.txt result__spmm_f64_n128.txt

mv result_ASpT_spmm_f64_n32.csv result_ASpT_spmm_f64_n32.txt
mv result_ASpT_spmm_f64_n128.csv result_ASpT_spmm_f64_n128.txt

sed 's/ //g' result_ASpT_spmm_f64_n32.txt > result_ASpT_spmm_f64_n32_1.txt
sed 's/ //g' result_ASpT_spmm_f64_n128.txt > result_ASpT_spmm_f64_n128_1.txt
sed 's/,/ /g' result_ASpT_spmm_f64_n32_1.txt > result_ASpT_spmm_f64_n32_2.txt
sed 's/,/ /g' result_ASpT_spmm_f64_n128_1.txt > result_ASpT_spmm_f64_n128_2.txt

./filter_line.sh result_ASpT_spmm_f64_n32_2.txt ASpT_32.txt 6
./filter_line.sh result_ASpT_spmm_f64_n128_2.txt ASpT_128.txt 6

rm result_ASpT_spmm_f64_n32.txt
rm result_ASpT_spmm_f64_n128.txt
rm result_ASpT_spmm_f64_n32_1.txt
rm result_ASpT_spmm_f64_n128_1.txt
rm ASpT_32.txt
rm ASpT_128.txt

mv result_ASpT_spmm_f64_n32_2.txt result_ASpT_spmm_f64_n32.txt
mv result_ASpT_spmm_f64_n128_2.txt result_ASpT_spmm_f64_n128.txt

sed -i '1d' result_ASpT_spmm_f64_n32.txt
sed -i '1d' result_ASpT_spmm_f64_n128.txt
sed -i '1d' result__spmm_f64_n32.txt
sed -i '1d' result__spmm_f64_n128.txt

./filter_line.sh results_Swift_FP64_32.txt Swift_32.txt 13
./filter_line.sh results_Swift_FP64_128.txt Swift_128.txt 13
rm Swift_32.txt
rm Swift_128.txt

#check 0.000000  inf