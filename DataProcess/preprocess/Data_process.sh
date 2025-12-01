#!/bin/bash

mv pre_result__spmm.csv pre_result__spmm.txt
mv pre_result_ASpT.csv pre_result_ASpT.txt

sed 's/ //g' pre_result__spmm.txt > pre_result__spmm_1.txt
sed 's/ //g' pre_result_ASpT.txt > pre_result_ASpT_1.txt
sed 's/,/ /g' pre_result__spmm_1.txt > pre_result__spmm_2.txt
sed 's/,/ /g' pre_result_ASpT_1.txt > pre_result_ASpT_2.txt

./filter_line.sh pre_result__spmm_2.txt spmm_32.txt 12
./filter_line.sh pre_result_ASpT_2.txt spmm_128.txt 7

rm pre_result__spmm.txt
rm pre_result_ASpT.txt
rm pre_result__spmm_1.txt
rm pre_result_ASpT_1.txt
rm spmm_32.txt
rm spmm_128.txt

mv pre_result__spmm_2.txt pre_result__spmm.txt
mv pre_result_ASpT_2.txt pre_result_ASpT.txt


sed -i '1d' pre_result__spmm.txt
sed -i '1d' pre_result_ASpT.txt

./filter_line.sh preprocess_Swift.txt Swift_32.txt 21
rm Swift_32.txt


#check 0.000000  inf