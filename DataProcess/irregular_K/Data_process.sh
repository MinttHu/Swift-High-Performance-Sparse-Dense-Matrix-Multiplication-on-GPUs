#!/bin/bash
./filter_line.sh FP32_K24.txt 32-24.txt 13
./filter_line.sh FP32_K48.txt 32-48.txt 13
./filter_line.sh FP32_K96.txt 32-96.txt 13
./filter_line.sh FP32_K192.txt 32-192.txt 13
./filter_line.sh FP32_K384.txt 32-384.txt 13    
./filter_line.sh FP32_K768.txt 32-768.txt 13

./fileter_line.sh FP64_K24.txt 64-24.txt 13
./filter_line.sh FP64_K48.txt 64-48.txt 13
./filter_line.sh FP64_K96.txt 64-96.txt 13
./filter_line.sh FP64_K192.txt 64-192.txt 13
./filter_line.sh FP64_K384.txt 64-384.txt 13
./filter_line.sh FP64_K768.txt 64-768.txt 13


rm 32-24.txt
rm 32-48.txt
rm 32-96.txt    
rm 32-192.txt   
rm 32-384.txt   
rm 32-768.txt

rm 64-24.txt
rm 64-48.txt
rm 64-96.txt
rm 64-192.txt
rm 64-384.txt
rm 64-768.txt

