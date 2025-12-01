# Swift


## A Code Structure
Swift

|- DataProcess

|- MTX

|- MTX-samples

|- script

|- SOTA_script

|- src

|- test

clean.sh

compile.sh

Makefile

Readme

SOTA_method_compile.sh

## Download test matrix dataset
cd MTX_samples
./Data_get.sh


## B How to compile

1 Modify Makefile file: change the install directory of CUDA

2 ./compile.sh

## C How to compile SOTA method:
git clone --recursive https://github.com/CRAFT-THU/RoDe.git
2 ./SOTA_method_compile.sh



## D How to test if compile successful:

1 cd script

2 ./Function_test.sh  

## E How to run small scale of test:

1 cd script

2 ./Run_CMA_compare.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

3 ./Run_FP32.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

4 ./Run_FP64.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

5 ./Run_idealCMA.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

6 ./Run_irr_opt.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

7 ./Run_irr_regular_ratio.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

8 ./Run_irregular_K.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

9 ./Run_LoadBalance.sh

10 ./Run_MTX_info.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

11 ./Run_preprocess.sh /Direct/to/Swift/MTX_samples test_mtx_list.txt

12 ./Run_Swift_ncu.sh /Direct/to/Swift/MTX_samples /Direct/to/Swift/DataProcess/ncu test_mtx_list.txt

13 ./Run_SOTA_method.sh /Direct/to/MTX test_mtx_list.txt

14 ./Run_SOTA_preprocess.sh /Direct/to/MTX test_mtx_list.txt

15 ./Run_SOTA_ncu.sh /home/hjy/Swift/MTX_samples /home/hjy/Swift/DataProcess/SOTA_ncu test_mtx_list.txt

## F How to run large scale of test:
1 Down load the dataset of sparse matrix (Section H)
2 Change the direct to dataset
3 Replace the test_mtx_list.txt to mtxlist.txt for 1-8,10-14 in Section E

## G The function of the other script in script file:

./Function_test.sh            (This script is used to test if the compilation is successful)

./Run_CMA_compare.sh          (This script is used to compare the Swift with or without coalesced memory access)

./Run_FP32.sh                 (This script is used to test the Swift with FP32)

./Run_FP64.sh                 (This script is used to test the Swift with FP64)

./Run_idealCMA                (This script is used to test the ideal situation of matrix multiplication with or without coalesced memory access)

./Run_irr_opt                 (This script is used to compare the Swift with or without optimization of the regular part)

./Run_irr_regular_ratio.sh    (This script is used to categorize the ratio of regular and irregular parts of the sparse matrix after the pre-processing of Swift)

./Run_irregular_K.sh          (This script is used to compare the performances of Swift with cuSPARSE under various irregular K)

./Run_LoadBalance.sh          (This script is used to compare the performance of SpMM with and without load balance)

./Run_preprocess.sh           (This script is used to test the preprocess time of Swift)

./Run_SOTA_method.sh          (This script is used to test the SOTA methods)

./Run_SOTA_preprocess.sh      (This script is used to test the preprocess of SOTA methods)

./Run_SOTA_ncu.sh             (This script is used to test the memory bandwidth of SOTA via the Nsight compute)

./Run_Swift_ncu.sh            (This script is used to test the memory bandwidth of Swift  via the Nsight compute)


## H How to download matrices:

cd MTX

./Data_get.sh
