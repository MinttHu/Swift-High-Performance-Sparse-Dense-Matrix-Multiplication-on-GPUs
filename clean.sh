#!/bin/bash

#clean executable files
rm src/test
rm test/irr_regular_ratio/test
rm test/preprocess/test
rm test/LoadBalance/test
rm test/MTX_info/test
rm test/ncu/test

rm test/irregular_K/FP32/test
rm test/irregular_K/FP64/test

rm test/k-32/CMA-compare/test
rm test/k-32/FP32/test
rm test/k-32/FP64/test
rm test/k-32/idealCMA/test
rm test/k-32/irr_opt/test

rm test/k-128/CMA-compare-128/test
rm test/k-128/FP32-128/test
rm test/k-128/FP64-128/test
rm test/k-128/idealCMA-128/test
rm test/k-128/irr_opt-128/test

# Clean up generated data files
rm src/data/*

rm test/irr_regular_ratio/data/*
rm test/preprocess/data/*   
rm test/LoadBalance/data/*  
rm test/MTX_info/data/*
rm test/ncu/data/*

rm test/k-32/CMA-compare/data/*
rm test/k-32/FP32/data/*
rm test/k-32/FP64/data/*
rm test/k-32/idealCMA/data/*
rm test/k-32/irr_opt/data/* 

rm test/k-128/CMA-compare-128/data/*
rm test/k-128/FP32-128/data/*
rm test/k-128/FP64-128/data/*
rm test/k-128/idealCMA-128/data/*   
rm test/k-128/irr_opt-128/data/*    

rm test/irregular_K/FP32/data/*
rm test/irregular_K/FP64/data/*

rm DataProcess/CMA-compare/*.txt
rm DataProcess/FP32/*.txt
rm DataProcess/FP64/*.txt
rm DataProcess/idealCMA/*.txt
rm DataProcess/irr_opt/*.txt    
rm DataProcess/irr_regular_ratio/*.txt 
rm DataProcess/irregular_K/*.txt
rm DataProcess/LoadBalance/*.txt
rm DataProcess/MTX_info/*.txt
rm DataProcess/ncu/*.ncu-rep
rm DataProcess/preprocess/*.txt
rm DataProcess/SOTA_ncu/*.ncu-rep

