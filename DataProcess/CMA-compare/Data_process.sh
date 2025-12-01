#!/bin/bash
./filter_line.sh results_CMA_VS_nonCMA_32.txt Swift_32.txt 13
./filter_line.sh results_CMA_VS_nonCMA_128.txt Swift_128.txt 13
rm Swift_32.txt
rm Swift_128.txt