#!/bin/bash
./filter_line.sh ideal-cma.txt Swift_32.txt 13
./filter_line.sh ideal-cma-128.txt Swift_128.txt 13
rm Swift_32.txt
rm Swift_128.txt