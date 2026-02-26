#!/bin/bash
cd /mnt/d/Code/qianshi0225/2/seed/iDC-MlSys_interview/example
g++ -O3 -fopenmp -mavx2 -mfma -o gcn_optimized gcn_optimized.cpp
./gcn_optimized 64 16 8 ../graph/100000_graph.txt ../embedding/100000.bin ../weight/W_64_16.bin ../weight/W_16_8.bin
