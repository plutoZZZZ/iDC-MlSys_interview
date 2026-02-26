#!/bin/bash
cd /mnt/d/Code/qianshi0225/2/seed/iDC-MlSys_interview/example
g++ -O3 -fopenmp -mavx2 -mfma -o gcn_chunk gcn_chunk_parse.cpp

echo "=== Run 1 ==="
./gcn_chunk 64 16 8 /tmp/100000_graph.txt /tmp/embedding.bin /tmp/W1.bin /tmp/W2.bin
echo "=== Run 2 ==="
./gcn_chunk 64 16 8 /tmp/100000_graph.txt /tmp/embedding.bin /tmp/W1.bin /tmp/W2.bin
echo "=== Run 3 ==="
./gcn_chunk 64 16 8 /tmp/100000_graph.txt /tmp/embedding.bin /tmp/W1.bin /tmp/W2.bin
