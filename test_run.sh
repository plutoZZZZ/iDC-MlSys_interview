#!/bin/bash
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active

for i in 1 2 3 4 5; do
    echo "Run $i:"
    ./gcn_cpu_final 64 16 8 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin weight/W_16_8.bin
done
