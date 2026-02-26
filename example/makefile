CXX = g++
NVCC = /usr/local/cuda-12.2/bin/nvcc
CXXFLAGS = -O3 -fopenmp -mavx2 -mfma -march=native
NVCCFLAGS = -O3 -arch=sm_89

all: cpu cuda

cpu: gcn_cpu_optimized.cpp
	$(CXX) $(CXXFLAGS) -o ../gcn_cpu.exe gcn_cpu_optimized.cpp

cuda: gcn_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o ../gcn_cuda.exe gcn_cuda.cu

baseline: gcn.cpp
	g++ -O0 -o ../gcn_baseline.exe gcn.cpp

clean:
	rm -f ../gcn_cpu.exe ../gcn_cuda.exe ../gcn_baseline.exe
