CXX = g++
CXXFLAGS = -O3 -ffast-math -fopenmp -march=native -mtune=native -std=c++17 -mavx2 -mfma
LDFLAGS = -fopenmp -lpthread

all: baseline cpu_opt

baseline: gcn.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

cpu_opt: gcn_cpu_opt.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f baseline cpu_opt
