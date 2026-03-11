CXX = g++
CXXFLAGS = -O3 -ffast-math -fopenmp -march=native -mtune=native -std=c++17
LDFLAGS = -fopenmp -lpthread

all: baseline cpu_final cpu_opt_new

baseline: example/gcn.cpp
	$(CXX) $(CXXFLAGS) -o baseline example/gcn.cpp $(LDFLAGS)

cpu_final: gcn_cpu_final.cpp
	$(CXX) $(CXXFLAGS) -o cpu_final gcn_cpu_final.cpp $(LDFLAGS)

cpu_opt_new: gcn_cpu_opt_new.cpp
	$(CXX) $(CXXFLAGS) -o cpu_opt_new gcn_cpu_opt_new.cpp $(LDFLAGS)

clean:
	rm -f baseline cpu_final cpu_opt_new
