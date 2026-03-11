#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdint>

using namespace std;

int v_num = 1024;
int e_num = 4096;

vector<int> raw_graph;
vector<int> csr_row_ptr;
vector<int> csr_col_idx;
vector<float> csr_val;
vector<int> degree;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)aligned_alloc(64, num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)aligned_alloc(64, num * sizeof(float));
    for (int i = 0; i < num; i++) {
        dst[i] = 0.0f;
    }
}

void XW_64_16(const float *__restrict__ in_X, float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * 64];
        float* y = &out_X[i * 16];
        
        __m256 y0_7 = _mm256_setzero_ps();
        __m256 y8_15 = _mm256_setzero_ps();
        
        #pragma unroll 16
        for (int k = 0; k < 64; k++) {
            float xv = x[k];
            const float* w = &W[k * 16];
            
            __m256 x_vec = _mm256_set1_ps(xv);
            __m256 w0_7 = _mm256_load_ps(w);
            __m256 w8_15 = _mm256_load_ps(w + 8);
            
            y0_7 = _mm256_fmadd_ps(x_vec, w0_7, y0_7);
            y8_15 = _mm256_fmadd_ps(x_vec, w8_15, y8_15);
        }
        
        _mm256_store_ps(y, y0_7);
        _mm256_store_ps(y + 8, y8_15);
    }
}

int main(int argc, char** argv) {
    char fname1[] = "embedding/1024.bin";
    char fname2[] = "weight/W_64_16.bin";
    char fname3[] = "weight/W_16_8.bin";
    
    readFloat(fname1, X0, 1024 * 64);
    readFloat(fname2, W1, 64 * 16);
    readFloat(fname3, W2, 16 * 8);
    
    initFloat(X1_inter, 1024 * 16);
    
    printf("X0: %p, W1: %p, W2: %p, X1_inter: %p\n", X0, W1, W2, X1_inter);
    printf("X0 alignment: %ld\n", ((uintptr_t)X0) % 64);
    printf("W1 alignment: %ld\n", ((uintptr_t)W1) % 64);
    printf("W2 alignment: %ld\n", ((uintptr_t)W2) % 64);
    printf("X1_inter alignment: %ld\n", ((uintptr_t)X1_inter) % 64);
    
    omp_set_num_threads(12);
    
    // Warmup
    XW_64_16(X0, X1_inter, W1);
    
    auto start = chrono::steady_clock::now();
    XW_64_16(X0, X1_inter, W1);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> diff = end - start;
    printf("XW_64_16 time with real data: %.8f ms\n", diff.count() * 1000);
    
    return 0;
}
