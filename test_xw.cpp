#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <string>
#include <cstring>

using namespace std;

int v_num = 1024;

float *X0, *W1, *X1_inter;

void readFloat(const char *fname, float *&dst, int num)
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

// Version from cpu_opt_new - chunk size 64
void XW_64_16_chunk64(const float *__restrict__ in_X, float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 64)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * 64];
        float* y = &out_X[i * 16];
        
        __m256 y0_7 = _mm256_setzero_ps();
        __m256 y8_15 = _mm256_setzero_ps();
        
        #pragma GCC unroll 16
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

// Version from cpu_final - chunk size 4
void XW_64_16_chunk4(const float *__restrict__ in_X, float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * 64];
        float* y = &out_X[i * 16];
        
        __m256 y0_7 = _mm256_setzero_ps();
        __m256 y8_15 = _mm256_setzero_ps();
        
        #pragma GCC unroll 16
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

// Version with dynamic schedule
void XW_64_16_dynamic(const float *__restrict__ in_X, float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * 64];
        float* y = &out_X[i * 16];
        
        __m256 y0_7 = _mm256_setzero_ps();
        __m256 y8_15 = _mm256_setzero_ps();
        
        #pragma GCC unroll 16
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

// Version without restrict
void XW_64_16_norestrict(float *in_X, float *out_X, float *W)
{
    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * 64];
        float* y = &out_X[i * 16];
        
        __m256 y0_7 = _mm256_setzero_ps();
        __m256 y8_15 = _mm256_setzero_ps();
        
        #pragma GCC unroll 16
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

template<typename F>
double benchmark(F func, const float* in_X, float* out_X, const float* W, const char* name, int iterations = 10) {
    // Warmup
    func(in_X, out_X, W);
    
    auto start = chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        func(in_X, out_X, W);
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> diff = end - start;
    double avg_time = (diff.count() * 1000) / iterations;
    printf("%-25s: %.6f ms avg (over %d runs)\n", name, avg_time, iterations);
    return avg_time;
}

int main() {
    readFloat("embedding/1024.bin", X0, 1024 * 64);
    readFloat("weight/W_64_16.bin", W1, 64 * 16);
    initFloat(X1_inter, 1024 * 16);
    
    printf("X0: %p (align %ld), W1: %p (align %ld), X1_inter: %p (align %ld)\n", 
           X0, ((uintptr_t)X0) % 64,
           W1, ((uintptr_t)W1) % 64,
           X1_inter, ((uintptr_t)X1_inter) % 64);
    
    omp_set_num_threads(12);
    omp_set_dynamic(0);
    printf("Threads: %d\n", omp_get_max_threads());
    
    printf("\n=== Benchmarking XW_64_16 variants ===\n");
    
    benchmark(XW_64_16_chunk64, X0, X1_inter, W1, "chunk64");
    benchmark(XW_64_16_chunk4, X0, X1_inter, W1, "chunk4");
    benchmark(XW_64_16_dynamic, X0, X1_inter, W1, "dynamic");
    benchmark(XW_64_16_norestrict, X0, X1_inter, W1, "norestrict");
    
    // Verify results are the same
    float *out1 = (float*)aligned_alloc(64, 1024 * 16 * sizeof(float));
    float *out2 = (float*)aligned_alloc(64, 1024 * 16 * sizeof(float));
    
    memset(out1, 0, 1024 * 16 * sizeof(float));
    memset(out2, 0, 1024 * 16 * sizeof(float));
    
    XW_64_16_chunk4(X0, out1, W1);
    XW_64_16_chunk64(X0, out2, W1);
    
    float max_diff = 0.0f;
    for (int i = 0; i < 1024 * 16; i++) {
        float diff = fabsf(out1[i] - out2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("\nMax difference between chunk4 and chunk64: %.10f\n", max_diff);
    
    free(X0);
    free(W1);
    free(X1_inter);
    free(out1);
    free(out2);
    
    return 0;
}
