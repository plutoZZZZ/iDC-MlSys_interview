#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <cstdlib>

int v_num = 1024;

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

void XW_out16(int in_dim, const float *__restrict__ in_X, 
              float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * 16];
        
        __m256 y0_7 = _mm256_setzero_ps();
        __m256 y8_15 = _mm256_setzero_ps();
        
        #pragma unroll 16
        for (int k = 0; k < in_dim; k++) {
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

int main() {
    float *in_X = (float*)aligned_alloc(64, 1024 * 64 * sizeof(float));
    float *out_X = (float*)aligned_alloc(64, 1024 * 16 * sizeof(float));
    float *W = (float*)aligned_alloc(64, 64 * 16 * sizeof(float));
    
    for (int i = 0; i < 1024 * 64; i++) in_X[i] = 1.0f;
    for (int i = 0; i < 64 * 16; i++) W[i] = 1.0f;
    
    omp_set_num_threads(12);
    
    // Warmup
    XW_64_16(in_X, out_X, W);
    XW_out16(64, in_X, out_X, W);
    
    // Test 1
    double start = omp_get_wtime();
    XW_64_16(in_X, out_X, W);
    double end = omp_get_wtime();
    printf("XW_64_16 time: %.8f ms\n", (end - start) * 1000);
    
    // Test 2
    start = omp_get_wtime();
    XW_out16(64, in_X, out_X, W);
    end = omp_get_wtime();
    printf("XW_out16 time: %.8f ms\n", (end - start) * 1000);
    
    printf("Alignment check: in_X: %p, out_X: %p, W: %p\n", in_X, out_X, W);
    
    free(in_X);
    free(out_X);
    free(W);
    return 0;
}
