#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <xmmintrin.h>
#include <immintrin.h>
#include <cassert>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<int> raw_graph;

// CSR format
vector<int> csr_row_ptr;
vector<int> csr_col_idx;
vector<float> csr_val;
vector<int> degree;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

// AMD Zen 4 specific: Prefer AVX512 for large operations, AVX2 for tight loops
// Zen 4 has 256-bit SIMD execution units, but AVX512 can still help with parallelism

void readGraph(char *fname)
{
    ifstream infile(fname);
    int source;
    int end;
    infile >> v_num >> e_num;
    raw_graph.reserve(e_num * 2);
    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF)
            break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

void raw_graph_to_CSR()
{
    vector<int> temp_degree(v_num, 0);
    degree.resize(v_num, 0);
    
    for (int i = 0; i < (int)raw_graph.size() / 2; i++) {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        temp_degree[dst]++;
        degree[src]++;
    }
    
    csr_row_ptr.resize(v_num + 1);
    csr_row_ptr[0] = 0;
    for (int i = 0; i < v_num; i++) {
        csr_row_ptr[i + 1] = csr_row_ptr[i] + temp_degree[i];
    }
    
    vector<int> temp_ptr = csr_row_ptr;
    csr_col_idx.resize(e_num);
    
    for (int i = 0; i < (int)raw_graph.size() / 2; i++) {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        csr_col_idx[temp_ptr[dst]++] = src;
    }
}

void compute_CSR_values()
{
    csr_val.resize(e_num);
    
    vector<float> inv_sqrt_degree(v_num);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++) {
        inv_sqrt_degree[i] = 1.0f / sqrtf(degree[i]);
    }
    
    #pragma omp parallel for schedule(static, 16)
    for (int i = 0; i < v_num; i++) {
        float inv_sqrt_i = inv_sqrt_degree[i];
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        for (int j = row_start; j < row_end; j++) {
            int nbr = csr_col_idx[j];
            csr_val[j] = inv_sqrt_i * inv_sqrt_degree[nbr];
        }
    }
}

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
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num; i++) {
        dst[i] = 0.0f;
    }
}

// XW1: 1024x64 * 64x16 = 1024x16
// For Zen 4: use AVX512 with proper unrolling - 2x256 operations per cycle
void XW_out16_zen4(int in_dim, const float *__restrict__ in_X, 
                   float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * 16];
        
        // Use two AVX2 registers for better ILP on Zen 4
        __m256 y0_7 = _mm256_setzero_ps();
        __m256 y8_15 = _mm256_setzero_ps();
        
        // Unroll by 4 to match Zen 4's FMA throughput
        #pragma unroll 4
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

// XW2: 1024x16 * 16x8 = 1024x8
void XW_out8_zen4(int in_dim, const float *__restrict__ in_X, 
                  float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * 8];
        
        __m256 y_vec = _mm256_setzero_ps();
        
        #pragma unroll 8
        for (int k = 0; k < in_dim; k++) {
            float xv = x[k];
            const float* w = &W[k * 8];
            
            __m256 x_vec = _mm256_set1_ps(xv);
            __m256 w_vec = _mm256_load_ps(w);
            
            y_vec = _mm256_fmadd_ps(x_vec, w_vec, y_vec);
        }
        
        _mm256_store_ps(y, y_vec);
    }
}

// General XW
void XW_general_zen4(int in_dim, int out_dim, const float *__restrict__ in_X, 
                     float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 64)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * out_dim];
        
        int j = 0;
        for (; j + 8 <= out_dim; j += 8) {
            __m256 y_vec = _mm256_setzero_ps();
            #pragma unroll 8
            for (int k = 0; k < in_dim; k++) {
                __m256 w_vec = _mm256_load_ps(&W[k * out_dim + j]);
                y_vec = _mm256_fmadd_ps(_mm256_set1_ps(x[k]), w_vec, y_vec);
            }
            _mm256_store_ps(&y[j], y_vec);
        }
        for (; j < out_dim; j++) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < in_dim; k++) {
                sum += x[k] * W[k * out_dim + j];
            }
            y[j] = sum;
        }
    }
}

void XW_optimized_zen4(int in_dim, int out_dim, const float *__restrict__ in_X, 
                       float *__restrict__ out_X, const float *__restrict__ W)
{
    if (out_dim == 16) {
        XW_out16_zen4(in_dim, in_X, out_X, W);
    } else if (out_dim == 8) {
        XW_out8_zen4(in_dim, in_X, out_X, W);
    } else {
        XW_general_zen4(in_dim, out_dim, in_X, out_X, W);
    }
}

// AX for dim=16 - Sparse Matrix Dense Matrix multiplication
// A is sparse CSR, X is dense (v_num x dim)
void AX_dim16_zen4(const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(static, 16)
    for (int i = 0; i < v_num; i++) {
        float* out_row = &out_X[i * 16];
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        
        __m256 sum0_7 = _mm256_setzero_ps();
        __m256 sum8_15 = _mm256_setzero_ps();
        
        for (int j = row_start; j < row_end; j++) {
            int nbr = csr_col_idx[j];
            float val = csr_val[j];
            const float* in_row = &in_X[nbr * 16];
            
            __m256 v = _mm256_set1_ps(val);
            __m256 x0_7 = _mm256_load_ps(in_row);
            __m256 x8_15 = _mm256_load_ps(in_row + 8);
            
            sum0_7 = _mm256_fmadd_ps(v, x0_7, sum0_7);
            sum8_15 = _mm256_fmadd_ps(v, x8_15, sum8_15);
        }
        
        _mm256_store_ps(out_row, sum0_7);
        _mm256_store_ps(out_row + 8, sum8_15);
    }
}

// AX for dim=8
void AX_dim8_zen4(const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(static, 32)
    for (int i = 0; i < v_num; i++) {
        float* out_row = &out_X[i * 8];
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (int j = row_start; j < row_end; j++) {
            int nbr = csr_col_idx[j];
            float val = csr_val[j];
            const float* in_row = &in_X[nbr * 8];
            
            __m256 v = _mm256_set1_ps(val);
            __m256 x = _mm256_load_ps(in_row);
            
            sum_vec = _mm256_fmadd_ps(v, x, sum_vec);
        }
        
        _mm256_store_ps(out_row, sum_vec);
    }
}

void AX_general_zen4(int dim, const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < v_num; i++) {
        float* out_row = &out_X[i * dim];
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        
        int k = 0;
        for (; k + 8 <= dim; k += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int j = row_start; j < row_end; j++) {
                int nbr = csr_col_idx[j];
                float val = csr_val[j];
                const float* in_row = &in_X[nbr * dim + k];
                
                __m256 v = _mm256_set1_ps(val);
                __m256 x = _mm256_load_ps(in_row);
                sum = _mm256_fmadd_ps(v, x, sum);
            }
            _mm256_store_ps(&out_row[k], sum);
        }
        for (; k < dim; k++) {
            float sum = 0.0f;
            for (int j = row_start; j < row_end; j++) {
                int nbr = csr_col_idx[j];
                sum += csr_val[j] * in_X[nbr * dim + k];
            }
            out_row[k] = sum;
        }
    }
}

void AX_optimized_zen4(int dim, const float *__restrict__ in_X, float *__restrict__ out_X)
{
    if (dim == 16) {
        AX_dim16_zen4(in_X, out_X);
    } else if (dim == 8) {
        AX_dim8_zen4(in_X, out_X);
    } else {
        AX_general_zen4(dim, in_X, out_X);
    }
}

// Fast ReLU with AVX2 - better for Zen 4
void ReLU_zen4(int dim, float *X)
{
    const __m256 zero = _mm256_setzero_ps();
    int total = v_num * dim;
    
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i < total; i += 8) {
        __m256 x = _mm256_load_ps(&X[i]);
        x = _mm256_max_ps(x, zero);
        _mm256_store_ps(&X[i], x);
    }
}

// Fast vectorized exp for AVX2
inline __m256 exp256_ps(__m256 x) {
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666667f);
    const __m256 c4 = _mm256_set1_ps(0.04166667f);
    const __m256 c5 = _mm256_set1_ps(0.00833333f);
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504089f);
    
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT);
    __m256 r = _mm256_fnmadd_ps(n, ln2, x);
    
    __m256 t = c5;
    t = _mm256_fmadd_ps(t, r, c4);
    t = _mm256_fmadd_ps(t, r, c3);
    t = _mm256_fmadd_ps(t, r, c2);
    t = _mm256_fmadd_ps(t, r, c1);
    t = _mm256_fmadd_ps(t, r, c0);
    
    __m256 power = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
    
    return _mm256_mul_ps(t, power);
}

// LogSoftmax for dim=8 - fully vectorized
void LogSoftmax_dim8_zen4(float *X)
{
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i < v_num; i++) {
        float* row = &X[i * 8];
        
        __m256 v = _mm256_load_ps(row);
        
        // Find max using AVX2
        __m128 v_low = _mm256_castps256_ps128(v);
        __m128 v_high = _mm256_extractf128_ps(v, 1);
        __m128 v_max = _mm_max_ps(v_low, v_high);
        v_max = _mm_max_ps(v_max, _mm_movehl_ps(v_max, v_max));
        v_max = _mm_max_ps(v_max, _mm_shuffle_ps(v_max, v_max, 0x55));
        float max_val = _mm_cvtss_f32(v_max);
        
        // Subtract max from all elements
        __m256 max_vec = _mm256_set1_ps(max_val);
        __m256 v_sub = _mm256_sub_ps(v, max_vec);
        
        // Compute exp values
        float exp_vals[8];
        _mm256_store_ps(exp_vals, v_sub);
        
        float sum = 0.0f;
        sum += __builtin_expf(exp_vals[0]);
        sum += __builtin_expf(exp_vals[1]);
        sum += __builtin_expf(exp_vals[2]);
        sum += __builtin_expf(exp_vals[3]);
        sum += __builtin_expf(exp_vals[4]);
        sum += __builtin_expf(exp_vals[5]);
        sum += __builtin_expf(exp_vals[6]);
        sum += __builtin_expf(exp_vals[7]);
        
        float log_sum = __builtin_logf(sum);
        float sub_val = max_val + log_sum;
        
        __m256 sub_vec = _mm256_set1_ps(sub_val);
        v = _mm256_sub_ps(v, sub_vec);
        _mm256_store_ps(row, v);
    }
}

void LogSoftmax_general_zen4(int dim, float *X)
{
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i < v_num; i++) {
        float* row = &X[i * dim];
        
        float max_val = row[0];
        #pragma unroll
        for (int j = 1; j < dim; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < dim; j++) {
            sum += __builtin_expf(row[j] - max_val);
        }
        
        float log_sum = __builtin_logf(sum);
        float sub_val = max_val + log_sum;
        
        #pragma unroll
        for (int j = 0; j < dim; j++) {
            row[j] -= sub_val;
        }
    }
}

void LogSoftmax_optimized_zen4(int dim, float *X)
{
    if (dim == 8) {
        LogSoftmax_dim8_zen4(X);
    } else {
        LogSoftmax_general_zen4(dim, X);
    }
}

// MaxRowSum optimized for Zen 4
float MaxRowSum_zen4(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    
    #pragma omp parallel for reduction(max:max_sum) schedule(static, 64)
    for (int i = 0; i < v_num; i++) {
        const float* row = &X[i * dim];
        float sum = 0.0f;
        
        if (dim == 8) {
            __m256 sum_vec = _mm256_load_ps(row);
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            sum = _mm_cvtss_f32(sum_low) + _mm_cvtss_f32(sum_high);
        } else if (dim == 16) {
            __m256 sum0_7 = _mm256_load_ps(row);
            __m256 sum8_15 = _mm256_load_ps(row + 8);
            sum0_7 = _mm256_add_ps(sum0_7, sum8_15);
            sum0_7 = _mm256_hadd_ps(sum0_7, sum0_7);
            sum0_7 = _mm256_hadd_ps(sum0_7, sum0_7);
            __m128 sum_low = _mm256_castps256_ps128(sum0_7);
            __m128 sum_high = _mm256_extractf128_ps(sum0_7, 1);
            sum = _mm_cvtss_f32(sum_low) + _mm_cvtss_f32(sum_high);
        } else {
            int j = 0;
            __m256 sum256 = _mm256_setzero_ps();
            for (; j + 8 <= dim; j += 8) {
                sum256 = _mm256_add_ps(sum256, _mm256_load_ps(&row[j]));
            }
            sum256 = _mm256_hadd_ps(sum256, sum256);
            sum256 = _mm256_hadd_ps(sum256, sum256);
            __m128 sum_low = _mm256_castps256_ps128(sum256);
            __m128 sum_high = _mm256_extractf128_ps(sum256, 1);
            sum = _mm_cvtss_f32(sum_low) + _mm_cvtss_f32(sum_high);
            for (; j < dim; j++) {
                sum += row[j];
            }
        }
        
        if (sum > max_sum) {
            max_sum = sum;
        }
    }
    return max_sum;
}

void freeFloats()
{
    free(X0);
    free(W1);
    free(W2);
    free(X1);
    free(X2);
    free(X1_inter);
    free(X2_inter);
}

int main(int argc, char **argv)
{
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);
    F2 = atoi(argv[3]);

    readGraph(argv[4]);
    readFloat(argv[5], X0, v_num * F0);
    readFloat(argv[6], W1, F0 * F1);
    readFloat(argv[7], W2, F1 * F2);

    initFloat(X1, v_num * F1);
    initFloat(X1_inter, v_num * F1);
    initFloat(X2, v_num * F2);
    initFloat(X2_inter, v_num * F2);

    omp_set_num_threads(16);
    omp_set_dynamic(0);
    // Don't set global schedule - let each pragma define its own
    
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    TimePoint start = chrono::steady_clock::now();

    TimePoint prepross_start = chrono::steady_clock::now();
    raw_graph_to_CSR();
    TimePoint prepross_end = chrono::steady_clock::now();
    chrono::duration<double> prepross_ = prepross_end - prepross_start;
    double prepross_time = prepross_.count() * 1e3;
    printf("prepross_time: %.8lf\n", prepross_time);

    TimePoint edgeNorm_start = chrono::steady_clock::now();
    compute_CSR_values();
    TimePoint edgeNorm_end = chrono::steady_clock::now();
    chrono::duration<double> edgeNorm_ = edgeNorm_end - edgeNorm_start;
    double edgeNorm_time = edgeNorm_.count() * 1e3;
    printf("edgeNorm_time: %.8lf\n", edgeNorm_time);

    TimePoint XW1_start = chrono::steady_clock::now();
    XW_optimized_zen4(F0, F1, X0, X1_inter, W1);
    TimePoint XW1_end = chrono::steady_clock::now();
    chrono::duration<double> XW1_ = XW1_end - XW1_start;
    double XW1_time = XW1_.count() * 1e3;
    printf("XW1_time: %.8lf\n", XW1_time);

    TimePoint AX1_start = chrono::steady_clock::now();
    AX_optimized_zen4(F1, X1_inter, X1);
    TimePoint AX1_end = chrono::steady_clock::now();
    chrono::duration<double> AX1_ = AX1_end - AX1_start;
    double AX1_time = AX1_.count() * 1e3;
    printf("AX1_time: %.8lf\n", AX1_time);

    TimePoint ReLU_start = chrono::steady_clock::now();
    ReLU_zen4(F1, X1);
    TimePoint ReLU_end = chrono::steady_clock::now();
    chrono::duration<double> ReLU_ = ReLU_end - ReLU_start;
    double ReLU_time = ReLU_.count() * 1e3;
    printf("ReLU_time: %.8lf\n", ReLU_time);

    TimePoint XW2_start = chrono::steady_clock::now();
    XW_optimized_zen4(F1, F2, X1, X2_inter, W2);
    TimePoint XW2_end = chrono::steady_clock::now();
    chrono::duration<double> XW2_ = XW2_end - XW2_start;
    double XW2_time = XW2_.count() * 1e3;
    printf("XW2_time: %.8lf\n", XW2_time);

    TimePoint AX2_start = chrono::steady_clock::now();
    AX_optimized_zen4(F2, X2_inter, X2);
    TimePoint AX2_end = chrono::steady_clock::now();
    chrono::duration<double> AX2_ = AX2_end - AX2_start;
    double AX2_time = AX2_.count() * 1e3;
    printf("AX2_time: %.8lf\n", AX2_time);

    TimePoint LogSoftmax_start = chrono::steady_clock::now();
    LogSoftmax_optimized_zen4(F2, X2);
    TimePoint LogSoftmax_end = chrono::steady_clock::now();
    chrono::duration<double> LogSoftmax_ = LogSoftmax_end - LogSoftmax_start;
    double LogSoftmax_time = LogSoftmax_.count() * 1e3;
    printf("LogSoftmax_time: %.8lf\n", LogSoftmax_time);

    TimePoint max_sum_start = chrono::steady_clock::now();
    float max_sum = MaxRowSum_zen4(X2, F2);
    TimePoint max_sum_end = chrono::steady_clock::now();
    chrono::duration<double> max_sum_ = max_sum_end - max_sum_start;
    double max_sum_time = max_sum_.count() * 1e3;
    printf("max_sum_time: %.8lf\n", max_sum_time);

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    printf("%.8f\n", max_sum);
    printf("total time: %.8lf\n\n", l_timeMs);

    freeFloats();
}