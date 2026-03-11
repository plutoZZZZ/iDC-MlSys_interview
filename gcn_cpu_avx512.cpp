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

// Threshold for conditional OpenMP - tuned for 1024 vertices
const int OMP_THRESHOLD_SMALL = 2048;
const int OMP_THRESHOLD_MEDIUM = 8192;

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
    #pragma omp parallel for if(v_num > OMP_THRESHOLD_SMALL)
    for (int i = 0; i < v_num; i++) {
        inv_sqrt_degree[i] = 1.0f / sqrtf(degree[i]);
    }
    
    #pragma omp parallel for schedule(static, 16) if(v_num > OMP_THRESHOLD_SMALL)
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
    #pragma omp parallel for if(num > OMP_THRESHOLD_MEDIUM)
    for (int i = 0; i < num; i++) {
        dst[i] = 0.0f;
    }
}

// AVX512 optimized XW for out_dim = 16
// Using 16x1 register blocking with AVX512
void XW_out16_avx512(int in_dim, const float *__restrict__ in_X, 
                      float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 4) if(v_num > OMP_THRESHOLD_SMALL)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * 16];
        
        __m512 y_vec = _mm512_setzero_ps();
        
        #pragma unroll 8
        for (int k = 0; k < in_dim; k++) {
            float xv = x[k];
            const float* w = &W[k * 16];
            
            __m512 x_vec = _mm512_set1_ps(xv);
            __m512 w_vec = _mm512_load_ps(w);
            
            y_vec = _mm512_fmadd_ps(x_vec, w_vec, y_vec);
        }
        
        _mm512_store_ps(y, y_vec);
    }
}

// AVX512 optimized XW for out_dim = 8
void XW_out8_avx512(int in_dim, const float *__restrict__ in_X, 
                     float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 8) if(v_num > OMP_THRESHOLD_SMALL)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * 8];
        
        __m256 y_vec = _mm256_setzero_ps();
        
        #pragma unroll
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

// AVX512 optimized XW for out_dim = 12
void XW_out12_avx512(int in_dim, const float *__restrict__ in_X, 
                      float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 8) if(v_num > OMP_THRESHOLD_SMALL)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * 12];
        
        __m512 y_vec = _mm512_setzero_ps();
        float y_last[4] __attribute__((aligned(16))) = {0, 0, 0, 0};
        
        #pragma unroll
        for (int k = 0; k < in_dim; k++) {
            float xv = x[k];
            const float* w = &W[k * 12];
            
            __m512 x_vec = _mm512_set1_ps(xv);
            __m512 w_vec = _mm512_loadu_ps(w);
            
            y_vec = _mm512_fmadd_ps(x_vec, w_vec, y_vec);
            
            y_last[0] += xv * w[8];
            y_last[1] += xv * w[9];
            y_last[2] += xv * w[10];
            y_last[3] += xv * w[11];
        }
        
        _mm512_storeu_ps(y, y_vec);
        y[8] = y_last[0];
        y[9] = y_last[1];
        y[10] = y_last[2];
        y[11] = y_last[3];
    }
}

// General XW with AVX512
void XW_general_avx512(int in_dim, int out_dim, const float *__restrict__ in_X, 
                        float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 64) if(v_num > OMP_THRESHOLD_SMALL)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * out_dim];
        
        int j = 0;
        for (; j + 16 <= out_dim; j += 16) {
            __m512 y_vec = _mm512_setzero_ps();
            #pragma unroll 8
            for (int k = 0; k < in_dim; k++) {
                __m512 w_vec = _mm512_load_ps(&W[k * out_dim + j]);
                y_vec = _mm512_fmadd_ps(_mm512_set1_ps(x[k]), w_vec, y_vec);
            }
            _mm512_store_ps(&y[j], y_vec);
        }
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

void XW_optimized_avx512(int in_dim, int out_dim, const float *__restrict__ in_X, 
                          float *__restrict__ out_X, const float *__restrict__ W)
{
    if (out_dim == 16) {
        XW_out16_avx512(in_dim, in_X, out_X, W);
    } else if (out_dim == 8) {
        XW_out8_avx512(in_dim, in_X, out_X, W);
    } else if (out_dim == 12) {
        XW_out12_avx512(in_dim, in_X, out_X, W);
    } else {
        XW_general_avx512(in_dim, out_dim, in_X, out_X, W);
    }
}

// AVX512 optimized AX for dim = 16
void AX_dim16_avx512(const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(static, 16) if(v_num > OMP_THRESHOLD_SMALL)
    for (int i = 0; i < v_num; i++) {
        float* out_row = &out_X[i * 16];
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        
        __m512 sum_vec = _mm512_setzero_ps();
        
        for (int j = row_start; j < row_end; j++) {
            int nbr = csr_col_idx[j];
            float val = csr_val[j];
            const float* in_row = &in_X[nbr * 16];
            
            __m512 v = _mm512_set1_ps(val);
            __m512 x = _mm512_load_ps(in_row);
            
            sum_vec = _mm512_fmadd_ps(v, x, sum_vec);
        }
        
        _mm512_store_ps(out_row, sum_vec);
    }
}

// AVX512 optimized AX for dim = 8
void AX_dim8_avx512(const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(static, 32) if(v_num > OMP_THRESHOLD_SMALL)
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

// General AX with AVX512
void AX_general_avx512(int dim, const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(dynamic, 16) if(v_num > OMP_THRESHOLD_SMALL)
    for (int i = 0; i < v_num; i++) {
        float* out_row = &out_X[i * dim];
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        
        int k = 0;
        for (; k + 16 <= dim; k += 16) {
            __m512 sum = _mm512_setzero_ps();
            for (int j = row_start; j < row_end; j++) {
                int nbr = csr_col_idx[j];
                float val = csr_val[j];
                const float* in_row = &in_X[nbr * dim + k];
                
                __m512 v = _mm512_set1_ps(val);
                __m512 x = _mm512_load_ps(in_row);
                sum = _mm512_fmadd_ps(v, x, sum);
            }
            _mm512_store_ps(&out_row[k], sum);
        }
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

void AX_optimized_avx512(int dim, const float *__restrict__ in_X, float *__restrict__ out_X)
{
    if (dim == 16) {
        AX_dim16_avx512(in_X, out_X);
    } else if (dim == 8) {
        AX_dim8_avx512(in_X, out_X);
    } else {
        AX_general_avx512(dim, in_X, out_X);
    }
}

// AVX512 ReLU - single threaded for small sizes
void ReLU_avx512(int dim, float *X)
{
    const __m512 zero = _mm512_setzero_ps();
    int total = v_num * dim;
    
    // Use single thread for small workloads to avoid OpenMP overhead
    if (total <= OMP_THRESHOLD_MEDIUM) {
        int i = 0;
        for (; i + 16 <= total; i += 16) {
            __m512 x = _mm512_load_ps(&X[i]);
            x = _mm512_max_ps(x, zero);
            _mm512_store_ps(&X[i], x);
        }
        for (; i < total; i++) {
            if (X[i] < 0) X[i] = 0;
        }
    } else {
        #pragma omp parallel for schedule(static, 256)
        for (int i = 0; i < total; i += 16) {
            if (i + 16 <= total) {
                __m512 x = _mm512_load_ps(&X[i]);
                x = _mm512_max_ps(x, zero);
                _mm512_store_ps(&X[i], x);
            } else {
                for (int j = i; j < total; j++) {
                    if (X[j] < 0) X[j] = 0;
                }
            }
        }
    }
}

// Fast exp approximation using AVX512
inline __m512 exp512_ps(__m512 x) {
    const __m512 c0 = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(1.0f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c3 = _mm512_set1_ps(0.16666667f);
    const __m512 c4 = _mm512_set1_ps(0.04166667f);
    const __m512 c5 = _mm512_set1_ps(0.00833333f);
    const __m512 c6 = _mm512_set1_ps(0.00138889f);
    const __m512 c7 = _mm512_set1_ps(0.00019841f);
    
    __m512 t1 = _mm512_fmadd_ps(x, c7, c6);
    __m512 t2 = _mm512_fmadd_ps(t1, x, c5);
    __m512 t3 = _mm512_fmadd_ps(t2, x, c4);
    __m512 t4 = _mm512_fmadd_ps(t3, x, c3);
    __m512 t5 = _mm512_fmadd_ps(t4, x, c2);
    __m512 t6 = _mm512_fmadd_ps(t5, x, c1);
    __m512 t7 = _mm512_fmadd_ps(t6, x, c0);
    
    return t7;
}

// Optimized LogSoftmax for dim=8 with AVX512 - single threaded
void LogSoftmax_dim8_avx512(float *X)
{
    // Single threaded for small v_num to avoid OpenMP overhead
    for (int i = 0; i < v_num; i++) {
        float* row = &X[i * 8];
        
        __m256 v = _mm256_load_ps(row);
        
        // Find max value using AVX2
        __m256 max_v = _mm256_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
        max_v = _mm256_max_ps(max_v, v);
        __m128 max128 = _mm_max_ps(_mm256_castps256_ps128(max_v), 
                                   _mm256_extractf128_ps(max_v, 1));
        max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 0x55));
        float max_val = _mm_cvtss_f32(max128);
        
        // Subtract max from all elements
        __m256 max_vec = _mm256_set1_ps(max_val);
        __m256 v_sub = _mm256_sub_ps(v, max_vec);
        
        // Compute exp using builtin for accuracy
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

// General LogSoftmax with AVX512
void LogSoftmax_general_avx512(int dim, float *X)
{
    // Single threaded for small v_num
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

void LogSoftmax_optimized_avx512(int dim, float *X)
{
    if (dim == 8) {
        LogSoftmax_dim8_avx512(X);
    } else {
        LogSoftmax_general_avx512(dim, X);
    }
}

// AVX512 optimized MaxRowSum
float MaxRowSum_avx512(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    
    // Single threaded for small v_num to avoid OpenMP overhead
    if (v_num <= OMP_THRESHOLD_SMALL) {
        for (int i = 0; i < v_num; i++) {
            const float* row = &X[i * dim];
            float sum = 0.0f;
            
            if (dim == 8) {
                __m256 sum_vec = _mm256_load_ps(row);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum = _mm256_cvtss_f32(sum_vec) + _mm_cvtss_f32(_mm256_extractf128_ps(sum_vec, 1));
            } else if (dim == 16) {
                __m512 sum_vec = _mm512_load_ps(row);
                sum = _mm512_reduce_add_ps(sum_vec);
            } else {
                int j = 0;
                __m512 sum512 = _mm512_setzero_ps();
                for (; j + 16 <= dim; j += 16) {
                    sum512 = _mm512_add_ps(sum512, _mm512_load_ps(&row[j]));
                }
                sum = _mm512_reduce_add_ps(sum512);
                for (; j < dim; j++) {
                    sum += row[j];
                }
            }
            
            if (sum > max_sum) {
                max_sum = sum;
            }
        }
    } else {
        #pragma omp parallel for reduction(max:max_sum) schedule(static, 64)
        for (int i = 0; i < v_num; i++) {
            const float* row = &X[i * dim];
            float sum = 0.0f;
            
            if (dim == 8) {
                __m256 sum_vec = _mm256_load_ps(row);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
                sum = _mm256_cvtss_f32(sum_vec);
            } else if (dim == 16) {
                __m512 sum_vec = _mm512_load_ps(row);
                sum = _mm512_reduce_add_ps(sum_vec);
            } else {
                int j = 0;
                __m512 sum512 = _mm512_setzero_ps();
                for (; j + 16 <= dim; j += 16) {
                    sum512 = _mm512_add_ps(sum512, _mm512_load_ps(&row[j]));
                }
                sum = _mm512_reduce_add_ps(sum512);
                for (; j < dim; j++) {
                    sum += row[j];
                }
            }
            
            if (sum > max_sum) {
                max_sum = sum;
            }
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
    XW_optimized_avx512(F0, F1, X0, X1_inter, W1);
    TimePoint XW1_end = chrono::steady_clock::now();
    chrono::duration<double> XW1_ = XW1_end - XW1_start;
    double XW1_time = XW1_.count() * 1e3;
    printf("XW1_time: %.8lf\n", XW1_time);

    TimePoint AX1_start = chrono::steady_clock::now();
    AX_optimized_avx512(F1, X1_inter, X1);
    TimePoint AX1_end = chrono::steady_clock::now();
    chrono::duration<double> AX1_ = AX1_end - AX1_start;
    double AX1_time = AX1_.count() * 1e3;
    printf("AX1_time: %.8lf\n", AX1_time);

    TimePoint ReLU_start = chrono::steady_clock::now();
    ReLU_avx512(F1, X1);
    TimePoint ReLU_end = chrono::steady_clock::now();
    chrono::duration<double> ReLU_ = ReLU_end - ReLU_start;
    double ReLU_time = ReLU_.count() * 1e3;
    printf("ReLU_time: %.8lf\n", ReLU_time);

    TimePoint XW2_start = chrono::steady_clock::now();
    XW_optimized_avx512(F1, F2, X1, X2_inter, W2);
    TimePoint XW2_end = chrono::steady_clock::now();
    chrono::duration<double> XW2_ = XW2_end - XW2_start;
    double XW2_time = XW2_.count() * 1e3;
    printf("XW2_time: %.8lf\n", XW2_time);

    TimePoint AX2_start = chrono::steady_clock::now();
    AX_optimized_avx512(F2, X2_inter, X2);
    TimePoint AX2_end = chrono::steady_clock::now();
    chrono::duration<double> AX2_ = AX2_end - AX2_start;
    double AX2_time = AX2_.count() * 1e3;
    printf("AX2_time: %.8lf\n", AX2_time);

    TimePoint LogSoftmax_start = chrono::steady_clock::now();
    LogSoftmax_optimized_avx512(F2, X2);
    TimePoint LogSoftmax_end = chrono::steady_clock::now();
    chrono::duration<double> LogSoftmax_ = LogSoftmax_end - LogSoftmax_start;
    double LogSoftmax_time = LogSoftmax_.count() * 1e3;
    printf("LogSoftmax_time: %.8lf\n", LogSoftmax_time);

    TimePoint max_sum_start = chrono::steady_clock::now();
    float max_sum = MaxRowSum_avx512(X2, F2);
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
