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
#include <pthread.h>

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
    
    #pragma omp parallel for schedule(static)
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

void XW_out8(int in_dim, const float *__restrict__ in_X, 
             float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * in_dim];
        float* y = &out_X[i * 8];
        
        __m256 y_vec = _mm256_setzero_ps();
        
        #pragma unroll 16
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

void XW_optimized(int in_dim, int out_dim, const float *__restrict__ in_X, 
                  float *__restrict__ out_X, const float *__restrict__ W)
{
    if (out_dim == 16) {
        XW_out16(in_dim, in_X, out_X, W);
    } else if (out_dim == 8) {
        XW_out8(in_dim, in_X, out_X, W);
    } else {
        #pragma omp parallel for schedule(static, 4)
        for (int i = 0; i < v_num; i++) {
            const float* x = &in_X[i * in_dim];
            float* y = &out_X[i * out_dim];
            memset(y, 0, out_dim * sizeof(float));
            
            for (int k = 0; k < in_dim; k++) {
                float xv = x[k];
                const float* w = &W[k * out_dim];
                int j = 0;
                for (; j + 8 <= out_dim; j += 8) {
                    __m256 yv = _mm256_load_ps(&y[j]);
                    __m256 wv = _mm256_load_ps(&w[j]);
                    yv = _mm256_fmadd_ps(_mm256_set1_ps(xv), wv, yv);
                    _mm256_store_ps(&y[j], yv);
                }
                for (; j < out_dim; j++) {
                    y[j] += xv * w[j];
                }
            }
        }
    }
}

void AX_dim16(const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(static, 4)
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

void AX_dim8(const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(static, 8)
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

void AX_optimized(int dim, const float *__restrict__ in_X, float *__restrict__ out_X)
{
    if (dim == 16) {
        AX_dim16(in_X, out_X);
    } else if (dim == 8) {
        AX_dim8(in_X, out_X);
    } else {
        #pragma omp parallel for schedule(static, 4)
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
}

void ReLU_optimized(int dim, float *X)
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

void LogSoftmax_optimized(int dim, float *X)
{
    #pragma omp parallel for schedule(static, 64)
    for (int i = 0; i < v_num; i++) {
        float* row = &X[i * dim];
        
        float max_val = row[0];
        for (int j = 1; j < dim; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            sum += __builtin_expf(row[j] - max_val);
        }
        
        float log_sum = __builtin_logf(sum);
        float sub_val = max_val + log_sum;
        
        for (int j = 0; j < dim; j++) {
            row[j] -= sub_val;
        }
    }
}

float MaxRowSum(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    
    #pragma omp parallel for reduction(max:max_sum) schedule(static, 64)
    for (int i = 0; i < v_num; i++) {
        const float* row = &X[i * dim];
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) {
            sum += row[j];
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

void warmup()
{
    volatile float sum = 0.0f;
    for (int i = 0; i < 1000000; i++) {
        sum += sinf((float)i);
    }
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

    // AMD Zen 4 specific: 12 physical cores, use only physical cores
    omp_set_num_threads(12);
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);
    
    // Set thread affinity to physical cores
    putenv((char*)"OMP_PROC_BIND=close");
    putenv((char*)"OMP_PLACES=cores");
    putenv((char*)"OMP_WAIT_POLICY=active");
    
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // Warmup
    warmup();

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
    XW_optimized(F0, F1, X0, X1_inter, W1);
    TimePoint XW1_end = chrono::steady_clock::now();
    chrono::duration<double> XW1_ = XW1_end - XW1_start;
    double XW1_time = XW1_.count() * 1e3;
    printf("XW1_time: %.8lf\n", XW1_time);

    TimePoint AX1_start = chrono::steady_clock::now();
    AX_optimized(F1, X1_inter, X1);
    TimePoint AX1_end = chrono::steady_clock::now();
    chrono::duration<double> AX1_ = AX1_end - AX1_start;
    double AX1_time = AX1_.count() * 1e3;
    printf("AX1_time: %.8lf\n", AX1_time);

    TimePoint ReLU_start = chrono::steady_clock::now();
    ReLU_optimized(F1, X1);
    TimePoint ReLU_end = chrono::steady_clock::now();
    chrono::duration<double> ReLU_ = ReLU_end - ReLU_start;
    double ReLU_time = ReLU_.count() * 1e3;
    printf("ReLU_time: %.8lf\n", ReLU_time);

    TimePoint XW2_start = chrono::steady_clock::now();
    XW_optimized(F1, F2, X1, X2_inter, W2);
    TimePoint XW2_end = chrono::steady_clock::now();
    chrono::duration<double> XW2_ = XW2_end - XW2_start;
    double XW2_time = XW2_.count() * 1e3;
    printf("XW2_time: %.8lf\n", XW2_time);

    TimePoint AX2_start = chrono::steady_clock::now();
    AX_optimized(F2, X2_inter, X2);
    TimePoint AX2_end = chrono::steady_clock::now();
    chrono::duration<double> AX2_ = AX2_end - AX2_start;
    double AX2_time = AX2_.count() * 1e3;
    printf("AX2_time: %.8lf\n", AX2_time);

    TimePoint LogSoftmax_start = chrono::steady_clock::now();
    LogSoftmax_optimized(F2, X2);
    TimePoint LogSoftmax_end = chrono::steady_clock::now();
    chrono::duration<double> LogSoftmax_ = LogSoftmax_end - LogSoftmax_start;
    double LogSoftmax_time = LogSoftmax_.count() * 1e3;
    printf("LogSoftmax_time: %.8lf\n", LogSoftmax_time);

    TimePoint max_sum_start = chrono::steady_clock::now();
    float max_sum = MaxRowSum(X2, F2);
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
