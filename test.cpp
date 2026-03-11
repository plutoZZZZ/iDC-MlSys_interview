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
    
    for (int i = 0; i < v_num; i++) {
        inv_sqrt_degree[i] = 1.0f / sqrtf(degree[i]);
    }
    
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
    for (int i = 0; i < num; i++) {
        dst[i] = 0.0f;
    }
}

// XW: 1024 x 64 @ 64 x 16 = 1024 x 16
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

// XW: 1024 x 16 @ 16 x 8 = 1024 x 8
void XW_16_8(const float *__restrict__ in_X, float *__restrict__ out_X, const float *__restrict__ W)
{
    #pragma omp parallel for schedule(static, 128)
    for (int i = 0; i < v_num; i++) {
        const float* x = &in_X[i * 16];
        float* y = &out_X[i * 8];
        
        __m256 y_vec = _mm256_setzero_ps();
        
        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
            float xv = x[k];
            const float* w = &W[k * 8];
            
            __m256 x_vec = _mm256_set1_ps(xv);
            __m256 w_vec = _mm256_load_ps(w);
            
            y_vec = _mm256_fmadd_ps(x_vec, w_vec, y_vec);
        }
        
        _mm256_store_ps(y, y_vec);
    }
}

// AX: 1024 x 16 with 4096 edges
void AX_16(const float *__restrict__ in_X, float *__restrict__ out_X)
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

// AX: 1024 x 8 with 4096 edges
void AX_8(const float *__restrict__ in_X, float *__restrict__ out_X)
{
    #pragma omp parallel for schedule(static, 128)
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

// ReLU for 1024 x 16
void ReLU_16(float *X)
{
    const __m256 zero = _mm256_setzero_ps();
    
    #pragma omp parallel for schedule(static, 256)
    for (int i = 0; i < v_num * 16; i += 8) {
        __m256 x = _mm256_load_ps(&X[i]);
        x = _mm256_max_ps(x, zero);
        _mm256_store_ps(&X[i], x);
    }
}

// Fast approximate exp2
inline __m256 mm256_exp2_ps(__m256 x) {
    const __m256 c1 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c2 = _mm256_set1_ps(1.0f);
    const __m256 c3 = _mm256_set1_ps(0.5f);
    const __m256 c4 = _mm256_set1_ps(0.166666666666f);
    const __m256 c5 = _mm256_set1_ps(0.0416666666667f);
    
    __m256 t = _mm256_fmadd_ps(x, c1, _mm256_set1_ps(12582912.0f));
    __m256 ti = _mm256_floor_ps(t);
    __m256 tf = _mm256_sub_ps(t, ti);
    
    __m256 r = _mm256_fmadd_ps(tf, c5, c4);
    r = _mm256_fmadd_ps(r, tf, c3);
    r = _mm256_fmadd_ps(r, tf, c2);
    r = _mm256_fmadd_ps(r, tf, c2);
    
    __m256i ei = _mm256_cvtps_epi32(ti);
    ei = _mm256_slli_epi32(ei, 23);
    __m256 pow2 = _mm256_castsi256_ps(ei);
    
    return _mm256_mul_ps(r, pow2);
}

inline __m256 mm256_exp_ps(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    return mm256_exp2_ps(_mm256_div_ps(x, ln2));
}

// Fast log2 approximation
inline __m256 mm256_log2_ps(__m256 x) {
    __m256i i = _mm256_castps_si256(x);
    __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(i, 23), _mm256_set1_epi32(127)));
    __m256 m = _mm256_castsi256_ps(_mm256_or_epi32(_mm256_and_si256(i, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000)));
    
    const __m256 c1 = _mm256_set1_ps(-0.64946f);
    const __m256 c2 = _mm256_set1_ps(1.4625f);
    const __m256 c3 = _mm256_set1_ps(-1.2136f);
    const __m256 c4 = _mm256_set1_ps(0.7108f);
    const __m256 c5 = _mm256_set1_ps(-0.2044f);
    
    __m256 r = c1;
    r = _mm256_fmadd_ps(r, m, c2);
    r = _mm256_fmadd_ps(r, m, c3);
    r = _mm256_fmadd_ps(r, m, c4);
    r = _mm256_fmadd_ps(r, m, c5);
    
    return _mm256_add_ps(e, _mm256_mul_ps(m, r));
}

inline __m256 mm256_log_ps(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    return _mm256_mul_ps(mm256_log2_ps(x), ln2);
}

// LogSoftmax for 1024 x 8 - NO OpenMP overhead, compute directly
void LogSoftmax_8(float *X) {
    const float ln2_hi = 0.69314718056f;
    
    for (int i = 0; i < v_num; i++) {
        float* row = &X[i * 8];
        
        // Find max - use scalar for 8 elements
        float max_val = row[0];
        max_val = max_val > row[1] ? max_val : row[1];
        max_val = max_val > row[2] ? max_val : row[2];
        max_val = max_val > row[3] ? max_val : row[3];
        max_val = max_val > row[4] ? max_val : row[4];
        max_val = max_val > row[5] ? max_val : row[5];
        max_val = max_val > row[6] ? max_val : row[6];
        max_val = max_val > row[7] ? max_val : row[7];
        
        // Load and subtract max
        __m256 x_vec = _mm256_load_ps(row);
        __m256 max_vec = _mm256_set1_ps(max_val);
        __m256 x_sub = _mm256_sub_ps(x_vec, max_vec);
        
        // Compute exp(x - max) using builtin exp
        float vals[8];
        _mm256_store_ps(vals, x_sub);
        
        float sum = 0.0f;
        float exp_vals[8];
        for (int j = 0; j < 8; j++) {
            exp_vals[j] = __builtin_expf(vals[j]);
            sum += exp_vals[j];
        }
        
        float log_sum = __builtin_logf(sum);
        float sub_val = max_val + log_sum;
        __m256 sub_vec = _mm256_set1_ps(sub_val);
        
        __m256 result = _mm256_sub_ps(x_vec, sub_vec);
        _mm256_store_ps(row, result);
    }
}

// MaxRowSum for 8
float MaxRowSum_8(float *X) {
    float max_sum = -__FLT_MAX__;
    
    #pragma omp parallel for reduction(max:max_sum) schedule(static, 256)
    for (int i = 0; i < v_num; i++) {
        const float* row = &X[i * 8];
        float sum = row[0] + row[1] + row[2] + row[3] + row[4] + row[5] + row[6] + row[7];
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
    for (int i = 0; i < 100000; i++) {
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

    // AMD Zen 4 specific: 12 physical cores
    omp_set_num_threads(12);
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);
    
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
    XW_64_16(X0, X1_inter, W1);
    TimePoint XW1_end = chrono::steady_clock::now();
    chrono::duration<double> XW1_ = XW1_end - XW1_start;
    double XW1_time = XW1_.count() * 1e3;
    printf("XW1_time: %.8lf\n", XW1_time);

    TimePoint AX1_start = chrono::steady_clock::now();
    AX_16(X1_inter, X1);
    TimePoint AX1_end = chrono::steady_clock::now();
    chrono::duration<double> AX1_ = AX1_end - AX1_start;
    double AX1_time = AX1_.count() * 1e3;
    printf("AX1_time: %.8lf\n", AX1_time);

    TimePoint ReLU_start = chrono::steady_clock::now();
    ReLU_16(X1);
    TimePoint ReLU_end = chrono::steady_clock::now();
    chrono::duration<double> ReLU_ = ReLU_end - ReLU_start;
    double ReLU_time = ReLU_.count() * 1e3;
    printf("ReLU_time: %.8lf\n", ReLU_time);

    TimePoint XW2_start = chrono::steady_clock::now();
    XW_16_8(X1, X2_inter, W2);
    TimePoint XW2_end = chrono::steady_clock::now();
    chrono::duration<double> XW2_ = XW2_end - XW2_start;
    double XW2_time = XW2_.count() * 1e3;
    printf("XW2_time: %.8lf\n", XW2_time);

    TimePoint AX2_start = chrono::steady_clock::now();
    AX_8(X2_inter, X2);
    TimePoint AX2_end = chrono::steady_clock::now();
    chrono::duration<double> AX2_ = AX2_end - AX2_start;
    double AX2_time = AX2_.count() * 1e3;
    printf("AX2_time: %.8lf\n", AX2_time);

    TimePoint LogSoftmax_start = chrono::steady_clock::now();
    LogSoftmax_8(X2);
    TimePoint LogSoftmax_end = chrono::steady_clock::now();
    chrono::duration<double> LogSoftmax_ = LogSoftmax_end - LogSoftmax_start;
    double LogSoftmax_time = LogSoftmax_.count() * 1e3;
    printf("LogSoftmax_time: %.8lf\n", LogSoftmax_time);

    TimePoint max_sum_start = chrono::steady_clock::now();
    float max_sum = MaxRowSum_8(X2);
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
