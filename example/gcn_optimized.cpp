#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<int> raw_graph;
vector<int> degree;

// CSR format for sparse matrix A
vector<int> csr_row_ptr;
vector<int> csr_col_idx;
vector<float> csr_val;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(char *fname)
{
    ifstream infile(fname);
    int source, end;
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
    degree.resize(v_num, 0);
    vector<vector<int>> temp_edges(v_num);
    
    // First pass: count degrees
    for (int i = 0; i < raw_graph.size() / 2; i++)
    {
        int src = raw_graph[2 * i];
        degree[src]++;
    }
    
    // Second pass: build adjacency list
    for (int i = 0; i < raw_graph.size() / 2; i++)
    {
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        temp_edges[dst].push_back(src);
    }
    
    csr_row_ptr.resize(v_num + 1, 0);
    for (int i = 0; i < v_num; i++)
    {
        csr_row_ptr[i + 1] = csr_row_ptr[i] + temp_edges[i].size();
    }
    
    csr_col_idx.resize(e_num);
    csr_val.resize(e_num);
    
    int pos = 0;
    for (int i = 0; i < v_num; i++)
    {
        for (int nbr : temp_edges[i])
        {
            csr_col_idx[pos] = nbr;
            pos++;
        }
    }
}

// Optimized edgeNormalization - remove OpenMP overhead for small v_num
void edgeNormalization()
{
    for (int i = 0; i < v_num; i++)
    {
        float deg_i = sqrt(degree[i]);
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        for (int j = row_start; j < row_end; j++)
        {
            int nbr = csr_col_idx[j];
            csr_val[j] = 1.0f / (deg_i * sqrt(degree[nbr]));
        }
    }
}

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)_mm_malloc(num * sizeof(float), 32);
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)_mm_malloc(num * sizeof(float), 32);
    memset(dst, 0, num * sizeof(float));
}

// Optimized matrix multiplication with SIMD and OpenMP
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
    #pragma omp parallel for
    for (int i = 0; i < v_num; i++)
    {
        for (int k = 0; k < in_dim; k++)
        {
            float x_val = in_X[i * in_dim + k];
            const float *w_row = &W[k * out_dim];
            float *out_row = &out_X[i * out_dim];
            
            int j;
            for (j = 0; j + 8 <= out_dim; j += 8)
            {
                __m256 x = _mm256_set1_ps(x_val);
                __m256 w = _mm256_load_ps(&w_row[j]);
                __m256 o = _mm256_load_ps(&out_row[j]);
                __m256 res = _mm256_fmadd_ps(x, w, o);
                _mm256_store_ps(&out_row[j], res);
            }
            for (; j < out_dim; j++)
            {
                out_row[j] += x_val * w_row[j];
            }
        }
    }
}

// Optimized sparse matrix multiplication using CSR with OpenMP
void AX(int dim, float *in_X, float *out_X)
{
    #pragma omp parallel for
    for (int i = 0; i < v_num; i++)
    {
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        float *out_row = &out_X[i * dim];
        
        for (int j = row_start; j < row_end; j++)
        {
            int nbr = csr_col_idx[j];
            float val = csr_val[j];
            const float *in_row = &in_X[nbr * dim];
            
            int k;
            for (k = 0; k + 8 <= dim; k += 8)
            {
                __m256 v = _mm256_set1_ps(val);
                __m256 x = _mm256_load_ps(&in_row[k]);
                __m256 o = _mm256_load_ps(&out_row[k]);
                __m256 res = _mm256_fmadd_ps(v, x, o);
                _mm256_store_ps(&out_row[k], res);
            }
            for (; k < dim; k++)
            {
                out_row[k] += val * in_row[k];
            }
        }
    }
}

// Optimized ReLU - simple and fast
void ReLU(int dim, float *X)
{
    int total = v_num * dim;
    for (int i = 0; i < total; i++)
    {
        if (X[i] < 0) X[i] = 0;
    }
}

// Optimized LogSoftmax - simple implementation
void LogSoftmax(int dim, float *X)
{
    for (int i = 0; i < v_num; i++)
    {
        float *row = &X[i * dim];
        
        // Find max
        float max_val = row[0];
        for (int j = 1; j < dim; j++)
        {
            if (row[j] > max_val) max_val = row[j];
        }
        
        // Compute sum of exp
        float sum = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            row[j] -= max_val;
            sum += exp(row[j]);
        }
        
        float log_sum = log(sum);
        
        // Subtract log_sum
        for (int j = 0; j < dim; j++)
        {
            row[j] -= log_sum;
        }
    }
}

// Optimized MaxRowSum - simple and fast
float MaxRowSum(float *X, int dim)
{
    float max = -__FLT_MAX__;
    for (int i = 0; i < v_num; i++)
    {
        const float *row = &X[i * dim];
        float sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += row[j];
        }
        if (sum > max) max = sum;
    }
    return max;
}

void freeFloats()
{
    _mm_free(X0);
    _mm_free(W1);
    _mm_free(W2);
    _mm_free(X1);
    _mm_free(X2);
    _mm_free(X1_inter);
    _mm_free(X2_inter);
}

void somePreprocessing()
{
    raw_graph_to_CSR();
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

    TimePoint start = chrono::steady_clock::now();

    TimePoint prepross_start = chrono::steady_clock::now();
    somePreprocessing();
    TimePoint prepross_end = chrono::steady_clock::now();
    chrono::duration<double> prepross_ = prepross_end - prepross_start;
    double prepross_time = prepross_.count() * 1e3;
    printf("prepross_time: %.8lf\n", prepross_time);

    TimePoint edgeNorm_start = chrono::steady_clock::now();
    edgeNormalization();
    TimePoint edgeNorm_end = chrono::steady_clock::now();
    chrono::duration<double> edgeNorm_ = edgeNorm_end - edgeNorm_start;
    double edgeNorm_time = edgeNorm_.count() * 1e3;
    printf("edgeNorm_time: %.8lf\n", edgeNorm_time);

    TimePoint XW1_start = chrono::steady_clock::now();
    XW(F0, F1, X0, X1_inter, W1);
    TimePoint XW1_end = chrono::steady_clock::now();
    chrono::duration<double> XW1_ = XW1_end - XW1_start;
    double XW1_time = XW1_.count() * 1e3;
    printf("XW1_time: %.8lf\n", XW1_time);

    TimePoint AX1_start = chrono::steady_clock::now();
    AX(F1, X1_inter, X1);
    TimePoint AX1_end = chrono::steady_clock::now();
    chrono::duration<double> AX1_ = AX1_end - AX1_start;
    double AX1_time = AX1_.count() * 1e3;
    printf("AX1_time: %.8lf\n", AX1_time);

    TimePoint ReLU_start = chrono::steady_clock::now();
    ReLU(F1, X1);
    TimePoint ReLU_end = chrono::steady_clock::now();
    chrono::duration<double> ReLU_ = ReLU_end - ReLU_start;
    double ReLU_time = ReLU_.count() * 1e3;
    printf("ReLU_time: %.8lf\n", ReLU_time);

    TimePoint XW2_start = chrono::steady_clock::now();
    XW(F1, F2, X1, X2_inter, W2);
    TimePoint XW2_end = chrono::steady_clock::now();
    chrono::duration<double> XW2_ = XW2_end - XW2_start;
    double XW2_time = XW2_.count() * 1e3;
    printf("XW2_time: %.8lf\n", XW2_time);

    TimePoint AX2_start = chrono::steady_clock::now();
    AX(F2, X2_inter, X2);
    TimePoint AX2_end = chrono::steady_clock::now();
    chrono::duration<double> AX2_ = AX2_end - AX2_start;
    double AX2_time = AX2_.count() * 1e3;
    printf("AX2_time: %.8lf\n", AX2_time);

    TimePoint LogSoftmax_start = chrono::steady_clock::now();
    LogSoftmax(F2, X2);
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
