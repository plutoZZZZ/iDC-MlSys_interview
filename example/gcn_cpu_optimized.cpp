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
#include <immintrin.h>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

// CSR format for sparse matrix (incoming edges)
vector<int> csr_row_ptr;
vector<int> csr_col_idx;
vector<float> csr_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(char *fname)
{
    ifstream infile(fname);
    int source;
    int end;

    infile >> v_num >> e_num;

    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF)
            break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

// Build CSR format for incoming edges
void buildCSR()
{
    degree.resize(v_num, 0);
    
    for (size_t i = 0; i < raw_graph.size() / 2; i++)
    {
        int src = raw_graph[2*i];
        int dst = raw_graph[2*i + 1];
        degree[dst]++;
    }
    
    csr_row_ptr.resize(v_num + 1);
    csr_row_ptr[0] = 0;
    for (int i = 0; i < v_num; i++)
    {
        csr_row_ptr[i + 1] = csr_row_ptr[i] + degree[i];
    }
    
    int total_edges = csr_row_ptr[v_num];
    csr_col_idx.resize(total_edges);
    csr_val.resize(total_edges);
    
    vector<int> row_ptr_copy = csr_row_ptr;
    
    for (size_t i = 0; i < raw_graph.size() / 2; i++)
    {
        int src = raw_graph[2*i];
        int dst = raw_graph[2*i + 1];
        csr_col_idx[row_ptr_copy[dst]] = src;
        row_ptr_copy[dst]++;
    }
}

// 优化edgeNormalization：减少并行开销，使用更粗的粒度
void edgeNormalization()
{
    vector<float> inv_sqrt_degree(v_num);
    for (int i = 0; i < v_num; i++)
    {
        inv_sqrt_degree[i] = 1.0f / sqrtf((float)degree[i]);
    }
    
    // 使用动态调度，但减少并行区域的创建开销
    // 对于小图，减少线程数
    int num_threads = omp_get_max_threads();
    int chunk_size = max(v_num / num_threads / 4, 64);
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < v_num; i++)
    {
        float inv_sqrt_d_i = inv_sqrt_degree[i];
        for (int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
        {
            int col = csr_col_idx[j];
            csr_val[j] = inv_sqrt_d_i * inv_sqrt_degree[col];
        }
    }
}

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)aligned_alloc(64, num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    size_t ret = fread(dst, num * sizeof(float), 1, fp);
    (void)ret;
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)aligned_alloc(64, num * sizeof(float));
    memset(dst, 0, num * sizeof(float));
}

// 高度优化的XW，使用更大的SIMD展开和更好的缓存策略
void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
    // 使用更大的块大小减少并行开销
    const int BM = 128;
    const int BN = 64;
    const int BK = 64;
    
    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < v_num; m += BM)
    {
        int m_end = min(m + BM, v_num);
        
        for (int n = 0; n < out_dim; n += BN)
        {
            int n_end = min(n + BN, out_dim);
            
            // 初始化输出块
            for (int i = m; i < m_end; i++)
            {
                for (int j = n; j < n_end; j++)
                {
                    out_X[i * out_dim + j] = 0.0f;
                }
            }
            
            for (int k = 0; k < in_dim; k += BK)
            {
                int k_end = min(k + BK, in_dim);
                
                for (int i = m; i < m_end; i++)
                {
                    for (int j = n; j < n_end; j++)
                    {
                        float sum = out_X[i * out_dim + j];
                        int kk = k;
                        
                        // 16路展开，更好地利用寄存器
                        for (; kk <= k_end - 16; kk += 16)
                        {
                            sum += in_X[i * in_dim + kk] * W[kk * out_dim + j];
                            sum += in_X[i * in_dim + kk + 1] * W[(kk + 1) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 2] * W[(kk + 2) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 3] * W[(kk + 3) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 4] * W[(kk + 4) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 5] * W[(kk + 5) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 6] * W[(kk + 6) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 7] * W[(kk + 7) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 8] * W[(kk + 8) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 9] * W[(kk + 9) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 10] * W[(kk + 10) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 11] * W[(kk + 11) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 12] * W[(kk + 12) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 13] * W[(kk + 13) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 14] * W[(kk + 14) * out_dim + j];
                            sum += in_X[i * in_dim + kk + 15] * W[(kk + 15) * out_dim + j];
                        }
                        
                        for (; kk < k_end; kk++)
                        {
                            sum += in_X[i * in_dim + kk] * W[kk * out_dim + j];
                        }
                        
                        out_X[i * out_dim + j] = sum;
                    }
                }
            }
        }
    }
}

// 融合AX和ReLU：减少一次内存遍历
void AX_ReLU(int dim, float *in_X, float *out_X)
{
    int chunk_size = max(v_num / omp_get_max_threads() / 4, 64);
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < v_num; i++)
    {
        for (int k = 0; k < dim; k++)
        {
            float sum = 0.0f;
            int row_start = csr_row_ptr[i];
            int row_end = csr_row_ptr[i + 1];
            
            // 手动展开4路
            int j = row_start;
            for (; j <= row_end - 4; j += 4)
            {
                sum += in_X[csr_col_idx[j] * dim + k] * csr_val[j];
                sum += in_X[csr_col_idx[j + 1] * dim + k] * csr_val[j + 1];
                sum += in_X[csr_col_idx[j + 2] * dim + k] * csr_val[j + 2];
                sum += in_X[csr_col_idx[j + 3] * dim + k] * csr_val[j + 3];
            }
            
            for (; j < row_end; j++)
            {
                sum += in_X[csr_col_idx[j] * dim + k] * csr_val[j];
            }
            
            // 融合ReLU
            out_X[i * dim + k] = sum > 0 ? sum : 0;
        }
    }
}

// 优化的AX（不融合ReLU，用于第二层）
void AX(int dim, float *in_X, float *out_X)
{
    int chunk_size = max(v_num / omp_get_max_threads() / 4, 64);
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < v_num; i++)
    {
        for (int k = 0; k < dim; k++)
        {
            float sum = 0.0f;
            int row_start = csr_row_ptr[i];
            int row_end = csr_row_ptr[i + 1];
            
            // 手动展开4路
            int j = row_start;
            for (; j <= row_end - 4; j += 4)
            {
                sum += in_X[csr_col_idx[j] * dim + k] * csr_val[j];
                sum += in_X[csr_col_idx[j + 1] * dim + k] * csr_val[j + 1];
                sum += in_X[csr_col_idx[j + 2] * dim + k] * csr_val[j + 2];
                sum += in_X[csr_col_idx[j + 3] * dim + k] * csr_val[j + 3];
            }
            
            for (; j < row_end; j++)
            {
                sum += in_X[csr_col_idx[j] * dim + k] * csr_val[j];
            }
            
            out_X[i * dim + k] = sum;
        }
    }
}

// 优化的LogSoftmax
void LogSoftmax(int dim, float *X)
{
    int chunk_size = max(v_num / omp_get_max_threads() / 4, 64);
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < v_num; i++)
    {
        // Find max
        float max_val = X[i * dim];
        for (int j = 1; j < dim; j++)
        {
            max_val = max(max_val, X[i * dim + j]);
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            X[i * dim + j] = expf(X[i * dim + j] - max_val);
            sum += X[i * dim + j];
        }
        
        // Compute log and subtract
        float log_sum = logf(sum);
        for (int j = 0; j < dim; j++)
        {
            X[i * dim + j] = logf(X[i * dim + j]) - log_sum;
        }
    }
}

float MaxRowSum(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    int chunk_size = max(v_num / omp_get_max_threads() / 4, 64);
    
    #pragma omp parallel for reduction(max:max_sum) schedule(dynamic, chunk_size)
    for (int i = 0; i < v_num; i++)
    {
        float sum = 0.0f;
        int j = 0;
        
        // 8路展开
        for (; j <= dim - 8; j += 8)
        {
            sum += X[i * dim + j];
            sum += X[i * dim + j + 1];
            sum += X[i * dim + j + 2];
            sum += X[i * dim + j + 3];
            sum += X[i * dim + j + 4];
            sum += X[i * dim + j + 5];
            sum += X[i * dim + j + 6];
            sum += X[i * dim + j + 7];
        }
        
        for (; j < dim; j++)
        {
            sum += X[i * dim + j];
        }
        
        if (sum > max_sum)
            max_sum = sum;
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

void somePreprocessing()
{
    buildCSR();
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

    // Layer 1: XW
    TimePoint XW1_start = chrono::steady_clock::now();
    XW(F0, F1, X0, X1_inter, W1);
    TimePoint XW1_end = chrono::steady_clock::now();
    chrono::duration<double> XW1_ = XW1_end - XW1_start;
    double XW1_time = XW1_.count() * 1e3;
    printf("XW1_time: %.8lf\n", XW1_time);

    // Layer 1: AX + ReLU (融合)
    TimePoint AX1_start = chrono::steady_clock::now();
    AX_ReLU(F1, X1_inter, X1);
    TimePoint AX1_end = chrono::steady_clock::now();
    chrono::duration<double> AX1_ = AX1_end - AX1_start;
    double AX1_time = AX1_.count() * 1e3;
    printf("AX1+ReLU_time: %.8lf\n", AX1_time);

    // Layer 2: XW
    TimePoint XW2_start = chrono::steady_clock::now();
    XW(F1, F2, X1, X2_inter, W2);
    TimePoint XW2_end = chrono::steady_clock::now();
    chrono::duration<double> XW2_ = XW2_end - XW2_start;
    double XW2_time = XW2_.count() * 1e3;
    printf("XW2_time: %.8lf\n", XW2_time);

    // Layer 2: AX
    TimePoint AX2_start = chrono::steady_clock::now();
    AX(F2, X2_inter, X2);
    TimePoint AX2_end = chrono::steady_clock::now();
    chrono::duration<double> AX2_ = AX2_end - AX2_start;
    double AX2_time = AX2_.count() * 1e3;
    printf("AX2_time: %.8lf\n", AX2_time);

    // Layer 2: LogSoftmax
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
