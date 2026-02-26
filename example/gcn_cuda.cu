#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<int> csr_row_ptr;
vector<int> csr_col_idx;
vector<float> csr_val;
vector<int> degree;
vector<int> raw_graph;

float *d_X0, *d_W1, *d_W2, *d_X1, *d_X1_inter, *d_X2, *d_X2_inter;
int *d_csr_row_ptr, *d_csr_col_idx;
float *d_csr_val;

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

void edgeNormalization()
{
    vector<float> inv_sqrt_degree(v_num);
    for (int i = 0; i < v_num; i++)
    {
        inv_sqrt_degree[i] = 1.0f / sqrtf((float)degree[i]);
    }
    
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
    dst = (float *)malloc(num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    size_t ret = fread(dst, num * sizeof(float), 1, fp);
    (void)ret;
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    memset(dst, 0, num * sizeof(float));
}

// XW kernel: each thread computes one output element
__global__ void XW_kernel(int v_num, int in_dim, int out_dim, 
                          float *in_X, float *out_X, float *W)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < v_num && col < out_dim)
    {
        float sum = 0.0f;
        for (int k = 0; k < in_dim; k++)
        {
            sum += in_X[row * in_dim + k] * W[k * out_dim + col];
        }
        out_X[row * out_dim + col] = sum;
    }
}

// AX kernel: one warp per row, each thread handles one feature dimension
__global__ void AX_kernel(int v_num, int dim, 
                          int *csr_row_ptr, int *csr_col_idx, float *csr_val,
                          float *in_X, float *out_X)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row >= v_num || col >= dim) return;
    
    float sum = 0.0f;
    int row_start = csr_row_ptr[row];
    int row_end = csr_row_ptr[row + 1];
    
    for (int j = row_start; j < row_end; j++)
    {
        int neighbor = csr_col_idx[j];
        sum += in_X[neighbor * dim + col] * csr_val[j];
    }
    
    out_X[row * dim + col] = sum;
}

// Fused AX + ReLU kernel
__global__ void AX_ReLU_kernel(int v_num, int dim, 
                               int *csr_row_ptr, int *csr_col_idx, float *csr_val,
                               float *in_X, float *out_X)
{
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (row >= v_num || col >= dim) return;
    
    float sum = 0.0f;
    int row_start = csr_row_ptr[row];
    int row_end = csr_row_ptr[row + 1];
    
    for (int j = row_start; j < row_end; j++)
    {
        int neighbor = csr_col_idx[j];
        sum += in_X[neighbor * dim + col] * csr_val[j];
    }
    
    out_X[row * dim + col] = sum > 0 ? sum : 0;
}

// LogSoftmax kernel: one block per row
__global__ void LogSoftmax_kernel(int v_num, int dim, float *X)
{
    int row = blockIdx.x;
    if (row >= v_num) return;
    
    extern __shared__ float sdata[];
    
    float max_val = X[row * dim];
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        max_val = fmaxf(max_val, X[row * dim + i]);
    }
    
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        float exp_val = expf(X[row * dim + i] - max_val);
        X[row * dim + i] = exp_val;
        sum += exp_val;
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();
    
    float log_sum = logf(sum);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        X[row * dim + i] = logf(X[row * dim + i]) - log_sum;
    }
}

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

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

    cudaMalloc(&d_X0, v_num * F0 * sizeof(float));
    cudaMalloc(&d_W1, F0 * F1 * sizeof(float));
    cudaMalloc(&d_W2, F1 * F2 * sizeof(float));
    cudaMalloc(&d_X1, v_num * F1 * sizeof(float));
    cudaMalloc(&d_X1_inter, v_num * F1 * sizeof(float));
    cudaMalloc(&d_X2, v_num * F2 * sizeof(float));
    cudaMalloc(&d_X2_inter, v_num * F2 * sizeof(float));
    cudaMalloc(&d_csr_row_ptr, (v_num + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_idx, csr_col_idx.size() * sizeof(int));
    cudaMalloc(&d_csr_val, csr_val.size() * sizeof(float));

    cudaMemcpy(d_X0, X0, v_num * F0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, F0 * F1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, F1 * F2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (v_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_idx, csr_col_idx.data(), csr_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_val, csr_val.data(), csr_val.size() * sizeof(float), cudaMemcpyHostToDevice);

    TimePoint compute_start = chrono::steady_clock::now();

    dim3 blockSizeXW(16, 16);
    dim3 gridSizeXW1((F1 + 15) / 16, (v_num + 15) / 16);
    dim3 gridSizeXW2((F2 + 15) / 16, (v_num + 15) / 16);

    XW_kernel<<<gridSizeXW1, blockSizeXW>>>(v_num, F0, F1, d_X0, d_X1_inter, d_W1);
    AX_ReLU_kernel<<<v_num, F1>>>(v_num, F1, d_csr_row_ptr, d_csr_col_idx, d_csr_val, d_X1_inter, d_X1);
    XW_kernel<<<gridSizeXW2, blockSizeXW>>>(v_num, F1, F2, d_X1, d_X2_inter, d_W2);
    AX_kernel<<<v_num, F2>>>(v_num, F2, d_csr_row_ptr, d_csr_col_idx, d_csr_val, d_X2_inter, d_X2);
    LogSoftmax_kernel<<<v_num, 256, 256 * sizeof(float)>>>(v_num, F2, d_X2);
    
    cudaDeviceSynchronize();

    TimePoint compute_end = chrono::steady_clock::now();
    chrono::duration<double> compute_ = compute_end - compute_start;
    double compute_time = compute_.count() * 1e3;
    printf("compute_time: %.8lf\n", compute_time);

    cudaMemcpy(X2, d_X2, v_num * F2 * sizeof(float), cudaMemcpyDeviceToHost);

    TimePoint max_sum_start = chrono::steady_clock::now();
    float max_sum = -__FLT_MAX__;
    for (int i = 0; i < v_num; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < F2; j++)
        {
            sum += X2[i * F2 + j];
        }
        if (sum > max_sum)
            max_sum = sum;
    }
    TimePoint max_sum_end = chrono::steady_clock::now();
    chrono::duration<double> max_sum_ = max_sum_end - max_sum_start;
    double max_sum_time = max_sum_.count() * 1e3;
    printf("max_sum_time: %.8lf\n", max_sum_time);

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    printf("%.8f\n", max_sum);
    printf("total time: %.8lf\n\n", l_timeMs);

    cudaFree(d_X0);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_X1);
    cudaFree(d_X1_inter);
    cudaFree(d_X2);
    cudaFree(d_X2_inter);
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_col_idx);
    cudaFree(d_csr_val);

    freeFloats();
}
