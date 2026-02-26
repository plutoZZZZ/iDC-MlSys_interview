#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

int v_num = 0, e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;
vector<int> raw_graph;

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA err: %s\n", cudaGetErrorString(err)); exit(1); } } while(0)

void readGraph(char *fname)
{
    ifstream infile(fname);
    int source, end;
    infile >> v_num >> e_num;
    while (!infile.eof()) {
        infile >> source >> end;
        if (infile.peek() == EOF) break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)aligned_alloc(64, num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

__global__ void XW_kernel(int v_num, int in_dim, int out_dim, float *X, float *W, float *out)
{
    int row = blockIdx.x;
    int feat = threadIdx.x;
    
    if (row >= v_num || feat >= out_dim) return;
    
    float sum = 0;
    for (int k = 0; k < in_dim; k++)
        sum += X[row * in_dim + k] * W[k * out_dim + feat];
    
    out[row * out_dim + feat] = sum;
}

__global__ void AX_kernel(int v_num, int dim, int *row_ptr, int *col_idx, float *val, float *in_X, float *out_X)
{
    int row = blockIdx.x;
    int feat = threadIdx.x;
    
    if (row >= v_num || feat >= dim) return;
    
    float sum = 0;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++)
    {
        int nbr = col_idx[j];
        sum += in_X[nbr * dim + feat] * val[j];
    }
    out_X[row * dim + feat] = sum;
}

__global__ void ReLU_kernel(int n, float *X)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && X[i] < 0) X[i] = 0;
}

__global__ void LogSoftmax_kernel(int v_num, int dim, float *X)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= v_num) return;
    
    float *r = X + row * dim;
    
    float max_val = -1e30f;
    for (int i = tid; i < dim; i += 32)
        max_val = fmaxf(max_val, r[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    float sum = 0;
    for (int i = tid; i < dim; i += 32)
        sum += expf(r[i] - max_val);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);
    
    float log_sum = logf(sum);
    for (int i = tid; i < dim; i += 32)
        r[i] = r[i] - max_val - log_sum;
}

__global__ void MaxRowSum_kernel(int v_num, int dim, float *X, float *result)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= v_num) return;
    
    float sum = 0;
    for (int i = tid; i < dim; i += 32)
        sum += X[row * dim + i];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    if (tid == 0) result[row] = sum;
}

int main(int argc, char **argv)
{
    F0 = atoi(argv[1]); F1 = atoi(argv[2]); F2 = atoi(argv[3]);
    readGraph(argv[4]);
    
    float *h_X0, *h_W1, *h_W2;
    readFloat(argv[5], h_X0, v_num * F0);
    readFloat(argv[6], h_W1, F0 * F1);
    readFloat(argv[7], h_W2, F1 * F2);
    
    float *h_X1 = (float *)calloc(v_num * F1, sizeof(float));
    float *h_X1_tmp = (float *)calloc(v_num * F1, sizeof(float));
    float *h_X2 = (float *)calloc(v_num * F2, sizeof(float));
    float *h_X2_tmp = (float *)calloc(v_num * F2, sizeof(float));
    
    vector<int> degree(v_num, 0), in_degree(v_num, 0);
    for (size_t i = 0; i < raw_graph.size() / 2; i++) {
        degree[raw_graph[2 * i]]++;
        in_degree[raw_graph[2 * i + 1]]++;
    }
    
    int *row_ptr = new int[v_num + 1];
    int *col_idx = new int[e_num];
    float *edge_val = new float[e_num];
    row_ptr[0] = 0;
    for (int i = 0; i < v_num; i++) row_ptr[i + 1] = row_ptr[i] + in_degree[i];
    
    vector<int> pos(v_num, 0);
    for (int i = 0; i < v_num; i++) pos[i] = row_ptr[i];
    for (size_t i = 0; i < raw_graph.size() / 2; i++) {
        int src = raw_graph[2 * i], dst = raw_graph[2 * i + 1];
        col_idx[pos[dst]] = src;
        pos[dst]++;
    }
    
    vector<float> sqrt_deg(v_num);
    for (int i = 0; i < v_num; i++) sqrt_deg[i] = sqrtf((float)degree[i]);
    for (int i = 0; i < v_num; i++)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            edge_val[j] = 1.0f / (sqrt_deg[i] * sqrt_deg[col_idx[j]]);
    
    float *d_X0, *d_W1, *d_W2, *d_X1, *d_X1_tmp, *d_X2, *d_X2_tmp;
    int *d_row_ptr, *d_col_idx;
    float *d_edge_val;
    
    CUDA_CHECK(cudaMalloc(&d_X0, v_num * F0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, F0 * F1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, F1 * F2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X1, v_num * F1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X1_tmp, v_num * F1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X2, v_num * F2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X2_tmp, v_num * F2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (v_num + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, e_num * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_val, e_num * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_X0, h_X0, v_num * F0 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1, F0 * F1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2, F1 * F2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (v_num + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, e_num * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edge_val, edge_val, e_num * sizeof(float), cudaMemcpyHostToDevice));
    
    auto start = chrono::steady_clock::now();
    
    XW_kernel<<<v_num, 64>>>(v_num, F0, F1, d_X0, d_W1, d_X1_tmp);
    AX_kernel<<<v_num, 32>>>(v_num, F1, d_row_ptr, d_col_idx, d_edge_val, d_X1_tmp, d_X1);
    ReLU_kernel<<<(v_num * F1 + 255) / 256, 256>>>(v_num * F1, d_X1);
    
    XW_kernel<<<v_num, 16>>>(v_num, F1, F2, d_X1, d_W2, d_X2_tmp);
    AX_kernel<<<v_num, 32>>>(v_num, F2, d_row_ptr, d_col_idx, d_edge_val, d_X2_tmp, d_X2);
    LogSoftmax_kernel<<<v_num, 32>>>(v_num, F2, d_X2);
    
    float *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, v_num * sizeof(float)));
    MaxRowSum_kernel<<<v_num, 32>>>(v_num, F2, d_X2, d_result);
    
    float *h_result = new float[v_num];
    CUDA_CHECK(cudaMemcpy(h_result, d_result, v_num * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_sum = h_result[0];
    for (int i = 1; i < v_num; i++) if (h_result[i] > max_sum) max_sum = h_result[i];
    
    auto end = chrono::steady_clock::now();
    
    printf("%.8f\n", max_sum);
    printf("total time: %.8lf\n", chrono::duration<double>(end - start).count() * 1e3);
    
    CUDA_CHECK(cudaFree(d_X0)); CUDA_CHECK(cudaFree(d_W1)); CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_X1)); CUDA_CHECK(cudaFree(d_X1_tmp));
    CUDA_CHECK(cudaFree(d_X2)); CUDA_CHECK(cudaFree(d_X2_tmp));
    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx)); CUDA_CHECK(cudaFree(d_edge_val));
    CUDA_CHECK(cudaFree(d_result));
    
    free(h_X0); free(h_W1); free(h_W2); free(h_X1); free(h_X1_tmp); free(h_X2); free(h_X2_tmp);
    delete[] row_ptr; delete[] col_idx; delete[] edge_val; delete[] h_result;
    
    return 0;
}
