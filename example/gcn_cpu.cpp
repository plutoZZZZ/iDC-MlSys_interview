#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <vector>
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

vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_tmp, *X2, *X2_tmp;

int *row_ptr;
int *col_idx;
float *edge_val_csr;

void readGraph(char *fname)
{
    ifstream infile(fname);
    int source, end;
    infile >> v_num >> e_num;
    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF) break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

void buildCSR()
{
    vector<int> degree(v_num, 0), in_degree(v_num, 0);
    for (size_t i = 0; i < raw_graph.size() / 2; i++)
    {
        degree[raw_graph[2 * i]]++;
        in_degree[raw_graph[2 * i + 1]]++;
    }
    row_ptr = new int[v_num + 1];
    col_idx = new int[e_num];
    edge_val_csr = new float[e_num];
    row_ptr[0] = 0;
    for (int i = 0; i < v_num; i++) row_ptr[i + 1] = row_ptr[i] + in_degree[i];
    vector<int> current_pos(v_num, 0);
    for (int i = 0; i < v_num; i++) current_pos[i] = row_ptr[i];
    for (size_t i = 0; i < raw_graph.size() / 2; i++)
    {
        int src = raw_graph[2 * i], dst = raw_graph[2 * i + 1];
        col_idx[current_pos[dst]] = src;
        current_pos[dst]++;
    }
    vector<float> sqrt_deg(v_num);
    for (int i = 0; i < v_num; i++) sqrt_deg[i] = sqrtf((float)degree[i]);
    for (int i = 0; i < v_num; i++)
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            edge_val_csr[j] = 1.0f / (sqrt_deg[i] * sqrt_deg[col_idx[j]]);
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
    memset(dst, 0, num * sizeof(float));
}

#ifdef __AVX512F__
void XW_AX_ReLU_fused(int in_dim, int out_dim, float *X, float *W, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *out_row = out + i * out_dim;
        __m512 zero = _mm512_setzero_ps();
        
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            int nbr = col_idx[j];
            float val = edge_val_csr[j];
            __m512 val_vec = _mm512_set1_ps(val);
            float *nbr_row = X + nbr * in_dim;
            
            for (int k = 0; k < in_dim; k++)
            {
                float x_val = nbr_row[k];
                __m512 x_vec = _mm512_set1_ps(x_val);
                float *w_row = W + k * out_dim;
                
                int col = 0;
                int simd_end = (out_dim / 16) * 16;
                for (; col < simd_end; col += 16)
                {
                    __m512 w_vec = _mm512_loadu_ps(w_row + col);
                    __m512 out_vec = _mm512_loadu_ps(out_row + col);
                    __m512 prod = _mm512_mul_ps(x_vec, w_vec);
                    __m512 scaled = _mm512_mul_ps(prod, val_vec);
                    out_vec = _mm512_add_ps(out_vec, scaled);
                    out_vec = _mm512_max_ps(out_vec, zero);
                    _mm512_storeu_ps(out_row + col, out_vec);
                }
                for (; col < out_dim; col++)
                {
                    float tmp = out_row[col] + x_val * w_row[col] * val;
                    out_row[col] = tmp > 0 ? tmp : 0;
                }
            }
        }
    }
}

void XW_fused(int in_dim, int out_dim, float *X, float *W, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *x_row = X + i * in_dim;
        float *out_row = out + i * out_dim;
        
        for (int k = 0; k < in_dim; k++)
        {
            float val = x_row[k];
            __m512 val_vec = _mm512_set1_ps(val);
            float *w_row = W + k * out_dim;
            
            int j = 0;
            int simd_end = (out_dim / 16) * 16;
            for (; j < simd_end; j += 16)
            {
                __m512 w_vec = _mm512_loadu_ps(w_row + j);
                __m512 out_vec = _mm512_loadu_ps(out_row + j);
                out_vec = _mm512_fmadd_ps(val_vec, w_vec, out_vec);
                _mm512_storeu_ps(out_row + j, out_vec);
            }
            for (; j < out_dim; j++) out_row[j] += val * w_row[j];
        }
    }
}

void AX_fused(int dim, float *in_X, float *out_X)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *out_row = out_X + i * dim;
        
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            int nbr = col_idx[j];
            float val = edge_val_csr[j];
            __m512 val_vec = _mm512_set1_ps(val);
            float *in_row = in_X + nbr * dim;
            
            int k = 0;
            int simd_end = (dim / 16) * 16;
            for (; k < simd_end; k += 16)
            {
                __m512 in_vec = _mm512_loadu_ps(in_row + k);
                __m512 out_vec = _mm512_loadu_ps(out_row + k);
                out_vec = _mm512_fmadd_ps(val_vec, in_vec, out_vec);
                _mm512_storeu_ps(out_row + k, out_vec);
            }
            for (; k < dim; k++) out_row[k] += in_row[k] * val;
        }
    }
}

void ReLU_fused(int dim, float *X)
{
    int total = v_num * dim;
    __m512 zero = _mm512_setzero_ps();
    int i = 0;
    int simd_end = (total / 16) * 16;
    for (; i < simd_end; i += 16)
    {
        __m512 vec = _mm512_loadu_ps(X + i);
        vec = _mm512_max_ps(vec, zero);
        _mm512_storeu_ps(X + i, vec);
    }
    for (; i < total; i++) if (X[i] < 0) X[i] = 0;
}

void LogSoftmax_fused(int dim, float *X)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *row = X + i * dim;
        __m512 max_vec = _mm512_set1_ps(row[0]);
        int j = 0;
        int simd_end = (dim / 16) * 16;
        for (; j < simd_end; j += 16)
        {
            __m512 vec = _mm512_loadu_ps(row + j);
            max_vec = _mm512_max_ps(max_vec, vec);
        }
        float max_val = _mm512_reduce_max_ps(max_vec);
        for (; j < dim; j++) if (row[j] > max_val) max_val = row[j];
        
        float sum = 0;
        j = 0;
        for (; j < simd_end; j += 16)
        {
            float temp[16];
            for (int k = 0; k < 16; k++) temp[k] = row[j + k] - max_val;
            for (int k = 0; k < 16; k++) sum += expf(temp[k]);
        }
        for (; j < dim; j++) sum += expf(row[j] - max_val);
        
        sum = logf(sum);
        __m512 max_v = _mm512_set1_ps(max_val);
        __m512 sum_v = _mm512_set1_ps(sum);
        j = 0;
        for (; j < simd_end; j += 16)
        {
            __m512 vec = _mm512_loadu_ps(row + j);
            vec = _mm512_sub_ps(vec, max_v);
            vec = _mm512_sub_ps(vec, sum_v);
            _mm512_storeu_ps(row + j, vec);
        }
        for (; j < dim; j++) row[j] -= max_val + sum;
    }
}

float MaxRowSum_fused(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    #pragma omp parallel for reduction(max:max_sum)
    for (int i = 0; i < v_num; i++)
    {
        float *row = X + i * dim;
        __m512 sum_vec = _mm512_setzero_ps();
        int j = 0;
        int simd_end = (dim / 16) * 16;
        for (; j < simd_end; j += 16)
        {
            __m512 vec = _mm512_loadu_ps(row + j);
            sum_vec = _mm512_add_ps(sum_vec, vec);
        }
        float sum = _mm512_reduce_add_ps(sum_vec);
        for (; j < dim; j++) sum += row[j];
        if (sum > max_sum) max_sum = sum;
    }
    return max_sum;
}
#else
void XW_fused(int in_dim, int out_dim, float *X, float *W, float *out)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *x_row = X + i * in_dim;
        float *out_row = out + i * out_dim;
        for (int k = 0; k < in_dim; k++)
        {
            float val = x_row[k];
            float *w_row = W + k * out_dim;
            #pragma omp simd
            for (int j = 0; j < out_dim; j++) out_row[j] += val * w_row[j];
        }
    }
}

void AX_fused(int dim, float *in_X, float *out_X)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *out_row = out_X + i * dim;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            int nbr = col_idx[j];
            float val = edge_val_csr[j];
            float *in_row = in_X + nbr * dim;
            #pragma omp simd
            for (int k = 0; k < dim; k++) out_row[k] += in_row[k] * val;
        }
    }
}

void ReLU_fused(int dim, float *X)
{
    int total = v_num * dim;
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < total; i++) if (X[i] < 0) X[i] = 0;
}

void LogSoftmax_fused(int dim, float *X)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *row = X + i * dim;
        float max_val = row[0];
        for (int j = 1; j < dim; j++) if (row[j] > max_val) max_val = row[j];
        float sum = 0;
        for (int j = 0; j < dim; j++) sum += expf(row[j] - max_val);
        sum = logf(sum);
        for (int j = 0; j < dim; j++) row[j] -= max_val + sum;
    }
}

float MaxRowSum_fused(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    #pragma omp parallel for reduction(max:max_sum)
    for (int i = 0; i < v_num; i++)
    {
        float *row = X + i * dim;
        float sum = 0;
        for (int j = 0; j < dim; j++) sum += row[j];
        if (sum > max_sum) max_sum = sum;
    }
    return max_sum;
}
#endif

void freeFloats()
{
    free(X0); free(W1); free(W2); free(X1); free(X1_tmp); free(X2); free(X2_tmp);
    delete[] row_ptr; delete[] col_idx; delete[] edge_val_csr;
}

int main(int argc, char **argv)
{
    F0 = atoi(argv[1]); F1 = atoi(argv[2]); F2 = atoi(argv[3]);
    readGraph(argv[4]);
    readFloat(argv[5], X0, v_num * F0);
    readFloat(argv[6], W1, F0 * F1);
    readFloat(argv[7], W2, F1 * F2);
    initFloat(X1, v_num * F1);
    initFloat(X1_tmp, v_num * F1);
    initFloat(X2, v_num * F2);
    initFloat(X2_tmp, v_num * F2);

    TimePoint start = chrono::steady_clock::now();
    
    TimePoint t1 = chrono::steady_clock::now();
    buildCSR();
    TimePoint t2 = chrono::steady_clock::now();
    printf("prepross_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    t1 = chrono::steady_clock::now();
    XW_fused(F0, F1, X0, W1, X1_tmp);
    t2 = chrono::steady_clock::now();
    printf("XW1_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    t1 = chrono::steady_clock::now();
    AX_fused(F1, X1_tmp, X1);
    t2 = chrono::steady_clock::now();
    printf("AX1_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    t1 = chrono::steady_clock::now();
    ReLU_fused(F1, X1);
    t2 = chrono::steady_clock::now();
    printf("ReLU_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    t1 = chrono::steady_clock::now();
    XW_fused(F1, F2, X1, W2, X2_tmp);
    t2 = chrono::steady_clock::now();
    printf("XW2_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    t1 = chrono::steady_clock::now();
    AX_fused(F2, X2_tmp, X2);
    t2 = chrono::steady_clock::now();
    printf("AX2_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    t1 = chrono::steady_clock::now();
    LogSoftmax_fused(F2, X2);
    t2 = chrono::steady_clock::now();
    printf("LogSoftmax_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    t1 = chrono::steady_clock::now();
    float max_sum = MaxRowSum_fused(X2, F2);
    t2 = chrono::steady_clock::now();
    printf("max_sum_time: %.8lf\n", chrono::duration<double>(t2 - t1).count() * 1e3);

    TimePoint end = chrono::steady_clock::now();
    printf("%.8f\n", max_sum);
    printf("total time: %.8lf\n\n", chrono::duration<double>(end - start).count() * 1e3);

    freeFloats();
    return 0;
}
