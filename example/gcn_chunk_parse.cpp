#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <chrono>
#include <x86intrin.h>
#include <cmath>
#include <sys/stat.h>
#include <algorithm>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

vector<int> raw_graph;
vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

inline int fast_atoi(const char *&p)
{
    unsigned char c;
    while (((c = static_cast<unsigned char>(*p)) - 9) < 5U || c == ' ') p++;
    int val = 0;
    while ((c = *p - '0') < 10U)
    {
        val = val * 10 + c;
        p++;
    }
    return val;
}

void readGraph(char *fname)
{
    struct stat sb;
    stat(fname, &sb);
    long fsize = sb.st_size;
    
    FILE *fp = fopen(fname, "rb");
    char *buffer = (char *)malloc(fsize + 1);
    fread(buffer, fsize, 1, fp);
    buffer[fsize] = '\0';
    fclose(fp);
    
    const char *p = buffer;
    v_num = fast_atoi(p);
    e_num = fast_atoi(p);
    
    raw_graph.reserve(e_num * 2);
    while (*p) {
        int src = fast_atoi(p);
        int dst = fast_atoi(p);
        raw_graph.push_back(src);
        raw_graph.push_back(dst);
    }
    
    free(buffer);
}

void raw_graph_to_AdjacencyList()
{
    int graph_edges = raw_graph.size() / 2;
    
    edge_index.resize(v_num);
    edge_val.resize(v_num);
    degree.resize(v_num, 0);
    
    vector<int> in_degree(v_num, 0);
    
    for (int i = 0; i < graph_edges; i++)
    {
        int src = raw_graph[2*i];
        int dst = raw_graph[2*i + 1];
        degree[src]++;
        in_degree[dst]++;
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        edge_index[i].reserve(in_degree[i]);
    }
    
    for (int i = 0; i < graph_edges; i++)
    {
        int src = raw_graph[2*i];
        int dst = raw_graph[2*i + 1];
        edge_index[dst].push_back(src);
    }
}

void edgeNormalization()
{
    vector<float> degree_sqrt(v_num);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        degree_sqrt[i] = sqrtf(static_cast<float>(degree[i]));
    }
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < v_num; i++)
    {
        const vector<int> &nlist = edge_index[i];
        int n = nlist.size();
        edge_val[i].resize(n);
        
        float di_sqrt = degree_sqrt[i];
        for (int j = 0; j < n; j++)
        {
            float dj_sqrt = degree_sqrt[nlist[j]];
            edge_val[i][j] = 1.0f / (di_sqrt * dj_sqrt);
        }
    }
}

void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

void initFloat(float *&dst, int num)
{
    dst = (float *)calloc(num, sizeof(float));
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < v_num; i++)
    {
        for (int k = 0; k < in_dim; k++)
        {
            float ival = in_X[i * in_dim + k];
            const float *wrow = &W[k * out_dim];
            float *out = &out_X[i * out_dim];
            
            int j = 0;
            __m256 v = _mm256_set1_ps(ival);
            for (; j + 8 <= out_dim; j += 8)
            {
                __m256 w = _mm256_loadu_ps(&wrow[j]);
                __m256 o = _mm256_loadu_ps(&out[j]);
                o = _mm256_fmadd_ps(v, w, o);
                _mm256_storeu_ps(&out[j], o);
            }
            for (; j < out_dim; j++)
            {
                out[j] += ival * wrow[j];
            }
        }
    }
}

void AX(int dim, float *in_X, float *out_X)
{
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < v_num; i++)
    {
        const vector<int> &nlist = edge_index[i];
        const vector<float> &vlist = edge_val[i];
        float *out = &out_X[i * dim];
        
        for (int j = 0; j < nlist.size(); j++)
        {
            int nbr = nlist[j];
            float val = vlist[j];
            const float *in = &in_X[nbr * dim];
            __m256 v = _mm256_set1_ps(val);
            
            int k = 0;
            for (; k + 8 <= dim; k += 8)
            {
                __m256 a = _mm256_loadu_ps(&out[k]);
                __m256 b = _mm256_loadu_ps(&in[k]);
                a = _mm256_fmadd_ps(v, b, a);
                _mm256_storeu_ps(&out[k], a);
            }
            for (; k < dim; k++)
            {
                out[k] += val * in[k];
            }
        }
    }
}

void ReLU(int dim, float *X)
{
    int total = v_num * dim;
    __m256 zero = _mm256_setzero_ps();
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total / 8; i++)
    {
        __m256 x = _mm256_loadu_ps(&X[i * 8]);
        x = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(&X[i * 8], x);
    }
    for (int i = (total / 8) * 8; i < total; i++)
    {
        if (X[i] < 0) X[i] = 0;
    }
}

void LogSoftmax(int dim, float *X)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float *row = &X[i * dim];
        
        float max_val = row[0];
        for (int j = 1; j < dim; j++)
            if (row[j] > max_val) max_val = row[j];
        
        float sum = 0;
        for (int j = 0; j < dim; j++)
            sum += std::exp(row[j] - max_val);
        
        float log_sum = std::log(sum);
        
        for (int j = 0; j < dim; j++)
            row[j] = row[j] - max_val - log_sum;
    }
}

float MaxRowSum(float *X, int dim)
{
    float max_sum = -__FLT_MAX__;
    #pragma omp parallel for reduction(max:max_sum) schedule(static)
    for (int i = 0; i < v_num; i++)
    {
        float sum = 0;
        const float *row = &X[i * dim];
        for (int j = 0; j < dim; j++)
        {
            sum += row[j];
        }
        if (sum > max_sum) max_sum = sum;
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

    TimePoint start = chrono::steady_clock::now();

    TimePoint prepross_start = chrono::steady_clock::now();
    raw_graph_to_AdjacencyList();
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
