#include <stdio.h>
#include <stdlib.h>

int main() {
    int v_num = 1024;
    int F0 = 64;
    float* X0 = (float*)malloc(v_num * F0 * sizeof(float));
    FILE* fp = fopen("embedding/1024.bin", "rb");
    fread(X0, v_num * F0 * sizeof(float), 1, fp);
    fclose(fp);
    
    int zero_count = 0;
    for (int i = 0; i < v_num * F0; i++) {
        if (X0[i] == 0.0f) zero_count++;
    }
    printf("Zero count: %d / %d (%.2f%%)\n", zero_count, v_num * F0, (zero_count * 100.0) / (v_num * F0));
    free(X0);
    return 0;
}
