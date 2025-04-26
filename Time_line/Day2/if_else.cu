#include <cuda_runtime.h>
#include <stdio.h>

__global__ void Varmul(float* c, int N) {
    // 正确的全局索引计算
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (tid >= N) return;

    float a = 0.0f, b = 0.0f;
    
    // 优化分支：减少Warp Divergence
    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    
    c[tid] = a + b;
}

int main() {
    const int N = 1024; // 数组长度
    float *d_c;
    float *h_c = (float*)malloc(N * sizeof(float));

    // 分配设备内存
    cudaMalloc(&d_c, N * sizeof(float));

    // 启动核函数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    Varmul<<<gridSize, blockSize>>>(d_c, N);

    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %.1f\n", i, h_c[i]);
    }

    // 释放内存
    free(h_c);
    cudaFree(d_c);
    return 0;
}