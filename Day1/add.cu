#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA核函数：每个线程计算一个数组元素的和
__global__ void vectorAdd(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程的全局索引
    if (tid < size) {
        c[tid] = a[tid] + b[tid]; // 仅对有效范围内的元素执行加法
    }
}

int main() {
    const int size = 1024; // 数组大小
    int *a, *b, *c;        // 主机端（CPU）数组指针
    int *d_a, *d_b, *d_c;  // 设备端（GPU）数组指针

    // 1. 在主机端分配内存并初始化数据
    a = (int *)malloc(size * sizeof(int));
    b = (int *)malloc(size * sizeof(int));
    c = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        a[i] = i;       // 数组a初始化为0,1,2,...,1023
        b[i] = i * 2;   // 数组b初始化为0,2,4,...,2046
    }

    // 2. 在设备端分配内存
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // 3. 将数据从主机复制到设备
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // 4. 定义线程块和网格大小
    int threadsPerBlock = 256; // 每个线程块256个线程
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; // 计算需要的线程块数

    // 5. 启动核函数（在GPU上执行）
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // 6. 将结果从设备复制回主机
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 7. 验证结果（检查前5个元素）
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %d (expected: %d)\n", i, c[i], a[i] + b[i]);
    }

    // 8. 释放内存
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}