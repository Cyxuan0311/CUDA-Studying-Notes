#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

__global__ void vectorAdd(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int size = 1024;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    // 1. 分配并初始化主机内存
    a = (int *)malloc(size * sizeof(int));
    b = (int *)malloc(size * sizeof(int));
    c = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // 2. 分配设备内存
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // 3. 拷贝数据到设备
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // 4. 启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // 5. 检查核函数执行
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 6. 拷贝结果回主机
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 7. 验证结果
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %d (expected: %d)\n", i, c[i], 3 * i);
        assert(c[i] == 3 * i);  // 验证结果是否正确
    }

    // 8. 释放内存
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    printf("Success!\n");
    return 0;
}