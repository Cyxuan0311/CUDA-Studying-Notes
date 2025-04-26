#include <cuda_runtime.h>
#include <stdio.h>

__global__ void printThreadIndex(float* a, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    // 添加边界检查
    if (ix < nx && iy < ny) {
        unsigned int idx = iy * nx + ix;
        printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d) "
               "global index %2d value %.2f\n", 
               threadIdx.x, threadIdx.y,
               blockIdx.x, blockIdx.y, ix, iy, idx, a[idx]);
    }
}

int main() {
    int nx = 8, ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // 分配主机内存
    float* A_host = (float*)malloc(nBytes);
    
    // 分配设备内存
    float* A_dev = nullptr;
    cudaMalloc((void**)&A_dev, nBytes);

    // 初始化主机数据
    for(int i = 0; i < nxy; i++) {
        A_host[i] = (float)i;  // 显式转换为float
    }

    // 将数据从主机拷贝到设备
    cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice);

    // 设置网格和块维度
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // 调用核函数
    printThreadIndex<<<grid, block>>>(A_dev, nx, ny);
    cudaDeviceSynchronize();

    // 释放内存
    free(A_host);
    cudaFree(A_dev);

    return 0;
}