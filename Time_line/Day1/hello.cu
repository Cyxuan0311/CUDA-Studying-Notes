#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}