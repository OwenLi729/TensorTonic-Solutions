#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N) {
        if(input[i] >= 0) {
            output[i] = input[i];
        } else {
            output[i] = alpha * input[i];
        }
    }
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}