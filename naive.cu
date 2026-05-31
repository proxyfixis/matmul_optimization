%%writefile naive.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 4096
#define NUM_RUNS 5

// CUDA error checking
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(-1); \
    }

__global__ void multmat(float *A, float *B, float *C, int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    float value = 0.0f;

    if (i < n && j < n) {
        for (int k = 0; k < n; k++)
            value += A[i * n + k] * B[k * n + j];
        C[i * n + j] = value;
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threads(32, 32);
    dim3 blocks((N + 31) / 32, (N + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark naive kernel
    float total_ms = 0.0f;
    for (int r = 0; r < NUM_RUNS; r++) {
        cudaEventRecord(start);
        multmat<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float avg_kernel_time = total_ms / NUM_RUNS;
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // cuBLAS benchmark
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    float total_cublas = 0.0f;
    for (int r = 0; r < NUM_RUNS; r++) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_B, N, d_A, N, &beta, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_cublas += ms;
    }

    float avg_cublas_time = total_cublas / NUM_RUNS;
    double flops = 2.0 * N * N * N;
    double gflops_kernel = flops / (avg_kernel_time * 1e6);
    double gflops_cublas = flops / (avg_cublas_time * 1e6);
    double efficiency = (gflops_kernel / gflops_cublas) * 100.0;

    printf("RESULT: KernelName=Naive, AvgTime=%.3f, GFLOPS=%.2f, CuBLAS=%.2f, Efficiency=%.2f\n",
           avg_kernel_time, gflops_kernel, gflops_cublas, efficiency);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
