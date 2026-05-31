%%writefile tiled4096.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 4096
#define TILE 32    // Tile size (use 16, 32 for best performance)
#define NUM_RUNS 5

// -----------------------------------------------------------------------------
// Tiled matrix multiplication kernel using shared memory
// -----------------------------------------------------------------------------
__global__ void matmul_tiled(float *A, float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float value = 0.0f;

    // Loop over tiles
    for (int t = 0; t < n / TILE; t++) {
        // Load one tile from A and B into shared memory
        As[ty][tx] = A[row * n + (t * TILE + tx)];
        Bs[ty][tx] = B[(t * TILE + ty) * n + col];

        __syncthreads(); // Wait for all threads in the block

        // Compute partial results
        
        for (int k = 0; k < TILE; k++) {
            value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads(); // Make sure all threads done before next tile
    }

    // Write result to global memory
    C[row * n + col] = value;
}

// -----------------------------------------------------------------------------
// CUDA error checking macro
// -----------------------------------------------------------------------------
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        return -1; \
    }

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize matrices
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

    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------------------------------------------------------------
    //  Benchmark custom tiled CUDA kernel
    // -------------------------------------------------------------------------
    float total_ms_kernel = 0.0f;
    for (int r = 0; r < NUM_RUNS; r++) {
        cudaEventRecord(start);
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaError_t err = cudaGetLastError();
        CUDA_CHECK(err);
        err = cudaDeviceSynchronize();
        CUDA_CHECK(err);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms_kernel += ms;
    }
    float avg_kernel_time = total_ms_kernel / NUM_RUNS;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    printf("Tiled Kernel Avg Time: %.3f ms\n", avg_kernel_time);
    printf("h_C[0] = %f, h_C[N*N-1] = %f\n", h_C[0], h_C[N*N - 1]);

    // -------------------------------------------------------------------------
    //  Benchmark cuBLAS SGEMM
    // -------------------------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float total_ms_cublas = 0.0f;
    for (int r = 0; r < NUM_RUNS; r++) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_B, N,
                    d_A, N,
                    &beta,
                    d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms_cublas += ms;
    }
    float avg_cublas_time = total_ms_cublas / NUM_RUNS;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    printf("cuBLAS Avg Time: %.3f ms\n", avg_cublas_time);
    printf("cuBLAS h_C[0] = %f, h_C[N*N-1] = %f\n", h_C[0], h_C[N*N - 1]);

    // -------------------------------------------------------------------------
    //  Compute GFLOPS and Efficiency
    // -------------------------------------------------------------------------
    double flops = 2.0 * N * N * N;
    double gflops_kernel = flops / (avg_kernel_time * 1e6);
    double gflops_cublas = flops / (avg_cublas_time * 1e6);
    double efficiency = (gflops_kernel / gflops_cublas) * 100.0;

    printf("RESULT: KernelName=Tiled, AvgTime=%.3f, GFLOPS=%.2f, CuBLAS=%.2f, Efficiency=%.2f\n",
       avg_kernel_time, gflops_kernel, gflops_cublas, efficiency);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
