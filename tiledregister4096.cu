
%%writefile tiledregister4096.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 4096
#define BLOCK 16          
#define MICRO 2           
#define TILE (BLOCK * MICRO) 
#define NUM_RUNS 5


// Tiled matrix multiplication kernel using shared memory + register tiling
// Each thread computes a MICRO x MICRO sub-block of C (stored in registers)

__global__ void matmul_reg_tiled(const float *A, const float *B, float *C, int n) {
   
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x; 
    int ty = threadIdx.y; 

  // Starting row and col for this thread's MICROxMICRO micro-tile in the global matrix
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row0 = blockRow * TILE + ty * MICRO; // top row of this thread's micro-tile
    int col0 = blockCol * TILE + tx * MICRO; // left col of this thread's micro-tile

    // Register accumulation for MICRO x MICRO outputs
    float c_reg[MICRO][MICRO];
    #pragma unroll
    for (int i = 0; i < MICRO; ++i)
        for (int j = 0; j < MICRO; ++j)
            c_reg[i][j] = 0.0f;

    int numTiles = n / TILE; 

    for (int t = 0; t < numTiles; ++t) {
        // Each thread loads MICRO x MICRO elements for A and B into shared memory
        // Compute the offsets inside the shared tile
        int shared_row_base = ty * MICRO;
        int shared_col_base = tx * MICRO;
        int a_row_base = row0;
        int a_col_base = t * TILE + tx * MICRO;
        int b_row_base = t * TILE + ty * MICRO;
        int b_col_base = col0;

        #pragma unroll
        for (int i = 0; i < MICRO; ++i) {
            #pragma unroll
            for (int j = 0; j < MICRO; ++j) {
                // Load A: global A[(a_row_base + i) * n + (a_col_base + j)] -> As[shared_row_base + i][shared_col_base + j]
                As[shared_row_base + i][shared_col_base + j] =
                    A[(a_row_base + i) * n + (a_col_base + j)];

                // Load B: global B[(b_row_base + i) * n + (b_col_base + j)] -> Bs[shared_row_base + i][shared_col_base + j]
                Bs[shared_row_base + i][shared_col_base + j] =
                    B[(b_row_base + i) * n + (b_col_base + j)];
            }
        }

        __syncthreads();

        // Compute on the loaded TILE: iterate k across TILE dimension
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            // Load a MICRO-length column of A (for the rows of this thread's micro-tile)
            float a_vals[MICRO];
            #pragma unroll
            for (int i = 0; i < MICRO; ++i) {
                a_vals[i] = As[shared_row_base + i][k];
            }
            // Load a MICRO-length row of B (for the cols of this thread's micro-tile)
            float b_vals[MICRO];
            #pragma unroll
            for (int j = 0; j < MICRO; ++j) {
                b_vals[j] = Bs[k][shared_col_base + j];
            }

            // Rank-1 update into the MICROxMICRO register block
            #pragma unroll
            for (int i = 0; i < MICRO; ++i) {
                #pragma unroll
                for (int j = 0; j < MICRO; ++j) {
                    c_reg[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // Write the MICROxMICRO register results back to global memory
    #pragma unroll
    for (int i = 0; i < MICRO; ++i) {
        #pragma unroll
        for (int j = 0; j < MICRO; ++j) {
            int global_r = row0 + i;
            int global_c = col0 + j;
            // Bounds check (safe if N divisible by TILE, but kept for safety)
            if (global_r < n && global_c < n) {
                C[global_r * n + global_c] = c_reg[i][j];
            }
        }
    }
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
    size_t size = (size_t)N * (size_t)N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        printf("Host allocation failed\n");
        return -1;
    }

    // Initialize matrices (simple test pattern)
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK, BLOCK);
    dim3 blocks(N / TILE, N / TILE); // assume N divisible by TILE

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------------------------------------------------------------------------
    // Benchmark custom register-tiled CUDA kernel
    // -------------------------------------------------------------------------
    float total_ms_kernel = 0.0f;
    for (int r = 0; r < NUM_RUNS; ++r) {
        cudaEventRecord(start);
        matmul_reg_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
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
    printf("Register-Tiled Kernel Avg Time: %.3f ms\n", avg_kernel_time);
    printf("h_C[0] = %f, h_C[N*N-1] = %f\n", h_C[0], h_C[N*N - 1]);

    // -------------------------------------------------------------------------
    // Benchmark cuBLAS SGEMM (for reference)
    // -------------------------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float total_ms_cublas = 0.0f;
    for (int r = 0; r < NUM_RUNS; ++r) {
        cudaEventRecord(start);
        // note: cublasSgemm uses column-major by default; to compare with our
        // row-major data, we swap A and B and transpose arguments (equivalent).
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
    printf("cuBLAS h_C[0] = %f, cuBLAS h_C[N*N-1] = %f\n", h_C[0], h_C[N*N - 1]);

    // -------------------------------------------------------------------------
    // Compute GFLOPS and Efficiency
    // -------------------------------------------------------------------------
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops_kernel = flops / (avg_kernel_time * 1e6);
    double gflops_cublas = flops / (avg_cublas_time * 1e6);
    double efficiency = (gflops_kernel / gflops_cublas) * 100.0;

    printf("RESULT: KernelName=RegisterTiled, AvgTime=%.3f, GFLOPS=%.2f, CuBLAS=%.2f, Efficiency=%.2f\n",
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
