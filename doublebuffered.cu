%%writefile doublebuffered.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 4096
#define TILE 32        // shared tile dimension (32 is a common sweet spot)
#define SUB 4          // register tile per thread (each thread computes SUB x SUB outputs) ( MICRO)
#define NUM_RUNS 5

#if (TILE % SUB) != 0
#error "TILE must be divisible by SUB"
#endif

#define TPB (TILE / SUB)   // threads per block dimension

// Double-buffered shared memory with +1 pad to reduce bank conflicts.
__global__ void matmul_tiled_reg_db_v4(const float * __restrict__ A,
                                       const float * __restrict__ B,
                                       float * __restrict__ C,
                                       int n) {
    __shared__ float As[2][TILE][TILE + 1];
    __shared__ float Bs[2][TILE][TILE + 1];

    int tx = threadIdx.x; // 0 .. TPB-1
    int ty = threadIdx.y; // 0 .. TPB-1

    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;

    int row0 = block_row + ty * SUB;
    int col0 = block_col + tx * SUB;

    // register accumulators (SUB x SUB). For SUB=4, unroll explicitly
    float acc[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; ++i)
        #pragma unroll
        for (int j = 0; j < SUB; ++j)
            acc[i][j] = 0.0f;

    int numTiles = n / TILE;
    if (numTiles == 0) return;

    int shared_row = ty * SUB;
    int shared_col = tx * SUB;

    // buffer indices
    int buf = 0;
    int nextbuf = 1;

    // --- Prefetch tile 0 into buf ---
    // We'll load SUB contiguous elements per thread using float4 when possible.
    // Each thread loads SUB contiguous columns of A (row-major) and SUB contiguous columns of B rows.
    // Use float4 trick: SUB must be multiple of 4 for full vectorization — here SUB=4.
    #pragma unroll
    for (int i = 0; i < SUB; ++i) {
        // A: load 4 contiguous elements from A[row0 + i][ t*TILE + shared_col + 0..3 ]
        int a_row = row0 + i;
        int a_col_base = 0 * TILE + shared_col;
        const float* a_ptr = A + (size_t)a_row * n + a_col_base;
        // load as float4 (aligned since N is multiple of 4)
        float4 a4 = *((const float4*)a_ptr);
        // store into shared memory
        #pragma unroll
        for (int j = 0; j < SUB; ++j)
            As[buf][shared_row + i][shared_col + j] = ((float*)&a4)[j];

        // B: load 4 contiguous elements from B[b_row ..][col0 + j] where b_row varies
        int b_row_base = 0 * TILE + shared_row + i;
        float4 b4 = *((const float4*)(B + (size_t)b_row_base * n + col0));
        #pragma unroll
        for (int j = 0; j < SUB; ++j)
            Bs[buf][shared_row + i][shared_col + j] = ((float*)&b4)[j];
    }

    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        // Start loading next tile (t+1) into nextbuf if exists — do scalar/vec loads without __syncthreads
        if (t + 1 < numTiles) {
            #pragma unroll
            for (int i = 0; i < SUB; ++i) {
                int a_row = row0 + i;
                int a_col_base = (t + 1) * TILE + shared_col;
                const float* a_ptr = A + (size_t)a_row * n + a_col_base;
                float4 a4 = *((const float4*)a_ptr);
                #pragma unroll
                for (int j = 0; j < SUB; ++j)
                    As[nextbuf][shared_row + i][shared_col + j] = ((float*)&a4)[j];

                int b_row_base = (t + 1) * TILE + shared_row + i;
                float4 b4 = *((const float4*)(B + (size_t)b_row_base * n + col0));
                #pragma unroll
                for (int j = 0; j < SUB; ++j)
                    Bs[nextbuf][shared_row + i][shared_col + j] = ((float*)&b4)[j];
            }
        }

        // Compute with current 'buf'
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            // load SUB elements of A column from shared into regs
            float a_vals[SUB];
            #pragma unroll
            for (int i = 0; i < SUB; ++i)
                a_vals[i] = As[buf][shared_row + i][k];

            // load SUB elements of B row from shared into regs
            float b_vals[SUB];
            #pragma unroll
            for (int j = 0; j < SUB; ++j)
                b_vals[j] = Bs[buf][k][shared_col + j];

            // rank-1 update
            #pragma unroll
            for (int i = 0; i < SUB; ++i)
                #pragma unroll
                for (int j = 0; j < SUB; ++j)
                    acc[i][j] = fmaf(a_vals[i], b_vals[j], acc[i][j]);
        }

        // barrier: ensure nextbuf loads are fully visible before next iteration uses nextbuf
        __syncthreads();

        // swap buffers
        buf = 1 - buf;
        nextbuf = 1 - nextbuf;
    }

    // Write back SUBxSUB results
    #pragma unroll
    for (int i = 0; i < SUB; ++i) {
        #pragma unroll
        for (int j = 0; j < SUB; ++j) {
            int gr = row0 + i;
            int gc = col0 + j;
            if (gr < n && gc < n) {
                C[(size_t)gr * n + gc] = acc[i][j];
            }
        }
    }
}

// Error macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        return -1; \
    }

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes), *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) { printf("malloc fail\n"); return -1; }

    for (int i = 0; i < N * N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; h_C[i] = 0.0f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes)); CUDA_CHECK(cudaMalloc(&d_B, bytes)); CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    dim3 threads(TPB, TPB);
    dim3 blocks(N / TILE, N / TILE);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int r = 0; r < NUM_RUNS; ++r) {
        cudaEventRecord(start);
        matmul_tiled_reg_db_v4<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float avg = total_ms / NUM_RUNS;
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    printf("Improved DB+Reg (SUB=%d) Avg Time: %.3f ms\n", SUB, avg);
    // cuBLAS for reference
    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;
    float total_cublas = 0.0f;
    for (int r = 0; r < NUM_RUNS; ++r) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_cublas += ms;
    }
    float avg_cublas = total_cublas / NUM_RUNS;
    printf("cuBLAS Avg Time: %.3f ms\n", avg_cublas);

    double flops = 2.0 * (double)N * N * N;
    double gk = flops / (avg * 1e6), gb = flops / (avg_cublas * 1e6);
    printf("RESULT: KernelName=DoubleBuffered, AvgTime=%.3f, GFLOPS=%.2f, CuBLAS=%.2f, Efficiency=%.2f\n",
       avg, gk, gb, (gk / gb) * 100.0);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
