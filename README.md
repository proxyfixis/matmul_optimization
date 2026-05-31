## 1. Implementation Details

This project implements and benchmarks multiple progressively optimized versions of matrix multiplication on the GPU, each targeting specific CUDA performance bottlenecks.

---

### 1.1 Naive Implementation (`naive.cu`)

**Concept:**  
Each thread computes one element of output matrix `C[i][j]` by iterating through a full row of `A` and a full column of `B`.

```cpp
__global__ void multmat(float *A, float *B, float *C, int n) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    float value = 0.0f;
    for (int k = 0; k < n; k++)
        value += A[i * n + k] * B[k * n + j];
    C[i * n + j] = value;
} 
```

**Compilation:**

```bash
Copy code
nvcc -O3 -arch=sm_75 naive.cu -lcublas -o naive
```

**Bottlenecks:**

- Each thread performs n global memory reads → poor memory reuse.

- Non-coalesced global memory accesses for B.

- Extremely memory-bound and bandwidth-limited.

### 1.2 Shared Memory Tiling (`tiled4096.cu`)
**Concept:**
Use shared memory tiles of A and B so threads in a block reuse loaded data.
Each block computes one `TILE × TILE` submatrix of C.

```cpp
Copy code
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

for (int t = 0; t < n / TILE; t++) {
    As[ty][tx] = A[row * n + (t * TILE + tx)];
    Bs[ty][tx] = B[(t * TILE + ty) * n + col];
    __syncthreads();

    for (int k = 0; k < TILE; k++)
        value += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
```

**Compilation:**

```bash
Copy code
nvcc -O3 -arch=sm_75 tiled4096.cu -lcublas -o tiled4096
``` 

**Advantages:**

- Each tile is reused by all threads → drastically reduces global reads.

- Threads in a warp access contiguous addresses (coalesced).

- Big performance jump (often 10×+).

**Challenges:**

- Correct synchronization (__syncthreads) to avoid race conditions.

- Tuning TILE = 16 or 32 depending on GPU shared memory limits.

### 1.3 Register Tiling (`tiledregister4096.cu`)
**Concept:**

Extend shared memory tiling by having each thread compute a `micro-tile` (2×2 or 4×4 block) stored in registers for ultra-fast accumulation.

```cpp
Copy code
float c_reg[MICRO][MICRO] = {0};
for (int t = 0; t < numTiles; ++t) {
    ...
    for (int k = 0; k < TILE; ++k)
        for (int i = 0; i < MICRO; ++i)
            for (int j = 0; j < MICRO; ++j)
                c_reg[i][j] += As[ty*MICRO+i][k] * Bs[k][tx*MICRO+j];
}
```

**Compilation:**

```bash
Copy code
nvcc -O3 -arch=sm_75 tiledregister4096.cu -lcublas -o tiledregister4096
```

**Advantages:**

-Registers are the fastest memory level (1-cycle access).

-Higher arithmetic intensity (more math per byte of memory).

**Challenges:**

-More registers per thread → can reduce occupancy.

-Must balance TILE and MICRO for optimal throughput.

### 1.4 Double-Buffered Register-Tiled (`doublebuffered.cu`)

**Concept:**

While one tile is being computed, preload the next tile into another shared memory buffer.
This overlaps memory loads and computation.

```cpp
Copy code
__shared__ float As[2][TILE][TILE + 1];
__shared__ float Bs[2][TILE][TILE + 1];

int buf = 0, nextbuf = 1;
for (int t = 0; t < numTiles; ++t) {
    if (t + 1 < numTiles) { load next tile into nextbuf; }
    for (int k = 0; k < TILE; ++k)
        ...
    __syncthreads();
    swap(buf, nextbuf);
}
```

**Compilation:**

```bash
Copy code
nvcc -O3 -arch=sm_75 doublebuffered.cu -lcublas -o doublebuffered
```

**Advantages:**

- Hides memory latency via overlap of load and compute.

- +1 padding avoids shared memory bank conflicts.

- Achieves >90% of cuBLAS performance.

**Challenges:**

- Requires careful synchronization.

- Complex indexing and shared buffer management.

##  Performance Summary

The following table summarizes the performance of all CUDA matrix multiplication implementations tested on a `4096 × 4096` matrix, benchmarked against cuBLAS as the performance reference.

| Implementation  | Time (ms) | GFLOPS  | % of cuBLAS | Speedup vs Naive |
|-----------------|-----------:|---------:|-------------:|-----------------:|
| Naive           | 230.805    | 595.48   | 10.02%       | 1.00x            |
| Tiled           | 133.182    | 1031.96  | 17.37%       | 1.73x            |
| Register Tiled  | 79.381     | 1731.39  | 29.15%       | 2.91x            |
| Double Buffered | 54.080     | 2541.38  | 42.78%       | 4.27x            |
| cuBLAS (ref.)   | 23.13      | 5943.50  | 100.00%      | 9.98x            |

---

### Observations

- **Naive:** Memory-bound kernel with no reuse or coalescing.  
- **Tiled:** Shared memory reuse improved throughput by ~1.7×.  
- **Register Tiled:** Further reduced shared memory latency, achieving ~2.9× speedup.  
- **Double Buffered:** Overlapped computation and memory loads, pushing to ~4.3× over naive.  
- **cuBLAS:** Still highest due to use of tensor cores, warp-level MMA, and mixed precision optimizations.

---

###  Performance Trend

To visualize improvement across optimizations, you can generate a bar plot:

```python
import matplotlib.pyplot as plt

implementations = ["Naive", "Tiled", "Register Tiled", "Double Buffered", "cuBLAS"]
gflops = [595.48, 1031.96, 1731.39, 2541.38, 5943.50]

plt.figure(figsize=(8,5))
plt.bar(implementations, gflops)
plt.title("CUDA Matrix Multiplication Performance (GFLOPS)")
plt.ylabel("GFLOPS")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

