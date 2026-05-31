COMPILE.md
# Compilation Instructions

This document describes how to compile and run all CUDA matrix multiplication implementations and benchmarks included in this project.

---

##  Requirements

Before compiling, ensure the following dependencies are installed:

- **CUDA Toolkit 12.x or newer**
- **NVIDIA GPU** with compute capability ≥ 7.0 (Volta or newer)
- **cuBLAS** (ships with the CUDA Toolkit)
- **gcc / g++** compatible with your CUDA version
- **Make** (optional if using the provided Makefile)

---

##  Compilation Commands

You can compile each file manually using `nvcc` or automatically using the provided `Makefile`.

### Manual Compilation

Run the following commands from the project root:

```bash
# Naive baseline kernel
nvcc -O3 -arch=sm_75 naive.cu -lcublas -o naive

# Shared memory tiled kernel
nvcc -O3 -arch=sm_75 tiled4096.cu -lcublas -o tiled4096

# Register tiled kernel
nvcc -O3 -arch=sm_75 tiledregister4096.cu -lcublas -o tiledregister4096

# Double-buffered + register tiled kernel
nvcc -O3 -arch=sm_75 doublebuffered.cu -lcublas -o doublebuffered

# Benchmark all implementations together
nvcc -O3 -arch=sm_75 benchmark.cu -o benchmark
```


 **Note:**

Replace sm_75 with your GPU’s compute architecture.
You can find it via:
``` bash
nvidia-smi
```


Running the Programs

After compilation, run each executable individually to test or benchmark:
``` bash
./naive
./tiled4096
./tiledregister4096
./doublebuffered
```


or run the benchmark to execute all and compare automatically:
```
./benchmark
```

Expected Output Format

Each executable prints a performance summary similar to:
```
==== Performance Summary ====
Matrix size: 4096 x 4096
Kernel: tiledregister4096
Average Time: 79.381 ms
Performance: 1731.39 GFLOPS (29.15% of cuBLAS)
```
