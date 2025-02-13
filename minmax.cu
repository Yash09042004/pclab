#include <stdio.h>
#include <cuda.h>

__global__ void findMinMax(int *input, int *min_vals, int *max_vals, int n) {
    extern __shared__ int shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n) {
        shared[tid] = input[idx];
        shared[blockDim.x + tid] = input[idx];
    } else {
        shared[tid] = INT_MAX;
        shared[blockDim.x + tid] = INT_MIN;
    }
    __syncthreads();

    // Parallel reduction for min and max
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = min(shared[tid], shared[tid + stride]);
            shared[blockDim.x + tid] = max(shared[blockDim.x + tid], shared[blockDim.x + tid + stride]);
        }
        __syncthreads();
    }

    // Write results for this block to global memory
    if (tid == 0) {
        min_vals[blockIdx.x] = shared[0];
        max_vals[blockIdx.x] = shared[blockDim.x];
    }
}

int main() {
    const int arraySize = 100;
    int h_input[arraySize];
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = i + 1;
    }

    int *d_input, *d_min_vals, *d_max_vals;
    int threads = 32;
    int numBlocks = (arraySize + threads-1) / threads;
    int sharedMemSize = 2 * threads * sizeof(int);

    cudaMalloc(&d_input, arraySize * sizeof(int));
    cudaMalloc(&d_min_vals, numBlocks * sizeof(int));
    cudaMalloc(&d_max_vals, numBlocks * sizeof(int));

    cudaMemcpy(d_input, h_input, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    findMinMax<<<numBlocks, threads, sharedMemSize>>>(d_input, d_min_vals, d_max_vals, arraySize);

    int h_min_vals[numBlocks], h_max_vals[numBlocks];
    cudaMemcpy(h_min_vals, d_min_vals, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_vals, d_max_vals, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    int min_val = h_min_vals[0];
    int max_val = h_max_vals[0];
    for (int i = 1; i < numBlocks; i++) {
        min_val = min(min_val, h_min_vals[i]);
        max_val = max(max_val, h_max_vals[i]);
    }

    printf("Minimum value: %d\n", min_val);
    printf("Maximum value: %d\n", max_val);

    cudaFree(d_input);
    cudaFree(d_min_vals);
    cudaFree(d_max_vals);

    return 0;
}