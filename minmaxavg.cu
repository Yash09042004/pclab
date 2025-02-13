#include <stdio.h>
#include <cuda.h>
#include <limits.h>

#define N 1000
#define THREADS_PER_BLOCK 256  

// CUDA kernel to find min, max, and sum
__global__ void computeMinMaxSum(int *arr1, int *arr2, int *minVal, int *maxVal, long long *sumVal) {
    __shared__ int minVals[THREADS_PER_BLOCK];
    __shared__ int maxVals[THREADS_PER_BLOCK];
    __shared__ long long sumVals[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Load data into shared memory
    if (tid < N) {
        minVals[local_tid] = min(arr1[tid], arr2[tid]);  
        maxVals[local_tid] = max(arr1[tid], arr2[tid]);  
        sumVals[local_tid] = (long long) arr1[tid] + arr2[tid];  
    } else {
        minVals[local_tid] = INT_MAX;
        maxVals[local_tid] = INT_MIN;
        sumVals[local_tid] = 0;
    }
    __syncthreads();

    // Parallel reduction for min, max, and sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            minVals[local_tid] = min(minVals[local_tid], minVals[local_tid + stride]);
            maxVals[local_tid] = max(maxVals[local_tid], maxVals[local_tid + stride]);
            sumVals[local_tid] += sumVals[local_tid + stride];
        }
        __syncthreads();
    }

    // Store results from each block to global memory
    if (local_tid == 0) {
        atomicMin(minVal, minVals[0]);
        atomicMax(maxVal, maxVals[0]);
        atomicAdd((unsigned long long int*)sumVal, (unsigned long long int)sumVals[0]);
    }
}

// Host function
int main() {
    int *d_A, *d_B, *d_min, *d_max;
    long long *d_sum;
    int h_A[N], h_B[N];
    int h_min = INT_MAX, h_max = INT_MIN;
    long long h_sum = 0;

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i + 1;        // 1 to 1000
        h_B[i] = N - i;        // 1000 to 1
    }

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));
    cudaMalloc(&d_sum, sizeof(long long));

    // Copy data to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(long long), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    computeMinMaxSum<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_min, d_max, d_sum);

    // Copy results back
    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);

    // Compute final average
    double avg = (double)h_sum / (2 * N);

    // Print results
    printf("Combined Min = %d\n", h_min);
    printf("Combined Max = %d\n", h_max);
    printf("Combined Avg = %.2f\n", avg);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_sum);

    return 0;
}
