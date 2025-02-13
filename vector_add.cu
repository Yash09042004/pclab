#include <stdio.h>
#include <cuda_runtime.h>

#define N 10000  // 100x100 = 10000 elements

// CUDA Kernel for Vector Addition
__global__ void vector_add(int *A, int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure within bounds
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int *A, *B, *C;         // Host pointers
    int *d_A, *d_B, *d_C;   // Device pointers

    int size = N * sizeof(int);

    // Allocate memory on host
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        A[i] = i;       // A = [0, 1, 2, ...]
        B[i] = 2 * i;   // B = [0, 2, 4, ...]
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Configure kernel launch
    int threadsPerBlock = 10;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA Kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
}
cudaDeviceSynchronize();


    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    printf("Sample Output:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, C[i]);
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
