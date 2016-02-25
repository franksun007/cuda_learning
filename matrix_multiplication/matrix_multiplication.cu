#include <cuda.h>
#include <cstdlib>
#include <stdio.h>

#define DIM 4
#define MAX_THREADS 32

__global__ void matrix_multiplication(float *A, float *B, float *C, int dim) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < dim && j < dim) {
        float dot = 0;
        for (int k = 0; k < dim; k++) {
            dot += A[i * dim + k] * B[k * dim + j];
        }
        C[i * dim + j] = dot;
    }
}

int main(int argc, char **argv) {


    int dim;
    if (argc == 2) {
        dim = atoi(argv[1]);
    } else {
        dim = DIM;
    }
    
    int memSize = dim * dim * sizeof(float);

    float *host_A, *host_B, *host_C;
    float *dev_A, *dev_B, *dev_C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    /*
       didn't use
       dim3 dimBlock(1024, 1024);
       dim3 dimGrid(1,1);
     */

    host_A = (float *) malloc(memSize);
    host_B = (float *) malloc(memSize);
    host_C = (float *) malloc(memSize);

    cudaMalloc(&dev_A, memSize);
    cudaMalloc(&dev_B, memSize);
    cudaMalloc(&dev_C, memSize);

    for (int i = 0; i < dim * dim; i++) {
        host_A[i] = 0.9;
        host_B[i] = 0.9;
    }

    cudaMemcpy(dev_A, host_A, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, memSize, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    
    int threads = dim < MAX_THREADS ? dim : MAX_THREADS;
    int blocks = dim <= MAX_THREADS ? 1 : dim / threads;

    dim3 dimGrid(blocks, blocks);
    dim3 dimBlock(threads, threads);

    matrix_multiplication<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, dim);


    cudaMemcpy(host_C, dev_C, memSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Cuda version\n");
    printf("The execution time is %f milliseconds\n", milliseconds);

    printf("\n");
    printf("Result - for sanity check\n");

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            printf("%08f\t", host_C[i * dim + j]);
        }
        printf("\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(host_A);
    free(host_B);
    free(host_C);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    return 0;
}
