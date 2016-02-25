#include <cuda.h>
#include <cstdlib>
#include <stdio.h>
#include <math.h>

#define DIM 4
#define MAX_THREADS 32

#define SHARED_MEM_CAPACITY (48 * 1024)
#define TILE_WIDTH 32

__global__ void matrix_multiplication
(float *A, float *B, float *C, int dim, int tile_width) {

    __shared__ float A_sub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_sub[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;

    float sum = 0;
    for (int k = 0; k < dim / tile_width; k++) {
        A_sub[ty][tx] = A[row * dim + (k * tile_width + tx)];
        B_sub[ty][tx] = B[(k * tile_width + ty) * dim + col];
        __syncthreads();

        for (int m = 0; m < tile_width; m++) {
            sum += A_sub[ty][m] * B_sub[m][tx];
        }
        __syncthreads();
    }
    C[row * dim + col] = sum;

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

    int tile_width = TILE_WIDTH;

    dim3 dimGrid(blocks, blocks);
    dim3 dimBlock(threads, threads);

    matrix_multiplication<<<dimGrid, dimBlock>>>
        (dev_A, dev_B, dev_C, dim, tile_width);

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
