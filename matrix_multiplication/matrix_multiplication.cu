#include <cuda.h>
#include <cstdlib>
#include <stdio.h>
#include <math.h>

#define DIM 4
#define MAX_THREADS 32

#define SHARED_MEM_CAPACITY (48 * 1024)
#define TILE_WIDTH 32

__global__ void matrix_multiplication
(float *A, float *B, float *C, int dim) {

    // init the block index, thread index etc. 
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // fixed row and col for a specific thread in a specific block
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Allocate the TILE on the shared memory.
    __shared__ float A_sub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_sub[TILE_WIDTH][TILE_WIDTH];

    // If condition to eliminate extra or dim which dim % 32 (or 2^n) != 0
    if (by * blockDim.y + ty < dim && bx * blockDim.y + tx < dim) {

        // partial sum
        float sum = 0;
        for (int k = 0; k < dim / TILE_WIDTH + 1; k++) {
            // in case that k * TILE_WIDTH + thread > dim, 
            // we assign those values to be 0.
            // Even doing the dot product everything will be 0.
            A_sub[ty][tx] = 
                (k * TILE_WIDTH + tx) < dim ? 
                A[row * dim + (k * TILE_WIDTH + tx)] : 0;
            B_sub[ty][tx] = 
                (k * TILE_WIDTH + ty) < dim ?
                B[(k * TILE_WIDTH + ty) * dim + col] : 0;
            // Wait until all the threads finish doing that 
            __syncthreads();

            // At this point, all of the TILES need for the 
            // target tile are loaded to shared mem.

            // The sum will be the cumulated sum for the 
            // specific thread it's computing
            for (int m = 0; m < TILE_WIDTH; m++) {
                sum += A_sub[ty][m] * B_sub[m][tx];
            }
            // Wait until all the threads finish so that 
            // the shared mem can be flashed
            __syncthreads();
        }
        // classic [][]
        C[row * dim + col] = sum;
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
    
    // If dim < MAX_THREADS, then the threads we using can be 
    // the dim itself. Otherwise, threads should be max threads
    // in order to use 1024 threads per block. 
    int threads = dim < MAX_THREADS ? dim : MAX_THREADS;

    // Calculate the number of blocks that is necessary for 
    // the calculation
    int blocks = dim * dim / threads / threads + 1;
    
    // Figure out the square-like block geometry
    // (which shouldn't be matter too much, but for simplicity)
    int block_x = (int) sqrt(blocks);
    int block_y = blocks / block_x;
    if (block_x * block_y < blocks) {
        block_x++;
    }

    dim3 dimGrid(block_x, block_y);
    dim3 dimBlock(threads, threads);

    matrix_multiplication<<<dimGrid, dimBlock>>>
        (dev_A, dev_B, dev_C, dim);

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
