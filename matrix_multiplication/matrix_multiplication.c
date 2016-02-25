#include <stdlib.h>
#include <stdio.h>
#include <sys/timeb.h>

#define DIM 4

void matrix_multiplication(float *A, float *B, float *C, int dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                C[i * dim + j] += A[j * dim + k] * B[k * dim + i];
            }
        }
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

    float *A, *B, *C;

    struct timeb tmb;
    struct timeb tmb2;

    A = (float *) malloc(memSize);
    B = (float *) malloc(memSize);
    C = (float *) malloc(memSize);

    for (int i = 0; i < dim * dim; i++) {
        A[i] = 0.9;
        B[i] = 0.9;
    }

    ftime(&tmb);

    matrix_multiplication(A, B, C, dim);

    ftime(&tmb2);

    printf("C version\n");
    printf("The execution time is %G milliseconds\n", (double)(tmb2.time * 1000 + tmb2.millitm - tmb.time * 1000 - tmb.millitm));

    printf("\n");
    printf("Result - for sanity check\n");

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            printf("%08f\t", C[i * dim + j]);
        }
        printf("\n");
    }
}
