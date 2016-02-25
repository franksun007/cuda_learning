C_SOURCE=matrix_multiplication.c
CU_SOURCE=matrix_multiplication.cu
C_EXEC=MM_host
CU_EXEC=MM_device

all:
	make mm_c
	make mm_cu

mm_c:
	gcc -O2 -Wall -std=c99 -g $(C_SOURCE) -o $(C_EXEC)

mm_cu:
	nvcc -g $(CU_SOURCE) -o $(CU_EXEC)

clean:
	rm $(C_EXEC) $(CU_EXEC)
