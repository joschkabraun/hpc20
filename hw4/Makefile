targets: matrix_vec jacobi2d

matrix_vec: matrix_vec.cu
	nvcc matrix_vec.cu -Xcompiler -fopenmp -o matrix_vec

jacobi2d: jacobi-2d.cu
	nvcc jacobi-2d.cu -Xcompiler -fopenmp -o jacobi-2d
