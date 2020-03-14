targets: MMulti gsd2D jacobi2D val1 val2 omp2 omp3 omp4 omp5 omp6

MMulti: MMulti1.cpp
	g++ -o3 -fopenmp -march=native MMulti1.cpp -o MMulti1

gsd2D: gsd2D-omp.cpp
	g++ -std=c++11 -o3 -fopenmp gsd2D-omp.cpp -o gsd2d-omp

jacobi2D: jacobi2D-omp.cpp
	g++ -std=c++11 -o3 -fopenmp jacobi2D-omp.cpp -o jacobi2D-omp

val1: val_test01_solved.cpp
	g++ -std=c++11 val_test01_solved.cpp -o val_test01_solved

val2: val_test02_solved.cpp
	g++ -std=c++11 val_test02_solved.cpp -o val_test02_solved

omp2: omp_solved2.c
	gcc -std=c11 -fopenmp omp_solved2.c -o omp_solved2

omp3: omp_solved3.c
	gcc -std=c11 -fopenmp omp_solved3.c -o omp_solved3

omp4: omp_solved4.c
	gcc -std=c11 -fopenmp omp_solved4.c -o omp_solved4

omp5: omp_solved5.c
	gcc -std=c11 -fopenmp omp_solved5.c -o omp_solved5

omp6: omp_solved6.c
	gcc -std=c11 -fopenmp omp_solved6.c -o omp_solved6
