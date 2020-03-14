#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <math.h>

#ifdef _OPENMP
  #include <omp.h>
  #define USE_OPENMP_FOR omp parallel for
#else
  #define USE_OPENMP_FOR
#endif

double l2_norm_shift(double const* u, double shift, long n) {
  double accum = 0.;
  for (long i = 1; i < n-1; i++) {
    for (long j=1; j< n-1; j++) {
      accum += (u[n*i+j] - shift) * (u[n*i+j] - shift);
    }
  }
  return sqrt(accum);
}

void calculate_A_times_u(const double *u, double *res, long N, double h) {
    res[0] = 0;
    
    for (long i = 1; i < N-1; i++) {
      for (long j=1; j< N-1; j++) {  
        res[N*i+j] = (4 * u[N*i+j] - u[N*(i-1)+j] - u[N*(i+1)+j] - u[N*i+j+1] - u[N*i+j-1]) / (h * h);
      }
    }
    
    res[N*(N-1) + N-1] = 0;
}

int main(int argc, char *argv[]) {
  // the utils.h was not compiled
  long N = read_option<long>("-n",argc,argv);  
  //long it = read_option<long>("-it",argc,argv);
  long it = 26000;
  double h = 1. / (N+1.);
  

  double* u_0 = (double*) calloc(N*N, sizeof(double));
  double* u_1 = (double*) calloc(N*N, sizeof(double));
  double* A_u = (double*) malloc(N*N*sizeof(double)); 
 
  Timer t;
  
  t.tic();
  printf("iteration     residue\n");
  printf("---------------------\n");
  for (long k=1; k < it+1; k++) {
    memcpy(u_0, u_1, N*N*sizeof(double));

    #pragma USE_OPENMP_FOR   
    for (long i=1; i < N-1; i++) {
      for (long j=1; j < N-1; j++) {
        u_1[N*i + j] = (h*h + u_0[N*(i-1)+j] + u_0[N*i+j-1] + u_0[N*(i+1)+j] + u_0[N*i + j+1]) / 4.0; 
      }
    }

    if ((k % 2000) == 0) {
      calculate_A_times_u(u_1, A_u, N, h);
      double res = l2_norm_shift(A_u, 1., N);
      printf(" %ld        %f\n",k, res);
    }
  }
  double time = t.toc();
  printf("%f seconds per iteration\n", time/it);

  free(u_0);
  free(u_1);
  free(A_u);

  return 0;
}
