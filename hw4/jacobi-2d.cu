#include <omp.h>
#include <math.h>
#include <stdio.h>

void jacobi_cpu(double* u, const double* u0, double h, long N) {
  #pragma omp parallel for schedule(static)
  for (long i=1; i<N-1; i++) {
    for (long j=1; j<N-1; j++) {
      u[N*i + j] = (h*h + u0[N*(i-1)+j] + u0[N*(i+1)+j] + u0[N*i+j+1] + u0[N*i+j-1]) / 4.0;
    }
  }
}

void calculate_A_times_u(const double *u, double *res, long N, double h) {
  res[0] = 0;
  
  #pragma omp parallel for schedule(static)
  for (long i = 1; i < N-1; i++) {
    for (long j=1; j< N-1; j++) {  
      res[N*i+j] = (4 * u[N*i+j] - u[N*(i-1)+j] - u[N*(i+1)+j] - u[N*i+j+1] - u[N*i+j-1]) / (h * h);
    }
  }
    
  res[N*(N-1) + N-1] = 0;
}

double l2_norm_shift(double const* u, double shift, long n) {
  double accum = 0.;
  for (long i = 1; i < n-1; i++) {
    for (long j=1; j< n-1; j++) {
      accum += (u[n*i+j] - shift) * (u[n*i+j] - shift);
    }
  }
  return sqrt(accum);
}

__global__ void jacobi_kernel(double* u, const double* u0, double h, long N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx = N*i + j
  
  int j = idx % N;
  int i = (idx - j) / N;
  
  u[idx] = (h*h + u0[N*(i+1)+j] + u0[N*(i-1)+j] + u0[idx-1] + u0[idx+1]) / 4.0;
}

__global__ void A_u_kernel(double* res, const double* u, long N, double h) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx = N*i + j
  int j = idx % N;
  int i = (idx - j) / N;
  
  if (idx == 0) res[0] = 0.0;
  else if (idx == N*(N-1) + N-1) res[N*(N-1) + N-1] = 0.0;
  else res[idx] = (4*u[idx] - u[idx+1] - u[idx-1] - u[N*(i-1)+j] - u[N*(i+1)+j]) / (h*h);
}

int main() {
  long N = 512;
  long it = 10*3000;
  const long BLOCK_SIZE = 1024;
  double h = 1./(N+1.);
  
  double *u, *u_0, *A_u;
  cudaMallocHost((void**) &u, N*N*sizeof(double));
  cudaMallocHost((void**) &u_0, N*N*sizeof(double));
  cudaMallocHost((void**) &A_u, N*N*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i=0; i<N*N; i++) {
    u[i] = 0.0;
    u_0[i] = 0.0;
  }
  
  double *v_0, *v, *A_v;
  cudaMalloc(&v_0, N*N*sizeof(double));
  cudaMalloc(&v, N*N*sizeof(double));
  cudaMalloc(&A_v, N*N*sizeof(double));
  cudaMemcpyAsync(v_0, u_0, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(v, u, N*N*sizeof(double), cudaMemcpyHostToDevice);

  double tt = omp_get_wtime();
  printf("CPU Jacobi2D\n");
  printf("iteration     residue\n");
  printf("---------------------\n");
  for (long k=1; k<1; k++) {
    cudaMemcpy(u_0, u, N*N*sizeof(double), cudaMemcpyHostToHost);
    jacobi_cpu(u, u_0, h, N);
    
    if ((k % 3000) == 0) {
      calculate_A_times_u(u, A_u, N, h);
      double res = l2_norm_shift(A_u, 1., N);
      printf(" %ld      %f\n", k, res);
    }
  }
  printf("CPU: %f it/s\n\n\n\n", 1.0*it / (omp_get_wtime()-tt));
  
  double *res_d;
  cudaMalloc(
  tt = omp_get_wtime();
  printf("GPU Jacobi2D\n");
  printf("iteration     residue\n");
  printf("---------------------\n");
  for (long k=1; k<it+1; k++) {
    printf("hi");
    cudaMemcpy(v_0, v, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
    printf("hi2");
    jacobi_kernel<<<N*N/BLOCK_SIZE, BLOCK_SIZE>>>(v, v_0, h, N);
    printf("hi3\n");
    printf("%ld\n",k);
    
    if ((k % 3000) == 0) {
      A_u_kernel<<<N*N/BLOCK_SIZE, BLOCK_SIZE>>>(A_v, v, N, h);
      double res = l2_norm_shift(A_v, 1., N);   // do this as in other task -> inner vector product
      printf(" %ld      %f\n", k, res);
    }
  }
  printf("GPU: %f it/s\n", 1.0*it / (omp_get_wtime()-tt));

  return 0;
}
