#include <stdio.h>
#include <algorithm>
#include <omp.h>

void vec_mul(double* sum_ptr, const double* a, const double* b, long N) {
  double sum = 0.0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i=0; i<N; i++) sum += a[i]*b[i];
  *sum_ptr = sum;
}

__global__ void vec_el_mul_kernel(double* c, const double* a, const double* b, long N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] * b[idx];
}

#define BLOCK_SIZE 1024

__global__ void vec_dot_kernel(double* sum, const double* a, const double* b, long N) {
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx] * b[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

__global__ void reduction_kernel(double* sum, const double*a, long N) {
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  // each thread reads data from global into shared memory
  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;
  __syncthreads();

  for(int s = 1; s < blockDim.x; s *= 2) {
    if(threadIdx.x % (2*s) == 0)
      smem[threadIdx.x] += smem[threadIdx.x + s];
     __syncthreads();
  }

  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

int main() {
  long N = 4*1024*1024;
  printf("%ld\n", N);

  double *x, *y;
  cudaMallocHost((void**) &x, N*sizeof(double));
  cudaMallocHost((void**) &y, N*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i=0; i<N; i++) {
    x[i] = 1.0/(i+1.0);
    y[i] = 2.0/(i+2.0);
  }
  
  double sum_ref, sum;
  double tt = omp_get_wtime();
  vec_mul(&sum_ref, x, y, N);
  printf("CPU Bandwidth %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *sum_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/BLOCK_SIZE; i>1; i = (i+BLOCK_SIZE-1)/BLOCK_SIZE) N_work += i;
  cudaMalloc(&sum_d, N_work*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  
  //double* sum_d = z_d;
  long Nb = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
  vec_dot_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, x_d, y_d, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/BLOCK_SIZE;
    reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d+N, sum_d, N);
    sum_d += N;
  }

  cudaMemcpyAsync(&sum, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwith = %f GB/s\n", 1*N*sizeof(double)/(omp_get_wtime()-tt)/1e9);
  printf("Error = %f\n", fabs(sum-sum_ref));

  // matrix vector mult
  long M = 512;

  double *A, *u, *B, *v;
  cudaMallocHost((void**) &A, M*N*sizeof(double));
  cudaMallocHost((void**) &u, N*sizeof(double));
  cudaMalloc(&B, M*N*sizeof(double));
  cudaMalloc(&v, N*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i=0; i<M*N; i++) A[i] = 1.0/(i+1.0);
  for (long j=0; j<N; j++) y[j] = 2.0/(i+3.0);
  

  cudaMemcpyAsync(B, A, M*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(v, u, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  


  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(sum_d);
  cudaFreeHost(x);

  return 0;
}
