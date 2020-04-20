#include <stdio.h>
#include <algorithm>
#include <omp.h>
#include <math.h>

double dist(double const* u, double const* v, long N) {
  double accum = 0.;
  for (long i=0; i<N; i++) {
    accum += (u[i]-v[i]) * (u[i]-v[i]);
  }
  return sqrt(accum);
}

void vec_mul(double* sum_ptr, const double* a, const double* b, long N) {
  double sum = 0.0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i=0; i<N; i++) sum += a[i]*b[i];
  *sum_ptr = sum;
}

void mat_vec_mul(double* A_u, const double* A, const double* u, long M, long N) {
  for (long i=0; i< M; i++) {
    double Au_ij = 0.;
    for (long j=0; j< N; j++) {
      Au_ij += A[i*N+j] * u[j];
    }
    A_u[i] = Au_ij;
  }
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
  long N = 4194304;
  //printf("%ld\n", N);

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

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(sum_d);
  cudaFreeHost(x);
  

  // matrix vector mult
  //N = 82944;
  N=82944;
  long M = 512;

  N_work = 1;
  for (long i=(1024+BLOCK_SIZE-1)/BLOCK_SIZE; i>1; i = (i+BLOCK_SIZE-1)/BLOCK_SIZE) N_work += i;

  double *A, *u, *B, *v, *entries, *entries_host, *A_u, *A_v;
  cudaMallocHost((void**) &A, M*N*sizeof(double));
  cudaMallocHost((void**) &u, N*sizeof(double));
  cudaMalloc(&B, M*N*sizeof(double));
  cudaMalloc(&v, N*sizeof(double));
  cudaMalloc(&entries, M*N_work*sizeof(double));
  cudaMallocHost((void**) &entries_host, M*N_work*sizeof(double));
  cudaMallocHost((void**) &A_u, M*sizeof(double));
  cudaMallocHost((void**) &A_v, M*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i=0; i<M*N; i++) A[i] = 1.0/(i+1.0);
  for (long j=0; j<N; j++) u[j] = 2.0/(j+3.0);
  //printf("N_work: %ld\n", N_work);

  tt = omp_get_wtime();
  mat_vec_mul(A_u, A, u, M, N); 
  printf("CPU Bandwidth for Mat_vec: %f GB/s\n", M*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  tt = omp_get_wtime();
  cudaMemcpyAsync(B, A, M*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(v, u, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  long *Nbm;
  cudaMallocHost((void**) &Nbm, M*sizeof(long));
  for (long i=0; i<M; i++) Nbm[i] = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
  for (long m=0; m<M; m++) {
    vec_dot_kernel<<<Nbm[m],BLOCK_SIZE>>>(&entries[m*N_work], &B[m*N], v, N);
    while (Nbm[m] > 1) {
      long N = Nbm[m];
      Nbm[m] = (Nbm[m]+BLOCK_SIZE-1)/BLOCK_SIZE;
      reduction_kernel<<<Nbm[m],BLOCK_SIZE>>>(&entries[m*N_work+N], &entries[m*N_work], N);
    }
  }
  cudaMemcpyAsync(&entries_host, entries, M*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth for Mat_vec: %f GB/s\n", M*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  for (long i=0; i<M*N_work; i++) {
    if (entries_host[i]>0.5) {
      printf("i: %ld, val: %f\n", i, entries_host[i]);
    }
  }

  for (long i=0; i<M; i++) {
    //printf("ent_h: %f\n", entries_host[i*N_work]);
    A_v[i] = entries_host[i*N_work+64+448];
  }
  printf("Error = %f\n", dist(A_u, A_v, M));

  for (long i=0; i<M; i++) {
    //printf("Au_%ld: %f\nAv_%ld: %f\n",i, A_u[i], i, A_v[i]);
  }

  cudaFreeHost(A);
  cudaFreeHost(u);
  cudaFree(B);
  cudaFree(v);
  cudaFree(entries);
  cudaFreeHost(entries_host);
  cudaFreeHost(A_u);
  cudaFreeHost(A_v);

  return 0;
}
