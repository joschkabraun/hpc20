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

double dist(double const* u, double const* v, long N) {
  double accum = 0.;
  for (long i=0; i < N; i++) {
    for (long j=0; j< N; j++) {
      accum += (u[i*N+j] - v[i*N+j]) * (u[i*N+j] - v[i*N+j]);
    }
  }
  return sqrt(accum);
}

__global__ void jacobi_kernel(double* u, const double* u0, double h, long N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx = N*i + j
  
  int j = idx % N;
  int i = (idx - j) / N;

  if ((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
    u[idx] = (h*h + u0[N*(i+1)+j] + u0[N*(i-1)+j] + u0[idx-1] + u0[idx+1]) / 4.0;
  }
}

__global__ void A_u_kernel(double* res, const double* u, long N, double h) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx = N*i + j
  int j = idx % N;
  int i = (idx - j) / N;
  printf("idx: %d\n", idx); 
  if (idx == 0) res[idx] = 0.0;
  else if (idx == N*(N-1) + N-1) res[idx] = 0.0;
  else res[idx] = (4*u[idx] - u[idx+1] - u[idx-1] - u[N*(i-1)+j] - u[N*(i+1)+j]) / (h*h);

  printf("A_v: %f at i:%d, j:%d\n", res[idx], i, j);
}

#define BLOCK_SIZE 1024

__global__ void l2_norm_shift_kernel(double* sum, const double* a, double shift, long N) {
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = (a[idx]-shift) * (a[idx]-shift);
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
  long N = 512;
  long it = 10*3000;
  //const long BLOCK_SIZE = 1024;
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
  for (long k=1; k<it+1; k++) {
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
  long N_work = 1;
  for (long i= (N*N+BLOCK_SIZE-1)/BLOCK_SIZE; i>1; i = (i+BLOCK_SIZE-1)/BLOCK_SIZE) N_work += i;
  cudaMalloc(&res_d, N_work*sizeof(double));
  //printf("N_work: %ld\n", N_work);


  tt = omp_get_wtime();
  printf("GPU Jacobi2D\n");
  printf("iteration     residue\n");
  printf("---------------------\n");
  for (long k=1; k<it+1; k++) {
    cudaMemcpy(v_0, v, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
    jacobi_kernel<<<N*N/BLOCK_SIZE, BLOCK_SIZE>>>(v, v_0, h, N);
    cudaDeviceSynchronize();
    
    if (((k % 3000) == 0) && (k > it)) {   // code does not work for some reason
      A_u_kernel<<<N*N/BLOCK_SIZE, BLOCK_SIZE>>>(A_v, v, N, h);

      // calculate residue as in other task      
      long Nb = (N*N+BLOCK_SIZE-1)/BLOCK_SIZE;
      l2_norm_shift_kernel<<<Nb,BLOCK_SIZE>>>(res_d, A_v, 1., N*N);
      while (Nb > 1) {
	//printf("hi\n");
        long N = Nb;
	Nb = (Nb+BLOCK_SIZE-1)/BLOCK_SIZE;
	reduction_kernel<<<Nb,BLOCK_SIZE>>>(res_d+N, res_d, N);
	res_d += N;
      }
      double res;
      cudaMemcpyAsync(&res, &res_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize(); 

      printf(" %ld      %f\n", k, res);
    }
  }
  printf("GPU: %f it/s\n", 1.0*it / (omp_get_wtime()-tt));

  cudaMemcpyAsync(u_0, v, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  printf("Difference is %f\n", dist(u,u_0, N));

  cudaFree(v_0);
  cudaFree(v);
  cudaFree(res_d);
  cudaFree(A_v);
  cudaFreeHost(u_0);
  cudaFreeHost(A_u);
  cudaFreeHost(u);
  return 0;
}
