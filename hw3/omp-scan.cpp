#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n, long p) {
  if (n==0) return;
  //int p = omp_get_max_threads();  		// num_threads
  long* pre_part_sum = (long*) calloc(p, sizeof(long));
  long* part_sum = (long*) calloc(p, sizeof(long));
  omp_set_num_threads(p);
//prefix_sum[0] = 0;
  
  #pragma omp parralel shared(p, prefix_part_sum, part_sum)
  {
  #pragma omp for
  for (long i = 0; i < p; i++) {
    long l = i*n/p;
    prefix_sum[l] = 0;
    for (long j = l+1; j < l+n/p; j++) {
      prefix_sum[j] = prefix_sum[j-1] + A[j-1];
    }
    part_sum[i] = prefix_sum[l+n/p-1] + A[l+n/p-1];
  }

  #pragma omp single
  {
  scan_seq(pre_part_sum, part_sum, p);
  }
 
 // correcting error
  #pragma omp for
    for (long i = 1; i < p; i++) {
      long l = i*n/p;
      for (long j = l; j < l+n/p; j++) {
        prefix_sum[j] = prefix_sum[j] + pre_part_sum[i];
      }
    }
  }
  
  // handling the remaining data
  for (long i = n/p*p; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }


  free(pre_part_sum);
  free(part_sum);
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);
  
  int thread_nums[] = {1,2,4};
  for (int p : thread_nums) {
    tt = omp_get_wtime();
    scan_omp(B1, A, N, p);
    printf("parallel-scan with %d threads = %fs\n", p, omp_get_wtime() - tt);
  }

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  //delete thread_nums;
  return 0;
}
