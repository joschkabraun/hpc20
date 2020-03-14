/******************************************************************************
* FILE: omp_bug5.c
* DESCRIPTION:
*   Using SECTIONS, two threads initialize their own array and then add
*   it to the other's array, however a deadlock occurs.
* AUTHOR: Blaise Barney  01/29/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 10000 // I had to decrease the N from the original 1000000 as otherwise my computer would not have been able to allocate that much space
#define PI 3.1415926535
#define DELTA .01415926535

int main (int argc, char *argv[]) 
{
int nthreads, tid, i;
float a[N], b[N], c[N], d[N], tmp[N]; // introduced temporary variable (cf. understanding of problem blow)
omp_lock_t locka, lockb, lockc, lockd;

/* Initialize the locks */
omp_init_lock(&locka);
omp_init_lock(&lockb);

/* There are two different interpretation of the work which is going to be exectued below:
 * 1st: initialize a and b. Then add a to b. Then this updated b to a.
 * 2nd: Initialize a and b. Then add a to b and b to a without using updated values of a and b in the mean time.
 * I will implement both interpretations
 */

// 1st interpretation implementation
/* Fork a team of threads giving them their own copies of variables */
printf("Executing first interpretation...\n");
#pragma omp parallel shared(a, b, nthreads, locka, lockb) private(tid)
  {

  /* Obtain thread number and number of threads */
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  #pragma omp barrier // to ensure first the number of threads is going to be printed
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      printf("Thread %d initializing a[]\n",tid);
      omp_set_lock(&locka);
      for (i=0; i<N; i++)
        a[i] = i * DELTA;
      
      omp_set_lock(&lockb);
      printf("Thread %d adding a[] to b[]\n",tid);
      for (i=0; i<N; i++)
        b[i] += a[i];
      omp_unset_lock(&lockb);
      omp_unset_lock(&locka);
      }

    #pragma omp section
      {
      printf("Thread %d initializing b[]\n",tid);
      omp_set_lock(&lockb);
      for (i=0; i<N; i++)
        b[i] = i * PI;
      omp_unset_lock(&lockb);
      omp_set_lock(&locka);
      printf("Thread %d adding b[] to a[]\n",tid);
      for (i=0; i<N; i++)
        a[i] += b[i];
      omp_unset_lock(&locka);
      omp_unset_lock(&lockb);
      }
    }  /* end of sections */
  }  /* end of parallel region */
 
omp_init_lock(&lockc);
omp_init_lock(&lockd);
  // 2nd interpreationi
printf("\n\nExecuting second interpretation...\n");
#pragma omp parallel shared(c, d, tmp, nthreads, lockc, lockd) private(tid)
  {

  /* Obtain thread number and number of threads */
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  #pragma omp barrier // to ensure first the number of threads is going to be printed
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      printf("Thread %d initializing a[]\n",tid);
      omp_set_lock(&lockc);
      for (i=0; i<N; i++)
        c[i] = i * DELTA;
      
      omp_set_lock(&lockd);
      printf("Thread %d adding a[] to b[]\n",tid);
      for (i=0; i<N; i++) {
        tmp[i] = d[i];
        d[i] += c[i];
      }
      omp_unset_lock(&lockd);
      omp_unset_lock(&lockc);
      }

    #pragma omp section
      {
      printf("Thread %d initializing b[]\n",tid);
      omp_set_lock(&lockd);
      for (i=0; i<N; i++)
        d[i] = i * PI;
      omp_unset_lock(&lockd);
      omp_set_lock(&lockc);
      printf("Thread %d adding b[] to a[]\n",tid);
      for (i=0; i<N; i++)
        c[i] += tmp[i];
      omp_unset_lock(&lockc);
      omp_unset_lock(&lockd);
      }
    }  /* end of sections */
  }  /* end of parallel region */

  // locks haave to be destroyed
  omp_destroy_lock(&locka);
  omp_destroy_lock(&lockb);
  omp_destroy_lock(&lockc);
  omp_destroy_lock(&lockd);
}

