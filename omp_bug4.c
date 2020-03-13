/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
int  a[N][N];
// we are writing integer to the double array a so changed it to int instead of double

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  // ensure number of threads is first printed
  //#pragma omp barrier
  printf("Thread %d starting...\n", tid);
  // ensure that all threads start at the same time
  //#pragma omp barrier

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;

  /* For confirmation */
  // to appropriatetly write the last element, the flag has to be for integers instead of doubles
  printf("Thread %d done. Last element= %d\n",tid,a[N-1][N-1]);

  }  /* All threads join master thread and disband */

}

