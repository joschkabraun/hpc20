/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
float total;

/*** Spawn parallel region ***/
#pragma omp parallel 
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  // we want first the number of threads printed and then start doing computations
  #pragma omp barrier

  printf("Thread %d is starting...\n",tid);

  //#pragma omp barrier

  /* do some work */
  total = 0.0;
  //#pragma omp for schedule(dynamic,10)
  // a static work load seems more appropriate than the dynamic load balancing as the workload per iteration is not increasing
  // to avoid the race condition in updating the total sum, I introduce a reduction on the total variable
  #pragma omp for schedule(static) reduction(+:total)
	for (i=0; i<1000000; i++) 
     total += i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
