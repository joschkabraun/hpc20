/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i, k, mpirank;
  double tmp, gres = 0.0, lres = 0.0;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

  for (i = 1; i <= lN; i++){
    for (k = 1; k <= lN; k++) {
      /*if (mpirank == 0) {
        printf("i: %d, k: %d\n", i, k);
      }*/
      tmp = ((4.0*lu[i*(lN+2)+k] - lu[(i-1)*(lN+2)+k] - lu[(i+1)*(lN+2)+k] - lu[i*(lN+2)+k+1] - lu[i*(lN+2)+k-1]) * invhsq - 1);
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, j, k, p, N, lN, iter, max_iters, pow2_j;
  MPI_Status status, status1, status2, status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  // read in j for p=4^j
  sscanf(argv[1], "%d", &j);
  sscanf(argv[2], "%d", &max_iters);
  sscanf(argv[3], "%d", &N);

  /* compute number of unknowns handled by each process */
  lN = (int) N / sqrt(p);
  pow2_j = (int) pow(2,j);
  if ( (N*N % (int) pow(4,j) != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N*N must be a multiple of 4\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  if ( (p != (int) pow(4,j)) && mpirank == 0) {
    printf("p: %d, j: %d\n", p, j);
    printf("Exiting. p must be equals 4**j\n");
  }
  if (mpirank == 0) {
    printf("all good\n");
    printf("p: %d, N: %d, j: %d, lN: %d\n", p, N, j, lN);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (int) pow(lN+2, 2));
  double * lunew = (double *) calloc(sizeof(double), (int) pow(lN+2, 2));
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
      for (k = 1; k <= lN; k++) {
        lunew[i]  = 0.25 * (hsq + lu[(i-1)*(lN+2)+k] + lu[(i+1)*(lN+2)+k] + lu[i*(lN+2)+k+1] + lu[i*(lN+2)+k-1]);
      }
    }

    /* communicate ghost values */
    if (mpirank >= pow2_j) {
      // If not the lower processes, send/recv bdry values to the bottom process
      MPI_Send(&(lunew[(lN+2)+1]), lN, MPI_DOUBLE, mpirank-pow2_j, 124, MPI_COMM_WORLD);                    // lowest inner line of points
      MPI_Recv(&(lunew[1]), lN, MPI_DOUBLE, mpirank-pow2_j, 123, MPI_COMM_WORLD, &status);                  // lowest outer line of points
    }
    if (mpirank <= pow2_j*pow2_j - pow2_j - 1) {
      // If not the upper processes, send/recv bdry values to the upper process
      MPI_Send(&(lunew[lN*(lN+2)+1]), lN, MPI_DOUBLE, mpirank+pow2_j, 123, MPI_COMM_WORLD);                 // highest inner line of poitns
      MPI_Recv(&(lunew[(lN+1)*(lN+2)+1]), lN, MPI_DOUBLE, mpirank+pow2_j, 124, MPI_COMM_WORLD, &status1);   // highest outer line of points
    }
    if (mpirank % pow2_j != 0) {
      // If not the most left processes, send/recv bdry values to the left /
      for (i = 1; i <= lN; i++) {
        MPI_Send(&(lunew[i*(lN+2)+1]), 1, MPI_DOUBLE, mpirank-1, 130+2*i, MPI_COMM_WORLD);                   // most left inner line of points
        MPI_Recv(&(lunew[i*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 129+2*i, MPI_COMM_WORLD, &status2);           // most left outer line of points
      }
    }
    if (mpirank % pow2_j != 1) {
      // If not the most rightprocesses, send/recv bdry values to the right /
      for (i = 1; i <= lN; i++) {
        MPI_Send(&(lunew[i*(lN+2)+lN]), 1, MPI_DOUBLE, mpirank+1, 129+2*i, MPI_COMM_WORLD);                     // most right inner line of points
        MPI_Recv(&(lunew[i*(lN+2)+lN+1]), 1, MPI_DOUBLE, mpirank+1, 130+2*i, MPI_COMM_WORLD, &status3);         // most right outer line of points
      }
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
