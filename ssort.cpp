// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  //for (int i = 0; i < N; i++)
   // printf("rank: %d, entry: %d\n", rank, vec[i]);

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* samples = (int*)malloc((p-1)*sizeof(int));
  int freq = N / p;
  for (int i = 1; i <= p-1; i++) {
    samples[i-1] = vec[freq*i];
  }

  //printf("rank: %d, sample: %d\n", rank, samples[0]);

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* tot_samples;

  if (rank == 0){
    tot_samples = (int*)malloc((p-1)*p*sizeof(int));
  }
  MPI_Gather(
      samples,
      p-1,
      MPI::INT,
      tot_samples,
      p-1,
      MPI::INT,
      0,
      MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int* splitters = (int*)malloc((p-1)*sizeof(int));
  if (rank == 0) {
    std::sort(tot_samples, tot_samples+p*(p-1));
    for (int i = 1; i <= p-1; i++) {
      splitters[i-1] = tot_samples[i*(p-1)];
    }
    //for (int i = 0; i < p*(p-1); i++) printf("%d", tot_samples[i]);
  //printf("rank: %d, splitter: %d\n", rank, splitters[0]);
    
  }
  // root process broadcasts splitters to all other processes
  MPI_Bcast(
      splitters, 
      p-1,
      MPI::INT, 
      0,
      MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int* send_counts =(int*)malloc(p*sizeof(int));
  int* last_pt = vec;
  int* pt;
  for (int i =0; i < p-1; i++) {
   // printf("Rank %d, Splitter: %d\n", rank, splitters[i]);
    pt = std::lower_bound(last_pt, vec+N, splitters[i]);
    send_counts[i] = pt - last_pt;
    last_pt = pt;
  }    
  send_counts[p-1] = vec+N-last_pt;


  //for (int i = 0; i < p; i++) printf("Rank: %d, ind: %d, size: %d\n", rank, i,send_counts[i]);

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int* rec_counts =(int*)malloc(p*sizeof(int));
  MPI_Alltoall(
      send_counts,
      1,
      MPI::INT,
      rec_counts,
      1, 
      MPI::INT,
      MPI_COMM_WORLD);

  //for (int i = 0; i < p; i++) printf("Rank: %d, rec size: %d\n", rank, rec_counts[i]);

  int send_off[p];
  send_off[0] = 0;
  for (int i = 1; i < p; i++)
    send_off[i] = send_counts[i-1]+send_off[i-1];

  int rec_off[p];
  rec_off[0] = 0;
  for (int i = 1; i < p; i++)
    rec_off[i] = rec_counts[i-1]+rec_off[i-1];

  int new_N = rec_off[p-1]+rec_counts[p-1];

  int* new_vec = (int*)malloc(new_N*sizeof(int));

  MPI_Alltoallv(
      vec, 
      send_counts,
      send_off,
      MPI::INT,
      new_vec, 
      rec_counts, 
      rec_off, 
      MPI::INT,
      MPI_COMM_WORLD);

  //for (int i = 0; i < new_N; i++)
   // printf("Final: rank: %d, entry: %d\n", rank, new_vec[i]);

  // do a local sort of the received data
  std::sort(new_vec, new_vec+new_N); 

  // every process writes its result to a file
  { // Write output to a file
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }

    fprintf(fd, "rank %d get the following sort:\n", rank);
    for(int i = 0; i < new_N; i++)
      fprintf(fd, "  %d\n", new_vec[i]);

    fclose(fd);
  }

  free(vec);
  MPI_Finalize();
  return 0;
}
