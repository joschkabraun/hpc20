#!/bin/bash
#
#SBATCH --verbose
#SBATCH --job-name=hpc20-hw5-jacobi2d
#SBATCH --nodes=8-32
#SBATCH --tasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=00:20:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jjb666@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load openmpi/gnu/4.0.2
rm jacobi_mpi
mpic++ -O3 -o jacobi_mpi jacobi_mpi.cpp
mpirun -np 4   jacobi_mpi 1  200 10000
mpirun -np 16  jacobi_mpi 2  400 10000
mpirun -np 64  jacobi_mpi 3  800 10000
mpirun -np 256 jacobi_mpi 4 1600 10000
mpirun -np 1   jacobi_mpi 0  400 10000
mpirun -np 4   jacobi_mpi 1  400 10000
mpirun -np 16  jacobi_mpi 2  400 10000
mpirun -np 64  jacobi_mpi 3  400 10000
mpirun -np 256 jacobi_mpi 4  400 10000
