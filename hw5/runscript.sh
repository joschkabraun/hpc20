#!/bin/bash
#
#SBATCH --job-name=hpc20-hw5-jacobi2d
#SBATCH --nodes=8-32
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=00:10:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jjb666@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load openmpi/gnu/4.0.2
rm jacobi_mpi
mpic++ -o jacobi_mpi jacobi_mpi.cpp

mpiexec -np 64 ./jacobi_mpi 3 300 1600
