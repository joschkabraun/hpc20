#!/bin/bash
#
#SBATCH --job-name=hpc20-hw5-jacobi2d
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=00:10:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jjb666@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load openmpi/gnu/4.0.2
./jacobi_mpi 2 10000 160
