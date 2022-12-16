# CNSE
A Python code for analysis of laser envelope resonant interactions

This is an implementation of a solver of multiple coupled nonlinear 2+1 Schrodinger equations (two spatial and one temporal coordinate) with an arbitrary number of equations involved in the system, with each equation tracking the evolution of an independent laser envelope. The numerical approach used is known as a symmetrized split-step Fourier method (see, e.g., G. P. Agrawal "Nonlinear Fiber Optics", Elsevier (2006)); to account for resonant interaction between different envelopes, transfer matrix method is used. The code supports both multiprocessing/pool and MPI parallelization for conducting large parameter scans. NumPy, mpi4py, Matplotlib libraries are used. Multiple tests are implemented to check envelope propagation, diffraction, and Raman scattering against known analytical solutions.

## How to use:

1. To use the MPI version, one should use main_mpi.py, to use Pool parallelization - main.py. There, besides required libraries, we load all the information from initialization files (see init_*.py files), where we specify initial envelopes, their group velocities, frequencies, self-focusing, and coupling constants. For instance, in main_mpi.py, we do `from init_scanwidth import *` to conduct a scan on pump laser pulse parameters, width, and duration, for the fixed pump energy. Also, there are multiple interaction models that you should specify in the main.py or main_mpi.py files. These models are described in ~/source/driver.py. In short: `driver.Simulation` is used when you consider single seed-single pump simulation; `driver.SimulationTwoPump` is used when two pumps with independent beatings are considered; `driver.SimulationTwoPumpOneBeat` - when two pumps and a single beating (sum of beatings between seed and first pump and seed and second pump) are desired; `driver.SimulationMultiPumpMultiBeat` and `driver.SimulationMultiPump` - same as above but for an arbitrary number of pumps.


2. MPI: here is an example of a Slurm submission script that works for Tiger cluster at Princeton University https://researchcomputing.princeton.edu/systems/tiger

```
#!/bin/bash

#SBATCH -N 2
#SBATCH --ntasks-per-node=40
#SBATCH -t 01:00:00
#SBATCH -p test
#SBATCH -J cnse_mpi
#SBATCH -o cnse_mpi.o%j
#SBATCH -e cnse_mpi.e%j
#export SLURM_WHOLE=1

module purge
module load anaconda3/2021.11
module load openmpi/gcc/3.1.5/64
conda activate fast-mpi4py

Output_DIR=/path/to/repository
cd $Output_DIR

date
pwd
echo $Output_DIR | mpirun -n 80 python main_mpi.py
date
```

fast-mpi4py environment is used to make mpi4py properly work on Tiger cluster, see the following link for the details:
https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py

3. Pool: here is an example of a Slurm submission script that works for Tiger cluster at Princeton University

```
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 40
#SBATCH -p test
#SBATCH -t 1:00:00
#SBATCH -J cnse_pool
#SBATCH -o cnse_pool.o%j
#SBATCH -e cnse_pool.e%j
#export SLURM_WHOLE=1

module load anaconda3

Output_DIR=/path/to/repository
cd $Output_DIR

date
pwd
echo $Output_DIR | python main.py
date
```

Note that in the case of Pool, only one computational node should be used.

4. The default output consists of (1) figures representing envelope and phase evolution and (2) .npy files with full envelope data, which are easily processed with Python.

## Limitations and possible developments:
As of now, the code is not vectorized; although there is an analytical expression for the transfer matrix for any number of pump pulses (nonlinear Schrodinger equations), I did not find a good way to construct a transfer matrix for vectorized calculation.
