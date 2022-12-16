# CNSE
A Python code for analysis of laser envelope resonant interactions

This is an implementation of a solver of multiple coupled 2+1 Schrodinger equations (two spatial and one temporal coordinate) with an arbitrary number of 
equations involved in the system, with each equation tracking the evolution of an independent laser envelope. The numerical approach used is known as a 
symmetrized split-step Fourier method (see, e.g., G. P. Agrawal "Nonlinear Fiber Optics", Elsevier (2006)); to account for resonant interaction between 
different envelopes, transfer matrix method is used. The code supports both multiprocessing/pool and MPI parallelization for conducting large parameter 
scans. NumPy, mpi4py, Matplotlib libraries are used. Multiple tests are implemented to check envelope propagation, diffraction, and Raman scattering against 
known analytical solutions.

## How to use:

1. To use MPI version, one should use main_mpi.py, to use Pool parallelization - main.py. There, besides required libraries, we load all the information
from initialization files (see init_*.py files), where we specify initial envelopes, their group velocities, frequencies, self-focusing and coupling constants.
For instance, in main_mpi.py, we do `from init_scanwidth import *` to conduct a scan on pump laser pulse parameters, width and duration, for the fixed 
pump energy.


## Limitations and possible developments:
As of now, the code is not vectorized; although there is an analytical expression for the transfer matrix for any number of equations, 
