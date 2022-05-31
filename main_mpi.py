#!/usr/bin/python3

import numpy as np
import os
from mpi4py import MPI
from source import solver
from source import driver
from init_scanwidth import *

if __name__ == '__main__':
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()
    nproc=comm.Get_size()
    name = MPI.Get_processor_name()
    print(rank,nproc,name)
    datasize=len(datainputs)
    chunks=datasize//nproc+1
    for i in range(chunks):
        if rank+i*nproc<datasize:
            data=datainputs[rank+i*nproc]
            driver.Simulation(*data)
