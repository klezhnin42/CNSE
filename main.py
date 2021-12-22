#!/usr/bin/python3

import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool
from source import solver
from source import driver
from init_npump import *

if __name__ == '__main__':
    print('Starting time:',datetime.now())
    pool = Pool(1)
    pool.starmap(driver.SimulationMultiPump,datainputs)
    pool.close()
    pool.join()
    print('Finishing time:',datetime.now())
