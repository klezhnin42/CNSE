#!/usr/bin/python3

import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool
import solver
import driver
from init import *

if __name__ == '__main__':
    pool = Pool(1)#os.cpu_count())
    print('Starting time:',datetime.now())
    pool.starmap(driver.Simulation,datainputs)
    pool.close()
    pool.join()
    print('Finishing time:',datetime.now())
