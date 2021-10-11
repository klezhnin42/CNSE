#!/usr/bin/python3

import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool
import solver
import driver
from init_angdel import *

if __name__ == '__main__':
    print('Starting time:',datetime.now())
    pool = Pool(40)
    pool.starmap(driver.Simulation,datainputs)
    pool.close()
    pool.join()
    print('Finishing time:',datetime.now())
