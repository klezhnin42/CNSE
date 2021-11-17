#!/usr/bin/python3

import numpy as np
import os
from datetime import datetime
from multiprocessing import Pool
from source import solver
from source import driver
from source.tests.init_units2 import *

if __name__ == '__main__':
    print('Starting time:',datetime.now())
    pool = Pool(1)
    pool.starmap(driver.Simulation,datainputs)
    pool.close()
    pool.join()
    print('Finishing time:',datetime.now())
