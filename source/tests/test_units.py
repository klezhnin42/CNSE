#!/usr/bin/python3

import numpy as np
import os
import sys
from datetime import datetime
from .. import solver
from .. import driver

def run_unit_test(maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt):
    driver.Simulation(maindir,cpl,ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,coupling,Nt)
