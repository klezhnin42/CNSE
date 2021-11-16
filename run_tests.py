#!/usr/bin/python3

import numpy as np
import os
from source import solver
from source import driver
from source.tests.init_units import *
from source.tests.test_units import *

assert run_unit_test(maindir,cpls[0],ua0,ub0,vga,vgb,cvph1,cvph2,x,y,kxm,kym,k2xm,k2ym,dt,Es,couplings[0],Nt)==0
