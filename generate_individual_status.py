#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

import sys, math
import random
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=60, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import scipy.io as sio
