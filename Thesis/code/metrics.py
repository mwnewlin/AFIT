#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:03:47 2019

@author: mnewlin
"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats


"""
    Calculates the Power distance between two matrices X and Y
    Defaults to Euclidean Distance unless parameters p and r are provided
"""
def l_p_distance(X,Y,p=2,r=2):
    X = np.array(X)
    Y = np.array(Y)
    if (X.shape != Y.shape):
        print("Usage: Matrices must be the same shape.")
        return -1
    num_rows = X.shape[0]
    distances = np.zeros((num_rows,1))
    for i in range(num_rows):
        x = X[i]
        y = Y[i]
        distances[i] = np.power(np.sum(np.power(np.abs(x-y),p)),(1/r))
    
    return np.sum(distances)

def cosine_similarity(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    num_rows = X.shape[0]
    cos_sims = np.zeros((num_rows,1))
    