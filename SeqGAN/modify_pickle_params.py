#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:28:27 2019

@author: mnewlin
"""

import pickle

pickle_file = 'save/target_params_py3.pkl'
pickle_in = open(pickle_file, "rb")

var = pickle.load(pickle_in)

