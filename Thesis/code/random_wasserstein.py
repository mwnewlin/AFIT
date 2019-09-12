# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean as euclidean_dist
from scipy.stats import wasserstein_distance as wasserstein

real_base = 'netflow_02_chunk_0_'
real_dir = '/run/media/mnewlin/_userdata/uhnds/network/converted/real/samples/'
fake_dir = '/run/media/mnewlin/_userdata/uhnds/network/converted/fake/'
real_samples = []
fake_samples = []
for x in range(100):
    fake_samples.append(real_base + 'random_generated_{}.txt'.format(x))
    real_samples.append(real_base + 'sample_{}.txt'.format(x))

wasserstein_dists = np.zeros((100,11))
wasserstein_dists_copy = np.zeros((100,11))
col_names=['Time', 'Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes']
# For each of the 100 samples: calculate mean wasserstein across the 11 columns
for x in range(100):
    real_df = pd.read_csv(real_dir + real_samples[x], names=['Time', 'Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], sep=' ')
    fake_df = pd.read_csv(fake_dir + fake_samples[x], names=['Time', 'Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], sep=' ')
    wass_dists = []
    wass_dists_copy = []
    for col in col_names:
        real_dist = real_df.loc[:,col]
        fake_dist = fake_df.loc[:,col]
        wass = wasserstein(real_dist, fake_dist)
        wass_copy = wasserstein(real_dist, real_dist)
        wass_dists.append(wass)
        wass_dists_copy.append(wass_copy)
        
    wasserstein_dists[x] = np.array([wass_dists])
    wasserstein_dists_copy[x] = np.array([wass_dists_copy])
    
wasserstein_means = np.mean(wasserstein_dists, axis=1)
wasserstein_means_copy = np.mean(wasserstein_dists_copy, axis=1)
plt.semilogy(wasserstein_means, 'b-')
plt.show()

plt.plot(wasserstein_means_copy, 'r-')
plt.show()