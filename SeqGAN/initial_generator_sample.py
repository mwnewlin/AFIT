# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt


# Read in Log File
real_filename = 'save/netflow_02_chunk_0.txt'

real_df = pd.read_csv(real_filename, names=['Time', 'Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], sep=' ')

desc = real_df.describe()

fake_array = np.zeros((10000,11))

for x in range(10000):
    # Generate fake time
    real_time_min = np.min(real_df['Time'])
    real_time_max = np.max(real_df['Time'])
    time = np.random.randint(real_time_min, real_time_max+1)
    
    # Generate fake Duration
    real_duration_min = np.min(real_df['Duration'])
    real_duration_max = np.max(real_df['Duration'])
    duration = np.random.randint(real_duration_min, real_duration_max+1)

    # Generate fake SrcDevice
    real_srcDev_min = np.min(real_df['SrcDevice'])
    real_srcDev_max = np.max(real_df['SrcDevice'])
    src_dev = np.random.randint(real_srcDev_min, real_srcDev_max+1)
    
    # Generate fake DstDevice
    real_dstDev_min = np.min(real_df['DstDevice'])
    real_dstDev_max = np.max(real_df['DstDevice'])
    dst_dev = np.random.randint(real_dstDev_min, real_dstDev_max+1)
    
    # Generate fake protocol (must be either TCP(6) or UDP(17))
    prot_num = np.random.randint(0,2)
    protocol = 0
    if prot_num == 0:
        protocol = 6
    else:
        protocol = 17
    
    # Generate fake srcPort
    src_port = np.random.randint(0,65536)
    
    # Generate fake DstPort
    dst_port = np.random.randint(0,65536)
    
    # Generate fake SrcPackets
    src_packet_max = np.max(real_df['SrcPackets'])
    src_packets = np.random.randint(0, src_packet_max+1)
    
    # Generate fake DstPackets
    dst_packet_max = np.max(real_df['DstPackets'])
    dst_packets = np.random.randint(0, dst_packet_max+1)
    
    # Generate fake SrcBytes
    src_bytes_max = np.max(real_df['SrcBytes'])
    src_bytes = np.random.randint(0, src_bytes_max+1)
    
    # Generate fake DstBytes
    dst_bytes_max = np.max(real_df['DstBytes'])
    dst_bytes = np.random.randint(0, dst_bytes_max+1)
    
    entry = np.array([time, duration, src_dev, dst_dev, protocol, src_port, 
                      dst_port, src_packets, dst_packets, src_bytes, dst_bytes])
    
    fake_array[x] = entry
    
fake_df = pd.DataFrame(data = fake_array, columns=['Time', 'Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], dtype=np.int64)
    
fake_filename = 'save/unhds_generator_sample_c0.txt'

fake_df.to_csv(fake_filename, sep=' ', index=False, header=False)
