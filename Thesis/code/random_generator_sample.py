# -*- coding: utf-8 -*-
"""
@author: mnewlin
"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

NUM_COLS = 10
fake_dir = '/run/media/mnewlin/_userdata/uhnds/network/converted/fake/'
real_dir = '/run/media/mnewlin/_userdata/uhnds/network/converted/real/'
real_file = 'netflow_day-02'

"""
    Function to read in a single real sample from a given directory based
    on the desired length of the sample.
"""
def load_real_sample(sample_length=100, random_state=69):
    np.random.RandomState(seed=random_state)
    data_dir = 'samples_{}/'.format(sample_length)
    sample_range = 0
    if sample_length < 10000:
        sample_range = 10000
    elif sample_length < 100000:
        sample_range = 2000
    else:
        sample_range = 1160
        
    random_sample = np.random.randint(0,sample_range)
    load_file = real_file + '_sample_{}.txt'.format(random_sample)
    filename = real_dir + data_dir + load_file
    df = pd.read_csv(filename, names=['Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], sep=' ')
    return df



"""
    Function to generate random netflow samples based on real samples,
    desired sample length, and the number of samples
"""
def generate_random_samples(random_state=69, sample_length=100, num_samples=1000):
    
    np.random.RandomState(seed=random_state)   
    
    for n in range(num_samples):
        
        real_df = load_real_sample(sample_length=sample_length, random_state=random_state)
        fake_array = np.zeros((sample_length,NUM_COLS))
        for x in range(sample_length):
            # Generate fake Duration
            real_duration_min = int(np.min(real_df['Duration']))
            real_duration_max = int(np.max(real_df['Duration']))
            duration = np.random.randint(real_duration_min, real_duration_max+1)
        
            # Generate fake SrcDevice
            real_srcDev_min = int(np.min(real_df['SrcDevice']))
            real_srcDev_max = int(np.max(real_df['SrcDevice']))
            src_dev = np.random.randint(real_srcDev_min, real_srcDev_max+1)
                
            # Generate fake DstDevice
            real_dstDev_min = int(np.min(real_df['DstDevice']))
            real_dstDev_max = int(np.max(real_df['DstDevice']))
            dst_dev = np.random.randint(real_dstDev_min, real_dstDev_max+1)
                
            # Generate fake protocol (must be either TCP(6) or UDP(17))
            prot_num = np.random.randint(0,3)
            prot_list = [1, 6, 17]
            protocol = prot_list[prot_num]
                
            # Generate fake srcPort
            src_port = np.random.randint(1,65536)
                
            # Generate fake DstPort
            dst_port = np.random.randint(1,65536)
                
            # Generate fake SrcPackets
            src_packet_max = int(np.max(real_df['SrcPackets']))
            src_packets = np.random.randint(0, src_packet_max+1)
                
            # Generate fake DstPackets
            dst_packet_max = int(np.max(real_df['DstPackets']))
            dst_packets = np.random.randint(0, dst_packet_max+1)
                
            # Generate fake SrcBytes
            src_bytes_max = int(np.max(real_df['SrcBytes']))
            src_bytes = np.random.randint(0, src_bytes_max+1)
                
            # Generate fake DstBytes
            dst_bytes_max = int(np.max(real_df['DstBytes']))
            dst_bytes = np.random.randint(0, dst_bytes_max+1)
                
            entry = np.array([duration, src_dev, dst_dev, protocol, src_port, 
                                  dst_port, src_packets, dst_packets, src_bytes, dst_bytes])
                
            fake_array[x] = entry
    
        fake_df = pd.DataFrame(data = fake_array, columns=['Duration', 'SrcDevice', 
                    'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
                    'SrcBytes', 'DstBytes'], dtype=np.int64)
        out_dir = 'samples_{}/'.format(sample_length)
        fake_filename = fake_dir + out_dir + real_file + '_' + 'random_sample_{}.txt'.format(n)
        fake_df.to_csv(fake_filename, sep=' ', index=False, header=False)


"""
    Code to create random samples
"""


# Generate 10000 samples of length 100
generate_random_samples(sample_length=100, num_samples=10000)
print("Finished with sample length 100.")
        
# Generate 10000 samples of length 1000
generate_random_samples(sample_length=1000, num_samples=10000)
print("Finished with sample length 1000.")

# Generate 2000 samples of length 10000
generate_random_samples(sample_length=10000, num_samples=2000)
print("Finished with sample length 10000.")

# Generate 1160 samples of length 10000
generate_random_samples(sample_length=100000, num_samples=1160)
print("Finished with sample length 100000.")
#print("All Done.")

