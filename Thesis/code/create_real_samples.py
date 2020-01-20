#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:33:57 2019

@author: mnewlin
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#make plots inline using jupyter magic

import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import datasets, linear_model, metrics


import matplotlib as mpl
import seaborn as sns






data_dir = '/run/media/mnewlin/_userdata/uhnds/'
original_netflow_data_dir = data_dir + 'network/extracted/'
original_netflow_file = 'netflow_day-02'

original_host_data_dir = data_dir + 'host/extracted/'
original_host_file = 'wls_day-02.json'

host_data_columns = ['AuthenticationPackage', 'Destination', 'DomainName', 'EventID',
       'FailureReason', 'LogHost', 'LogonID', 'LogonType',
       'LogonTypeDescription', 'ParentProcessID', 'ParentProcessName',
       'ProcessID', 'ProcessName', 'ServiceName', 'Source', 'Status',
       'SubjectDomainName', 'SubjectLogonID', 'SubjectUserName', 'Time', 'UserName']

"""
    Creates real samples of data based on number of samples and desired sample length.
"""
def create_real_samples_json(directory='host', original_file=original_host_file, num_samples=1000, sample_length=1000, random_seed=69):
    curr_chunk=0
    if directory == 'host':
        reader = pd.read_json(original_host_data_dir+original_file, chunksize=sample_length, lines=True, orient='records')
        for chunk in reader:
            
            real_df = chunk
            if real_df.shape[0] < len(host_data_columns):
                for col in host_data_columns:
                    if col not in real_df.columns:
                        real_df.at[:,col] = np.NaN
            elif real_df.shape[0] > len(host_data_columns):
                for col in real_df.columns:
                    if col not in host_data_columns:
                        real_df = real_df.drop([col], axis=1)
            # Replace Text from original file
            #real_df = real_df.replace(to_replace=[r"^Comp",r"^IP", r"Port"], 
                                                #value="", regex=True)
            # Convert EnterpriseAppServer to integer. Done by conversion to hex and then first five
            # nibbles to integer
            #real_df = real_df.replace(to_replace="EnterpriseAppServer", value="284391")
            #real_df = real_df.replace(to_replace="ActiveDirectory", value="267831")
            #real_df = real_df.replace(to_replace="VPN", value="56566")

            out_dir = 'samples_{}/'.format(sample_length)
            outfile = original_file+'_sample_{}.txt'.format(curr_chunk)
            outpath = data_dir + directory + '/unconverted/real/' + out_dir + outfile 
            real_df.to_csv(outpath, sep=',', index=False)
            curr_chunk += 1
            #if curr_chunk >= num_samples:
                #break
    
    return 0

"""
    Creates real samples of data based on number of samples and desired sample length.
"""
def create_real_samples(directory='network', original_file=original_netflow_file, num_samples=1000, sample_length=1000, random_seed=69):
    curr_chunk=0
    if directory == 'network':
        for chunk in pd.read_csv(original_netflow_data_dir+original_file, names=['Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], chunksize=sample_length, sep=' '):
            
            real_df = chunk
            # Replace Text from original file
            #real_df = real_df.replace(to_replace=[r"^Comp",r"^IP", r"Port"], 
                                                #value="", regex=True)
            # Convert EnterpriseAppServer to integer. Done by conversion to hex and then first five
            # nibbles to integer
            #real_df = real_df.replace(to_replace="EnterpriseAppServer", value="284391")
            #real_df = real_df.replace(to_replace="ActiveDirectory", value="267831")
            #real_df = real_df.replace(to_replace="VPN", value="56566")

            out_dir = 'samples_{}/'.format(sample_length)
            outfile = original_file+'_sample_{}.txt'.format(curr_chunk)
            outpath = data_dir + directory + '/converted/real/' + out_dir + outfile 
            real_df.to_csv(outpath, sep=' ', index=False, header=False)
            curr_chunk += 1
            if curr_chunk >= num_samples:
                break
    
    return 0

create_real_samples_json(num_samples=10000)
print("Finished")

"""
# Create 100 line samples (10000)
create_real_samples(num_samples=10000,sample_length=100)
print("Finished with sample length 100.")

# Create 1000 line samples (10000)
create_real_samples(num_samples=10000,sample_length=1000)
print("Finished with sample length 1000.")

# Create 10000 line samples (2000)
create_real_samples(num_samples=2000,sample_length=10000)
print("Finished with sample length 10000.")

# Create 100000 line samples (2000)
create_real_samples(num_samples=2000,sample_length=100000)
print("Finished with sample length 100000.")
"""

