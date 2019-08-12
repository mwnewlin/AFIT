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
filename = 'save/unhds_netflow_sample.csv'
file_in_1 = '/run/media/mnewlin/_userdata/uhnds/network/extracted/netflow_day-02'

chunksize = 10**4
i=0
for chunk in pd.read_csv(file_in_1, names=['Time', 'Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], chunksize=chunksize):
    # Replace Comp, IP, and Port
    log_file_df = chunk
    log_file_df_conv = log_file_df
    log_file_df_conv = log_file_df_conv.replace(to_replace=[r"^Comp",r"^IP", r"Port"], 
                                                value="", regex=True)
    # Convert EnterpriseAppServer to integer. Done by conversion to hex and then first five
    # nibbles to integer
    log_file_df_conv = log_file_df_conv.replace(to_replace="EnterpriseAppServer", value="284391")
    log_file_df_conv = log_file_df_conv.replace(to_replace="ActiveDirectory", value="267831")
    log_file_df_conv = log_file_df_conv.replace(to_replace="VPN", value="56566")
        
    export_file = '/run/media/mnewlin/_userdata/uhnds/network/extracted/netflow_day-02_chunked/netflow_02_chunk_{}'.format(i)
    log_file_df_conv.to_csv(export_file, sep=' ', index=False, header=False)
    i=i+1
    if i >= 200:
        break