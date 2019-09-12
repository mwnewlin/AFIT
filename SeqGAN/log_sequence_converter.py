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
file_in_1 = '/run/media/mnewlin/_userdata/uhnds/network/extracted/netflow_day-02_original'
file_out_1 = '/run/media/mnewlin/_userdata/uhnds/network/extracted/netflow_day-02'
temp_file = '/run/media/mnewlin/_userdata/uhnds/network/extracted/netflow_day-02_chunked/netflow_02_chunk_0'

chunk_size = 10**5
i=0
real_df = pd.DataFrame()
for chunk in  pd.read_csv(file_in_1, names=['Time', 'Duration', 'SrcDevice', 
                'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], chunksize=chunk_size, sep=','):
    chunk = chunk.drop(['Time'], axis=1)
    temp_df = chunk
    # Convert EnterpriseAppServer to integer. Done by conversion to hex and then first five
    # nibbles to integer
    temp_df = temp_df.replace(to_replace="EnterpriseAppServer", value="284391")
    temp_df = temp_df.replace(to_replace="ActiveDirectory", value="267831")
    temp_df = temp_df.replace(to_replace="VPN", value="56566")
    temp_df = temp_df.replace(to_replace="VScanner", value="353590")
    temp_df = temp_df.replace(to_replace=[r"^Comp",r"^IP", r"Port"], value="", regex=True)
    real_df = real_df.append(temp_df)
    if (i%100==0):
        print(i)
    i+=1
    
    
# Replace Text from original file
#real_df = real_df.replace(to_replace=[r"^[a-zA-Z]+"], value="", regex=True)

real_df.to_csv(file_out_1, sep=' ', index=False, header=False)
print("Finished")