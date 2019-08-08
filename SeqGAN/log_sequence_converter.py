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
file_in_1 = ''
log_file_df = pd.read_csv(filename, names=['Time', 'Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'])

# Replace Comp, IP, and Port
log_file_df_conv = log_file_df
log_file_df_conv = log_file_df_conv.replace(to_replace=[r"^Comp",r"^IP", r"Port"], 
                                            value="", regex=True)
# Convert EnterpriseAppServer to integer. Done by conversion to hex and then first five
# nibbles to integer
log_file_df_conv = log_file_df_conv.replace(to_replace="EnterpriseAppServer", value="284391")
    
export_file = 'save/unhds_netflow_converted.csv'
log_file_df_conv.to_csv(export_file, sep=' ', index=False, header=False)

