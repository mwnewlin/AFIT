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
import pandas


sentiment_singularity_csv = '~/git/afit_mlperf_training/sentiment_analysis/results/1xP100_sentiment_analysis_singularity.csv'
sentiment_native_csv = '~/git/afit_mlperf_training/sentiment_analysis/results/1xP100_sentiment_analysis_native.csv'

native_df = pandas.read_csv(sentiment_native_csv, '\n', names=['Native Runtime (Seconds)'])
singularity_df = pandas.read_csv(sentiment_singularity_csv, '\n', names=['Singularity Runtime (Seconds)'])
native_df=native_df.drop([35])
singularity_df=singularity_df.drop([18])

df = native_df.join(singularity_df).dropna()
print(df.describe())
print('p-value:\t 0.05\n')
print('degrees of freedom:\t ~200\n')
print('Critical t-val:\t 1.980\n')

t_val_rel = stats.ttest_rel(df.loc[:,'Native Runtime (Seconds)'],df.loc[:,'Singularity Runtime (Seconds)'])
print(t_val_rel)
t_val_ind = stats.ttest_ind(df.loc[:,'Native Runtime (Seconds)'],df.loc[:,'Singularity Runtime (Seconds)'])
print(t_val_ind)

ax = plt.gca()

df.plot(kind='hist', y='Native Runtime (Seconds)', color='red', ax=ax)
df.plot(kind='hist', y='Singularity Runtime (Seconds)', color='blue', ax=ax)


plt.savefig('P100_Histogram.png')




