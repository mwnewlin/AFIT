#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:33:57 2019

@author: mnewlin
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import datasets, linear_model, metrics


import matplotlib as mpl
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.neighbors import KNeighborsClassifier
#Balanced RF Classifier
from imblearn.ensemble import BalancedRandomForestClassifier as BRF

from IPython.display import Markdown as md  #enable markdown within code cell
from IPython.display import display, Math, Latex

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import confusion_matrix
import time

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, precision_recall_curve, make_scorer,f1_score
from sklearn.metrics import precision_recall_curve as PRC

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist

from collections import Counter

from scipy.stats import norm
import scipy
from io import StringIO

import metrics
import ijson
import random




working_data_dir = '/run/media/mnewlin/_userdata/uhnds/'
original_netflow_data_dir = working_data_dir + 'network/extracted/'
original_netflow_file = 'netflow_day-02'

original_host_data_dir = working_data_dir + 'host/extracted/'
original_host_file = 'wls_day-02.json'
current_data_dir = working_data_dir + 'host/unconverted/real/'

N_COLS = 20

COLUMNS = ['AuthenticationPackage', 'Destination', 'DomainName', 'EventID',
       'FailureReason', 'LogHost', 'LogonID', 'LogonType',
       'LogonTypeDescription', 'ParentProcessID', 'ParentProcessName',
       'ProcessID', 'ProcessName', 'ServiceName', 'Source', 'Status',
       'SubjectDomainName', 'SubjectLogonID', 'SubjectUserName', 'UserName']

def calculate_tf(counter):
    num_words = sum(counter.values())
    tf_dict = {}
    for item in counter:
        tf_dict[item] = counter[item]/float(num_words)
    return tf_dict

def calculate_idf(docList):
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            word = str(word)
            if val > 0:
                if word in idfDict.keys():
                    idfDict[word] += 1
                else:
                    idfDict[word] = 1
    for word, val in idfDict.items():
        word = str(word)
        idfDict[word] = np.log10(N/float(val))
    return idfDict

def compute_tfidf(tf_dict, idf_dict):
    tfidf_dict = {}
    for word, val in tf_dict.items():
        tfidf_dict[word] = val * idf_dict[word]
    return tfidf_dict




def load_real_host_sample(sample_num, sample_length=1000):

    data_dir = 'samples_{}/'.format(sample_length)
    if sample_length < 10000:
        if sample_num >= 10000:
            return -1
    elif sample_length < 100000:
        if sample_num >= 2000:
            return -1
    else:
        if sample_num >= 2000:
            return -1
        
    load_file = original_host_file + '_sample_{}.txt'.format(sample_num)
    filename = current_data_dir + data_dir + load_file
    df = pd.read_csv(filename)
    df = df.drop(['Time'], axis=1)
    if df.columns.size < len(COLUMNS):
        for col in COLUMNS:
            if col not in df.columns:
                df.at[:,col] = np.NaN
    df.loc[:,'EventID'] = df.loc[:,'EventID'].astype(str)
    df = df.replace(to_replace=[np.NaN], value='none', regex=False)
    df = df.replace(to_replace=[r"[\$]+"], value='', regex=True)
    df = df.replace(to_replace=["0x0"], value='success', regex=False)
    data = np.array(df)
    return data


"""
    Function to create a set of samples of real or fake data
"""
def load_n_host_samples(sample_length=1000, num_samples=1000, random_state=69):
    sample_set = np.array([])
    sample_range = 0
    if sample_length <= 1000:
        sample_range = 64845
    elif sample_length <= 10000:
        sample_range = 2000
    elif sample_length <= 100000:
        sample_range = 1160
    # Seed random samples for repeatability    
    random.seed(a=random_state)
    sample_list = random.sample(range(sample_range), num_samples)
    for num in sample_list:
        data = load_real_host_sample(sample_length=sample_length, sample_num=num)
        sample_set = np.append(sample_set, data)
            
    sample_set = np.reshape(sample_set, newshape=(num_samples, sample_length, N_COLS))
    return sample_set


def create_host_samples(sample_length=1000, num_samples=1000, random_state=69):
    tfidf_data_dir = '/run/media/mnewlin/_userdata/uhnds/host/unconverted/real/tfidf/'
    work_dir = 'samples_{}/'.format(sample_length)
    
    
    for i in range(num_samples):
        original_data = load_real_host_sample(sample_num=i, sample_length=sample_length)
        #print(original_data.shape)
        if i%1000 == 0:
            print(i)
        tfidf_converted_data = tfidf_convert_host_sample(original_data)
        tfidf_data = pd.DataFrame(data=tfidf_converted_data, columns=COLUMNS )
        outfile = tfidf_data_dir + work_dir+ 'tfidf_sample_{}.csv'.format(i)
        tfidf_data.to_csv(outfile, index=False)






def tfidf_convert_host_sample(sample):
    
    doc = sample
    docList = []
    doc_tfs = []
    for i in range(sample.shape[1]):
        d = doc[:,i].astype(str)
        tf_col = calculate_tf(Counter(d))
        doc_tfs.append(tf_col)
        docList.append(dict(tf_col))
    idf_doc = calculate_idf(docList)
    doc_tfidfs = []
    for tf in doc_tfs:
        doc_tfidf = compute_tfidf(tf, idf_doc)
        doc_tfidfs.append(doc_tfidf)
    tfidf_vals = np.zeros(doc.shape)
    for j in range(tfidf_vals.shape[1]):
        curr_col = doc[:,j].astype(str)
        for i in range(tfidf_vals.shape[0]):
            col_tfidf = doc_tfidfs[j]
            tfidf_vals[i,j] = col_tfidf[curr_col[i]]    
    return tfidf_vals

    


"""
    Creates real samples of data based on number of samples and desired sample length.
"""
def create_real_samples_json(directory='host', original_file=original_host_file, num_samples=1000, sample_length=1000, random_seed=69):
    curr_chunk=0
    if directory == 'host':
        reader = pd.read_json(original_host_data_dir+original_file, chunksize=sample_length, lines=True, orient='records')
        for chunk in reader:
            
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
            outpath = data_dir + directory + '/unconverted/real/' + out_dir + outfile 
            real_df.to_csv(outpath, sep=',', index=False)
            curr_chunk += 1
            #if curr_chunk >= num_samples:
                #break
    
    return 0

create_host_samples(num_samples=20000)
print("Finished")

