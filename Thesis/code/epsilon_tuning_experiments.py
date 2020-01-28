#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 07:50:39 2019

@author: mnewlin
"""
import os
import numpy as np
import matplotlib.pyplot as plt

#make plots inline using jupyter magic
#%matplotlib inline

import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import datasets, linear_model, metrics


import matplotlib as mpl
import seaborn as sns

import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.neighbors import KNeighborsClassifier
#Balanced RF Classifier
from imblearn.ensemble import BalancedRandomForestClassifier as BRF

#from IPython.display import Markdown as md  #enable markdown within code cell
#from IPython.display import display, Math, Latex

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import confusion_matrix
import time
import random
import scipy

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, precision_recall_curve, make_scorer,f1_score
from sklearn.metrics import precision_recall_curve as PRC
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from numpy.fft import fftn

## Homemade code imports
import metrics

data_dir = '/run/media/mnewlin/_userdata/uhnds/'
original_netflow_data_dir = data_dir + 'network/extracted/'
original_netflow_file = 'netflow_day-02'
fake_dir = '/run/media/mnewlin/_userdata/uhnds/network/converted/fake/'
real_dir = '/run/media/mnewlin/_userdata/uhnds/network/converted/real/'
real_file = 'netflow_day-02'
# Dataset dependent number of cols
N_COLS = 10
N_HOST_COLS = 10
np.seterr(all='ignore')

"""
    Function to read in a single real sample from a given directory based
    on the desired length of the sample.
"""
def load_real_sample(sample_num, sample_length=100):

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
        
    load_file = real_file + '_sample_{}.txt'.format(sample_num)
    filename = real_dir + data_dir + load_file
    df = pd.read_csv(filename, names=['Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], sep=' ', dtype=np.float64)
    data = np.array(df)
    return data

"""
    Function to read in a single fake sample from a given directory based
    on the desired length of the sample.
"""
def load_fake_sample(sample_num, sample_length=100):
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

    load_file = real_file + '_random_sample_{}.txt'.format(sample_num)
    filename = fake_dir + data_dir + load_file
    df = pd.read_csv(filename, names=['Duration', 'SrcDevice', 
            'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 
            'SrcBytes', 'DstBytes'], sep=' ', dtype=np.float64)
    data = np.array(df)
    return data
"""
    Function to create a set of samples of real or fake data
"""
def load_n_samples(real=True, sample_length=100, num_samples=100, random_state=69):
    sample_set = np.array([])
    sample_range = 0
    if sample_length <= 1000:
        sample_range = 10000
    elif sample_length <= 10000:
        sample_range = 2000
    elif sample_length <= 100000:
        sample_range = 1160
    # Seed random samples for repeatability    
    random.seed(a=random_state)
    sample_list = random.sample(range(sample_range), num_samples)
    if real:
        for num in sample_list:
            data = load_real_sample(sample_length=sample_length, sample_num=num)
            sample_set = np.append(sample_set, data)
    else:
        for num in sample_list:
            data = load_fake_sample(sample_length=sample_length, sample_num=num)
            sample_set = np.append(sample_set, data)
    sample_set = np.reshape(sample_set, newshape=(num_samples, sample_length, N_COLS))
    return sample_set

def load_real_host_sample(sample_num, sample_length=1000):
    directory = '/run/media/mnewlin/_userdata/uhnds/host/unconverted/real/tfidf/'
    real_host_data_dir = directory + 'samples_{}/'.format(sample_length)
        
    load_file = real_host_data_dir + 'tfidf_sample_{}.csv'.format(sample_num)
    df = pd.read_csv(load_file, dtype=np.float64)
    data = np.array(df)
    return data

def load_fake_host_sample(sample_num, sample_length=1000,dist='uniform'):
    directory = '/run/media/mnewlin/_userdata/uhnds/host/unconverted/fake/{}/'.format(dist)
    
    fake_host_data_dir = directory + 'samples_{}/'.format(sample_length)
        
    load_file = fake_host_data_dir + 'tfidf_sample_{}.csv'.format(sample_num)
    df = pd.read_csv(load_file, dtype=np.float64)
    data = np.array(df)
    return data

def load_n_host_samples(real=True, sample_length=100, num_samples=100, random_state=69, dist='uniform'):

    sample_set = np.array([])
    sample_range= 10000
    random.seed(a=random_state)
    sample_list = random.sample(range(sample_range), num_samples)
    if real:
        for num in sample_list:
            data = load_real_host_sample(sample_length=sample_length, sample_num=num)
            sample_set = np.append(sample_set, data)
    else:
        for num in sample_list:
            data = None
            if dist == 'uniform':
                data = load_fake_host_sample(sample_length=sample_length, sample_num=num, dist='uniform')
            else:
                data = load_fake_host_sample(sample_length=sample_length, sample_num=num, dist='normal')
            sample_set = np.append(sample_set, data)
    sample_set = np.reshape(sample_set, newshape=(num_samples, sample_length, N_COLS))
        
    return sample_set

"""
    Function to create a mix of real and fake data
"""
def create_sample_mix(ratio, sample_length=100, num_samples=100, random_state=69):
    sample_range = 0
    if sample_length <= 1000:
        sample_range = 10000
    elif sample_length <= 10000:
        sample_range = 2000
    elif sample_length <= 100000:
        sample_range = 1160
    
    #mix_set = np.zeros((num_samples, sample_length, N_COLS))
    bound_val_real = np.around(((1-ratio)*num_samples), decimals=2)
    bound_val_fake = np.around((ratio)*num_samples, decimals=2)
    bound_val_real = int(bound_val_real) # How many real samples there should be
    bound_val_fake = int(bound_val_fake) # How many fake samples there should be
    
    real_data = load_n_samples(real=True, num_samples=num_samples, 
                               sample_length=sample_length, random_state=random_state) 
    fake_data = load_n_samples(real=False, num_samples=num_samples, 
                               sample_length=sample_length, random_state=random_state)
    real_section = real_data[:bound_val_real]
    fake_section = fake_data[:bound_val_real]
    mix_set = np.append(real_section, fake_section)
    mix_set = np.reshape(mix_set, newshape=(num_samples, sample_length, N_COLS))
    return mix_set

"""
    Gets the scores between real-real samples and real-fake samples
    Assumes that data passed in has all transforms completed
"""
def get_scores(real_set_1, real_set_2, real_set_3,real_set_4, fake_set_1, fake_set_2,
               num_samples, sample_length, pca=False):
    
    
    _,_,d_mat_rr_euc = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=2,r=2)
    _,_,d_mat_rf_euc = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=2,r=2)
    
    _,_,d_mat_rr_lp1 = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=0.5,r=0.5)
    _,_,d_mat_rf_lp1 = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=0.5,r=0.5)
    
    _,_,d_mat_rr_lp2 = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=0.75,r=0.75)
    _,_,d_mat_rf_lp2 = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=0.75,r=0.75)
    
    _,_,d_mat_rr_man = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=1,r=1)
    _,_,d_mat_rf_man = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='lp', p=1,r=1)
    
    _,_,d_mat_rr_cos = metrics.score_set(S1=np.abs(real_set_1), S2=np.abs(real_set_2), 
        num_samples=num_samples, sample_length=sample_length, metric='cosine')
    _,_,d_mat_rf_cos = metrics.score_set(S1=np.abs(real_set_1), S2=np.abs(fake_set_1), 
        num_samples=num_samples, sample_length=sample_length, metric='cosine')
    
    _,_,d_mat_rr_mah = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='mahalanobis')
    _,_,d_mat_rf_mah = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='mahalanobis')
    
    _,_,d_mat_rr_wass = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='wasserstein')
    _,_,d_mat_rf_wass = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='wasserstein')
    
    _,_,d_mat_rr_ent = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='entropy', standardized=False)
    _,_,d_mat_rf_ent = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='entropy', standardized=False)
    
    _,_,d_mat_rr_perp = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        num_samples=num_samples, sample_length=sample_length, metric='perplexity', standardized=False)
    _,_,d_mat_rf_perp = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
        num_samples=num_samples, sample_length=sample_length, metric='perplexity', standardized=False)
    
    _,_,d_mat_rr_mmd = metrics.score_set(S1=real_set_1, S2=real_set_2,
        G1=real_set_3, G2=real_set_4, num_samples=num_samples, 
        sample_length=sample_length, metric='mmd')
    _,_,d_mat_rf_mmd = metrics.score_set(S1=real_set_1, S2=real_set_2, 
        G1=fake_set_1, G2=fake_set_2, num_samples=num_samples, 
        sample_length=sample_length, metric='mmd')
    d_mat_rr_fid = np.zeros((num_samples,))
    d_mat_rf_fid = np.zeros((num_samples,))
    if pca:
        _,_,d_mat_rr_fid = metrics.score_set(S1=real_set_1, S2=real_set_2, 
            num_samples=num_samples, sample_length=sample_length, metric='fid')
        _,_,d_mat_rf_fid = metrics.score_set(S1=real_set_1, S2=fake_set_1, 
            num_samples=num_samples, sample_length=sample_length, metric='fid')
        
    names = ['Manhattan', 'Euclidean', 'lp: p=r=0.5', 'lp: p=r=0.75', 'cosine', 
        'mahalanobis', 'wasserstein', 'entropy', 'perplexity', 'mmd', 'fid']
    real_data = pd.DataFrame(columns=names)
    fake_data = pd.DataFrame(columns=names)
    real_data.loc[:,'Manhattan'] = d_mat_rr_man
    real_data.loc[:,'Euclidean'] = d_mat_rr_euc
    real_data.loc[:,'lp: p=r=0.5'] = d_mat_rr_lp1
    real_data.loc[:,'lp: p=r=0.75'] = d_mat_rr_lp2
    real_data.loc[:,'cosine'] = d_mat_rr_cos
    real_data.loc[:,'mahalanobis'] = d_mat_rr_mah
    real_data.loc[:,'wasserstein'] = d_mat_rr_wass
    real_data.loc[:,'entropy'] = d_mat_rr_ent
    real_data.loc[:,'perplexity'] = d_mat_rr_perp
    real_data.loc[:,'mmd'] = d_mat_rr_mmd
    real_data.loc[:,'fid'] = d_mat_rr_fid
    
    fake_data.loc[:,'Manhattan'] = d_mat_rf_man
    fake_data.loc[:,'Euclidean'] = d_mat_rf_euc
    fake_data.loc[:,'lp: p=r=0.5'] = d_mat_rf_lp1
    fake_data.loc[:,'lp: p=r=0.75'] = d_mat_rf_lp2
    fake_data.loc[:,'cosine'] = d_mat_rf_cos
    fake_data.loc[:,'mahalanobis'] = d_mat_rf_mah
    fake_data.loc[:,'wasserstein'] = d_mat_rf_wass
    fake_data.loc[:,'entropy'] = d_mat_rf_ent
    fake_data.loc[:,'perplexity'] = d_mat_rf_perp
    fake_data.loc[:,'mmd'] = d_mat_rf_mmd
    fake_data.loc[:,'fid'] = d_mat_rf_fid
    return real_data, fake_data

def untransformed_sets(sample_length=1000, num_samples=1000, random_state=69):
    real_set_1 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)
    real_set_2 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)
    fake_set_1 = load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)
    fake_set_2 = load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)
    real_set_3 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=3*random_state)
    real_set_4 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=4*random_state)
    
    r_scaler = RobustScaler()
    
    for sample in range(num_samples):
        real_set_1[sample] = r_scaler.fit_transform(real_set_1[sample])
        real_set_2[sample] = r_scaler.fit_transform(real_set_2[sample])
        fake_set_1[sample] = r_scaler.fit_transform(fake_set_1[sample])
        fake_set_2[sample] = r_scaler.fit_transform(fake_set_2[sample])
        real_set_3[sample] = r_scaler.fit_transform(real_set_3[sample])
        real_set_4[sample] = r_scaler.fit_transform(real_set_4[sample])
    return real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2

def sqrt_sets(sample_length=1000, num_samples=1000, random_state=69):
    real_set_1 = np.sqrt(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state))
    real_set_2 = np.sqrt(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state))
    fake_set_1 = np.sqrt(load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state))
    fake_set_2 = np.sqrt(load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state))
    real_set_3 = np.sqrt(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=3*random_state))
    real_set_4 = np.sqrt(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=4*random_state))
    
    r_scaler = RobustScaler()
    
    for sample in range(num_samples):
        real_set_1[sample] = r_scaler.fit_transform(real_set_1[sample])
        real_set_2[sample] = r_scaler.fit_transform(real_set_2[sample])
        fake_set_1[sample] = r_scaler.fit_transform(fake_set_1[sample])
        fake_set_2[sample] = r_scaler.fit_transform(fake_set_2[sample])
        real_set_3[sample] = r_scaler.fit_transform(real_set_3[sample])
        real_set_4[sample] = r_scaler.fit_transform(real_set_4[sample])
    return real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2

def log_sets(sample_length=1000, num_samples=1000, random_state=69):
    real_set_1 = np.log(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)+1)
    real_set_2 = np.log(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)+1)
    fake_set_1 = np.log(load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)+1)
    fake_set_2 = np.log(load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)+1)
    real_set_3 = np.log(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=3*random_state)+1)
    real_set_4 = np.log(load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=4*random_state)+1)
    
    r_scaler = RobustScaler()
    
    for sample in range(num_samples):
        real_set_1[sample] = r_scaler.fit_transform(real_set_1[sample])
        real_set_2[sample] = r_scaler.fit_transform(real_set_2[sample])
        fake_set_1[sample] = r_scaler.fit_transform(fake_set_1[sample])
        fake_set_2[sample] = r_scaler.fit_transform(fake_set_2[sample])
        real_set_3[sample] = r_scaler.fit_transform(real_set_3[sample])
        real_set_4[sample] = r_scaler.fit_transform(real_set_4[sample])
    return real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2

def pca_sets(sample_length=1000, num_samples=1000, random_state=69):
    real_set_1_ = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)
    real_set_2_ = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)
    fake_set_1_ = load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)
    fake_set_2_ = load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)
    real_set_3_ = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=3*random_state)
    real_set_4_ = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=4*random_state)
    
    r_scaler = RobustScaler()
    
    for sample in range(num_samples):
        real_set_1_[sample] = r_scaler.fit_transform(real_set_1_[sample])
        real_set_2_[sample] = r_scaler.fit_transform(real_set_2_[sample])
        fake_set_1_[sample] = r_scaler.fit_transform(fake_set_1_[sample])
        fake_set_2_[sample] = r_scaler.fit_transform(fake_set_2_[sample])
        real_set_3_[sample] = r_scaler.fit_transform(real_set_3_[sample])
        real_set_4_[sample] = r_scaler.fit_transform(real_set_4_[sample])
        
    real_set_1 = np.zeros((num_samples, N_COLS, N_COLS))
    real_set_2 = np.zeros((num_samples, N_COLS, N_COLS))
    fake_set_1 = np.zeros((num_samples, N_COLS, N_COLS))
    fake_set_2 = np.zeros((num_samples, N_COLS, N_COLS))
    real_set_3 = np.zeros((num_samples, N_COLS, N_COLS))
    real_set_4 = np.zeros((num_samples, N_COLS, N_COLS))
    
    pca = PCA()
    for x in range(num_samples):
        pca.fit(real_set_1_[x])
        real_set_1[x] = pca.components_
        
        pca.fit(real_set_2_[x])
        real_set_2[x] = pca.components_
        
        pca.fit(fake_set_1_[x])
        fake_set_1[x] = pca.components_
        
        pca.fit(fake_set_2_[x])
        fake_set_2[x] = pca.components_
        
        pca.fit(real_set_3_[x])
        real_set_3[x] = pca.components_
        
        pca.fit(real_set_4_[x])
        real_set_4[x] = pca.components_ 
    return real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2

def fft_sets(sample_length=1000, num_samples=1000, random_state=69):
    real_set_1 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)
    real_set_2 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)
    fake_set_1 = load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=random_state)
    fake_set_2 = load_n_samples(real=False, num_samples=num_samples, 
                                sample_length=sample_length, random_state=2*random_state)
    real_set_3 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=3*random_state)
    real_set_4 = load_n_samples(real=True, num_samples=num_samples, 
                                sample_length=sample_length, random_state=4*random_state)
    
    # Fourier Transform
    for x in range(num_samples):
        real_set_1[x] = fftn(real_set_1[x]).real
        real_set_2[x] = fftn(real_set_2[x]).real
        fake_set_1[x] = fftn(fake_set_1[x]).real
        fake_set_2[x] = fftn(fake_set_2[x]).real
        real_set_3[x] = fftn(real_set_3[x]).real
        real_set_4[x] = fftn(real_set_4[x]).real
    
    r_scaler = RobustScaler()
    
    for sample in range(num_samples):
        real_set_1[sample] = r_scaler.fit_transform(real_set_1[sample])
        real_set_2[sample] = r_scaler.fit_transform(real_set_2[sample])
        fake_set_1[sample] = r_scaler.fit_transform(fake_set_1[sample])
        fake_set_2[sample] = r_scaler.fit_transform(fake_set_2[sample])
        real_set_3[sample] = r_scaler.fit_transform(real_set_3[sample])
        real_set_4[sample] = r_scaler.fit_transform(real_set_4[sample])
    return real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2

def run_untrans(num_samples=1000, sample_length=1000, random_state=69):
    real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2 = untransformed_sets(num_samples=num_samples,
                sample_length=sample_length, random_state=random_state)
    real_data, fake_data = get_scores(real_set_1, real_set_2, real_set_3,real_set_4, fake_set_1, fake_set_2,
               num_samples, sample_length, pca=False)
    return real_data, fake_data
def run_sqrt(num_samples=1000, sample_length=1000, random_state=69):
    real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2 = sqrt_sets(num_samples=num_samples,
                sample_length=sample_length, random_state=random_state)
    real_data, fake_data = get_scores(real_set_1, real_set_2, real_set_3,real_set_4, fake_set_1, fake_set_2,
               num_samples, sample_length, pca=False)
    return real_data, fake_data
def run_log(num_samples=1000, sample_length=1000, random_state=69):
    real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2 = log_sets(num_samples=num_samples,
                sample_length=sample_length, random_state=random_state)
    real_data, fake_data = get_scores(real_set_1, real_set_2, real_set_3,real_set_4, fake_set_1, fake_set_2,
               num_samples, sample_length, pca=False)
    return real_data, fake_data
def run_pca(num_samples=1000, sample_length=1000, random_state=69):
    real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2 = pca_sets(num_samples=num_samples,
                sample_length=sample_length, random_state=random_state)
    real_data, fake_data = get_scores(real_set_1, real_set_2, real_set_3,real_set_4, fake_set_1, fake_set_2,
               num_samples, sample_length, pca=True)
    return real_data, fake_data
def run_fft(num_samples=1000, sample_length=1000, random_state=69):
    real_set_1, real_set_2, real_set_3, real_set_4, fake_set_1, fake_set_2 = fft_sets(num_samples=num_samples,
                sample_length=sample_length, random_state=random_state)
    real_data, fake_data = get_scores(real_set_1, real_set_2, real_set_3,real_set_4, fake_set_1, fake_set_2,
               num_samples, sample_length, pca=False)
    return real_data, fake_data


"""
    Actual Script running
"""
num_sample_list = [(n+1)*100 for n in range(10)]
#num_sample_list.insert(0,100)
# Adding Additional sample sizes up to 10k
#num_sample_list = [5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
NUM_SAMPLES = 1000
SAMPLE_LENGTH = 1000
RANDOM_STATE = 10
# Untransformed data
random_seeds = [1, 13, 42, 21, 33, 69, 56, 12, 27, 99]
real_data_untrans = pd.DataFrame()
fake_data_untrans = pd.DataFrame()
t_start = time.time()
for num_samples in num_sample_list:
    for random_state in random_seeds:
        real_data, fake_data = run_untrans(num_samples=num_samples, sample_length=SAMPLE_LENGTH,
                                       random_state=random_state)
        real_data_untrans = real_data_untrans.append(real_data)
        fake_data_untrans = fake_data_untrans.append(fake_data)
    outdir = '/run/media/mnewlin/_userdata/results_data/results/untrans/'
    real_data_untrans.to_csv(outdir+'real_data_exp_eff_{}.csv'.format(num_samples))
    fake_data_untrans.to_csv(outdir+'fake_data_exp_eff_{}.csv'.format(num_samples))
print("Untransformed")

real_data_sqrt = pd.DataFrame()
fake_data_sqrt = pd.DataFrame()
for num_samples in num_sample_list:
    for random_state in random_seeds:
        real_data, fake_data = run_sqrt(num_samples=num_samples, sample_length=SAMPLE_LENGTH,
                                       random_state=random_state)
        real_data_sqrt = real_data_sqrt.append(real_data)
        fake_data_sqrt = fake_data_sqrt.append(fake_data)
    outdir = '/run/media/mnewlin/_userdata/results_data/results/sqrt/'
    real_data_sqrt.to_csv(outdir+'real_data_exp_eff_{}.csv'.format(num_samples))
    fake_data_sqrt.to_csv(outdir+'fake_data_exp_eff_{}.csv'.format(num_samples))
print("Square Root")
t_elapsed = time.time()
t_so_far = t_elapsed - t_start
print("Total time elapsed: {:.2f}".format(t_so_far))
real_data_log = pd.DataFrame()
fake_data_log = pd.DataFrame()
for num_samples in num_sample_list:
    for random_state in random_seeds:
        real_data, fake_data = run_log(num_samples=num_samples, sample_length=SAMPLE_LENGTH,
                                       random_state=random_state)
        real_data_log = real_data_log.append(real_data)
        fake_data_log = fake_data_log.append(fake_data)
    outdir = '/run/media/mnewlin/_userdata/results_data/results/log/'
    real_data_log.to_csv(outdir+'real_data_exp_eff_{}.csv'.format(num_samples))
    fake_data_log.to_csv(outdir+'fake_data_exp_eff_{}.csv'.format(num_samples))
print("Log")
t_elapsed = time.time()
t_so_far = t_elapsed - t_start
print("Total time elapsed: {:.2f}".format(t_so_far))

real_data_pca = pd.DataFrame()
fake_data_pca = pd.DataFrame()
for num_samples in num_sample_list:
    for random_state in random_seeds:
        real_data, fake_data = run_pca(num_samples=num_samples, sample_length=SAMPLE_LENGTH,
                                       random_state=random_state)
        real_data_pca = real_data_pca.append(real_data)
        fake_data_pca = fake_data_pca.append(fake_data)
    outdir = '/run/media/mnewlin/_userdata/results_data/results/pca/'
    real_data_pca.to_csv(outdir+'real_data_exp_eff_{}.csv'.format(num_samples))
    fake_data_pca.to_csv(outdir+'fake_data_exp_eff_{}.csv'.format(num_samples))
print("PCA")
t_elapsed = time.time()
t_so_far = t_elapsed - t_start
print("Total time elapsed: {:.2f}".format(t_so_far))

real_data_fft = pd.DataFrame()
fake_data_fft = pd.DataFrame()
for num_samples in num_sample_list:
    for random_state in random_seeds:
        real_data, fake_data = run_fft(num_samples=num_samples, sample_length=SAMPLE_LENGTH,
                                       random_state=random_state)
        real_data_fft = real_data_fft.append(real_data)
        fake_data_fft = fake_data_fft.append(fake_data)
    outdir = '/run/media/mnewlin/_userdata/results_data/results/fft/'
    real_data_fft.to_csv(outdir+'real_data_exp_eff_{}.csv'.format(num_samples))
    fake_data_fft.to_csv(outdir+'fake_data_exp_eff_{}.csv'.format(num_samples))
print("FFT")

t_end = time.time()
t_diff = t_end - t_start
print("Total time: {:.2f}".format(t_diff))