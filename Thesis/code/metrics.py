#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:03:47 2019

@author: mnewlin
"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.linalg import sqrtm

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist

from scipy.stats import wasserstein_distance as wasserstein

from scipy.stats import norm, entropy


"""
    Generates probabilities for matrices X and Y, assuming given distribution
    distribution defaults to normal (may add other distributions later)
    
"""
def generate_probs(X,Y, dist='norm'):
    X = np.array(X)
    Y = np.array(Y)
    num_rows = X.shape[0]
    num_cols = X.shape[1]
    norm_x = np.zeros((num_rows, num_cols))
    norm_y = np.zeros((num_rows, num_cols))
    if dist == 'norm':
        for j in range(num_cols):
            xj = X[:,j]

            prob_xj = norm.pdf(xj, loc=xj.mean(), scale=xj.var())
            norm_x[:,j] = prob_xj

            yj = Y[:,j]

            prob_yj = norm.pdf(yj, loc=yj.mean(), scale=yj.var())
            norm_y[:,j] = prob_yj
    return np.nan_to_num(norm_x), np.nan_to_num(norm_y)

"""
    Calculates the Power distance between two matrices X and Y
    Defaults to Euclidean Distance unless parameters p and r are provided
"""
def l_p_distance(X,Y,p=2,r=2):
    X = np.array(X)
    Y = np.array(Y)
    if (X.shape != Y.shape):
        print("Usage: Matrices must be the same shape.")
        return -1
    num_rows = X.shape[0]
    distances = np.zeros((num_rows,1))
    for i in range(num_rows):
        x = X[i]
        y = Y[i]
        distances[i] = np.power(np.sum(np.power(np.abs(x-y),p)),(1/r))
    
    return np.mean(np.nan_to_num(distances))

"""
    Calculates the cosine similarity between two matrices X and Y
"""
def cosine_similarity(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    num_cols = X.shape[1]
    cos_sims = np.array([])
    for i in range(num_cols):
        x = X[:,i]
        y = Y[:,i]
        cos_sim = cosine(x,y)
        cos_sims = np.append(cos_sims, cos_sim)
    return np.mean(np.nan_to_num(cos_sims))
    
"""
    Calculates the Mahalanobis distance between 2 matrices X and Y
"""  
def mahalanobis_distance(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    # Generate Positive Definite Matrix
    XXT = np.matmul(X,X.T)
    YYT = np.matmul(Y,Y.T)
    mahalanobis = cdist(XXT, YYT, 'mahalanobis')
    return np.mean(mahalanobis)    

"""
    Calculates the chi squared distance between 2 matrices X and Y
    This function relies on the generate_probs function to generate
    probabilities for the values of the matrices X and Y in order to calculate
    the chi-squared distance.
"""
def chi_squared_dist(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    epsilon = 1e-10
    num_cols = X.shape[1]
    prob_x, prob_y = generate_probs(X,Y)
    chi_squares = np.array([])
    for j in range(num_cols):
        chi_squares = np.append(chi_squares, np.sum(np.divide(np.power(prob_x[:,j]-prob_y[:,j], 2), prob_y[:,j]+epsilon)))
    return np.mean(np.nan_to_num(chi_squares))

def wasserstein_dist(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    #x_prob, y_prob = generate_probs(X,Y)
    num_cols = X.shape[1]
    wass_dists = np.array([])
    for x in range(num_cols):
        u = X[:, x]
        v = Y[:, x]
        d = wasserstein(u,v)
        wass_dists = np.append(wass_dists, d)
    return np.mean(np.nan_to_num(wass_dists))

"""
    Calculates the Difference in standardized entropy between two matrices X and Y
"""
def calc_entropy(X,Y, sample_length):
    X = np.array(X)
    Y = np.array(Y)
    norm_x,norm_y = generate_probs(X,Y)
    num_cols = X.shape[1]
    ents = np.array([])
    for j in range(num_cols):
        ent_x = np.nan_to_num(entropy(norm_x[:,j])/np.log(sample_length))
        ent_y = np.nan_to_num(entropy(norm_y[:,j])/np.log(sample_length))
        diff = np.abs(ent_x - ent_y)
        ents = np.append(ents, diff)
    return np.mean(ents)

"""
    Calculates the Difference in perplexity between two matrices X and Y
"""
def calc_perplexity(X,Y, sample_length):
    X = np.array(X)
    Y = np.array(Y)
    norm_x,norm_y = generate_probs(X,Y)
    num_cols = X.shape[1]
    perps = np.array([])
    for j in range(num_cols):
        ent_x = np.nan_to_num(entropy(norm_x[:,j])/np.log(sample_length))
        ent_y = np.nan_to_num(entropy(norm_y[:,j])/np.log(sample_length))
        perp_x = np.power(2,ent_x)
        perp_y = np.power(2,ent_y)
        diff = np.abs(perp_x - perp_y)
        perps = np.append(perps, diff)
    return np.mean(np.nan_to_num(perps))
 
"""
    Calculates the Frechet Inception Distance between matrices X and Y
    Implementation details taken from 
    https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
"""       
def fid(X,Y):
    X = np.nan_to_num(np.array(X))
    Y = np.nan_to_num(np.array(Y))
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    C_x = np.nan_to_num(np.cov(X, rowvar=False))
    C_y = np.nan_to_num(np.cov(Y, rowvar=False))
    ssdiff = np.sum(np.square(mu_x - mu_y))
    covmean = np.nan_to_num(np.abs(np.matmul(C_x,C_y)))
    
    score = ssdiff + np.trace(C_x + C_y - 2.0*np.sqrt(covmean))
    return score
"""
    Scores a set of samples pairwise based on the given metric
"""
def score_samples(data, sample_length, num_samples, metric='lp', p=2, r=2):
    # Reshape data into 3d array of n*l*w from 2d array of nl*w
    sample_list = np.reshape(np.array(data), (num_samples, sample_length, data.shape[1]))
    dist_matrix = np.zeros((num_samples, num_samples))
    if metric == 'lp':
        # Do pairwise metrics
        for i in range(num_samples):
            for j in range(num_samples):
                d = l_p_distance(sample_list[i], sample_list[j], p=p, r=r)
                dist_matrix[i,j] = d
    elif metric == 'cosine':
        for i in range(num_samples):
            for j in range(num_samples):
                d = cosine_similarity(sample_list[i], sample_list[j])
                dist_matrix[i,j] = d
    elif metric == 'mahalanobis':
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    d = mahalanobis_distance(sample_list[i], sample_list[j])
                    dist_matrix[i,j] = d
    elif metric == 'chi_squared':
        for i in range(num_samples):
            for j in range(num_samples):
                d = chi_squared_dist(sample_list[i], sample_list[j])
                dist_matrix[i,j] = d
    elif metric == 'wasserstein':
        for i in range(num_samples):
            for j in range(num_samples):
                d = wasserstein_dist(sample_list[i], sample_list[j])
                dist_matrix[i,j] = d
    return np.sum(dist_matrix)

"""
    Score two sets of samples based on a given metric
"""
def score_set(S1, S2, sample_length, num_samples, metric='lp', p=2, r=2):
    dist_matrix = np.array([])
    if metric == 'lp':
        for x in range(num_samples):
            d = l_p_distance(S1[x], S2[x], p=p, r=r)
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'cosine':
        for x in range(num_samples):
            d = cosine_similarity(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'mahalanobis':
        for x in range(num_samples):
            d = mahalanobis_distance(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'chi_squared':
        for x in range(num_samples):
            d = chi_squared_dist(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'wasserstein':
        for x in range(num_samples):
            d = wasserstein_dist(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'fid':
        for x in range(num_samples):
            d = fid(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'entropy':
        for x in range(num_samples):
            d = calc_entropy(S1[x], S2[x], sample_length=sample_length)
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'perplexity':
        for x in range(num_samples):
            d = calc_perplexity(S1[x], S2[x], sample_length=sample_length)
            dist_matrix = np.append(dist_matrix, d)
    return np.mean(dist_matrix)   

"""
    Score two sets of samples based on a given metric, treating each collection
    as a giant sample
"""
def score_set_all(S1, S2, sample_length, num_samples, metric='lp', p=2, r=2):
    dist_matrix = np.array([])
    if metric == 'lp':
        d = l_p_distance(S1, S2, p=p, r=r)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'cosine':
        d = cosine_similarity(S1, S2)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'mahalanobis':
        d = mahalanobis_distance(S1, S2)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'chi_squared':
        d = chi_squared_dist(S1, S2)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'wasserstein':
        d = wasserstein_dist(S1, S2)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'fid':
        d = fid(S1, S2)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'entropy':
        d = calc_entropy(S1, S2, sample_length=sample_length*num_samples)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'perplexity':
        d = calc_perplexity(S1, S2, sample_length=sample_length*num_samples)
        dist_matrix = np.append(dist_matrix, d)
    return np.mean(dist_matrix)   