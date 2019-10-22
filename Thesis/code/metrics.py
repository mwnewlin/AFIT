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
    X = np.nan_to_num(np.array(X))
    Y = np.nan_to_num(np.array(Y))
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
    norm_x = np.nan_to_num(norm_x)
    norm_y = np.nan_to_num(norm_y)
    return norm_x, norm_y

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
    num_cols = X.shape[1]
    distances = np.zeros((num_cols,1))
    for i in range(num_cols):
        x = X[:,i]
        y = Y[:,i]
        diff = np.abs(x-y)
        distances[i] = np.power(np.sum(np.power(diff,p)),(1/r))
    
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
        cos_sim = cosine(X[:,i], Y[:,i])
        cos_sims = np.append(cos_sims, cos_sim)
    return np.mean(np.nan_to_num(cos_sims))
    
"""
    Calculates the Mahalanobis distance between 2 matrices X and Y
    In order to ensure non-singularity for arbitrary matrices X and Y
    Actual Mahalanobis Distance calculations are between X*X^T and Y*Y^T
"""  
def mahalanobis_distance(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    prob_x, prob_y = generate_probs(X,Y)
    # Generate Positive Definite Matrix
    #XXT = np.matmul(X.T,X)
    #YYT = np.matmul(Y.T,Y)
    #stack = np.vstack([XXT, YYT])
    #VI = np.linalg.pinv(np.cov(stack, rowvar=False))
    #mahalanobis = cdist(XXT, YYT, 'mahalanobis', VI=VI)
    stack = np.vstack([prob_x, prob_y])
    VI = np.linalg.pinv(np.cov(stack, rowvar=False))
    mahalanobis = cdist(prob_x, prob_y, 'mahalanobis', VI=VI)
    return np.mean(np.nan_to_num(mahalanobis))

def alt_mahalanobis(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    stack = np.vstack([X, Y])
    VI = np.linalg.pinv(np.cov(stack, rowvar=False))
    mahalanobis = cdist(X, Y, 'mahalanobis', VI=VI)
    return np.mean(np.nan_to_num(mahalanobis))

"""
    Calculates the chi squared distance between 2 matrices X and Y
    This function relies on the generate_probs function to generate
    probabilities for the values of the matrices X and Y in order to calculate
    the chi-squared distance.
"""
def chi_squared_dist(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    num_cols = X.shape[1]
    prob_x, prob_y = generate_probs(X,Y)
    chi_squares = np.array([])
    for j in range(num_cols):
        prob_yj = prob_y[:,j]
        epsilon = 1e-6
        prob_yj = np.where(prob_yj == 0, epsilon, prob_yj)
        chi_squares = np.append(chi_squares, np.sum(np.divide(np.square(prob_x[:,j]-prob_y[:,j]), prob_yj)))
    return np.mean(np.nan_to_num(chi_squares))
"""
    Calculate the Wasserstein Distance between Matrices X and Y
"""
def wasserstein_dist(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    x_prob, y_prob = generate_probs(X,Y)
    num_cols = X.shape[1]
    wass_dists = np.array([])
    for x in range(num_cols):
        u = x_prob[:, x]
        v = y_prob[:, x]
        d = wasserstein(u,v)
        wass_dists = np.append(wass_dists, d)
    return np.mean(np.nan_to_num(wass_dists))

"""
    Calculates the Difference in standardized entropy between two matrices X and Y
"""
def calc_entropy(X,Y, sample_length, standardized=True):
    sample_size = 1
    if standardized:
        sample_size = np.log(sample_length)
    X = np.array(X)
    Y = np.array(Y)
    norm_x,norm_y = generate_probs(X,Y)
    num_cols = X.shape[1]
    ents = np.array([])
    for j in range(num_cols):
        ent_x = np.nan_to_num(entropy(norm_x[:,j])/sample_size)
        ent_y = np.nan_to_num(entropy(norm_y[:,j])/sample_size)
        diff = np.abs(ent_x - ent_y)
        ents = np.append(ents, diff)
    return np.mean(np.nan_to_num(ents))

"""
    Calculates the Difference in perplexity between two matrices X and Y
"""
def calc_perplexity(X,Y, sample_length, standardized=True):
    sample_size = 1
    if standardized:
        sample_size = np.log(sample_length)
    X = np.array(X)
    Y = np.array(Y)
    norm_x,norm_y = generate_probs(X,Y)
    num_cols = X.shape[1]
    perps = np.array([])
    for j in range(num_cols):
        ent_x = np.nan_to_num(entropy(norm_x[:,j])/sample_size)
        ent_y = np.nan_to_num(entropy(norm_y[:,j])/sample_size)
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
    X = np.array(X)
    Y = np.array(Y)
    prob_x, prob_y = generate_probs(X,Y)
    mu_x = np.mean(prob_x, axis=0)
    mu_y = np.mean(prob_y, axis=0)
    
    Cx = np.cov(prob_x,rowvar=False)
    Cy = np.cov(prob_y, rowvar=False)
    ssdiff = np.sum(np.square(mu_x-mu_y))
    covmean = scipy.linalg.sqrtm(Cx.dot(Cy))
    score = ssdiff + np.trace(Cx + Cy - 2.0*covmean)
    return np.abs(score)
"""
    Calculates (X-Y)^2 for matrices X and Y 
    Returns distance matrix M
"""
def distance(X,Y, sqrt=False):
    X = np.array(X)
    Y = np.array(Y)
    X2 = np.matmul(X,X.T)
    Y2 = np.matmul(Y,Y.T)
    XY = np.matmul(X,Y.T)
    M = X2+Y2-2*XY
    if sqrt:
       M = np.sqrt(np.abs(M))
    return M

"""
    Calculated the Maximum Mean Discrepancy between
    real and fake distributions using the Gaussian Kernel (RBF)
"""
def mmd(Mxx,Mxy, Myy, sigma):
    mu = np.mean(Mxx)
    Mxx = np.nan_to_num(np.exp(np.divide(-Mxx,mu*2*sigma*sigma)))
    Mxy = np.nan_to_num(np.exp(np.divide(-Mxy,mu*2*sigma*sigma)))
    Myy = np.nan_to_num(np.exp(np.divide(-Myy,mu*2*sigma*sigma)))
    a = Mxx.mean() + Myy.mean() - 2*Mxy.mean()
    mmd = np.sqrt(np.maximum(a,0))
    return mmd

"""
    Calculates the Bhattacharyya distance between X and Y
"""
def bhattacharyya(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    num_cols = X.shape[1]
    prob_x, prob_y = generate_probs(X,Y)
    dist = np.array([])
    for j in range(num_cols):
        x = prob_x[:,j]
        y = prob_y[:,j]
        bc = np.sum(np.sqrt(np.multiply(x,y)))
        bd = -np.log(bc)
        dist = np.append(dist, bd)
    return np.mean(np.nan_to_num(dist))

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
def score_set(S1, S2, sample_length, num_samples, metric='lp', p=2, r=2, standardized=True, G1=None, G2=None):
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
            d = calc_entropy(S1[x], S2[x], sample_length=sample_length, standardized=standardized)
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'perplexity':
        for x in range(num_samples):
            d = calc_perplexity(S1[x], S2[x], sample_length=sample_length, standardized=standardized)
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'bd':
        for x in range(num_samples):
            d = bhattacharyya(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
    elif metric == 'mmd':
        for x in range(num_samples):
            Mxx = distance(S1[x],S2[x], sqrt=True)
            Myy = distance(G1[x],G2[x], sqrt=True)
            Mxy = distance(S1[x], G1[x], sqrt=True)
            d = mmd(Mxx, Mxy, Myy, sigma=1)
            dist_matrix = np.append(dist_matrix, d)
    return np.mean(dist_matrix)   

"""
    Score two sets of samples based on a given metric, treating each collection
    as a giant sample
"""
def score_set_all(S1, S2, sample_length, num_samples, metric='lp', p=2, r=2, standardized=True):
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
        d = calc_entropy(S1, S2, sample_length=sample_length*num_samples, standardized=standardized)
        dist_matrix = np.append(dist_matrix, d)
    elif metric == 'perplexity':
        d = calc_perplexity(S1, S2, sample_length=sample_length*num_samples, standardized=standardized)
        dist_matrix = np.append(dist_matrix, d)
    return np.mean(dist_matrix)   