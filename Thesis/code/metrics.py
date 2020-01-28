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
#from scipy.spatial.distance import jensonshannon as js

from scipy.stats import wasserstein_distance as wasserstein
from scipy.special import rel_entr

from scipy.stats import norm, entropy
from scipy.stats.mstats import gmean
import time


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
    0 --> X and Y are the same
    1 --> X and Y are orthogonal
"""
def cosine_similarity(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    num_cols = X.shape[1]
    cos_sims = np.array([])
    for i in range(num_cols):
        cos_sim = cosine(X[:,i], Y[:,i])
        cos_sims = np.append(cos_sims, cos_sim)
        cos_sims = np.nan_to_num(cos_sims)
        cos_sims = np.where(cos_sims > 1, 1, cos_sims)
    return np.mean(cos_sims)
    
"""
    Calculates the Mahalanobis distance between 2 matrices X and Y
"""  
def mahalanobis_distance(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    stack = np.vstack([X, Y])
    VI = np.linalg.pinv(np.cov(stack, rowvar=False))
    d_mat = cdist(X,Y, metric='mahalanobis', VI=VI)
    return np.trace(d_mat)

def alt_mahalanobis(X,Y):
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

def KL(P,Q, eps=1e-5):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = eps

    # You may want to instead make copies to avoid changing the np arrays.
    P_prime = np.where(P==0, P+epsilon, P)
    Q_prime = np.where(Q==0, Q+epsilon, Q)
    

    divergence = np.sum(np.multiply(P_prime,np.log(P_prime/Q_prime)))
    return divergence

"""
    Wrapper function for scipy jenson shannon divergence
    https://scipy.github.io/devdocs/generated/scipy.spatial.distance.jensenshannon.html
    As of writing this file, this is still in a dev version of scipy so 
    this function was copied out of scipy source github at
    https://github.com/scipy/scipy/blob/089e3b2/scipy/spatial/distance.py#L1235-L1292
"""
def jensenshannon(p, q, base=None):
    """
    Compute the Jensen-Shannon distance (metric) between
    two 1-D probability arrays. This is the square root
    of the Jensen-Shannon divergence.
    The Jensen-Shannon distance between two probability
    vectors `p` and `q` is defined as,
    .. math::
       \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
    where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
    and :math:`D` is the Kullback-Leibler divergence.
    This routine will normalize `p` and `q` if they don't sum to 1.0.
    Parameters
    ----------
    p : (N,) array_like
        left probability vector
    q : (N,) array_like
        right probability vector
    base : double, optional
        the base of the logarithm used to compute the output
        if not given, then the routine uses the default base of
        scipy.stats.entropy.
    Returns
    -------
    js : double
        The Jensen-Shannon distance between `p` and `q`
    .. versionadded:: 1.2.0
    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
    1.0
    >>> distance.jensenshannon([1.0, 0.0], [0.5, 0.5])
    0.46450140402245893
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    0.0
    """
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=0)
    q = q / np.sum(q, axis=0)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    js = np.sum(left, axis=0) + np.sum(right, axis=0)
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)
    

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
    return gmean(np.where(wass_dists==0,1,wass_dists))

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
    mmd = np.sqrt(np.abs(a))
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
    return np.mean(dist_matrix), np.std(dist_matrix), dist_matrix   

"""
    Score two sets of samples based on a given metric
"""
def time_score_set(S1, S2, sample_length, num_samples, metric='lp', p=2, r=2, standardized=True, G1=None, G2=None):
    dist_matrix = np.array([])
    t_start = -1.0
    t_end = -1.0
    if metric == 'lp':
        t_start = time.time()
        for x in range(num_samples):
            d = l_p_distance(S1[x], S2[x], p=p, r=r)
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'cosine':
        t_start = time.time()
        for x in range(num_samples):
            d = cosine_similarity(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'mahalanobis':
        t_start = time.time()
        for x in range(num_samples):
            d = mahalanobis_distance(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'chi_squared':
        t_start = time.time()
        for x in range(num_samples):
            d = chi_squared_dist(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'wasserstein':
        t_start = time.time()
        for x in range(num_samples):
            d = wasserstein_dist(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'fid':
        t_start = time.time()
        for x in range(num_samples):
            d = fid(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'entropy':
        t_start = time.time()
        for x in range(num_samples):
            d = calc_entropy(S1[x], S2[x], sample_length=sample_length, standardized=standardized)
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'perplexity':
        t_start = time.time()
        for x in range(num_samples):
            d = calc_perplexity(S1[x], S2[x], sample_length=sample_length, standardized=standardized)
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'bd':
        t_start = time.time()
        for x in range(num_samples):
            d = bhattacharyya(S1[x], S2[x])
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    elif metric == 'mmd':
        t_start = time.time()
        for x in range(num_samples):
            Mxx = distance(S1[x],S2[x], sqrt=True)
            Myy = distance(G1[x],G2[x], sqrt=True)
            Mxy = distance(S1[x], G1[x], sqrt=True)
            d = mmd(Mxx, Mxy, Myy, sigma=1)
            dist_matrix = np.append(dist_matrix, d)
        t_end = time.time()
    t_diff = t_end - t_start
    return dist_matrix, t_diff   