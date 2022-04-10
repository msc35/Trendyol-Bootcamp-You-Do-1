# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 01:15:04 2022

@author: mehme
"""


import pandas as pd
import numpy as np

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


################ Similarity Matrix Creator Function ###########################
def sim(r:np.ndarray) -> np.ndarray:
    m,_ = r.shape
    
    s = np.zeros((m,m))
    
    mu = np.nanmean(r,axis=1) 
    
    for i in range(m):
        for j in range(m):
            mask  = ~np.isnan(r[i,:]) & ~np.isnan(r[j,:])
            
            num = np.dot(r[i,mask] - mu[i], r[j,mask] - mu[j])
            
            
            denum = np.linalg.norm(r[i,mask] - mu[i]) *  np.linalg.norm(r[j,mask] - mu[j])
            if denum == 0:
                s[i][j] = 0.0000000001
            else:
                s[i][j] = num/denum
         
    return s

def sim_item(r:np.ndarray) -> np.ndarray:
    _,m = r.shape
    
    s = np.zeros((m,m))
    
    mu = np.nanmean(r,axis=0) 
    
    for i in range(m):
        for j in range(m):
            mask  = ~np.isnan(r[:,i]) & ~np.isnan(r[:,j])
            
            num = np.dot(r[mask,i] - mu[i], r[mask,j] - mu[j])
            
            
            denum = np.linalg.norm(r[mask,i] - mu[i]) *  np.linalg.norm(r[mask,j] - mu[j])
            if denum == 0:
                s[i][j] = 0.0000000001
            else:
                s[i][j] = num/denum
            
    return s

######################### Read Data + Pivot Table "r" #########################

df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

df2 = df.sample(n = 25)

r = df2.pivot(index='user_id', columns='item_id', values='rating').values

R = r.copy()

print(R.shape)

####################### Similarity Matrix of USERS ############################
s = sim(R)

def top_n_similar(s,n):
    T = np.zeros((R_copy.shape[0], n))
    for i in range(s.shape[0]):
        ind = np.argpartition(s[:,i], -n-1)[-n-1:]
        ind = ind[ind != i]
        for j in range(n):
            T[i][j] = ind[j]
    return T.astype(int)


T = top_n_similar(s,5) #Top 5 Similar User indicies.

####################### Similarity Matrix of ITEMS ############################

s_item = sim_item(R)

def top_n_similar_item(s,n):
    T = np.zeros((R_copy.shape[1], n))
    for i in range(s.shape[1]):
        ind = np.argpartition(s[:,i], -n-1)[-n-1:]
        ind = ind[ind != i]
        for j in range(n):
            T[i][j] = ind[j]
    return T.astype(int)



R_copy = R.copy()


################## Top n Similar Rated Users Rated Item j #####################
def top_n_similar_rated_j(s,n,j): 
    T = np.zeros((R_copy.shape[0], n))
    rated_j = np.where(~np.isnan(R_copy[:,j]))
    for i in range(s.shape[0]):
        ind = np.argpartition(s[rated_j,i], -n-1)[-n-1:] # Indexler karışıyor, s[rated_j,i] kendi indexlerini yaratıyor.
        ind = ind[ind != i]
        for j in range(n):
            T[i][j] = ind[j]
    return T.astype(int)


def top_k_sim(s,u,j,k):
    rated_j = np.nonzero(~np.isnan(R_copy[:, j]))[0]
    topk_users = rated_j[s[u,rated_j].argsort()[::-1][:k]]
    return topk_users



################# User + Item Based Prediction Function #######################

def r_u_j(R, s, w_u, w_j ,u ,j, n=5):  # also gives weights
    mu = np.nanmean(R,axis=1)
    T = top_n_similar_rated_j(s, n, j)
    I = top_n_similar_item(s,n,u)
    zero_centered_score_user = R[T[u],j] - mu[T[u]]
    zero_centered_score_item = R[u,I[j]] 
    user_weight = w_u[T[u]][j]
    item_weight = w_j[j][I[j]]

    y_pred = np.dot(zero_centered_score_user,user_weight)/np.abs(user_weight).sum() + mu[u] + np.dot(zero_centered_score_item,item_weight)/np.abs(item_weight).sum()
    return y_pred




############################### Loss Function #################################

def top_k_similar_users(i,j,k,r):
    m, _ = r.shape
    rated_j = np.nonzero(~np.isnan(R_copy[:, j]))[0]
    sim_matrix = sim(r)
    similar_ind = rated_j[sim_matrix[i,rated_j].argsort()[::-1][:k]]
    return similar_ind.astype(int)


def loss(r, k, w):
    loss = 0
    mu = np.nanmean(R, axis = 1)
    for i in range(r.shape[0]):
        
        for j in range(r.shape[1]):
            v = top_k_similar_users(i,j,k,R)
            if np.isnan(r[i,j]):
                continue
            loss += np.square(r[i,j] - (mu[i] + np.dot(w[v,j],(r[v,j] - mu[v]))))
            
    return loss      
            

def gradient(r,k,w):
    m,_ = r.shape
    grad = np.zeros((m,k))
    mu = np.nanmean(r,axis = 1)
    for i in range(r.shape[0]): 
        for j in range(r.shape[1]):
            v = top_k_similar_users(i,j,k,r)
            if np.isnan(r[i,j]):
                continue
                grad[i, j] = -2*(r[i,j] - (mu[i] + np.dot(w[v,j],(r[v,j] - mu[v]).T))) * np.sum(r[v,j] - mu[v])
            
    return grad

def gradient_descent(r,k,w,learning_rate,iterations):
    m,_ = r.shape
    for i in range(iterations):
        grad = gradient(r, k, w)
        w = w - learning_rate * grad
        return w

            
####################### Indexes of *not Nan* Values #############################
irow, jcol = np.where(~np.isnan(R))

idx = np.random.choice(np.arange(100), 100, replace=False)
test_irow = irow[idx]
test_jcol = jcol[idx]

rated_j = np.where(~np.isnan(R[:,0]))
ind = np.argpartition(s[rated_j,1], -5-1)[-5-1:]
s_j = s[rated_j,1].T

############################### Prediction ####################################
err = []
w1 = np.ones(shape = (R.shape[0], R.shape[1]))
w2 = np.ones(shape = (R.shape[0], R.shape[1]))


for u, j in zip(test_irow, test_jcol):
    w_u = gradient_descent(R,5,w1, 0.001 ,1000)
    w_j = gradient_descent(R,5,w2, 0.001 ,1000)
    y_pred = r_u_j(R, s, w_u, w_j ,u ,j, n=5)
    y = R[u, j]

    err.append((y_pred - y) ** 2)


Total_error = err.sum()
MSE = Total_error/len(err)




















