# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:04:38 2023

@author: julia
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma
from scipy.linalg import sqrtm
from tqdm import tqdm

from IS_misc_functions import *
import time



def create_sample_MtM(ELGD_bounds,EAD_bounds,rho_bounds,c_bounds,g_bounds,D_bounds,obligor_bounds,B):
    
    ELGD = np.random.uniform(ELGD_bounds[0],ELGD_bounds[1],(B,obligor_bounds[1]))
    EAD = np.random.uniform(EAD_bounds[0],EAD_bounds[1],(B,obligor_bounds[1]))
    EAD = EAD / np.repeat(np.expand_dims(np.sum(EAD,1),1),obligor_bounds[1],axis = 1)
    
    rho = np.random.uniform(rho_bounds[0],rho_bounds[1],(B,obligor_bounds[1]))
    c = np.random.uniform(c_bounds[0],c_bounds[1],(B,obligor_bounds[1]))
    g = np.random.randint(g_bounds[0],g_bounds[1],(B,obligor_bounds[1]))
    D = np.random.uniform(D_bounds[0],D_bounds[1],(B,obligor_bounds[1]))
    # Create vector with number of obligors
    N_obligors = np.random.randint(obligor_bounds[0], obligor_bounds[1],size = B)
    N = np.zeros((B,obligor_bounds[1]))
    for i in range(B):
        N[i,:] = np.pad(np.repeat(1,N_obligors[i]),(0,obligor_bounds[1]-N_obligors[i])) # Number of Obligors    
    
    return ELGD*N,EAD*N,rho*N,c*N,g*N,D*N

def MC_IS_MtM(ELGD,EAD,rho,c,g,trans_dict,D,r,q=0.999,T=1,n = 10000,n1 = 50,LGD_constant = False,
                  nu = 0.25,
              default_only=False,ImportanceSampling = False, return_cond_loss = False): 
    
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta   
    
    A = EAD/np.sum(EAD) # Normalize the EADs
    
    # Reduce the vectors considered to a smaller dimension
    greater_zero = A>0
    A = A[greater_zero]
    ELGD = ELGD[greater_zero]
    rho = rho[greater_zero]
    c = c[greater_zero]
    g = g[greater_zero].astype(int)
    D = D[greater_zero]
    
    L = []
    p_list = []
    # Compute the Bounds for state transitions
    bounds = rating_bounds(ELGD,A,rho,c,g,D,trans_dict,default_only=default_only) #length = S+2
    # Compute Lambda 
    if LGD_constant:
        lambda_0 = np.matrix(lambda_vector(ELGD,A,rho,c,g,D,np.arange(S+1),trans_dict,T=T,default_only=default_only)) #S+1 times nr obligor
        if default_only:
            lambda_0[0,:] =1-ELGD
        # Sample n times the systematic Risk Factor X
        quantile = norm.ppf( 1-q)
        if ImportanceSampling:
            y = -np.sum(np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0)*pi_func(quantile,rho,bounds))
            mu = to_optimize_mu(y,rho,lambda_0,ELGD,A,c,g,bounds)
            X = np.random.normal(mu,1,  size = n)  
            
            for i in range(n):
                L_inner = []
                p_inner = []
                # Determine t
                t_X = t(ELGD,A,rho,c,g,D,X[i],y,bounds,lambda_0)
                for j in range(n1):
                    loss_var = - np.sum(np.multiply(np.sum(np.multiply(lambda_0,Z_S(ELGD,A,rho,c,g,X[i],bounds,t_X,lambda_0) ),0),A)) 
                    L_inner.append(loss_var) #(loss_var>y)*
                    p_inner.append(np.exp(-t_X*loss_var))
                L.append(np.mean(L_inner))
                p_list.append(M_L(ELGD,A,rho,c,g,X[i],t_X,lambda_0,bounds)*r_mu(X[i],mu)*np.mean(p_inner))
        
                
            # sort (L,p) simultaneously by L
            sorted_data = sorted(zip(L, p_list))
            sorted_L, sorted_p = zip(*sorted_data)
            quantile_index = np.searchsorted(np.cumsum(sorted_p), q*np.sum(p_list), side='right')   
            MC = sorted_L[quantile_index-1]   
        elif ImportanceSampling == False:
            X = np.random.normal(0,1,  size = n)  
            L = []
            for i in range(n):
                Y = np.sqrt(rho)*X[i]+np.sqrt(1-rho)*np.random.normal(size = len(A))
                Loss_n = - np.sum(np.multiply(np.sum(np.multiply(lambda_0,Z(ELGD,A,rho,c,g,Y,bounds) ),0),A)) 
                L.append(Loss_n)
            MC = np.quantile(L,q)     
        Cond_loss = -np.sum(np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0)*pi_func(norm.ppf(1-q),rho,bounds))
        ER = np.sum([A[i]*np.sum([lambda_0[s,i]*trans_dict['trans_prob'].iloc[S-g[i],S-s] for s in range(S)]) for i in range(lambda_0.shape[1])])

    elif LGD_constant == False:

        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = len(ELGD)) for _ in range(n)]))
        Term1,Term2,Term3,tenor,lam_01,lam_02,P_00 = lambda_vector_terms_const_1(ELGD,A,rho,c,g,D,np.arange(S+1),trans_dict,T=T,default_only=default_only)        
        lambda_0 = [np.matrix(lambda_vector_terms_const_2(ELGD,A,rho,c,g,D,np.arange(S+1),trans_dict,
                                                          Term1,Term2,Term3*np.expand_dims(ELGD,1),
                                                          tenor,lam_01,lam_02,lam_02*LGD[i],P_00,T = T)) for  i in range(n)]
        lambda_0_ELGD = np.matrix(lambda_vector(ELGD,A,rho,c,g,D,np.arange(S+1),trans_dict,T=T,default_only=default_only))
        
        #print("Price:",np.round([lambda_0_ELGD[s,0]*P_0(ELGD,A,rho,c,g,D,tenor,trans_dict,r =r, default_only=False)[0] for s in range(S)],3))
        #print("Price:",np.round([lambda_0_ELGD[s,0] for s in range(S)],3))
        
        #print("Cond Prob:", np.round(pi_func(norm.ppf( 1-q),rho,bounds)[:,0],3))
        #
        # expected return
        #ER = np.sum([A[i]*lambda_0_ELGD[g[i],i] for i in range(lambda_0_ELGD.shape[1])])
        ER = np.sum([A[i]*np.sum([lambda_0_ELGD[s,i]*trans_dict['trans_prob'].iloc[S-g[i],S-s] for s in range(S)]) for i in range(lambda_0_ELGD.shape[1])])
        if default_only:
            for  i in range(n):
                lambda_0[i][0,:] =1-ELGD
        # Sample n times the systematic Risk Factor X
        quantile = norm.ppf( 1-q)
        X = np.random.normal(0,1,  size = n)  
        L = []
        for i in range(n):
            Y = np.sqrt(rho)*X[i]+np.sqrt(1-rho)*np.random.normal(size = len(A))
            Loss_n = - np.sum(np.multiply(np.sum(np.multiply(lambda_0[i],Z(ELGD,A,rho,c,g,Y,bounds) ),0),A)) 
            L.append(Loss_n)
        MC = np.quantile(L,q)     
        Cond_loss = -np.sum(np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0_ELGD)*pi_func(norm.ppf(1-q),rho,bounds))
    if return_cond_loss:
        return_val = [np.exp(-r(T))*(ER+MC),np.exp(-r(T))*(ER+Cond_loss)]    
    else:
        return_val =  np.exp(-r(T))*(MC-Cond_loss)    
    return return_val