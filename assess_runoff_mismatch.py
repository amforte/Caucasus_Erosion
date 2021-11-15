#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 06:46:58 2021

@author: amforte
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import stochastic_threshold as stim

def min_R(X,k_e,tau_c,ks,e,c,s):
    cl=stim.set_constants(X,k_e,dist_type='weibull',tau_c=tau_c)
    [ep,_,_]=stim.stim_one(ks,c,cl,sc=s)
    return (ep-e)**2

edf=pd.read_csv('data_tables/gc_ero_master_table.csv')
erun=pd.read_csv('result_tables/estimate_runoff_power.csv')
edf2=pd.read_csv('result_tables/optimized_ero_k_e_tau_c.csv')
clustmdf=pd.read_csv('result_tables/grdc_mean_clusters.csv')

cmb=clustmdf['c_aggr'].to_numpy()
smb=clustmdf['s_aggr'].to_numpy()

ecluster_label=edf2['cluster'].to_numpy().astype(int)
num_clustb=np.max(ecluster_label)+1
k_e_optim=edf2['k_e_median'].to_numpy()
tau_c_optim=edf2['tau_c_median'].to_numpy()
k_e_fix=np.median(k_e_optim)
tau_c_fix=np.median(tau_c_optim)

emR=erun['mean_runoff'].to_numpy()
ksn=edf['mean_ksn'].to_numpy()
e=edf['St_E_rate_m_Myr'].to_numpy()
eu=edf['St_Ext_Unc'].to_numpy()
ksnu=edf['se_ksn'].to_numpy()

tuneR=np.zeros((len(ksn)))
for i in range(len(ksn)):
    # args=(k_e_fix,tau_c_fix,ksn[i],e[i],cmb[ecluster_label[i]],smb[ecluster_label[i]])
    args=(k_e_fix,tau_c_fix,ksn[i],e[i],cmb[1],smb[1])
    res=minimize_scalar(min_R,args=args,bounds=[0.1,10],
                        method='bounded',
                        options={'maxiter':500000,'xatol':1e-20})
    tuneR[i]=res.x

color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']    
plt.figure(num=1,figsize=(15,15))
plt.plot([0,8],[0,8],c='k',linestyle=':',zorder=0)
for i in range(num_clustb):
    plt.scatter(emR[ecluster_label==i],tuneR[ecluster_label==i],c=color_list[i],zorder=1)
plt.xlabel('Mean Estimated Runoff [mm/day]')
plt.ylabel('Tuned Runoff [mm/day]')
plt.ylim((0,8))
plt.xlim((0,8))
