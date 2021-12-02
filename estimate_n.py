#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approximates the cluster STIM relationships with power laws to estimate a
of n (i.e., the slope exponent in the stream power incision model)

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
"""
import pandas as pd
import numpy as np
from scipy import odr
import matplotlib.pyplot as plt

import stochastic_threshold as stim

def lin(B,x):
    return B[1]*x+B[0]

ll_model=odr.Model(lin)

# Load ke value
edf2=pd.read_csv('result_tables/optimized_ero_k_e_tau_c.csv')
k_e_optim=edf2['k_e_median'].to_numpy()
k_e=np.median(k_e_optim)

# Load cluster values
clustmdf=pd.read_csv('result_tables/grdc_mean_clusters.csv')
mR_pop=clustmdf['r_mean'].to_numpy()
cmb=clustmdf['c_aggr'].to_numpy()
smb=clustmdf['s_aggr'].to_numpy()

# Generate n output
n=np.zeros((4))

# Set colors
color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']

plt.figure(1,figsize=(10,10))
ax1=plt.subplot(2,1,1)
ax1.set_xlim((0,10000))
ax1.set_xlabel('Erosion Rate [m/Myr]')
ax1.set_ylabel('$k_{sn}$')


ax2=plt.subplot(2,1,2)
ax2.set_xlabel('Approximate n')
ax2.set_ylabel('c')

# Start loop
for i in range(4):
    wcl=stim.set_constants(mR_pop[i],k_e,dist_type='weibull')
    [KS,E,_]=stim.stim_range(cmb[i],wcl,sc=smb[i],max_ksn=550,space_type='lin')
    
    # Prep data
    E=np.ravel(E)
    ix=E>10 
    x=np.log10(E[ix])
    y=np.log10(KS[ix])

    # FIT
    ll_data=odr.RealData(x,y)
    llodr_obj=odr.ODR(ll_data,ll_model,beta0=[2,0.05])
    llres_ins=llodr_obj.run()
    lls_C=10**llres_ins.beta[0]
    lls_phi=llres_ins.beta[1]
    K=lls_C**(-1/lls_phi)
    n[i]=1/lls_phi
    
    
    ax2.scatter(n[i],cmb[i],c=color_list[i])
    
    ax1.scatter(K*KS**n[i],KS,c=color_list[i],s=10,alpha=0.5,label='Power Law Fit; Cluster '+str(i+1))
    ax1.plot(E,KS,c=color_list[i],zorder=2,linewidth=2,label='STIM; Cluster '+str(i+1),linestyle='-')
    ax1.legend()




