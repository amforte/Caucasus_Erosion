#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares the power law fit, assuming a simple stream power model, to
the "fit" assuming a stochastic thershold incision model. 

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

import stochastic_threshold as stim

def find_ksn(eO,eP,ksnP):
    ksnO=np.zeros((len(eO)))
    for i in range(len(eO)):
        ix=np.argmin(np.abs(eO[i]-eP))
        ksnO[i]=ksnP[ix]
    return ksnO
        
        
def rmse(y,yp):
    mse=(np.sum((yp-y)**2))/len(y)
    return np.sqrt(mse)
    

# Load in Data from Erosion Rate basins
edf=pd.read_csv('data_tables/gc_ero_master_table.csv')
ksn=edf['mean_ksn'].to_numpy()
ksnq=edf['mean_ksn_q'].to_numpy()
e=edf['St_E_rate_m_Myr'].to_numpy()
eu=edf['St_Ext_Unc'].to_numpy()
ksnu=edf['se_ksn'].to_numpy()

# Load in results from optimization
edf2=pd.read_csv('result_tables/optimized_ero_k_e_tau_c.csv')
ecluster_label=edf2['cluster'].to_numpy().astype(int)
num_clustb=np.max(ecluster_label)+1
k_e_optim=edf2['k_e_median'].to_numpy()
tau_c_optim=edf2['tau_c_median'].to_numpy()

# Load in SPIM fits
sdf=pd.read_csv('result_tables/w_hi__bootstrap.csv')
s_K=sdf['K'].to_numpy()
s_K25=sdf['K_q25'].to_numpy()
s_K75=sdf['K_q75'].to_numpy()
s_n=sdf['n'].to_numpy()
s_n25=sdf['n_q25'].to_numpy()
s_n75=sdf['n_q75'].to_numpy()

# Fix values
k_e_fix=np.median(k_e_optim)
k_e_lo=np.percentile(k_e_optim,25)
k_e_hi=np.percentile(k_e_optim,75)
tau_c_fix=np.median(tau_c_optim)
k_e=np.ones((4))*k_e_fix
t_c=np.ones((4))*tau_c_fix

# Load in results of clustering of GRDC basins
clustdf=pd.read_csv('result_tables/grdc_basin_clusters.csv')
# ID=clustdf['grdc_id'].to_numpy().astype(int)
cluster_label=clustdf['cluster'].to_numpy().astype(int)

# Load in cluster population info
clustmdf=pd.read_csv('result_tables/grdc_mean_clusters.csv')
mR_pop=clustmdf['r_mean'].to_numpy()
cmb=clustmdf['c_aggr'].to_numpy()
smb=clustmdf['s_aggr'].to_numpy()

### Start Plotting   
color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']

f1=plt.figure(num=1,figsize=(25,20))
for i in range(num_clustb):
    idx=cluster_label==i
    
    eOI=e[ecluster_label==i]
    ksnOI=ksn[ecluster_label==i]
    
    plt.subplot(3,4,i+1)
    
    plt.title('Cluster '+str(i+1)) 
    
    plt.errorbar(e[ecluster_label==i],ksn[ecluster_label==i],ksnu[ecluster_label==i],eu[ecluster_label==i],ecolor=color_list[i],linestyle='',elinewidth=0.5)
    plt.scatter(e[ecluster_label==i],ksn[ecluster_label==i],s=10,marker='s',c=color_list[i],alpha=0.5,zorder=0)
            
    wclb=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    [KSb,Eb,_]=stim.stim_range(cmb[i],wclb,sc=smb[i],max_ksn=550,num_points=1000)    
    plt.plot(Eb,KSb,c=color_list[i],zorder=2,linewidth=2,linestyle='-',label='Fit to Composite')
     
    wclb=stim.set_constants(mR_pop[i],k_e_lo,dist_type='weibull',tau_c=t_c[i])
    [KSb1,Eb1,_]=stim.stim_range(cmb[i],wclb,sc=smb[i],max_ksn=550)    
    # plt.plot(Eb1,KSb1,c=color_list[i],zorder=2,linewidth=2,linestyle=':',label='25% $k_{e}$')

    wclb=stim.set_constants(mR_pop[i],k_e_hi,dist_type='weibull',tau_c=t_c[i])
    [KSb2,Eb2,_]=stim.stim_range(cmb[i],wclb,sc=smb[i],max_ksn=550)    
    # plt.plot(Eb2,KSb2,c=color_list[i],zorder=2,linewidth=2,linestyle=':',label='75% $k_{e}$')
    
    plt.fill_betweenx(KSb1,Eb1.ravel(),Eb2.ravel(),color=color_list[i],alpha=0.25)
    
    plt.plot(s_K*KSb**s_n,KSb,c='k',zorder=1,linewidth=2,linestyle='-',label='SPIM')
    # plt.plot(s_K25*KSb**s_n75,KSb,c='k',zorder=1,linewidth=2,linestyle=':',label='SPIM Bounds')
    # plt.plot(s_K75*KSb**s_n25,KSb,c='k',zorder=1,linewidth=2,linestyle=':')
    
    plt.fill_betweenx(KSb,s_K25*KSb**s_n75,s_K75*KSb**s_n25,color='k',alpha=0.25)
    
    plt.legend(loc='best')  

    plt.xlabel('Erosion Rate [m/Myr]')
    plt.ylabel('$k_{sn}$ [m]')
    plt.xlim((10,10000))
    plt.xscale('log')
    plt.ylim((0,550))
    
    plt.subplot(3,4,i+5)
    
    # Estimate y position
    SPIM_ksn=find_ksn(eOI,s_K*KSb**s_n,KSb)
    STIM_ksn=find_ksn(eOI,Eb,KSb)
    
    SPIM_RMSE=rmse(ksnOI,SPIM_ksn)
    STIM_RMSE=rmse(ksnOI,STIM_ksn)

    plt.stem(eOI,SPIM_ksn-ksnOI,linefmt='k',markerfmt='ko',
             basefmt=' ',label='SPIM - RMSE = '+str(np.round(SPIM_RMSE,0).astype('int')))
    (ma,st,ba)=plt.stem(eOI,STIM_ksn-ksnOI,linefmt=color_list[i],
                        label='STIM - RMSE = '+str(np.round(STIM_RMSE,0).astype('int')))
    plt.setp(ma,markerfacecolor=color_list[i],markeredgecolor=color_list[i])
    plt.setp(ba,color='gray',linestyle=':')

    plt.legend(loc='best')
    plt.xlabel('Erosion Rate [m/Myr]')
    plt.ylabel('$\Delta k_{sn}$ [m]')
    plt.xlim((10,10000))
    plt.xscale('log')
    
    plt.subplot(3,4,i+9)
    
    SPIM_E=s_K*ksnOI**s_n
    STIM_E=np.zeros((len(eOI)))
    
    SPIM_RMSE=rmse(eOI,SPIM_E)
  
    wclb=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    for j in range(len(eOI)):
        [STIM_E[j],_]=stim.stim_one(ksnOI[j],cmb[i],wclb,sc=smb[i])
    
    STIM_RMSE=rmse(eOI,STIM_E) 
    
    plt.stem(ksnOI,SPIM_E-eOI,linefmt='k',orientation='horizontal',markerfmt='ko',
             basefmt=' ',label='SPIM - RMSE = '+str(np.round(SPIM_RMSE,0).astype('int')))
    (ma,st,ba)=plt.stem(ksnOI,STIM_E-eOI,linefmt=color_list[i],orientation='horizontal',
                        label='STIM - RMSE = '+str(np.round(STIM_RMSE,0).astype('int')))
    plt.setp(ma,markerfacecolor=color_list[i],markeredgecolor=color_list[i])
    plt.setp(ba,color='gray',linestyle=':')    
    plt.legend(loc='best')
    
    plt.xlabel('$\Delta Erosion Rate [m/Myr]$')
    plt.ylabel('$k_{sn}$ [m]')
    plt.ylim((0,550))
    
    
    
    
    
    
    