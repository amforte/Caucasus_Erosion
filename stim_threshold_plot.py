#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates plots of thresholds for incision for the different clusters. 

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

import stochastic_threshold as stim

# Load in Data from Erosion Rate basins
edf=pd.read_csv('data_tables/gc_ero_master_table.csv')
ksn=edf['mean_ksn'].to_numpy()
e=edf['St_E_rate_m_Myr'].to_numpy()
eu=edf['St_Ext_Unc'].to_numpy()
ksnu=edf['se_ksn'].to_numpy()
ksnus=edf['std_ksn'].to_numpy()

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

f1=plt.figure(num=1,figsize=(20,25))

s1=np.arange(1,8,2)
s2=np.arange(2,9,2)

for i in range(num_clustb):
    idx=cluster_label==i
        
    plt.subplot(4,2,s1[i])
    
    plt.title('Cluster '+str(i+1)) 
    
    plt.errorbar(e[ecluster_label==i],ksn[ecluster_label==i],ksnu[ecluster_label==i],eu[ecluster_label==i],ecolor=color_list[i],linestyle='',elinewidth=0.5)
    plt.scatter(e[ecluster_label==i],ksn[ecluster_label==i],s=10,marker='s',c=color_list[i],alpha=0.5,zorder=0)
            
    wclb=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    [KSb,Eb,Qc]=stim.stim_range(cmb[i],wclb,sc=smb[i],max_ksn=550) 
    plt.plot(Eb,KSb,c=color_list[i],zorder=2,linewidth=2,linestyle='-',label='STIM')
    
    wclb=stim.set_constants(mR_pop[i],k_e_lo,dist_type='weibull',tau_c=t_c[i])
    [KSb1,Eb1,Qc1]=stim.stim_range(cmb[i],wclb,sc=smb[i],max_ksn=550)    
    wclb=stim.set_constants(mR_pop[i],k_e_hi,dist_type='weibull',tau_c=t_c[i])
    [KSb2,Eb2,Qc2]=stim.stim_range(cmb[i],wclb,sc=smb[i],max_ksn=550)    
    plt.fill_betweenx(KSb1,Eb1.ravel(),Eb2.ravel(),color=color_list[i],alpha=0.25)
    
    plt.xlabel('Erosion Rate [m/Myr]')
    plt.ylabel('$k_{sn}$ [m]')
    plt.xlim((10,10000))
    plt.xscale('log')
    plt.ylim((0,550))
    
    Q_star=np.logspace(-1,5,1000)
    R_vec=Q_star*mR_pop[i]
    cdf=weibull_min.sf(Q_star,cmb[i],loc=0,scale=smb[i])
    
    ax1=plt.subplot(4,2,s2[i])
    
    plt1=ax1.plot(Qc*mR_pop[i],KSb,linewidth=2,c=color_list[i],label='Critical Runoff')
    plt2=ax1.plot([mR_pop[i],mR_pop[i]],[0,550],linewidth=2,c=color_list[i],linestyle='--',label='Mean Runoff')
    ax1.set_xlabel('Runoff [mm/day]')
    ax1.set_xlim((0.5,1000))
    ax1.set_xscale('log')
    ax1.set_ylabel('$k_{sn}$ [m]')
    ax1.set_ylim((0,550))
    
    ax2=ax1.twinx()
    plt3=ax2.plot(R_vec,cdf,c=color_list[i],linewidth=2,linestyle=':',label='Probability of Exceedance')
    ax2.set_yscale('log')
    ax2.set_ylim((1e-7,1))
    ax2.set_ylabel('Exceedance Frequency')
    
    ax2.axhline(1/7,c='k',linestyle=':',zorder=0)
    ax2.axhline(1/30,c='k',linestyle=':',zorder=0)
    ax2.axhline(1/(1*365.25),c='k',linestyle=':',zorder=0)
    ax2.axhline(1/(10*365.25),c='k',linestyle=':',zorder=0)
    ax2.axhline(1/(100*365.25),c='k',linestyle=':',zorder=0)
    ax2.axhline(1/(1000*365.25),c='k',linestyle=':',zorder=0)
    ax2.axhline(1/(10000*365.25),c='k',linestyle=':',zorder=0)

    ax2.text(200,1/7,'1 week',fontsize='large',va='bottom')
    ax2.text(200,1/30,'30 days',fontsize='large',va='bottom')
    ax2.text(200,1/(1*365.25),'1 Year',fontsize='large',va='bottom')
    ax2.text(200,1/(10*365.25),'10 Year',fontsize='large',va='bottom')
    ax2.text(200,1/(100*365.25),'100 Year',fontsize='large',va='bottom')
    ax2.text(200,1/(1000*365.25),'1000 Year',fontsize='large',va='bottom')
    ax2.text(200,1/(10000*365.25),'10000 Year',fontsize='large',va='bottom')
    
    plts=plt1+plt2+plt3
    lbls=[l.get_label() for l in plts]
    
    plt.legend(plts,lbls,loc='best')
    
f1.savefig('threshold.pdf')
    
    
    