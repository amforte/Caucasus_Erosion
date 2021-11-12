#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
seed=5

# Read in data
df=pd.read_csv('data_tables/gc_ero_master_table.csv')
# Extract main variables of interest
mR=df['mean_runoff'].to_numpy()
mP=df['corrected_mean_trmm'].to_numpy()
RR=df['runoff_ratio'].to_numpy()
mS=df['mean_SNOWstd'].to_numpy()
ksn=df['mean_ksn'].to_numpy()
ksn_u=df['se_ksn'].to_numpy()
e=df['St_E_rate_m_Myr'].to_numpy()
e_u=df['St_Ext_Unc'].to_numpy()
# Choice of k estimation technique
k_var='k_z_est'
k_var1='k_SSN_est'

k_values=df[[k_var,k_var1]]
k=k_values.mean(1).to_numpy()
k_std=k_values.std(1)

qdf=pd.read_csv('data_tables/grdc_seasonal_values.csv')
G_mR=qdf['mean_runoff_mm_day'].to_numpy()
G_k=qdf['k'].to_numpy()


### Cluster Ero Basins
X=np.concatenate((k.reshape(len(k),1),mR.reshape(len(mR),1)),axis=1)
# Scale data
scaler=StandardScaler().fit(X)
XS=scaler.transform(X)


#### Optimal Cluster Number Based on Elbow ####
num_clust=3
km=KMeans(n_clusters=num_clust,max_iter=5000,random_state=seed).fit(XS)

# PLOT
color_list=['maroon','dodgerblue','darkorange','crimson''darkolivegreen',]

fig1=plt.figure(num=1,figsize=(10,10))
for i in range(num_clust):  
    plt.scatter(X[km.labels_==i,1],X[km.labels_==i,0],c=color_list[i],zorder=1)
    if i==0:
        idx=np.logical_and(G_mR<4,G_k>4)
        plt.scatter(G_mR[idx],G_k[idx],c=color_list[i],zorder=1,marker='s',edgecolor='k')
    elif i==1:
        idx=np.logical_and(G_mR<4,G_k<4)
        plt.scatter(G_mR[idx],G_k[idx],c=color_list[i],zorder=1,marker='s',edgecolor='k') 
    else:
        idx=G_mR>=4
        plt.scatter(G_mR[idx],G_k[idx],c=color_list[i],zorder=1,marker='s',edgecolor='k')        
    
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Estimated Variability')
# plt.xlim((0,6))
# plt.ylim((2,6))

