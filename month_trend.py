#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:37:19 2021

@author: amforte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from cmcrameri import cm

# Build File List    
gdf=pd.read_csv('data_tables/grdc_summary_values.csv') 
maxZ=gdf['maxz'].to_numpy()/1000
minZ=gdf['minz'].to_numpy()/1000
mnZ=gdf['mnz'].to_numpy()/1000
ID=gdf['ID'].to_numpy()
N=len(ID)

mnth_trend=np.zeros((N,12))

for i in range(N):
    # Read Files
    df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[i])+'.csv')
    R=df['R'].to_numpy()
    yr=df['yr'].to_numpy()
    mnth=df['mnth'].to_numpy()
    # Find year list
    yru=np.unique(yr)
    num_year=len(yru)
    mnrs=np.zeros((num_year,12))
    for j in range(num_year):
        idx=yr==yru[j]
        mnoi=mnth[idx]
        roi=R[idx]
        for k in range(12):
            mnrs[j,k]=np.mean(roi[mnoi==k+1])
    for j in range(12):
        x=yru-np.min(yru)
        y=mnrs[:,j]
        nanix=~np.isnan(y)
        x=x[nanix]
        y=y[nanix]
        lin=np.polyfit(x,y,1)
        mnth_trend[i,j]=lin[0]
        

plt.figure(1,figsize=(10,15))

ax1=plt.subplot(2,1,1)
ax1.axhline(0,c='k',linestyle=':')
ax1.violinplot(mnth_trend,showmeans=True)
ax1.set_xlabel('Month')
ax1.set_ylabel('Slope of Trend in Monthly Means')

norm=colors.Normalize(vmin=0.2,vmax=2.8)
ax2=plt.subplot(2,1,2)
for i in range(12):
    y=mnth_trend[:,i]
    x=np.ones(y.shape)*(i+1)
    sc1=ax2.scatter(x,y,c=mnZ,s=20,norm=norm,cmap=cm.vik)
ax2.set_xlabel('Month')
ax2.set_ylabel('Slope of Trend in Monthly Means')    
cbar1=plt.colorbar(sc1,ax=ax2)
cbar1.ax.set_ylabel('Mean Elevation [km]')