#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for meaningful changes in monthly mean runoffs through time within the
available time series for gauged basins.

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
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


plt.figure(2,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax1.set_title('Winter')
ax1.set_xlim((-0.7,0.7))
ax1.axvline(0,c='k',linestyle=':')
ax1.set_xlabel('Slope of Trend in Monthly Means')
ax1.set_ylabel('Mean Elevation [m]')

ax2=plt.subplot(2,2,2)
ax2.set_title('Spring')
ax2.set_xlim((-0.7,0.7))
ax2.axvline(0,c='k',linestyle=':')
ax2.set_xlabel('Slope of Trend in Monthly Means')
ax2.set_ylabel('Mean Elevation [m]')

ax3=plt.subplot(2,2,3)
ax3.set_title('Summer')
ax3.set_xlim((-0.7,0.7))
ax3.axvline(0,c='k',linestyle=':')
ax3.set_xlabel('Slope of Trend in Monthly Means')
ax3.set_ylabel('Mean Elevation [m]')

ax4=plt.subplot(2,2,4)
ax4.set_title('Fall')
ax4.set_xlim((-0.7,0.7))
ax4.axvline(0,c='k',linestyle=':')
ax4.set_xlabel('Slope of Trend in Monthly Means')
ax4.set_ylabel('Mean Elevation [m]')

for i in range(12):
    if i==0 or i==1 or i==11:
        ax1.scatter(mnth_trend[:,i],mnZ)
    elif i==2 or i==3 or i==4:
        ax2.scatter(mnth_trend[:,i],mnZ)
    elif i==5 or i==6 or i==7:
        ax3.scatter(mnth_trend[:,i],mnZ)
    elif i==8 or i==9 or i==10:
        ax4.scatter(mnth_trend[:,i],mnZ)
