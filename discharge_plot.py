#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots exceedance frequency and the relationships between discharge and drainage 
area for gauged basins

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


def survive(Q):
    Qstar=Q/np.mean(Q)
    Qstar_sort=np.sort(Qstar)
    Qn=len(Qstar)
    Qrank=np.arange(1,Qn+1,1)
    Q_freq_excd=(Qn+1-Qrank)/Qn
    return Qstar_sort,Q_freq_excd

# Load data from gauged basins
qdf=pd.read_csv('data_tables/grdc_summary_values.csv')
mR=qdf['mean_runoff_mm_day'].to_numpy()
da=qdf['DA_km2'].to_numpy()
mRain=qdf['mnTRMM_mm_day'].to_numpy()
ID=qdf['ID'].to_numpy()
N=len(ID)

f1=plt.figure(num=1,figsize=(14,5))

ax1=plt.subplot(1,2,1)
ax1.set_ylim((10**-4,1))
ax1.set_xlim((0.01,500))
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('Runoff [mm/day]')
ax1.set_ylabel('Exceedance Frequency')

rain_norm=colors.Normalize(vmin=1,vmax=6)
mQ=np.zeros(ID.shape)
for i in range(N):
    df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[i])+'.csv')
    Q=df['Q'].to_numpy()
    mQ[i]=np.mean(Q)/(60*60*24)
    [Qstar_sort,Q_freq_excd]=survive(Q)
    Rainv=np.ones(Q.shape)*mRain[i]
    ax1.scatter(Qstar_sort*mR[i],Q_freq_excd,c=Rainv,norm=rain_norm,cmap=cm.batlow_r,s=2)
    
ax2=plt.subplot(1,2,2)
ax2.set_ylim((10**-1,10**3))
ax2.set_xlim((50,5000))
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Catchment Area [km]')
ax2.set_ylabel('Mean Q [m3/s]')


runs=np.arange(1,7,1)
inter=np.linspace(0,1,len(runs))
colors=[cm.batlow_r(x) for x in inter]

sc1=ax2.scatter(da,mQ,c=mRain,marker='o',s=40,norm=rain_norm,cmap=cm.batlow_r)
xx=np.linspace(50,5000,100)
con=(1000**2)/(1000*24*60*60)
for i, color in enumerate(colors):
    ax2.plot(xx,xx*runs[i]*con,c=color,zorder=0,linestyle=':')
cbar1=plt.colorbar(sc1,ax=ax2)
cbar1.ax.set_ylabel('Mean Rainfall [mm/day]')

f1.savefig('dischage.pdf')
f1.savefig('dischage.tif',dpi=300)