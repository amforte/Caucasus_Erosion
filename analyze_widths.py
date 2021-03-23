#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


df=pd.read_csv('data_tables/gc_ero_master_table.csv')
wd=pd.read_csv('data_tables/width_key.csv')
qdf=pd.read_csv('data_tables/gauged_discharges.csv')

rml=wd['rm'].to_numpy()
smpl=wd['smpl'].to_numpy()
all_rm=df['river_mouth'].to_numpy()
E=df['St_E_rate_m_Myr'].to_numpy()
mR=df['mean_runoff'].to_numpy()

fig1=plt.figure(num=1,figsize=(20,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,3)
ax3=plt.subplot(2,2,2)
ax4=plt.subplot(2,2,4)
cnorm=colors.Normalize(vmin=0,vmax=2500)
rnorm=colors.Normalize(vmin=0,vmax=4)

da_l=[]
w_l=[]
das_l=[]
ws_l=[]

for i in range(len(all_rm)):
    rmOI=all_rm[i]
    idx=rml==rmOI
    if any(idx):
        fp=smpl[idx]
        raw_file='data_tables/width_tables/rwidth_'+fp[0]+'_raw.csv'
        mvmn_file='data_tables/width_tables/rwidth_'+fp[0]+'_movmean.csv'
        raw=pd.read_csv(raw_file)
        mvmn=pd.read_csv(mvmn_file)
        # Extract smooth
        ws=mvmn['smooth_w'].to_numpy()
        das=mvmn['drn_ar'].to_numpy()
        das=das/1e6
        # Extract raw
        da=raw['drain_area'].to_numpy()
        da=da/1e6
        w=raw['river_w'].to_numpy()
        EOI=np.ones(da.shape)*E[i]
        EOI2=np.ones(ws.shape)*E[i]
        ROI=np.ones(da.shape)*mR[i]
        ROI2=np.ones(ws.shape)*mR[i]
        sc1=ax1.scatter(da,w,s=5,c=EOI,norm=cnorm,cmap='plasma',zorder=1)
        sc2=ax2.scatter(das,ws,s=5,c=EOI2,norm=cnorm,cmap='plasma',zorder=1)
        sc3=ax3.scatter(da,w,s=5,c=ROI,norm=rnorm,cmap='Spectral',zorder=1)
        sc4=ax4.scatter(das,ws,s=5,c=ROI2,norm=rnorm,cmap='Spectral',zorder=1)
        da_l.append(da)
        w_l.append(w)
        das_l.append(das)
        ws_l.append(ws)
        

Qb=qdf['Qb'].to_numpy()
Rb=qdf['Rb'].to_numpy()
Qbf=qdf['Qbf'].to_numpy()
Rbf=qdf['Rbf'].to_numpy()

kw=15
b=0.5

da_vec=np.logspace(0,2,10)
da_vec_m=da_vec*1e6
for i in range(len(da_vec)):
    Wb=kw*(Rb**b)*(da_vec_m[i]**b)
    Wbf=kw*(Rbf**b)*(da_vec_m[i]**b)
    da_vec_OI=np.ones(Wb.shape)*da_vec[i]  
    ax3.scatter(da_vec_OI,Wbf,c='w',s=40,marker='^',edgecolor='k',zorder=4)
    ax3.scatter(da_vec_OI,Wb,c='k',s=40,marker='s',zorder=3)
    ax4.scatter(da_vec_OI,Wbf,c='w',s=40,marker='^',edgecolor='k',zorder=4)
    ax4.scatter(da_vec_OI,Wb,c='k',s=40,marker='s',zorder=3)

        
cbar1=plt.colorbar(sc1,ax=ax1)
cbar2=plt.colorbar(sc2,ax=ax2)
cbar1.ax.set_ylabel('Erosion Rate [m/Myr]')
cbar2.ax.set_ylabel('Erosion Rate [m/Myr]')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Drainage Area [$km^2$]')
ax1.set_ylabel('Width [m]')   

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Binned Drainage Area [$km^2$]')
ax2.set_ylabel('Smoothed (1 km) Width [m]') 
         
cbar3=plt.colorbar(sc3,ax=ax3)
cbar4=plt.colorbar(sc4,ax=ax4)
cbar3.ax.set_ylabel('Mean Daily Runoff [mm/day]')
cbar4.ax.set_ylabel('Mean Daily Runoff [mm/day]')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Drainage Area [$km^2$]')
ax3.set_ylabel('Width [m]')   

ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Binned Drainage Area [$km^2$]')
ax4.set_ylabel('Smoothed (1 km) Width [m]')   








