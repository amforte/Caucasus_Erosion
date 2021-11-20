#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of width, discharge, and drainage area scaling relationships in
ungagued, 10Be basins in the Greater Caucasus. Main goal is is to assess an 
appropriate value the k_w parameter.

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import stochastic_threshold as stim

df=pd.read_csv('data_tables/gc_ero_master_table.csv')
wd=pd.read_csv('data_tables/width_key.csv')
qdf=pd.read_csv('data_tables/gauged_discharges.csv')
erun=pd.read_csv('result_tables/estimate_runoff_power.csv')

rml=wd['rm'].to_numpy()
smpl=wd['smpl'].to_numpy()
all_rm=df['river_mouth'].to_numpy()
E=df['St_E_rate_m_Myr'].to_numpy()
mR=erun['mean_runoff'].to_numpy()

fig1=plt.figure(num=1,figsize=(20,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,3)
ax3=plt.subplot(2,2,2)
ax4=plt.subplot(2,2,4)
cnorm=colors.Normalize(vmin=0,vmax=2500)
rnorm=colors.Normalize(vmin=0.5,vmax=2)

# Rebuttal Figure
fig2=plt.figure(num=2,figsize=(20,15))
f2ax1=plt.subplot(2,3,1)
f2ax2=plt.subplot(2,3,2)
f2ax3=plt.subplot(2,3,3)
f2ax4=plt.subplot(2,1,2)

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
        
        f2ax1.scatter(das,ws,s=5,c=ROI2,norm=rnorm,cmap='Spectral',zorder=1)
        f2ax2.scatter(das,ws,s=5,c=ROI2,norm=rnorm,cmap='Spectral',zorder=1)
        f2ax3.scatter(das,ws,s=5,c=ROI2,norm=rnorm,cmap='Spectral',zorder=1)
        
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
cbar3.ax.set_ylabel('Log Mean Daily Runoff [mm/day]')
cbar4.ax.set_ylabel('Log Mean Daily Runoff [mm/day]')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Drainage Area [$km^2$]')
ax3.set_ylabel('Width [m]')   

ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Binned Drainage Area [$km^2$]')
ax4.set_ylabel('Smoothed (1 km) Width [m]')   


## REBUTAL FIGURE
kw=15
b=0.5
da_vec=np.logspace(0,2,10)
da_vec_m=da_vec*1e6
for i in range(len(da_vec)):
    Wb=kw*(Rb**b)*(da_vec_m[i]**b)
    Wbf=kw*(Rbf**b)*(da_vec_m[i]**b)
    da_vec_OI=np.ones(Wb.shape)*da_vec[i]  
    f2ax2.scatter(da_vec_OI,Wbf,c='w',s=40,marker='^',edgecolor='k',zorder=4)
    f2ax2.scatter(da_vec_OI,Wb,c='k',s=40,marker='s',zorder=3)
    
kw=10
b=0.5
da_vec=np.logspace(0,2,10)
da_vec_m=da_vec*1e6
for i in range(len(da_vec)):
    Wb=kw*(Rb**b)*(da_vec_m[i]**b)
    Wbf=kw*(Rbf**b)*(da_vec_m[i]**b)
    da_vec_OI=np.ones(Wb.shape)*da_vec[i]  
    f2ax1.scatter(da_vec_OI,Wbf,c='w',s=40,marker='^',edgecolor='k',zorder=4)
    f2ax1.scatter(da_vec_OI,Wb,c='k',s=40,marker='s',zorder=3)
    
kw=20
b=0.5
da_vec=np.logspace(0,2,10)
da_vec_m=da_vec*1e6
for i in range(len(da_vec)):
    Wb=kw*(Rb**b)*(da_vec_m[i]**b)
    Wbf=kw*(Rbf**b)*(da_vec_m[i]**b)
    da_vec_OI=np.ones(Wb.shape)*da_vec[i]  
    f2ax3.scatter(da_vec_OI,Wbf,c='w',s=40,marker='^',edgecolor='k',zorder=4)
    f2ax3.scatter(da_vec_OI,Wb,c='k',s=40,marker='s',zorder=3)    

f2ax1.set_xscale('log')
f2ax1.set_yscale('log')
f2ax1.set_xlabel('Drainage Area [$km^2$]')
f2ax1.set_ylabel('Width [m]')
f2ax1.set_title('$k_w$=10')
f2ax2.set_xscale('log')
f2ax2.set_yscale('log')
f2ax2.set_xlabel('Drainage Area [$km^2$]')
f2ax2.set_ylabel('Width [m]')
f2ax2.set_title('$k_w$=15')
f2ax3.set_xscale('log')
f2ax3.set_yscale('log')
f2ax3.set_xlabel('Drainage Area [$km^2$]')
f2ax3.set_ylabel('Width [m]')
f2ax3.set_title('$k_w$=20')


clustmdf=pd.read_csv('result_tables/grdc_mean_clusters.csv')
cmb=clustmdf['c_aggr'].to_numpy()
smb=clustmdf['s_aggr'].to_numpy()
rm=clustmdf['r_mean'].to_numpy()
k_e_o=clustmdf['k_e'].to_numpy()

cl=stim.set_constants(rm[1],1.5e-11,dist_type='weibull',k_w=15)
[KS1,E1,_,_]=stim.stim_range(cmb[1],cl,sc=smb[1],max_ksn=550)

cl=stim.set_constants(rm[1],5e-11,dist_type='weibull',k_w=20)
[KS2,E2,_,_]=stim.stim_range(cmb[1],cl,sc=smb[1],max_ksn=550)

cl=stim.set_constants(rm[1],3.5e-12,dist_type='weibull',k_w=10)
[KS3,E3,_,_]=stim.stim_range(cmb[1],cl,sc=smb[1],max_ksn=550)

f2ax4.plot(E1,KS1,c='k',linestyle='-',label='$k_w$=15; $k_e$=1.5e-11')
f2ax4.plot(E2,KS2,c='k',linestyle=':',label='$k_w$=20; $k_e$=1.5e-11')
f2ax4.plot(E3,KS3,c='k',linestyle='--',label='$k_w$=10; $k_e$=1.5e-11')
f2ax4.legend(loc='best')
f2ax4.set_xlabel('Erosion Rate [m/Myr]')
f2ax4.set_ylabel('$k_{sn}$ [m]')


fig1.savefig('widths.pdf')