# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

qdf=pd.read_csv('data_tables/grdc_seasonal_values.csv')
grdc_id=qdf['grdc_id'].to_numpy()
mR=qdf['mean_runoff_mm_day'].to_numpy()
k=qdf['k'].to_numpy()
cluster=qdf['cluster_num'].to_numpy()
ssn_frac=qdf['seasonal_frac'].to_numpy()
anu_frac=qdf['annual_frac'].to_numpy()
evnt_frac=qdf['event_frac'].to_numpy()
da=qdf['drain_area_km2'].to_numpy()
mz=qdf['maxz_m'].to_numpy()
do=qdf['dist_from_sw_km'].to_numpy()
d=np.copy(do)
d[np.isnan(d)]=150

djf_run=qdf['DJF_mean_runoff_mm_day'].to_numpy()
mam_run=qdf['MAM_mean_runoff_mm_day'].to_numpy()
jja_run=qdf['JJA_mean_runoff_mm_day'].to_numpy()
son_run=qdf['SON_mean_runoff_mm_day'].to_numpy()

djf_rain=qdf['DJF_mean_TRMMrainfall_mm_day'].to_numpy()
mam_rain=qdf['MAM_mean_TRMMrainfall_mm_day'].to_numpy()
jja_rain=qdf['JJA_mean_TRMMrainfall_mm_day'].to_numpy()
son_rain=qdf['SON_mean_TRMMrainfall_mm_day'].to_numpy()

# Colors for clusters
color_list=['maroon','dodgerblue','darkorange']

## Figure 1
f1=plt.figure(num=1,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

for i in range(3):
    idx=cluster==i+1
    ax1.scatter(k[idx],anu_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax2.scatter(k[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax3.scatter(k[idx],evnt_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax4.scatter(mR[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

ax1.set_ylabel('Annual Fraction')
ax2.set_ylabel('Seasonal Fraction')
ax3.set_ylabel('Event Fraction')
ax4.set_ylabel('Seasonal Fraction') 
ax1.set_xlabel('Variability (k)')        
ax2.set_xlabel('Variability (k)')       
ax3.set_xlabel('Variability (k)') 
ax4.set_xlabel('Mean Runoff [mm/day]')

## Figure 2
f2=plt.figure(num=2,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

for i in range(3):
    idx=cluster==i+1
    ax1.scatter(k[idx],djf_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax2.scatter(k[idx],mam_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax3.scatter(k[idx],jja_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax4.scatter(k[idx],son_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

ax1.set_ylabel('Winter Mean Runoff [mm/day]')
ax2.set_ylabel('Spring Mean Runoff [mm/day]')
ax3.set_ylabel('Summer Mean Runoff [mm/day]')
ax4.set_ylabel('Fall Mean Runoff [mm/day]') 
ax1.set_xlabel('Variability (k)')        
ax2.set_xlabel('Variability (k)')       
ax3.set_xlabel('Variability (k)') 
ax4.set_xlabel('Variability (k)')

## Figure 3
f3=plt.figure(num=3,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

ax1.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
ax2.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
ax3.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
ax4.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')

for i in range(3):
    idx=cluster==i+1
    ax1.scatter(djf_rain[idx],djf_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax2.scatter(mam_rain[idx],mam_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax3.scatter(jja_rain[idx],jja_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax4.scatter(son_rain[idx],son_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

ax1.set_ylabel('Winter Mean Runoff [mm/day]')
ax2.set_ylabel('Spring Mean Runoff [mm/day]')
ax3.set_ylabel('Summer Mean Runoff [mm/day]')
ax4.set_ylabel('Fall Mean Runoff [mm/day]') 
ax1.set_xlabel('Winter Mean Rainfall [mm/day]')
ax2.set_xlabel('Spring Mean Rainfall [mm/day]')
ax3.set_xlabel('Summer Mean Rainfall [mm/day]')
ax4.set_xlabel('Fall Mean Rainfall [mm/day]') 

## Figure 4
f4=plt.figure(num=4,figsize=(15,15))
for i in range(3):
    idx=cluster==i+1
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    plt.subplot(3,1,i+1)
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        if mzOI[j]<2700:
            plt.plot(dn,mnR,c=color_list[i],linewidth=1,linestyle=':')
        else:
            plt.plot(dn,mnR,c=color_list[i],linewidth=2)
    plt.xlabel('Day in Year')
    plt.ylabel('Smoothed Daily Mean Runoff [mm]')
    plt.xlim((0,365))
    plt.ylim((0,18))

## Figure 5
f5=plt.figure(num=5,figsize=(20,10))
for i in range(3):
    idx=cluster==i+1
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    plt.subplot(2,3,i+1)
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
        if mzOI[j]<2700:
            plt.plot(dn,mnP,c=color_list[i],linewidth=1,linestyle=':')
        else:
            plt.plot(dn,mnP,c=color_list[i],linewidth=2)
    plt.xlabel('Day in Year')
    plt.ylabel('Smoothed Daily Mean Rainfall [mm]')
    plt.xlim((0,365))
    plt.ylim((0,8))
    plt.subplot(2,3,4+i)
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        if mzOI[j]<2700:
            plt.plot(dn,mnR,c=color_list[i],linewidth=1,linestyle=':')
        else:
            plt.plot(dn,mnR,c=color_list[i],linewidth=2)
    plt.xlabel('Day in Year')
    plt.ylabel('Smoothed Daily Mean Runoff [mm]')
    plt.xlim((0,365))
    plt.ylim((0,18))    
    
    
## Figure 6
f6=plt.figure(num=6,figsize=(15,15))
for i in range(3):
    idx=cluster==i+1
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    plt.subplot(3,1,i+1)
    plt.plot(np.array([0,365]),np.array([0,0]),c='k',linestyle=':',linewidth=0.5,zorder=0)
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        if mzOI[j]<2700:
            plt.plot(dn,mnP-mnR,c=color_list[i],linewidth=1,linestyle=':')
        else:
            plt.plot(dn,mnP-mnR,c=color_list[i],linewidth=2)
    plt.xlabel('Day in Year')
    plt.ylabel('Rainfall - Runoff')
    plt.xlim((0,365))
    plt.ylim((-12,5))  
    
## Figure 7
f7=plt.figure(num=7,figsize=(20,15))
for i in range(3):
    idx=cluster==i+1
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    dOI=d[idx]
    plt.subplot(3,3,i+1)
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
        if np.logical_and(mzOI[j]<2700,dOI[j]<100):
            plt.plot(dn,mnP,c=color_list[i],linewidth=0.5)
        elif np.logical_and(mzOI[j]<2700,dOI[j]>100):
            plt.plot(dn,mnP,c=color_list[i],linewidth=0.5,linestyle='--')
        elif np.logical_and(mzOI[j]>=2700,dOI[j]<100):
            plt.plot(dn,mnP,c=color_list[i],linewidth=2)
        elif np.logical_and(mzOI[j]>=2700,dOI[j]>100):
            plt.plot(dn,mnP,c=color_list[i],linewidth=2,linestyle='--')
    plt.xlabel('Day in Year')
    plt.ylabel('Smoothed Daily Mean Rainfall [mm]')
    plt.xlim((0,365))
    plt.ylim((0,8))
    plt.title('Cluster '+str(i+1))
    ax=plt.gca()
    ax.axvline(59,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(151,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(243,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(334,c='k',linewidth=0.5,linestyle='--')    
    
    plt.subplot(3,3,4+i)
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        if np.logical_and(mzOI[j]<2700,dOI[j]<100):
            plt.plot(dn,mnR,c=color_list[i],linewidth=0.5)
        elif np.logical_and(mzOI[j]<2700,dOI[j]>100):
            plt.plot(dn,mnR,c=color_list[i],linewidth=0.5,linestyle='--')
        elif np.logical_and(mzOI[j]>=2700,dOI[j]<100):
            plt.plot(dn,mnR,c=color_list[i],linewidth=2)
        elif np.logical_and(mzOI[j]>=2700,dOI[j]>100):
            plt.plot(dn,mnR,c=color_list[i],linewidth=2,linestyle='--')
    plt.xlabel('Day in Year')
    plt.ylabel('Smoothed Daily Mean Runoff [mm]')
    plt.xlim((0,365))
    plt.ylim((0,18))
    ax=plt.gca()
    ax.axvline(59,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(151,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(243,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(334,c='k',linewidth=0.5,linestyle='--')
    
    plt.subplot(3,3,i+7)
    plt.plot(np.array([0,365]),np.array([0,0]),c='k',linestyle=':',linewidth=0.5,zorder=0)
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        if np.logical_and(mzOI[j]<2700,dOI[j]<100):
            plt.plot(dn,mnP-mnR,c=color_list[i],linewidth=0.5)
        elif np.logical_and(mzOI[j]<2700,dOI[j]>100):
            plt.plot(dn,mnP-mnR,c=color_list[i],linewidth=0.5,linestyle='--')
        elif np.logical_and(mzOI[j]>=2700,dOI[j]<100):
            plt.plot(dn,mnP-mnR,c=color_list[i],linewidth=2)
        elif np.logical_and(mzOI[j]>=2700,dOI[j]>100):
            plt.plot(dn,mnP-mnR,c=color_list[i],linewidth=2,linestyle='--')
    plt.xlabel('Day in Year')
    plt.ylabel('Rainfall - Runoff')
    plt.xlim((0,365))
    plt.ylim((-12,5))
    ax=plt.gca()
    ax.axvline(59,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(151,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(243,c='k',linewidth=0.5,linestyle='--')
    ax.axvline(334,c='k',linewidth=0.5,linestyle='--')    