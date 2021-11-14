#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:10:22 2021

@author: amforte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('result_tables/grdc_basin_clusters.csv')
cluster=df['cluster'].to_numpy().astype('int')
grdc_id=df['grdc_id'].to_numpy().astype('int')

qdf=pd.read_csv('data_tables/grdc_summary_values.csv')
mR=qdf['mean_runoff_mm_day'].to_numpy()
mz=qdf['maxz'].to_numpy()
do=qdf['dist_from_sw_km'].to_numpy()
d=np.copy(do)
d[np.isnan(d)]=150

color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']

## Figure 4
f4=plt.figure(num=4,figsize=(15,15))
for i in range(4):
    idx=cluster==i
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    plt.subplot(4,1,i+1)
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
for i in range(4):
    idx=cluster==i
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    plt.subplot(2,4,i+1)
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
    plt.subplot(2,4,5+i)
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
for i in range(4):
    idx=cluster==i
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    plt.subplot(4,1,i+1)
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
for i in range(4):
    idx=cluster==i
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    dOI=d[idx]
    plt.subplot(3,4,i+1)
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
    
    plt.subplot(3,4,i+5)
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
    
    plt.subplot(3,4,i+9)
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