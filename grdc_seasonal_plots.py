# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

qdf=pd.read_csv('data_tables/grdc_summary_values.csv')
mR=qdf['mean_runoff_mm_day'].to_numpy()
ssn_frac=qdf['seasonal_frac'].to_numpy()
anu_frac=qdf['annual_frac'].to_numpy()
evnt_frac=qdf['event_frac'].to_numpy()
da=qdf['DA_km2'].to_numpy()
mz=qdf['maxz'].to_numpy()
do=qdf['dist_from_sw_km'].to_numpy()
d=np.copy(do)
d[np.isnan(d)]=150

djf_run=qdf['DJF_mean_runoff_mm_day'].to_numpy()
mam_run=qdf['MAM_mean_runoff_mm_day'].to_numpy()
jja_run=qdf['JJA_mean_runoff_mm_day'].to_numpy()
son_run=qdf['SON_mean_runoff_mm_day'].to_numpy()

djf_rain=qdf['mnTRMM_djf_mm_day'].to_numpy()
mam_rain=qdf['mnTRMM_mam_mm_day'].to_numpy()
jja_rain=qdf['mnTRMM_jja_mm_day'].to_numpy()
son_rain=qdf['mnTRMM_son_mm_day'].to_numpy()

pdf=pd.read_csv('result_tables/GRDC_Distribution_Fits.csv')
c=pdf['c_best'].to_numpy()
s=pdf['s_best'].to_numpy()

df=pd.read_csv('result_tables/GRDC_Clusters.csv')
cluster=df['Cluster'].to_numpy().astype('int')
grdc_id=df['GRDC_ID'].to_numpy().astype('int')


# Colors for clusters
color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']

## Figure 1
f1=plt.figure(num=1,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

for i in range(4):
    idx=cluster==i
    ax1.scatter(c[idx],anu_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax2.scatter(c[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax3.scatter(c[idx],evnt_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax4.scatter(mR[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

ax1.set_ylabel('Annual Fraction')
ax2.set_ylabel('Seasonal Fraction')
ax3.set_ylabel('Event Fraction')
ax4.set_ylabel('Seasonal Fraction') 
ax1.set_xlabel('Shape')        
ax2.set_xlabel('Shape')       
ax3.set_xlabel('Shape') 
ax4.set_xlabel('Mean Runoff [mm/day]')

## Figure 2
f2=plt.figure(num=2,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

for i in range(4):
    idx=cluster==i
    ax1.scatter(s[idx],anu_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax2.scatter(s[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax3.scatter(s[idx],evnt_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax4.scatter(mR[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

ax1.set_ylabel('Annual Fraction')
ax2.set_ylabel('Seasonal Fraction')
ax3.set_ylabel('Event Fraction')
ax4.set_ylabel('Seasonal Fraction') 
ax1.set_xlabel('Scale')        
ax2.set_xlabel('Scale')       
ax3.set_xlabel('Scale') 
ax4.set_xlabel('Mean Runoff [mm/day]')

## Figure 3
f3=plt.figure(num=3,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

for i in range(4):
    idx=cluster==i
    ax1.scatter(c[idx],djf_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax2.scatter(c[idx],mam_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax3.scatter(c[idx],jja_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax4.scatter(c[idx],son_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

ax1.set_ylabel('Winter Mean Runoff [mm/day]')
ax2.set_ylabel('Spring Mean Runoff [mm/day]')
ax3.set_ylabel('Summer Mean Runoff [mm/day]')
ax4.set_ylabel('Fall Mean Runoff [mm/day]') 
ax1.set_xlabel('Shape')        
ax2.set_xlabel('Shape')       
ax3.set_xlabel('Shape') 
ax4.set_xlabel('Shape')

## Figure 4
f4=plt.figure(num=4,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

for i in range(4):
    idx=cluster==i
    ax1.scatter(s[idx],djf_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax2.scatter(s[idx],mam_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax3.scatter(s[idx],jja_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
    ax4.scatter(s[idx],son_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

ax1.set_ylabel('Winter Mean Runoff [mm/day]')
ax2.set_ylabel('Spring Mean Runoff [mm/day]')
ax3.set_ylabel('Summer Mean Runoff [mm/day]')
ax4.set_ylabel('Fall Mean Runoff [mm/day]') 
ax1.set_xlabel('Scale')        
ax2.set_xlabel('Scale')       
ax3.set_xlabel('Scale') 
ax4.set_xlabel('Scale')

## Figure 5
f5=plt.figure(num=5,figsize=(15,15))
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,2,3)
ax4=plt.subplot(2,2,4)

ax1.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
ax2.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
ax3.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
ax4.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')

for i in range(4):
    idx=cluster==i
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
   