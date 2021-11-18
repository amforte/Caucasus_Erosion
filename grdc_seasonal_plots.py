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
mz=qdf['maxz'].to_numpy()/1000
snow=qdf['ssnstd'].to_numpy()
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

df=pd.read_csv('result_tables/grdc_basin_clusters.csv')
cluster=df['cluster'].to_numpy().astype('int')
grdc_id=df['grdc_id'].to_numpy().astype('int')


# Colors for clusters
color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']

# Difference in peak
diff_peak=np.zeros((len(grdc_id),3))
for i in range(len(grdc_id)):
    fn='data_tables/grdc_daily_means/grdc_'+str(grdc_id[i])+'_mean_daily.csv'
    bdf=pd.read_csv(fn)
    dn=bdf['day_number'].to_numpy()
    mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
    mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
    rmax=np.argmax(mnR)
    rdn=dn[rmax] # Day in year of max runoff        
    pmax=np.argmax(mnP)
    pdn=dn[pmax] # Day in year of max rainfall
    # Convert to radians
    r_theta=(rdn/365)*2*np.pi
    p_theta=(pdn/365)*2*np.pi
    # Normalize to runoff angle
    p_theta=p_theta-r_theta
    r_theta=r_theta-r_theta
    # Convert to cartesian
    rx=np.cos(r_theta)
    ry=np.sin(r_theta)
    px=np.cos(p_theta)
    py=np.sin(p_theta)
    rv=np.array([rx,ry,0])
    pv=np.array([px,py,0])
        
    # Find angle between
    a=np.arctan2(np.linalg.norm(np.cross(rv,pv)),np.dot(rv,pv))
    diff_peak[i,0]=rdn
    diff_peak[i,1]=pdn
    diff_peak[i,2]=(a/(2*np.pi))*365
dp=diff_peak[:,2]


## Master Figure - Shape
f1=plt.figure(num=100,figsize=(15,20))

axl1=plt.subplot(4,2,1)
axl2=plt.subplot(4,2,3)
axl3=plt.subplot(4,2,5)
axl4=plt.subplot(4,2,7)
axr1=plt.subplot(3,2,2)
axr2=plt.subplot(3,2,4)
axr3=plt.subplot(3,2,6)

lcnum=np.arange(1,8,2)
for i in range(4):
    idx=cluster==i
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    dOI=d[idx]
    plt.subplot(4,2,lcnum[i])
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
        pks_max=np.argmax(mnP)
        
        if mzOI[j]<2.7:
            if dOI[j]<100:
                plt.plot(dn,mnR,c=color_list[i],linewidth=1)
                plt.scatter(dn[pks_max],mnP[pks_max],c=color_list[i],s=20)
            else:
                plt.plot(dn,mnR,c=color_list[i],linewidth=1,linestyle=':')
                plt.scatter(dn[pks_max],mnP[pks_max],edgecolors=color_list[i],c='w',s=20)                
        else:
            if dOI[j]<100:
                plt.plot(dn,mnR,c=color_list[i],linewidth=2)
                plt.scatter(dn[pks_max],mnP[pks_max],c=color_list[i],s=40,marker='s')
            else:
                plt.plot(dn,mnR,c=color_list[i],linewidth=2,linestyle=':')
                plt.scatter(dn[pks_max],mnP[pks_max],edgecolors=color_list[i],c='w',s=40,marker='s')                    
    plt.axvline(59,c='k',linewidth=0.5,linestyle='--')
    plt.axvline(151,c='k',linewidth=0.5,linestyle='--')
    plt.axvline(243,c='k',linewidth=0.5,linestyle='--')
    plt.axvline(334,c='k',linewidth=0.5,linestyle='--')
    plt.xlabel('Day in Year')
    plt.ylabel('Smoothed Daily Mean Runoff [mm]')
    plt.xlim((0,365))
    plt.ylim((0,18))
    
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        # Find peak in runoff
        rmax=np.argmax(mnR)
        rdn=dn[rmax]
        # Determine seasons
        if np.logical_or(rdn<=59,rdn>334):
            # DJF
            if dOI[j]<100:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
            else:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='o')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='o')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='o')                    
        elif np.logical_and(rdn>59,rdn<=151):
            # MAM '^'
            if dOI[j]<100:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
            else:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='^')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='^')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='^')
        elif np.logical_and(rdn>151,rdn<=243):
            # JJA 's'
            if dOI[j]<100:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
            else:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='s')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='s')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='s')
        elif np.logical_and(rdn>243,rdn<=334):
            # SON 'D'
            if dOI[j]<100:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
            else:
                axr1.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='D')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='D')
                axr3.scatter(c[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='D')

## DO IT YOURSELF EXPLANATION
axr2.scatter(80,0.15,s=5*25,c='k')
axr2.scatter(80,0.12,s=4*25,c='k')
axr2.scatter(80,0.09,s=3*25,c='k')
axr2.scatter(80,0.06,s=2*25,c='k')
axr2.scatter(80,0.03,s=1*25,c='k')
axr2.text(90,0.15,'5 km')
axr2.text(90,0.12,'4 km')
axr2.text(90,0.09,'3 km')
axr2.text(90,0.06,'2 km')
axr2.text(90,0.03,'1 km')
axr2.text(80,0.18,'Max Elev.')

axr2.scatter(120,0.15,s=3*25,marker='o',c='k')
axr2.scatter(120,0.12,s=3*25,marker='^',c='k')
axr2.scatter(120,0.09,s=3*25,marker='s',c='k')
axr2.scatter(120,0.06,s=3*25,marker='D',c='k')
axr2.text(130,0.15,'DJF')
axr2.text(130,0.12,'MAM')
axr2.text(130,0.09,'JJA')
axr2.text(130,0.06,'SON')
axr2.text(110,0.18,'Peak Runoff Season')

axr2.scatter(35,0.06,s=3*25,c='k')
axr2.scatter(35,0.03,s=3*25,c='w',edgecolors='k')
axr2.text(45,0.06,'In GC')
axr2.text(45,0.03,'Outside GC')
axr2.text(35,0.09,'Position')

axr1.set_xlabel('Shape')
axr2.set_xlabel('Difference in Peaks [days]')
axr3.set_xlabel('Shape')

axr1.set_ylabel('Seasonal Fraction')        
axr2.set_ylabel('Seasonal Fraction')       
axr3.set_ylabel('Seasonal Snow STD')


## Master Figure - Scale
f2=plt.figure(num=200,figsize=(15,20))

axl1=plt.subplot(4,2,1)
axl2=plt.subplot(4,2,3)
axl3=plt.subplot(4,2,5)
axl4=plt.subplot(4,2,7)
axr1=plt.subplot(3,2,2)
axr2=plt.subplot(3,2,4)
axr3=plt.subplot(3,2,6)

lcnum=np.arange(1,8,2)
for i in range(4):
    idx=cluster==i
    idOI=grdc_id[idx]
    mzOI=mz[idx]
    dOI=d[idx]
    plt.subplot(4,2,lcnum[i])
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
        pks_max=np.argmax(mnP)
        
        if mzOI[j]<2.7:
            if dOI[j]<100:
                plt.plot(dn,mnR,c=color_list[i],linewidth=1)
                plt.scatter(dn[pks_max],mnP[pks_max],c=color_list[i],s=20)
            else:
                plt.plot(dn,mnR,c=color_list[i],linewidth=1,linestyle=':')
                plt.scatter(dn[pks_max],mnP[pks_max],edgecolors=color_list[i],c='w',s=20)                
        else:
            if dOI[j]<100:
                plt.plot(dn,mnR,c=color_list[i],linewidth=2)
                plt.scatter(dn[pks_max],mnP[pks_max],c=color_list[i],s=40,marker='s')
            else:
                plt.plot(dn,mnR,c=color_list[i],linewidth=2,linestyle=':')
                plt.scatter(dn[pks_max],mnP[pks_max],edgecolors=color_list[i],c='w',s=40,marker='s')                    
    plt.axvline(59,c='k',linewidth=0.5,linestyle='--')
    plt.axvline(151,c='k',linewidth=0.5,linestyle='--')
    plt.axvline(243,c='k',linewidth=0.5,linestyle='--')
    plt.axvline(334,c='k',linewidth=0.5,linestyle='--')
    plt.xlabel('Day in Year')
    plt.ylabel('Smoothed Daily Mean Runoff [mm]')
    plt.xlim((0,365))
    plt.ylim((0,18))
    
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        # Find peak in runoff
        rmax=np.argmax(mnR)
        rdn=dn[rmax]
        # Determine seasons
        if np.logical_or(rdn<=59,rdn>334):
            # DJF
            if dOI[j]<100:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
            else:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='o')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='o')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='o')                    
        elif np.logical_and(rdn>59,rdn<=151):
            # MAM '^'
            if dOI[j]<100:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
            else:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='^')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='^')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='^')
        elif np.logical_and(rdn>151,rdn<=243):
            # JJA 's'
            if dOI[j]<100:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
            else:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='s')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='s')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='s')
        elif np.logical_and(rdn>243,rdn<=334):
            # SON 'D'
            if dOI[j]<100:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
            else:
                axr1.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='D')
                axr2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='D')
                axr3.scatter(s[idx][j],snow[idx][j],s=mz[idx][j]*25,edgecolors=color_list[i],c='w',marker='D')


## DO IT YOURSELF EXPLANATION
axr2.scatter(80,0.15,s=5*25,c='k')
axr2.scatter(80,0.12,s=4*25,c='k')
axr2.scatter(80,0.09,s=3*25,c='k')
axr2.scatter(80,0.06,s=2*25,c='k')
axr2.scatter(80,0.03,s=1*25,c='k')
axr2.text(90,0.15,'5 km')
axr2.text(90,0.12,'4 km')
axr2.text(90,0.09,'3 km')
axr2.text(90,0.06,'2 km')
axr2.text(90,0.03,'1 km')
axr2.text(80,0.18,'Max Elev.')

axr2.scatter(120,0.15,s=3*25,marker='o',c='k')
axr2.scatter(120,0.12,s=3*25,marker='^',c='k')
axr2.scatter(120,0.09,s=3*25,marker='s',c='k')
axr2.scatter(120,0.06,s=3*25,marker='D',c='k')
axr2.text(130,0.15,'DJF')
axr2.text(130,0.12,'MAM')
axr2.text(130,0.09,'JJA')
axr2.text(130,0.06,'SON')
axr2.text(110,0.18,'Peak Runoff Season')

axr2.scatter(35,0.06,s=3*25,c='k')
axr2.scatter(35,0.03,s=3*25,c='w',edgecolors='k')
axr2.text(45,0.06,'In GC')
axr2.text(45,0.03,'Outside GC')
axr2.text(35,0.09,'Position')

axr1.set_xlabel('Scale')
axr2.set_xlabel('Difference in Peaks [days]')
axr3.set_xlabel('Scale')

axr1.set_ylabel('Seasonal Fraction')        
axr2.set_ylabel('Seasonal Fraction')       
axr3.set_ylabel('Seasonal Snow STD')


# f1.savefig('seasonal_shape.pdf')
# f2.savefig('seasonal_scale.pdf')

# ## Figure 3
# f3=plt.figure(num=3,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)

# for i in range(4):
#     idx=cluster==i
#     ax1.scatter(c[idx],anu_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax2.scatter(c[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax3.scatter(c[idx],evnt_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax4.scatter(mR[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

# ax1.set_ylabel('Annual Fraction')
# ax2.set_ylabel('Seasonal Fraction')
# ax3.set_ylabel('Event Fraction')
# ax4.set_ylabel('Seasonal Fraction') 
# ax1.set_xlabel('Shape')        
# ax2.set_xlabel('Shape')       
# ax3.set_xlabel('Shape') 
# ax4.set_xlabel('Mean Runoff [mm/day]')

# ## Figure 4
# f4=plt.figure(num=4,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)

# for i in range(4):
#     idx=cluster==i
#     ax1.scatter(s[idx],anu_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax2.scatter(s[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax3.scatter(s[idx],evnt_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax4.scatter(mR[idx],ssn_frac[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

# ax1.set_ylabel('Annual Fraction')
# ax2.set_ylabel('Seasonal Fraction')
# ax3.set_ylabel('Event Fraction')
# ax4.set_ylabel('Seasonal Fraction') 
# ax1.set_xlabel('Scale')        
# ax2.set_xlabel('Scale')       
# ax3.set_xlabel('Scale') 
# ax4.set_xlabel('Mean Runoff [mm/day]')

# ## Figure 5
# f5=plt.figure(num=5,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)

# for i in range(4):
#     idx=cluster==i
#     ax1.scatter(c[idx],djf_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax2.scatter(c[idx],mam_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax3.scatter(c[idx],jja_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax4.scatter(c[idx],son_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

# ax1.set_ylabel('Winter Mean Runoff [mm/day]')
# ax2.set_ylabel('Spring Mean Runoff [mm/day]')
# ax3.set_ylabel('Summer Mean Runoff [mm/day]')
# ax4.set_ylabel('Fall Mean Runoff [mm/day]') 
# ax1.set_xlabel('Shape')        
# ax2.set_xlabel('Shape')       
# ax3.set_xlabel('Shape') 
# ax4.set_xlabel('Shape')

# ## Figure 6
# f6=plt.figure(num=6,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)

# for i in range(4):
#     idx=cluster==i
#     ax1.scatter(s[idx],djf_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax2.scatter(s[idx],mam_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax3.scatter(s[idx],jja_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax4.scatter(s[idx],son_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

# ax1.set_ylabel('Winter Mean Runoff [mm/day]')
# ax2.set_ylabel('Spring Mean Runoff [mm/day]')
# ax3.set_ylabel('Summer Mean Runoff [mm/day]')
# ax4.set_ylabel('Fall Mean Runoff [mm/day]') 
# ax1.set_xlabel('Scale')        
# ax2.set_xlabel('Scale')       
# ax3.set_xlabel('Scale') 
# ax4.set_xlabel('Scale')

# ## Figure 7
# f7=plt.figure(num=7,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)

# ax1.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
# ax2.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
# ax3.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')
# ax4.plot(np.array([0,12]),np.array([0,12]),c='k',linestyle=':')

# for i in range(4):
#     idx=cluster==i
#     ax1.scatter(djf_rain[idx],djf_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax2.scatter(mam_rain[idx],mam_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax3.scatter(jja_rain[idx],jja_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')
#     ax4.scatter(son_rain[idx],son_run[idx],s=da[idx]/10,c=color_list[i],edgecolors='k')

# ax1.set_ylabel('Winter Mean Runoff [mm/day]')
# ax2.set_ylabel('Spring Mean Runoff [mm/day]')
# ax3.set_ylabel('Summer Mean Runoff [mm/day]')
# ax4.set_ylabel('Fall Mean Runoff [mm/day]') 
# ax1.set_xlabel('Winter Mean Rainfall [mm/day]')
# ax2.set_xlabel('Spring Mean Rainfall [mm/day]')
# ax3.set_xlabel('Summer Mean Rainfall [mm/day]')
# ax4.set_xlabel('Fall Mean Rainfall [mm/day]') 

# ## Figure 8
# f8=plt.figure(num=8,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)

# for i in range(4):
#     idx=cluster==i
#     idOI=grdc_id[idx]
#     for j in range(len(idOI)):
#         fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
#         bdf=pd.read_csv(fn)
#         dn=bdf['day_number'].to_numpy()
#         mnP=bdf['trmm_smoothed_mean_daily_rainfall_mm_day'].to_numpy()
#         # Find peak in rainfall
#         pmax=np.argmax(mnP)
#         pdn=dn[pmax]

#         # Determine seasons
#         if np.logical_or(pdn<=59,pdn>334):
#             # DJF
#             ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax2.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax3.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax4.scatter(mR[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#         elif np.logical_and(pdn>59,pdn<=151):
#             # MAM
#             ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax2.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax3.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax4.scatter(mR[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#         elif np.logical_and(pdn>151,pdn<=243):
#             # JJA
#             ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax2.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax3.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax4.scatter(mR[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#         elif np.logical_and(pdn>243,pdn<=334):
#             # SON
#             ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax2.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax3.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax4.scatter(mR[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')                       
# ax1.set_ylabel('Annual Fraction')
# ax2.set_ylabel('Seasonal Fraction')
# ax3.set_ylabel('Event Fraction')
# ax4.set_ylabel('Seasonal Fraction') 
# ax1.set_xlabel('Shape')        
# ax2.set_xlabel('Shape')       
# ax3.set_xlabel('Shape') 
# ax4.set_xlabel('Mean Runoff [mm/day]')

## Figure 9
f9=plt.figure(num=9,figsize=(15,20))
ax1=plt.subplot(3,2,1)
ax2=plt.subplot(3,2,2)
ax3=plt.subplot(3,2,3)
ax4=plt.subplot(3,2,4)
ax5=plt.subplot(3,2,5)
ax6=plt.subplot(3,2,6)

for i in range(4):
    idx=cluster==i
    idOI=grdc_id[idx]
    for j in range(len(idOI)):
        fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
        bdf=pd.read_csv(fn)
        dn=bdf['day_number'].to_numpy()
        mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
        # Find peak in runoff
        rmax=np.argmax(mnR)
        rdn=dn[rmax]
        # Determine seasons
        if np.logical_or(rdn<=59,rdn>334):
            # DJF
            ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
            ax3.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
            ax5.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
            ax2.scatter(s[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
            ax4.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
            ax6.scatter(s[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
        elif np.logical_and(rdn>59,rdn<=151):
            # MAM
            ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
            ax3.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
            ax5.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
            ax2.scatter(s[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
            ax4.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
            ax6.scatter(s[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
        elif np.logical_and(rdn>151,rdn<=243):
            # JJA
            ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
            ax3.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
            ax5.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
            ax2.scatter(s[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
            ax4.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
            ax6.scatter(s[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
        elif np.logical_and(rdn>243,rdn<=334):
            # SON
            ax1.scatter(c[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
            ax3.scatter(c[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
            ax5.scatter(c[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
            ax2.scatter(s[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
            ax4.scatter(s[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
            ax6.scatter(s[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')                      
ax1.set_ylabel('Annual Fraction')
ax2.set_ylabel('Annual Fraction')
ax3.set_ylabel('Seasonal Fraction')
ax4.set_ylabel('Seasonal Fraction') 
ax5.set_ylabel('Event Fraction')
ax6.set_ylabel('Event Fraction')

ax1.set_xlabel('Shape')        
ax3.set_xlabel('Shape')       
ax5.set_xlabel('Shape') 
ax2.set_xlabel('Scale')
ax4.set_xlabel('Scale')
ax6.set_xlabel('Scale')


ax4.scatter(0.8,0.15,s=5*25,c='k')
ax4.scatter(0.8,0.12,s=4*25,c='k')
ax4.scatter(0.8,0.09,s=3*25,c='k')
ax4.scatter(0.8,0.06,s=2*25,c='k')
ax4.scatter(0.8,0.03,s=1*25,c='k')
ax4.text(0.9,0.15,'5 km')
ax4.text(0.9,0.12,'4 km')
ax4.text(0.9,0.09,'3 km')
ax4.text(0.9,0.06,'2 km')
ax4.text(0.9,0.03,'1 km')
ax4.text(0.9,0.18,'Max Elev.')

ax4.scatter(1.1,0.15,s=3*25,marker='o',c='k')
ax4.scatter(1.1,0.12,s=3*25,marker='^',c='k')
ax4.scatter(1.1,0.09,s=3*25,marker='s',c='k')
ax4.scatter(1.1,0.06,s=3*25,marker='D',c='k')
ax4.text(1.2,0.15,'DJF')
ax4.text(1.2,0.12,'MAM')
ax4.text(1.2,0.09,'JJA')
ax4.text(1.2,0.06,'SON')
ax4.text(1.2,0.18,'Peak Runoff Season')

ax4.scatter(0.4,0.06,s=3*25,c='k')
ax4.scatter(0.4,0.03,s=3*25,c='w',edgecolors='k')
ax2.text(0.45,0.06,'In GC')
ax4.text(0.45,0.03,'Outside GC')
ax4.text(0.35,0.09,'Position')

f9.savefig('fractions.pdf')

# ## FIgure 10
# f10=plt.figure(num=10,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)

# for i in range(4):
#     idx=cluster==i
#     idOI=grdc_id[idx]
#     for j in range(len(idOI)):
#         fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
#         bdf=pd.read_csv(fn)
#         dn=bdf['day_number'].to_numpy()
#         mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
#         # Find peak in runoff
#         rmax=np.argmax(mnR)
#         rdn=dn[rmax]
#         # Determine seasons
#         if np.logical_or(rdn<=59,rdn>334):
#             # DJF
#             ax1.scatter(dp[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax3.scatter(dp[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax4.scatter(mR[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#         elif np.logical_and(rdn>59,rdn<=151):
#             # MAM
#             ax1.scatter(dp[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax3.scatter(dp[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax4.scatter(mR[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#         elif np.logical_and(rdn>151,rdn<=243):
#             # JJA
#             ax1.scatter(dp[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax3.scatter(dp[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax4.scatter(mR[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#         elif np.logical_and(rdn>243,rdn<=334):
#             # SON
#             ax1.scatter(dp[idx][j],anu_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax2.scatter(dp[idx][j],ssn_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax3.scatter(dp[idx][j],evnt_frac[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax4.scatter(mR[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')                       
# ax1.set_ylabel('Annual Fraction')
# ax2.set_ylabel('Seasonal Fraction')
# ax3.set_ylabel('Event Fraction')
# ax4.set_ylabel('Difference in Peaks [days]') 
# ax1.set_xlabel('Difference in Peaks [days]')        
# ax2.set_xlabel('Difference in Peaks [days]')       
# ax3.set_xlabel('Difference in Peaks [days]') 
# ax4.set_xlabel('Mean Runoff [mm/day]')

# ## Figure 11
# f11=plt.figure(num=11,figsize=(15,15))
# ax1=plt.subplot(2,2,1)
# ax2=plt.subplot(2,2,2)
# ax3=plt.subplot(2,2,3)
# ax4=plt.subplot(2,2,4)
# for i in range(4):
#     idx=cluster==i
#     idOI=grdc_id[idx]
#     for j in range(len(idOI)):
#         fn='data_tables/grdc_daily_means/grdc_'+str(idOI[j])+'_mean_daily.csv'
#         bdf=pd.read_csv(fn)
#         dn=bdf['day_number'].to_numpy()
#         mnR=bdf['grdc_smoothed_mean_daily_runoff_mm_day'].to_numpy()
#         # Find peak in runoff
#         rmax=np.argmax(mnR)
#         rdn=dn[rmax]
#         # Determine seasons
#         if np.logical_or(rdn<=59,rdn>334):
#             # DJF
#             ax1.scatter(c[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax2.scatter(s[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax3.scatter(snow[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#             ax4.scatter(snow[idx][j],c[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='o')
#         elif np.logical_and(rdn>59,rdn<=151):
#             # MAM
#             ax1.scatter(c[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax2.scatter(s[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax3.scatter(snow[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#             ax4.scatter(snow[idx][j],c[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='^')
#         elif np.logical_and(rdn>151,rdn<=243):
#             # JJA
#             ax1.scatter(c[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax2.scatter(s[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax3.scatter(snow[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#             ax4.scatter(snow[idx][j],c[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='s')
#         elif np.logical_and(rdn>243,rdn<=334):
#             # SON
#             ax1.scatter(c[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax2.scatter(s[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax3.scatter(snow[idx][j],dp[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
#             ax4.scatter(snow[idx][j],c[idx][j],s=mz[idx][j]*25,c=color_list[i],edgecolors='k',marker='D')
                    
# ax1.set_ylabel('Difference in Peaks [days]')
# ax2.set_ylabel('Difference in Peaks [days]')
# ax3.set_ylabel('Difference in Peaks [days]')
# ax4.set_ylabel('Shape')

# ax1.set_xlabel('Shape')        
# ax2.set_xlabel('Scale')       
# ax3.set_xlabel('Seasonal Snow STD') 
# ax4.set_xlabel('Seasonal Snow STD') 

 
