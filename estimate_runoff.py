#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 08:49:58 2021

@author: aforte
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

# Load data from gauged basins
qdf=pd.read_csv('data_tables/grdc_summary_values.csv')
mR=qdf['mean_runoff_mm_day'].to_numpy()
da=qdf['DA_km2'].to_numpy()
mxz=qdf['maxz'].to_numpy()/1000
mRain=qdf['mnTRMM_mm_day'].to_numpy()

# Seasonal Values
djf_run=qdf['DJF_mean_runoff_mm_day'].to_numpy()
mam_run=qdf['MAM_mean_runoff_mm_day'].to_numpy()
jja_run=qdf['JJA_mean_runoff_mm_day'].to_numpy()
son_run=qdf['SON_mean_runoff_mm_day'].to_numpy()

djf_rain=qdf['mnTRMM_djf_mm_day'].to_numpy()
mam_rain=qdf['mnTRMM_mam_mm_day'].to_numpy()
jja_rain=qdf['mnTRMM_jja_mm_day'].to_numpy()
son_rain=qdf['mnTRMM_son_mm_day'].to_numpy()

rr=mR/mRain
djf_rr=djf_run/djf_rain
mam_rr=mam_run/mam_rain
jja_rr=jja_run/jja_rain
son_rr=son_run/son_rain

edf=pd.read_csv('data_tables/gc_ero_master_table.csv')
emxZ=edf['max_el'].to_numpy()/1000
epdf=pd.read_csv('data_tables/ero_TRMM.csv')
emRain=epdf['TRMM_mn_mm_day'].to_numpy()
edjf_rain=epdf['TRMM_DJFmn_mm_day'].to_numpy()
emam_rain=epdf['TRMM_MAMmn_mm_day'].to_numpy()
ejja_rain=epdf['TRMM_JJAmn_mm_day'].to_numpy()
eson_rain=epdf['TRMM_SONmn_mm_day'].to_numpy()


# Generate empty array
power_est=np.zeros((len(emRain),5))
linear_est=np.zeros((len(emRain),5))

# Define functions
def lin(B,x):
    return B[0]*x + B[1]

def powr(B,x):
    return (B[0]*x**B[1])+B[2]
lm=odr.Model(lin)
pm=odr.Model(powr)

def rmse(obs,pred):
    return np.sqrt(np.sum((obs-pred)**2)/len(obs))

f2=plt.figure(2,figsize=(10,5))
ax1=plt.subplot(1,2,1)
ax1.set_xlim((1,6))
ax1.set_ylim((0.4,10))
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel('Mean Runoff [mm/day]')
ax1.set_xlabel('Mean TRMM Rainfall [mm/day]')
ax1.tick_params(direction='in',which='both')
ax2=plt.subplot(1,2,2)
ax2.set_xlim((1,6))
ax2.set_ylim((0,1.75))
ax2.set_ylabel('Mean Runoff / Mean Rainfall')
ax2.set_xlabel('Mean TRMM Rainfall [mm/day]')
ax2.tick_params(direction='in',which='both')

plt.figure(1,figsize=(10,25))

rain_vec=np.linspace(0.5,11,100)

plt.subplot(5,2,1)
plt.scatter(mRain,mR,s=20,c='k')

data=odr.RealData(mRain,mR)
lodr=odr.ODR(data,lm,beta0=[2,-1])
lres=lodr.run()
podr=odr.ODR(data,pm,beta0=[1,2,1])
pres=podr.run()


r=np.round(rmse(mR,powr(pres.beta,mRain)),2)

a=np.round(pres.beta[0],4)
b=np.round(pres.beta[1],2)
c=np.round(pres.beta[2],2)
plt.plot(rain_vec,lin(lres.beta,rain_vec),c='r',label='y = '+str(np.round(lres.beta[0],2))+'x + '+str(np.round(lres.beta[1],2)))
plt.plot(rain_vec,powr(pres.beta,rain_vec),c='b',label='y = '+str(a)+'$x^{'+str(b)+'}$ + '+str(c))
plt.legend(loc='best')
power_est[:,0]=powr(pres.beta,emRain)
linear_est[:,0]=lin(lres.beta,emRain)
plt.ylabel('Mean Runoff [mm/day]')
plt.xlim((0.75,11))
plt.xlim((0.75,11))
plt.xscale('log')
plt.yscale('log')
plt.title('Annual')

plt.subplot(5,2,2)
plt.scatter(mRain,rr,s=20,c='k',label='Observed GRDC')
plt.scatter(emRain,lin(lres.beta,emRain)/emRain,c='r',s=20,label='Linear Fit')
plt.scatter(emRain,powr(pres.beta,emRain)/emRain,c='b',s=20,label='Power Fit')
plt.legend(loc='best')
plt.ylabel('Runoff Ratio')
plt.xlim((0,11))
plt.ylim((0,2.6))

ax1.scatter(mRain,mR,s=20,c='k')
ax1.plot(rain_vec,powr(pres.beta,rain_vec),c='b',label='y = '+str(a)+'$x^{'+str(b)+'}$ + '+str(c)+'; RMSE = '+str(r))
ax1.legend(loc='best')
ax2.scatter(mRain,rr,s=20,c='k',label='Observed GRDC')
ax2.scatter(emRain,powr(pres.beta,emRain)/emRain,c='b',s=20,label='Power Fit')
ax2.legend(loc='best')
f2.savefig('estimate_runoff.pdf')


plt.subplot(5,2,3)
plt.scatter(djf_rain,djf_run,s=20,c='k')
data=odr.RealData(djf_rain,djf_run)
lodr=odr.ODR(data,lm,beta0=[2,-1])
lres=lodr.run()
podr=odr.ODR(data,pm,beta0=[1,2,1])
pres=podr.run()
a=np.round(pres.beta[0],4)
b=np.round(pres.beta[1],2)
c=np.round(pres.beta[2],2)
plt.plot(rain_vec,lin(lres.beta,rain_vec),c='r',label='y = '+str(np.round(lres.beta[0],2))+'x + '+str(np.round(lres.beta[1],2)))
plt.plot(rain_vec,powr(pres.beta,rain_vec),c='b',label='y = '+str(a)+'$x^{'+str(b)+'}$ + '+str(c))
plt.legend(loc='best')
power_est[:,1]=powr(pres.beta,edjf_rain)
linear_est[:,1]=lin(lres.beta,edjf_rain)
plt.ylabel('Mean Runoff [mm/day]')
plt.title('Winter')
plt.xlim((0.75,11))
plt.xlim((0.75,11))
plt.xscale('log')
plt.yscale('log')

plt.subplot(5,2,4)
plt.scatter(djf_rain,djf_rr,s=20,c='k',label='Observed GRDC')
plt.scatter(edjf_rain,lin(lres.beta,edjf_rain)/edjf_rain,c='r',s=20,label='Linear Fit')
plt.scatter(edjf_rain,powr(pres.beta,edjf_rain)/edjf_rain,c='b',s=20,label='Power Fit')
plt.legend(loc='best')
plt.ylabel('Runoff Ratio')
plt.xlim((0,11))
plt.ylim((0,2.6))

plt.subplot(5,2,5)
plt.scatter(mam_rain,mam_run,s=20,c='k')
data=odr.RealData(mam_rain,mam_run)
lodr=odr.ODR(data,lm,beta0=[2,-1])
lres=lodr.run()
podr=odr.ODR(data,pm,beta0=[1,2,1])
pres=podr.run()
a=np.round(pres.beta[0],4)
b=np.round(pres.beta[1],2)
c=np.round(pres.beta[2],2)
plt.plot(rain_vec,lin(lres.beta,rain_vec),c='r',label='y = '+str(np.round(lres.beta[0],2))+'x + '+str(np.round(lres.beta[1],2)))
plt.plot(rain_vec,powr(pres.beta,rain_vec),c='b',label='y = '+str(a)+'$x^{'+str(b)+'}$ + '+str(c))
power_est[:,2]=powr(pres.beta,emam_rain)
linear_est[:,2]=lin(lres.beta,emam_rain)
plt.legend(loc='best')
plt.ylabel('Mean Runoff [mm/day]')
plt.title('Spring')
plt.xlim((0.75,11))
plt.xlim((0.75,11))
plt.xscale('log')
plt.yscale('log')

plt.subplot(5,2,6)
plt.scatter(mam_rain,mam_rr,s=20,c='k',label='Observed GRDC')
plt.scatter(emam_rain,lin(lres.beta,emam_rain)/emam_rain,c='r',s=20,label='Linear Fit')
plt.scatter(emam_rain,powr(pres.beta,emam_rain)/emam_rain,c='b',s=20,label='Power Fit')
plt.legend(loc='best')
plt.ylabel('Runoff Ratio') 
plt.xlim((0,11))
plt.ylim((0,2.6))

plt.subplot(5,2,7)
plt.scatter(jja_rain,jja_run,s=20,c='k')
data=odr.RealData(jja_rain,jja_run)
lodr=odr.ODR(data,lm,beta0=[2,-1])
lres=lodr.run()
podr=odr.ODR(data,pm,beta0=[1,2,1])
pres=podr.run()
a=np.round(pres.beta[0],4)
b=np.round(pres.beta[1],2)
c=np.round(pres.beta[2],2)
plt.plot(rain_vec,lin(lres.beta,rain_vec),c='r',label='y = '+str(np.round(lres.beta[0],2))+'x + '+str(np.round(lres.beta[1],2)))
plt.plot(rain_vec,powr(pres.beta,rain_vec),c='b',label='y = '+str(a)+'$x^{'+str(b)+'}$ + '+str(c))
plt.legend(loc='best')
power_est[:,3]=powr(pres.beta,ejja_rain)
linear_est[:,3]=lin(lres.beta,ejja_rain)
plt.ylabel('Mean Runoff [mm/day]')
plt.title('Summer')
plt.xlim((0.75,11))
plt.xlim((0.75,11))
plt.xscale('log')
plt.yscale('log')

plt.subplot(5,2,8)
plt.scatter(jja_rain,jja_rr,s=20,c='k',label='Observed GRDC')
plt.scatter(ejja_rain,lin(lres.beta,ejja_rain)/ejja_rain,c='r',s=20,label='Linear Fit')
plt.scatter(ejja_rain,powr(pres.beta,ejja_rain)/ejja_rain,c='b',s=20,label='Power Fit')
plt.legend(loc='best')
plt.ylabel('Runoff Ratio')
plt.xlim((0,11))
plt.ylim((0,2.6))

plt.subplot(5,2,9)
plt.scatter(son_rain,son_run,s=20,c='k')
data=odr.RealData(son_rain,son_run)
lodr=odr.ODR(data,lm,beta0=[2,-1])
lres=lodr.run()
podr=odr.ODR(data,pm,beta0=[1,2,1])
pres=podr.run()
a=np.round(pres.beta[0],4)
b=np.round(pres.beta[1],2)
c=np.round(pres.beta[2],2)
plt.plot(rain_vec,lin(lres.beta,rain_vec),c='r',label='y = '+str(np.round(lres.beta[0],2))+'x + '+str(np.round(lres.beta[1],2)))
plt.plot(rain_vec,powr(pres.beta,rain_vec),c='b',label='y = '+str(a)+'$x^{'+str(b)+'}$ + '+str(c))
plt.legend(loc='best')
power_est[:,4]=powr(pres.beta,eson_rain)
linear_est[:,4]=lin(lres.beta,eson_rain)
plt.xlabel('Mean TRMM Rainfall [mm/day]')
plt.ylabel('Mean Runoff [mm/day]')
plt.title('Fall')
plt.xlim((0.75,11))
plt.xlim((0.75,11))
plt.xscale('log')
plt.yscale('log')

plt.subplot(5,2,10)
plt.scatter(son_rain,son_rr,s=20,c='k',label='Observed GRDC')
plt.scatter(eson_rain,lin(lres.beta,eson_rain)/eson_rain,c='r',s=20,label='Linear Fit')
plt.scatter(eson_rain,powr(pres.beta,eson_rain)/eson_rain,c='b',s=20,label='Power Fit')
plt.legend(loc='best')
plt.xlabel('Mean TRMM Rainfall [mm/day]')
plt.ylabel('Runoff Ratio') 
plt.xlim((0,11))
plt.ylim((0,2.6))   

powdf=pd.DataFrame(power_est,columns=['mean_runoff','mean_djf_runoff','mean_mam_runoff','mean_jja_runoff','mean_son_runoff'])
lindf=pd.DataFrame(linear_est,columns=['mean_runoff','mean_djf_runoff','mean_mam_runoff','mean_jja_runoff','mean_son_runoff'])

powdf.to_csv('result_tables/estimate_runoff_power.csv',index=False)
lindf.to_csv('result_tables/estimate_runoff_linear.csv',index=False)