#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:38:39 2021

@author: aforte
"""
import pandas as pd
import numpy as np
from cmcrameri import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import odr

def budyko_func(phi,omega):
    return 1+phi-(1+(phi**omega))**(1/omega)

def min_bud(X,R,PET):
    ET=(X-R)/X
    AR=PET/X
    ETobj=1+AR-(1+(AR)**2.6)**(1/2.6)
    return ETobj-ET


df=pd.read_csv('result_tables/GRDC_Distribution_Fits.csv')
cb=df['c_best'].to_numpy()
sb=df['s_best'].to_numpy()

gdf=pd.read_csv('data_tables/grdc_budyko.csv')
arid=gdf['aridity_index'].to_numpy()
et=gdf['et_ratio'].to_numpy()
r=gdf['mn_runoff_mm_yr'].to_numpy()
rain=gdf['mn_rainfall_mm_yr'].to_numpy()
pet=gdf['mn_PET_mm_yr'].to_numpy()

a_vec=np.linspace(0,4,100)

cnorm=colors.Normalize(vmin=0.2,vmax=1.6)
snorm=colors.Normalize(vmin=0.1,vmax=1.3)

plt.figure(1,figsize=(10,10))
ax1=plt.subplot(2,1,1)
sc1=plt.scatter(arid,et,s=50,c=cb,norm=cnorm,cmap=cm.vik,edgecolors='k')
plt.plot(a_vec,budyko_func(a_vec,2.6),c='k',linestyle=':')
plt.xlabel('Aridity Index')
plt.ylabel('ET Ratio')
cbar1=plt.colorbar(sc1,ax=ax1)
cbar1.ax.set_ylabel('Shape Parameter')

ax2=plt.subplot(2,1,2)
sc2=plt.scatter(arid,et,s=50,c=sb,norm=snorm,cmap=cm.lapaz,edgecolors='k')
plt.plot(a_vec,budyko_func(a_vec,2.6),c='k',linestyle=':')
plt.xlabel('Aridity Index')
plt.ylabel('ET Ratio')
cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel('Scale Parameter')


## Estimate precipitation assuming Budyko holds
bud_precip=np.zeros((len(cb)))
for i in range(len(cb)):
    bud_precip[i]=fsolve(min_bud,[1000],args=(r[i],pet[i]))

# Convert to mm/day
eP=bud_precip/365.25

qdf=pd.read_csv('data_tables/grdc_summary_values.csv')
mR=qdf['mean_runoff_mm_day'].to_numpy()
mRain=qdf['mnTRMM_mm_day'].to_numpy()

epdf=pd.read_csv('data_tables/ero_TRMM.csv')
emRain=epdf['TRMM_mn_mm_day'].to_numpy()

def lin(B,x):
    return B[0]*x + B[1]

def powr(B,x):
    return (B[0]*x**B[1])+B[2]
lm=odr.Model(lin)
pm=odr.Model(powr)

data1=odr.RealData(mRain,mR)
podr=odr.ODR(data1,pm,beta0=[1,2,1])
pres=podr.run()


data2=odr.RealData(mRain,eP)
podr2=odr.ODR(data2,pm,beta0=[1,2,3])
pres2=podr2.run()

data3=odr.RealData(eP,mR)
lodr=odr.ODR(data3,lm,beta0=[1,-5])
lres=lodr.run()


plt.figure(2,figsize=(10,10))
rnorm=colors.Normalize(vmin=0,vmax=6)
ax3=plt.subplot(3,1,1)
plt.plot([0,8],[0,8],c='gray',linestyle=':',zorder=1)
plt.plot(np.linspace(0,8),powr(pres2.beta,np.linspace(0,8)),c='k')
sc3=plt.scatter(mRain,eP,s=50,cmap=cm.vik_r,c=mR,norm=rnorm)
plt.xlim((0,8))
plt.ylim((0,8))
plt.xlabel('TRMM Rainfall [mm/day]')
plt.ylabel('Implied Precipitation from Budyko [mm/day]')
cbar3=plt.colorbar(sc3,ax=plt.gca())
cbar3.ax.set_ylabel('Runoff [mm/day]')  

plt.subplot(3,1,2)
plt.scatter(mRain,mR,s=50,c='k',label='TRMM')
rain_vec=np.linspace(0,10)
plt.plot(rain_vec,powr(pres.beta,rain_vec),c='k',linestyle='-')
plt.plot(rain_vec,lin(lres.beta,rain_vec),c='b',linestyle='-')
plt.scatter(eP,mR,s=50,c='b',label='Implied from Budyko')
plt.xlabel('Rainfall or Precipitation [mm/day]')
plt.ylabel('Runoff [mm/day]')
plt.legend(loc='best') 
plt.ylim((0,9))
plt.xlim((0,9))

impR=powr(pres.beta,emRain)
impRB=lin(lres.beta,powr(pres2.beta,emRain))
plt.subplot(3,1,3)
plt.plot([0,7],[0,7],c='gray',linestyle=':')
plt.scatter(impR,impRB,s=50,c='k')
plt.xlabel('Estimated Runoff from TRMM to Runoff')
plt.ylabel('Estimated Runoff from TRMM to EP to Runoff')
plt.ylim((0,7))
plt.xlim((0,7))



    