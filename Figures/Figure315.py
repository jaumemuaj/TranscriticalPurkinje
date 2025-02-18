#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:48:35 2025

@author: jaume
"""

from TCplusplus import *

import time
import matplotlib.pyplot as plt
from functools import partial
import typing
import jax
import jax.numpy as jnp
import seaborn as sns
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d


#f/I plots for PF stimulus
pct=0.05
variation=['Iint','zmax','Deltaeps_z','max_vCas','dres_coeff','m','tauR','Ca_half','slp','tauCa','C',
      'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p',]

N=25

spike_raster=[]
avg_freqs=[]
std_freqs=[]


Ipfs=np.linspace(0,0.4,17)
batch_id=0

for d in [1,2,3]:
    for i in Ipfs:
        for n in range(N):
            with open('params1606.json', 'r') as file:
                params=json.load(file)
            var=pct*(np.round(np.random.rand(len(variation)),3)*2-1)
            for param,v in zip(variation,var):
                params[param]+=v*params[param]
            tr=simulation(params,PF=(1200,i,d),plots=False)
            spike_raster.append(np.array(tr['spikes']))
            
        batch=spike_raster[N*batch_id:N*(batch_id+1)]
        _, id_freq_avg,_,_=smooth_freq(batch, bin_size=15, start_time=500, end_time=1500, sigma=30,plot=False)
        avg_freqs.append(np.mean(id_freq_avg))
        std_freqs.append(np.std(id_freq_avg))
        batch_id+=1



for n,color in zip(np.arange(0,len(avg_freqs),len(Ipfs)),['dimgrey','darkmagenta','olive']):
    n=int(n)
    plt.errorbar(Ipfs*1000,avg_freqs[n:n+len(Ipfs)],yerr=std_freqs[n:n+len(Ipfs)],label=f'S{int(n/len(avg_freqs)*3)+1}',color=color,
                 capsize=5,fmt='-')

plt.legend()
plt.show()

#%%
with open('params1606.json', 'r') as file:
    params=json.load(file)

sns.set(style="ticks", context="paper", font_scale=1.5)

fig= plt.figure(figsize=(12,5)) 
gs= plt.GridSpec(5,2,height_ratios=[2,2,2,1,1],width_ratios=[2,3])
gs.update(hspace=0)
gs.update(wspace=0.5)


ax1=plt.subplot(gs[:,1])
for n,color in zip(np.arange(0,len(avg_freqs),len(Ipfs)),['dimgrey','darkmagenta','olive']):
    n=int(n)
    ax1.errorbar(Ipfs*1000,avg_freqs[n:n+len(Ipfs)],yerr=std_freqs[n:n+len(Ipfs)],label=f'S{int(n/len(avg_freqs)*3)+1}',color=color,
                 capsize=5,fmt='-')    

ax1.legend(loc='upper right')
ax1.set_xlabel('PF current amplitude (pA)')
ax1.set_ylabel('Firing frequency (Hz)')

tr1=simulation(params,PF=(1000,0.1,2),plots=False)
tr2=simulation(params,PF=(1000,0.2,2),plots=False)
tr3=simulation(params,PF=(1000,0.3,2),plots=False)

ax2=plt.subplot(gs[0,0])
ax2.axis('off')
ax3=plt.subplot(gs[1,0],sharex=ax2)
ax3.axis('off')
ax4=plt.subplot(gs[2,0],sharex=ax2)
#ax4.axis('off')
ax5=plt.subplot(gs[3,0],sharex=ax2)


ax2.plot(tr1['t']/ms,tr1['Soma'],color='tomato',alpha=0.7)
ax2.set_xlim((200,800))

ax3.plot(tr2['t']/ms,tr2['Soma'],color='red',alpha=0.7)
ax4.plot(tr3['t']/ms,tr3['Soma'],color='darkred',alpha=0.7)

ax5.plot(tr1['t']/ms,np.where(tr['t']/ms>300,0.1,0),color='tomato',label='100 pA',alpha=0.7)
ax5.plot(tr1['t']/ms,np.where(tr['t']/ms>300,0.2,0),color='red',label='200 pA',alpha=0.7)
ax5.plot(tr1['t']/ms,np.where(tr['t']/ms>300,0.3,0),color='darkred',label='300 pA',alpha=0.7)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.yaxis.set_ticks_position('none')
ax5.tick_params(axis='y', which='both', left=False, labelleft=False)
ax5.spines['bottom'].set_position(('outward', 10))  # You can adjust the outward position if needed
ax5.xaxis.set_ticks_position('bottom')
ax5.xaxis.set_label_position('bottom')
ax5.set_xlabel('ms')
#ax5.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('adscadf.png',dpi=500)
plt.show()