from TCplusplus import *


import json
from cmaes import CMA
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import itertools
import time
from functools import partial
import typing
import jax
import jax.numpy as jnp
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

import seaborn as sns

tr=np.load('trace_5phen.npz')


Ca=tr['Ca']/1e-9
Ca_half=50
slp=5e-4
max_alpha=0.1
n=0.2

exp=np.exp
alpha   = (Ca<Ca_half)* ( slp*(Ca_half-Ca) \
                                  + max_alpha / (1+exp(-n*(Ca-Ca_half))) ) \
    + (Ca>Ca_half)* ( max_alpha / (1+exp(-n*(Ca-Ca_half))) )

ms=1e-3; mV=1e-3; nA=1e-9

sns.set(style="ticks", context="paper", font_scale=1.5)

fig= plt.figure(figsize=(10,12)) 
gs= plt.GridSpec(7,1,height_ratios=[0.5,1.5,1,1,1,1,0.5])
gs.update(hspace=0.1)

time=tr['t']/ms - 100
ax5=fig.add_subplot(gs[6])
#ax5.set_xticks([0,200,400,600,800,1000])


ax0=fig.add_subplot(gs[0])
ax1=fig.add_subplot(gs[1],sharex=ax5)
ax1.plot(time,tr['Soma']/mV, label='Soma')
ax1.plot(time,tr['M0']/mV,color='darkorange',label='M0')
ax1.plot(time,tr['S1']/mV,color='dimgrey',label='S1')
ax1.plot(time,tr['S2']/mV,color='darkmagenta',label='S2')
ax1.plot(time,tr['S3']/mV,color='olive',label='S3')
ax1.legend(loc='upper right',fontsize=10)
ax1.set_ylabel('mV')


ax1.set_yticks([-60,-35,-10])

PF=np.zeros(len(tr['t']))
ax0.axis('off')

PF[int(400/0.02):int(450/0.02)]=0.2;PF[int(550/0.02):int(600/0.02)]=0.2;PF[int(700/0.02):int(750/0.02)]=0.2
CF=np.zeros(len(time))
CF[int(855/0.02):int(860/0.02)]=1
inh=np.zeros(len(time))
inh[int(250/0.02):int(300/0.02)]=-0.25

ax2=fig.add_subplot(gs[5],sharex=ax5)
ax5.plot(time,CF,linewidth=2, color='darkred',label='CF stimulus')
ax5.plot(time,inh,linewidth=2,color='orange',label='Inhibitory stimulus')
ax5.plot(time,PF,color='tomato',linewidth=2,label='PF stimulus')
ax5.legend(loc='right',fontsize=10)
ax5.set_yticks([-0.25,0.2,1],[-0.2,0.2,10]) 
ax5.set_ylabel('nA')


ax3=fig.add_subplot(gs[2],sharex=ax5)
ax3.axhline(y=50,linestyle='dotted',color='dimgrey')
ax3.plot(time,tr['Ca']/nA,label=r'$[Ca^{2+}]$',color='dodgerblue')
ax3.legend(loc='right',fontsize=10)
ax3.set_yticks([20,50,80],[20,r'50',80])
ax3.set_ylabel('nM')


ax6=fig.add_subplot(gs[3],sharex=ax5)
ax6.plot(time,tr['Ca']*alpha/nA, label=r'$\alpha\,[Ca^{2+}]$',color='steelblue')
ax6.legend(loc='right',fontsize=10)
ax6.set_yticks([0,4,8])
ax6.set_ylabel('nA')

ax4=fig.add_subplot(gs[4],sharex=ax5)
ax4.plot(time,tr['act'],label=r'$g_{KCa}$',color='grey')
ax4.legend(loc='right',fontsize=10)
ax4.set_yticks([0,0.5,1])
ax4.set_ylabel('a.u.')

#ax5=fig.add_subplot(gs[6])
ax2.plot(time,tr['z']/nA,label=r'$z$')
ax2.plot(time,tr['dres']/nA,label=r'$d_{res}$')
ax2.plot(time,(tr['dres']+tr['z'])/nA,label=r'$z$ + $d_{res}$')
ax2.legend(loc='right',fontsize=10)
ax2.set_yticks([0,50,100])
ax2.set_ylabel('nA')

for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    #ax.tick_params(axis='x', labelsize=10)  # Adjusts only x-axis tick labels
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))  # You can adjust the outward position if needed

    #ax.set_xticks([])
    if ax!=ax5:
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_ticklabels([])  
        ax.xaxis.set_visible(False)  


ax5.spines['bottom'].set_visible(True)
ax5.spines['bottom'].set_position(('outward', 10))  # You can adjust the outward position if needed
ax5.xaxis.set_ticks_position('bottom')
ax5.xaxis.set_label_position('bottom')
ax5.set_xlabel('ms')
ax5.label_outer()
ax5.set_xticks([0,200,400,600,800,1000],[0,200,400,600,800,1000])


#ax1.axis('off')
#ax2.axis('off')
#ax3.axis('off')
#ax4.axis('off')
#ax6.axis('off')
#ax5.axis('off')
ax5.set_xlim((0,1100))

plt.tight_layout()
plt.show()

