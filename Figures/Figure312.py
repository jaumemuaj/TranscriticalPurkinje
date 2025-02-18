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

dres=np.linspace(0,5,25)
delta=np.linspace(0,10,25)
mesh1, mesh2 = np.meshgrid(dres, delta)

# Stack and reshape to get all combinations
combinations = np.vstack([mesh1.ravel(), mesh2.ravel()]).T

# Print the combinations
print(len(combinations))

N=10
def mean_cf_pause(params,N):
    pause_list=[]
    pct=0.05
    variation=['Iint','zmax','max_vCas','m','tauR','Ca_half','slp','tauCa','C',
          'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p',]
    for n in range(N):
        run_params=params.copy()
        var=pct*(np.round(np.random.rand(len(variation)),3)*2-1)
        for param,v in zip(variation,var):
            run_params[param]+=v*run_params[param]
            
        tr=simulation(run_params,CF=10,plots=False)
        pause_list.append(np.max(tr['pauses']))
    
    return np.mean(pause_list)

with open('params1606.json', 'r') as file:
        params=json.load(file)

pause_scores=[]
count=0
per=0
for run in combinations:
    if count%(len(combinations)//100)==0:
        print(per,'%',f' {count}/{len(combinations)}')
        per+=1
    dr,de=run
    params['dres_coeff']=dr
    params['Deltaeps_z']=de
    pause_scores.append(mean_cf_pause(params,N))
    count+=1

with open('params1606.json', 'r') as file:
        params=json.load(file)
# Create a meshgrid for plotting
x = dres
y = delta
X, Y = np.meshgrid(x, y)
#%%
# Reshape values to fit the meshgrid
pause_scores=np.array(pause_scores)
Z = pause_scores.reshape(len(y), len(x))

sns.set(style="ticks", context="paper", font_scale=1.5)

fig = plt.figure(figsize=(10, 4))
gs= plt.GridSpec(3,2,height_ratios=[1,1,1],width_ratios=[1.5,1])
gs.update(hspace=0)


# Plot the colormap
ax1=plt.subplot(gs[:,0])

sc=ax1.imshow(Z, extent=(x[0], x[-1], y[0], y[-1]), origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(sc,ax=ax1,label='Pause length (ms)')
ax1.set_xlabel(r'$\overline{d_{res}}$  (nA/ms)')
ax1.set_ylabel(r'$\Delta \epsilon _z$  (a.u.)')

# Highlight the scores within the range 100 to 150
highlight_mask = (pause_scores >= 100) & (pause_scores <= 150)
highlight_x = X.ravel()[highlight_mask]
highlight_y = Y.ravel()[highlight_mask]

# Overlay scatter plot
ax1.scatter(highlight_x, highlight_y, color='red', edgecolors='white', label=r'100ms < Pause < 150ms')

# Add legend
ax1.legend(loc='upper right',fontsize=10)

ax4=plt.subplot(gs[2,1])
params['dres_coeff']=3.1
params['Deltaeps_z']=0
tr=simulation(params,CF=10,plots=False)
ax4.plot(tr['t']/ms,tr['z']/nA)
ax4.plot(tr['t']/ms,tr['dres']/nA)
ax4.plot(tr['t']/ms,(tr['z']+tr['dres'])/nA)
ax4.set_xlim((400,700))
ax4.set_yticks([75],[r'$z_{lim}$'])
ax4.set_xlabel('Time (ms)')

ax2=plt.subplot(gs[0,1],sharex=ax4)
params['dres_coeff']=1
params['Deltaeps_z']=9.5
tr=simulation(params,CF=10,plots=False)

ax2.plot(tr['t']/ms,tr['z']/nA,label='z')
ax2.plot(tr['t']/ms,tr['dres']/nA,label=r'$d_{res}$')
ax2.plot(tr['t']/ms,(tr['z']+tr['dres'])/nA,label=r'$z_{eff}$')
ax2.set_xlim((400,700))
ax2.set_yticks([75],[r'$z_{lim}$'])


ax3=plt.subplot(gs[1,1],sharex=ax4)
params['dres_coeff']=2.5
params['Deltaeps_z']=5
tr=simulation(params,CF=10,plots=False)
ax3.plot(tr['t']/ms,tr['z']/nA)
ax3.plot(tr['t']/ms,tr['dres']/nA)
ax3.plot(tr['t']/ms,(tr['z']+tr['dres'])/nA)
ax3.set_xlim((400,700))
ax3.set_xticks([])
ax3.set_yticks([75],[r'$z_{lim}$'])
ax2.legend(fontsize=10)



plt.tight_layout()
plt.show()