

from TCplusplus import *
import seaborn as sns
import numpy as np

sns.set(style="ticks", context="paper", font_scale=1.5)

with open('params1606.json', 'r') as file:
    params=json.load(file)


# params['Iint']=93*nA
# params['Deltaeps_z']=4
# params['dres_coeff']=3

params['Iint']=70*nA #value low enough where no spontaneous spikes occur
params['max_vCad']=0 #deactivate calcium mechanisms
tr=simulation(params,PF=(50,0.2,1),plots=False)

#np.savez('trace_5phen.npz', **tr)


tr['t']=tr['t']/mV


fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(12, 6))  # Adjust figsize as needed

ax1.plot(tr['t'],tr['Soma'], label='Soma')
ax1.plot(tr['t'],tr['M0'],color='darkorange', label='M0')
ax1.plot(tr['t'],tr['S1'],color='dimgrey', label='S1')
ax1.plot(tr['t'],tr['S2'],color='darkmagenta', label='S2')
ax1.plot(tr['t'],tr['S3'],color='olive', label='S3')
#ax1.set_xlim((400,700))
ax1.axis('off')
ax1.legend(loc='upper right')

with open('params1606.json', 'r') as file:
    params=json.load(file)
params['Iint']=70*nA
tr=simulation(params,PF=(50,0.2,1),plots=False)

tr['t']=tr['t']/mV


ax2.plot(tr['t'],tr['Soma'])
ax2.plot(tr['t'],tr['M0'],color='darkorange')
ax2.plot(tr['t'],tr['S1'],color='dimgrey')
ax2.plot(tr['t'],tr['S2'],color='darkmagenta')
ax2.plot(tr['t'],tr['S3'],color='olive')

for ax in (ax1,ax2,ax3):
    ax.set_xlim((400,700))
    ax.axis('off')
plt.tight_layout()
#plt.savefig('PFSS502001.png',dpi=500)
plt.show()