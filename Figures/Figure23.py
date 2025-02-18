
from TCplusplus import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


with open('params1606.json', 'r') as file:
        params=json.load(file)
params['Iint']=90.17e-9
tr=simulation(params,CF=10,plots=False)
print(np.max(tr['pauses']))

cftr=np.load('/Users/jaume/Desktop/TC+/thesis_code/singleCF/cfdelay0-0-0-0/traces.npz')
cf=np.load('/Users/jaume/Desktop/TC+/thesis_code/singleCF/cfdelay0-0-0-0/spike_times.npz')
spike_times=cf['arr_0']
fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(3, 1, height_ratios=[3,3,2])  # Added a small height for the ghost axis

ax1=plt.subplot(gs[0])
ax1.plot(cftr['t'],cftr['Soma'])
ax1.set_xlim((300,800))
ax1.axis('off')
ax1.label_outer()  # Only show outer labels to avoid overlap

ax2=plt.subplot(gs[1])
ax2.plot(tr['t']/ms,tr['Soma']/mV, color='tomato')
ax2.set_xlim((300,800))
ax2.label_outer()  # Only show outer labels to avoid overlap
ax2.axis('off')


ax3=plt.subplot(gs[2])
_,_,tfreq,freq=smooth_freq([spike_times],bin_size=10, start_time=300, end_time=800, sigma=30,plot=False)
ax3.plot(tfreq,freq[0],label='ZdS')
_,_,tfreq,freq=smooth_freq([tr['spikes']],bin_size=10, start_time=300, end_time=800, sigma=30,plot=False)
ax3.plot(tfreq,freq[0],label='TC+',color='tomato')
ax3.legend(loc='upper right')

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.yaxis.set_ticks_position('none')
ax3.tick_params(axis='y', which='both', left=False, labelleft=False)
ax3.spines['bottom'].set_position(('outward', 10))  # You can adjust the outward position if needed
ax3.xaxis.set_ticks_position('bottom')
ax3.xaxis.set_label_position('bottom')
ax3.set_xlabel('ms')
ax3.set_xticks([300,400,500,600,700,800],[-200,-100,0,100,200,300])
plt.tight_layout()
