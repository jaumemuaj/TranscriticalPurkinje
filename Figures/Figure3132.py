
from TCplusplus import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm



with open('params1606.json', 'r') as file:
    params=json.load(file)
    


def open_trace_data(dur_stim,iamp,dend,CF=False,isi=0):
    """
    Parameters
    ----------
    dur_stim, iamp, dend are arrays but so far (13/06) only with 1 element
    CF : TYPE, optional
        DESCRIPTION. The default is False.
    isi  is a number

    """
    compartments=['Soma','M0','S1','S2','S3']
    data_traces={}
    if CF==False:
        file_prefix = f'singlePF/{dur_stim[0]}-{int(1000 * iamp[0])}-{dend[0]}'
        with np.load(f'{file_prefix}/traces_{dur_stim[0]}{int(1000*iamp[0])}{dend[0]}.npz') as data:
            data_traces['t']=data['t']
            for comp in compartments:
                data_traces[comp]=data[comp]
        with np.load(f'{file_prefix}/spikes_{dur_stim[0]}{int(1000*iamp[0])}{dend[0]}.npz') as spikes:
            data_traces['spikes']=spikes['arr_0']
    elif CF==True:
        with np.load(f'singleCF/cfdelay{isi}-0-0-0/traces.npz') as data:
            data_traces['t']=data['t']
            for comp in compartments:
                data_traces[comp]=data[comp]
            data_traces['S4']=data['S4']
        with np.load(f'singleCF/cfdelay{isi}-0-0-0/spike_times.npz') as spikes:
            data_traces['spikes']=spikes['arr_0']
    return data_traces


#steady state – tonic firing
zds_ss=open_trace_data([0],[0],[0])
tcplus_ss=simulation(params,plots=False)

#Inhibited state 
zds_inh=np.load('inhrun/400--250-2/traces_400-2502.npz')
tcplus_inh=np.load('tc_inh.npz')


#PF input – no pause
zds_pfnp=open_trace_data([100],[0.1],[2])
tcplus_pfnp=simulation(params,PF=(100,0.1,2),plots=False)

#PF input – pause
zds_pfp=open_trace_data([100],[0.1],[3])
tcplus_pfp=simulation(params,PF=(100,0.1,3),plots=False)

#CF input
zds_cf=open_trace_data([0],[0],[0],CF=True)
tcplus_cf=simulation(params,CF=10,plots=False)


def Iline(dur,iamp,start):
    dt=0.02
    I=np.zeros(50001)
    I[int(start/dt):int((start+dur)/dt)]=iamp
    return I

x=zds_inh['t']
# Create a figure
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(15, 1, height_ratios=[1,0.25, 0.6, 1,0.25, 0.6, 1,0.25, 0.6, 1,0.25, 0.6, 1,0.25, 0.5])
gs.update(hspace=0.2)
# Plot 1
ax1 = plt.subplot(gs[0])
zdsss = np.zeros(50001)
zdsss[9999:9999+len(zds_ss['Soma'])] = zds_ss['Soma']
ax1.plot(x, zdsss)
#ax1.set_title('Intrinsic firing')
ax1.set_xlim((400, 900))
ax1.axis('off')
ax11=plt.subplot(gs[1],sharex=ax1)
ax11.plot(x,Iline(0,0,100),color='tomato')
ax11.axis('off')

# Ghost axis
ghost_ax1 = plt.subplot(gs[2])
ghost_ax1.axis('off')  # Hide the ghost axis

# Plot 2
ax2 = plt.subplot(gs[3])
ax2.plot(x, zds_inh['Soma'])
#ax2.set_title('Inhibition')
ax2.set_xlim((400, 900))
ax22 = plt.subplot(gs[4],sharex=ax1)
ax22.plot(x,Iline(300,-1,500),color='tomato')

# Ghost axis
ghost_ax2 = plt.subplot(gs[5])
ghost_ax2.axis('off')  # Hide the ghost axis

# Plot 3
ax3 = plt.subplot(gs[6])
ax3.plot(x, zds_pfnp['Soma'])
#ax3.set_title('PF input – no dendritic spike')
ax3.set_xlim((400, 900))
ax33=plt.subplot(gs[7],sharex=ax1)
ax33.plot(x,Iline(100,1,500),color='tomato')

# Ghost axis
ghost_ax3 = plt.subplot(gs[8])
ghost_ax3.axis('off')  # Hide the ghost axis

# Plot 4
ax4 = plt.subplot(gs[9])
ax4.plot(zds_inh['t'], zds_pfp['Soma'])
#ax4.set_title('PF input – dendritic spike')
ax4.set_xlim((400, 900))
ax44=plt.subplot(gs[10],sharex=ax1)
ax44.plot(x,Iline(100,1,500),color='tomato')

# Ghost axis
ghost_ax4 = plt.subplot(gs[11])
ghost_ax4.axis('off')  # Hide the ghost axis

# Plot 5
ax5 = plt.subplot(gs[12])
ax5.plot(zds_inh['t'], zds_cf['Soma'])
#ax5.set_title('CF input')
ax5.set_xlim((400, 900))
ax55=plt.subplot(gs[13],sharex=ax1)
ax55.plot(x,Iline(5,5,500),color='tomato')


# Ghost axis
ghost_ax5 = plt.subplot(gs[14])
ghost_ax5.axis('off')  # Hide the ghost axis

ax2.axis('off')
ax22.axis('off')
ax3.axis('off')
ax33.axis('off')
ax4.axis('off')
ax44.axis('off')
ax5.axis('off')
#ax55.axis('off')

#plt.savefig('zds_phenomenology2.png',dpi=500)

plt.tight_layout()
plt.show()


def Iline(dur,iamp,start):
    dt=0.02
    I=np.zeros(50000)
    I[int(start/dt):int((start+dur)/dt)]=iamp
    return I

x=tcplus_inh['t']/ms
# Create a figure
fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(15, 1, height_ratios=[1,0.25, 0.6, 1,0.25, 0.6, 1,0.25, 0.6, 1,0.25, 0.6, 1,0.25, 0.5])
gs.update(hspace=0.2)



# Plot 1
ax1 = plt.subplot(gs[0])
#zdsss = np.zeros(50001)
#zdsss[9999:9999+len(zds_ss['Soma'])] = zds_ss['Soma']
ax1.plot(x, tcplus_ss['Soma']/mV)
#ax1.set_title('Intrinsic firing')
ax1.set_xlim((400, 900))
ax1.axis('off')
ax11=plt.subplot(gs[1],sharex=ax1)
ax11.plot(x,Iline(0,0,100),color='tomato')
ax11.axis('off')

# Ghost axis
ghost_ax1 = plt.subplot(gs[2])
ghost_ax1.axis('off')  # Hide the ghost axis

# Plot 2
ax2 = plt.subplot(gs[3])
ax2.plot(x, tcplus_inh['Soma'])
#ax2.set_title('Inhibition')
ax2.set_xlim((400, 900))
ax22 = plt.subplot(gs[4],sharex=ax1)
ax22.plot(x,Iline(300,-1,500),color='tomato')

# Ghost axis
ghost_ax2 = plt.subplot(gs[5])
ghost_ax2.axis('off')  # Hide the ghost axis

# Plot 3
ax3 = plt.subplot(gs[6])
ax3.plot(x, tcplus_pfnp['Soma'])
#ax3.set_title('PF input – no dendritic spike')
ax3.set_xlim((400, 900))
ax33=plt.subplot(gs[7],sharex=ax1)
ax33.plot(x,Iline(100,1,500),color='tomato')

# Ghost axis
ghost_ax3 = plt.subplot(gs[8])
ghost_ax3.axis('off')  # Hide the ghost axis

# Plot 4
ax4 = plt.subplot(gs[9])
ax4.plot(x, tcplus_pfp['Soma'])
#ax4.set_title('PF input – dendritic spike')
ax4.set_xlim((400, 900))
ax44=plt.subplot(gs[10],sharex=ax1)
ax44.plot(x,Iline(100,1,500),color='tomato')

# Ghost axis
ghost_ax4 = plt.subplot(gs[11])
ghost_ax4.axis('off')  # Hide the ghost axis

# Plot 5
ax5 = plt.subplot(gs[12])
ax5.plot(x, tcplus_cf['Soma'])
#ax5.set_title('CF input')
ax5.set_xlim((400, 900))
ax55=plt.subplot(gs[13],sharex=ax1)
ax55.plot(x,Iline(5,5,500),color='tomato')


# Ghost axis
ghost_ax5 = plt.subplot(gs[14])
ghost_ax5.axis('off')  # Hide the ghost axis

ax2.axis('off')
ax22.axis('off')
ax3.axis('off')
ax33.axis('off')
ax4.axis('off')
ax44.axis('off')
ax5.axis('off')
ax55.axis('off')

plt.savefig('tcplus_phenomenology.png',dpi=500)

plt.tight_layout()
plt.show()