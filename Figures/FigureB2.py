#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:13:33 2025

@author: jaume
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a figure with a specific size to fill a full page
fig = plt.figure(figsize=(8, 11))  # Approximate size of an A4 page in inches

# Set up GridSpec layout
gs = GridSpec(5, 2)  # 5 rows, 2 columns

compartments=['Soma','M0','S1','S2','S3']


# List of example data to plot - replace with your actual data retrieval method
runs = [(25,100,3),(25,150,3),(30,150,2),(30,200,2),(50,50,3),
        (50,100,3),(100,100,2),(150,50,2),(100,100,4),(150,50,4)]  # Replace 'your_data_function' with actual data generation

# Loop through the data and create each subplot
for n in range(10):
    d,i,dend=runs[n]
    ax = fig.add_subplot(gs[n // 2, n % 2])  # Determine position based on index
    data_traces={}
    data=np.load(f'singlePF/{d}-{i}-{dend}/traces_{d}{i}{dend}.npz')
    data_traces['t']=data['t']
    
    for comp in compartments:
        data_traces[comp]=data[comp] 
        ax.plot(data_traces['t'],data_traces[comp])
    
    if dend==4:
        didx=1
    else:
        didx=dend
        
    ax.set_title(f'{d}ms – {i}pA – S{didx}')
    ax.set_xlabel('ms')
    ax.set_ylabel('mV')
    ax.set_xlim((400,800))
    ax.set_xticks([400,500,600,700,800],[-100,0,100,200,300])
    

# Adjust layout
plt.tight_layout()
plt.savefig('zdsdata.png',dpi=300)
plt.show()
