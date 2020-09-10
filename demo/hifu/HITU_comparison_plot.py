# Plot of paper
# Comparison of VINES with HITU simulator (https://github.com/jsoneson/HITU_Simulator)

# This code reproduces the plot in the preprint 
# "Fast non-linear ultrasound simulations with an integral equation method of nested grids"

# HITU results, for comparison
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

# file_contents = sio.loadmat('results/matlab/H101_HITU.mat')
file_contents = sio.loadmat('results/matlab/H101_HITU_water_power100.mat')
# file_contents = sio.loadmat('results/matlab/H101_HITU_liver_power100.mat')
# file_contents = sio.loadmat('results/matlab/H131_HITU.mat')
hitu = file_contents['p5']
axis_hitu = file_contents['z_axis']

# filename = 'results/pickles/H101_power100_water_nPerLam6.pickle'
# filename = 'results/pickles/H101_linear_power100_liver_nPerLam6.pickle'
# filename = 'results/pickles/H131_power50_water_test_nPerLam6.pickle'
filename = 'results/pickles/H101_linear_power100_water_nPerLam6.pickle'

with open(filename, 'rb') as f:
    bits = pickle.load(f)

axis_vines = bits[0]
p_axis = bits[1]

# matplotlib.rcParams.update({'font.size': 26})
plt.rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 22})
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
# fig = plt.figure(figsize=(14, 8))
fig = plt.figure(figsize=(9, 7))
ax = fig.gca()
# Plot HITU
plt.plot(axis_hitu[0,:], np.abs(hitu[0,:])/1e6, 'k-', linewidth=1)
plt.plot(axis_hitu[0,:], np.abs(hitu[1,:])/1e6, 'r-', linewidth=1,
         label='_nolegend_')
plt.plot(axis_hitu[0,:], np.abs(hitu[2,:])/1e6, 'b-', linewidth=1,
         label='_nolegend_')
plt.plot(axis_hitu[0,:], np.abs(hitu[3,:])/1e6, 'g-', linewidth=1,
         label='_nolegend_')
plt.plot(axis_hitu[0,:], np.abs(hitu[4,:])/1e6, 'm-', linewidth=1,
         label='_nolegend_')
# # Plot volume potential
plt.plot(100*(axis_vines[0]), np.abs(p_axis[0])/1e6,'k--', linewidth=1)
plt.plot(100*(axis_vines[1]), np.abs(p_axis[1])/1e6,'r--', linewidth=1,
         label='_nolegend_')
plt.plot(100*(axis_vines[2]), np.abs(p_axis[2])/1e6,'b--', linewidth=1,
         label='_nolegend_')
plt.plot(100*(axis_vines[3]), np.abs(p_axis[3])/1e6,'g--', linewidth=1,
         label='_nolegend_')
plt.plot(100*(axis_vines[4]), np.abs(p_axis[4])/1e6,'m--', linewidth=1,
         label='_nolegend_')

plt.grid(True)
plt.xlim([axis_hitu[0,0], axis_hitu[0,-1]])
plt.ylim([0, 12])
# plt.yticks(np.arange(1, 10, step=1), ['2','','4','','6','','8',''])
# plt.yticks(np.arange(1, 6, step=1), ['1','','3','','5'])
plt.xlabel(r'Axial distance (cm)')
plt.ylabel(r'Pressure (MPa)')

# legend
plt.legend(('HITU simulator', 'Volume potential'),
           shadow=False, loc=(0.03, 0.8), handlelength=1.5,
           fontsize=20)

fig.savefig('results/H101_HITU_comparison_water_test_power100.pdf')
plt.close()