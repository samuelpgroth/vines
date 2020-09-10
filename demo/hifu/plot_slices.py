# Plot slices of right-hand side integrands and also the harmonics themselves

import numpy as np
from IPython import embed
import pickle
import matplotlib
import matplotlib.pyplot as plt
import itertools

# Name of transducer
# transducer_name = 'H131'
# power = 100
# material = 'water'
# nPerLam = 20

transducer_name = 'H101'
power = 100
material = 'liver'
nPerLam = 10



filename = 'results/pickles/slice_' + transducer_name + '_power' + str(power) + \
            '_' + material + '_nPerLam' + str(nPerLam) + '.pickle'

with open(filename, 'rb') as f:
    bits = pickle.load(f)

x_line = bits[0]
y_line = bits[1]
slices = bits[2]

xmin = min(x_line)*100
xmax = max(x_line)*100
ymin = min(y_line)*100
ymax = max(y_line)*100

rel_p1 = slices[0]**2 /  np.max(np.abs(slices[0]**2))
rel_p2 = slices[0]*slices[1] /  np.max(np.abs(slices[0]*slices[1]))
temp = slices[1]**2 + 2 * slices[0]*slices[2]
rel_p3 = temp /  np.max(np.abs(temp))
temp = slices[0]*slices[3] + slices[1]*slices[2]
rel_p4 = temp /  np.max(np.abs(temp))

# embed()

matplotlib.rcParams.update({'font.size': 10})
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# fig = plt.figure(figsize=(9, 7))
# ax = fig.gca()
# # plt.imshow(np.log10(np.abs(P[:, :, np.int(np.floor(N/2))].T / 1e6)),
# #            extent=[xmin, xmax, ymin, ymax],
# #            cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
# plt.imshow(np.log10(np.abs(rel_p1)).T,
#            extent=[xmin, xmax, ymin, ymax],
#            cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
# plt.xlabel(r'$x$ (cm)')
# plt.ylabel(r'$y/z$ (cm)')

fig, axs = plt.subplots(2, 2)
CS = axs[0, 0].contourf(np.log10(np.abs(rel_p1)).T,
             extent=[xmin, xmax, ymin, ymax],
             levels=[-4, -3.5, -2.5, -2.0, -1.5, -1, -0.5, 0],
             cmap=plt.cm.viridis,
             extend='both')
axs[0, 0].get_xaxis().set_visible(False)
axs[0, 0].set_title('$f=p_1^2$')
# axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].contourf(np.log10(np.abs(rel_p2)).T,
             extent=[xmin, xmax, ymin, ymax],
             levels=[-4, -3.5, -2.5, -2.0, -1.5, -1, -0.5, 0],
             cmap=plt.cm.viridis,
             extend='both')
axs[0, 1].get_xaxis().set_visible(False)
axs[0, 1].get_yaxis().set_visible(False)
axs[0, 1].set_title('$f=p_1p_2$')
axs[1, 0].contourf(np.log10(np.abs(rel_p3)).T,
             extent=[xmin, xmax, ymin, ymax],
             levels=[-4, -3.5, -2.5, -2.0, -1.5, -1, -0.5, 0],
             cmap=plt.cm.viridis,
             extend='both')
axs[1, 0].set_title('$f=p_2^2 + 2p_1p_3$')
axs[1, 1].contourf(np.log10(np.abs(rel_p4)).T,
             extent=[xmin, xmax, ymin, ymax],
             levels=[-4, -3.5, -2.5, -2.0, -1.5, -1, -0.5, 0],
             cmap=plt.cm.viridis,
             extend='both')
axs[1, 1].get_yaxis().set_visible(False)
axs[1, 1].set_title('$f=p_1p_4+p_2p_3$')

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

# # Plot figure
# fig = plt.figure(figsize=(9, 7))
# ax = fig.gca()
# CS = plt.contourf(np.log10(np.abs(rel_p1)).T,
#              extent=[xmin, xmax, ymin, ymax],
#              levels=[-4, -3.5, -2.5, -2.0, -1.5, -1, -0.5, 0],
#              cmap=plt.cm.viridis,
#              extend='both')

fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(CS, cax=cbar_ax)

# cbar = plt.colorbar(CS)
CS.cmap.set_under('black')
CS.changed()

cbar.ax.set_ylabel('log$_{10}(|f|$/max$|f|)$')

# labels
# axs[0, 0].ylabel('$y/z$ (cm)')
axs[0, 0].set_ylabel('$y/z$ (cm)')
axs[1, 0].set_ylabel('$y/z$ (cm)')
axs[1, 0].set_xlabel('$x$ (cm)')
axs[1, 1].set_xlabel('$x$ (cm)')
# plt.xlabel('$z$ (cm)')
filename = 'results/relative_magnitudes0_' + transducer_name + '_power' + str(power) + '_material_' + \
        material + '.png'
fig.savefig(filename, dpi=300)
plt.close()


# Plot harmonics themselves
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(np.abs(slices[1]).T,
             extent=[xmin, xmax, ymin, ymax],
           cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
axs[0, 0].get_xaxis().set_visible(False)
axs[0, 1].imshow(np.abs(slices[2]).T,
             extent=[xmin, xmax, ymin, ymax],
           cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
axs[0, 1].get_xaxis().set_visible(False)
axs[0, 1].get_yaxis().set_visible(False)
axs[1, 0].imshow(np.abs(slices[3]).T,
             extent=[xmin, xmax, ymin, ymax],
           cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
# axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].imshow(np.abs(slices[4]).T,
             extent=[xmin, xmax, ymin, ymax],
           cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
axs[1, 1].get_yaxis().set_visible(False)
# axs[1, 1].set_title('Axis [1,1]')

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

# # Plot figure
# fig = plt.figure(figsize=(9, 7))
# ax = fig.gca()
# CS = plt.contourf(np.log10(np.abs(rel_p1)).T,
#              extent=[xmin, xmax, ymin, ymax],
#              levels=[-4, -3.5, -2.5, -2.0, -1.5, -1, -0.5, 0],
#              cmap=plt.cm.viridis,
#              extend='both')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(CS, cax=cbar_ax)

# # cbar = plt.colorbar(CS)
# CS.cmap.set_under('black')
# CS.changed()

# cbar.ax.set_ylabel('log$_{10}(|p_1^2|$/max$|p_1^2|)$')

# labels
# axs[0, 0].ylabel('$y/z$ (cm)')
axs[0, 0].set_ylabel('$y/z$ (cm)')
axs[1, 0].set_ylabel('$y/z$ (cm)')
axs[1, 0].set_xlabel('$x$ (cm)')
axs[1, 1].set_xlabel('$x$ (cm)')
# plt.xlabel('$z$ (cm)')
fig.savefig('results/blah1.png', dpi=300)
plt.close()

