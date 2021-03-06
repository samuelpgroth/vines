#
# Nonlinear field generated in by a bowl-shaped HIFU transducer
# ==========================================================
#
# This demo illustrates how to:
#
# * Compute the nonlinear time-harmonic field in a homogeneous medium
# * Use incident field routines to generate the field from a HIFU transducer
# * Make a nice plot of the solution in the domain
#
#
# We consider the field generated by the Sonic Concepts H101 transducer:
# https://sonicconcepts.com/transducer-selection-guide/
# This transducer operates at 1.1 MHz, has a 63.2 mm radius of curvature and a 
# diameter of 64 mm. It has no central aperture.
# The medium of propagation we consider is water.

import os
import sys
# FIXME: figure out how to avoid this sys.path stuff
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import numpy as np
from vines.geometry.geometry import shape
from vines.fields.plane_wave import PlaneWave
from vines.operators.acoustic_operators import volume_potential, volume_potential_cylindrical
from vines.precondition.threeD import circulant_embed_fftw
from vines.operators.acoustic_matvecs import mvp_volume_potential, mvp_vec_fftw
from scipy.sparse.linalg import LinearOperator, gmres
from vines.mie_series_function import mie_function
from matplotlib import pyplot as plt
from vines.geometry.geometry import generatedomain2d, generatedomain
from vines.fields.transducers import bowl_transducer, normalise_power
import time
import matplotlib
from matplotlib import pyplot as plt
import itertools
import scipy.io as sio

file_contents = sio.loadmat('results/matlab/H101_HITU_liver_power100.mat')
# file_contents = sio.loadmat('results/matlab/H131_HITU.mat')
# file_contents = sio.loadmat('results/matlab/H151_HITU_water_power100.mat')
# file_contents = sio.loadmat('results/matlab/H151_HITU_liver_power100.mat')
hitu = file_contents['p5']
axis_hitu = file_contents['z_axis']
'''                        Define medium parameters                         '''
# * speed of sound (c)
# * medium density (\rho)
# * the attenuation power law info (\alpha_0, \eta)
# * nonlinearity parameter (\beta)
# material = 'water'
# c = 1487.0
# rho = 998.0
# alpha0 = 0.217
# eta = 2
# beta = 3.5e0

material = 'liver'
c = 1590.0
rho = 1060
alpha0 = 90.0
eta = 1.1
beta = 4.4


def attenuation(f, alpha0, eta):
    'Attenuation function'
    alpha = alpha0 * (f * 1e-6)**eta
    alpha = alpha / 8.686
    return alpha


'''                      Define transducer parameters                       '''
# * operating/fundamental frequency f1
# * radius of curvature, focal length (roc)
# * inner diameter (inner_D)
# * outer diameter (outer_D)
# * total acoustic power (power)
# f1 = 1.1e6
# roc = 0.0632
# inner_D = 0.0
# outer_D = 0.064
# power = 50

# f1 = 1.1e6
# transducername = 'H131'
# roc = 0.035
# inner_D = 0.0
# outer_D = 0.033
# power = 100

f1 = 1.1e6
transducername = 'H101'
roc = 0.0632
inner_D = 0.0
outer_D = 0.064
power = 100

# f1 = 1.1e6
# transducername = 'H151'
# roc = 0.1
# inner_D = 0.0
# outer_D = 0.064
# power = 100


# FIXME: don't need to define focus location but perhaps handy for clarity?
focus = [roc, 0., 0.]
# FIXME: need source pressure as input

# How many harmonics to compute
# (mesh resolution should be adjusted accordingly, I recommend setting
# nPerLam  >= 3 * n_harm, depending on desired speed and/or accuracy)
n_harm = 2

# Mesh resolution (number of voxels per fundamental wavelength)
nPerLam = 10

# Compute useful quantities: wavelength (lam), wavenumber (k0),
# angular frequency (omega)
lam = c / f1
k1 = 2 * np.pi * f1 / c + 1j * attenuation(f1, alpha0, eta)
omega = 2 * np.pi * f1

# Create voxel mesh
dx = lam / nPerLam

# Dimension of computation domain
# x_start needs to be close to the transducer
# x_end can be just beyond the focus
# the width in the y,z directions should be around the width of outer_D,
# but you can shrink this to speed up computations if required
# x_start = 0.001
x_start_0 = roc - 0.99 * np.sqrt(roc**2 - (outer_D/2)**2)
x_start = 0
x_end = roc + 0.01
wx = x_end - x_start
wy = outer_D*1
wz = wy/1000
# wz = wy

# start = time.time()
# # r, L, M = generatedomain2d(dx, wx, wy)
# r, L, M, N = generatedomain(dx, wx, wy, wz)

# # Adjust r by shifting x locations
# r[:, :, :, 0] = r[:, :, :, 0] - r[0, 0, 0, 0] + x_start
# # r[:, :, :, 1] = r[:, :, :, 1] - r[0, 0, 0, 1]
# end = time.time()
# print('Mesh generation time:', end-start)
# points = r.reshape(L*M*N, 3, order='F')

n_line = 10000
x_line = np.linspace(x_start_0, x_end, n_line)
points = np.vstack((x_line, np.zeros(n_line), np.zeros(n_line))).T



start = time.time()
n_elements = 2**12
x, y, z, p = bowl_transducer(k1, roc, focus, outer_D / 2, n_elements,
                             inner_D / 2, points.T, 'x')
end = time.time()
print('Incident field evaluation time (s):', end-start)
dist_from_focus = np.sqrt((points[:, 0]-focus[0])**2 + points[:, 1]**2 +
                           points[:,2]**2)
idx_near = np.abs(dist_from_focus - roc) < 5e-4
p[idx_near] = 0.0

# Normalise incident field to achieve desired total acoutic power
p0 = normalise_power(power, rho, c, outer_D/2, k1, roc,
                     focus, n_elements, inner_D/2)

p *= p0

# from IPython import embed; embed()
# Create a pretty plot of the first harmonic in the domain
matplotlib.rcParams.update({'font.size': 22})
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(9, 7))
ax = fig.gca()
plt.plot(x_line*100, np.abs(p)/1e6)
plt.plot(axis_hitu[0,:], np.abs(hitu[0,:]/1e6))
# plt.xlabel(r'$x$ (cm)')
# plt.ylabel(r'$y/z$ (cm)')

fig.savefig('results/H101_line.png')
plt.close()