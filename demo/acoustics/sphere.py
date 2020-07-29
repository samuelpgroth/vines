#
# Scattering of a plane wave by a homogeneous sphere
# ==================================================
#
# This demo illustrates how to:
#
# * Compute the scattering of a plane wave by a homogeneous dielectric obstable
# * Solve the volume integral equation using an iterative method
# * Postprocess the solution to evaluate the total field
# * Check the accuracy by comparing to the analytical solution
# * Make a nice plot of the solution in the domain

import os
import sys
# FIXME: figure out how to avoid this sys.path stuff
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import numpy as np
from vines.geometry.geometry import shape
from vines.fields.plane_wave import PlaneWave
from vines.operators.acoustic_operators import volume_potential
from vines.precondition.threeD import circulant_embed_fftw
from vines.operators.acoustic_matvecs import mvp_vec_fftw, mvp_domain, mvp_potential_x_perm
from scipy.sparse.linalg import LinearOperator, gmres
from vines.mie_series_function import mie_function, mie_function_density_contrast
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D 
from plotting import set_axes_equal
import time

'''                         Define parameters                               '''
# We consider a sphere of radius 1mm and refractive index 1.2
# The incident field is a plane wave of wavelength 0.5mm and unit amplitude
# travelling in the positive x-direction
# * Sphere info
geom = 'sphere'
radius = 2.5e-3
refInd = 1.2 + 1j * 0.0
# * Wavelength
lambda_ext = 1.5e-3
# * Plane wave info
Ao = 1
direction = np.array((1, 0, 0))
ko = 2 * np.pi / lambda_ext  # exterior wavenumber

print('Size parameter = ', ko * radius)


# Define the resolution of the voxel mesh - this is given in terms of number
# of voxels per wavelength. 10 voxels per wavelength typically gives a
# reasonable (<5%) accuracy. See demo_convergence.py for an example script in
# which the convergence of the scheme is considered w.r.t. mesh resolution
nPerLam = 10


# Get mesh geometry and interior wavelength
r, idx, res, P, lambda_int = shape(geom, refInd, lambda_ext, radius,
                                   nPerLam, 1)

(L, M, N) = r.shape[0:3]  # number of voxels in x-, y-, z-directions

# Plot geometry

(L, M, N, _) = r.shape  # number of voxels in x-, y-, z-directions
#  Arrays containing the x, y, z coordinates, these are handy for plotting the geometry
xd = r[:, :, :, 0]
yd = r[:, :, :, 1]
zd = r[:, :, :, 2]

matplotlib.rcParams.update({'font.size': 20})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure(figsize=(12, 8))
ax = fig.gca(projection='3d')
# ax.set_aspect('equal')

ax.scatter(xd[idx], yd[idx], zd[idx])

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
set_axes_equal(ax)
fig.savefig('results/sphere_voxels.pdf')
plt.close()


# Get plane wave incident field
Uinc = PlaneWave(Ao, ko, direction, r)

# Voxel permittivities
Mr = np.zeros((L, M, N), dtype=np.complex128)
Mr[idx] = refInd**2 - 1

# Assemble volume potential operator
toep = volume_potential(ko, r)
toep = ko**2 * toep

# Circulant embedding of volume potential operator
circ_op = circulant_embed_fftw(toep, L, M, N)

# Create array that has the incident field values in sphere, and zero outside
xIn = np.zeros((L, M, N), dtype=np.complex128)
xIn[idx] = Uinc[idx]
xInVec = xIn.reshape((L*M*N, 1), order='F')


def mvp(x):
    'Matrix-vector product operator'
    return mvp_vec_fftw(x, circ_op, idx, Mr)


# Linear oper
A = LinearOperator((L*M*N, L*M*N), matvec=mvp)


def residual_vector(rk):
    'Function to store residual vector in iterative solve'
    global resvec
    resvec.append(rk)


# Iterative solve with GMRES (could equally use BiCG-Stab, for example)
start = time.time()
resvec = []
sol, info = gmres(A, xInVec, tol=1e-5, callback=residual_vector)
print("The linear system was solved in {0} iterations".format(len(resvec)))
end = time.time()
print('Solve time = ', end-start, 's')

# Reshape solution
J = sol.reshape(L, M, N, order='F')

# Get the analytical solution for comparison
# P = mie_function(ko * radius, refInd, L)
P = mie_function_density_contrast(ko * radius, refInd, L, 1, 1)

idx_n = np.ones((L, M, N), dtype=bool)

# Utemp = mvp_domain(sol, circ_op, idx_n, Mr).reshape(L, M, N, order='F')
# U = Uinc - Utemp + J
Utemp = mvp_potential_x_perm(sol, circ_op, idx_n, Mr).reshape(L, M, N,
                                                              order='F')
U = Uinc + Utemp
U_centre = U[:, :, np.int(np.round(N/2))]

error = np.linalg.norm(U_centre-np.conj(P)) / np.linalg.norm(P)
print('Error = ', error)


# Create pretty plot of field over central slice of the sphere
matplotlib.rcParams.update({'font.size': 22})
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(12, 9))
ax = fig.gca()
# Domain extremes
xmin, xmax = r[0, 0, 0, 0], r[-1, 0, 0, 0]
ymin, ymax = r[0, 0, 0, 1], r[0, -1, 0, 1]
plt.imshow(np.real(U_centre.T),
           extent=[xmin*1e3, xmax*1e3, ymin*1e3, ymax*1e3],
           cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
circle = plt.Circle((0., 0.), radius*1e3, color='black', fill=False,
                    linestyle=':')
ax.add_artist(circle)
plt.colorbar()
fig.savefig('results/sphere.pdf')
plt.close()
