#
# Linear scattering of a focused field by a homogeneous sphere
# ============================================================
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
from vines.fields.transducers import bowl_transducer, normalise_power
from matplotlib import pyplot as plt
import matplotlib
import time

'''                      Define transducer parameters                       '''
# * operating/fundamental frequency f1
# * radius of curvature / focal length (roc)
# * inner diameter (inner_D)
# * outer diameter (outer_D)
# * total acoustic power (power)
f1 = 1.1e6
roc = 0.0632
inner_D = 0.0
outer_D = 0.064
power = 44
# FIXME: don't need to define focus location but perhaps handy for clarity?
focus = [roc, 0., 0.]
# FIXME: need source pressure as input

'''                    Define scatterer parameters                          '''
# We consider a sphere of radius 1mm and refractive index 1.2
# The centre  of the sphere is at scat_loc (scatterer location)
# * Sphere info
geom = 'sphere'
radius = 2e-3
refInd = 1.2 + 1j * 0.0
scat_loc = [roc, 0., 0.]

'''                        Define medium parameters                         '''
# * speed of sound (c)
# * medium density (\rho)
# * the attenuation power law info (\alpha_0, \eta)
# * nonlinearity parameter (\beta)
c = 1487.0
rho = 998.0
alpha0 = 0.217
eta = 2
beta = 3.5e0


def attenuation(f, alpha0, eta):
    'Attenuation function'
    alpha = alpha0 * (f * 1e-6)**eta
    return alpha


# Compute useful quantities: wavelength (lam), wavenumber (k0),
# angular frequency (omega)
lam = c / f1
k1 = 2 * np.pi * f1 / c + 1j * attenuation(f1, alpha0, eta)
omega = 2 * np.pi * f1

print('Size parameter = ', np.real(k1) * radius)


# Define the resolution of the voxel mesh - this is given in terms of number
# of voxels per wavelength. 5-10 voxels per wavelength typically gives a
# reasonable (<5%) accuracy. See demo_convergence.py for an example script in
# which the convergence of the scheme is considered w.r.t. mesh resolution
nPerLam = 4


# Get mesh geometry and interior wavelength
r, idx, res, P, lambda_int = shape(geom, refInd, lam, radius,
                                   nPerLam, 1)

(L, M, N) = r.shape[0:3]  # number of voxels in x-, y-, z-directions

# Shift the coordinates
r[:, :, :, 0] = r[:, :, :, 0] + scat_loc[0]

points = r.reshape(L*M*N, 3, order='F')

# Voxel permittivities
Mr = np.zeros((L, M, N), dtype=np.complex128)
Mr[idx] = refInd**2 - 1

# Assemble volume potential operator
toep = volume_potential(k1, r)
toep = k1**2 * toep

# Circulant embedding of volume potential operator
circ_op = circulant_embed_fftw(toep, L, M, N)

# Generate incident field
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
Uinc = p.reshape(L, M, N, order='F')

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
sol, info = gmres(A, xInVec, tol=1e-4, callback=residual_vector)
print("The linear system was solved in {0} iterations".format(len(resvec)))
end = time.time()
print('Solve time = ', end-start, 's')

# Reshape solution
J = sol.reshape(L, M, N, order='F')

idx_n = np.ones((L, M, N), dtype=bool)

# Utemp = mvp_domain(sol, circ_op, idx_n, Mr).reshape(L, M, N, order='F')
# U = Uinc - Utemp + J
Utemp = mvp_potential_x_perm(sol, circ_op, idx_n, Mr).reshape(L, M, N, order='F')
# U = Uinc + Utemp
U = Utemp
U_centre = U[:, :, np.int(np.round(N/2))]

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
fig.savefig('results/sphere_focused.pdf')
plt.close()


# Create a bigger grid over which we evaluate the total field
from  IPython import embed; embed()
dx = res
wx = max(r[:, 0, 0, 0]) - min(r[:, 0, 0, 0]) + dx
wy = max(r[0, :, 0, 1]) - min(r[0, :, 0, 1]) + dx
wz = max(r[0, 0, :, 2]) - min(r[0, 0, :, 2]) + dx
# embed()

start = time.time()
from vines.geometry.geometry import generatedomain, grid3d
r_b, L_b, M_b, N_b = generatedomain(dx, wx, wy, wz)
r_b[:, :, :, 0] = r_b[:, :, :, 0] + scat_loc[0]
# Adjust r
r[:, :, :, 0] = r[:, :, :, 0] - r[0, 0, 0, 0] + x_start
end = time.time()
print('Mesh generation time:', end-start)
# embed()
points = r.reshape(L*M*N, 3, order='F')