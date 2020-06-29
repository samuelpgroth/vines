# Preconditioning versus no Preconditioning
# We demonstrate the gain achieved by employing a circulant preconditioning 
# strategy when compared to using no preconditioner.

# FIXME: currently a sphere - worth switching to a different shape

# Scattering of a plane wave by a homogeneous sphere
# This BVP permits analytical solution by the method of separation of variables
# FFT-accelerated VIE solver using a Cartesian grid.
# Currently using "DDA" evaluation of all the integrals.
# In this example we demonstrate the use of the circulant preconditioning 
# approach for accelerating the convergence of the iterative solve.

import os
import sys
from IPython import embed
# FIXME: figure out how to avoid this sys.path stuff
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import numpy as np
from vines.geometry.geometry import shape
from vines.fields.plane_wave import PlaneWave
from vines.operators.acoustic_operators import volume_potential
from vines.precondition.threeD import circulant_embed_fftw
from vines.operators.acoustic_matvecs import mvp_vec_fftw, mvp_domain
from scipy.sparse.linalg import LinearOperator, gmres
from vines.precondition.circulant_acoustic import mvp_circ2_acoustic
import matplotlib
from matplotlib import pyplot as plt
import time

geom = 'sphere'

radius = 1e-3
lambda_ext = 5e-4
ko = 2 * np.pi / lambda_ext  # exterior wavenumber
sizeParam = ko * radius  # Size parameter
nPerLam = 10  # number of voxels per interior wavelength
aspectRatio = 1

# Refractive index of scatterer (real and imaginary parts)
refInd = 1.2 + 1j * 0.0

# Polyhedral geometries
r, idx, res, P, lambda_int = shape(geom, refInd, lambda_ext, radius,
                                   nPerLam, aspectRatio)

print('Size parameter ka = ', sizeParam)

(L, M, N, _) = r.shape  # number of voxels in x-, y-, z-directions

# Define incident plane wave of unit amplitude in x-direction
Ao = 1
direction = np.array((1, 0, 0))
Uinc = PlaneWave(Ao, ko, direction, r)

# Voxel permittivities
Mr = np.zeros((L, M, N), dtype=np.complex128)
Mr[idx] = refInd**2 - 1

# Assemble volume potential operator
start = time.time()
toep = volume_potential(ko, r)
end = time.time()
print('Operator assembly time:', end-start)

toep = ko**2 * toep

start = time.time()
# Circulant embedding
circ_op = circulant_embed_fftw(toep, L, M, N)
end = time.time()
print('Time for circulant embedding and FFT:', end-start)


xIn = np.zeros((L, M, N), dtype=np.complex128)
xIn[idx] = Uinc[idx]
xInVec = xIn.reshape((L*M*N, 1), order='F')

mvp = lambda x: mvp_vec_fftw(x, circ_op, idx, Mr)

# Solving the linear system
A = LinearOperator((L*M*N, L*M*N), matvec=mvp)


def residual_vector(rk):
    global resvec
    resvec.append(rk)


start = time.time()
resvec = []
sol, info = gmres(A, xInVec, tol=1e-4, callback=residual_vector)
print("The linear system was solved in {0} iterations".format(len(resvec)))
end = time.time()
resvec0 = resvec
print('Solve time = ', end-start, 's')
print('Relative residual = ',
      np.linalg.norm(mvp(sol)-xInVec)/np.linalg.norm(xInVec))

J = sol.reshape(L, M, N, order='F')


start = time.time()
# 2-level circulant preconditioner for acoustic problem
from vines.precondition.circulant_acoustic import circ_1_level_acoustic, circ_2_level_acoustic
_, circ_L_opToep = circ_1_level_acoustic(toep, L, M, N, 'off')
circ2, circ_M_opToep = circ_2_level_acoustic(circ_L_opToep, L, M, N)
end = time.time()
print('Circulant preconditioner construction (s):', end-start)

# Invert preconditioner in parallel
start = time.time()
def processInput(i):
    inverse_blocks = np.zeros((M, N, N), dtype=np.complex128)
    for j in range(0, M):
            inverse_blocks[j, :, :] = np.linalg.inv(np.identity(N) - (refInd**2 - 1) * circ2[i, j, :, :])
    return inverse_blocks

from joblib import Parallel, delayed  
import multiprocessing
inputs = range(L)

num_cores = multiprocessing.cpu_count()

print("numCores = " + str(num_cores))

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs) 
circ2_inv = np.asarray(results)

end = time.time()
print('Preconditioner inversion (s):', end-start)

mvp_prec = lambda x: mvp_circ2_acoustic(x, circ2_inv, L, M, N, idx)
prec = LinearOperator((L*M*N, L*M*N), matvec=mvp_prec)

start = time.time()
resvec = []
sol1, info1 = gmres(A, xInVec, M=prec, tol=1e-4, restart=500, callback=residual_vector)
end = time.time()
print("The linear system was solved in {0} iterations".format(len(resvec)))
print('Solve time = ', end-start,'s')
print('True residual = ',
      np.linalg.norm(mvp(sol1)-xInVec)/np.linalg.norm(xInVec))

from vines.mie_series_function import mie_function
P = mie_function(sizeParam, refInd, L)

idx_n = np.ones((L, M, N), dtype=bool)
mvp_all = lambda x:mvp_domain(x, circ_op, idx_n, Mr)

temp = mvp_all(sol1)

Utemp = temp.reshape(L, M, N, order='F')
U = Uinc - Utemp + J
U_centre = U[:, :, np.int(np.round(N/2))]

error = np.linalg.norm(U_centre-np.conj(P)) / np.linalg.norm(P)
print('Error = ', error)

# Plot the convergence of iterative solver
matplotlib.rcParams.update({'font.size': 20})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure(figsize=(9, 6))
ax = fig.gca()
plt.semilogy(resvec0/resvec0[0],'-ro')
plt.semilogy(resvec/resvec[0],'-ks')
plt.grid()

# labels
plt.ylabel('Relative residual')
plt.xlabel('\# iterations')
# ax.yaxis.major.formatter._useMathText = True

plt.legend(('No preconditioning', 'Circulant preconditioning'),
           shadow=True, loc=(0.37, 0.7), handlelength=1.5, fontsize=20)

fig.savefig('covergence.png')
plt.close()

# Plot field on central slice
fig = plt.figure(figsize=(12, 8))
ax = fig.gca()
plt.imshow(np.real(U_centre.T),
        extent=[r[0, 0, 0, 0], r[-1, 0, 0, 0], r[0, 0, 0, 1], r[0, -1, 0, 1]],
        cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
plt.xlabel('$x$ (m)')
plt.ylabel('$y$ (m)')
circle = plt.Circle((0., 0.), radius, color='black', fill=False)
ax.add_artist(circle)
plt.colorbar()

fig.savefig('sphere.png')
plt.close()
