# Scattering of a plane wave by a homogeneous square
# FFT-accelerated VIE solver using a Cartesian grid.
# Currently using "DDA" evaluation of all the integrals.

import numpy as np
from scipy.special import hankel1
from scipy.sparse.linalg import LinearOperator, gmres
from analytical import penetrable_circle
from scipy.linalg import toeplitz
import time

ko = 80  # Wavenumber
domain_width = 1.5  # radius of domain (half width)
square_side = 1
lam = 2*np.pi/ko
refInd = 1.2
n_per_lam = 10  # Pixels per wavelength

h_temp = lam / n_per_lam  # temp pixel dimension

wx = domain_width
wy = domain_width

# How many points in x and y directions
M = np.int(np.ceil(wx / h_temp))
N = np.int(np.ceil(wy / h_temp))

dx = wx/M
dy = wy/N

A = dx * dy      # pixel area
a = np.sqrt(A / np.pi)  # radius of equivalent-area circle

# Get coordinates of points on grid
# FIX ME: I thought this complex number stuff was elegant at first, but it's
# actually just annoying. Switch to using meshgrid
x = np.zeros((M*N, 1), dtype=np.complex128)
counter = 0
for j in range(N):
    for i in range(M):
        x[counter] = -wx/2 + dx/2+dx*i \
            - 1j*wy/2 + 1j * (dy/2+dy*j)
        counter = counter + 1

x_coord = (np.arange(M)+1) * dx - dx/2 - wx/2
y_coord = (np.arange(N)+1) * dy - dy/2 - wy/2

perm = np.ones(M*N)  # permittivities

# Find the indices of the pixels lying inside the square
IDX = (np.abs(np.real(x)) <= square_side/2) * \
      (np.abs(np.imag(x)) <= square_side/2)
idx = np.where(IDX)  # locate indices of points inside circle

perm[idx[0]] = refInd**2  # assume permittivity of scatterer is 2 for now
Mr = perm - 1

# Incident plane wave
dInc = np.array([1, 0])
eInc = np.zeros((M * N, 1), dtype=np.complex128)
eInc[idx[0]] = np.exp(1j * ko * (np.real(x[idx[0]]) * dInc[0] +
                      np.imag(x[idx[0]] * dInc[1])))

MR = Mr.reshape(M, N, order='F')

# Fundamental solution of the Helmholtz equation
g = lambda x, y: A * 1j/4 * hankel1(0, ko * np.abs(x - y))

# Self term
self = a**2 * 1j * np.pi/2 * ((1 + 1j * np.euler_gamma) / 2
                              - 1j / np.pi + 1j / np.pi * np.log(ko * a / 2))

# Sparse matvec
def potential(x):
    toep = np.zeros((M, N), dtype=np.complex128)
    for i in range(M):
        for j in range(N):
            if i == 0 and j == 0:
                toep[i, j] = self
            else:
                toep[i, j] = g(x[0], x[j * M + i])
    return toep


toep = ko**2 * potential(x)

circ = np.zeros((2 * M, 2 * N), dtype=np.complex128)

# Circulant embedding
circ[0:M, 0:N] = toep[0:M, 0:N]
circ[0:M, N+1:2*N] = toep[0:M, -1:0:-1]
circ[M+1:2*M, 0:N] = toep[-1:0:-1, 0:N]
circ[M+1:2*M, N+1:2*N] = toep[-1:0:-1, -1:0:-1]

opCirc = np.fft.fftn(circ)

xx = np.arange(M*N)
X = xx.reshape(N, M).T
X = (xx.T).reshape(N, M).T
XFFT = np.fft.fftn(X, [2*M, 2*N])
Y = np.fft.ifftn(opCirc*XFFT)
xOut = Y[0:M, 0:N]

def mvp(xIn):
    xInRO = xIn.reshape(M, N, order='F')
    XFFT = np.fft.fftn(xInRO, [2 * M, 2 * N])
    Y = np.fft.ifftn(opCirc * XFFT)
    xTemp = Y[0:M, 0:N]
    xPerm = MR * xTemp
    xOutArray = xInRO - xPerm
    xOut = np.zeros((M * N, 1), dtype=np.complex128)
    xOut[idx[0]] = (xOutArray.reshape(M * N, 1, order='F'))[idx[0]]
    return xOut


A = LinearOperator((M*N, M*N), matvec=mvp)

it_count = 0


def iteration_counter(x):
    global it_count
    it_count += 1


def mvp_domain(xIn, opCirc, M, N, MR):
    xInRO = xIn.reshape(M, N, order='F')
    XFFT = np.fft.fftn(MR * xInRO, [2*M, 2*N])
    Y = np.fft.ifftn(opCirc * XFFT)
    xTemp = Y[0:M, 0:N]
    xPerm = xTemp
    xOutArray = xInRO - xPerm
    xOut = np.zeros((M * N, 1), dtype=np.complex128)
    xOut = (xOutArray.reshape(M*N, 1, order='F'))
    return xOut


xIn = xx
# This transpose is a hack caused by different reshapes in Matlab and Python
xInRO = xIn.reshape(M, N, order='F')
XFFT = np.fft.fftn(xInRO, [2*M, 2*N])
Y = np.fft.ifftn(opCirc * XFFT)
xTemp = Y[0:M, 0:N]
Mr = Mr.reshape(M, N, order='F')
xPerm = Mr * xTemp
xOutArray = xInRO - xPerm
xOut = np.zeros((M*N, 1), dtype=np.complex128)
xOut[idx[0]] = (xOutArray.reshape(M * N, 1, order='F'))[idx[0]]

xmin, xmax, ymin, ymax = [-wx/2+dx/2, wx/2-dx/2, -wy/2+dy/2, wy/2-dy/2]
plot_grid = np.mgrid[xmin:xmax:M * 1j, ymin:ymax:N * 1j]

# Construct circulant approximation in x-direction
start = time.time()
c = np.zeros((M, N), dtype=np.complex128)

for i in range(1, M):
    c[i, :] = (M - i) / M * toep[i, :] + i/M * toep[(M - 1) - i + 1, :]

# Fix up 1st entry
c[0, :] = toep[0, :]

c_fft = np.fft.fft(c.T).T

# Construct 1-level preconditioner
circ = np.zeros((M, N, N), dtype=np.complex128)
for i_loop in range(0, M):
    temp = np.zeros((N, N), dtype=np.complex128)
    temp[0:N, 0:N] = toeplitz(c_fft[i_loop, 0:N],
                              c_fft[i_loop, 0:N])
    circ[i_loop, :, :] = temp

# Invert preconditioner
circ_inv = np.zeros_like(circ)
for i in range(0, M):
    circ_inv[i, :, :] = np.linalg.inv(np.identity(N) - (refInd**2 - 1) *
                                      circ[i, :, :])

end = time.time()
print('Preconditioner assembly time = ', end - start)

def mvp_circ(x, circ_inv, M, N, IDX):
    x_r = x
    # from IPython import embed; embed()
    x_r[np.invert(IDX)] = 0.0 
    x_rhs = x_r.reshape(M*N, 1, order='F')

    temp = x_rhs.reshape(M, N, order='F')
    temp = np.fft.fft(temp, axis=0).T
    for i in range(0, M):
        temp[:, i] = np.matmul(circ_inv[i, :, :], temp[:, i])

    temp = np.fft.ifft(temp.T, axis=0)
    TEMP = temp.reshape(M*N, 1, order='F')
    TEMP_RO = TEMP
    TEMP_RO[np.invert(IDX)] = 0.0 + 0.0j
    matvec = TEMP_RO.reshape(M*N, 1, order='F')
    return matvec

idx_all = np.ones((M*N, 1), dtype=bool)
mvp_prec = lambda x: mvp_circ(x, circ_inv, M, N, IDX[:, 0])

prec = LinearOperator((M*N, M*N), matvec=mvp_prec)

it_count = 0
start = time.time()
solp, info = gmres(A, eInc, M=prec, tol=1e-4, callback=iteration_counter)
# solp, info = gmres(A, eInc, tol=1e-4, callback=iteration_counter)
print("The linear system was solved in {0} iterations".format(it_count))
end = time.time()
print('Solve time = ', end-start,'s')

print('Relative residual = ',
      np.linalg.norm(mvp(solp)-eInc)/np.linalg.norm(eInc))

mvp_eval = mvp_domain(solp, opCirc, M, N, MR)

EINC = np.zeros((M * N, 1), dtype=np.complex128)
EINC = np.exp(1j * ko * (np.real(x)*dInc[0] + np.imag(x*dInc[1])))

E_tot = EINC.reshape(M, N, order='F') \
    - mvp_eval.reshape(M, N, order='F') \
    + solp.reshape(M, N, order='F')

E = mvp_eval.reshape(M, N, order='F')

verts = np.array([[-square_side/2, -square_side/2],
                  [-square_side/2, square_side/2],
                  [square_side/2, square_side/2],
                  [square_side/2, -square_side/2]])

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
fig = plt.figure(figsize=(12, 8))
ax = fig.gca()
plt.imshow(np.real(E_tot.T), extent=[-wx/2,wx/2,-wy/2,wy/2],
           cmap=plt.cm.get_cmap('viridis'), interpolation='spline16')
polygon = Polygon(verts, facecolor="none", 
              edgecolor='black', lw=0.8)
plt.gca().add_patch(polygon)

plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

fig.savefig('results/square.png')
plt.close()
