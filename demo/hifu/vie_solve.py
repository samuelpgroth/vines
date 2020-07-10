# Integral equation solve routine
#
# The steps taken to solve the volume integral equation are:
#
# 1. Assemble Toeplitz integral operator, T, over the scatterer's bounding box
# 2. Embed Toeplitz operator, T, in a circulant operator and take FFT
# 3. Set up a matrix-vector product function (I - M*T)
# 4. Set up the iterative solver (GMRES, BiCGStab,...)
# 5. Perform iterative solve

# Inputs required:
#
# 1. Mr - permittivities of voxels
# 2. r - voxel coordinates
# 3. idx - indices of scatter voxels (True if in scatterer, False if not)
# 4. k - wavenumber
# 5. u_inc - incident field evaluated over voxel grid

import numpy as np
from vines.operators.acoustic_operators import volume_potential
from vines.operators.acoustic_matvecs import mvp_vec_fftw, mvp_potential_x_perm
from scipy.sparse.linalg import LinearOperator, gmres
from vines.precondition.threeD import circulant_embed_fftw
from scipy.sparse.linalg import LinearOperator, gmres
import time


def vie_solver(Mr, r, idx, u_inc, k):
    # Toeplitz operator
    T = k**2 * volume_potential(k, r)

    # Get shape of voxel grid
    (L, M, N, _) = r.shape
    n_voxel = L * M * N

    # Circulant embedding of Toeplitz operator
    circ = circulant_embed_fftw(T, L, M, N)

    # Create array that has the incident field values inside scatterer
    xIn = np.zeros((L, M, N), dtype=np.complex128)
    xIn[idx] = u_inc
    xInVec = xIn.reshape((n_voxel, 1), order='F')

    def mvp(x):
        'Matrix-vector product operator'
        return mvp_vec_fftw(x, circ, idx, Mr)

    # Linear operator
    A = LinearOperator((n_voxel, n_voxel), matvec=mvp)

    def residual_vector(rk):
        'Function to store residual vector in iterative solve'
        # global resvec
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

    # Evaluate scattered field in domain using representation formula
    idx_all = np.ones((L, M, N), dtype=bool)
    u_sca = mvp_potential_x_perm(sol, circ, idx_all,
                                 Mr).reshape(L, M, N, order='F')

    return sol, J, u_sca
