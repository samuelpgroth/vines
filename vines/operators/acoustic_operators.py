import numpy as np
from numba import jit, njit, prange
def volume_potential(ko, r):
    ''' Create Toeplitz operator '''
    (L, M, N, _) = r.shape
    toep = np.zeros((L, M, N), dtype=np.complex128)
    dx = r[1, 0, 0, 0] - r[0, 0, 0, 0]
    vol = (dx)**3  # voxel volume
    a = (3/4 * vol / np.pi)**(1/3)  # radius of sphere of same volume
    R0 = r[0, 0, 0, :]

    # self = np.pi * 1j / 2 * (np.pi / 2 + ko * a)
    # self = (np.exp(1j*ko*a)-1)/(1j*ko)
    self = (1/ko**2 - 1j*a/ko) * np.exp(1j*ko*a) - 1/ko**2

    # self = np.pi * 1j / 2 * (-np.pi / 2 + ko * a - 1/18*(ko * a)**3)
    nearby_quad = 'off'
    n_quad = 10
    xG, wG = np.polynomial.legendre.leggauss(n_quad)
    XG, YG, ZG = np.meshgrid(xG, xG, xG)
    XW, YW, ZW = np.meshgrid(wG*0.5, wG*0.5, wG*0.5)

    @njit(parallel=True)
    def potential_fast(ko):
        toep = np.zeros((L, M, N), dtype=np.complex128)
        for i in prange(0, L):
            for j in range(0, M):
                for k in range(0, N):
                    R1 = r[i, j, k, :]
                    rk_to_rj = R1-R0
                    rjk = np.linalg.norm(rk_to_rj)
                    if nearby_quad in 'on':
                        if rjk < 5 * dx and rjk > 1e-15:
                            x_grid = R1[0] + dx/2 * XG
                            y_grid = R1[1] + dx/2 * YG
                            z_grid = R1[2] + dx/2 * ZG

                            temp = 0.0+0.0j
                            for iQ in range(0, n_quad):
                                for jQ in range(0, n_quad):
                                    for kQ in range(0, n_quad):
                                        RQ = np.array([x_grid[iQ, jQ, kQ],
                                                       y_grid[iQ, jQ, kQ],
                                                       z_grid[iQ, jQ, kQ]])

                                        rk_to_rj = RQ - R0

                                        rjk = np.linalg.norm(rk_to_rj)
                                        rjk_hat = rk_to_rj / rjk
                                        rjkrjk = np.outer(rjk_hat, rjk_hat)

                                        Ajk = np.exp(1j * ko * rjk) / \
                                            (4 * np.pi * rjk) * dx**3
                                        # Draine & Flatau
                                        temp = temp + Ajk * XW[iQ, jQ, kQ] * YW[iQ, jQ, kQ] * ZW[iQ, jQ, kQ]
                            toep[i, j, k] = temp
                        else:
                            if np.abs(rjk) > 1e-15:
                                toep[i, j, k] = \
                                    np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
                            else:
                                toep[i, j, k] = self
                    else:
                        if np.abs(rjk) > 1e-15:
                            toep[i, j, k] = \
                                np.exp(1j * ko * rjk) / (4 * np.pi * rjk) * dx**3
                        else:
                            toep[i, j, k] = self
        return toep

    return potential_fast(ko)