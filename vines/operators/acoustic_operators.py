import numpy as np
from numba import njit, prange
from scipy.special import hankel1


def volume_potential(ko, r):
    ''' Create Toeplitz operator '''
    (L, M, N, _) = r.shape
    dx = r[1, 0, 0, 0] - r[0, 0, 0, 0]
    vol = (dx)**3  # voxel volume
    a = (3/4 * vol / np.pi)**(1/3)  # radius of sphere of same volume
    R0 = r[0, 0, 0, :]

    self = (1/ko**2 - 1j*a/ko) * np.exp(1j*ko*a) - 1/ko**2

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

                                        Ajk = np.exp(1j * ko * rjk) / \
                                            (4 * np.pi * rjk) * dx**3
                                        temp += Ajk * XW[iQ, jQ, kQ] * \
                                            YW[iQ, jQ, kQ] * ZW[iQ, jQ, kQ]
                            toep[i, j, k] = temp
                        else:
                            if np.abs(rjk) > 1e-15:
                                toep[i, j, k] = np.exp(1j * ko * rjk) / \
                                        (4 * np.pi * rjk) * dx**3
                            else:
                                toep[i, j, k] = self
                    else:
                        if np.abs(rjk) > 1e-15:
                            toep[i, j, k] = np.exp(1j * ko * rjk) / \
                                (4 * np.pi * rjk) * dx**3
                        else:
                            toep[i, j, k] = self
        return toep

    return potential_fast(ko)


def volume_potential_cylindrical(ko, r):
    ''' Create Toeplitz operator for cylindrically symmetric case '''
    (L, M, N, _) = r.shape
    dx = r[1, 0, 0, 0] - r[0, 0, 0, 0]
    vol = (dx)**3  # voxel volume
    a = (3/4 * vol / np.pi)**(1/3)  # radius of sphere of same volume
    R0 = r[0, 0, 0, :]

    # self = (1/ko**2 - 1j*a/ko) * np.exp(1j*ko*a) - 1/ko**2
    self = 1/(2j*ko) * (np.exp(1j*ko*a) - 1)

    # @njit(parallel=True)
    # def potential_fast_cylindrical(ko):
    #     toep = np.zeros((L, M, N), dtype=np.complex128)
    #     for i in prange(0, L):
    #         for j in range(0, M):
    #             for k in range(0, N):
    #                 R1 = r[i, j, k, :]
    #                 rk_to_rj = R1-R0
    #                 rjk = np.linalg.norm(rk_to_rj)

    #                 if np.abs(rjk) > 1e-15:
    #                     # toep[i, j, k] = np.exp(1j * ko * rjk) / \
    #                     #     (4 * np.pi * rjk) * dx**2 * np.abs(R1[1])
    #                     toep[i, j, k] = np.exp(1j * ko * rjk) / \
    #                         (4 * np.pi * rjk) * dx**2
    #                 else:
    #                     toep[i, j, k] = self
    #     return toep

    @njit(parallel=True)
    def potential_fast_cylindrical(ko):
        ntheta = 400
        dtheta = 2*np.pi / ntheta
        # theta = np.linspace(dtheta/2, np.pi - dtheta/2, ntheta)
        theta = np.linspace(0, 2*np.pi - dtheta, ntheta)
        # theta = 0
        toep = np.zeros((L, M, N), dtype=np.complex128)
        for i in prange(0, L):
            for j in range(0, M):
                for k in range(0, N):
                    temp = 0
                    for THETA in theta:
                        R1_temp = r[i, j, k, :]
                        R1 = np.array([R1_temp[0],
                                       R1_temp[1] * np.cos(THETA),
                                       R1_temp[1] * np.sin(THETA)])
                        # R1 = np.array([R1_temp[0],
                        #                R1_temp[1],
                        #                np.abs(R1_temp[1]) * np.sin(THETA)])
                        # R1 = r[i, j, k, :]
                        rk_to_rj = R1-R0
                        rjk = np.linalg.norm(rk_to_rj)

                        if np.abs(rjk) > 1e-15:
                            # toep[i, j, k] = np.exp(1j * ko * rjk) / \
                            #     (4 * np.pi * rjk) * dx**2 * np.abs(R1[1])
                            temp += np.exp(1j * ko * rjk) / \
                                (4 * np.pi * rjk) * dx**2
                        else:
                            temp += self
                    toep[i, j, k] = temp * dtheta  #* np.abs(R1_temp[1])
        return toep

    return potential_fast_cylindrical(ko)


def grad_potential(ko, r):
    ''' Create Toeplitz operator '''
    (L, M, N, _) = r.shape
    dx = r[1, 0, 0, 0] - r[0, 0, 0, 0]
    vol = (dx)**3  # voxel volume
    a = (3/4 * vol / np.pi)**(1/3)  # radius of sphere of same volume
    R0 = r[0, 0, 0, :]

    self = 0.0

    nearby_quad = 'off'
    n_quad = 10
    xG, wG = np.polynomial.legendre.leggauss(n_quad)
    XG, YG, ZG = np.meshgrid(xG, xG, xG)
    XW, YW, ZW = np.meshgrid(wG*0.5, wG*0.5, wG*0.5)

    @njit(parallel=True)
    def grad_potential_fast(ko):
        toep = np.zeros((L, M, N, 3), dtype=np.complex128)
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

                                        # Ajk = np.exp(1j * ko * rjk) / \
                                        #     (4 * np.pi * rjk) * dx**3

                                        Ajk = np.exp(1j * ko * rjk) * \
                                            (1j * ko * rjk - 1) / \
                                            (4 * np.pi * rjk**3) * dx**3 * \
                                            rk_to_rj

                                        temp += Ajk * XW[iQ, jQ, kQ] * \
                                            YW[iQ, jQ, kQ] * ZW[iQ, jQ, kQ]
                            toep[i, j, k, :] = temp
                        else:
                            if np.abs(rjk) > 1e-15:
                                # toep[i, j, k] = np.exp(1j * ko * rjk) / \
                                #         (4 * np.pi * rjk) * dx**3
                                toep[i, j, k, :] = np.exp(1j * ko * rjk) * \
                                            (1j * ko * rjk - 1) / \
                                            (4 * np.pi * rjk**3) * dx**3 * \
                                            rk_to_rj
                            else:
                                toep[i, j, k, :] = self
                    else:
                        if np.abs(rjk) > 1e-15:
                            toep[i, j, k, :] = np.exp(1j * ko * rjk) * \
                                            (1j * ko * rjk - 1) / \
                                            (4 * np.pi * rjk**3) * dx**3 * \
                                            rk_to_rj
                        else:
                            toep[i, j, k, :] = self
        return toep

    return grad_potential_fast(ko)


def get_operator_2d(A, ko, x, a):
    M, N, _ = x.shape
    
    # Self term
    self_term = a**2 * 1j * np.pi/2 * ((1 + 1j * np.euler_gamma) / 2
                - 1j / np.pi + 1j / np.pi * np.log(ko * a / 2))

    toep = np.zeros((M, N), dtype=np.complex128)
    for i in range(M):
        for j in range(N):
            if i == 0 and j == 0:
                toep[i, j] = self_term
            else:
                toep[i, j] = A * 1j/4 * hankel1(0,
                    ko * np.linalg.norm(x[0, 0, :] - x[i, j, :]))

    return ko**2 * toep


def circulant_embedding(toep, M, N):
    circ = np.zeros((2 * M, 2 * N), dtype=np.complex128)

    # Circulant embedding
    circ[0:M, 0:N] = toep[0:M, 0:N]
    circ[0:M, N+1:2*N] = toep[0:M, -1:0:-1]
    circ[M+1:2*M, 0:N] = toep[-1:0:-1, 0:N]
    circ[M+1:2*M, N+1:2*N] = toep[-1:0:-1, -1:0:-1]

    opCirc = np.fft.fftn(circ)
    return opCirc