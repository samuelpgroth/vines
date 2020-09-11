import numpy as np
from scipy.special import hankel1
from scipy.linalg import toeplitz

def geometry2d(h_temp, wx, wy):
    # FIXME: currently this allows dx and dy to differ. Just make the same size
    # to get square pixels
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

    return x, A, a, M, N, dx, dy


def get_operator(A, ko, x, a, M, N):
    # Fundamental solution of the Helmholtz equation
    g = lambda x, y: A * 1j/4 * hankel1(0, ko * np.abs(x - y))

    # Self term
    self_term = a**2 * 1j * np.pi/2 * ((1 + 1j * np.euler_gamma) / 2
                                - 1j / np.pi + 1j / np.pi * np.log(ko * a / 2))
    
    # Sparse matvec
    def potential(x):
        toep = np.zeros((M, N), dtype=np.complex128)
        for i in range(M):
            for j in range(N):
                if i == 0 and j == 0:
                    toep[i, j] = self_term
                else:
                    toep[i, j] = g(x[0], x[j * M + i])
        return toep

    toep = ko**2 * potential(x)
    return toep


def circulant_embedding(toep, M, N):
    circ = np.zeros((2 * M, 2 * N), dtype=np.complex128)

    # Circulant embedding
    circ[0:M, 0:N] = toep[0:M, 0:N]
    circ[0:M, N+1:2*N] = toep[0:M, -1:0:-1]
    circ[M+1:2*M, 0:N] = toep[-1:0:-1, 0:N]
    circ[M+1:2*M, N+1:2*N] = toep[-1:0:-1, -1:0:-1]

    opCirc = np.fft.fftn(circ)
    return opCirc


def circulant_preconditioner(toep, M, N, refInd):
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

    return circ_inv