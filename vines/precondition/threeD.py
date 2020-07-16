def gperiodic_coeff_nop(cube):
    import numpy as np
    if  cube in 'L':
        Gp = np.array((+1.0, -1.0, -1.0, +1.0, +1.0, +1.0))

    elif cube in 'M':
        Gp = np.array((+1.0, -1.0, +1.0, +1.0, -1.0, +1.0))

    elif cube in 'N':
        Gp = np.array((+1.0, +1.0, -1.0, +1.0, -1.0, +1.0))

    elif cube in 'LM':
        Gp = np.array((+1.0, +1.0, -1.0, +1.0, -1.0, +1.0))

    elif cube in 'LN':
        Gp = np.array((+1.0, -1.0, +1.0, +1.0, -1.0, +1.0))

    elif cube in 'MN':    
        Gp = np.array((+1.0, -1.0, -1.0, +1.0, +1.0, +1.0))

    elif cube in 'LMN':
        Gp = np.array((+1.0, +1.0, +1.0, +1.0, +1.0, +1.0))

    return Gp


def circulant_nop_const(Toep, L, M, N):
    import numpy as np
    G_mn = Toep
    Gp_mn = np.zeros((2 * L, 2 * M, 2 * N, 6), dtype=np.complex128)

    # Cube 'L'
    Gp_coeff_L = gperiodic_coeff_nop('L')
    # Cube 'M'
    Gp_coeff_M = gperiodic_coeff_nop('M')
    # Cube 'N'
    Gp_coeff_N = gperiodic_coeff_nop('N')
    # Cube 'LM'
    Gp_coeff_LM = gperiodic_coeff_nop('LM')
    # Cube 'LN'
    Gp_coeff_LN = gperiodic_coeff_nop('LN')
    # Cube 'MN'
    Gp_coeff_MN = gperiodic_coeff_nop('MN')

    Gp_mn[0:L, 0:M, 0:N, :] = G_mn
    # %
    for ii in range(0, 6):
        # Cube 'L'
        Gp_mn[L+1:2*L,0:M,0:N,ii] = G_mn[-1:0:-1, 0:M, 0:N, ii] * Gp_coeff_L[ii]
        # Cube 'M'
        Gp_mn[0:L,M+1:2*M,0:N,ii] = G_mn[0:L, -1:0:-1, 0:N, ii] * Gp_coeff_M[ii]
        # Cube 'N'
        Gp_mn[0:L, 0:M, N+1:2*N, ii]     = G_mn[0:L, 0:M, -1:0:-1, ii] * Gp_coeff_N[ii]
        # Cube 'LM'
        Gp_mn[L+1:2*L, M+1:2*M, 0:N, ii] = G_mn[-1:0:-1, -1:0:-1, 0:N, ii] * Gp_coeff_LM[ii]
        # Cube 'LN'
        Gp_mn[L+1:2*L, 0:M, N+1:2*N, ii] = G_mn[-1:0:-1, 0:M, -1:0:-1, ii] * Gp_coeff_LN[ii]
        # Cube 'MN'
        Gp_mn[0:L, M+1:2*M, N+1:2*N, ii] = G_mn[0:L, -1:0:-1, -1:0:-1,ii] * Gp_coeff_MN[ii]

    # Cube 'LMN'
    Gp_mn[L+1:2*L, M+1:2*M, N+1:2*N, :]  = G_mn[-1:0:-1, -1:0:-1, -1:0:-1, :]
    
    return Gp_mn

def fft_operator(A):
    import numpy as np
    L,M,N,d = A.shape
    fA = np.zeros((L,M,N,d), dtype=np.complex128)
    # 3D-FFT of A
    for p in range(0, d):
        fA[:, :, :, p] = np.fft.fftn(A[:, :, :, p])
        
    return fA


def circulant_embed(toep, L, M, N):
    import numpy as np
    # Circulant embedding
    circ = np.zeros((2 * L, 2 * M, 2 * N), dtype=np.complex128)

    circ[0:L, 0:M, 0:N] = toep
    circ[0:L, 0:M, N+1:2*N] = toep[0:L, 0:M, -1:0:-1]
    circ[0:L, M+1:2*M, 0:N] = toep[0:L, -1:0:-1, 0:N]
    circ[0:L, M+1:2*M, N+1:2*N] = toep[0:L, -1:0:-1, -1:0:-1]
    circ[L+1:2*L, 0:M, 0:N] = toep[-1:0:-1, 0:M, 0:N]
    circ[L+1:2*L, 0:M, N+1:2*N] = toep[-1:0:-1, 0:M, -1:0:-1]
    circ[L+1:2*L, M+1:2*M, 0:N] = toep[-1:0:-1, -1:0:-1, 0:N]
    circ[L+1:2*L, M+1:2*M, N+1:2*N] = toep[-1:0:-1, -1:0:-1, -1:0:-1]

    # FFT of circulant operator
    circ_op = np.fft.fftn(circ)
    return circ_op


def circulant_embed_fftw(toep, L, M, N):
    import numpy as np
    import pyfftw, multiprocessing
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    # Circulant embedding
    circ = np.zeros((2 * L, 2 * M, 2 * N), dtype=np.complex128)

    circ[0:L, 0:M, 0:N] = toep
    circ[0:L, 0:M, N+1:2*N] = toep[0:L, 0:M, -1:0:-1]
    circ[0:L, M+1:2*M, 0:N] = toep[0:L, -1:0:-1, 0:N]
    circ[0:L, M+1:2*M, N+1:2*N] = toep[0:L, -1:0:-1, -1:0:-1]
    circ[L+1:2*L, 0:M, 0:N] = toep[-1:0:-1, 0:M, 0:N]
    circ[L+1:2*L, 0:M, N+1:2*N] = toep[-1:0:-1, 0:M, -1:0:-1]
    circ[L+1:2*L, M+1:2*M, 0:N] = toep[-1:0:-1, -1:0:-1, 0:N]
    circ[L+1:2*L, M+1:2*M, N+1:2*N] = toep[-1:0:-1, -1:0:-1, -1:0:-1]

    # FFT of circulant operator
    circ_op = pyfftw.interfaces.numpy_fft.fftn(circ)
    return circ_op


def fftw_operator(A):
    import numpy as np
    import pyfftw, multiprocessing
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    L, M, N, d = A.shape
    fA = np.zeros((L, M, N, d), dtype=np.complex128)
    # 3D-FFT of A
    for p in range(0, d):
        fA[:, :, :, p] = pyfftw.interfaces.numpy_fft.fftn(A[:, :, :, p])

    return fA


def circulant_gradient_embed(toep, L, M, N):
    import numpy as np
    # Circulant embedding
    circ = np.zeros((2 * L, 2 * M, 2 * N, 3), dtype=np.complex128)

    for i in range(0, 3):
        circ[0:L, 0:M, 0:N, i] = toep[:, :, :, i]
        circ[0:L, 0:M, N+1:2*N, i] = toep[0:L, 0:M, -1:0:-1, i]
        circ[0:L, M+1:2*M, 0:N, i] = toep[0:L, -1:0:-1, 0:N, i]
        circ[0:L, M+1:2*M, N+1:2*N, i] = toep[0:L, -1:0:-1, -1:0:-1, i]
        circ[L+1:2*L, 0:M, 0:N, i] = toep[-1:0:-1, 0:M, 0:N, i]
        circ[L+1:2*L, 0:M, N+1:2*N, i] = toep[-1:0:-1, 0:M, -1:0:-1, i]
        circ[L+1:2*L, M+1:2*M, 0:N, i] = toep[-1:0:-1, -1:0:-1, 0:N, i]
        circ[L+1:2*L, M+1:2*M, N+1:2*N, i] = toep[-1:0:-1, -1:0:-1, -1:0:-1, i]

    # FFT of circulant operator
    circ_op = fftw_operator(circ)
    return circ_op
