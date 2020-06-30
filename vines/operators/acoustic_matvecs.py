import pyfftw
import multiprocessing
import numpy as np
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
def mvp_vec_fftw(xIn, circ_op, idx, Mr):
    ''' Matrix-vector product with FFTW'''
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order='F')
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = pyfftw.interfaces.numpy_fft.fftn(xInRO, [2 * L, 2 * M, 2 * N])
    Y = pyfftw.interfaces.numpy_fft.ifftn(circ_op * xFFT)
    xPerm = Mr * Y[0:L, 0:M, 0:N]
    xOut = xInRO - xPerm
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


def mvp_vec(xIn, circ_op, idx, Mr):
    ''' Matrix-vector product with numpy's FFT'''
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order='F')
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = np.fft.fftn(xInRO, [2 * L, 2 * M, 2 * N])
    Y = np.fft.ifftn(circ_op * xFFT)
    xPerm = Mr * Y[0:L, 0:M, 0:N]
    xOut = xInRO - xPerm
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


# Evaluate solution in domain - requires on MVP
def mvp_domain(xIn, circ_op, idx, Mr):
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order='F')
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = np.fft.fftn(Mr * xInRO, [2 * L, 2 * M, 2 * N])
    Y = np.fft.ifftn(circ_op * xFFT)
    xPerm = Y[0:L, 0:M, 0:N]
    xOut = xInRO - xPerm
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec
