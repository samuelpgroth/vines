import pyfftw
import multiprocessing
import numpy as np
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
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


def mvp_vec_test(xIn, circ_op, rho_ratio, idx, Mr):
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


def mvp_vec_rho_fftw(xIn, circ_op, circ_op_grad, idx, Mr, Dr_grad,
                     rho_ratio):
    ''' Matrix-vector product with FFTW'''
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order='F')
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = pyfftw.interfaces.numpy_fft.fftn(xInRO, [2 * L, 2 * M, 2 * N])
    Y = pyfftw.interfaces.numpy_fft.ifftn(circ_op * xFFT)

    # MVP with gradient of operator
    # x component
    Y_grad_x = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 0]
                                                 * xFFT)
    # y component
    Y_grad_y = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 1]
                                                 * xFFT)
    # z component
    Y_grad_z = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 2]
                                                 * xFFT)

    dot_grad = Dr_grad[0, :, :, :] * Y_grad_x[0:L, 0:M, 0:N] + \
        Dr_grad[1, :, :, :] * Y_grad_y[0:L, 0:M, 0:N] + \
        Dr_grad[2, :, :, :] * Y_grad_z[0:L, 0:M, 0:N]

    xPerm = rho_ratio * Mr * Y[0:L, 0:M, 0:N]
    xOut = rho_ratio * xInRO - xPerm - dot_grad
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


pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
def mvp_volume_potential(xIn, circ_op, idx, Mr):
    ''' Matrix-vector product with FFTW'''
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order='F')
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = pyfftw.interfaces.numpy_fft.fftn(xInRO, [2 * L, 2 * M, 2 * N])
    Y = pyfftw.interfaces.numpy_fft.ifftn(circ_op * xFFT)
    xOut = Mr * Y[0:L, 0:M, 0:N]
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


# Evaluate scattered field in domain - requires on MVP
def mvp_potential_x_perm(xIn, circ_op, idx, Mr):
    (L, M, N) = Mr.shape
    xInRO = xIn.reshape(L, M, N, order='F')
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    xFFT = np.fft.fftn(Mr * xInRO, [2 * L, 2 * M, 2 * N])
    Y = np.fft.ifftn(circ_op * xFFT)
    xOut = Y[0:L, 0:M, 0:N]
    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


# Evaluate portion of scattered field from gradient of potential
# operator in domain (for density contrast problems) - requires on MVP
def mvp_potential_grad(xIn, circ_op_grad, idx, Dr_grad, rho_ratio):
    ''' Matrix-vector product with FFTW'''
    (_, L, M, N) = Dr_grad.shape
    xInRO = xIn.reshape(L, M, N, order='F')
    xInRO[np.invert(idx)] = 0.0

    xOut = np.zeros((L, M, N), dtype=np.complex128)
    xOutVec = np.zeros((L * M * N, 1), dtype=np.complex128)

    # MVP with gradient of operator
    # x component
    xFFT = pyfftw.interfaces.numpy_fft.fftn(Dr_grad[0, :, :, :] *
                                            xInRO,
                                            [2 * L, 2 * M, 2 * N])
    Y_x = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 0] * xFFT)
    # y component
    yFFT = pyfftw.interfaces.numpy_fft.fftn(Dr_grad[1, :, :, :] *
                                            xInRO,
                                            [2 * L, 2 * M, 2 * N])
    Y_y = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 1] * yFFT)
    # z component
    zFFT = pyfftw.interfaces.numpy_fft.fftn(Dr_grad[2, :, :, :] *
                                            xInRO,
                                            [2 * L, 2 * M, 2 * N])
    Y_z = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 2] * zFFT)

    xOut = Y_x[0:L, 0:M, 0:N] + Y_y[0:L, 0:M, 0:N] + Y_z[0:L, 0:M, 0:N]

    # # DIFFERENT VERSION
    # xFFT = pyfftw.interfaces.numpy_fft.fftn(xInRO, [2 * L, 2 * M, 2 * N])

    # # MVP with gradient of operator
    # # x component
    # Y_grad_x = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 0]
    #                                              * xFFT)
    # # y component
    # Y_grad_y = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 1]
    #                                              * xFFT)
    # # z component
    # Y_grad_z = pyfftw.interfaces.numpy_fft.ifftn(circ_op_grad[:, :, :, 2]
    #                                              * xFFT)

    # dot_grad = Dr_grad[0, :, :, :] * Y_grad_x[0:L, 0:M, 0:N] + \
    #     Dr_grad[1, :, :, :] * Y_grad_y[0:L, 0:M, 0:N] + \
    #     Dr_grad[2, :, :, :] * Y_grad_z[0:L, 0:M, 0:N]

    # xOut = -dot_grad

    xOut[np.invert(idx)] = 0.0
    xOutVec = xOut.reshape(L * M * N, 1, order='F')
    return xOutVec


