# Circulant acoustic
def circ_1_level_acoustic(Toep, L, M, N, on_off):
    import numpy as np
    from scipy.linalg import toeplitz
    # Create 1-level circulant approximation to Toeplitz operator
    circ_L_opToep = np.zeros((L, M, N), dtype=np.complex128)

    A = Toep

    #  Now construct circulant approximation
    c1 = np.zeros((L, M, N), dtype=np.complex128)

    for i in range(1, L):
        c1[i, :, :] = (L - i)/L * A[i, :, :] + i/L * A[(L-1)-i+1, :, :]
    # from IPython import embed; embed()
    # Fix up for 1st element
    c1[0, :, :] = A[0, :, :]
    c1_fft = np.fft.fft(c1.T).T
    circ_L_opToep = c1_fft

    if (on_off in 'on'):
        # Construct 1-level preconditioner
        circ = np.zeros((L, M*N, M*N), dtype=np.complex128)
        for i_loop in range(0, L):
            temp = np.zeros((M*N, M*N), dtype=np.complex128)
            chan = np.zeros((N, M, M), dtype=np.complex128)
            # First block
            for i in range(0, N):
                chan[i, :, :] = toeplitz(c1_fft[i_loop, 0:M, i], c1_fft[i_loop, 0:M, i])

            result = chan[toeplitz(np.arange(0, N))].transpose(0, 2, 1, 3).reshape(M*N, M*N).copy()
            temp[0:M*N, 0:M*N] = result

            circ[i_loop, :, :] = temp
    else:
        circ = 0
        
    return circ, circ_L_opToep

    
def circ_2_level_acoustic(circ_L_opToep, L, M, N):
    import numpy as np
    from scipy.linalg import toeplitz
    circ_M_opToep = np.zeros((L, M, N), dtype=np.complex128)

    circ2 = np.zeros((L, M, N, N), dtype=np.complex128)

    for i_loop in range(0, L):
        # FIX ME: Don't need to create new A-F arrays, get rid of them
        A = circ_L_opToep[i_loop, :, :]

        c1 = np.zeros((M, N), dtype=np.complex128)

        for i in range(1, M):
            c1[i, :] =  (M - i)/M * A[i, :] + i/M * A[(M-1)-i+1, :]

        c1[0, :] = A[0, :]

        c1_fft = np.fft.fft(c1, axis=0)

        circ_M_opToep[i_loop, :, :] = c1_fft

        for j_loop in range(0, M):
            temp = np.zeros((N, N), dtype=np.complex128)

            # First block
            temp[0:N, 0:N] = toeplitz(c1_fft[j_loop, 0:N], c1_fft[j_loop, 0:N])

            circ2[i_loop, j_loop, :, :] = temp

    return circ2, circ_L_opToep


# Matrix-vector product with 2-level circulant preconditioner
def mvp_circ2_acoustic(JInVec, circ2_inv, L, M, N, idx):
    import numpy as np
    V_R = JInVec.reshape(L, M, N, order='F')
    V_R[np.invert(idx)] = 0.0 
    
    Vrhs = V_R.reshape(L*M*N, 1, order='F')

    temp = Vrhs.reshape(L,M*N, order='F')
    temp = np.fft.fft(temp, axis=0).T  # transpose is application of permutation matrix

    for i in range(0, L):
        TEMP = temp[:, i].reshape(M,N, order='F')
        TEMP = np.fft.fft(TEMP, axis=0).T
        for j in range(0, M):
            TEMP[:,j] = np.matmul(circ2_inv[i, j, :, :], TEMP[:, j])

        TEMP = np.fft.ifft(TEMP.T, axis=0)
        temp[:, i] = TEMP.reshape(1,M*N, order='F')


    temp = np.fft.ifft(temp.T, axis=0)  # transpose is application of permutation matrix transpose
    TEMP = temp.reshape(L*M*N,1, order='F')
    TEMP_RO = TEMP.reshape(L, M, N, order='F')
    TEMP_RO[np.invert(idx)] = 0.0 +0j 
    matvec = TEMP_RO.reshape(L*M*N, 1, order='F')
    return matvec