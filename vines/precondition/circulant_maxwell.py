def circ_1_level(Toep, L, M, N):
    import numpy as np
    from scipy.linalg import toeplitz
    # Create 1-level circulant approximation to Toeplitz operator
    circ_L_opToep = np.zeros((L, M, N, 6), dtype=np.complex128)

    A = Toep[:,:,:,0]

    #  Now construct circulant approximation
    c1 = np.zeros((L, M, N), dtype=np.complex128)

    for i in range(1, L):
        c1[i, :, :] = (L - i)/L * A[i, :, :] + i/L * A[(L-1)-i+1, :, :]

    # Fix up for 1st element
    c1[0, :, :] = A[0, :, :]
    c1_fft = np.fft.fft(c1.T).T
    circ_L_opToep[:, :, :, 0] = c1_fft

    #----- 2nd block
    B = Toep[:, :, :, 1]

    # Now construct circulant approximation
    c2 = np.zeros((L, M, N), dtype=np.complex128)
    for i in range(1, L):
        c2[i, :, :] = -(L - i)/L * B[i, :, :] + i/L * B[(L-1)-i+1,:,:]

    c2[0, :, :] = B[0, :, :]

    c2_fft = np.fft.fft(c2.T).T

    circ_L_opToep[:, :, :, 1] = c2_fft

    #----- 3rd block
    C = Toep[:, :, :, 2]

    # Now construct circulant approximation
    c3 = np.zeros((L, M, N), dtype=np.complex128)
    for i in range(1, L):
        c3[i, :, :] = -(L - i)/L * C[i, :, :] + i/L * C[(L-1)-i+1, :, :]

    c3[0, :, :] = C[0, :, :]
    c3_fft = np.fft.fft(c3.T).T

    circ_L_opToep[:, :, :, 2] = c3_fft

    #---- 4th block
    D = Toep[:, :, :, 3]

    # Now construct circulant approximation
    c4 = np.zeros((L, M, N), dtype=np.complex128)
    for i in range(1, L):
        c4[i, :, :] = (L - i)/L * D[i, :, :] + i/L * D[(L-1)-i+1, :, :]

    c4[0, :, :] = D[0, :, :]
    c4_fft = np.fft.fft(c4.T).T;

    circ_L_opToep[:, :, :, 3] = c4_fft

    #---- 5th block
    E = Toep[:, :, :, 4]
    c5 = np.zeros((L, M, N), dtype=np.complex128)
    # Now construct circulant approximation
    for i in range(1, L):
        c5[i, :, :] = (L - i)/L * E[i, :, :] + i/L * E[(L-1)-i+1, :, :]

    c5[0, :, :] = E[0, :, :]

    c5_fft = np.fft.fft(c5.T).T
    circ_L_opToep[:, :, :, 4] = c5_fft

    #---- 6th block
    F = Toep[:, :, :, 5]

    c6 = np.zeros((L, M, N), dtype=np.complex128)
    # Now construct circulant approximation
    for i in range(1, L):
        c6[i, :, :] = (L-i)/L * F[i, :, :] + i/L * F[(L-1)-i+1, :, :]

    c6[0, :, :] = F[0, :, :]
    c6_fft = np.fft.fft(c6.T).T

    circ_L_opToep[:, :, :, 5] = c6_fft

    # Construct 1-level preconditioner
    circ = np.zeros((L, 3*M*N, 3*M*N), dtype=np.complex128)
    for i_loop in range(0, L):
        temp = np.zeros((3*M*N, 3*M*N), dtype=np.complex128)
        chan = np.zeros((N, M, M), dtype=np.complex128)
        # First block
        for i in range(0, N):
            chan[i, :, :] = toeplitz(c1_fft[i_loop, 0:M, i], c1_fft[i_loop, 0:M, i])

        result = chan[toeplitz(np.arange(0, N))].transpose(0, 2, 1, 3).reshape(M*N, M*N).copy()
        temp[0:M*N, 0:M*N] = result

        # Second block
        for i in range(0, N):
            chan[i, :, :] = toeplitz(np.concatenate((c2_fft[i_loop, 0, i], -c2_fft[i_loop, 1:M, i]), axis=None),
                                     c2_fft[i_loop, 0:M, i])

        result = chan[toeplitz(np.arange(0, N))].transpose(0, 2, 1, 3).reshape(M*N, M*N).copy()
        temp[0:M*N, M*N:2*M*N] = result
        temp[M*N:2*M*N,0:M*N] = result

        # Third block
        for i in range(0, N):
            chan[i, :, :] = toeplitz(c3_fft[i_loop, 0:M, i], c3_fft[i_loop, 0:M, i])

        result = chan[toeplitz(np.arange(0, N))].transpose(0, 2, 1, 3).reshape(M*N, M*N).copy()
        Upper = np.triu(result)
        result = Upper-Upper.T+np.diag(np.diag(Upper));
        temp[0:M*N, 2*M*N:3*M*N] = result
        temp[2*M*N:3*M*N,0:M*N] = result

        # Fourth block
        for i in range(0, N):
            chan[i, :, :] = toeplitz(c4_fft[i_loop, 0:M, i], c4_fft[i_loop, 0:M, i])

        result = chan[toeplitz(np.arange(0, N))].transpose(0, 2, 1, 3).reshape(M*N, M*N).copy()
        temp[M*N:2*M*N, M*N:2*M*N] = result

        # Fifth block
        for i in range(0, N):
            chan[i, :, :] = toeplitz(np.concatenate((c5_fft[i_loop, 0, i], -c5_fft[i_loop, 1:M, i]), axis=None),
                                     c5_fft[i_loop, 0:M, i])

        result = chan[toeplitz(np.arange(0, N))].transpose(0, 2, 1, 3).reshape(M*N, M*N).copy()
        Upper = np.triu(result)
        result = Upper + Upper.T - np.diag(np.diag(Upper));
        temp[M*N:2*M*N, 2*M*N:3*M*N] = result
        temp[2*M*N:3*M*N,M*N:2*M*N] = result

        # Sixth block
        for i in range(0, N):
            chan[i, :, :] = toeplitz(c6_fft[i_loop, 0:M, i], c6_fft[i_loop, 0:M, i])

        result = chan[toeplitz(np.arange(0, N))].transpose(0, 2, 1, 3).reshape(M*N, M*N).copy()
        temp[2*M*N:3*M*N, 2*M*N:3*M*N] = result

        circ[i_loop, :, :] = temp
        
        return circ, circ_L_opToep
        
def circ_2_level(circ_L_opToep, L, M, N):
    import numpy as np
    from scipy.linalg import toeplitz
    circ_M_opToep = np.zeros((L, M, N, 6), dtype=np.complex128)

    circ2 = np.zeros((L, M, 3*N, 3*N), dtype=np.complex128)

    for i_loop in range(0, L):
        # FIX ME: Don't need to create new A-F arrays, get rid of them
        A = circ_L_opToep[i_loop, :, :, 0]
        B = circ_L_opToep[i_loop, :, :, 1]
        C = circ_L_opToep[i_loop, :, :, 2]
        D = circ_L_opToep[i_loop, :, :, 3]
        E = circ_L_opToep[i_loop, :, :, 4]
        F = circ_L_opToep[i_loop, :, :, 5]

        c1 = np.zeros((M, N), dtype=np.complex128)
        c2 = np.zeros((M, N), dtype=np.complex128)
        c3 = np.zeros((M, N), dtype=np.complex128)
        c4 = np.zeros((M, N), dtype=np.complex128)
        c5 = np.zeros((M, N), dtype=np.complex128)
        c6 = np.zeros((M, N), dtype=np.complex128)
        for i in range(1, M):
            c1[i, :] =  (M - i)/M * A[i, :] + i/M * A[(M-1)-i+1, :]
            c2[i, :] = -(M - i)/M * B[i, :] + i/M * B[(M-1)-i+1, :]
            c3[i, :] =  (M - i)/M * C[i, :] + i/M * C[(M-1)-i+1, :]
            c4[i, :] =  (M - i)/M * D[i, :] + i/M * D[(M-1)-i+1, :]
            c5[i, :] = -(M - i)/M * E[i, :] + i/M * E[(M-1)-i+1, :]
            c6[i, :] =  (M - i)/M * F[i, :] + i/M * F[(M-1)-i+1, :]

        c1[0, :] = A[0, :]
        c2[0, :] = B[0, :]
        c3[0, :] = C[0, :]
        c4[0, :] = D[0, :]
        c5[0, :] = E[0, :]
        c6[0, :] = F[0, :]

        c1_fft = np.fft.fft(c1, axis=0)
        c2_fft = np.fft.fft(c2, axis=0)
        c3_fft = np.fft.fft(c3, axis=0)
        c4_fft = np.fft.fft(c4, axis=0)
        c5_fft = np.fft.fft(c5, axis=0)
        c6_fft = np.fft.fft(c6, axis=0)

        circ_M_opToep[i_loop, :, :, 0] = c1_fft
        circ_M_opToep[i_loop, :, :, 1] = c2_fft
        circ_M_opToep[i_loop, :, :, 2] = -c3_fft
        circ_M_opToep[i_loop, :, :, 3] = c4_fft
        circ_M_opToep[i_loop, :, :, 4] = c5_fft
        circ_M_opToep[i_loop, :, :, 5] = c6_fft

        for j_loop in range(0, M):
            temp = np.zeros((3*N, 3*N), dtype=np.complex128)

            # First block
            temp[0:N, 0:N] = toeplitz(c1_fft[j_loop, 0:N], c1_fft[j_loop, 0:N])

            # Second block
            result = toeplitz(np.concatenate((c2_fft[j_loop, 0], c2_fft[j_loop, 1:N]), axis=None),
                                     c2_fft[j_loop, 0:N])
            temp[0:N, N:2*N] = result
            temp[N:2*N, 0:N] = result

            # Third block
            result = toeplitz(np.concatenate((c3_fft[j_loop, 0], -c3_fft[j_loop, 1:N]), axis=None),
                                     c3_fft[j_loop, 0:N])
            temp[0:N, 2*N:3*N] = result
            temp[2*N:3*N, 0:N] = result

            # Fourth block
            temp[N:2*N, N:2*N] = toeplitz(c4_fft[j_loop, 0:N], c4_fft[j_loop, 0:N])

            # Fifth block
            result = toeplitz(np.concatenate((c5_fft[j_loop, 0], -c5_fft[j_loop, 1:N]), axis=None),
                                     c5_fft[j_loop, 0:N])
            temp[N:2*N, 2*N:3*N] = result
            temp[2*N:3*N, N:2*N] = result

            # Sixth block
            temp[2*N:3*N, 2*N:3*N] = toeplitz(c6_fft[j_loop, 0:N], c6_fft[j_loop, 0:N])

            circ2[i_loop, j_loop, :, :] = temp
            
    return circ2, circ_L_opToep