# Matrix-vector product with Toeplitz operator
def mvp_vec(JIn0, op_out, idx, Gram, Mr, Mc):    
    import numpy as np

    (L, M, N) = Mr.shape
    JIn = JIn0.reshape(L, M, N, 3, order = 'F') 
    JIn[np.invert(idx)] = 0.0  

    JOut = np.zeros((L, M, N, 3), dtype=np.complex128)
    JOutVec = np.zeros((3 * L * M* N, 1), dtype=np.complex128)
    
    # x component of JIn, store contribution on 3 components of Jout
    fJ = np.fft.fftn(JIn[:, :, :, 0], (2*L, 2*M, 2*N))
    Jout1 = op_out[:, :, :, 0] * fJ
    Jout2 = op_out[:, :, :, 1] * fJ
    Jout3 = op_out[:, :, :, 2] * fJ
    
    # y component of JIn, add contribution on 3 components of Jout
    fJ = np.fft.fftn(JIn[:, :, :, 1], (2*L, 2*M, 2*N))
    Jout1 = Jout1 + op_out[:, :, :, 1] * fJ
    Jout2 = Jout2 + op_out[:, :, :, 3] * fJ
    Jout3 = Jout3 + op_out[:, :, :, 4] * fJ
    
    # z component of JIn, add contribution on 3 components of Jout
    fJ = np.fft.fftn(JIn[:, :, :, 2], (2*L, 2*M, 2*N))
    Jout1 = Jout1 + op_out[:, :, :, 2] * fJ
    Jout2 = Jout2 + op_out[:, :, :, 4] * fJ
    Jout3 = Jout3 + op_out[:, :, :, 5] * fJ
                           
    # apply ifft, multiply by material properties and Gram
    Jout1 = np.fft.ifftn(Jout1)
    JOut[:, :, :, 0] = Gram * Mr * JIn[:, :, :, 0] - Mc * Jout1[0:L, 0:M, 0:N]
    Jout2 = np.fft.ifftn(Jout2)
    JOut[:, :, :, 1] = Gram * Mr * JIn[:, :, :, 1] - Mc * Jout2[0:L, 0:M, 0:N]
    Jout3 = np.fft.ifftn(Jout3)
    JOut[:, :, :, 2] = Gram * Mr * JIn[:, :, :, 2] - Mc * Jout3[0:L, 0:M, 0:N]
    
    JOut[np.invert(idx)] = 0.0
    
    JOutVec = JOut.reshape(L*M*N*3, 1, order = 'F') 
    
    return JOutVec

# Matrix-vector product with 2-level circulant preconditioner
def mvp_circ2(JInVec, circ2_inv, L, M, N, idx):
    import numpy as np
    V_R = JInVec.reshape(L, M, N, 3, order='F')
    V_R[np.invert(idx)] = 0.0 
    
    Vrhs = V_R.reshape(3*L*M*N, 1, order='F')

    temp = Vrhs.reshape(L,3*M*N, order='F')
    temp = np.fft.fft(temp, axis=0).T  # transpose is application of permutation matrix

    for i in range(0, L):
        TEMP = temp[:, i].reshape(M,3*N, order='F')
        TEMP = np.fft.fft(TEMP, axis=0).T
        for j in range(0, M):
            TEMP[:,j] = np.matmul(circ2_inv[i, j, :, :], TEMP[:, j])

        TEMP = np.fft.ifft(TEMP.T, axis=0)
        temp[:, i] = TEMP.reshape(1,3*M*N, order='F')


    temp = np.fft.ifft(temp.T, axis=0)  # transpose is application of permutation matrix transpose
    TEMP = temp.reshape(3*L*M*N,1, order='F')
    TEMP_RO = TEMP.reshape(L, M, N, 3, order='F')
    TEMP_RO[np.invert(idx)] = 0.0 +0j 
    matvec = TEMP_RO.reshape(3*L*M*N, 1, order='F')
    return matvec