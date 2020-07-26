def getOPERATOR_DDA(r, ko, refInd, kvec, Eo, nearby_quad):
    import numpy as np
    (L, M, N, _) = r.shape
    # Self-interaction Classius-Mossotti stuff
    dx=r[1, 0, 0, 0] - r[0, 0, 0, 0]
    b1 = -1.8915316
    b2 = 0.1648469
    b3 = -1.7700004
    msqr = refInd ** 2
    dcube = dx ** 3
    d=dx

    a_hat = kvec/np.linalg.norm(kvec)
    e_hat = Eo/np.linalg.norm(Eo)
    S = 0
    for j in range(0, 3):
        S = S + (a_hat[j] * e_hat[j]) ** 2


    alpha_CM = 3/(4*np.pi)*(msqr - 1) /(msqr + 2)  # Clausius-Mossotti
    alpha_LDR = alpha_CM/(1 + (alpha_CM)*((b1+msqr*b2+msqr*b3*S)*(ko*d)**2 \
                -2/3*1j*ko**3*dcube))

    I = np.identity(3)

    Toep = np.zeros((L,M,N,6), dtype=np.complex128)
    R0 = r[0, 0, 0, :]

    n_quad = 10
    xG, wG = np.polynomial.legendre.leggauss(n_quad)
    XG, YG, ZG = np.meshgrid(xG, xG, xG)
    XW, YW, ZW = np.meshgrid(wG*0.5, wG*0.5, wG*0.5)

    for i in range(0, L):
        for j in range(0, M):
            for k in range(0, N):
                R1 = r[i,j,k,:]
                rk_to_rj = R1-R0
                rjk = np.linalg.norm(rk_to_rj)

                if nearby_quad in 'on':
                    if rjk < 5 * dx and rjk > 1e-15:
                        x_grid = R1[0] + dx/2 * XG
                        y_grid = R1[1] + dx/2 * YG
                        z_grid = R1[2] + dx/2 * ZG

                        temp = np.zeros((3, 3), dtype=np.complex128)
                        for iQ in range(0, n_quad):
                            for jQ in range(0, n_quad):
                                for kQ in range(0, n_quad):
                                    RQ = np.array([x_grid[iQ, jQ, kQ],
                                         y_grid[iQ, jQ, kQ],z_grid[iQ, jQ, kQ]])

                                    rk_to_rj = RQ - R0

                                    rjk = np.linalg.norm(rk_to_rj)
                                    rjk_hat = rk_to_rj / rjk
                                    rjkrjk = np.outer(rjk_hat, rjk_hat)

                                    Ajk = np.exp(1j*ko*rjk)/rjk * \
                                        (ko**2*(I - rjkrjk) + 
                                        (1j*ko*rjk-1)/rjk**2*(I - 3*rjkrjk))
                                         # Draine & Flatau
                                    temp = temp + Ajk * XW[iQ, jQ, kQ] * YW[iQ, jQ, kQ] * ZW[iQ, jQ, kQ]

                        Toep[i,j,k,0] = temp[0, 0]
                        Toep[i,j,k,1] = temp[0, 1]
                        Toep[i,j,k,2] = temp[0, 2]
                        Toep[i,j,k,3] = temp[1, 1]
                        Toep[i,j,k,4] = temp[1, 2]
                        Toep[i,j,k,5] = temp[2, 2]
                    else:
                        if np.abs(rjk)>1e-15:
                            R1 = r[i,j,k,:]
                            rk_to_rj = R1-R0
                            rjk = np.linalg.norm(rk_to_rj)
                            rjk_hat = (rk_to_rj)/rjk
                            rjkrjk = np.outer(rjk_hat, rjk_hat)

                            Ajk = np.exp(1j*ko*rjk)/rjk*(ko**2*(I - rjkrjk) + 
                                        (1j*ko*rjk-1)/rjk**2*(I - 3*rjkrjk)) 
                            Toep[i,j,k,0] = Ajk[0, 0]
                            Toep[i,j,k,1] = Ajk[0, 1]
                            Toep[i,j,k,2] = Ajk[0, 2]
                            Toep[i,j,k,3] = Ajk[1, 1]
                            Toep[i,j,k,4] = Ajk[1, 2]
                            Toep[i,j,k,5] = Ajk[2, 2] 


                else:
                    if np.abs(rjk)>1e-15:
                        rjk_hat = (rk_to_rj)/rjk
                        rjkrjk = np.outer(rjk_hat, rjk_hat)

                        Ajk = np.exp(1j*ko*rjk)/rjk*(ko**2*(I - rjkrjk) + 
                                    (1j*ko*rjk-1)/rjk**2*(I - 3*rjkrjk))
                        Toep[i,j,k,0] = Ajk[0, 0]
                        Toep[i,j,k,1] = Ajk[0, 1]
                        Toep[i,j,k,2] = Ajk[0, 2]
                        Toep[i,j,k,3] = Ajk[1, 1]
                        Toep[i,j,k,4] = Ajk[1, 2]
                        Toep[i,j,k,5] = Ajk[2, 2]
                        
                        
    opCirculant = circulant_nop_const(Toep, L, M, N)
    op_out = fft_operator(opCirculant)

    return op_out, Toep, alpha_LDR

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
    Gp_mn = np.zeros((2*L,2*M,2*N ,6), dtype=np.complex128)

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