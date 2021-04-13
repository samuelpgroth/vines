import time
def rhs_lookup(P, beta, omega, rho, c, points2, L2, M2, N2, i_harm, interp_funs):
    # Create vector for matrix-vector product
    if i_harm == 1:
        # Second harmonic
        xIn = -2 * beta * omega**2 / (rho * c**4) * P[0] * P[0]
    elif i_harm == 2:
        # Third harmonic
        # Interpolate P1 and P2 onto finer grid
        start = time.time()
        p1_interp = interp_funs[0](points2)
        P1_interp = p1_interp.reshape(L2, M2, N2, order='F')
        p2_interp = interp_funs[1](points2)
        P2_interp = p2_interp.reshape(L2, M2, N2, order='F')
        end = time.time()
        print('Interpolation time = ', end-start)
        xIn = -9 * beta * omega**2 / (rho * c**4) * P1_interp * P2_interp
    elif i_harm == 3:
        # Fourth harmonic
        start = time.time()
        p1_interp = interp_funs[0](points2)
        P1 = p1_interp.reshape(L2, M2, N2, order='F')
        p2_interp = interp_funs[1](points2)
        P2 = p2_interp.reshape(L2, M2, N2, order='F')
        p3_interp = interp_funs[2](points2)
        P3 = p3_interp.reshape(L2, M2, N2, order='F')
        end = time.time()
        print('Interpolation time = ', end-start)
        xIn = -8 * beta * omega**2 / (rho * c**4) * \
            (P2 * P2 + 2 * P1 * P3)
    elif i_harm == 4:
        # Fifth harmonic
        start = time.time()
        p1_interp = interp_funs[0](points2)
        P1 = p1_interp.reshape(L2, M2, N2, order='F')
        p2_interp = interp_funs[1](points2)
        P2 = p2_interp.reshape(L2, M2, N2, order='F')
        p3_interp = interp_funs[2](points2)
        P3 = p3_interp.reshape(L2, M2, N2, order='F')
        p4_interp = interp_funs[3](points2)
        P4 = p4_interp.reshape(L2, M2, N2, order='F')
        end = time.time()
        print('Interpolation time = ', end-start)
        xIn = -25 * beta * omega**2 / (rho * c**4) * \
            (P1 * P4 + P2 * P3)
    elif i_harm == 5:
        # Sixth harmonic
        xIn = -18 * beta * omega**2 / (rho * c**4) * \
            (2 * P[0] * P[4] + 2 * P[1] * P[3] + P[2]**2)
    elif i_harm == 6:
        # Seventh harmonic
        xIn = -49 * beta * omega**2 / (rho * c**4) * \
            (P[0] * P[5] + P[1] * P[4] + P[2] * P[3])

    return xIn