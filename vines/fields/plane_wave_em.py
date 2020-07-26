def PlaneWaveEM(Eo, kvec, r):
    import numpy as np

    (L, M, N, junk) = r.shape
    
    krx = kvec[0] * r[:, :, :, 0]
    kry = kvec[1] * r[:, :, :, 1]
    krz = kvec[2] * r[:, :, :, 2]

    kr = krx + kry + krz

    expKr = np.exp(1j * kr)

    Einc = np.zeros((L, M, N, 3), dtype=np.complex128)
    Einc[:, :, :, 0] = Eo[0] * expKr
    Einc[:, :, :, 1] = Eo[1] * expKr
    Einc[:, :, :, 2] = Eo[2] * expKr
    
    return Einc