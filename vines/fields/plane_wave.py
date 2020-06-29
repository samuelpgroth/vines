def PlaneWave(Uo, k, dInc, r):
    import numpy as np

    (L, M, N, _) = r.shape

    krx = k * dInc[0] * r[:, :, :, 0]
    kry = k * dInc[1] * r[:, :, :, 1]
    krz = k * dInc[2] * r[:, :, :, 2]

    kr = krx + kry + krz

    expKr = np.exp(1j * kr)

    Uinc = np.zeros((L, M, N), dtype=np.complex128)
    Uinc = Uo * expKr

    return Uinc
