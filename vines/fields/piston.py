def plane_circular_piston(rad, k, points):
    import numpy as np
    import quadpy

    scheme_near = quadpy.disk.lether(50)
    w_near = scheme_near.weights * rad**2
    nodes_near = scheme_near.points * rad

    import time
    from numba import njit, prange
    start = time.time()
    @njit(parallel=True)
    def eval_source(points, k, nodes_near, w_near):
        p = np.zeros_like(points[0], dtype=np.complex128)
        for i in prange(points.shape[1]):
            dist = np.sqrt(points[0, i]**2 + (points[2, i] - nodes_near[:, 0])**2 + (points[1, i] - nodes_near[:, 1])**2)
            integrand = np.exp(1j * k * dist) / (4 * np.pi * dist)
            p[i] = np.sum(integrand * w_near)
        return p

    p = eval_source(points, k, nodes_near, w_near)
    end = time.time()
    print('Time taken (parallel) = ', end-start)

    return p