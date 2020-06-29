
from numba import njit, prange
import numpy as np


def generatedomain(res, dx, dy, dz):
    # Get minimum number of voxels in each direction
    nx = np.int(np.max((1.0, np.round(dx / res))))
    ny = np.int(np.max((1.0, np.round(dy / res))))
    nz = np.int(np.max((1.0, np.round(dz / res))))

    Dx = nx * res
    Dy = ny * res
    Dz = nz * res

    x = np.arange(0.0, Dx, res) + (-Dx / 2 + res/2)
    y = np.arange(0.0, Dy, res) + (-Dy / 2 + res/2)
    z = np.arange(0.0, Dz, res) + (-Dz / 2 + res/2)

    r, L, M, N = grid3d(x, y, z)
    return r, L, M, N


@njit(parallel=True)
def grid3d(x, y, z):
    # define the dimensions
    L = x.shape[0]
    M = y.shape[0]
    N = z.shape[0]

    # allocate space
    r = np.zeros((L, M, N, 3))

    for ix in prange(0, L):
        xx = x[ix]
        for iy in range(0, M):
            yy = y[iy]
            for iz in range(0, N):
                zz = z[iz]
                r[ix, iy, iz, :] = np.array((xx, yy, zz))

    return r, L, M, N


def koch_snowflake(order, scale=1):
    """
    Return two lists x, y of point coordinates of the Koch snowflake.

    Arguments
    ---------
    order : int
        The recursion depth.
    scale : float
        The extent of the snowflake (edge length of the base triangle).
    """
    def _koch_snowflake_complex(order):
        if order == 0:
            # initial triangle
            angles = np.array([0, 120, 240]) + 90
            return scale * np.exp(np.deg2rad(angles) * 1j)
#             return scale / np.sqrt(3) * np.exp(np.deg2rad(angles) * 1j)
        else:
            ZR = 0.5 - 0.5j * np.sqrt(3) / 3

            p1 = _koch_snowflake_complex(order - 1)  # start points
            p2 = np.roll(p1, shift=-1)  # end points
            dp = p2 - p1  # connection vectors

            new_points = np.empty(len(p1) * 4, dtype=np.complex128)
            new_points[::4] = p1
            new_points[1::4] = p1 + dp / 3
            new_points[2::4] = p1 + dp * ZR
            new_points[3::4] = p1 + dp / 3 * 2
            return new_points

    points = _koch_snowflake_complex(order)
    x, y = points.real, points.imag

    # Stick coordinates into an array (useful for contains_points function)
    P = np.zeros((x.shape[0], 2), dtype=np.float64)
    P[:, 0] = x
    P[:, 1] = y

    return x, y, P


def shape_size_param(geom, refInd, sizeParam, nPerLam, aspectRatio):
    import numpy as np
    from matplotlib import path

    if geom in 'hex':
        a = 1
        b = np.sqrt(3)/2 * a
        dom_x = 2 * a
        dom_y = 2 * b
        dom_z = a * aspectRatio
        theta = np.arange(0, 7) * 2*np.pi/6
        verts = a * np.exp(1j * theta)
        P = np.zeros((verts.shape[0], 2), dtype=np.float64)
        P[:, 0] = verts.real
        P[:, 1] = verts.imag
    elif geom in 'koch':
        a = 1
        x, y, _ = koch_snowflake(order=5, scale=a)
        dom_x = np.max(x) - np.min(x)
        dom_y = np.max(y) - np.min(y)
        dom_z = a * aspectRatio
        P = np.zeros((x.shape[0]+1, 2), dtype=np.float64)
        P[:-1, 0] = x
        P[:-1, 1] = y
        P[-1, 0] = x[0]
        P[-1, 1] = y[0]
    elif geom in 'sphere':
        a = 1
        dom_x = 2 * a
        dom_y = dom_x
        dom_z = dom_x
        P = []  # vertices leave blank

    lambda_ext = 2 * np.pi * a / sizeParam     # exterior wavelength
    lambda_int = lambda_ext / np.real(refInd)  # interior wavelength

    # Discretise geometry into voxels
    h_pref = dom_x  # enforce precise discretisation in x-direction
    res_temp = lambda_int / nPerLam  # provisional resolution
    N = np.int(np.ceil(h_pref / res_temp))
    res = h_pref / N

    r, L, M, N = generatedomain(res, dom_x, dom_y, dom_z)

    # Determine which points lie inside shape
    if geom in 'sphere':
        r_sq = r[:, :, :, 0]**2 + r[:, :, :, 1]**2 + r[:, :, :, 2]**2
        idx = (r_sq <= a)
        # from IPython import embed; embed()
    else:
        # Polyhedron
        points = r[:, :, :, 0:2].reshape(L*M*N, 2, order='F')
        p = path.Path(P)
        idx = p.contains_points(points).reshape(L, M, N, order='F')

    return r, idx, res, P, lambda_ext, lambda_int


def shape(geom, refInd, lambda_ext, radius, nPerLam, aspectRatio):
    import numpy as np
    from matplotlib import path

    if geom in 'hex':
        a = radius
        b = np.sqrt(3)/2 * a
        dom_x = 2 * a
        dom_y = 2 * b
        dom_z = a * aspectRatio
        theta = np.arange(0, 7) * 2*np.pi/6
        verts = a * np.exp(1j * theta)
        P = np.zeros((verts.shape[0], 2), dtype=np.float64)
        P[:, 0] = verts.real
        P[:, 1] = verts.imag
    elif geom in 'koch':
        a = radius
        x, y, _ = koch_snowflake(order=5, scale=a)
        dom_x = np.max(x) - np.min(x)
        dom_y = np.max(y) - np.min(y)
        dom_z = a * aspectRatio
        P = np.zeros((x.shape[0]+1, 2), dtype=np.float64)
        P[:-1, 0] = x
        P[:-1, 1] = y
        P[-1, 0] = x[0]
        P[-1, 1] = y[0]
    elif geom in 'sphere':
        a = radius
        dom_x = 2 * a
        dom_y = dom_x
        dom_z = dom_x
        P = []  # vertices leave blank

    # lambda_ext = 2 * np.pi * a / sizeParam     # exterior wavelength
    lambda_int = lambda_ext / np.real(refInd)  # interior wavelength

    # Discretise geometry into voxels
    h_pref = dom_x  # enforce precise discretisation in x-direction
    res_temp = lambda_int / nPerLam  # provisional resolution
    N = np.int(np.ceil(h_pref / res_temp))
    res = h_pref / N

    r, L, M, N = generatedomain(res, dom_x, dom_y, dom_z)

    # Determine which points lie inside shape
    if geom in 'sphere':
        r_sq = r[:, :, :, 0]**2 + r[:, :, :, 1]**2 + r[:, :, :, 2]**2
        idx = (r_sq <= a**2)
        # from IPython import embed; embed()
    else:
        # Polyhedron
        points = r[:, :, :, 0:2].reshape(L*M*N, 2, order='F')
        p = path.Path(P)
        idx = p.contains_points(points).reshape(L, M, N, order='F')

    return r, idx, res, P, lambda_int
