def bowl_transducer(k, focal_length, focus, radius,
                    n_elements, aperture_radius, points,
                    axis):
    ''' Generates a field from a uniform bowl transducer with or without
    an aperture. This is essentially a segment of a sphere's surface.
    We compute it in a slightly crude way by spreading many point sources
    evenly over the surface. This is done according to
    `How to generate equidistributed points on the surface of a sphere' by
    Markus Deserno (https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf).
    Note that in practice such tranducers with uniformly distributed sources
    are not used in practice.
    '''
    import numpy as np
    theta1 = np.arcsin(aperture_radius / focal_length)
    theta2 = np.arcsin(radius / focal_length)

    r = 1.0  # radius of the sphere
    n_count = 0
    a = 2 * np.pi * r**2 * (np.cos(theta1) - np.cos(theta2)) / n_elements
    d = np.sqrt(a)
    M_theta = np.int(np.round((theta2 - theta1) / d))
    d_theta = (theta2 - theta1) / M_theta
    d_phi = a / d_theta
    x = []
    y = []
    z = []
    for m in range(0, M_theta):
        theta = (theta2 - theta1) * (m + 0.5) / M_theta + theta1
        M_phi = np.int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(0, M_phi):
            phi = 2 * np.pi * n / M_phi
            x.append(focal_length * np.sin(theta) * np.cos(phi))
            y.append(focal_length * np.sin(theta) * np.sin(phi))
            z.append(focal_length * np.cos(theta))
            n_count += 1

    # Ro-order so that the transducer is behind the axis, rather than in front
    if axis in 'z':
        x = np.array(x)
        y = np.array(y)
        z = -np.array(z)
    elif axis in 'x':
        x_t = np.array(x)
        y_t = np.array(y)
        z_t = -np.array(z)
        z = -y_t
        y = -x_t
        x = z_t

    # if axis in 'z':
    #     x = np.array(x)
    #     y = np.array(y)
    #     z = focus[2] - np.array(z)
    # elif axis in 'x':
    #     x_t = np.array(x)
    #     y_t = np.array(y)
    #     z_t = focus[0] - np.array(z)
    #     z = -y_t
    #     y = -x_t
    #     x = z_t

    # Rotate about the z-axis
    # FIXME: assumes that the x-axis is the central axis of the transducer
    rot_angle = 0
    rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                        [np.sin(rot_angle), np.cos(rot_angle), 0],
                        [0, 0, 1]])

    coord_array = np.zeros((x.shape[0], 3))
    coord_array[:, 0] = x[:]
    coord_array[:, 1] = y[:]
    coord_array[:, 2] = z[:]

    new_array = np.zeros_like(coord_array)
    for ii in range(x.shape[0]):
        new_array[ii, :] = rot_mat.dot(coord_array[ii, :])

    x[:] = new_array[:, 0]
    y[:] = new_array[:, 1]
    z[:] = new_array[:, 2]

    # Shift to make focus at focus[0,1,2]
    if axis in 'z':
        z = focus[2] + z
    elif axis in 'x':
        x = focus[0] + x

    # def eval_source_vec(x, y, z, points, k):
    #     # p = np.zeros_like(points[0])
    #     p = 0.0
    #     for i in range(x.shape[0]):
    #         dist = np.sqrt((points[0] - x[i])**2 + (points[1] - y[i])**2 +
    #                        (points[2] - z[i])**2)
    #         p += np.exp(1j * k * dist) / (4 * np.pi * dist)
    #     return p


    from numba import njit, prange
    @njit(parallel=True)
    def eval_source(x, y, z, points, k):
        p = np.zeros_like(points[0], dtype=np.complex128)
        # from IPython import embed; embed()
        for i in prange(points.shape[1]):
            temp = 0.0
            for j in range(x.shape[0]):
                dist = np.sqrt((points[0, i] - x[j])**2 +
                               (points[1, i] - y[j])**2 +
                               (points[2, i] - z[j])**2)
                if dist > 1e-3:
                    temp += np.exp(1j * k * dist) / (4 * np.pi * dist)
            p[i] = temp
        return p


    p = eval_source(x, y, z, points, k)
    # p = eval_source_vec(x, y, z, points, k)

    return x, y, z, p*a


def normalise_power(power, rho, c0, radius, k1, focal_length,
                    focus, n_elements, aperture_radius):
    import numpy as np
    n_quad = 500
    r_quad_dim = radius * 1.0
    r_quad = np.linspace(0, r_quad_dim, n_quad)
    # x_location_disk = x_location
    x_location_disk = focal_length - 0.98 * np.sqrt(focal_length**2 - radius**2)
    points_quad = np.vstack((x_location_disk * np.ones(n_quad),
                            np.zeros(n_quad),
                            r_quad))

    _, _, _, p_quad = bowl_transducer(np.real(k1), focal_length, focus, radius,
                        n_elements, aperture_radius, points_quad,'x')
    integral = 2*np.pi*np.sum(np.abs(p_quad)**2 * r_quad)*r_quad_dim/n_quad 
    p0 = np.sqrt(2*rho*c0*power/integral)
    return p0


def bowl_transducer_rotate(k, focal_length, focus, radius,
                    n_elements, aperture_radius, points,
                    axis, rot_angle):
    ''' Generates a field from a uniform bowl transducer with or without
    an aperture. This is essentially a segment of a sphere's surface.
    We compute it in a slightly crude way by spreading many point sources
    evenly over the surface. This is done according to
    `How to generate equidistributed points on the surface of a sphere' by
    Markus Deserno (https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf).
    Note that in practice such tranducers with uniformly distributed sources
    are not used in practice.
    '''
    import numpy as np
    theta1 = np.arcsin(aperture_radius / focal_length)
    theta2 = np.arcsin(radius / focal_length)

    r = 1.0  # radius of the sphere
    n_count = 0
    a = 2 * np.pi * r**2 * (np.cos(theta1) - np.cos(theta2)) / n_elements
    d = np.sqrt(a)
    M_theta = np.int(np.round((theta2 - theta1) / d))
    d_theta = (theta2 - theta1) / M_theta
    d_phi = a / d_theta
    x = []
    y = []
    z = []
    for m in range(0, M_theta):
        theta = (theta2 - theta1) * (m + 0.5) / M_theta + theta1
        M_phi = np.int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(0, M_phi):
            phi = 2 * np.pi * n / M_phi
            x.append(focal_length * np.sin(theta) * np.cos(phi))
            y.append(focal_length * np.sin(theta) * np.sin(phi))
            z.append(focal_length * np.cos(theta))
            n_count += 1

    # Ro-order so that the transducer is behind the axis, rather than in front
    if axis in 'z':
        x = np.array(x)
        y = np.array(y)
        z = -np.array(z)
    elif axis in 'x':
        x_t = np.array(x)
        y_t = np.array(y)
        z_t = -np.array(z)
        z = -y_t
        y = -x_t
        x = z_t

    # if axis in 'z':
    #     x = np.array(x)
    #     y = np.array(y)
    #     z = focus[2] - np.array(z)
    # elif axis in 'x':
    #     x_t = np.array(x)
    #     y_t = np.array(y)
    #     z_t = focus[0] - np.array(z)
    #     z = -y_t
    #     y = -x_t
    #     x = z_t

    # Rotate about the z-axis
    # FIXME: assumes that the x-axis is the central axis of the transducer
    # rot_angle = np.pi / 2
    rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                        [np.sin(rot_angle), np.cos(rot_angle), 0],
                        [0, 0, 1]])

    coord_array = np.zeros((x.shape[0], 3))
    coord_array[:, 0] = x[:]
    coord_array[:, 1] = y[:]
    coord_array[:, 2] = z[:]

    new_array = np.zeros_like(coord_array)
    for ii in range(x.shape[0]):
        new_array[ii, :] = rot_mat.dot(coord_array[ii, :])

    x[:] = new_array[:, 0]
    y[:] = new_array[:, 1]
    z[:] = new_array[:, 2]

    # Shift to make focus at focus[0,1,2]
    if axis in 'z':
        z = focus[2] + z
    elif axis in 'x':
        x = focus[0] + x

    # def eval_source_vec(x, y, z, points, k):
    #     # p = np.zeros_like(points[0])
    #     p = 0.0
    #     for i in range(x.shape[0]):
    #         dist = np.sqrt((points[0] - x[i])**2 + (points[1] - y[i])**2 +
    #                        (points[2] - z[i])**2)
    #         p += np.exp(1j * k * dist) / (4 * np.pi * dist)
    #     return p


    from numba import njit, prange
    @njit(parallel=True)
    def eval_source(x, y, z, points, k):
        p = np.zeros_like(points[0], dtype=np.complex128)
        # from IPython import embed; embed()
        for i in prange(points.shape[1]):
            temp = 0.0
            for j in range(x.shape[0]):
                dist = np.sqrt((points[0, i] - x[j])**2 +
                               (points[1, i] - y[j])**2 +
                               (points[2, i] - z[j])**2)
                if dist > 1e-3:
                    temp += np.exp(1j * k * dist) / (4 * np.pi * dist)
            p[i] = temp
        return p


    p = eval_source(x, y, z, points, k)
    # p = eval_source_vec(x, y, z, points, k)

    return x, y, z, p*a


def normalise_power_rotate(power, rho, c0, radius, k1, focal_length,
                    focus, n_elements, aperture_radius, rot_angle):
    import numpy as np
    n_quad = 500
    r_quad_dim = radius * 1.0
    r_quad = np.linspace(0, r_quad_dim, n_quad)
    # x_location_disk = x_location
    x_location_disk = focal_length - 0.98 * np.sqrt(focal_length**2 - radius**2)
    points_quad = np.vstack((x_location_disk * np.ones(n_quad),
                            np.zeros(n_quad),
                            r_quad))
    # Rotate points_quad by rot_angle
    rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0],
                        [np.sin(rot_angle), np.cos(rot_angle), 0],
                        [0, 0, 1]])
    
    # Perform rotation
    for i in range(n_quad):
        # Need to shift points so that focus is at origin, then rotate, then
        # shift back
        shifted = points_quad[:, i] - np.array([focal_length, 0, 0])
        points_quad[:, i] = rot_mat.dot(shifted) + np.array([focal_length, 0, 0])

    _, _, _, p_quad = bowl_transducer_rotate(np.real(k1), focal_length, focus, radius,
                        n_elements, aperture_radius, points_quad,'x', rot_angle)
    integral = 2*np.pi*np.sum(np.abs(p_quad)**2 * r_quad)*r_quad_dim/n_quad 
    p0 = np.sqrt(2*rho*c0*power/integral)
    return p0
