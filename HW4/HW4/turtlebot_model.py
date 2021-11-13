import numpy as np

EPSILON_OMEGA = 1e-3


def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    V = u[0]
    om = u[1]
    theta_0 = xvec[2]
    theta = xvec[2] + u[1] * dt

    if abs(om) > EPSILON_OMEGA:
        x = xvec[0] + V / om * (np.sin(theta) - np.sin(theta_0))
        y = xvec[1] + V / om * (-np.cos(theta) + np.cos(theta_0))

        Gx = np.array([[1, 0, V / om * (np.cos(theta_0 + om * dt) - np.cos(theta_0))],
                      [0, 1, V / om * (np.sin(theta_0 + om * dt) - np.sin(theta_0))],
                      [0, 0, 1]]
                      )
        Gu = np.array([[1/om * (np.sin(theta) - np.sin(theta_0)), V/om * (dt * np.cos(theta) + 1/om * (np.sin(theta_0) - np.sin(theta)))],
                      [1/om * (np.cos(theta_0) - np.cos(theta)),  V/om * (dt * np.sin(theta) + 1/om * (np.cos(theta) - np.cos(theta_0)))],
                      [0, dt]])

    else:
        x = xvec[0] + V * np.cos(theta_0) * dt
        y = xvec[1] + V * np.sin(theta_0) * dt

        # lim (d/dt) = d/dt (lim )
        Gx = np.array([[1, 0, - V * np.sin(theta_0) * dt],
                      [0, 1, V * np.cos(theta_0) * dt],
                      [0, 0, 1]]
                      )

        # lim (d/dt)
        Gu = np.array([[np.cos(theta_0) * dt, -(1/2) * V * dt**2 * np.sin(theta_0)],
                      [np.sin(theta_0) * dt, (1/2) * V * dt**2 * np.cos(theta_0)],
                      [0, dt]])
        # d/dt (lim)
        # Gu = np.array([[np.cos(theta_0) * dt, 0],
        #                [np.sin(theta_0) * dt, 0],
        #                [0, dt]])

    g = np.array([x, y, theta])


    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu


def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

    def rotation_matrix(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, -s], [s, c]])

    x_base, y_base, th_base = x
    r_base = x[:2]
    x_cam_H, y_cam_H, th_cam_H = tf_base_to_camera

    # First rotate back and then translate
    camera_xy_in_world = (rotation_matrix(th_base) @ tf_base_to_camera[:2]) + r_base
    x_cam, y_cam = camera_xy_in_world[0], camera_xy_in_world[1]
    th_cam = x[2] + tf_base_to_camera[2]

    # Second, get the (alpha_in_cam, r_in_cam)
    alpha_in_cam = alpha - th_cam
    angle = np.arctan2(y_cam, x_cam)
    r_in_cam = r - np.linalg.norm(camera_xy_in_world) * np.cos(alpha - angle)
    h = np.array([alpha_in_cam, r_in_cam])

    print("alpha", alpha)
    print("alpha_in_cam", alpha_in_cam)
    print("angle", angle)
    print("th_cam_H", th_cam_H)
    print("th_base", th_base)
    print("th_cam", th_cam)

    # Third, get the Jacobian
    # partial h / x  = [ 0  (see below)]
    # partial h / y  = [ 0  (see below)]
    # partial h / th = [-1  (see below)]
    cos_th_base = np.cos(th_base)
    sin_th_base = np.sin(th_base)

    denominator_term_1 = x_base + x_cam_H * cos_th_base - y_cam_H * sin_th_base
    denominator_term_2 = y_base + y_cam_H * cos_th_base + x_cam_H * sin_th_base
    denominator = np.sqrt(denominator_term_1 ** 2 + denominator_term_2 ** 2)

    projection_factor = np.cos(alpha - angle)

    # H12 = - projection_factor * denominator_term_1 / denominator
    # H22 = - projection_factor * denominator_term_2 / denominator
    H32 = - np.sin(alpha - th_base) * denominator
    ca = np.cos(alpha)
    ca2 = np.cos(2 * alpha)
    sa = np.sin(alpha)
    sa2 = np.sin(2 * alpha)

    H12 = (- ca / 2) - (ca * ca2 / 2) - (sa * sa2 / 2)
    H22 = (- sa / 2) + (ca2 * sa / 2) - (ca * sa2 / 2)
    # H32 = (- ca / 2) + (ca * ca2)

    Hx = np.array([[0, 0, -1], [H12, H22, H32]])

    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
