import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
from . import turtlebot_model as tb


EPSILON_OMEGA = 1e-3


class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx, :2].mean(axis=0)
        th = self.xs[idx, 2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # TODO: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.

        us = np.random.multivariate_normal(u, self.R, self.M)  # return Mx2
        self.xs = self.transition_model(us, dt)

        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # TODO: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.
        
        i = 0
        c = ws[0]
        
        x = np.zeros((xs.shape))
        w = np.zeros((ws.shape))
        
        for m in range(ws.size):
            u = np.sum(ws)*(r+m/self.M)
            while c < u:
                i = i + 1
                c = c + ws[i]
            x[m,0:3] = xs[i,:]
            w[m] = ws[i]
#            w[m] = 1/self.M - not sure why this isn't right, the algorithm in the assignment tells us to do this
            
        self.xs = x
        self.ws = w

        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # TODO: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each partical
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate
        #       updates for them

        # g = np.zeros((self.M, 3))

        # for i in range(self.M):
        #     g[i, :], Gx, Gu = tb.compute_dynamics(self.xs[i, :], us[i, :], dt, compute_jacobians=True)
        g = tb.compute_dynamics_vectorized(self.xs, us, dt, compute_jacobians=False)
        assert g.shape == (self.M, 3)

        ########## Code ends here ##########

        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # TODO: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()
        
        vs, Q = self.measurement_model(z_raw, Q_raw)
        I = z_raw.size // 2
        p = np.zeros(self.M)

        for m in range(self.M):
            ws[m] = scipy.stats.multivariate_normal.pdf(vs[m,:], np.zeros(2*I), Q)

        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw))   # M x 2I

        ########## Code starts here ##########
        # TODO: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful
        Q = scipy.linalg.block_diag(*Q_raw)

        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        ########## Code starts here ##########
        # TODO: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each
        #       observed line, find the most likely map entry (the entry with
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(),
        #       np.linalg.solve(), np.meshgrid() useful.

        I = z_raw.shape[1]
        J = self.map_lines.shape[1]

        vs = np.empty((self.M, I, 2))

        hs = self.compute_predicted_measurements()

        # Start the vectorization
        # We will use the computations to reduce (M, I, 2, J) to (M, I, 2)
        vijm = np.empty([self.M, I, 2, J])

        # Tiled version
        # I x M x J for every single one
        z0_tiled = np.tile(z_raw[0, :, np.newaxis, np.newaxis], (1, self.M, J))
        z1_tiled = np.tile(z_raw[1, :, np.newaxis, np.newaxis], (1, self.M, J))
        hs0_tiled = np.tile(hs[np.newaxis, :, 0, :], (I, 1, 1))
        hs1_tiled = np.tile(hs[np.newaxis, :, 1, :], (I, 1, 1))

        # Change I x M x J to M x I x J
        vijm[:, :, 0, :] = np.transpose(angle_diff(z0_tiled, hs0_tiled), (1, 0, 2))
        vijm[:, :, 1, :] = np.transpose(z1_tiled - hs1_tiled, (1, 0, 2))

        # PRIOR CODE WITH FOR LOOPS, for reference!
        # for m in range(self.M):
        #     vij = np.empty([I, 2, J])
        #
        #     for i in range(I):  # for each measurement
        #         dij = np.empty([J, ])
        #         vij = np.empty([2, J])
        #
        #         vijm[m, i, 0, :] = angle_diff(z_raw[0, i], hs[m, 0, :])
        #         vijm[m, i, 1, :] = z_raw[1, i] - hs[m, 1, :]
        #
        #         for j in range(J):  # for each line
        #             # vij[0, j] = angle_diff(z_raw[0, i], hs[m, 0, j])
        #             # vij[1, j] = z_raw[1, i] - hs[m, 1, j]
        #             dij[j] = vij[:, j].T @ np.linalg.solve(Q_raw[i, :, :], vij[:, j])
        #
        #         min_idx = np.argmin(dij)
        #         vs[m, i, :] = vij[i, :, min_idx]

        # This is (I x 2 x 2)^{-1} @ (M x I x 2 x J) = M x I x 2 x J
        solve_pre = np.linalg.solve(Q_raw, vijm)
        # This is M x I x 2 x J inside to M x I x J after
        dij = np.sum(vijm * solve_pre, axis=2)  # Sum over the columns of each to get the distance
        # This selects on the last column, so M x I x J to M x I
        min_idx = np.argmin(dij, axis=2)  # argmin over the J

        ii, mm = np.meshgrid(range(I), range(self.M))
        vs[:, :, :] = vijm[mm, ii, :, min_idx[mm, ii]]

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M, -1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # TODO: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.

        J = self.map_lines.shape[1]
        hs = np.zeros((self.M, 2, J))

        for i in range(self.M):
            h = np.empty([2, J])
            Hx = np.empty([2, self.xs.shape[1], J])

            # for j in range(J):  # loop over map lines
            #     ########## Code starts here ##########
            #     # TODO: Compute h, Hx using tb.transform_line_to_scanner_frame() for the j'th map line.
            #     # HINT: This should be a single line of code.
            #     line = self.map_lines[:, j]
            #     h[:,j], Hx[:,:,j] = tb.transform_line_to_scanner_frame(line, self.xs[i,:], self.tf_base_to_camera)

            h = tb.transform_line_to_scanner_frame(self.map_lines, self.xs[i, :], self.tf_base_to_camera,
                                                   compute_jacobian=False)

            ########## Code ends here ##########

            h = tb.normalize_line_parameters(h)
            hs[i, :, :] = h

        ########## Code ends here ##########

        return hs

