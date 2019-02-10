import math
from math import cos, sin

import art
import numpy as np
import parameters as p

DEGREE = np.pi / 180.

# Have a state class is better
STATE_SIZE = 5
CONTROL_SIZE = 2
AUG_STATE_SIZE = 7

X_IDX = 0
Y_IDX = 1
VX_IDX = 2
VY_IDX = 3
THETA_IDX = 4
THETA_DOT_IDX = 5
TRUST_IDX = 6

"""
Earth model with its data
"""


def norm_l2(x):
    """
        helper for l2 norm
    """
    return np.linalg.norm(x)


from collections import namedtuple

Star = namedtuple('Star', ['position', 'mass'])


class GravitationalField:
    def __init__(self, stars):
        self.stars = stars

    def acceleration(self, my_pos):
        """
        total accl due to stars
        """
        accl = np.array([0, 0.])
        for star in self.stars:
            accl += self.gravity(my_pos, star)
        return accl

    @staticmethod
    def gravity(my_pos, star):
        """
        accl form a star
        """
        r = star.position - my_pos

        dist = np.linalg.norm(r)

        G = 1
        assert G > 0 and dist > 0
        return G * star.mass * r / dist ** 2


class EarthMotionWithControlResidual:

    def __init__(self, cur_state, next_state, cur_control, next_control, dt, accl_field):
        assert len(cur_state) == STATE_SIZE
        assert len(next_state) == STATE_SIZE
        self.cur_state = cur_state.copy()
        self.next_state = next_state.copy()
        self.cur_control = cur_control.copy()
        self.next_control = next_control.copy()
        self.dt = dt
        self.accl_field = accl_field

    def unpack_state(self):
        """
            helper
        """
        x, y, v_x, v_y, theta = self.cur_state
        return x, y, v_x, v_y, theta

    def unpack_control(self):
        """
            helper
        """
        theta_dot, thrust = self.cur_control
        return theta_dot, thrust

    def motion(self):
        """
            motion model for cur_state
        """
        x, y, v_x, v_y, theta = self.unpack_state()
        theta_dot, thrust = self.unpack_control()

        g_x_dot, g_y_dot = self.accl_field.acceleration(np.array([x, y]))

        new_state = np.array([
            x + self.dt * v_x,
            y + self.dt * v_y,
            v_x + self.dt * cos(theta) * thrust + self.dt * g_x_dot,
            v_y + self.dt * sin(theta) * thrust + self.dt * g_y_dot,
            theta + self.dt * theta_dot,
            theta_dot,
            thrust,
        ])

        return new_state

    def motion_constrain_residual(self):
        """
            return the residual vector
        """
        predict_state_next = self.motion()
        aug_next_state = np.hstack([self.next_state, self.cur_control])
        return predict_state_next - aug_next_state

    def jacobian_residual_wrt_current_state_and_control(self):
        """
            J with respect to cur_state
            Put d_r/d_state with d_r/d_control together for simplicity
        """
        x, y, v_x, v_y, theta = self.unpack_state()
        theta_dot, thrust = self.unpack_control()

        # Assume the order is [x, y, v_x, v_y, theta, theta_dot, thrust]
        jacobian = np.array([
            [1, 0, self.dt, 0, 0, 0, 0],
            [0, 1, 0, self.dt, 0, 0, 0],
            [0, 0, 1, 0, - self.dt * thrust * sin(theta), 0, self.dt * cos(theta)],
            [0, 0, 0, 1, self.dt * thrust * cos(theta), 0, self.dt * sin(theta)],
            [0, 0, 0, 0, 1, self.dt, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        return jacobian

    def jacobian_residual_wrt_next_state(self):
        """
            J with respect to next_state
        """
        return - np.identity(AUG_STATE_SIZE)

    def jacobian_residual_wrt_dt(self):
        """
            J with respect to dt
        """
        x, y, v_x, v_y, theta = self.unpack_state()
        theta_dot, thrust = self.unpack_control()
        g_x_dot, g_y_dot = self.accl_field.acceleration(np.array([x, y]))

        jacobian = np.array([v_x,
                             v_y,
                             cos(theta) * thrust + g_x_dot,
                             sin(theta) * thrust + g_y_dot,
                             theta_dot,
                             0.,
                             0.]).T

        return jacobian

    def weight(self):
        """
            assume weight is same for all motion constrains
        """
        return p.motion_residual_weights

    @staticmethod
    def size_of_residual():
        """
            size of residual vector
        """
        # return STATE_SIZE
        return AUG_STATE_SIZE


class EarthPriorResidual:

    def __init__(self, state, prior_state):
        self.state = state
        self.prior_state = prior_state.copy()

    def residual(self):
        """
        The residual vector
        """
        return self.state - self.prior_state

    def jacobian_wrt_state(self):
        """
        Jacobian
        """
        return np.identity(STATE_SIZE)

    def weight(self):
        """
        weight
        """
        return p.prior_residual_weights

    @staticmethod
    def size_of_residual():
        return STATE_SIZE


class EarthTargetState:
    def __init__(self, state, control, final_state, final_control):
        self.aug_state = np.hstack([state, control])
        self.prior_aug_state = np.hstack([final_state, final_control])

    def residual(self):
        """
        The residual vector
        """
        return self.aug_state - self.prior_aug_state

    def jacobian_wrt_state(self):
        return np.identity(AUG_STATE_SIZE)

    def weight(self):
        w = p.target_residual_weights

        return w

    @staticmethod
    def size_of_residual():
        return AUG_STATE_SIZE


class EarthTrajectoryLeastSquares:
    MIN_THRUST = -0.01

    def __init__(self, prior_state, target_state, target_control,
                 num_state, dt, accl_field):
        self.prior_state = prior_state
        self.target_state = target_state
        self.target_control = target_control
        self.dt = dt
        self.num_states = num_state
        self.accl_field = accl_field

    def least_square_optimization(self):
        states = self.__init_states_linear()
        # states = self.__init_states_circles(self.accl_field.stars[0])

        controls = self.__init_controls()

        for i in range(p.max_iter):
            if i % 20 == 0:
                print('Iteration: ', i)
            J, W, r = self.__construct_least_square(states, controls)

            jt_dot_W = np.dot(J.T, W)
            A = np.dot(jt_dot_W, J)
            b = -np.dot(jt_dot_W, r)

            A, b = self.__regularization(A, b, states, controls)

            if p.enable_thrust_constrain:
                A, b = self.__apply_constrains(A, b, states, controls, i)

            self.__newton(A, b, states, controls)

            if p.show_trajectory_in_optimization:
                art.plot_trajectory(states, controls, is_movie=True)

        return states.copy(), controls.copy(), self.dt

    def __update_variables(self, states, controls, dx, step=0.5):
        """
        update all variables given dx
        """
        for i in range(self.num_states):
            states[i] += step * dx[i * AUG_STATE_SIZE: i * AUG_STATE_SIZE + STATE_SIZE]

        for i in range(self.num_states - 1):
            controls[i] += step * dx[i * AUG_STATE_SIZE + STATE_SIZE: i * AUG_STATE_SIZE + AUG_STATE_SIZE]
            if p.enable_thrust_constrain and controls[i][1] < self.MIN_THRUST:
                assert len(controls[i]) == 2
                print('warning! trust constrain violated. Hard set THRUST.')
                controls[i][1] = self.MIN_THRUST + 0.01

        self.dt += step * dx[-1]

    def __newton(self, A, b, states, controls):
        """
        Newton method
        """
        dx = self.__solve_linear_syste(A, b)
        self.__update_variables(states, controls, dx, step=0.2)

    def __gradient_descent(self, neg_cost_gradient, states, controls):
        """
         Newton method
        """
        self.__update_variables(states, controls, neg_cost_gradient, step=0.001)

    def __solve_linear_syste(self, A, b):
        """
        Solve the Ax = b
        TODO(yimu): need a better solve
        """
        dx = np.linalg.solve(A, b)
        return dx

    def __init_states_linear(self):
        """
        init states given prior state and target state
        """
        all_states = [np.zeros([STATE_SIZE, 1]).squeeze() for i in range(self.num_states)]

        dx = self.target_state[0] - self.prior_state[0]
        dy = self.target_state[1] - self.prior_state[1]

        time = self.dt * (self.num_states - 1.)
        vx = dx / time
        vy = dy / time

        all_states[0] = self.prior_state.copy()
        all_states[self.num_states - 1] = self.target_state.copy()

        for i in range(1, self.num_states - 1):
            t = i / (self.num_states - 1.)

            delta = np.array([
                t * dx,
                t * dy,
                vx,
                vy,
                0.,
            ])

            all_states[i] = self.prior_state + delta
        return all_states

    def __init_states_circles(self, star, num_circles=2):
        """
        Curve line initialization
        """
        s_p = star.position
        p_x, p_y = s_p

        init_x, init_y = self.prior_state[:2]

        dist = norm_l2(self.prior_state[:2] - s_p)
        theta_0 = math.atan2(init_y - p_y, init_x - p_x)

        all_angle = - num_circles * (2 * np.pi)

        all_states = [np.zeros([STATE_SIZE, 1]).squeeze() for i in range(self.num_states)]

        all_states[0] = self.prior_state
        all_states[self.num_states - 1] = self.target_state

        for i in range(1, self.num_states - 1):
            t = 1 - i / (self.num_states - 1.)

            circle_x = cos(t * all_angle + theta_0) * dist * t + p_x
            circle_y = sin(t * all_angle + theta_0) * dist * t + p_y

            all_states[i] = compute_in_orbit_state(star, np.array([circle_x, circle_y]))

        return all_states

    def __init_controls(self):
        """
        init control variables
        """
        controls = [np.zeros([CONTROL_SIZE, 1]).squeeze() for i in range(self.num_states)]
        return controls

    def __regularization(self, A, b, states, controls):
        """
        Apply regularization
        """
        theta_dot_weight = p.regularization_theta_dot_weight
        thurst_weight = p.regularization_thurst_weight
        dt_weight = p.regularization_dt_weight * self.num_states

        VX_WEIGHT = 1e-3
        VY_WEIGHT = 1e-3

        A_reg = A.copy()
        b_reg = b.copy()

        dt_index = self.num_states * AUG_STATE_SIZE
        A_reg[dt_index, dt_index] += dt_weight
        b_reg[dt_index] -= dt_weight * self.dt

        for i in range(self.num_states):
            theta_dot_idx = i * AUG_STATE_SIZE + THETA_DOT_IDX
            trust_idx = i * AUG_STATE_SIZE + TRUST_IDX
            vx_idx = i * AUG_STATE_SIZE + VX_IDX
            vy_idx = i * AUG_STATE_SIZE + VY_IDX

            theta_dot, thrust = controls[i]
            x, y, v_x, v_y, theta = states[i]

            A_reg[theta_dot_idx, theta_dot_idx] += theta_dot_weight
            b_reg[theta_dot_idx] -= theta_dot_weight * theta_dot

            A_reg[trust_idx, trust_idx] += thurst_weight
            b_reg[trust_idx] -= thurst_weight * thrust

            A_reg[vx_idx, vx_idx] += VX_WEIGHT
            b_reg[vx_idx] -= VX_WEIGHT * v_x

            A_reg[vy_idx, vy_idx] += VY_WEIGHT
            b_reg[vy_idx] -= VY_WEIGHT * v_y

        return A_reg, b_reg

    def __apply_constrains(self, A, b, states, controls, iters):
        """
        constrain thrust to be positive
        """
        LOG_BARRIER_P = 1. / (iters + 1) ** 1.

        A_reg = A.copy()
        b_reg = b.copy()

        log_barrier_val = 0
        # Regularization for controls
        for i in range(self.num_states - 1):
            theta_dot, thrust = controls[i]
            trust_idx = i * AUG_STATE_SIZE + TRUST_IDX

            MAX_NEG_TRUST = - 0.01

            u_scale = 30
            u = u_scale * (thrust - MAX_NEG_TRUST)

            if u <= 0.1:
                print('Warning! small u')
                u = 0.1
            log_barrier_val += -math.log(u)
            A_reg[trust_idx, trust_idx] += LOG_BARRIER_P * u_scale ** 2 * 1 / u ** 2
            b_reg[trust_idx] -= - LOG_BARRIER_P * u_scale * 1 / u

        return A_reg, b_reg

    def __construct_least_square(self, states, controls):
        """
        compute the J,r,W for a least square problem
        """
        num_variables = self.num_states * AUG_STATE_SIZE + 1
        time_var_time = self.num_states * AUG_STATE_SIZE
        num_residual_equations = (self.num_states - 1) * EarthMotionWithControlResidual.size_of_residual() \
                                 + EarthPriorResidual.size_of_residual() \
                                 + EarthTargetState.size_of_residual()

        jacobian = np.zeros([num_residual_equations, num_variables])
        weights = np.identity(num_residual_equations)
        residual = np.zeros([num_residual_equations, 1]).squeeze()

        residual_idx = 0
        # construct motion residuals
        # No equations for last control. But it doesn't hurt.
        for i in range(self.num_states - 1):
            motion = EarthMotionWithControlResidual(states[i], states[i + 1],
                                                    controls[i], controls[i + 1],
                                                    dt=self.dt, accl_field=self.accl_field)
            rsize = motion.size_of_residual()

            cur_state_idx = i * AUG_STATE_SIZE
            next_state_idx = (i + 1) * AUG_STATE_SIZE
            ridx = i * rsize

            # The jacobian wrt cur state and cur control
            jacobian[ridx: ridx + rsize, cur_state_idx: cur_state_idx + AUG_STATE_SIZE] \
                = motion.jacobian_residual_wrt_current_state_and_control()
            jacobian[ridx: ridx + rsize, time_var_time] = motion.jacobian_residual_wrt_dt()

            # The jacobian wrt next state
            jacobian[ridx: ridx + rsize, next_state_idx: next_state_idx + AUG_STATE_SIZE] \
                = motion.jacobian_residual_wrt_next_state()

            weights[ridx: ridx + rsize, ridx: ridx + rsize] = motion.weight()

            residual[ridx: ridx + rsize] = motion.motion_constrain_residual()

            residual_idx += rsize

        # prior state residual
        rpsize = EarthPriorResidual.size_of_residual()
        prior = EarthPriorResidual(states[0], prior_state=self.prior_state)
        jacobian[residual_idx: residual_idx + rpsize, 0: 0 + STATE_SIZE] \
            = prior.jacobian_wrt_state()
        weights[residual_idx: residual_idx + rpsize, residual_idx: residual_idx + rpsize] = prior.weight()
        residual[residual_idx: residual_idx + rpsize] = prior.residual()
        residual_idx += rpsize

        # target state residual
        rtsize = EarthTargetState.size_of_residual()
        last_state_idx = self.num_states - 1
        last_state_idx_flatten = last_state_idx * AUG_STATE_SIZE
        target = EarthTargetState(states[last_state_idx], controls[last_state_idx]
                                  , final_state=self.target_state, final_control=self.target_control)
        jacobian[residual_idx: residual_idx + rtsize, last_state_idx_flatten: last_state_idx_flatten + AUG_STATE_SIZE] \
            = target.jacobian_wrt_state()
        weights[residual_idx: residual_idx + rtsize, residual_idx: residual_idx + rtsize] = target.weight()
        residual[residual_idx: residual_idx + rtsize] = target.residual()
        residual_idx += rtsize

        assert residual_idx == num_residual_equations

        if p.print_residual:
            abs_res = np.abs(residual)
            m_idx = (self.num_states - 1) * AUG_STATE_SIZE
            m_res = abs_res[:m_idx].reshape(self.num_states - 1, AUG_STATE_SIZE)
            print('motion residual: ', np.sum(m_res, axis=0))
            print('prior residual: ', residual[-10: -5])
            print('target residual: ', residual[-5:])
            print('total residual cost: ', np.dot(np.dot(residual.T, weights), residual))

        return jacobian, weights, residual


def compute_in_orbit_state(star, init_xy):
    init_x, init_y = init_xy

    accl_field = GravitationalField([star])
    accl = accl_field.acceleration(init_xy)

    r = star.position - init_xy
    dist = norm_l2(r)
    w = - math.sqrt(norm_l2(accl) / dist)

    rx, ry = r
    v_3d = np.cross(np.array([0, 0, w]), np.array([rx, ry, 0]))
    v_x, v_y, v_z = v_3d
    assert v_z == 0

    return np.array([
        init_x,
        init_y,
        v_x,
        v_y,
        0,
    ])
