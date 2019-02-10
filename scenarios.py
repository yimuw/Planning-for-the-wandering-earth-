import art
import numpy as np
import wandering_earth as we


def generate_trajectory_demo():
    init_state = np.array([
        0.,
        0,
        0,
        0,
        0,
    ])

    NUM_STATES = 200

    CONTROL_THETA_DOT = 0.01 * we.DEGREE
    CONTROL_TRUST = 0.01
    control_vars = np.vstack([
        CONTROL_THETA_DOT * np.ones([1, NUM_STATES]),
        CONTROL_TRUST * np.ones([1, NUM_STATES]),
    ])

    state = init_state
    all_states = [state.copy()]
    for i in range(NUM_STATES):
        control_var = control_vars[:, i]

        motion_model = we.EarthMotionWithControlResidual(
            cur_state=state, next_state=None,
            cur_control=control_var, next_control=None, dt=60)

        state = motion_model.motion()
        all_states.append(state.copy())

    we.plot_trajectory(all_states)


def wandering_earth_era_3_4_planning():
    init_state = np.array([
        0.,
        0,
        0,
        0,
        0.01,
    ])

    target_state = np.array([
        100.,
        0,
        1,
        0,
        0,
    ])

    print('init_state:', init_state)
    print('target_state:', target_state)


    target_control = np.array([0, 0.])

    accl_field = we.GravitationalField([])

    planning_problem = we.EarthTrajectoryLeastSquares(init_state, target_state, target_control,
                                          num_state=100, dt=2, accl_field=we.GravitationalField([]))

    states, controls, dt = planning_problem.least_square_optimization()

    art.trajectory_movie(states, controls, dt, accl_field)


def wandering_earth_era2_jupyter_planning():
    jupyter = we.Star(position=np.array([30, 1]), mass=5.)
    accl_field = we.GravitationalField([jupyter])

    init_state = np.array([
        0.,
        0,
        0,
        0,
        0,
    ])
    target_state = np.array([
        100.,
        0,
        1,
        0,
        0.,
    ])

    target_control = np.array([0, 0.])

    print('init_state:', init_state)
    print('target_state:', target_state)

    planning_problem = we.EarthTrajectoryLeastSquares(init_state, target_state, target_control,
                                          num_state=100, dt=0.5, accl_field=accl_field)

    states, controls, dt = planning_problem.least_square_optimization()

    art.trajectory_movie(states, controls, dt, accl_field)


def wandering_earth_era2_planning():
    sun = we.Star(position=np.array([0, 0.]), mass=5.)
    accl_field = we.GravitationalField([sun])
    init_xy = np.array([0, 5])

    init_state = we.compute_in_orbit_state(sun, init_xy)

    target_state = np.array([
        100.,
        0,
        1,
        0,
        0,
    ])

    print('init_state:', init_state)
    print('target_state:', target_state)

    target_control = np.array([0, 0.])

    planning_problem = we.EarthTrajectoryLeastSquares(init_state, target_state, target_control,
                                          num_state=100, dt=0.5, accl_field=accl_field)

    states, controls, dt = planning_problem.least_square_optimization()

    art.trajectory_movie(states, controls, dt, accl_field)


def wandering_earth_era_5_planning():
    alpha = we.Star(position=np.array([0, 0.]), mass=5)

    accl_field = we.GravitationalField([alpha])

    final_xy = np.array([0, 10]) + alpha.position

    init_state = np.array([
        100.,
        0,
        -1,
        0,
        np.pi,
    ])

    target_state = we.compute_in_orbit_state(alpha, final_xy)
    target_control = np.array([0, 0.])

    print('init_state:', init_state)
    print('target_state:', target_state)

    planning_problem = we.EarthTrajectoryLeastSquares(init_state, target_state, target_control,
                                          num_state=100, dt=0.5, accl_field=accl_field)

    states, controls, dt = planning_problem.least_square_optimization()

    art.trajectory_movie(states, controls, dt, accl_field)


if __name__ == '__main__':
    wandering_earth_era2_planning()

    # wandering_earth_era2_jupyter_planning()
    #
    # wandering_earth_era_3_4_planning()
    #
    # wandering_earth_era_5_planning()

    pass
