import os
from math import cos, sin

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import parameters as p
import pylab


def save_cur_figure(dir, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = os.path.join(dir, file_name)
    plt.savefig(path)


class Artist:
    def __init__(self):
        fire_im = plt.imread('art_files/blue_fire.png')

        # may need filter
        self.fire_im = fire_im[::5, ::5, :]
        self.earth_im = plt.imread('art_files/images.jpeg')

    def draw_at(self, x, y, theta_raw, fire_len_scale, scale=1.):
        theta = theta_raw - np.pi / 2.
        im_fire = plt.imshow(self.fire_im)
        flen_y, flen_x, _ = self.fire_im.shape
        trans_data = mtransforms.Affine2D().translate(-flen_x / 2., -flen_y * 0.8) \
                     + mtransforms.Affine2D().scale(0.25 * scale, 0.5 * scale * fire_len_scale) \
                     + mtransforms.Affine2D().rotate(theta) \
                     + mtransforms.Affine2D().translate(x, y)

        trans0 = im_fire.get_transform()
        trans_data = trans_data + trans0
        im_fire.set_transform(trans_data)

        im_earth = plt.imshow(self.earth_im)
        elen_y, elen_x, _ = self.earth_im.shape
        trans_data = \
            mtransforms.Affine2D().translate(-elen_x / 2., -elen_y / 2.) \
            + mtransforms.Affine2D().scale(0.1 * scale) \
            + mtransforms.Affine2D().rotate(theta) \
            + mtransforms.Affine2D().translate(x, y)

        trans0 = im_earth.get_transform()
        trans_data = trans_data + trans0
        im_earth.set_transform(trans_data)


def plot_trajectory(states, controls, is_movie=False):
    """
    plot trajectory
    """
    states_stacked = np.vstack(states)
    controls_stacked = np.vstack(controls)

    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(states_stacked[:, 0], states_stacked[:, 1], '-*')
    plt.title('x,y')
    plt.axis('equal')

    plt.subplot(2, 2, 2)
    plt.plot(np.linalg.norm(states_stacked[:, 2:4], axis=1), '-')
    plt.title('speed')

    plt.subplot(2, 2, 3)
    plt.plot(controls_stacked)
    plt.title('control v')

    plt.subplot(2, 2, 4)
    plt.plot(states_stacked[:, 4], 'r')
    plt.title('red : theta')

    if is_movie:
        plt.pause(0.05)
    else:
        plt.show()


def plot_circle(ax, mu, r, color='b'):
    t = np.linspace(0, 2 * np.pi, 100)
    circle = r * np.vstack([np.cos(t), np.sin(t)])
    eclipse = circle
    ax.plot(eclipse[0, :] + mu[0], eclipse[1, :] + mu[1], color=color)


def trajectory_movie(states, controls, dt, gravitational):
    """
    Plotting method for CLI doesn't work for python notebook.
    Plotting method for python notebook doesn't work for CLI.
    """

    if p.plot_python_notebook:
        trajectory_movie_for_python_notebook(states, controls, dt, gravitational)
    else:
        trajectory_movie_for_normal_python(states, controls, dt, gravitational)


def trajectory_movie_for_normal_python(states, controls, dt, gravitational):
    """
    show a trajectory moive
    """
    states_stacked = np.vstack(states)
    max_x = np.max(states_stacked[:, 0])
    max_y = np.max(states_stacked[:, 1])
    min_x = np.min(states_stacked[:, 0])
    min_y = np.min(states_stacked[:, 1])
    max_diff = max(max_x - min_x, max_y - min_y)

    controls_stacked = np.vstack(controls)

    max_thurst = max(controls_stacked[:, 1])

    speed = np.linalg.norm(states_stacked[:, 2:4], axis=1)
    max_speed = max(speed)

    total_control = np.sum(np.abs(controls_stacked), axis=0)
    sum_theta_dot, sum_thrust = total_control
    # print('sum_theta_dot, sum_thrust :', sum_theta_dot, sum_thrust)

    scale = 100  # (states[-1] - states[0])[0] / 30

    g_max = 0
    for i in range(len(states)):
        x, y, v_x, v_y, theta = states[i]
        g = gravitational.acceleration(np.array([x, y]))
        gn = np.linalg.norm(g)
        g_max = max(g_max, gn)

    for i in range(len(states)):
        plt.clf()
        time = i * dt
        state = states[i]
        control = controls[i]
        x, y, v_x, v_y, theta = state
        theta_dot, thrust = control
        thrust_x = cos(theta) * thrust
        thrust_y = sin(theta) * thrust

        gx, gy = gravitational.acceleration(np.array([x, y]))

        ax1 = plt.subplot(2, 3, 1)
        plt.plot(states_stacked[:, 0], states_stacked[:, 1], '-o', alpha=0.2)
        for s in gravitational.stars:
            plot_circle(ax1, s.position, 3, 'r')
        plot_circle(ax1, (x, y), 3, 'b')
        plt.axis('equal')
        plt.autoscale(False)
        # plt.xlim([-20, 120])
        # plt.ylim([-40, 40])

        if abs(thrust) > 0:
            plt.arrow(x - thrust_x * scale, y - thrust_y * scale,
                      thrust_x * scale, thrust_y * scale,
                      width=1.)
            # earth_drawer.draw_at(x, y, theta, fire_len_scale=thrust * 6, scale=0.5)
        plt.title('trajectory idx: {} time :{:10.4f} sec'.format(i, time))

        plt.subplot(2, 3, 2)
        plt.plot(controls_stacked[:, :], 'g', alpha=0.1)
        plt.plot(controls_stacked[:i, 0], 'r', label='theta_dot')
        plt.plot(controls_stacked[:i, 1], 'b', label='thurst')
        pylab.legend(loc='upper left')
        plt.xlabel('index')
        plt.ylabel('thrust / theta_dot')
        plt.title('control')

        plt.subplot(2, 3, 3)
        plt.plot(states_stacked[:, 2:4], 'g', alpha=0.1)
        plt.plot(states_stacked[:i, 2], 'r', label='vx')
        plt.plot(states_stacked[:i, 3], 'b', label='vy')
        pylab.legend(loc='upper left')
        plt.title('speed')
        plt.xlabel('idx')
        plt.ylabel('meter/sec')

        plt.subplot(2, 3, 4)
        max_acc = max(max_thurst, g_max)
        plt.plot([0, thrust_x], [0, thrust_y], alpha=0.5, color='r', label='thrust')
        plt.scatter(thrust_x, thrust_y, color='r')
        plt.plot([0, gx], [0, gy], alpha=0.5, color='g', label='gravitation')
        plt.scatter(gx, gy, color='g')
        plt.xlim([-max_acc, max_acc])
        plt.ylim([-max_acc, max_acc])
        pylab.legend(loc='upper left')
        plt.title('gravity/thrust')

        plt.subplot(2, 3, 5)
        plt.plot([0, thrust_x + gx], [0, thrust_y + gy], alpha=0.5, label='thrust')
        plt.scatter(thrust_x + gx, thrust_y + gy)
        max_acc = max(max_thurst, g_max)
        plt.xlim([-max_acc, max_acc])
        plt.ylim([-max_acc, max_acc])
        plt.title('acceleration')

        plt.subplot(2, 3, 6)
        plt.plot([0, v_x], [0, v_y], alpha=0.5)
        plt.scatter(v_x, v_y)
        plt.xlim([-max_speed, max_speed])
        plt.ylim([-max_speed, max_speed])
        plt.title('speed vector')
        plt.xlabel('vx')
        plt.ylabel('vy')

        plt.pause(0.05)

        # save_cur_figure('test', '{}.png'.format(i))
    plt.show()


def trajectory_movie_for_python_notebook(states, controls, dt, gravitational):
    """
    show a trajectory moive
    """
    states_stacked = np.vstack(states)
    max_x = np.max(states_stacked[:, 0])
    max_y = np.max(states_stacked[:, 1])
    min_x = np.min(states_stacked[:, 0])
    min_y = np.min(states_stacked[:, 1])
    max_diff = max(max_x - min_x, max_y - min_y)

    controls_stacked = np.vstack(controls)

    max_thurst = max(controls_stacked[:, 1])

    speed = np.linalg.norm(states_stacked[:, 2:4], axis=1)
    max_speed = max(speed)

    total_control = np.sum(np.abs(controls_stacked), axis=0)
    sum_theta_dot, sum_thrust = total_control
    # print('sum_theta_dot, sum_thrust :', sum_theta_dot, sum_thrust)

    scale = 100  # (states[-1] - states[0])[0] / 30

    g_max = 0
    for i in range(len(states)):
        x, y, v_x, v_y, theta = states[i]
        g = gravitational.acceleration(np.array([x, y]))
        gn = np.linalg.norm(g)
        g_max = max(g_max, gn)

    earth_drawer = Artist()

    fig_planal = plt.figure()
    ax1 = fig_planal.add_subplot(321)
    ax2 = fig_planal.add_subplot(322)
    ax3 = fig_planal.add_subplot(323)
    ax4 = fig_planal.add_subplot(324)
    ax5 = fig_planal.add_subplot(325)
    ax6 = fig_planal.add_subplot(326)

    fig_earth = plt.figure()
    axe = fig_earth.add_subplot(111)

    axis = [ax1, ax2, ax3, ax4, ax5, ax6, axe]

    plt.ion()
    fig_planal.show()
    fig_earth.show()

    for i in range(len(states)):

        for ax in axis:
            ax.clear()

        time = i * dt
        state = states[i]
        control = controls[i]
        x, y, v_x, v_y, theta = state
        theta_dot, thrust = control
        thrust_x = cos(theta) * thrust
        thrust_y = sin(theta) * thrust

        gx, gy = gravitational.acceleration(np.array([x, y]))

        axe.plot(states_stacked[:, 0], states_stacked[:, 1], '-o', alpha=0.2)
        for s in gravitational.stars:
            plot_circle(axe, s.position, 3, 'r')
        plot_circle(axe, (x, y), 2, 'b')
        # plt.axis('equal')
        # plt.autoscale(False)
        axe.set_xlim([-20, 120])
        axe.set_ylim([-40, 40])

        if abs(thrust) > 0:
            axe.arrow(x - thrust_x * scale, y - thrust_y * scale,
                      thrust_x * scale, thrust_y * scale,
                      width=1.)
            # earth_drawer.draw_at(x, y, theta, fire_len_scale=thrust * 6, scale=0.5)
        axe.set_title('trajectory idx: {} time :{:10.4f} sec'.format(i, time))

        ax1.plot(states_stacked[:, 4], 'g', alpha=0.1)
        ax1.plot(states_stacked[:i, 4], 'r', label='theta')
        ax1.legend(loc='upper left')
        ax1.set_title('theta')
        ax1.set_xlabel('idx')
        ax1.set_ylabel('rad/sec')

        ax2.plot(controls_stacked[:, 0:2], 'g', alpha=0.1)
        ax2.plot(controls_stacked[:i, 0], 'r', label='theta_dot')
        ax2.plot(controls_stacked[:i, 1], 'b', label='thurst')
        ax2.legend(loc='upper left')
        ax2.set_xlabel('index')
        ax2.set_ylabel('thrust / theta_dot')
        ax2.set_title('control')

        ax3.plot(states_stacked[:, 2:4], 'g', alpha=0.1)
        ax3.plot(states_stacked[:i, 2], 'r', label='vx')
        ax3.plot(states_stacked[:i, 3], 'b', label='vy')
        ax3.legend(loc='upper left')
        ax3.set_title('speed')
        ax3.set_xlabel('idx')
        ax3.set_ylabel('meter/sec')

        max_acc = max(max_thurst, g_max)
        ax4.plot([0, thrust_x], [0, thrust_y], alpha=0.5, color='r', label='thrust')
        ax4.scatter(thrust_x, thrust_y, color='r')
        ax4.plot([0, gx], [0, gy], alpha=0.5, color='g', label='gravitation')
        ax4.scatter(gx, gy, color='g')
        ax4.set_xlim([-max_acc, max_acc])
        ax4.set_ylim([-max_acc, max_acc])
        # pylab.legend(loc='upper left')
        ax4.set_title('gravity/thrust')

        ax5.plot([0, thrust_x + gx], [0, thrust_y + gy], alpha=0.5, label='thrust')
        ax5.scatter(thrust_x + gx, thrust_y + gy)
        max_acc = max(max_thurst, g_max)
        ax5.set_xlim([-max_acc, max_acc])
        ax5.set_ylim([-max_acc, max_acc])
        ax5.set_title('acceleration')

        ax6.plot([0, v_x], [0, v_y], alpha=0.5)
        ax6.scatter(v_x, v_y)
        ax6.set_xlim([-max_speed, max_speed])
        ax6.set_ylim([-max_speed, max_speed])
        ax6.set_title('speed vector')
        ax6.set_xlabel('vx')
        ax6.set_ylabel('vy')

        # save_cur_figure('test', '{}.png'.format(i))

        fig_planal.canvas.draw()
        fig_earth.canvas.draw()
    plt.show()
