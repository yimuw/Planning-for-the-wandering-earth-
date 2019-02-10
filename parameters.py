import numpy as np


max_iter = 100

enable_thrust_constrain = True

print_residual = False

show_trajectory_in_optimization = False

plot_python_notebook = False

motion_residual_weights = np.diag([1000, 1000, 1000, 1000, 1000, 50, 50])

prior_residual_weights = np.diag([1000, 1000, 1000, 1000, 10000])

target_residual_weights = np.diag([50, 50, 100, 100, 100, 10, 10])

regularization_dt_weight = 1e-1

regularization_theta_dot_weight = 5e-1

regularization_thurst_weight = 1e1
