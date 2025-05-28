import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

import os
import sys



# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

project_root = os.path.abspath(os.path.join(__file__, '..', '..'))

print("Project root:", project_root)
# 添加lib目录
lib_path = os.path.join(project_root, 'lib')
if lib_path not in sys.path:
    sys.path.append(lib_path)

import DE_cuda_solver

def compute_foot_yaw(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def generate_footstep_and_com_trajectory(sol_x, sol_u, n_repeat=8, start_with_left=True):
    """
    Generate left/right foot trajectories and CoM trajectory from optimization outputs.

    Returns:
        left_foot, right_foot, com
        Each is a list of length N * n_repeat + 1
    """
    left_foot = []
    right_foot = []
    com = []

    current_left = None
    current_right = None

    # yaw = sol_x[0,4]
    for i in range(len(sol_u)):
        com_xy = (sol_x[i, 0], sol_x[i, 1])
        foot_dx, foot_dy, foot_dyaw = sol_u[i]
        # absolute_foot_yaw = com_yaw + foot_dyaw
        # yaw += foot_dyaw
        yaw = sol_x[i, 4]
        foot_pos = (sol_x[i, 0] + foot_dx, sol_x[i, 1] + foot_dy, yaw)

        # Update only the swinging foot
        if (i % 2 == 0 and start_with_left) or (i % 2 == 1 and not start_with_left):
            current_left = foot_pos
        else:
            current_right = foot_pos

        for _ in range(n_repeat):
            # Fix the swing logic: only one foot changes, the other stays None
            if (i % 2 == 0 and start_with_left) or (i % 2 == 1 and not start_with_left):
                left_foot.append(current_left)
                right_foot.append(None)
            else:
                left_foot.append(None)
                right_foot.append(current_right)
            com.append(com_xy)

    # Append final foot and CoM state
    com.append((sol_x[-1, 0], sol_x[-1, 1]))
    left_foot.append(left_foot[-1])
    right_foot.append(right_foot[-1])

    return left_foot, right_foot, com



def solve_footstep_planning_with_cuda(x0_MLD, N, nx=5, nu=3):
    """
    Uses DE_cuda_solver to solve a footstep planning problem.

    Args:
        x0_MLD (np.ndarray): Initial state [x, y, vx, vy, theta]
        N (int): Planning horizon (number of steps)
        nx (int): State dimension (default 5)
        nu (int): Control dimension (default 3)

    Returns:
        sol_x (np.ndarray): State trajectory (N+1, nx)
        sol_u (np.ndarray): Control trajectory (N, nu)
        fitness (float): Solution fitness value
        solve_time (float): Runtime in seconds
    """
    print("============= Start Footstep Planning =============")

    # Create and initialize solver
    solver = DE_cuda_solver.Create()
    solver.init_solver(0)

    # Solve optimization problem
    start_time = time.time()
    solution = solver.Solve()
    solve_time = time.time() - start_time

    # Extract and reshape solution
    fitness = solution["fitness"]
    sol_x = np.reshape(solution["state"], (N+1, nx))
    sol_u = np.reshape(solution["param"], (N, nu))

    # print("Solver time:", solve_time)
    # print("Fitness:", fitness)
    # print("sol_x shape:", sol_x.shape)
    # print("sol_u shape:", sol_u.shape)
    print("============= Finish Footstep Planning =============")

    return sol_x, sol_u, fitness, solve_time