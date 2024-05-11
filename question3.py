import numpy as np
import scipy
import scipy.stats as sci
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from confidence_ellipse import nicegrid

g = 9.8
mat_A = np.array([  [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
mat_B = np.eye(4)
mat_C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]
                  ])
u_t = np.array([0, 0, 0, -g])
x_initial = np.array([0, 0, 100, 100]).T
P_initial = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 25, 5],
                      [0, 0, 5, 25]
                      ])
Q = np.array([[20, 0, 5, 0],
             [0, 20, 0, 5],
             [5, 0 ,10 ,0],
             [0, 5, 0, 10]]
             )
R = np.array([ [9,3],
             [3,9]
             ])
I = np.eye(4)
# Generate samples for w_t and r_t
num_samples = 100
w_t_samples = np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q, size=num_samples).T
r_t_samples = np.random.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R, size=num_samples).T

#Part A
# Initialize variables
T = 21 # Total time steps
ground_truth = np.zeros((4, T+1))

# Generate ground truth trajectory using x_t=1 = Ax_t + Bu_t + w_t
for t in range(T+1):
    if t == 0:
        ground_truth[:, t] = x_initial
    else:
        ground_truth[:, t] = mat_A @ ground_truth[:, t-1] + mat_B @ u_t + w_t_samples[:, t]

# Plot the ground truth trajectory 
plt.figure(figsize=(10, 6))
plt.plot(ground_truth[0, :], ground_truth[1, :], label = "x_t+1: ground truth trajectory")
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Projectile Trajectory')
plt.legend()
plt.grid(True)
plt.show()


#Part B
radar_obs = np.zeros((2, T+1))
for t in range(1, T+1):
    radar_obs[:, t] = mat_C @ ground_truth[:, t] + r_t_samples[:, t]
plt.figure(figsize=(10, 6))
plt.plot(radar_obs[0, :], radar_obs[1, :], label = "y_t: radar observations", color = "purple")
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Radar Observations')
plt.legend()
plt.grid(True)
plt.show()


#Part C
new_T = 10
x_hat = np.zeros((4, new_T+1))
P = np.zeros((4, 4, new_T+1))
x_hat[:, 0] = x_initial
P[:, :, 0] = P_initial

for t in range(1, new_T+1):
    # Prediction step
    x_hat[:, t] = mat_A @ x_hat[:, t-1] + mat_B @ u_t
    P[:, :, t] = mat_A @ P[:, :, t-1] @ mat_A.T + Q

    # Update step
    K = P[:, :, t] @ mat_C.T @ np.linalg.inv(mat_C @ P[:, :, t] @ mat_C.T + R)
    x_hat[:, t] = x_hat[:, t] + K @ (radar_obs[:, t] - mat_C @ x_hat[:, t])
    P[:, :, t] = (I - K @ mat_C) @ P[:, :, t]

final_xhat = x_hat[:,new_T]
final_P = P[:,:,new_T]
print("The Kalman Filter shows the following data for the position of the projectile after t=10 seconds:")
print("Updated State(x_t+1|t+1): ", final_xhat)
print("Updated Covariance(P_t+1|t+1): ", final_P)

#Part D




