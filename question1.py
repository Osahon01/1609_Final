import numpy as np
from confidence_ellipse import confidence_ellipse, nicegrid
import matplotlib.pyplot as plt
# Part C #

g = 9.8
mat_B = np.eye(4)
mat_C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]
                  ])


x_initial = np.array([0, 0, 100, 100]).T
u_initial = np.array([0, 0, 0, 0]).T

P_initial = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 25, 5],
                      [0, 0, 5, 25]
                      ])


def calc_prior_mean(A_t, x_t, u_t, t):
    if t == 0:
        return A_t @ x_t + mat_B @ u_t

    else:
        mat_A = np.array([[1, 0, t, 0],
                          [0, 1, 0, t],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        new_u_t = np.array([0, 0, 0, -g*t])

        return mat_A @ calc_prior_mean(A_t, x_t, new_u_t, t-1) + mat_B @ new_u_t


Q = np.eye(4)


def calc_prior_cov(A_t, P_t, Q, t):
    if t == 0:
        return (A_t @ P_t @ A_t.T) + Q

    else:
        mat_A = np.array([[1, 0, t, 0],
                          [0, 1, 0, t],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        P_new = calc_prior_cov(mat_A, P_t, Q, t-1)
        return (mat_A @ P_new @ mat_A.T) + Q


mean_5 = calc_prior_mean(mat_B, x_initial, u_initial, 5)
cov_5 = calc_prior_cov(mat_B, P_initial, Q, 5)


def plot_confidence_ellipses_d():
    alphas = [0.9, 0.95]
    colors = ['red', 'blue']
    for interval, color in zip(alphas, colors):
        confidence_ellipse(mean_5, cov_5, interval, color)


print("Prior Mean: ", mean_5)
print("Covariance matrix for 5|0", cov_5)


def plot_confidence_ellipses_e():
    alphas = [0.9]
    colors = ['green', 'blue']
    for time, color in zip([10, 15], colors):
        mean_time = calc_prior_mean(mat_B, x_initial, u_initial, time)
        cov_time = calc_prior_cov(mat_B, P_initial, Q, time)
        confidence_ellipse(mean_time, cov_time, alphas[0], color)


plt.figure(figsize=(8, 6))
plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')
plt.title('Confidence Ellipse(s) for $\\alpha$ and $t$')
plt.gca().set_aspect('equal', adjustable='box')
nicegrid(plt.gca())

plot_confidence_ellipses_d()
plt.legend(['90% Confidence Ellipse (t=5)', '95% Confidence Ellipse (t=5)'])
plt.show()

plt.savefig('answer_1d.svg', format='svg')


plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')
plt.title('Confidence Ellipse(s) for $\\alpha$ and $t$')
nicegrid(plt.gca())
plot_confidence_ellipses_e()
plt.legend(['90% Confidence Ellipse (t=10)', '90% Confidence Ellipse (t=15)'])
plt.show()
plt.savefig('answer_1e.svg', format='svg')  # Save the plot for part e
