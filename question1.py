import numpy as np
from confidence_ellipse import confidence_ellipse, nicegrid
import matplotlib.pyplot as plt
# Part C #

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
             [0, 20, 0,5],
             [5, 0 ,10 ,0],
             [0, 5, 0, 10]]
             )


def calc_prior_mean(x_t, t):
    if t == 1:
        return mat_A @ x_t + mat_B @ u_t
    else:
        return mat_A @ calc_prior_mean(x_t, t-1) + mat_B @ u_t



def calc_prior_cov(P_t, t):
    if t == 1:
        return (mat_A @ P_t @ mat_A.T) + Q

    else:
        P_new = calc_prior_cov(P_t, t-1)
        return (mat_A @ P_new @ mat_A.T) + Q


mean_5 = calc_prior_mean(x_initial, 5)
cov_5 = calc_prior_cov(P_initial, 5)

print("Prior Mean: ", mean_5)
print("Covariance matrix for 5|0", cov_5)  # print statements to answer 1d


def plot_confidence_ellipses_d():
    alphas = [0.9, 0.95]
    colors = ['red', 'blue']
    for interval, color in zip(alphas, colors):
        confidence_ellipse(mean_5, cov_5, interval, color)


def plot_confidence_ellipses_e():
    alphas = [0.9]
    colors = ['green', 'blue']
    for time, color in zip([10, 15], colors):
        mean_time = calc_prior_mean(x_initial, time)
        cov_time = calc_prior_cov(P_initial, time)
        confidence_ellipse(mean_time, cov_time, alphas[0], color)


plt.figure(figsize=(8, 6))
plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')
plt.title('Confidence Ellipse(s) for $\\alpha$ and $t$')
nicegrid(plt.gca())

plot_confidence_ellipses_d()
plt.legend(['90% Confidence Ellipse (t=5)', '95% Confidence Ellipse (t=5)'])
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('answer_1d.svg', format='svg', bbox_inches='tight')
plt.show()

plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')
plt.title('Confidence Ellipse(s) for $\\alpha$ and $t$')
nicegrid(plt.gca())
plot_confidence_ellipses_e()
plt.legend(['90% Confidence Ellipse (t=10)', '90% Confidence Ellipse (t=15)'])
plt.savefig('answer_1e.svg', format='svg', bbox_inches='tight')
plt.show()
