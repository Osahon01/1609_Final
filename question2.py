import numpy as np
import scipy
import scipy.stats as sci
import scipy.integrate as integrate
import matplotlib.pyplot as plt


g = 9.8
mat_B = np.eye(4)
mat_C = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0]
         ])

       
x_initial = np.array([0, 0, 100, 100]).T 
u_initial = np.array([0,0,0,0]).T

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
        new_u_t = np.array([0,0,0,-g*t])

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

t_chosen = 20
# plt.show()

# Define the normal distribution for x2 at time t = 20
mean_x2 = calc_prior_mean(mat_B, x_initial, u_initial, t_chosen)[1]  # Mean of x2 at time t = 20
cov_x2 = calc_prior_cov(mat_B, P_initial, Q, t_chosen)[1][1]  # Covariance of x2 at time t = 20
norm_x2 = scipy.stats.norm(mean_x2, cov_x2)

# Calculate the cumulative distribution function (CDF) at t = 20
prob_landed = norm_x2.cdf(0)  # Probability that x2 <= 0 at t = 20

# Plot the PDF of x2 at time t = 20
x_values = np.linspace(mean_x2 - 1000, mean_x2 + 1000, 100)
pdf_values_x2 = norm_x2.pdf(x_values)

plt.figure()
plt.plot(x_values, pdf_values_x2, label='PDF of x2 at t = 20')
plt.xlabel('x2')
plt.ylabel('Probability Density')
plt.title('PDF of x2 at t = 20')
plt.legend()
plt.grid(True)
plt.show()

print("Probability that the projectile has landed by time 20:", prob_landed)
print(mean_x2)