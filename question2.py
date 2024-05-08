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
mean = calc_prior_mean(mat_B, x_initial, u_initial, t_chosen) 
Cov = calc_prior_cov(mat_B, P_initial, Q, t_chosen)

# Extract the submatrix for the first two elements of mean and covariance
mean_2 = mean[0]
Cov_2 = Cov[0][-1]  # Fix: Extract the last element of the first row of Cov
print(Cov_2)

norm = scipy.stats.multivariate_normal(mean=mean_2, cov=Cov_2) 
x = np.linspace(-1, 19, num=20)  # Test
pdf_values = norm.pdf(x)
integral = integrate.cumtrapz(pdf_values, x, initial=float('inf'))
print(integral)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(x, pdf_values, label='PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Probability Density Function (PDF)')
plt.grid(True)
plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(x, integral, label='Integral')
# plt.fill_between(x[:-1], integral, alpha=0.3)
# plt.xlabel('x')
# plt.ylabel('Integrated Probability')
# plt.title('Visual Integration of PDF')
# plt.grid(True)
# plt.legend()
plt.show()