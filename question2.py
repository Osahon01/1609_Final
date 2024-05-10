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

#Part A

t_chosen = 20
mean_x1x2 = calc_prior_mean(x_initial, t_chosen)
cov_x1x2 = calc_prior_cov(P_initial,t_chosen)

# Define the normal distribution for x2 at time t = 20
x2_mean = mean_x1x2[1]  # Mean of x2 at time t = 20
x2_var = cov_x1x2[1][1]  # Variance of x2 at time t = 20
norm_x2 = scipy.stats.norm(x2_mean, x2_var**(1/2))

# Calculate the cumulative distribution function (CDF) at t = 20
prob_landed = norm_x2.cdf(0)  # Probability that x2 <= 0 at t = 20

# Plot the PDF of x2 at time t = 20
x_values = np.linspace(x2_mean - 1000, x2_mean + 1000, 100)
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
print(x2_mean)


#Part B
t_vals = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


def prob_landed_by_t(t):
    x2_mean = calc_prior_mean(x_initial, t)[1]
    x2_var = calc_prior_cov(P_initial, t)[1][1]
    norm_x2 = scipy.stats.norm(x2_mean, x2_var**(1/2))
    return norm_x2.cdf(0)


# Calculate the CDF values for each t in t_vals
cdf_vals = [prob_landed_by_t(t) for t in t_vals]

# Compute the PMF of Tland using the differences in CDF values
pmf_vals = [cdf_vals[0]] + [max(0, cdf_vals[i] - cdf_vals[i - 1])
                            for i in range(1, len(cdf_vals))]

# Print the PMF values
print("CDF of Tland:", cdf_vals)
print("PMF of Tland:", pmf_vals)


plt.figure()
plt.bar(t_vals, pmf_vals, width=0.5, align='center', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('PMF of Landing Time')
plt.xticks(t_vals)
# plt.grid(axis='y', alpha=0.75)
nicegrid(plt.gca())
plt.show()


#Part C
x1_mean = mean_x1x2[0]
x1_var = cov_x1x2[0][0]
x1x2_corr = cov_x1x2[0][1]/ (x1_var*x2_var)**(1/2)
c_mean_ans = x1_mean+(x1_var/x2_var)**(1/2)*x1x2_corr*-x2_mean
c_var_ans = (1-x1x2_corr**2)*x1_var
print("Mean Answer for Part C: ", c_mean_ans)
print("Var Answer to Part C: ", c_var_ans)
