import numpy as np

#Part C#

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
def calc_prior_cov(P_new, mat_A, t, P_initial, Q):
    if t == 0:
        A_new = mat_A
        curr = (A_new @ P_initial @ A_new.T) + Q
        return curr, t
    elif t > 0:
        P_sub, new_t = calc_prior_cov(P_new, mat_A, t - 1, P_initial, Q)
        A_sub = np.array([[1, 0, new_t, 0],
                          [0, 1, 0, new_t],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        curr = (A_sub @ P_sub @ A_sub.T) + Q
        return curr, new_t + 1
    return None, None 
print("Covariance matrix for 5|5", calc_prior_cov(P_initial, mat_B, 5, P_initial, Q))
c_mean_answer = calc_prior_mean(mat_B, x_initial,u_initial, 5)
print("Prior Mean: ", c_mean_answer)
