import numpy as np

#Part C#

g = 9.8
mat_B = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
         ])
mat_C = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0]
         ])

       
x_initial = np.array([0, 0, 100, 100]).T 

P_initial = np.array([[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 25, 5],
         [0, 0, 5, 25]
         ])


def calc_prior_mean(x_t, u_t, t):
    if t == 0:
        return (x_initial, np.array([0,0,0,0]).T)

    else:
        mat_A = np.array([[1, 0, t, 0],
         [0, 1, 0, t],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         ])
        u_t = np.array([0,0, 0, -g*t]).T
        return mat_A @ calc_prior_mean(x_t, u_t, t-1)[0] + mat_B @ calc_prior_mean(x_t, u_t, t-1) 

def calc_prior_cov(P_new):
    pass

print("asnwer")
