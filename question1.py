import numpy as np

#Part C#
delta_t = 5
g = 9.8
mat_A = np.array([[1, 0, delta_t, 0],
         [0, 1, 0, delta_t],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         ])
mat_B = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
         ])
mat_C = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0]
         ])
u_t = np.array([0,0, 0, -g*delta_t]).T
       
x_initial = np.array([0, 0, 100, 100]).T 

P_initial = np.array([[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 25, 5],
         [0, 0, 5, 25]
         ])


def calc_prior_mean(x_t, u_t, t):
    if t == 0:
        pass
    pass

def calc_prior_cov(P_new):
    pass

print("asnwer")
