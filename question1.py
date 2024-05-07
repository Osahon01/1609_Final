import numpy as np

#Part C#
delta_t = 5
g = 9.8
mat_A = [[1, 0, delta_t, 0],
         [0, 1, 0, delta_t],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         ]
mat_B = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]
         ]
mat_C = [[1, 0, 0, 0],
         [0, 1, 0, 0]
         ]
u_t = [[0],
       [0],
       [0],
       [-g*delta_t]
       ]

def calc_priors(x_t, u_t, r_t):
    pass

print("asnwer")
