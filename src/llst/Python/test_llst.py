from galahad import llst
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: llst")

# set parameters
n = 3
m = 1

#  describe objective function

A_type = 'coordinate'
A_ne = 3
A_row = np.array([0,0,0])
A_col = np.array([0,1,2])
A_ptr = None
A_val = np.array([1.0,1.0,1.0])
b = np.array([1.0])

#  describe norm

S_type = 'coordinate'
S_ne = 3
S_row = np.array([0,1,2])
S_col = np.array([0,1,2])
S_ptr = None
S_val = np.array([1.0,2.0,1.0])

# set trust-region radius

radius = 0.1

# allocate internal data and set default options
options = llst.initialize()

# set some non-default options
options['print_level'] = 0
options['definite_linear_solver'] = 'sytr '
#print("options:", options)

# load data (and optionally non-default options)
llst.load(m, n, A_type, A_ne, A_row, A_col, A_ptr, options)

# find minimum of linear least-squares objective within the trust region
print("\n solve problem 1")
x = llst.solve_problem(m, n, radius, A_ne, A_val, b)
print(" x:",x)

# get information
inform = llst.information()
print(" ||r||: %.4f" % inform['r_norm'])

# load data (and optionally non-default options)
llst.load_scaling(n, S_type, S_ne, S_row, S_col, S_ptr)

# find minimum of linear least-squares objective within the trust region
print("\n solve problem 2 with additional non-unit scaling")
x = llst.solve_problem(m, n, radius, A_ne, A_val, b, S_ne, S_val)
print(" x:",x)

# get information
inform = llst.information()
print(" ||r||: %.4f" % inform['r_norm'])

# deallocate internal data

llst.terminate()

