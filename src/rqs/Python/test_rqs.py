from galahad import rqs
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: rqs")

# set parameters
p = 1.0
n = 3
m = 1
infinity = float("inf")

#  describe objective function

f = 0.96
g = np.array([0.0,2.0,0.0])
H_type = 'coordinate'
H_ne = 4
H_row = np.array([0,1,2,2])
H_col = np.array([0,1,2,0])
H_ptr = None
H_val = np.array([1.0,2.0,3.0,4.0])

#  describe norm

M_type = 'coordinate'
M_ne = 3
M_row = np.array([0,1,2])
M_col = np.array([0,1,2])
M_ptr = None
M_val = np.array([1.0,2.0,1.0])

#  describe constraint

A_type = 'coordinate'
A_ne = 3
A_row = np.array([0,0,0])
A_col = np.array([0,1,2])
A_ptr = None
A_val = np.array([1.0,1.0,1.0])

# set regularization parameters

power = 3.0
weight = 1.0

# allocate internal data and set default options
options = rqs.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
rqs.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)

# find minimum of quadratic within the trust region
print("\n solve problem 1")
x = rqs.solve_problem(n, power, weight, f, g, H_ne, H_val)
print(" x:",x)

# get information
inform = rqs.information()
print(" f: %.4f" % inform['obj'])

# load data (and optionally non-default options)
rqs.load_m(n, M_type, M_ne, M_row, M_col, M_ptr)

# find minimum of quadratic within the trust region
print("\n solve problem 2 with additional non-unit norm")
x = rqs.solve_problem(n, power, weight, f, g, H_ne, H_val, M_ne, M_val)
print(" x:",x)

# get information
inform = rqs.information()
print(" f: %.4f" % inform['obj'])

# load data (and optionally non-default options)
rqs.load_a(m, A_type, A_ne, A_row, A_col, A_ptr)

# find minimum of quadratic within the trust region
print("\n solve problem 3 with additional linear constraint")
x, y = rqs.solve_problem(n, power, weight, f, g, H_ne, H_val,
                         M_ne, M_val, m, A_ne, A_val)
print(" x:",x)
print(" y:",y)

# get information
inform = rqs.information()
print(" f: %.4f" % inform['obj'])
print('** rqs exit status:', inform['status'])

# deallocate internal data

rqs.terminate()

