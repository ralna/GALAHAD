from galahad import nrek
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: nrek")

# set parameters
p = 1.0
n = 3
m = 1
infinity = float("inf")

#  describe objective function

f = 0.96
c = np.array([0.0,2.0,0.0])
H_type = 'coordinate'
H_ne = 4
H_row = np.array([0,1,2,2])
H_col = np.array([0,1,2,0])
H_ptr = None
H_val = np.array([1.0,2.0,3.0,4.0])

#  describe norm

S_type = 'coordinate'
S_ne = 3
S_row = np.array([0,1,2])
S_col = np.array([0,1,2])
S_ptr = None
S_val = np.array([1.0,2.0,1.0])

# allocate internal data and set default options
options = nrek.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
nrek.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)

# set regularization power and weight

power = 3.0
weight = 1.0

# find minimum of the norm-regularized quadratic
print("\n solve problem 1")
x = nrek.solve_problem(n, H_ne, H_val, c, power, weight)
print(" x:",x)

# get information
inform = nrek.information()
print(" f: %.4f" % inform['obj'])

# reset regularization weight to the suggested larger value

weight = inform['next_weight']
options['new_weight'] = True
nrek.reset_options(options)

# find minimum of the norm-regularized quadratic
print("\n solve problem 2 with smaller weight")
x = nrek.solve_problem(n, H_ne, H_val, c, power, weight)
print(" x:",x)

# get information
inform = nrek.information()
print(" f: %.4f" % inform['obj'])

# reinitialize the regularization weight

weight = 1.0
options['new_weight'] = False
nrek.reset_options(options)

# load data (and optionally non-default options)
nrek.load_s(n, S_type, S_ne, S_row, S_col, S_ptr)

# find minimum of the norm-regularized quadratic
print("\n solve problem 3 with additional non-unit norm")
x = nrek.solve_problem(n, H_ne, H_val, c, power, weight, S_ne, S_val)
print(" x:",x)

# get information
inform = nrek.information()
print(" f: %.4f" % inform['obj'])

# deallocate internal data

nrek.terminate()

