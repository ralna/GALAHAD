from galahad import trek
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: trek")

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

S_type = 'coordinate'
S_ne = 3
S_row = np.array([0,1,2])
S_col = np.array([0,1,2])
S_ptr = None
S_val = np.array([1.0,2.0,1.0])

# allocate internal data and set default options
options = trek.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
trek.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)

# set trust-region radius

radius = 1.0

# find minimum of quadratic within the trust region
print("\n solve problem 1")
x = trek.solve_problem(n, H_ne, H_val, g, radius)
print(" x:",x)

# get information
inform = trek.information()
print(" f: %.4f" % inform['obj'])

# reset trust-region radius to the suggested smaller value

radius = inform['next_radius']
options['new_radius'] = True
trek.reset_options(options)

# find minimum of quadratic within the trust region
print("\n solve problem 2 with smaller radius")
x = trek.solve_problem(n, H_ne, H_val, g, radius)
print(" x:",x)

# get information
inform = trek.information()
print(" f: %.4f" % inform['obj'])

# reinitialize trust-region radius

radius = 1.0
options['new_radius'] = False
trek.reset_options(options)

# load data (and optionally non-default options)
trek.load_s(n, S_type, S_ne, S_row, S_col, S_ptr)

# find minimum of quadratic within the trust region
print("\n solve problem 3 with additional non-unit norm")
x = trek.solve_problem(n, H_ne, H_val, g, radius, S_ne, S_val)
print(" x:",x)

# get information
inform = trek.information()
print(" f: %.4f" % inform['obj'])

# deallocate internal data

trek.terminate()

