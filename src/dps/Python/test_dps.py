from galahad import dps
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: dps")

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

# allocate internal data and set default options
options = dps.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
dps.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)

# set trust-region radius
radius = 1.0

# find minimum of quadratic within the trust region
print("\n solve trust-region problem")
x = dps.solve_tr_problem(n, radius, f, g, H_ne, H_val)
print(" x:",x)

# get information
inform = dps.information()
print(" f: %.4f" % inform['obj'])

# set regualization weight and power
weight = 1.0
power = 3.0

# find minimum of regularized quadratic
print("\n solve regularization problem")
x = dps.solve_rq_problem(n, weight, power, f, g, H_ne, H_val)
print(" x:",x)

# get information
inform = dps.information()
print(" f: %.4f" % inform['obj'])
print('** dps exit status:', inform['status'])

# deallocate internal data

dps.terminate()

