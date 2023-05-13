from galahad import bqp
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: bqp")

# set parameters
n = 3
infinity = float("inf")

#  describe objective function

f = 1.0
g = np.array([0.0,2.0,0.0])
H_type = 'coordinate'
H_ne = 4
H_row = np.array([0,1,2,2])
H_col = np.array([0,1,1,2])
H_ptr = None
H_val = np.array([1.0,2.0,1.0,3.0])

#  describe constraints

x_l = np.array([-1.0,-infinity,-infinity])
#x_u = np.array([1.0,infinity,2.0])
x_u = np.array([1.0,infinity,0.0])

# allocate internal data and set default options
options = bqp.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
bqp.load(n, H_type, H_ne, H_row, H_col, H_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
z = np.array([0.0,0.0,0.0])

# find optimum of qp
#print("\nsolve bqp")
x, z, x_stat = bqp.solve_qp(n, f, g, H_ne, H_val, x_l, x_u, x, z)
print(" x:",x)
print(" z:",z)
print(" x_stat:",x_stat)

# get information
inform = bqp.information()
print(" f: %.4f" % inform['obj'])
print('** bqp exit status:', inform['status'])

# deallocate internal data

bqp.terminate()

