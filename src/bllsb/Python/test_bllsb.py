from galahad import bllsb
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: bllsb")

# set parameters
n = 3
o = 4
infinity = float("inf")

#  describe objective function

b = np.array([2.0,2.0,3.0,1.0])
Ao_type = 'coordinate'
Ao_ne = 7
Ao_row = np.array([0,0,1,1,2,2,3])
Ao_col = np.array([0,1,1,2,0,2,1])
Ao_ptr = None
Ao_val = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0])

#  describe constraints

x_l = np.array([-1.0,-infinity,-infinity])
x_u = np.array([1.0,infinity,2.0])

sigma = 1.0
w = np.array([1.0,1.0,1.0,2.0])

# allocate internal data and set default options
options = bllsb.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
bllsb.load(n, o, Ao_type, Ao_ne, Ao_row, Ao_col, 0, Ao_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
z = np.array([0.0,0.0,0.0])

# find optimum of bllsb
print("\n problem: solve bllsb")
x, r, z, x_stat \
  = bllsb.solve_bllsb(n, o, Ao_ne, Ao_val, b, sigma, x_l, x_u, x, z, w)
print(" x:",x)
print(" r:",r)
print(" z:",z)
print(" x_stat:",x_stat)

# get information
inform = bllsb.information()
print(" f: %.4f" % inform['obj'])

# deallocate internal data

bllsb.terminate()
