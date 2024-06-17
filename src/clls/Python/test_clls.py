from galahad import clls
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: clls")

# set parameters
n = 3
o = 4
m = 2
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

A_type = 'coordinate'
A_ne = 4
A_row = np.array([0,0,1,1])
A_col = np.array([0,1,1,2])
A_ptr = None
A_val = np.array([2.0,1.0,1.0,1.0])
c_l = np.array([1.0,2.0])
c_u = np.array([2.0,2.0])
x_l = np.array([-1.0,-infinity,-infinity])
x_u = np.array([1.0,infinity,2.0])

sigma = 1.0
w = np.array([1.0,1.0,1.0,2.0])

# allocate internal data and set default options
options = clls.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
clls.load(n, o, m, Ao_type, Ao_ne, Ao_row, Ao_col, 0, Ao_ptr, 
          A_type, A_ne, A_row, A_col, 0, A_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
y = np.array([0.0,0.0])
z = np.array([0.0,0.0,0.0])

# find optimum of clls
print("\n problem: solve clls")
x, r, c, y, z, x_stat, c_stat \
  = clls.solve_clls(n, o, m, Ao_ne, Ao_val, b, sigma, A_ne, A_val, 
                    c_l, c_u, x_l, x_u, x, y, z, w)
print(" x:",x)
print(" r:",r)
print(" c:",c)
print(" y:",y)
print(" z:",z)
print(" x_stat:",x_stat)
print(" c_stat:",c_stat)

# get information
inform = clls.information()
print(" f: %.4f" % inform['obj'])

# deallocate internal data

clls.terminate()
