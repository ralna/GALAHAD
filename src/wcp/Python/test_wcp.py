from galahad import wcp
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: wcp")

# set parameters
n = 3
m = 2
infinity = float("inf")

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

# allocate internal data and set default options
options = wcp.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
wcp.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
g = np.array([0.0,0.0,0.0])
y_l = np.array([0.0,0.0])
y_u = np.array([0.0,0.0])
z_l = np.array([0.0,0.0,0.0])
z_u = np.array([0.0,0.0,0.0])

# find optimum of lp
print("\n find well-centred point")
x, c, y_l, y_u, z_l, z_u, x_stat, c_stat \
  = wcp.find_wcp(n, m, A_ne, A_val, c_l, c_u, x_l, x_u, x, y_l, y_u,
                 z_l, z_u, g)
print(" x:",x)
print(" c:",c)
print(" y_l:",y_l)
print(" y_u:",y_u)
print(" z_l:",z_l)
print(" z_u:",z_u)
print(" x_stat:",x_stat)
print(" c_stat:",c_stat)

# get information
inform = wcp.information()
print(" strictly feasible:",inform['feasible'])
print('** wcp exit status:', inform['status'])

# deallocate internal data

wcp.terminate()

