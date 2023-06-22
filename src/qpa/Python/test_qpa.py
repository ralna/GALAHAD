from galahad import qpa
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# set parameters
n = 3
m = 2
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
options = qpa.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
qpa.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, 
         A_type, A_ne, A_row, A_col, A_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
y = np.array([0.0,0.0])
z = np.array([0.0,0.0,0.0])

# find optimum of qp
print("\n 1st problem: solve qp")
x, c, y, z, x_stat, c_stat \
  = qpa.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, 
                 c_l, c_u, x_l, x_u, x, y, z)
print("x:",x)
print("c:",c)
print("y:",y)
print("z:",z)
print("x_stat:",x_stat)
print("c_stat:",c_stat)

# get information
inform = qpa.information()
print("f:",inform['obj'])

# deallocate internal data

qpa.terminate()

# l1qp problem
# allocate internal data and set default options
options = qpa.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
qpa.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, 
         A_type, A_ne, A_row, A_col, A_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
y = np.array([0.0,0.0])
z = np.array([0.0,0.0,0.0])

#  provide weights

rho_g = 1.0
rho_b = 1.0

# find optimum of qp
print("\n 2nd problem: solve l1qp")
x, c, y, z, x_stat, c_stat \
  = qpa.solve_l1qp(n, m, f, g, H_ne, H_val, rho_g, rho_b, A_ne, A_val, 
                 c_l, c_u, x_l, x_u, x, y, z)
print("x:",x)
print("c:",c)
print("y:",y)
print("z:",z)
print("x_stat:",x_stat)
print("c_stat:",c_stat)

# get information
inform = qpa.information()
print("f:",inform['obj'])

# deallocate internal data

qpa.terminate()

# bcl1qp problem
# allocate internal data and set default options
options = qpa.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
qpa.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, 
         A_type, A_ne, A_row, A_col, A_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
y = np.array([0.0,0.0])
z = np.array([0.0,0.0,0.0])

#  provide weights

rho_g = 1.0
rho_b = 1.0

# find optimum of qp
print("\n 3rd problem: solve bcl1qp")
x, c, y, z, x_stat, c_stat \
  = qpa.solve_bcl1qp(n, m, f, g, H_ne, H_val, rho_g, A_ne, A_val, 
                 c_l, c_u, x_l, x_u, x, y, z)
print("x:",x)
print("c:",c)
print("y:",y)
print("z:",z)
print("x_stat:",x_stat)
print("c_stat:",c_stat)

# get information
inform = qpa.information()
print("f:",inform['obj'])

# deallocate internal data

qpa.terminate()

