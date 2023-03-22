from galahad import cqp
import numpy as np

# allocate internal data and set default options
cqp.initialize()

# set some non-default options
#options = {'print_level' : 1, 'jacobian_available' : 2,
#           'hessian_available' : 2, 'model' : 6 }
options = {'print_level' : 1 }
print(options)

# set parameters
p = 1.0
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

# load data (and optionally non-default options)
cqp.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, 
         A_type, A_ne, A_row, A_col, A_ptr, options)

# find optimum
#x = cqp.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c_l, c_u, x_l, x_u)
x, c, y, z, x_stat, c_stat = cqp.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c_l, c_u, x_l, x_u)
#i, x, c, y, z = cqp.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c_l, c_u, x_l, x_u)
x_copy=x.copy()
c_copy=c.copy()
y_copy=y.copy()
z_copy=z.copy()
x_stat_copy=x_stat.copy()
c_stat_copy=c_stat.copy()
print("x:",x.copy())
print("x_copy:",x_copy)
print("c:",c.copy())
print("c_copy:",c_copy)
print("y:",y.copy())
print("y_copy:",y_copy)
print("z:",z.copy())
print("z_copy:",z_copy)
print("x_stat:",x_stat.copy())
print("x_stat_copy:",x_stat_copy)
print("c_stat:",c_stat.copy())
print("c_stat_copy:",c_stat_copy)

#print("c:",c)
#print("y:",y)
#print("z:",z)
#print("x_stat:",x_stat)
#print("c:",c_stat)

# get information
inform = cqp.information()
#print(inform)
print("f:",inform['obj'])

w = np.array([1.0,1.0,1.0])
x0 = np.array([1.0,1.0,1.0])

# deallocate internal data
cqp.terminate()
