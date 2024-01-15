from galahad import cro
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: cro")

# set parameters
n = 11
m = 3
m_equal = 1
infinity = float("inf")

#  describe objective function

g = np.array([0.5,-0.5,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,-0.5])
H_ne = 21
H_val = np.array([1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,
                  0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0])
H_col = np.array([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10])
H_ptr = np.array([0,1,3,5,7,9,11,13,15,17,19,21])

#  describe constraints

A_ne = 30
A_val = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                  1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                  1.0,1.0,1.0,1.0,1.0,1.0])
A_col = np.array([0,1,2,3,4,5,6,7,8,9,10,2,3,4,5,6,7,8,9,10,
                  1,2,3,4,5,6,7,8,9,10])
A_ptr = np.array([0,11,20,30])
c_l = np.array([10.0,9.0,-infinity])
c_u = np.array([10.0,infinity,10.0])
x_l = np.array([0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
x_u  = np.array([infinity,infinity,infinity,infinity,infinity,infinity,
          infinity,infinity,infinity,infinity,infinity])

# allocate internal data and set default options
options = cro.initialize()

# set some non-default options
options['print_level'] = 0
options['symmetric_linear_solver'] = 'sytr '
options['unsymmetric_linear_solver'] = 'getr '
options['sbls_options']['symmetric_linear_solver'] = 'sytr '
options['sbls_options']['definite_linear_solver'] = 'sytr '
options['sbls_options']['unsymmetric_linear_solver'] = 'getr '
#print("options:", options)

# provide optimal variables, Lagrange multipliers and dual variables
x = np.array([0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0])
c = np.array([10.0,9.0,10.0])
y = np.array([ -1.0,1.5,-2.0])
z = np.array([2.0,4.0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5])

# provide interior-point constraint and variable status
c_stat = np.array([-1,-1,1])
x_stat = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

# crossover from an interior-point to a basic solution
x, c, y, z, x_stat, c_stat, inform \
  = cro.crossover_solution(n, m, m_equal, g, H_ne, H_val, H_col, H_ptr,
                           A_ne, A_val, A_col, A_ptr, c_l, c_u, x_l, x_u,
                           x, c, y, z, x_stat, c_stat, options)
print(" x:",x)
print(" c:",c)
print(" y:",y)
print(" z:",z)
print(" x_stat:",x_stat)
print(" c_stat:",c_stat)
print(" number of dependent constraints:",inform['dependent'])
print('** cro exit status:', inform['status'])

# deallocate internal data

cro.terminate()

