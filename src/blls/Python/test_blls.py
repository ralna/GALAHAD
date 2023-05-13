from galahad import blls
import numpy as np
np.set_printoptions(precision=2,suppress=True,floatmode='fixed')
print("\n** python test: blls")

# set parameters
n = 10
m = n + 1
infinity = float("inf")

#  describe A = (  I  ) and b = ( i * e )
#               ( e^T )         ( n + 1 )

A_type = 'coordinate'
A_ne = 2 * n
A_row = np.empty(A_ne, int)
A_col = np.empty(A_ne, int)
A_val = np.empty(A_ne)
A_ptr = None
b = np.empty(m)
b[n] = m
for i in range(n):
  A_row[i] = i
  A_row[n+i] = m - 1
  A_col[i] = i
  A_col[n+i] = i
  A_val[i] = 1.0
  A_val[n+i] = 1.0
  b[i] = i + 1

#  set the weights

w = np.empty(m)
w[0] = 2.0
for i in range(1,m):
  w[i] = 1.0

#  specify the bounds on the variables

x_l = np.empty(n)
x_u = np.empty(n)
x_l[0] = - 1.0
x_u[0] = 1.0
x_l[1] = - infinity
x_u[1] = infinity
for i in range(2,n):
  x_l[i] = - infinity
  x_u[i] = 2.0

# allocate internal data and set default options
options = blls.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
blls.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options)

#  provide starting values (not crucial)

x = np.empty(n)
z = np.empty(n)
for i in range(n):
  x[i] = 0.0
  z[i] = 0.0

# find minimizer
#print("\n solve blls")
x, c, z, g, x_stat \
  = blls.solve_ls(n, m, w, A_ne, A_val, b, x_l, x_u, x, z)
print(" x:",x)
print(" c:",c)
print(" z:",z)
print(" g:",g)
print(" x_stat:",x_stat)

# get information
inform = blls.information()
print(" r: %.4f" % inform['obj'])
print('** blls exit status:', inform['status'])

# deallocate internal data

blls.terminate()
