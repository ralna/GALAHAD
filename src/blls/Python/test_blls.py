from galahad import blls
import numpy as np
np.set_printoptions(precision=2,suppress=True,floatmode='fixed')
print("\n** python test: blls")

# set parameters
n = 10
o = n + 1
sigma = 1.0
infinity = float("inf")

#  describe A = (  I  ) and b = ( i * e )
#               ( e^T )         ( n + 1 )

Ao_type = 'coordinate'
Ao_ne = 2 * n
Ao_row = np.empty(Ao_ne, int)
Ao_col = np.empty(Ao_ne, int)
Ao_val = np.empty(Ao_ne)
Ao_ptr = None
b = np.empty(o)
b[n] = o
for i in range(n):
  Ao_row[i] = i
  Ao_row[n+i] = o - 1
  Ao_col[i] = i
  Ao_col[n+i] = i
  Ao_val[i] = 1.0
  Ao_val[n+i] = 1.0
  b[i] = i + 1

#  set the weights and shifts

w = np.empty(o)
x_s = np.empty(n)
w[0] = 2.0
for i in range(1,o):
  w[i] = 1.0
for i in range(n):
  x_s[i] = 0.0

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
blls.load(n, o, Ao_type, Ao_ne, Ao_row, Ao_col, 0, Ao_ptr, options)

#  provide starting values (not crucial)

x = np.empty(n)
z = np.empty(n)
for i in range(n):
  x[i] = 0.0
  z[i] = 0.0

# find minimizer
#print("\n solve blls")
x, z, r, g, x_stat \
  = blls.solve(n, o, Ao_ne, Ao_val, b, sigma, x_l, x_u, x, z, w, x_s)
print(" x:",x)
print(" z:",z)
print(" r:",r)
print(" g:",g)
print(" x_stat:",x_stat)

# get information
inform = blls.information()
print(" r: %.4f" % inform['obj'])
print('** blls exit status:', inform['status'])

# deallocate internal data

blls.terminate()
