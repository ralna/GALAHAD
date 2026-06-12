from galahad import sllsb
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: sllsb")

# set parameters
n = 10
o = n + 1
m = 1
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

# allocate internal data and set default options
options = sllsb.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
sllsb.load(n, o, m, Ao_type, Ao_ne, Ao_row, Ao_col, 0, Ao_ptr, None, options)

#  provide starting values (not crucial)

x = np.empty(n)
y = np.empty(m)
z = np.empty(n)
for i in range(n):
  x[i] = 0.0
  z[i] = 0.0

for i in range(m):
  y[i] = 0.0

# find minimizer
print("\n solve sllsb")
x, y, z, r, x_stat \
  = sllsb.solve(n, o, m, Ao_ne, Ao_val, b, sigma, x, y, z, None, None)
print(" x:",x)
print(" y:",y)
print(" z:",z)
print(" r:",r)
print(" x_stat:",x_stat)

# get information
inform = sllsb.information()
print(" r: %.4f" % inform['obj'])
print('** sllsb exit status:', inform['status'])

# deallocate internal data

sllsb.terminate()

# use an explicit cohort

# allocate internal data and set default options
options = sllsb.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

cohort = np.empty(n, int)
for i in range(n):
  cohort[i] = 0

# load data (and optionally non-default options)
sllsb.load(n, o, m, Ao_type, Ao_ne, Ao_row, Ao_col, 0, Ao_ptr, cohort, options)

#  provide starting values (not crucial)

x_s = np.empty(n)
for i in range(n):
  x[i] = 0.0
  z[i] = 0.0
  x_s[i] = 0.0

for i in range(m):
  y[i] = 0.0

w = np.empty(o)
for i in range(o):
  w[i] = 1.0

# find minimizer
print("\n solve sllsb with an explicit cohort")
x, y, z, r, x_stat \
  = sllsb.solve(n, o, m, Ao_ne, Ao_val, b, sigma, x, y, z, w, x_s)
print(" x:",x)
print(" y:",y)
print(" z:",z)
print(" r:",r)
print(" x_stat:",x_stat)

# get information
inform = sllsb.information()
print(" r: %.4f" % inform['obj'])
print('** sllsb exit status:', inform['status'])

# deallocate internal data

sllsb.terminate()
