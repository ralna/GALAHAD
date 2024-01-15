from galahad import eqp
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: eqp")

# set parameters
n = 3
m = 2

#  describe objective function

f = 1.0
g = np.array([0.0,2.0,0.0])
H_type = 'coordinate'
H_ne = 3
H_row = np.array([0,1,2])
H_col = np.array([0,1,2])
H_ptr = None
H_val = np.array([1.0,1.0,1.0])

#  describe constraints

A_type = 'coordinate'
A_ne = 4
A_row = np.array([0,0,1,1])
A_col = np.array([0,1,1,2])
A_ptr = None
A_val = np.array([2.0,1.0,1.0,1.0])
c = np.array([3.0,0.0])

# allocate internal data and set default options
options = eqp.initialize()

# set some non-default options
options['print_level'] = 0
options['fdc_options']['use_sls'] = True
options['fdc_options']['symmetric_linear_solver'] = 'sytr '
options['sbls_options']['symmetric_linear_solver'] = 'sytr '
options['sbls_options']['definite_linear_solver'] = 'sytr '
#print("options:", options)

# load data (and optionally non-default options)
eqp.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
         A_type, A_ne, A_row, A_col, A_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
y = np.array([0.0,0.0])

# find optimum of qp
print("\n 1st problem: solve qp")
x, y = eqp.solve_qp(n, m, f, g, H_ne, H_val, A_ne, A_val, c, x, y)
print("x:",x)
print("y:",y)

# get information
inform = eqp.information()
print("f:",inform['obj'])

# deallocate internal data

eqp.terminate()

#  describe shifted-least-distance qp

w = np.array([1.0,1.0,1.0])
x0 = np.array([0.0,0.0,0.0])
H_type = 'shifted_least_distance'

# allocate internal data
eqp.initialize()

# load data (and optionally non-default options)
eqp.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
         A_type, A_ne, A_row, A_col, A_ptr, options)

#  provide starting values (not crucial)

x = np.array([0.0,0.0,0.0])
y = np.array([0.0,0.0])

# find optimum of sldqp
print("\n 2nd problem: solve sldqp")
x, y = eqp.solve_sldqp(n, m, f, g, w, x0, A_ne, A_val, c, x, y)
print("x:",x)
print("y:",y)

# get information
inform = eqp.information()
print(" f: %.4f" % inform['obj'])
print('** eqp exit status:', inform['status'])

# deallocate internal data

eqp.terminate()
