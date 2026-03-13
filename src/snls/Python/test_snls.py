from galahad import snls
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: snls")

# set parameters
p = 4.0
n = 5
m_r = 4

# set Jacobian sparsity
Jr_type = 'coordinate'
Jr_ne = 8
Jr_row = np.array([0,0,1,1,2,2,3,3])
Jr_col = np.array([0,1,1,2,2,3,3,4])
Jr_ptr = None

# set cohorts
m_c = 2
cohort = np.array([0,1,-1,0,1])

# set weights
w = np.array([1.0,1.0,1.0,1.0])

# allocate internal data and set default options
options = snls.initialize()

# set some non-default options
options['print_level'] = 0
options['jacobian_available'] = 2
options['stop_pg_absolute'] =  0.00001
options['slls_options']['sbls_options']['symmetric_linear_solver'] = 'sytr '
options['slls_options']['sbls_options']['definite_linear_solver'] = 'potr '
#print("options:", options)

# load data (and optionally non-default options)
snls.load(n, m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, 0, Jr_ptr, 
          cohort, options)

# define residual function and its Jacobian
def eval_r(x):
    return np.array([x[0]*x[1] - p, x[1]*x[2] - 1.0, 
                     x[2]*x[3] - 1.0, x[3]*x[4] - 1.0])
def eval_jr(x):
    return np.array([x[1], x[0], x[2], x[1], x[3], x[2], x[4], x[3]])

# set starting point
x = np.array([0.5,0.5,0.5,0.5,0.5])

# find minimizer
print("\n sparse solve snls")
x, y, z, r, g, x_stat = snls.solve(n, m_r, m_c, x, eval_r, Jr_ne, eval_jr, w)
print(" x:",x)
print(" y:",y)
print(" z:",z)
print(" r:",r)
print(" g:",g)
print(" x_stat:",x_stat)

# get information
inform = snls.information()
#print("inform:", inform)
print(" f: %.4f" % inform['obj'])
print('** snls exit status:', inform['status'])

# deallocate internal data
snls.terminate()

# repeat problem using dense format

# set Jacobian sparsity
Jr_type = 'dense'
Jr_ne = n * m_r
Jr_row = None
Jr_col = None
Jr_ptr = None

# allocate internal data and set default options
options = snls.initialize()

# set some non-default options
options['print_level'] = 0
options['jacobian_available'] = 2
options['stop_pg_absolute'] =  0.00001
options['slls_options']['sbls_options']['symmetric_linear_solver'] = 'sytr '
options['slls_options']['sbls_options']['definite_linear_solver'] = 'potr '
#print("options:", options)

# load data (and optionally non-default options)
snls.load(n, m_r, m_c, Jr_type, Jr_ne, Jr_row, Jr_col, 0, Jr_ptr, 
          cohort, options)

# define the dense Jacobian
def eval_jr_dense(x):
    return np.array([x[1], x[0], 0.0,  0.0,  0.0,
                     0.0,  x[2], x[1], 0.0,  0.0,
                     0.0,  0.0,  x[3], x[2], 0.0,
                     0.0,  0.0,  0.0,  x[4], x[3]])

# set starting point
x = np.array([0.5,0.5,0.5,0.5,0.5])

# find minimizer
print("\n dense solve snls")
x, y, z, r, g, x_stat = snls.solve(n, m_r, m_c, x, eval_r, Jr_ne, eval_jr_dense, w)
print(" x:",x)
print(" y:",y)
print(" z:",z)
print(" r:",r)
print(" g:",g)
print(" x_stat:",x_stat)

# get information
inform = snls.information()
#print("inform:", inform)
print(" f: %.4f" % inform['obj'])
print('** snls exit status:', inform['status'])

# deallocate internal data
snls.terminate()



