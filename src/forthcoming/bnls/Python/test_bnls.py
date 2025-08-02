from galahad import bnls
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: bnls")

# allocate internal data and set default options
options = bnls.initialize()

# set some non-default options
options['print_level'] = 0
options['jacobian_available'] = 2
options['hessian_available'] = 2
options['model'] = 6
#print("options:", options)

# set parameters
p = 1.0
n = 2
m = 3

# set Jacobian sparsity
J_type = 'coordinate'
J_ne = 5
J_row = np.array([0,1,1,2,2])
J_col = np.array([0,0,1,0,1])
J_ptr = None

# set Hessian sparsity
H_type = 'coordinate'
H_ne = 2
H_row = np.array([0,1])
H_col = np.array([0,1])
H_ptr = None

# set Hessian product sparsity
P_type = 'sparse_by_columns'
P_ne = 2
P_row = np.array([0,1])
P_col = None
P_ptr = np.array([0,1,2,2])

w = np.array([1.0,1.0,1.0])

# load data (and optionally non-default options)
bnls.load(n, m,
         J_type, J_ne, J_row, J_col, J_ptr,
         H_type, H_ne, H_row, H_col, H_ptr,
         P_type, P_ne, P_row, P_col, P_ptr,
         w, options)

# define residual function and its derivatives
def eval_c(x):
    return np.array([(x[0])**2 + p, x[0] + (x[1])**2, x[0] - x[1]])
def eval_j(x):
    return np.array([2.0 * x[0], 1.0, 2.0 * x[1], 1.0, - 1.0])
def eval_h(x,y):
    return np.array([2.0 * y[0], 2.0 * y[1]])
def eval_hprod(x,v):
    return np.array([2.0 * v[0], 2.0 * v[1]])

# set starting point
x = np.array([1.5,1.5])

# find optimum
x, c, g = bnls.solve(n, m, x, eval_c, J_ne, eval_j, H_ne, eval_h,
                    P_ne, eval_hprod)
print(" x:",x)
print(" c:",c)
print(" g:",g)

# get information
inform = bnls.information()
#print(inform)
print(" f: %.4f" % inform['obj'])
print('** bnls exit status:', inform['status'])

# deallocate internal data
bnls.terminate()
