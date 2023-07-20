from galahad import bgo
import numpy as np

# allocate internal data and set default options
bgo.initialize()

# set some non-default options
options = {'print_level' : 1, 'ugo_options' : {'print_level' : 0}}
print(options)

# set bounds
n = 2
x_l = np.array([3.,4.])
x_u = np.array([5.,6.])

# set Hessian sparsity
H_type = 'sparse_by_rows'
H_ne = 2
H_row = None
H_col = np.array([0,1])
H_ptr = np.array([0,1,2])
#H_ptr = np.array([3,5,7])

# load data (and optionally non-default options)
bgo.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=options)

# define objective function and its derivatives
def eval_f(x):
    return x[0]**2 + x[1]**2
def eval_g(x):
    return np.array([2*x[0],2*x[1]])
def eval_h(x):
    return np.array([2.,2.])

# starting value
x = np.array([2.,2.])

# find optimum
x, g = bgo.solve(n, H_ne, x, eval_f, eval_g, eval_h)
print("x:",x)
print("g:",g)

# get information
inform = bgo.information()
print(inform)

# deallocate internal data
bgo.terminate()
