from galahad import arc
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: arc")

# allocate internal data and set default options
options = arc.initialize()

# set some non-default options
options['print_level'] = 0
#options['rqs_options']['print_level'] = 0
#print("options:", options)

# set parameters
p = 4
# set bounds
n = 3

# set Hessian sparsity
H_type = 'coordinate'
H_ne = 5
H_row = np.array([0,2,1,2,2])
H_col = np.array([0,0,1,1,2])
H_ptr = None

# load data (and optionally non-default options)
arc.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=options)

# define objective function and its derivatives
def eval_f(x):
    return (x[0] + x[2] + p)**2 + (x[1] + x[2])**2 + np.cos(x[0])
def eval_g(x):
    return np.array([2.0* ( x[0] + x[2] + p ) - np.sin(x[0]),
                     2.0* ( x[1] + x[2] ),
                     2.0* ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] )])
def eval_h(x):
    return np.array([2. - np.cos(x[0]),2.0,2.0,2.0,4.0])

# set starting point
x = np.array([1.,1.,1.])

# find optimum
x, g = arc.solve(n, H_ne, x, eval_f, eval_g, eval_h)
print(" x:",x)
print(" g:",g)

# get information
inform = arc.information()
#print(inform)
print(" f: %.4f" % inform['obj'])
print('** arc exit status:', inform['status'])

# deallocate internal data
arc.terminate()
