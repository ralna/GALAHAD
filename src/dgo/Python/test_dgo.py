from galahad import dgo
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: dgo")

# allocate internal data and set default options
options = dgo.initialize()

# set some non-default options
options['print_level'] = 0
options['ugo_options']['print_level'] = 0
#print("options:", options)

# set parameters
p = 4
freq = 10
mag = 1000
# set bounds
n = 3
x_l = np.array([-10.0,-10.0,-10.0])
x_u = np.array([1.0,1.0,1.0])

# set Hessian sparsity
H_type = 'coordinate'
H_ne = 5
H_row = np.array([0,2,1,2,2])
H_col = np.array([0,0,1,1,2])
H_ptr = None

# load data (and optionally non-default options)
dgo.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=options)

# define objective function and its derivatives
def eval_f(x):
    return (x[0] + x[2] + p)**2 + (x[1] + x[2])**2 + mag * np.cos(freq * x[0]) + x[0] + x[1] + x[2]
def eval_g(x):
    return np.array([2. * ( x[0] + x[2] + p )
                     - mag * freq * np.sin(freq * x[0]) + 1.,
                     2. * ( x[1] + x[2] ) + 1.,
                     2. * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] ) + 1.])
def eval_h(x):
    return np.array([2. - mag * freq * freq * np.cos(freq * x[0]),2.,2.,2.,4.])

# set starting point
x = np.array([1.,1.,1.])

# find optimum
x, g = dgo.solve(n, H_ne, x, eval_f, eval_g, eval_h)
print(" x:",x)
print(" g:",g)

# get information
inform = dgo.information()
print(" f: %.4f" % inform['obj'])
print('** dgo exit status:', inform['status'])

# deallocate internal data
dgo.terminate()
