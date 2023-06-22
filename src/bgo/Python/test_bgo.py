from galahad import bgo
import numpy as np
import sys

# allocate internal data and set default options
options = bgo.initialize()

# set some non-default options
options['print_level'] = 1
options['ugo_options']['print_level'] = 0
#print("options:", options)

# set parameters
p = 4
freq = 10
mag = 1000
# set bounds
n = 3
x_l = np.array([-10.0,-10.0,-10.0])
x_u = np.array([0.5,0.5,0.5])

# set Hessian sparsity
H_type = 'coordinate'
H_ne = 5
H_row = np.array([0,1,2,2,2])
H_col = np.array([0,1,0,1,2])
H_ptr = None

# load data (and optionally non-default options)
bgo.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=options)

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
x = np.array([0.,0.,0.])

# find optimum
x, g = bgo.solve(n, H_ne, x, eval_f, eval_g, eval_h )

x_copy = x.copy()
g_copy = g.copy()
print("x:",x_copy)
print("g:",g_copy)
#print("g:",g)
#print("g:",g)
#print("ref count x", sys.getrefcount(x))
#print("ref count g", sys.getrefcount(g))

# get information
inform = bgo.information()
print("f:",inform['obj'])

# deallocate internal data
bgo.terminate()
