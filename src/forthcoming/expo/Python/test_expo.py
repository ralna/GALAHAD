from galahad import expo
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: expo")

# allocate internal data and set default options
options = expo.initialize()

# set some non-default options
#options['print_level'] = 0
options['max_it'] = 20
options['max_eval'] = 100
options['stop_abs_p'] = 0.00001
options['stop_abs_d'] = 0.00001
options['stop_abs_c'] = 0.00001

#print("options:", options)

# set parameters
p = 7.0
n = 2
m = 5

# set Jacobian sparsity
J_type = 'coordinate'
J_ne = 10
J_row = np.array([0,0,1,1,2,2,3,3,4,4])
J_col = np.array([0,1,0,1,0,1,0,1,0,1])
J_ptr = None

# set Hessian sparsity
H_type = 'coordinate'
H_ne = 2
H_row = np.array([0,1])
H_col = np.array([0,1])
H_ptr = None

# set upper and lower bounds
x_l = [-50.0, -50.0] # lower variable bounds
x_u = [50.0, 50.0] # upper variable bounds
c_l = [0.0, 0.0, 0.0, 0.0, 0.0] # lower constraint bounds
c_u = [np.inf, np.inf, np.inf, np.inf, np.inf] # upper constraint bounds

# load data (and optionally non-default options)
expo.load(n, m,
         J_type, J_ne, J_row, J_col, J_ptr,
         H_type, H_ne, H_row, H_col, H_ptr,
         options)

# define residual function and its derivatives
def eval_fc(x):
    f = x[0]^2 + x[1]^2
    c = np.array([x[0] + x[1] - 1.0,
                  x[0]^2 + x[1]^2 - 1.0,
                  p * x[0]^2 + x[1]^2 - p,
                  x[0]^2 - x[1],
                  x[1]^2 - x[0]])
    return f, c

def eval_gj(x):
    g = np.array([2.0 * x[0],2.0 * x[1]])
    j = np.array([1.0, 1.0,
                  2.0 * x[0], 2.0 * x[1],
                  2.0 * p * x[0], 2.0 * x[1],
                  2.0 * x[0], -1.0,
                  -1.0, 2.0 * x[1]])
    return g, j

def eval_hl(x,y):
    hval =  np.array([2.0 - 2.0 * (y[1] + p * y[2] + y[3]),
                       2.0 - 2.0 * (y[1] + y[2] + y[4])])
    return hval

# set starting point
x = np.array([3.0,1.0])

# find optimum
x, y, z, c, gl = expo.solve(n, m, J_ne, H_ne, c_l, c_u, x_l, x_u, x,
                            eval_fc, eval_gj, eval_hl)

print(" x:",x)
#print(" c:",c)
#print(" g:",g)

# get information
inform = expo.information()
#print(inform)
print(" f: %.4f" % inform['obj'])
print('** expo exit status:', inform['status'])

# deallocate internal data
expo.terminate()
