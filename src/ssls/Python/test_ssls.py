from galahad import ssls
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: ssls")

#  describe problem (only the lower triangle of matrix H is required)
#  ( 1     |  2   )     (   3   )
#  (   2 1 |  1 1 ) x = (   5   )
#  (   1 3 |    1 )     (   5   )
#  ( ------------ )     ( ----- )
#  ( 2 1   | -t   )     ( 3 - t )
#  (   1 1 |   -t )     ( 2 - t )

n = 3
m = 2

H_type = 'coordinate'
H_ne = 4
H_row = np.array([0,1,2,2])
H_col = np.array([0,1,1,2])
H_ptr = None
H_val = np.array([1.0,2.0,1.0,3.0])

A_type = 'coordinate'
A_ne = 4
A_row = np.array([0,0,1,1])
A_col = np.array([0,1,1,2])
A_ptr = None
A_val = np.array([2.0,1.0,1.0,1.0])

C_type = 'scaled_identity'
C_ne = 1
C_row = None
C_col = None
C_ptr = None
#t = 0.00000001
t = 0.0
C_val = np.array([t])

# allocate internal data and set default options
options = ssls.initialize()

# set some non-default options
#options['print_level'] = 1
options['symmetric_linear_solver'] = 'sytr '
#print("options:", options)

# load data (and optionally non-default options), and analyse matrix structure
ssls.load(n, m, H_type, H_ne, H_row, H_col, H_ptr,
                A_type, A_ne, A_row, A_col, A_ptr,
                C_type, C_ne, C_row, C_col, C_ptr, options)

# factorize matrix
ssls.factorize_matrix(H_ne, H_val, A_ne, A_val, C_ne, C_val)

# solve system
rhs = np.array([3.0,5.0,5.0,3.0-t,2.0-t])
x = ssls.solve_system(n, m, rhs)
print(" x:",x)

# get information
inform = ssls.information()
print(" rank:",inform['rank'])
print('** ssls exit status:', inform['status'])

# deallocate internal data

ssls.terminate()
