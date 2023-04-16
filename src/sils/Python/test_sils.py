from galahad import sls
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

#  describe problem (only the lower triangle of matrix is required)
#  ( 1     )     ( 1 )
#  (   2 1 ) x = ( 3 )
#  (   1 3 )     ( 4 )

n = 3
A_type = 'coordinate'
A_ne = 4
A_row = np.array([0,1,2,2])
A_col = np.array([0,1,1,2])
A_ptr = None
A_val = np.array([1.0,2.0,1.0,3.0])
b = np.array([1.0,3.0,4.0])

# set solver name, allocate internal data and set default options
solver = 'sytr'
options = sls.initialize(solver)

# set some non-default options
options['print_level'] = 1
#print("options:", options

# load data (and optionally non-default options), and analyse matrix structure
#sls.analyse_matrix(n, A_type, A_ne, A_row, A_col, A_ptr, options)

# factorize matrix
#sls.factorize_matrix(A_ne, A_val)

# solve system
#x = sls.solve_system(n, b)
#print("x:",x)

# get information
inform = sls.information()
print("rank:",inform['rank'])

# deallocate internal data

sls.terminate()
