from galahad import psls
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: psls")

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

# allocate internal data and set default options
options = psls.initialize()

# set some non-default options
options['preconditioner'] = 6
#print("options:", options)

# load data (and optionally non-default options)
psls.load(n, A_type, A_ne, A_row, A_col, A_ptr, options)

# form the preconditioner
psls.form_preconditioner(A_ne, A_val)

# apply the preconditioner
x = psls.apply_preconditioner(n, b)
print(" x:",x)

# get information
inform = psls.information()
print(" preconditioner:",inform['preconditioner'])
print('** psls exit status:', inform['status'])

# deallocate internal data

psls.terminate()
