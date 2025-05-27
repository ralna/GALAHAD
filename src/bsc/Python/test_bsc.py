from galahad import bsc
import numpy as np
np.set_printoptions(precision=2,suppress=True,floatmode='fixed')
print("\n** python test: bsc")

# set parameters
n = 4
m = 3

# describe matrix
A_type = 'coordinate'
A_ne = 6
A_row = np.array([0,0,1,1,2,2])
A_col = np.array([0,1,2,3,0,3])
A_ptr = None
A_val = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
D = np.array([1.0,2.0,3.0,4.0])

# allocate internal data and set default options
options = bsc.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
S_ne = bsc.load(m, n, A_type, A_ne, A_row, A_col, A_ptr, options)
print(" S_ne:",S_ne)

# form S = A D A'
S_row, S_col, S_ptr, S_val = bsc.form(m, n, A_ne, A_val, S_ne, D)
print(" S_row:",S_row)
print(" S_col:",S_col)
print(" S_ptr:",S_ptr)
print(" S_val:",S_val)

# get information
inform = bsc.information()
print('** bsc exit status:', inform['status'])

# deallocate internal data

bsc.terminate()
