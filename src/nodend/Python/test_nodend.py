from galahad import nodend
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: nodend")

# set parameters
n = 3

A_type = 'coordinate'
A_ne = 4
A_row = np.array([0,1,2,2])
A_col = np.array([0,1,1,2])
A_ptr = None

# allocate internal data and set default options
options = nodend.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# find the ordering
perm = nodend.order(n, A_type, A_ne, A_row, A_col, A_ptr, options)
print(" perm",perm)

# get information
inform = nodend.information()
print('** nodend exit status:', inform['status'])

# deallocate internal data

nodend.terminate()
