from galahad import lhs
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: lhs")

# allocate internal data and set default options
options, inform = lhs.initialize()
print(" initialize status:",inform['status'])

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# compute 5 Latin hypercube samples in 2 dimensions, starting from seed 1
n_dimen = 2
n_points = 5
seed = 1
X, seed = lhs.ihs(n_dimen, n_points, seed, options)
print(" samples:")
print(X)

# get information
inform = lhs.information()
print('** lhs exit status:', inform['status'])

# deallocate internal data

lhs.terminate()
