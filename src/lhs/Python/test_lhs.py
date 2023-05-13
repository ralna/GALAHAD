from galahad import lhs
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: lhs")

# allocate internal data and set default options
options, inform = lhs.initialize()
print(" initiliaze status:",inform['status'])

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = lhs.information()
print('** lhs exit status:', inform['status'])

# deallocate internal data

lhs.terminate()
