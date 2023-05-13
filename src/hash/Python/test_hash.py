from galahad import hash
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: hash")

# allocate internal data and set default options
nchar = 20
length = 200
options, inform = hash.initialize(nchar,length)

# set some non-default options
options['print_level'] = 0
#print("options:", options)
print(" initialize status:",inform['status'])

# call main solver(s)
# ....

# get information
inform = hash.information()
print('** hash exit status:', inform['status'])

# deallocate internal data

hash.terminate()
