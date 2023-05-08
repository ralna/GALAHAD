from galahad import sha
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# allocate internal data and set default options
options = sha.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = sha.information()
print(" status:",inform['status'])

# deallocate internal data

sha.terminate()
