from galahad import bsc
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# allocate internal data and set default options
options = bsc.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = bsc.information()
print(" status:",inform['status'])

# deallocate internal data

bsc.terminate()
