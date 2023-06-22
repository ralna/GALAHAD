from galahad import lms
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# allocate internal data and set default options
options = lms.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = lms.information()
print(" status:",inform['status'])

# deallocate internal data

lms.terminate()
