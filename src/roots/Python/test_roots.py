from galahad import roots
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# allocate internal data and set default options
options = roots.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = roots.information()
print(" status:",inform['status'])

# deallocate internal data

roots.terminate()
