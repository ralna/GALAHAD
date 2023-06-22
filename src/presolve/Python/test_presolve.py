from galahad import presolve
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# allocate internal data and set default options
options = presolve.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = presolve.information()
print(" status:",inform['status'])

# deallocate internal data

presolve.terminate()
