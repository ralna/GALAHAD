from galahad import ir
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# allocate internal data and set default options
options = ir.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = ir.information()
print(" status:",inform['status'])

# deallocate internal data

ir.terminate()
