from galahad import ir
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: ir")

# allocate internal data and set default options
options = ir.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = ir.information()
print('** ir exit status:', inform['status'])

# deallocate internal data

ir.terminate()
