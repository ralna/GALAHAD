from galahad import nodend
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: nodend")

# allocate internal data and set default options
options = nodend.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = nodend.information()
print('** nodend exit status:', inform['status'])

