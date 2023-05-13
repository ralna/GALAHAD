from galahad import convert
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: convert")

# allocate internal data and set default options
options = convert.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# call main solver(s)
# ....

# get information
inform = convert.information()
print('** convert exit status:', inform['status'])

# deallocate internal data

convert.terminate()
