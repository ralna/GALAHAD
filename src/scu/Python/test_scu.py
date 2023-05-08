from galahad import scu
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')

# call main solver(s)
# ....

# get information
inform = scu.information()
print(" status:",inform['status'])

# deallocate internal data

scu.terminate()
