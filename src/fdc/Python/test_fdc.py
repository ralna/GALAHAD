from galahad import fdc
import numpy as np

#  describe problem:
#  ( 1  2 3  4 )     (  5 )
#  ( 2 -4 6 -8 ) x = ( 10 )
#  (    5   10 )     (  0 )

m = 3
n = 4
A_ptr = np.array([0,4,8,10])
A_col = np.array([0,1,2,3,0,1,2,3,1,3])
A_val = np.array([1.0,2.0,3.0,4.0,2.0,-4.0,6.0,-8.0,5.0,10.0])
b = np.array([5.0,10.0,0.0])

# allocate internal data and set default options
options = fdc.initialize()

# set some non-default options
#options['print_level'] = 1
options['use_sls'] = True
options['symmetric_linear_solver'] = "sytr "
#print("options:", options)

# load data (and optionally non-default options), and find dependent rows
n_depen, depen, inform = fdc.find_dependent_rows(m, n, A_ptr, A_col, A_val,
                                                 b, options)
print("# dependent rows:", n_depen)
print("these are:", depen[0:n_depen])

# deallocate internal data

fdc.terminate()
