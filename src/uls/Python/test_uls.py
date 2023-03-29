from galahad import uls
import numpy as np

#  describe problem:
#  ( 1     )     ( 1 )
#  (   2 2 ) x = ( 4 )
#  (   1 3 )     ( 4 )

m = 3
n = 3
A_type = 'coordinate'
A_ne = 5
A_row = np.array([0,1,1,2,2])
A_col = np.array([0,1,2,1,2])
A_ptr = None
A_val = np.array([1.0,2.0,2.0,1.0,3.0])
b = np.array([1.0,4.0,4.0])

# set solver name, allocate internal data and set default options
solver = 'getr'
options = uls.initialize(solver)

# set some non-default options
options['print_level'] = 1
#print("options:", options

# load data (and optionally non-default options), and factorize matrix
uls.factorize_matrix(m, n, A_type, A_ne, A_row, A_col, A_ptr, A_val, options)

# solve system
trans = False
x = uls.solve_system(m, n, b, trans)

x_copy=x.copy()
print("x:",x_copy)

# solve transpose system
b = np.array([1.0,3.0,5.0])
trans = True
xt = uls.solve_system(m, n, b, trans)

xt_copy=xt.copy()
print("transpose x:",x_copy)

# get information
inform = uls.information()
print("rank:",inform['rank'])

# deallocate internal data

uls.terminate()
