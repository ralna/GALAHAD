from galahad import ugo
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: ugo")

# allocate internal data and set default options
options = ugo.initialize()

# set some non-default options
options['print_level'] = 0
#print("options:", options)

# load data (and optionally non-default options)
x_l = -1
x_u = 2
ugo.load(x_l,x_u,options=options)

# define objective function
# NB python functions have access to external variables
# So no need for userdata like in C or Fortran
def eval_fgh(x):
   a = 10
   f = x * x * np.cos( a*x )
   g = - a * x * x * np.sin( a*x ) + 2.0 * x * np.cos( a*x )
   h = - a * a* x * x * np.cos( a*x ) - 4.0 * a * x * np.sin( a*x ) \
       + 2.0 * np.cos( a*x )
   return f, g, h


# find optimum
x, f, g, h = ugo.solve(eval_fgh)
print(" x: %.4f" % x)
print(" f: %.4f" % f)
print(" g: %.4f" % g)
print(" h: %.4f" % h)

# get information
inform = ugo.information()
print('** ugo exit status:', inform['status'])

# deallocate internal data
ugo.terminate()
