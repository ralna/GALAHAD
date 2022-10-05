from galahad import ugo
import numpy as np

# allocate internal data and set default options
ugo.initialize()

# set some non-default options
options = {'print_level' : 3}
print(options)

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
print("x:",x)
print("f:",f)
print("g:",g)
print("h:",h)

# get information
inform = ugo.information()
print(inform)

# deallocate internal data
ugo.terminate()
