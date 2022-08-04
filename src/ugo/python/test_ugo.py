from galahad import ugo

# allocate internal data and set default options
ugo.initialize()

# set some non-default options
options = {'print_level' : 3}
print(options)

# load data (and optionally non-default options)
x_l = 3
x_u = 2.4
ugo.load(x_l,x_u,options=options)

# define objective function
# NB python functions have access to external variables
# So no need for userdata like in C or Fortran
def eval_fgh(x):
    return x*x, 2*x, 2

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
