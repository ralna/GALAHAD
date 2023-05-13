from galahad import l2rt
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: l2rt")

# set problem data

n = 50
m = 2 * n
shift = 1.0
power = 3.0
weight = 1.0
u = np.empty(m)
v = np.empty(n)
b = np.empty(m)
b[:] = 1.0

# set default options
options = l2rt.initialize()

# set some non-default options
options['unitm'] = False
options['print_level'] = 0
#print("options:", options)

# load non-default options
l2rt.load_options(options)

# problem

print(" solve problem")
status = 1
u = b
while status > 0:
  status, x, u, v = l2rt.solve_problem(status, m, n, power, weight, shift, u, v)
  if status == 2: # u -> u + A v
    for i in range(n):
      u[i] = u[i] + v[i]
      u[n+i] = u[n+i] + (i+1) * v[i]
  elif status == 3: # v -> v + A^T u
    for i in range(n):
      v[i] = v[i] + u[i] + (i+1) * u[n+i]
  elif status == 4: # reset r to b
    u = b

# get information

inform = l2rt.information()
print(" optimal f: %.4f" % inform['obj'])
#print(" x:",x)
print('** l2rt exit status:', inform['status'])

# deallocate internal data

l2rt.terminate()
