from galahad import lsrt
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: lsrt")

# set problem data

n = 50
m = 2 * n
power = 3.0
u = np.empty(m)
v = np.empty(n)
b = np.empty(m)
b[:] = 1.0

# set default options
options = lsrt.initialize()

# set some non-default options
options['unitm'] = False
options['print_level'] = 0
#print("options:", options)

# load non-default options
lsrt.load_options(options)

# problem

print(" solve problem")
status = 1
weight = 1.0
u = b
while status > 0:
  status, x, u, v = lsrt.solve_problem(status, m, n, power, weight, u, v)
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

inform = lsrt.information()
print(" optimal f: %.4f" % inform['obj'])
#print(" x:",x)
print('** lsrt exit status:', inform['status'])

# deallocate internal data

lsrt.terminate()
