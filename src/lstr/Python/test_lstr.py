from galahad import lstr
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: lstr")

# set problem data

n = 50
m = 2 * n
u = np.empty(m)
v = np.empty(n)
b = np.empty(m)
b[:] = 1.0

# set default options
options = lstr.initialize()

# set some non-default options
options['unitm'] = False
options['print_level'] = 0
#print("options:", options)

# load non-default options
lstr.load_options(options)

# first problem

print(" solve problem 1")
status = 1
radius = 1.0
u = b
while status > 0:
  status, x, u, v = lstr.solve_problem(status, m, n, radius, u, v)
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

inform = lstr.information()
print(" optimal ||r||: %.4f" % inform['r_norm'])
#print(" x:",x)

# second problem, same data with smaller radius

print("\n solve problem 2 with smaller radius")
status = 5
radius = 0.1
u = b
while status > 0:
  status, x, u, v = lstr.solve_problem(status, m, n, radius, u, v)
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

inform = lstr.information()
print(" optimal ||r||: %.4f" % inform['r_norm'])
#print(" x:",x)
print('** lstr exit status:', inform['status'])

# deallocate internal data

lstr.terminate()
