from galahad import gltr
import numpy as np
np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
print("\n** python test: gltr")

# set problem data

n = 100
v = np.empty(n)
hv = np.empty(n)
g = np.empty(n)
g[:] = 1.0

# set default options
options = gltr.initialize()

# set some non-default options
options['unitm'] = False
options['print_level'] = 0
#print("options:", options)

# load non-default options
gltr.load_options(options)

# first problem

print(" solve problem 1")
status = 1
radius = 1.0
r = g
while status > 0:
  status, x, r, v = gltr.solve_problem(status, n, radius, r, v)
  if status == 2: # precondition v -> v/2
    for i in range(n):
      v[i] = v[i] / 2.0;
  elif status == 3: # form v -> H v
    hv[0] = 2.0 * v[0] + v[1]
    for i in range(1,n-1):
       hv[i] = v[i-1] + 2.0 * v[i] + v[i+1]
    hv[n-1] = v[n-2] + 2.0 * v[n-1]
    for i in range(n):
      v[i] = hv[i]
  elif status == 5: # reset r
    r = g

# get information

inform = gltr.information()
print(" optimal f: %.4f" % inform['obj'])
#print(" x:",x)

# second problem, same data with smaller radius

print("\n solve problem 2 with smaller radius")
status = 4
radius = 0.1
r = g
while status > 0:
  status, x, r, v = gltr.solve_problem(status, n, radius, r, v)
  if status == 2: # precondition v -> v/2
    for i in range(n):
      v[i] = v[i] / 2.0;
  elif status == 3: # form v -> H v
    hv[0] =  2.0 * v[0] + v[1]
    for i in range(1,n-1):
       hv[i] = v[i-1] + 2.0 * v[i] + v[i+1]
    hv[n-1] = v[n-2] + 2.0 * v[n-1]
    for i in range(n):
      v[i] = hv[i]
  elif status == 5: # reset r
    r = g

# get information

inform = gltr.information()
print(" optimal f: %.4f" % inform['obj'])
#print(" x:",x)
print('** gltr exit status:', inform['status'])

# deallocate internal data

gltr.terminate()
