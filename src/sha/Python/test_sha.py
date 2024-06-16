from galahad import sha
import numpy as np
import random
np.set_printoptions(precision=2,suppress=True,floatmode='fixed')
print("\n** python test: sha")

# set parameters
n = 5
ne = 9
extra_differences = 1

#  Hessian co-ordinate indices, NB upper triangle

row = np.array([0,0,0,0,0,1,2,3,4])
col = np.array([0,1,2,3,4,1,2,3,4])
#val = np.empty(ne)

#  set space for the differences s and y

m_max = 9
ls1 = m_max
ls2 = n
ly1 = m_max
ly2 = n
strans = np.empty([ls1,ls2])
ytrans = np.empty([ly1,ly2])

#  for reference, the Hessian H we shall try to recover
val_h = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])

for algorithm in range(1,6):

# allocate internal data and set default options
  options = sha.initialize()

  # set some non-default options
  options['approximation_algorithm'] = algorithm
  options['extra_differences'] = extra_differences
  # print("options:", options)

  # analyse the co-ordinate structure
  m = sha.analyse_matrix(n, ne, row, col, options)
  print('\n Algorithm ' + str(algorithm) + ' - ' + str(m) +
        ' differences required, one extra might help')

  # add a little flexibility just in case of (accidental) dependence of s
  m = m + extra_differences
  order = np.empty(m,dtype=int)

  # construct random s in [-1,1]
  random.seed(a=1)
  for k in range(m):

    # assign a precedence order
    order[k] = m - k - 1

    # compute s and initialize y as 0
    for i in range(n):
      rr = random.random()
      strans[k,i] = -1.0 + 2.0 * rr
      ytrans[k,i] = 0.0

    # construct y = H s
    for l in range(ne):
      i = row[l]
      j = col[l]
      v = val_h[l]
      ytrans[k,i] = ytrans[k,i] + v * strans[k,j]
      if i != j: ytrans[k,j] = ytrans[k,j] + v * strans[k,i]

  # recover the values of the Hessian
  val = sha.recover_matrix(ne, m, ls1, ls2, strans, ly1, ly2, ytrans, order)

  print(" H from", m, "differences:", val)

  # recover the values of the Hessian
  print(" now use default order")
  val = sha.recover_matrix(ne, m, ls1, ls2, strans, ly1, ly2, ytrans )

  print(" H from", m, "differences:", val)

  # get information
  inform = sha.information()
  print(' ** sha exit status:', inform['status'])

# deallocate internal data

  sha.terminate()
