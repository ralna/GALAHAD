/* bnlstf.c */
/* Full test for the BNLS interface using Fortran sparse matrix indexing */
/* Jari Fowkes  Nick Gould, STFC-Rutherford Appleton Laboratory, 2024 */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_bnls.h"

# Define imax
int imax(int a, int b) {
return (a > b) ? a : b
]

# Custom userdata struct
struct userdata_type {
   real_wp_ p
]

# Function prototypes

int res(int n, int m, var::Vector{T}, real_wp_ c[], const void *)
int jac(int n, int m, int jne, var::Vector{T}, real_wp_ jval[],
 const void *)
int hess(int n, int m, int hne, var::Vector{T}, const real_wp_ y[],
  hval::Vector{T}, const void *)
int jacprod(int n, int m, var::Vector{T}, const bool transpose,
 u::Vector{T}, var::Vector{T}, bool got_j, const void *)
int hessprod(int n, int m, var::Vector{T}, const real_wp_ y[],
  u::Vector{T}, var::Vector{T}, bool got_h, const void *)
int rhessprods(int n, int m, int pne, var::Vector{T}, var::Vector{T},
real_wp_ pval[], bool got_h, const void *)
int scale(int n, int m, var::Vector{T}, u::Vector{T},
   var::Vector{T}, const void *)
int jac_dense(int n, int m, int jne, var::Vector{T}, real_wp_ jval[],
   const void *)
int hess_dense(int n, int m, int hne, var::Vector{T}, const real_wp_ y[],
hval::Vector{T}, const void *)
int rhessprods_dense(int n, int m, int pne, var::Vector{T},
  var::Vector{T}, real_wp_ pval[], bool got_h,
  const void *)

# test_bnls.jl
# Simple code to test the Julia interface to BNLS

using GALAHAD
using Test
using Printf
using Accessors

function test_bnls(::Type{T}, ::Type{INT}) where {T,INT}
# Derived types
data = Ref{Ptr{Cvoid}}()
control = Ref{bnls_control_type{T}}()
inform = Ref{bnls_inform_type{T}}()

# Set user data
struct userdata_type userdata
userdata.p = 1.0

# Set problem data
n = 2 # # variables
m = 3 # # residuals
j_ne = 5 # Jacobian elements
h_ne = 2 # Hesssian elements
p_ne = 2 # residual-Hessians-vector products elements
J_row = INT[1, 2, 2, 3, 3]  # Jacobian J
J_col = INT[1, 1, 2, 1, 2]  #
J_ptr = INT[1, 2, 4, 6]  # row pointers
H_row = INT[1, 2]  # Hessian H
H_col = INT[1, 2]  # NB lower triangle
H_ptr = INT[1, 2, 3]  # row pointers
P_row = INT[1, 2]  # residual-Hessians-vector product matrix
P_ptr = INT[1, 2, 3, 3]  # column pointers

# Set storage
g = zeros(T, n) # gradient
c = zeros(T, m) # residual
real_wp_ y[m] # multipliers
st = ' ' = ' '
status = Ref{INT}()

@printf(" Fortran sparse matrix indexing\n\n")

@printf(" tests options for all-in-one storage format\n\n")

for d = 1:5
#  for(int d=5 d <= 5 d++)

# Initialize BNLS
bnls_initialize(T, INT, data, control, inform)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
# control[].print_level = 1
control[].jacobian_available = 2
control[].hessian_available = 2
control[].model = 6
x = T[1.5,1.5]  # starting point
W = T[1.0, 1.0, 1.0]  # weights

# sparse co-ordinate storage
if d == 1
st = 'C'
bnls_import(T, INT, control, data, status, n, m,
"coordinate", j_ne, J_row, J_col, C_NULL,
"coordinate", h_ne, H_row, H_col, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
bnls_solve_with_mat(T, INT, data, userdata, status,
n, m, x, c, g, res, j_ne, jac,
h_ne, hess, p_ne, rhessprods)
end
# sparse by rows
if d == 2
st = 'R'
bnls_import(T, INT, control, data, status, n, m,
"sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
"sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
bnls_solve_with_mat(T, INT, data, userdata, status,
n, m, x, c, g, res, j_ne, jac,
h_ne, hess, p_ne, rhessprods)
end
# dense
if d == 3
st = 'D'
bnls_import(T, INT, control, data, status, n, m,
"dense", j_ne, C_NULL, C_NULL, C_NULL,
"dense", h_ne, C_NULL, C_NULL, C_NULL,
"dense", p_ne, C_NULL, C_NULL, C_NULL, W)
bnls_solve_with_mat(T, INT, data, userdata, status,
n, m, x, c, g, res, j_ne, jac_dense,
h_ne, hess_dense, p_ne, rhessprods_dense)
end
# diagonal
if d == 4
st = 'I'
bnls_import(T, INT, control, data, status, n, m,
"sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
"diagonal", h_ne, C_NULL, C_NULL, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
bnls_solve_with_mat(T, INT, data, userdata, status,
n, m, x, c, g, res, j_ne, jac,
h_ne, hess, p_ne, rhessprods)
end
case 5: # access by products
st = 'P'
bnls_import(T, INT, control, data, status, n, m,
"absent", j_ne, C_NULL, C_NULL, C_NULL,
"absent", h_ne, C_NULL, C_NULL, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
bnls_solve_without_mat(T, INT, data, userdata, status,
   n, m, x, c, g, res, jacprod,
   hessprod, p_ne, rhessprods)
end
]

bnls_information(T, INT, data, inform, status)

if inform[].status == 0
@printf("%c:%6i iterations. Optimal objective value = %5.2f"
   " status = %1i\n",
   st, inform[].iter, inform[].obj, inform[].status)
else
@printf("%c: BNLS_solve exit status = %1i\n", st, inform[].status)

# Delete internal workspace
bnls_terminate(T, INT, data, control, inform)
@printf("\n tests reverse-communication options\n\n")

# reverse-communication input/output
int eval_status
real_wp_ u[imax(m,n)], v[imax(m,n)]
real_wp_ J_val[j_ne], J_dense[m*n]
real_wp_ H_val[h_ne], H_dense[n*(n+1)/2], H_diag[n]
real_wp_ P_val[p_ne], P_dense[m*n]
bool transpose
bool got_j = false
bool got_h = false

for d = 1:5
#  for(int d=1 d <= 4 d++)

# Initialize BNLS
bnls_initialize(T, INT, data, control, inform)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
#control[].print_level = 1
control[].jacobian_available = 2
control[].hessian_available = 2
control[].model = 6
x = T[1.5,1.5]  # starting point
W = T[1.0, 1.0, 1.0]  # weights

# sparse co-ordinate storage
if d == 1
st = 'C'
bnls_import(T, INT, control, data, status, n, m,
"coordinate", j_ne, J_row, J_col, C_NULL,
"coordinate", h_ne, H_row, H_col, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
while true # reverse-communication loop
  bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
  n, m, x, c, g, j_ne, J_val, y,
  h_ne, H_val, v, p_ne, P_val)
  if status == 0 # successful termination
end
  elseif status < 0) # error exit
  end
  elseif status == 2) # evaluate c
  eval_status = res(n, m, x, c, userdata)
  elseif status == 3) # evaluate J
  eval_status = jac(n, m, j_ne, x, J_val, userdata)
  elseif status == 4) # evaluate H
  eval_status = hess(n, m, h_ne, x, y, H_val, userdata)
  elseif status == 7) # evaluate P
  eval_status = rhessprods(n, m, p_ne, x, v, P_val,
got_h, userdata)
  else
  @printf(" the value %1i of status should not occur\n",
status)
  end
 ]
]
end
# sparse by rows
if d == 2
st = 'R'
bnls_import(T, INT, control, data, status, n, m,
"sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
"sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
while true # reverse-communication loop
  bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
  n, m, x, c, g, j_ne, J_val, y,
  h_ne, H_val, v, p_ne, P_val)
  if status == 0 # successful termination
end
  elseif status < 0) # error exit
  end
  elseif status == 2) # evaluate c
  eval_status = res(n, m, x, c, userdata)
  elseif status == 3) # evaluate J
  eval_status = jac(n, m, j_ne, x, J_val, userdata)
  elseif status == 4) # evaluate H
  eval_status = hess(n, m, h_ne, x, y, H_val, userdata)
  elseif status == 7) # evaluate P
  eval_status = rhessprods(n, m, p_ne, x, v, P_val,
got_h, userdata)
  else
  @printf(" the value %1i of status should not occur\n",
status)
  end
 ]
]
end
# dense
if d == 3
st = 'D'
bnls_import(T, INT, control, data, status, n, m,
"dense", j_ne, C_NULL, C_NULL, C_NULL,
"dense", h_ne, C_NULL, C_NULL, C_NULL,
"dense", p_ne, C_NULL, C_NULL, C_NULL, W)
while true # reverse-communication loop
  bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
  n, m, x, c, g, m*n, J_dense, y,
  n*(n+1)/2, H_dense, v, m*n,
  P_dense)
  if status == 0 # successful termination
end
  elseif status < 0) # error exit
  end
  elseif status == 2) # evaluate c
  eval_status = res(n, m, x, c, userdata)
  elseif status == 3) # evaluate J
  eval_status = jac_dense(n, m, j_ne, x, J_dense,
   userdata)
  elseif status == 4) # evaluate H
  eval_status = hess_dense(n, m, h_ne, x, y, H_dense,
userdata)
  elseif status == 7) # evaluate P
  eval_status = rhessprods_dense(n, m, p_ne, x, v, P_dense,
  got_h, userdata)
  else
  @printf(" the value %1i of status should not occur\n",
status)
  end
 ]
]
end
# diagonal
if d == 4
st = 'I'
bnls_import(T, INT, control, data, status, n, m,
"sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
"diagonal", h_ne, C_NULL, C_NULL, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
while true # reverse-communication loop
  bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
  n, m, x, c, g, j_ne, J_val, y,
  n, H_diag, v, p_ne, P_val)
  if status == 0 # successful termination
end
  elseif status < 0) # error exit
  end
  elseif status == 2) # evaluate c
  eval_status = res(n, m, x, c, userdata)
  elseif status == 3) # evaluate J
  eval_status = jac(n, m, j_ne, x, J_val, userdata)
  elseif status == 4) # evaluate H
  eval_status = hess(n, m, h_ne, x, y, H_diag, userdata)
  elseif status == 7) # evaluate P
  eval_status = rhessprods(n, m, p_ne, x, v, P_val,
got_h, userdata)
  else
  @printf(" the value %1i of status should not occur\n",
status)
  end
 ]
]
end
case 5: # access by products
st = 'P'
#  control[].print_level = 1
bnls_import(T, INT, control, data, status, n, m,
"absent", j_ne, C_NULL, C_NULL, C_NULL,
"absent", h_ne, C_NULL, C_NULL, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
while true # reverse-communication loop
  bnls_solve_reverse_without_mat(T, INT, data, status, eval_status,
 n, m, x, c, g, transpose,
 u, v, y, p_ne, P_val)
  if status == 0 # successful termination
end
  elseif status < 0) # error exit
  end
  elseif status == 2) # evaluate c
  eval_status = res(n, m, x, c, userdata)
  elseif status == 5) # evaluate u + J v or u + J'v
  eval_status = jacprod(n, m, x, transpose, u, v, got_j,
 userdata)
  elseif status == 6) # evaluate u + H v
  eval_status = hessprod(n, m, x, y, u, v, got_h,
  userdata)
  elseif status == 7) # evaluate P
  eval_status = rhessprods(n, m, p_ne, x, v, P_val,
got_h, userdata)
  else
  @printf(" the value %1i of status should not occur\n",
status)
  end
 ]
]
end
]

bnls_information(T, INT, data, inform, status)

if inform[].status == 0
@printf("%c:%6i iterations. Optimal objective value = %5.2f"
   " status = %1i\n",
   st, inform[].iter, inform[].obj, inform[].status)
else
@printf("%c: BNLS_solve exit status = %1i\n", st, inform[].status)

# Delete internal workspace
bnls_terminate(T, INT, data, control, inform)
@printf("\n basic tests of models used, direct access\n\n")

for(int model=3 model <= 8 model++)

# Initialize BNLS
bnls_initialize(T, INT, data, control, inform)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
#control[].print_level = 1
control[].jacobian_available = 2
control[].hessian_available = 2
control[].model = model
x = T[1.5,1.5]  # starting point
W = T[1.0, 1.0, 1.0]  # weights

bnls_import(T, INT, control, data, status, n, m,
"sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
"sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
bnls_solve_with_mat(T, INT, data, userdata, status,
n, m, x, c, g, res, j_ne, jac,
h_ne, hess, p_ne, rhessprods)

bnls_information(T, INT, data, inform, status)

if inform[].status == 0
@printf(" %1i:%6" i_int 
   " iterations. Optimal objective value = %5.2f"
   " status = %1i\n",
   model, inform[].iter, inform[].obj, inform[].status)
else
@printf(" %i: BNLS_solve exit status = %1" i_int 
   "\n", model, inform[].status)

# Delete internal workspace
bnls_terminate(T, INT, data, control, inform)
@printf("\n basic tests of models used, access by products\n\n")

for(int model=3 model <= 8 model++)

# Initialize BNLS
bnls_initialize(T, INT, data, control, inform)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
#control[].print_level = 1
control[].jacobian_available = 2
control[].hessian_available = 2
control[].model = model
x = T[1.5,1.5]  # starting point
W = T[1.0, 1.0, 1.0]  # weights

bnls_import(T, INT, control, data, status, n, m,
"absent", j_ne, C_NULL, C_NULL, C_NULL,
"absent", h_ne, C_NULL, C_NULL, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
bnls_solve_without_mat(T, INT, data, userdata, status,
   n, m, x, c, g, res, jacprod,
   hessprod, p_ne, rhessprods)
bnls_information(T, INT, data, inform, status)

if inform[].status == 0
@printf("P%1i:%6i iterations. Optimal objective value = %5.2f"
   " status = %1i\n",
   model, inform[].iter, inform[].obj, inform[].status)
else
@printf("P%i: BNLS_solve exit status = %1i\n", model, inform[].status)

# Delete internal workspace
bnls_terminate(T, INT, data, control, inform)
@printf("\n basic tests of models used, reverse access\n\n")

for(int model=3 model <= 8 model++)

# Initialize BNLS
bnls_initialize(T, INT, data, control, inform)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
#control[].print_level = 1
control[].jacobian_available = 2
control[].hessian_available = 2
control[].model = model
x = T[1.5,1.5]  # starting point
W = T[1.0, 1.0, 1.0]  # weights

bnls_import(T, INT, control, data, status, n, m,
"sparse_by_rows", j_ne, C_NULL, J_col, J_ptr,
"sparse_by_rows", h_ne, C_NULL, H_col, H_ptr,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
while true # reverse-communication loop
  bnls_solve_reverse_with_mat(T, INT, data, status, eval_status,
  n, m, x, c, g, j_ne, J_val, y,
  h_ne, H_val, v, p_ne, P_val)
  if status == 0 # successful termination
end
  elseif status < 0) # error exit
  end
  elseif status == 2) # evaluate c
  eval_status = res(n, m, x, c, userdata)
  elseif status == 3) # evaluate J
  eval_status = jac(n, m, j_ne, x, J_val, userdata)
  elseif status == 4) # evaluate H
  eval_status = hess(n, m, h_ne, x, y, H_val, userdata)
  elseif status == 7) # evaluate P
  eval_status = rhessprods(n, m, p_ne, x, v, P_val,
got_h, userdata)
  else
  @printf(" the value %1i of status should not occur\n",
status)
  end
 ]
]

bnls_information(T, INT, data, inform, status)

if inform[].status == 0
@printf("P%1i:%6i iterations. Optimal objective value = %5.2f"
   " status = %1i\n",
   model, inform[].iter, inform[].obj, inform[].status)
else
@printf(" %i: BNLS_solve exit status = %1i\n", model, inform[].status)

# Delete internal workspace
bnls_terminate(T, INT, data, control, inform)
@printf("\n basic tests of models used, reverse access by products\n\n")

for(int model=3 model <= 8 model++)

# Initialize BNLS
bnls_initialize(T, INT, data, control, inform)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
#control[].print_level = 1
control[].jacobian_available = 2
control[].hessian_available = 2
control[].model = model
x = T[1.5,1.5]  # starting point
W = T[1.0, 1.0, 1.0]  # weights

bnls_import(T, INT, control, data, status, n, m,
"absent", j_ne, C_NULL, C_NULL, C_NULL,
"absent", h_ne, C_NULL, C_NULL, C_NULL,
"sparse_by_columns", p_ne, P_row, C_NULL, P_ptr, W)
while true # reverse-communication loop
  bnls_solve_reverse_without_mat(T, INT, data, status, eval_status,
 n, m, x, c, g, transpose,
 u, v, y, p_ne, P_val)
  if status == 0 # successful termination
end
  elseif status < 0) # error exit
  end
  elseif status == 2) # evaluate c
  eval_status = res(n, m, x, c, userdata)
  elseif status == 5) # evaluate u + J v or u + J'v
  eval_status = jacprod(n, m, x, transpose, u, v, got_j,
 userdata)
  elseif status == 6) # evaluate u + H v
  eval_status = hessprod(n, m, x, y, u, v, got_h,
  userdata)
  elseif status == 7) # evaluate P
  eval_status = rhessprods(n, m, p_ne, x, v, P_val,
got_h, userdata)
  else
  @printf(" the value %1i of status should not occur\n",
status)
  end
 ]
]

bnls_information(T, INT, data, inform, status)

if inform[].status == 0
@printf("P%1i:%6i iterations. Optimal objective value = %5.2f"
   " status = %1i\n",
   model, inform[].iter, inform[].obj, inform[].status)
else
@printf("P%i: BNLS_solve exit status = %1i\n", model, inform[].status)

# Delete internal workspace
bnls_terminate(T, INT, data, control, inform)
# compute the residuals
int res(int n, int m, var::Vector{T}, real_wp_ c[], const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ p = myuserdata->p
c[0] = pow(x[0],2.0) + p
c[1] = x[0] + pow(x[1],2.0)
c[2] = x[0] - x[1]
return 0
]

# compute the Jacobian
int jac(int n, int m, int jne, var::Vector{T}, real_wp_ jval[],
 const void *userdata)
jval[0] = 2.0 * x[0]
jval[1] = 1.0
jval[2] = 2.0 * x[1]
jval[3] = 1.0
jval[4] = - 1.0
return 0
]

# compute the Hessian
int hess(int n, int m, int hne, var::Vector{T}, const real_wp_ y[],
   hval::Vector{T}, const void *userdata)
hval[0] = 2.0 * y[0]
hval[1] = 2.0 * y[1]
return 0
]

# compute Jacobian-vector products
int jacprod(int n, int m, var::Vector{T}, const bool transpose, u::Vector{T},
 var::Vector{T}, bool got_j, const void *userdata)
if (transpose) {
  u[0] = u[0] + 2.0 * x[0] * v[0] + v[1] + v[2]
  u[1] = u[1] + 2.0 * x[1] * v[1] - v[2]
else
  u[0] = u[0] + 2.0 * x[0] * v[0]
  u[1] = u[1] + v[0]  + 2.0 * x[1] * v[1]
  u[2] = u[2] + v[0] - v[1]
]
return 0
]

# compute Hessian-vector products
int hessprod(int n, int m, var::Vector{T}, const real_wp_ y[], u::Vector{T},
  var::Vector{T}, bool got_h, const void *userdata)
u[0] = u[0] + 2.0 * y[0] * v[0]
u[1] = u[1] + 2.0 * y[1] * v[1]
return 0
]

# compute residual-Hessians-vector products
int rhessprods(int n, int m, int pne, var::Vector{T}, var::Vector{T},
real_wp_ pval[], bool got_h, const void *userdata)
pval[0] = 2.0 * v[0]
pval[1] = 2.0 * v[1]
return 0
]

# scale v
int scale(int n, int m, var::Vector{T}, u::Vector{T},
   var::Vector{T}, const void *userdata)
u[0] = v[0]
u[1] = v[1]
return 0
]

# compute the dense Jacobian
int jac_dense(int n, int m, int jne, var::Vector{T}, real_wp_ jval[],
   const void *userdata)
jval[0] = 2.0 * x[0]
jval[1] = 0.0
jval[2] = 1.0
jval[3] = 2.0 * x[1]
jval[4] = 1.0
jval[5] = - 1.0
return 0
]

# compute the dense Hessian
int hess_dense(int n, int m, int hne, var::Vector{T}, const real_wp_ y[],
hval::Vector{T}, const void *userdata)
hval[0] = 2.0 * y[0]
hval[1] = 0.0
hval[2] = 2.0 * y[1]
return 0
]

# compute dense residual-Hessians-vector products
int rhessprods_dense(int n, int m, int pne, var::Vector{T},
  var::Vector{T}, real_wp_ pval[], bool got_h,
  const void *userdata)
pval[0] = 2.0 * v[0]
pval[1] = 0.0
pval[2] = 0.0
pval[3] = 2.0 * v[1]
pval[4] = 0.0
pval[5] = 0.0
return 0
]
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "BNLS -- $T -- $INT" begin
      @test test_bnls(T, INT) == 0
    end
  end
end
