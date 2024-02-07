/* bgot2.c */
/* Full test for the BGO C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_bgo.h"

# Custom userdata struct
struct userdata_type {
   real_wp_ p
   real_wp_ freq
   real_wp_ mag
]

# Function prototypes
int fun(int n, const real_wp_ x[], real_wp_ *f, const void *)
int grad(int n, const real_wp_ x[], real_wp_ g[], const void *)
int hess(int n, int ne, const real_wp_ x[], real_wp_ hval[], const void *)
int hess_dense(int n, int ne, const real_wp_ x[], real_wp_ hval[],
const void *)
int hessprod(int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[],
  bool got_h, const void *)
int shessprod(int n, const real_wp_ x[], int nnz_v, const int index_nz_v[],
   const real_wp_ v[], int *nnz_u, int index_nz_u[], real_wp_ u[],
   bool got_h, const void *)
int prec(int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[],
  const void *)
int fun_diag(int n, const real_wp_ x[], real_wp_ *f, const void *)
int grad_diag(int n, const real_wp_ x[], real_wp_ g[], const void *)
int hess_diag(int n, int ne, const real_wp_ x[], real_wp_ hval[],
   const void *)
int hessprod_diag(int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[],
   bool got_h, const void *)
int shessprod_diag(int n, const real_wp_ x[], int nnz_v,
const int index_nz_v[],
const real_wp_ v[], int *nnz_u, int index_nz_u[],
real_wp_ u[], bool got_h, const void *)

# test_bgo.jl
# Simple code to test the Julia interface to BGO

using GALAHAD
using Test
using Printf
using Accessors

function test_bgo()
# Derived types
data = Ref{Ptr{Cvoid}}()
control = Ref{bgo_control_type{Float64}}()
inform = Ref{bgo_inform_type{Float64}}()

# Set user data
struct userdata_type userdata
userdata.p = 4.0
userdata.freq = 10
userdata.mag = 1000

# Set problem data
n = 3 # dimension
ne = 5 # Hesssian elements
x_l = Float64[-10,-10,-10]
x_u = Float64[0.5,0.5,0.5]
H_row = Cint[1, 2, 3, 3, 3]  # Hessian H
H_col = Cint[1, 2, 1, 2, 3]  # NB lower triangle
H_ptr = Cint[1, 2, 3, 6]  # row pointers

# Set storage
g = zeros(Float64, n) # gradient
st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")

@printf(" tests options for all-in-one storage format\n\n")

for(int d=1 d <= 5 d++)

# Initialize BGO
bgo_initialize(data, control, status)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
control[].attempts_max = 10000
control[].max_evals = 20000
control[].sampling_strategy = 3
control[].trb_control[].maxit = 100
#control[].print_level = 1

# Start from 0
x = Float64[0,0,0]

switch(d)
# sparse co-ordinate storage
if d == 1
st = 'C'
bgo_import(control, data, status, n, x_l, x_u,
"coordinate", ne, H_row, H_col, Cint[])
bgo_solve_with_mat(data, userdata, status, n, x, g,
ne, fun, grad, hess, hessprod, prec)
end
# sparse by rows
if d == 2
st = 'R'
bgo_import(control, data, status, n, x_l, x_u,
"sparse_by_rows", ne, Cint[], H_col, H_ptr)
bgo_solve_with_mat(data, userdata, status, n, x, g,
ne, fun, grad, hess, hessprod, prec)
end
# dense
if d == 3
st = 'D'
bgo_import(control, data, status, n, x_l, x_u,
"dense", ne, Cint[], Cint[], Cint[])
bgo_solve_with_mat(data, userdata, status, n, x, g,
ne, fun, grad, hess_dense, hessprod, prec)
end
# diagonal
if d == 4
st = 'I'
bgo_import(control, data, status, n, x_l, x_u,
"diagonal", ne, Cint[], Cint[], Cint[])
bgo_solve_with_mat(data, userdata, status, n, x, g,
ne, fun_diag, grad_diag, hess_diag,
hessprod_diag, prec)
end
case 5: # access by products
st = 'P'
bgo_import(control, data, status, n, x_l, x_u,
"absent", ne, Cint[], Cint[], Cint[])
bgo_solve_without_mat(data, userdata, status, n, x, g,
   fun, grad, hessprod, shessprod, prec)
end
]

# Record solution information
bgo_information(data, inform, status)

if inform[].status == 0
@printf("%c:%6i evaluations. Optimal objective value = %5.2f"
  " status = %1i\n", st, inform[].f_eval, inform[].obj, inform[].status)
else
@printf("%c: BGO_solve exit status = %1i\n", st, inform[].status)

# @printf("x: ")
# for(int i = 0 i < n i++) @printf("%f ", x[i])
# @printf("\n")
# @printf("gradient: ")
# for(int i = 0 i < n i++) @printf("%f ", g[i])
# @printf("\n")

# Delete internal workspace
bgo_terminate(data, control, inform)
@printf("\n tests reverse-communication options\n\n")

# reverse-communication input/output
int eval_status, nnz_u, nnz_v
f = 0.0
u = zeros(Float64, n), v[n]
index_nz_u = zeros(Cint, n), index_nz_v[n]
real_wp_ H_val[ne], H_dense[n*(n+1)/2], H_diag[n]

for(int d=1 d <= 5 d++)

# Initialize BGO
bgo_initialize(data, control, status)

# Set user-defined control options
control[].f_indexing = true # Fortran sparse matrix indexing
control[].attempts_max = 10000
control[].max_evals = 20000
control[].sampling_strategy = 3
control[].trb_control[].maxit = 100
#control[].print_level = 1

# Start from 0
x = Float64[0,0,0]

switch(d)
# sparse co-ordinate storage
if d == 1
st = 'C'
bgo_import(control, data, status, n, x_l, x_u,
"coordinate", ne, H_row, H_col, Cint[])
while true # reverse-communication loop
bgo_solve_reverse_with_mat(data, status, eval_status,
n, x, f, g, ne, H_val, u, v)
if status == 0 # successful termination
end
}elseif status < 0) # error exit
end
}elseif status == 2) # evaluate f
eval_status = fun(n, x, f, userdata)
elseif status == 3) # evaluate g
eval_status = grad(n, x, g, userdata)
elseif status == 4) # evaluate H
eval_status = hess(n, ne, x, H_val, userdata)
elseif status == 5) # evaluate Hv product
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 6) # evaluate the product with P
eval_status = prec(n, x, u, v, userdata)
elseif status == 23) # evaluate f and g
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
elseif status == 25) # evaluate f and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 35) # evaluate g and Hv product
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 235) # evaluate f, g and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
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
bgo_import(control, data, status, n, x_l, x_u,
"sparse_by_rows", ne, Cint[], H_col, H_ptr)
while true # reverse-communication loop
bgo_solve_reverse_with_mat(data, status, eval_status,
n, x, f, g, ne, H_val, u, v)
if status == 0 # successful termination
end
}elseif status < 0) # error exit
end
}elseif status == 2) # evaluate f
eval_status = fun(n, x, f, userdata)
elseif status == 3) # evaluate g
eval_status = grad(n, x, g, userdata)
elseif status == 4) # evaluate H
eval_status = hess(n, ne, x, H_val, userdata)
elseif status == 5) # evaluate Hv product
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 6) # evaluate the product with P
eval_status = prec(n, x, u, v, userdata)
elseif status == 23) # evaluate f and g
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
elseif status == 25) # evaluate f and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 35) # evaluate g and Hv product
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 235) # evaluate f, g and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
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
bgo_import(control, data, status, n, x_l, x_u,
"dense", ne, Cint[], Cint[], Cint[])
while true # reverse-communication loop
bgo_solve_reverse_with_mat(data, status, eval_status,
n, x, f, g, n*(n+1)/2,
H_dense, u, v)
if status == 0 # successful termination
end
}elseif status < 0) # error exit
end
}elseif status == 2) # evaluate f
eval_status = fun(n, x, f, userdata)
elseif status == 3) # evaluate g
eval_status = grad(n, x, g, userdata)
elseif status == 4) # evaluate H
eval_status = hess_dense(n, n*(n+1)/2, x, H_dense,
  userdata)
elseif status == 5) # evaluate Hv product
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 6) # evaluate the product with P
eval_status = prec(n, x, u, v, userdata)
elseif status == 23) # evaluate f and g
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
elseif status == 25) # evaluate f and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 35) # evaluate g and Hv product
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 235) # evaluate f, g and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
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
bgo_import(control, data, status, n, x_l, x_u,
"diagonal", ne, Cint[], Cint[], Cint[])
while true # reverse-communication loop
bgo_solve_reverse_with_mat(data, status, eval_status,
n, x, f, g, n, H_diag, u, v)
if status == 0 # successful termination
end
}elseif status < 0) # error exit
end
}elseif status == 2) # evaluate f
eval_status = fun_diag(n, x, f, userdata)
elseif status == 3) # evaluate g
eval_status = grad_diag(n, x, g, userdata)
elseif status == 4) # evaluate H
eval_status = hess_diag(n, n, x, H_diag, userdata)
elseif status == 5) # evaluate Hv product
eval_status = hessprod_diag(n, x, u, v, false,
 userdata)
elseif status == 6) # evaluate the product with P
eval_status = prec(n, x, u, v, userdata)
elseif status == 23) # evaluate f and g
eval_status = fun_diag(n, x, f, userdata)
eval_status = grad_diag(n, x, g, userdata)
elseif status == 25) # evaluate f and Hv product
eval_status = fun_diag(n, x, f, userdata)
eval_status = hessprod_diag(n, x, u, v, false,
 userdata)
elseif status == 35) # evaluate g and Hv product
eval_status = grad_diag(n, x, g, userdata)
eval_status = hessprod_diag(n, x, u, v, false,
 userdata)
elseif status == 235) # evaluate f, g and Hv product
eval_status = fun_diag(n, x, f, userdata)
eval_status = grad_diag(n, x, g, userdata)
eval_status = hessprod_diag(n, x, u, v, false,
 userdata)
else
@printf(" the value %1i of status should not occur\n",
   status)
end
]
]
end
case 5: # access by products
st = 'P'
bgo_import(control, data, status, n, x_l, x_u,
"absent", ne, Cint[], Cint[], Cint[])
nnz_u = 0
while true # reverse-communication loop
bgo_solve_reverse_without_mat(data, status, eval_status,
   n, x, f, g, u, v, index_nz_v,
   nnz_v, index_nz_u, nnz_u)
if status == 0 # successful termination
end
}elseif status < 0) # error exit
end
}elseif status == 2) # evaluate f
eval_status = fun(n, x, f, userdata)
elseif status == 3) # evaluate g
eval_status = grad(n, x, g, userdata)
elseif status == 5) # evaluate Hv product
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 6) # evaluate the product with P
eval_status = prec(n, x, u, v, userdata)
elseif status == 7) # evaluate sparse Hess-vect product
eval_status = shessprod(n, x, nnz_v, index_nz_v, v,
 nnz_u, index_nz_u, u,
 false, userdata)
elseif status == 23) # evaluate f and g
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
elseif status == 25) # evaluate f and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 35) # evaluate g and Hv product
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
elseif status == 235) # evaluate f, g and Hv product
eval_status = fun(n, x, f, userdata)
eval_status = grad(n, x, g, userdata)
eval_status = hessprod(n, x, u, v, false, userdata)
else
@printf(" the value %1i of status should not occur\n",
   status)
end
]
]
end
]

# Record solution information
bgo_information(data, inform, status)

if inform[].status == 0
@printf("%c:%6i evaluations. Optimal objective value = %5.2f"
  " status = %1i\n", st, inform[].f_eval, inform[].obj, inform[].status)
else
@printf("%c: BGO_solve exit status = %1i\n", st, inform[].status)

# @printf("x: ")
# for(int i = 0 i < n i++) @printf("%f ", x[i])
# @printf("\n")
# @printf("gradient: ")
# for(int i = 0 i < n i++) @printf("%f ", g[i])
# @printf("\n")

# Delete internal workspace
bgo_terminate(data, control, inform)
]

# Objective function
int fun(int n,
 const real_wp_ x[],
 real_wp_ *f,
 const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ p = myuserdata->p
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

*f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) + mag * cos(freq*x[0])
 + x[0] + x[1] + x[2]
return 0
]

# Gradient of the objective
int grad(int n,
  const real_wp_ x[],
  real_wp_ g[],
  const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ p = myuserdata->p
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

g[0] = 2.0 * (x[0] + x[2] + p) - mag * freq * sin(freq*x[0]) + 1
g[1] = 2.0 * (x[1] + x[2]) + 1
g[2] = 2.0 * (x[0] + x[2] + p) + 2.0 * (x[1] + x[2]) + 1
return 0
]

# Hessian of the objective
int hess(int n,
  int ne,
  const real_wp_ x[],
  real_wp_ hval[],
  const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

hval[0] = 2.0 - mag * freq * freq * cos(freq*x[0])
hval[1] = 2.0
hval[2] = 2.0
hval[3] = 2.0
hval[4] = 4.0
return 0
]

# Dense Hessian
int hess_dense(int n,
int ne,
const real_wp_ x[],
real_wp_ hval[],
const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

hval[0] = 2.0 - mag * freq * freq * cos(freq*x[0])
hval[1] = 0.0
hval[2] = 2.0
hval[3] = 2.0
hval[4] = 2.0
hval[5] = 4.0
return 0
]

# Hessian-vector product
int hessprod(int n,
  const real_wp_ x[],
  real_wp_ u[],
  const real_wp_ v[],
  bool got_h,
  const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

u[0] = u[0] + 2.0 * (v[0] + v[2])
   - mag * freq * freq * cos(freq*x[0]) * v[0]
u[1] = u[1] + 2.0 * (v[1] + v[2])
u[2] = u[2] + 2.0 * (v[0] + v[1] + 2.0 * v[2])
return 0
]

# Sparse Hessian-vector product
int shessprod(int n,
   const real_wp_ x[],
   int nnz_v,
   const int index_nz_v[],
   const real_wp_ v[],
   int *nnz_u,
   int index_nz_u[],
   real_wp_ u[],
   bool got_h,
   const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

real_wp_ p[] = {0., 0., 0.]
bool used[] = {false, false, false]
for(int i = 0 i < nnz_v i++)
int j = index_nz_v[i]
switch(j)
case 1:
p[0] = p[0] + 2.0 * v[0]
   - mag * freq * freq * cos(freq*x[0]) * v[0]
used[0] = true
p[2] = p[2] + 2.0 * v[0]
used[2] = true
end
case 2:
p[1] = p[1] + 2.0 * v[1]
used[1] = true
p[2] = p[2] + 2.0 * v[1]
used[2] = true
end
case 3:
p[0] = p[0] + 2.0 * v[2]
used[0] = true
p[1] = p[1] + 2.0 * v[2]
used[1] = true
p[2] = p[2] + 4.0 * v[2]
used[2] = true
end
]
]
*nnz_u = 0
for(int j = 0 j < 3 j++)
if used[j])
u[j] = p[j]
*nnz_u = *nnz_u + 1
index_nz_u[*nnz_u-1] = j+1
]
]
return 0
]

# Apply preconditioner
int prec(int n,
  const real_wp_ x[],
  real_wp_ u[],
  const real_wp_ v[],
  const void *userdata)
   u[0] = 0.5 * v[0]
   u[1] = 0.5 * v[1]
   u[2] = 0.25 * v[2]
   return 0
]

# Objective function
int fun_diag(int n,
  const real_wp_ x[],
  real_wp_ *f,
  const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ p = myuserdata->p
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

*f = pow(x[2] + p, 2) + pow(x[1], 2) + mag * cos(freq*x[0])
 + x[0] + x[1] + x[2]
return 0
]

# Gradient of the objective
int grad_diag(int n,
   const real_wp_ x[],
   real_wp_ g[],
   const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ p = myuserdata->p
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

g[0] = -mag * freq * sin(freq*x[0]) + 1
g[1] = 2.0 * x[1] + 1
g[2] = 2.0 * (x[2] + p) + 1
return 0
]

# Hessian of the objective
int hess_diag(int n,
   int ne,
   const real_wp_ x[],
   real_wp_ hval[],
   const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

hval[0] = -mag * freq * freq * cos(freq*x[0])
hval[1] = 2.0
hval[2] = 2.0
return 0
]

# Hessian-vector product
int hessprod_diag(int n,
   const real_wp_ x[],
   real_wp_ u[],
   const real_wp_ v[],
   bool got_h,
   const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

u[0] = u[0] + -mag * freq * freq * cos(freq*x[0]) * v[0]
u[1] = u[1] + 2.0 * v[1]
u[2] = u[2] + 2.0 * v[2]
return 0
]

# Sparse Hessian-vector product
int shessprod_diag(int n,
const real_wp_ x[],
int nnz_v,
const int index_nz_v[],
const real_wp_ v[],
int *nnz_u,
int index_nz_u[],
real_wp_ u[],
bool got_h,
const void *userdata)
struct userdata_type *myuserdata = (struct userdata_type *) userdata
real_wp_ freq = myuserdata->freq
real_wp_ mag = myuserdata->mag

real_wp_ p[] = {0., 0., 0.]
bool used[] = {false, false, false]
for(int i = 0 i < nnz_v i++)
int j = index_nz_v[i]
switch(j)
case 1:
p[0] = p[0] - mag * freq * freq * cos(freq*x[0]) * v[0]
used[0] = true
end
case 2:
p[1] = p[1] + 2.0 * v[1]
used[1] = true
end
case 3:
p[2] = p[2] + 2.0 * v[2]
used[2] = true
end
]
]
*nnz_u = 0
for(int j = 0 j < 3 j++)
if used[j])
u[j] = p[j]
*nnz_u = *nnz_u + 1
index_nz_u[*nnz_u-1] = j+1
]
]
return 0
]
end

@testset "BGO" begin
  @test test_bgo() == 0
end
