# test_arc.jl
# Simple code to test the Julia interface to ARC

using GALAHAD
using Printf

# Objective function
function fun(n, x, f, userdata)
  p = userdata.p
  f[] = (x[1] + x[3] + p)^2) + (x[2] + x[3])^2) + cos(x[1])
  return 0
end

# Gradient of the objective
function grad(n, x, g, userdata)
  p = userdata.p
  g[1] = 2.0 * ( x[1] + x[3] + p ) - sin(x[1])
  g[2] = 2.0 * ( x[2] + x[3] )
  g[3] = 2.0 * ( x[1] + x[3] + p ) + 2.0 * ( x[2] + x[3] )
  return 0
end

# Hessian of the objective
function hess(n, ne, x, hval, userdata)
  hval[1] = 2.0 - cos(x[1])
  hval[2] = 2.0
  hval[3] = 2.0
  hval[4] = 2.0
  hval[5] = 4.0
  return 0
end

# Dense Hessian
function hess_dense(n, ne, x, hval, userdata) 
  hval[1] = 2.0 - cos(x[1])
  hval[2] = 0.0
  hval[3] = 2.0
  hval[4] = 2.0
  hval[5] = 2.0
  hval[6] = 4.0
  return 0
end

# Hessian-vector product
function hessprod(n, x, u, v, got_h, userdata)
  u[1] = u[1] + 2.0 * ( v[1] + v[3] ) - cos(x[1]) * v[1]
  u[2] = u[2] + 2.0 * ( v[2] + v[3] )
  u[3] = u[3] + 2.0 * ( v[1] + v[2] + 2.0 * v[3] )
  return 0
end

# Apply preconditioner
function prec(n, x, u, v, userdata)
  u[1] = 0.5  * v[1]
  u[2] = 0.5  * v[2]
  u[3] = 0.25 * v[3]
  return 0
end

 # Objective function 
function fun_diag(n, x, f, userdata)
  p = userdata.p
  f[] = (x[3] + p)^2 + x[2]^2 + cos(x[1])
  return 0
end

# Gradient of the objective
function grad_diag(n, x, g, userdata)
  p = userdata.p
  g[1] = - sin(x[1])
  g[2] = 2.0 * x[2]
  g[3] = 2.0 * ( x[3] + p )
  return 0
end

# Hessian of the objective
function hess_diag(n, ne, x, hval, userdata)
  hval[1] = - cos(x[1])
  hval[2] = 2.0
  hval[3] = 2.0
  return 0
end  

# Hessian-vector product
function hessprod_diag(n, x, u, v, got_h, userdata)
  u[1] = u[1] + - cos(x[1]) * v[1]
  u[2] = u[2] + 2.0 * v[2]
  u[3] = u[3] + 2.0 * v[3]
  return 0
end

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = arc_control_type{Float64}()
inform = arc_inform_type{Float64}()

# Set user data
mutable struct userdata_type
  p::Float64
end

userdata = userdata_type(4.0)

# Set problem data
n = 3  # dimension
ne = 5  # Hesssian elements
H_row = Cint[1, 2, 3, 3, 3]  # Hessian H
H_col = Cint[1, 2, 1, 2, 3]  # NB lower triangle
H_ptr = Cint[1, 2, 3, 6]  # row pointers

# Set storage
g = zeros(Float64, n) # gradient
st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" tests options for all-in-one storage format\n\n")

for d = 1:5

  # Initialize ARC
  arc_initialize( data, control, status )

  # Set user-defined control options
  control.f_indexing = true  # Fortran sparse matrix indexing

  # control.print_level = 1

  # Start from 1.5
  x = Float64[1.5,1.5,1.5]

  # sparse co-ordinate storage
  if d == 1
    global st = 'C'

    arc_import( control, data, status, n, "coordinate", 
                ne, H_row, H_col, Cint[] )

    arc_solve_with_mat( data, userdata, status,
                        n, x, g, ne, fun, grad, hess, prec )
  end

  # sparse by rows
  if d == 2  
  
    global st = 'R'
  
    arc_import( control, data, status, n, "sparse_by_rows", 
                ne, Cint[], H_col, H_ptr)

    arc_solve_with_mat( data, userdata, status,
                        n, x, g, ne, fun, grad, hess, prec )
  end

  # dense
  if d == 3
    global st = 'D'
    
    arc_import( control, data, status, n, "dense", 
                ne, Cint[], Cint[], Cint[] )

    arc_solve_with_mat( data, userdata, status,
                        n, x, g, ne, fun, grad, hess_dense, prec )
  end

  # diagonal
  if d == 4
    global st = 'I'

    arc_import( control, data, status, n, "diagonal", 
                ne, Cint[], Cint[], Cint[] )

    arc_solve_with_mat( data, userdata, status, n, x, g, 
                        ne, fun_diag, grad_diag, hess_diag, prec) 
  end

  # access by products
  if d == 5
    global st = 'P'
  
    arc_import( control, data, status, n, "absent", 
                ne, Cint[], Cint[], Cint[] )
  
    arc_solve_without_mat( data, userdata, status,
                           n, x, g, fun, grad, hessprod, prec )
  end

  arc_information( data, inform, status)

  if inform.status == 0
    @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st, inform.iter, inform.obj, inform.status)
  else
    @printf("%c: ARC_solve exit status = %1i\n", st, inform.status)
  end

  # @printf("x: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # end
  # @printf("\n")
  # @printf("gradient: ")
  # for i = 1:n
  #   @printf("%f ", g[i])
  # end
  # @printf("\n")

  # Delete internal workspace
  arc_terminate( data, control, inform )
  @printf("\n tests reverse-communication options\n\n")
end

# reverse-communication input/output
eval_status = Ref{Cint}()
f = 0.0
u = zeros(Float64, n)
v = zeros(Float64, n)
index_nz_u = zeros(Cint, n)
index_nz_v = zeros(Cint, n)
H_val = zeros(Float64, ne)
H_dense = zeros(Float64, n*(n+1)/2)
H_diag = zeros(Float64, n)
 
for d = 1:5

  # Initialize ARC
  arc_initialize( data, control, status )

  # Set user-defined control options
  control.f_indexing = true  # Fortran sparse matrix indexing
  #control.print_level = 1

  # Start from 1.5
  x = Float64[1.5,1.5,1.5]

  # sparse co-ordinate storage
  if d == 1
    global st = 'C'

    arc_import( control, data, status, n, "coordinate", 
                ne, H_row, H_col, Cint[] )

    while true  # reverse-communication loop
      arc_solve_reverse_with_mat( data, status, eval_status, 
                                  n, x, f, g, ne, H_val, u, v )
      if status[] == 0  # successful termination
        break
      elseif status[] < 0  # error exit
        error("ARC -- status = $(status[])")
      elseif status[] == 2  # evaluate f
        eval_status = fun( n, x, f, userdata )
      elseif status[] == 3  # evaluate g
        eval_status = grad( n, x, g, userdata )
      elseif status[] == 4  # evaluate H
        eval_status = hess( n, ne, x, H_val, userdata ) 
      elseif status[] == 6  # evaluate the product with P
        eval_status = prec( n, x, u, v, userdata )
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
  end

  # sparse by rows
  if d == 2
    global st = 'R'

    arc_import( control, data, status, n, "sparse_by_rows", ne, 
                Cint[], H_col, H_ptr)

    while true  # reverse-communication loop
      arc_solve_reverse_with_mat( data, status, eval_status, 
                                  n, x, f, g, ne, H_val, u, v )
      if status[] == 0  # successful termination
        break
      elseif status[] < 0  # error exit
        error("ARC -- status = $(status[])")
      elseif status[] == 2  # evaluate f
        eval_status = fun( n, x, f, userdata )
      elseif status[] == 3  # evaluate g
        eval_status = grad( n, x, g, userdata )
      elseif status[] == 4  # evaluate H
        eval_status = hess( n, ne, x, H_val, userdata ) 
      elseif status[] == 6  # evaluate the product with P
        eval_status = prec( n, x, u, v, userdata )
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
  end

  # dense
  if d == 3
    global st = 'D'

    arc_import( control, data, status, n, "dense", 
                ne, Cint[], Cint[], Cint[] )

    while true  # reverse-communication loop
      arc_solve_reverse_with_mat( data, status, eval_status, 
                                  n, x, f, g, n*(n+1)/2, H_dense, u, v )
      if status[] == 0  # successful termination
        break
      elseif status[] < 0  # error exit
        error("ARC -- status = $(status[])")
      elseif status[] == 2  # evaluate f
        eval_status = fun( n, x, f, userdata )
      elseif status[] == 3  # evaluate g
        eval_status = grad( n, x, g, userdata )
      elseif status[] == 4  # evaluate H
        eval_status = hess_dense( n, n*(n+1)/2, x, H_dense, userdata ) 
      elseif status[] == 6  # evaluate the product with P
        eval_status = prec( n, x, u, v, userdata )
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
    end
  end

  # diagonal
  if d == 4
    global st = 'I'

    arc_import( control, data, status, n, "diagonal", 
                ne, Cint[], Cint[], Cint[] )

    while true  # reverse-communication loop
      arc_solve_reverse_with_mat( data, status, eval_status, 
                                  n, x, f, g, n, H_diag, u, v )
      if status[] == 0  # successful termination
        break
      elseif status[] < 0  # error exit
        error("ARC -- status = $(status[])")
      elseif status[] == 2  # evaluate f
        eval_status = fun_diag( n, x, f, userdata )
      elseif status[] == 3  # evaluate g
        eval_status = grad_diag( n, x, g, userdata )
      elseif status[] == 4  # evaluate H
        eval_status = hess_diag( n, n, x, H_diag, userdata ) 
      elseif status[] == 6  # evaluate the product with P
        eval_status = prec( n, x, u, v, userdata )
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
    end
  end

  # access by products
  if d == 5
    global st = 'P'

    arc_import( control, data, status, n, "absent", 
                ne, Cint[], Cint[], Cint[] )

    while true  # reverse-communication loop
      arc_solve_reverse_without_mat( data, status, eval_status,
                                     n, x, f, g, u, v )
      if status[] == 0  # successful termination
        break
      elseif status[] < 0  # error exit
        error("ARC -- status = $(status[])")
      elseif status[] == 2  # evaluate f
        eval_status = fun( n, x, f, userdata )
      elseif status[] == 3  # evaluate g
        eval_status = grad( n, x, g, userdata )
      elseif status[] == 5  # evaluate H
        eval_status = hessprod( n, x, u, v, false, userdata )
      elseif status[] == 6  # evaluate the product with P
        eval_status = prec( n, x, u, v, userdata )
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
    end
  end

  arc_information( data, inform, status )

  if inform.status == 0
    @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st, inform.iter, inform.obj, inform.status)
  else
    @printf("%c: ARC_solve exit status = %1i\n", st, inform.status)
  end

  # @printf("x: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # end
  # @printf("\n")
  # @printf("gradient: ")
  # for i = 1:n
  #   @printf("%f ", g[i])
  # end
  # @printf("\n")

  # Delete internal workspace
  arc_terminate( data, control, inform )
end
