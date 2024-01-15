# test_wcp.jl
# Simple code to test the Julia interface to WCP

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = wcp_control_type{Float64}()
inform = wcp_inform_type{Float64}()

# Set problem data
n = 3  # dimension
m = 2  # number of general constraints
g = Float64[0.0, 2.0, 0.0]  # linear term in the objective
A_ne = 4  # Jacobian elements
A_row = Cint[1, 1, 2, 2]  # row indices
A_col = Cint[1, 2, 2, 3]  # column indices
A_ptr = Cint[1, 3, 5]  # row pointers
A_val = Float64[2.0, 1.0, 1.0, 1.0]  # values
c_l = Float64[1.0, 2.0]  # constraint lower bound
c_u = Float64[2.0, 2.0]  # constraint upper bound
x_l = Float64[-1.0, -Inf, -Inf]  # variable lower bound
x_u = Float64[1.0, Inf, 2.0]  # variable upper bound

# Set output storage
c = zeros(Float64, m) # constraint values
x_stat = zeros(Cint, n) # variable status
c_stat = zeros(Cint, m) # constraint status
st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of wcp storage formats\n\n")

for d = 1:3
  # Initialize WCP
  wcp_initialize( data, control, status )

  # Set user-defined control options
  control.f_indexing = true # Fortran sparse matrix indexing

  # Start from 0
  x = Float64[0.0,0.0,0.0]
  y_l = Float64[0.0,0.0]
  y_u = Float64[0.0,0.0]
  z_l = Float64[0.0,0.0,0.0]
  z_u = Float64[0.0,0.0,0.0]

  if d == 1
    # sparse co-ordinate storage
    global st = 'C'
    wcp_import( control, data, status, n, m,
                "coordinate", A_ne, A_row, A_col, Cint[] )

    wcp_find_wcp( data, status, n, m, g, A_ne, A_val,
                  c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u,
                  x_stat, c_stat )
  end

  if d == 2
    # sparse by rows
    global st = 'R'
    wcp_import( control, data, status, n, m,
                "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )

    wcp_find_wcp( data, status, n, m, g, A_ne, A_val,
                  c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u,
                  x_stat, c_stat )
  end

  if d == 3
    # dense
    global st = 'D'
    A_dense_ne = 6 # number of elements of A
    A_dense = Float64[2.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    wcp_import( control, data, status, n, m,
                "dense", A_dense_ne, Cint[], Cint[], Cint[] )

    wcp_find_wcp( data, status, n, m, g, A_dense_ne, A_dense,
                  c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u,
                  x_stat, c_stat )
  end

  wcp_information( data, inform, status )

  if inform.status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform.iter, inform.obj, inform.status)
  else
      @printf("%c: WCP_solve exit status = %1i\n", st, inform.status)
  end
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
wcp_terminate( data, control, inform )
