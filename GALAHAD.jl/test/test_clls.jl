# test_clls.jl
# Simple code to test the Julia interface to CLLS

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = clls_control_type{Float64}()
inform = clls_inform_type{Float64}()

# Set problem data
n = 3  # dimension
o = 4  # number of observations
m = 2  # number of general constraints
Ao_ne = 7  # design matrix elements
Ao_row = Cint[1, 1, 2, 2, 3, 3, 4]  # row indices
Ao_col = Cint[1, 2, 2, 1, 3, 2]  # column indices
Ao_ptr_ne = o + 1 # row pointer length
Ao_ptr = Cint[1, 3, 5, 7, 8]  # row pointers
Ao_val = Float64[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # values
b = Float64[2.0, 2.0, 3.0, 1.0]  # observations
A_ne = 4  # Jacobian elements
A_row = Cint[1, 1, 2, 2]  # row indices
A_col = Cint[1, 2, 2, 3]  # column indices
Ao_ptr_ne = m + 1 # row pointer length
A_ptr = Cint[1, 3, 5]  # row pointers
A_val = Float64[2.0, 1.0, 1.0, 1.0]  # values
c_l = Float64[1.0, 2.0]  # constraint lower bound
c_u = Float64[2.0, 2.0]  # constraint upper bound
x_l = Float64[-1.0, -Inf, -Inf]  # variable lower bound
x_u = Float64[1.0, Inf, 2.0]  # variable upper bound
sigma = 1.0  # regularization weight
w = Float64[1.0, 1.0, 1.0, 2.0]  # values

# Set output storage
c = zeros(Float64, m)  # constraint values
x_stat = zeros(Cint, n)  # variable status
c_stat = zeros(Cint, m)  # constraint status
st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of clls storage formats\n\n")

for d = 1:3

  # Initialize CLLS
  clls_initialize( data, control, status )

  # Set user-defined control options
  control.f_indexing = true  # Fortran sparse matrix indexing

  # Start from 0
  x = Float64[0.0,0.0,0.0]
  y = Float64[0.0,0.0]
  z = Float64[0.0,0.0,0.0]

  # sparse co-ordinate storage
  if d == 1
    global st = 'C'

    clls_import( control, data, status, n, o, m,
                "coordinate", Ao_ne, Ao_row, Ao_col, 0, Cint[],
                "coordinate", A_ne, A_row, A_col, 0, Cint[] )

    clls_solve_clls( data, status, n, o, m, Ao_ne, Ao_val, b, sigma,
                     A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                     x_stat, c_stat, w )
  end

  # sparse by rows
  if d == 2
    global st = 'R'

    clls_import( control, data, status, n, o, m,
                "sparse_by_rows", Ao_ne, Cint[], Ao_col, Ao_ptr_ne, Ao_ptr,
                "sparse_by_rows", A_ne, Cint[], A_col, A_ptr_ne, A_ptr )

    clls_solve_clls( data, status, n, o, m, Ao_ne, Ao_val, b, sigma,
                     A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                     x_stat, c_stat, w )
  end

  # dense
  if d == 3
    global st = 'D'

    Ao_dense_ne = 12  # number of elements of Ao
    A_dense_ne = 6  # number of elements of A
    Ao_dense = Float64[1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                       1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    A_dense = Float64[2.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    clls_import( control, data, status, n, o, m,
                "dense", Ao_ne, Cint[], Cint[], 0, Cint[],
                "dense", A_ne, Cint[], Cint[], 0, Cint[] )

    clls_solve_clls( data, status, n, o, m, Ao_dense_ne, Ao_dense, b, sigma,
                     A_dense_ne, A_dense, c_l, c_u, x_l, x_u,
                     x, c, y, z, x_stat, c_stat, w )
  end

  clls_information( data, inform, status )

  if inform.status == 0
    @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st, inform.iter, inform.obj, inform.status)
  else
    @printf("%c: CLLS_solve exit status = %1i\n", st, inform.status)
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
  clls_terminate( data, control, inform )
end
