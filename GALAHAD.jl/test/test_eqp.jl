# test_eqp.jl
# Simple code to test the Julia interface to EQP

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = eqp_control_type{Float64}()
inform = eqp_inform_type{Float64}()

# Set problem data
n = 3  # dimension
m = 2  # number of general constraints
H_ne = 3  # Hesssian elements
H_row = Cint[1, 2, 3]  # row indices, NB lower triangle
H_col = Cint[1, 2, 3]  # column indices, NB lower triangle
H_ptr = Cint[1, 2, 3, 4]  # row pointers
H_val = Float64[1.0, 1.0, 1.0]  # values
g = Float64[0.0, 2.0, 0.0]  # linear term in the objective
f = 1.0  # constant term in the objective
A_ne = 4  # Jacobian elements
A_row = Cint[1, 1, 2, 2]  # row indices
A_col = Cint[1, 2, 2, 3]  # column indices
A_ptr = Cint[1, 3, 5]  # row pointers
A_val = Float64[2.0, 1.0, 1.0, 1.0]  # values
c = Float64[3.0, 0.0]  # rhs of the constraints

# Set output storage
x_stat = zeros(Cint, n)  # variable status
c_stat = zeros(Cint, m)  # constraint status
st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of qp storage formats\n\n")

for d = 1:6

  # Initialize EQP
  eqp_initialize( data, control, status )

  # Set user-defined control options
  control.f_indexing = true  # Fortran sparse matrix indexing
  control.fdc_control.use_sls = true
  solver = "sytr"
  control.fdc_control.symmetric_linear_solver = (115, 121, 116, 114, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0)
  control.sbls_control.symmetric_linear_solver = (115, 121, 116, 114, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0)
  control.sbls_control.definite_linear_solver = (115, 121, 116, 114, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0)

  # Start from 0
  x = Float64[0.0,0.0,0.0]
  y = Float64[0.0,0.0]
  z = Float64[0.0,0.0,0.0]

  # sparse co-ordinate storage
  if d == 1
    global st = 'C'

    eqp_import( control, data, status, n, m,
                "coordinate", H_ne, H_row, H_col, Cint[],
                "coordinate", A_ne, A_row, A_col, Cint[] )

    eqp_solve_qp( data, status, n, m, H_ne, H_val, g, f,
                  A_ne, A_val, c, x, y )
  end

  # sparse by rows
  if d == 2
    global st = 'R'

    eqp_import( control, data, status, n, m,
                "sparse_by_rows", H_ne, Cint[], H_col, H_ptr,
                "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )

    eqp_solve_qp( data, status, n, m, H_ne, H_val, g, f,
                  A_ne, A_val, c, x, y )
  end

  # dense
  if d == 3
    global st = 'D'

    H_dense_ne = 6  # number of elements of H
    A_dense_ne = 6  # number of elements of A
    H_dense = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    A_dense = Float64[2.0, 1.0, 0.0, 0.0, 1.0, 1.0]

    eqp_import( control, data, status, n, m,
                "dense", H_ne, Cint[], Cint[], Cint[],
                "dense", A_ne, Cint[], Cint[], Cint[] )

    eqp_solve_qp( data, status, n, m, H_dense_ne, H_dense, g, f,
                  A_dense_ne, A_dense, c, x, y )
  end

  # diagonal
  if d == 4
    global st = 'L'

    eqp_import( control, data, status, n, m,
                "diagonal", H_ne, Cint[], Cint[], Cint[],
                "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )

    eqp_solve_qp( data, status, n, m, H_ne, H_val, g, f,
                  A_ne, A_val, c, x, y )
  end

  # scaled identity
  if d == 5
    global st = 'S'

    eqp_import( control, data, status, n, m,
                "scaled_identity", H_ne, Cint[], Cint[], Cint[],
                "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )

    eqp_solve_qp( data, status, n, m, H_ne, H_val, g, f,
                  A_ne, A_val, c, x, y )
  end

  # identity
  if d == 6
    global st = 'I'

    eqp_import( control, data, status, n, m,
                "identity", H_ne, Cint[], Cint[], Cint[],
                "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )

    eqp_solve_qp( data, status, n, m, H_ne, H_val, g, f,
                  A_ne, A_val, c, x, y )
  end

  # zero
  if d == 7
    global st = 'Z'

    eqp_import( control, data, status, n, m,
                "zero", H_ne, Cint[], Cint[], Cint[],
                "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )

    eqp_solve_qp( data, status, n, m, H_ne, H_val, g, f,
                  A_ne, A_val, c, x, y )
  end

  eqp_information( data, inform, status )

  if inform.status == 0
    @printf("%c:%6i cg iterations. Optimal objective value = %5.2f status = %1i\n", st, inform.cg_iter, inform.obj, inform.status)
  else
    @printf("%c: EQP_solve exit status = %1i\n", st, inform.status)
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
  eqp_terminate( data, control, inform )
end

# test shifted least-distance interface
for d = 1:1

  # Initialize EQP
  eqp_initialize( data, control, status )
  control.fdc_control.use_sls = true
  strcpy(control.fdc_control.symmetric_linear_solver, "sytr ")
  strcpy(control.sbls_control.symmetric_linear_solver, "sytr ")
  strcpy(control.sbls_control.definite_linear_solver, "sytr ")

  # Set user-defined control options
  control.f_indexing = true  # Fortran sparse matrix indexing

  # Start from 0
  x = Float64[0.0,0.0,0.0]
  y = Float64[0.0,0.0]
  z = Float64[0.0,0.0,0.0]

  # Set shifted least-distance data

  w = [1.0,1.0,1.0]
  x_0 = [0.0,0.0,0.0]

  # sparse co-ordinate storage
  if d == 1
    global st = 'W'

    eqp_import( control, data, status, n, m,
                "shifted_least_distance", H_ne, Cint[], Cint[], Cint[],
                "coordinate", A_ne, A_row, A_col, Cint[] )

    eqp_solve_sldqp( data, status, n, m, w, x_0, g, f,
                     A_ne, A_val, c, x, y )
  end

  eqp_information( data, inform, status )

  if inform.status == 0
    @printf("%c:%6i cg iterations. Optimal objective value = %5.2f status = %1i\n", st, inform.cg_iter, inform.obj, inform.status)
  else
    @printf("%c: EQP_solve exit status = %1i\n", st, inform.status)
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
  eqp_terminate( data, control, inform )
end
