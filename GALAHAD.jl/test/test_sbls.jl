# test_sbls.jl
# Simple code to test the Julia interface to SBLS

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = sbls_control_type{Float64}()
inform = sbls_inform_type{Float64}()

# Set problem data
n = 3  # dimension of H
m = 2  # dimension of C
H_ne = 4  # number of elements of H
A_ne = 3  # number of elements of A
C_ne = 3  # number of elements of C
H_dense_ne = 6  # number of elements of H
A_dense_ne = 6  # number of elements of A
C_dense_ne = 3  # number of elements of C
H_row = Cint[1, 2, 3, 3]  # row indices, NB lower triangle
H_col = Cint[1, 2, 3, 1]
H_ptr = Cint[1, 2, 3, 5]
A_row = Cint[1, 1, 2]
A_col = Cint[1, 2, 3]
A_ptr = Cint[1, 3, 4]
C_row = [1, 2, 2]  # row indices, NB lower triangle
C_col = [1, 1, 2]
C_ptr = [1, 2, 4]
H_val = Float64[1.0, 2.0, 3.0, 1.0]
A_val = Float64[2.0, 1.0, 1.0]
C_val = Float64[4.0, 1.0, 2.0]
H_dense = Float64[1.0, 0.0, 2.0, 1.0, 0.0, 3.0]
A_dense = Float64[2.0, 1.0, 0.0, 0.0, 0.0, 1.0]
C_dense = Float64[4.0, 1.0, 2.0]
H_diag = Float64[1.0, 1.0, 2.0]
C_diag = Float64[4.0, 2.0]
H_scid = Float64[2.0]
C_scid = Float64[2.0]

st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of storage formats\n\n")

for d = 1:7
  # Initialize SBLS
  sbls_initialize( data, control, status )
  control.preconditioner = 2
  control.factorization = 2
  control.get_norm_residual = true

  # Set user-defined control options
  control.f_indexing = true  # fortran sparse matrix indexing

  # sparse co-ordinate storage
  if d == 1
    global st = 'C'

    sbls_import( control, data, status, n, m,
                 "coordinate", H_ne, H_row, H_col, Cint[],
                 "coordinate", A_ne, A_row, A_col, Cint[],
                 "coordinate", C_ne, C_row, C_col, Cint[] )

    sbls_factorize_matrix( data, status, n,
                           H_ne, H_val,
                           A_ne, A_val,
                           C_ne, C_val, Cint[] )
  end

  # sparse by rows
  if d == 2
    global st = 'R'

    sbls_import( control, data, status, n, m,
                 "sparse_by_rows", H_ne, Cint[], H_col, H_ptr,
                 "sparse_by_rows", A_ne, Cint[], A_col, A_ptr,
                 "sparse_by_rows", C_ne, Cint[], C_col, C_ptr )

    sbls_factorize_matrix( data, status, n,
                           H_ne, H_val,
                           A_ne, A_val,
                           C_ne, C_val, Cint[] )
  end

  # dense
  if d == 3
    global st = 'D'

    sbls_import( control, data, status, n, m,
                 "dense", H_ne, Cint[], Cint[], Cint[],
                 "dense", A_ne, Cint[], Cint[], Cint[],
                 "dense", C_ne, Cint[], Cint[], Cint[] )

    sbls_factorize_matrix( data, status, n,
                           H_dense_ne, H_dense,
                           A_dense_ne, A_dense,
                           C_dense_ne, C_dense,
                           Cint[] )
  end

  # diagonal
  if d == 4
    global st = 'L'

    sbls_import( control, data, status, n, m,
                 "diagonal", H_ne, Cint[], Cint[], Cint[],
                 "dense", A_ne, Cint[], Cint[], Cint[],
                 "diagonal", C_ne, Cint[], Cint[], Cint[] )

    sbls_factorize_matrix( data, status, n,
                           n, H_diag,
                           A_dense_ne, A_dense,
                           m, C_diag,
                           Cint[] )
  end

  # scaled identity
  if d == 5
    global st = 'S'

    sbls_import( control, data, status, n, m,
                 "scaled_identity", H_ne, Cint[], Cint[], Cint[],
                 "dense", A_ne, Cint[], Cint[], Cint[],
                 "scaled_identity", C_ne, Cint[], Cint[], Cint[] )

    sbls_factorize_matrix( data, status, n,
                           1, H_scid,
                           A_dense_ne, A_dense,
                           1, C_scid,
                           Cint[] )
  end

  # identity
  if d == 6
    global st = 'I'

    sbls_import( control, data, status, n, m,
                 "identity", H_ne, Cint[], Cint[], Cint[],
                 "dense", A_ne, Cint[], Cint[], Cint[],
                 "identity", C_ne, Cint[], Cint[], Cint[] )

    sbls_factorize_matrix( data, status, n,
                           0, H_val,
                           A_dense_ne, A_dense,
                           0, C_val, Cint[] )
  end

  # zero
  if d == 7
    global st = 'Z'

    sbls_import( control, data, status, n, m,
                 "identity", H_ne, Cint[], Cint[], Cint[],
                 "dense", A_ne, Cint[], Cint[], Cint[],
                 "zero", C_ne, Cint[], Cint[], Cint[] )

    sbls_factorize_matrix( data, status, n,
                           0, H_val,
                           A_dense_ne, A_dense,
                           0, Cint[], Cint[] )
  end

  # Set right-hand side ( a, b )
  sol = Float64[3.0, 2.0, 4.0, 2.0, 0.0]  # values

  sbls_solve_system( data, status, n, m, sol )

  sbls_information( data, inform, status )

  if inform.status == 0
    @printf("%c: residual = %9.1e status = %1i\n", st, inform.norm_residual, inform.status)
  else
    @printf("%c: SBLS_solve exit status = %1i\n", st, inform.status)
  end

  # @printf("sol: ")
  # for i = 1:n+m
  #   @printf("%f ", x[i])
  # end

  # Delete internal workspace
  sbls_terminate( data, control, inform )
end
