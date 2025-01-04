# test_sbls.jl
# Simple code to test the Julia interface to SBLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_sbls(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sbls_control_type{T,INT}}()
  inform = Ref{sbls_inform_type{T,INT}}()

  # Set problem data
  n = INT(3)  # dimension of H
  m = INT(2)  # dimension of C
  H_ne = INT(4)  # number of elements of H
  A_ne = INT(3)  # number of elements of A
  C_ne = INT(3)  # number of elements of C
  H_dense_ne = INT(6)  # number of elements of H
  A_dense_ne = INT(6)  # number of elements of A
  C_dense_ne = INT(3)  # number of elements of C
  H_row = INT[1, 2, 3, 3]  # row indices, NB lower triangle
  H_col = INT[1, 2, 3, 1]
  H_ptr = INT[1, 2, 3, 5]
  A_row = INT[1, 1, 2]
  A_col = INT[1, 2, 3]
  A_ptr = INT[1, 3, 4]
  C_row = INT[1, 2, 2]  # row indices, NB lower triangle
  C_col = INT[1, 1, 2]
  C_ptr = INT[1, 2, 4]
  H_val = T[1.0, 2.0, 3.0, 1.0]
  A_val = T[2.0, 1.0, 1.0]
  C_val = T[4.0, 1.0, 2.0]
  H_dense = T[1.0, 0.0, 2.0, 1.0, 0.0, 3.0]
  A_dense = T[2.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  C_dense = T[4.0, 1.0, 2.0]
  H_diag = T[1.0, 1.0, 2.0]
  C_diag = T[4.0, 2.0]
  H_scid = T[2.0]
  C_scid = T[2.0]

  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  for d in 1:7

    # Initialize SBLS
    sbls_initialize(T, INT, data, control, status)
    @reset control[].preconditioner = INT(2)
    @reset control[].factorization = INT(2)
    @reset control[].get_norm_residual = true
    @reset control[].symmetric_linear_solver = galahad_linear_solver("sytr")
    @reset control[].definite_linear_solver = galahad_linear_solver("sytr")

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      sbls_import(T, INT, control, data, status, n, m,
                  "coordinate", H_ne, H_row, H_col, C_NULL,
                  "coordinate", A_ne, A_row, A_col, C_NULL,
                  "coordinate", C_ne, C_row, C_col, C_NULL)

      sbls_factorize_matrix(T, INT, data, status, n,
                            H_ne, H_val,
                            A_ne, A_val,
                            C_ne, C_val, C_NULL)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      sbls_import(T, INT, control, data, status, n, m,
                  "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr,
                  "sparse_by_rows", C_ne, C_NULL, C_col, C_ptr)

      sbls_factorize_matrix(T, INT, data, status, n,
                            H_ne, H_val,
                            A_ne, A_val,
                            C_ne, C_val, C_NULL)
    end

    # dense
    if d == 3
      st = 'D'
      sbls_import(T, INT, control, data, status, n, m,
                  "dense", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "dense", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(T, INT, data, status, n,
                            H_dense_ne, H_dense,
                            A_dense_ne, A_dense,
                            C_dense_ne, C_dense,
                            C_NULL)
    end

    # diagonal
    if d == 4
      st = 'L'
      sbls_import(T, INT, control, data, status, n, m,
                  "diagonal", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "diagonal", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(T, INT, data, status, n,
                            n, H_diag,
                            A_dense_ne, A_dense,
                            m, C_diag,
                            C_NULL)
    end

    # scaled identity
    if d == 5
      st = 'S'
      sbls_import(T, INT, control, data, status, n, m,
                  "scaled_identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "scaled_identity", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(T, INT, data, status, n,
                            1, H_scid,
                            A_dense_ne, A_dense,
                            1, C_scid,
                            C_NULL)
    end

    # identity
    if d == 6
      st = 'I'
      sbls_import(T, INT, control, data, status, n, m,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "identity", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(T, INT, data, status, n,
                            0, H_val,
                            A_dense_ne, A_dense,
                            0, C_val, C_NULL)
    end

    # zero
    if d == 7
      st = 'Z'
      sbls_import(T, INT, control, data, status, n, m,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "zero", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(T, INT, data, status, n,
                            0, H_val,
                            A_dense_ne, A_dense,
                            0, C_NULL, C_NULL)
    end

    # check that the factorization succeeded
    if status[] != 0
      sbls_information(T, INT, data, inform, status)
      @printf("%c: SBLS_solve factorization exit status = %1i\n", st, inform[].status)
      return 1
    end

    # Set right-hand side (a, b)
    sol = T[3.0, 2.0, 4.0, 2.0, 0.0]  # values

    sbls_solve_system(T, INT, data, status, n, m, sol)

    sbls_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%c: residual = %9.1e status = %1i\n",
              st, inform[].norm_residual, inform[].status)
    else
      @printf("%c: SBLS_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("sol: ")
    # for i = 1:n+m
    #  @printf("%f ", x[i])
    # end

    # Delete internal workspace
    sbls_terminate(T, INT, data, control, inform)
  end

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "SBLS -- $T -- $INT" begin
      @test test_sbls(T, INT) == 0
    end
  end
end
