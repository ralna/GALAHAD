# test_ssls.jl
# Simple code to test the Julia interface to SSLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_ssls(::Type{T}, ::Type{INT}; sls::String="sytr") where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ssls_control_type{T,INT}}()
  inform = Ref{ssls_inform_type{T,INT}}()

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

    # Initialize SSLS
    ssls_initialize(T, INT, data, control, status)

    # Linear solvers
    @reset control[].symmetric_linear_solver = galahad_linear_solver(sls)

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      ssls_import(T, INT, control, data, status, n, m,
                  "coordinate", H_ne, H_row, H_col, C_NULL,
                  "coordinate", A_ne, A_row, A_col, C_NULL,
                  "coordinate", C_ne, C_row, C_col, C_NULL)

      ssls_factorize_matrix(T, INT, data, status,
                            H_ne, H_val,
                            A_ne, A_val,
                            C_ne, C_val)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      ssls_import(T, INT, control, data, status, n, m,
                  "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr,
                  "sparse_by_rows", C_ne, C_NULL, C_col, C_ptr)

      ssls_factorize_matrix(T, INT, data, status,
                            H_ne, H_val,
                            A_ne, A_val,
                            C_ne, C_val)
    end

    # dense
    if d == 3
      st = 'D'
      ssls_import(T, INT, control, data, status, n, m,
                  "dense", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "dense", C_ne, C_NULL, C_NULL, C_NULL)

      ssls_factorize_matrix(T, INT, data, status,
                            H_dense_ne, H_dense,
                            A_dense_ne, A_dense,
                            C_dense_ne, C_dense)
    end

    # diagonal
    if d == 4
      st = 'L'
      ssls_import(T, INT, control, data, status, n, m,
                  "diagonal", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "diagonal", C_ne, C_NULL, C_NULL, C_NULL)

      ssls_factorize_matrix(T, INT, data, status,
                            n, H_diag,
                            A_dense_ne, A_dense,
                            m, C_diag )
    end

    # scaled identity
    if d == 5
      st = 'S'
      ssls_import(T, INT, control, data, status, n, m,
                  "scaled_identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "scaled_identity", C_ne, C_NULL, C_NULL, C_NULL)

      ssls_factorize_matrix(T, INT, data, status,
                            1, H_scid,
                            A_dense_ne, A_dense,
                            1, C_scid)
    end

    # identity
    if d == 6
      st = 'I'
      ssls_import(T, INT, control, data, status, n, m,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "identity", C_ne, C_NULL, C_NULL, C_NULL)

      ssls_factorize_matrix(T, INT, data, status,
                            0, H_val,
                            A_dense_ne, A_dense,
                            0, C_val)
    end

    # zero
    if d == 7
      st = 'Z'
      ssls_import(T, INT, control, data, status, n, m,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "zero", C_ne, C_NULL, C_NULL, C_NULL)

      ssls_factorize_matrix(T, INT, data, status,
                            0, H_val,
                            A_dense_ne, A_dense,
                            0, C_NULL)
    end

    # check that the factorization succeeded
    if status[] != 0
      ssls_information(T, INT, data, inform, status)
      @printf("%c: SSLS_solve factorization exit status = %1i\n", st, inform[].status)
      return 1
    end

    # Set right-hand side (a, b)
    if d == 4
      sol = T[3.0, 2.0, 3.0, -1.0, -1.0]
    elseif d == 5
      sol = T[4.0, 3.0, 3.0, 1.0, -1.0]
    elseif d == 6
      sol = T[3.0, 2.0, 2.0, 2.0, 0.0]
    elseif d == 7
      sol = T[3.0, 2.0, 2.0, 3.0, 1.0]
    else
      sol = T[4.0, 3.0, 5.0, -2.0, -2.0]
    end

    ssls_solve_system(T, INT, data, status, n, m, sol)

    if status[] != 0
      ssls_information(T, INT, data, inform, status)
      @printf("%c: SSLS_solve exit status = %i\n", st, inform[].status)
      continue
    end

    ssls_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%c: status = %1i\n", st, inform[].status)
    else
      @printf("%c: SSLS_solve exit status = %1i\n", st, inform[].status)
    end

    # Delete internal workspace
    ssls_terminate(T, INT, data, control, inform)
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
    @testset "SSLS -- $T -- $INT" begin
      @test test_ssls(T, INT) == 0
    end
  end
end
