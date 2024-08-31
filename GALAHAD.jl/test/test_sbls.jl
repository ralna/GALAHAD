# test_sbls.jl
# Simple code to test the Julia interface to SBLS

using GALAHAD
using Test
using Printf
using Accessors

function test_sbls(::Type{T}) where T
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sbls_control_type{Float64}}()
  inform = Ref{sbls_inform_type{Float64}}()

  # Set problem data
  n = 3 # dimension of H
  m = 2 # dimension of C
  H_ne = 4 # number of elements of H
  A_ne = 3 # number of elements of A
  C_ne = 3 # number of elements of C
  H_dense_ne = 6 # number of elements of H
  A_dense_ne = 6 # number of elements of A
  C_dense_ne = 3 # number of elements of C
  H_row = Cint[1, 2, 3, 3]  # row indices, NB lower triangle
  H_col = Cint[1, 2, 3, 1]
  H_ptr = Cint[1, 2, 3, 5]
  A_row = Cint[1, 1, 2]
  A_col = Cint[1, 2, 3]
  A_ptr = Cint[1, 3, 4]
  C_row = Cint[1, 2, 2]  # row indices, NB lower triangle
  C_col = Cint[1, 1, 2]
  C_ptr = Cint[1, 2, 4]
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

  for d in 1:7

    # Initialize SBLS
    sbls_initialize(Float64, data, control, status)
    @reset control[].preconditioner = Cint(2)
    @reset control[].factorization = Cint(2)
    @reset control[].get_norm_residual = true

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      sbls_import(Float64, control, data, status, n, m,
                  "coordinate", H_ne, H_row, H_col, C_NULL,
                  "coordinate", A_ne, A_row, A_col, C_NULL,
                  "coordinate", C_ne, C_row, C_col, C_NULL)

      sbls_factorize_matrix(Float64, data, status, n,
                            H_ne, H_val,
                            A_ne, A_val,
                            C_ne, C_val, C_NULL)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      sbls_import(Float64, control, data, status, n, m,
                  "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr,
                  "sparse_by_rows", C_ne, C_NULL, C_col, C_ptr)

      sbls_factorize_matrix(Float64, data, status, n,
                            H_ne, H_val,
                            A_ne, A_val,
                            C_ne, C_val, C_NULL)
    end

    # dense
    if d == 3
      st = 'D'
      sbls_import(Float64, control, data, status, n, m,
                  "dense", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "dense", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(Float64, data, status, n,
                            H_dense_ne, H_dense,
                            A_dense_ne, A_dense,
                            C_dense_ne, C_dense,
                            C_NULL)
    end

    # diagonal
    if d == 4
      st = 'L'
      sbls_import(Float64, control, data, status, n, m,
                  "diagonal", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "diagonal", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(Float64, data, status, n,
                            n, H_diag,
                            A_dense_ne, A_dense,
                            m, C_diag,
                            C_NULL)
    end

    # scaled identity
    if d == 5
      st = 'S'
      sbls_import(Float64, control, data, status, n, m,
                  "scaled_identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "scaled_identity", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(Float64, data, status, n,
                            1, H_scid,
                            A_dense_ne, A_dense,
                            1, C_scid,
                            C_NULL)
    end

    # identity
    if d == 6
      st = 'I'
      sbls_import(Float64, control, data, status, n, m,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "identity", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(Float64, data, status, n,
                            0, H_val,
                            A_dense_ne, A_dense,
                            0, C_val, C_NULL)
    end

    # zero
    if d == 7
      st = 'Z'
      sbls_import(Float64, control, data, status, n, m,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL,
                  "zero", C_ne, C_NULL, C_NULL, C_NULL)

      sbls_factorize_matrix(Float64, data, status, n,
                            0, H_val,
                            A_dense_ne, A_dense,
                            0, C_NULL, C_NULL)
    end

    # Set right-hand side (a, b)
    sol = Float64[3.0, 2.0, 4.0, 2.0, 0.0]  # values

    sbls_solve_system(Float64, data, status, n, m, sol)

    sbls_information(Float64, data, inform, status)

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
    sbls_terminate(Float64, data, control, inform)
  end

  return 0
end

@testset "SBLS" begin
  @test test_sbls(Float32) == 0
  @test test_sbls(Float64) == 0
end
