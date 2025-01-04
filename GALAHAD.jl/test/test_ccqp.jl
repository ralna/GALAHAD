# test_ccqp.jl
# Simple code to test the Julia interface to CCQP

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_ccqp(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ccqp_control_type{T,INT}}()
  inform = Ref{ccqp_inform_type{T,INT}}()

  # Set problem data
  n = INT(3)  # dimension
  m = INT(2)  # number of general constraints
  H_ne = INT(3)  # Hesssian elements
  H_row = INT[1, 2, 3]  # row indices, NB lower triangle
  H_col = INT[1, 2, 3]  # column indices, NB lower triangle
  H_ptr = INT[1, 2, 3, 4]  # row pointers
  H_val = T[1.0, 1.0, 1.0]  # values
  g = T[0.0, 2.0, 0.0]  # linear term in the objective
  f = one(T)  # constant term in the objective
  A_ne = 4 # Jacobian elements
  A_row = INT[1, 1, 2, 2]  # row indices
  A_col = INT[1, 2, 2, 3]  # column indices
  A_ptr = INT[1, 3, 5]  # row pointers
  A_val = T[2.0, 1.0, 1.0, 1.0]  # values
  c_l = T[1.0, 2.0]  # constraint lower bound
  c_u = T[2.0, 2.0]  # constraint upper bound
  x_l = T[-1.0, -Inf, -Inf]  # variable lower bound
  x_u = T[1.0, Inf, 2.0]  # variable upper bound

  # Set output storage
  c = zeros(T, m) # constraint values
  x_stat = zeros(INT, n) # variable status
  c_stat = zeros(INT, m) # constraint status
  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of qp storage formats\n\n")

  for d in 1:7
    # Initialize CCQP
    ccqp_initialize(T, INT, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # Start from 0
    x = T[0.0, 0.0, 0.0]
    y = T[0.0, 0.0]
    z = T[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      ccqp_import(T, INT, control, data, status, n, m,
                  "coordinate", H_ne, H_row, H_col, C_NULL,
                  "coordinate", A_ne, A_row, A_col, C_NULL)

      ccqp_solve_qp(T, INT, data, status, n, m, H_ne, H_val, g, f,
                    A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      ccqp_import(T, INT, control, data, status, n, m,
                  "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)

      ccqp_solve_qp(T, INT, data, status, n, m, H_ne, H_val, g, f,
                    A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat)
    end

    # dense
    if d == 3
      st = 'D'
      H_dense_ne = 6 # number of elements of H
      A_dense_ne = 6 # number of elements of A
      H_dense = T[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
      A_dense = T[2.0, 1.0, 0.0, 0.0, 1.0, 1.0]

      ccqp_import(T, INT, control, data, status, n, m,
                  "dense", H_ne, C_NULL, C_NULL, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, C_NULL)

      ccqp_solve_qp(T, INT, data, status, n, m, H_dense_ne, H_dense, g, f,
                    A_dense_ne, A_dense, c_l, c_u, x_l, x_u,
                    x, c, y, z, x_stat, c_stat)
    end

    # diagonal
    if d == 4
      st = 'L'
      ccqp_import(T, INT, control, data, status, n, m,
                  "diagonal", H_ne, C_NULL, C_NULL, C_NULL,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)

      ccqp_solve_qp(T, INT, data, status, n, m, H_ne, H_val, g, f,
                    A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat)
    end

    # scaled identity
    if d == 5
      st = 'S'
      ccqp_import(T, INT, control, data, status, n, m,
                  "scaled_identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)

      ccqp_solve_qp(T, INT, data, status, n, m, H_ne, H_val, g, f,
                    A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat)
    end

    # identity
    if d == 6
      st = 'I'
      ccqp_import(T, INT, control, data, status, n, m,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)

      ccqp_solve_qp(T, INT, data, status, n, m, H_ne, H_val, g, f,
                    A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat)
    end

    # zero
    if d == 7
      st = 'Z'
      ccqp_import(T, INT, control, data, status, n, m,
                  "zero", H_ne, C_NULL, C_NULL, C_NULL,
                  "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)

      ccqp_solve_qp(T, INT, data, status, n, m, H_ne, H_val, g, f,
                    A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat)
    end

    ccqp_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: CCQP_solve exit status = %1i\n", st, inform[].status)
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
    ccqp_terminate(T, INT, data, control, inform)
  end

  # test shifted least-distance interface
  for d in 1:1

    # Initialize CCQP
    ccqp_initialize(T, INT, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # Start from 0
    x = T[0.0, 0.0, 0.0]
    y = T[0.0, 0.0]
    z = T[0.0, 0.0, 0.0]

    # Set shifted least-distance data

    w = T[1.0, 1.0, 1.0]
    x_0 = T[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'W'
      ccqp_import(T, INT, control, data, status, n, m,
                  "shifted_least_distance", H_ne, C_NULL, C_NULL, C_NULL,
                  "coordinate", A_ne, A_row, A_col, C_NULL)

      ccqp_solve_sldqp(T, INT, data, status, n, m, w, x_0, g, f,
                       A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                       x_stat, c_stat)
    end

    ccqp_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: CCQP_solve exit status = %1i\n", st, inform[].status)
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
    ccqp_terminate(T, INT, data, control, inform)
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
    @testset "CCQP -- $T -- $INT" begin
      @test test_ccqp(T, INT) == 0
    end
  end
end
