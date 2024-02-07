# test_cqp.jl
# Simple code to test the Julia interface to CQP

using GALAHAD
using Test
using Printf
using Accessors

function test_cqp()
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{cqp_control_type{Float64}}()
  inform = Ref{cqp_inform_type{Float64}}()

  # Set problem data
  n = 3 # dimension
  m = 2 # number of general constraints
  H_ne = 3 # Hesssian elements
  H_row = Cint[1, 2, 3]  # row indices, NB lower triangle
  H_col = Cint[1, 2, 3]  # column indices, NB lower triangle
  H_ptr = Cint[1, 2, 3, 4]  # row pointers
  H_val = Float64[1.0, 1.0, 1.0]  # values
  g = Float64[0.0, 2.0, 0.0]  # linear term in the objective
  f = 1.0  # constant term in the objective
  A_ne = 4 # Jacobian elements
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
  @printf(" basic tests of qp storage formats\n\n")

  for d in 1:7
    # Initialize CQP
    cqp_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # Start from 0
    x = Float64[0.0, 0.0, 0.0]
    y = Float64[0.0, 0.0]
    z = Float64[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      cqp_import(control, data, status, n, m,
                 "coordinate", H_ne, H_row, H_col, Cint[],
                 "coordinate", A_ne, A_row, A_col, Cint[])

      cqp_solve_qp(data, status, n, m, H_ne, H_val, g, f,
                   A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                   x_stat, c_stat)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      cqp_import(control, data, status, n, m,
                 "sparse_by_rows", H_ne, Cint[], H_col, H_ptr,
                 "sparse_by_rows", A_ne, Cint[], A_col, A_ptr)

      cqp_solve_qp(data, status, n, m, H_ne, H_val, g, f,
                   A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                   x_stat, c_stat)
    end
    # dense
    if d == 3
      st = 'D'
      H_dense_ne = 6 # number of elements of H
      A_dense_ne = 6 # number of elements of A
      H_dense = Float64[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
      A_dense = Float64[2.0, 1.0, 0.0, 0.0, 1.0, 1.0]

      cqp_import(control, data, status, n, m,
                 "dense", H_ne, Cint[], Cint[], Cint[],
                 "dense", A_ne, Cint[], Cint[], Cint[])

      cqp_solve_qp(data, status, n, m, H_dense_ne, H_dense, g, f,
                   A_dense_ne, A_dense, c_l, c_u, x_l, x_u,
                   x, c, y, z, x_stat, c_stat)
    end

    # diagonal
    if d == 4
      st = 'L'
      cqp_import(control, data, status, n, m,
                 "diagonal", H_ne, Cint[], Cint[], Cint[],
                 "sparse_by_rows", A_ne, Cint[], A_col, A_ptr)

      cqp_solve_qp(data, status, n, m, H_ne, H_val, g, f,
                   A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                   x_stat, c_stat)
    end

    # scaled identity
    if d == 5
      st = 'S'
      cqp_import(control, data, status, n, m,
                 "scaled_identity", H_ne, Cint[], Cint[], Cint[],
                 "sparse_by_rows", A_ne, Cint[], A_col, A_ptr)

      cqp_solve_qp(data, status, n, m, H_ne, H_val, g, f,
                   A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                   x_stat, c_stat)
    end

    # identity
    if d == 6
      st = 'I'
      cqp_import(control, data, status, n, m,
                 "identity", H_ne, Cint[], Cint[], Cint[],
                 "sparse_by_rows", A_ne, Cint[], A_col, A_ptr)

      cqp_solve_qp(data, status, n, m, H_ne, H_val, g, f,
                   A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                   x_stat, c_stat)
    end

    # zero
    if d == 7
      st = 'Z'
      cqp_import(control, data, status, n, m,
                 "zero", H_ne, Cint[], Cint[], Cint[],
                 "sparse_by_rows", A_ne, Cint[], A_col, A_ptr)

      cqp_solve_qp(data, status, n, m, H_ne, H_val, g, f,
                   A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                   x_stat, c_stat)
    end

    cqp_information(data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: CQP_solve exit status = %1i\n", st, inform[].status)
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
    cqp_terminate(data, control, inform)
  end

  # test shifted least-distance interface
  for d in 1:1
    # Initialize CQP
    cqp_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # Start from 0
    x = Float64[0.0, 0.0, 0.0]
    y = Float64[0.0, 0.0]
    z = Float64[0.0, 0.0, 0.0]

    # Set shifted least-distance data

    w = Float64[1.0, 1.0, 1.0]
    x_0 = Float64[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'W'
      cqp_import(control, data, status, n, m,
                 "shifted_least_distance", H_ne, Cint[], Cint[], Cint[],
                 "coordinate", A_ne, A_row, A_col, Cint[])

      cqp_solve_sldqp(data, status, n, m, w, x_0, g, f,
                      A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                      x_stat, c_stat)
    end

    cqp_information(data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: CQP_solve exit status = %1i\n", st, inform[].status)
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
    cqp_terminate(data, control, inform)
  end

  return 0
end

@testset "CQP" begin
  @test test_cqp() == 0
end
