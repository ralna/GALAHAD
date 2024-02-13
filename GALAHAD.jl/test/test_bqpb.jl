# test_bqpb.jl
# Simple code to test the Julia interface to BQPB

using GALAHAD
using Test
using Printf
using Accessors

function test_bqpb()
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bqpb_control_type{Float64}}()
  inform = Ref{bqpb_inform_type{Float64}}()

  # Set problem data
  n = 3 # dimension
  H_ne = 3 # Hesssian elements
  H_row = Cint[1, 2, 3]  # row indices, NB lower triangle
  H_col = Cint[1, 2, 3]  # column indices, NB lower triangle
  H_ptr = Cint[1, 2, 3, 4]  # row pointers
  H_val = Float64[1.0, 1.0, 1.0]  # values
  g = Float64[2.0, 0.0, 0.0]  # linear term in the objective
  f = 1.0  # constant term in the objective
  x_l = Float64[-1.0, -Inf, -Inf]  # variable lower bound
  x_u = Float64[1.0, Inf, 2.0]  # variable upper bound

  # Set output storage
  x_stat = zeros(Cint, n) # variable status
  st = ' '
  status = Ref{Cint}()

  @printf(" Fortran sparse matrix indexing\n\n")

  @printf(" basic tests of qp storage formats\n\n")

  for d in 1:7

    # Initialize BQPB
    bqpb_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # Start from 0
    x = Float64[0.0, 0.0, 0.0]
    z = Float64[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      bqpb_import(control, data, status, n,
                  "coordinate", H_ne, H_row, H_col, C_NULL)

      bqpb_solve_qp(data, status, n, H_ne, H_val, g, f,
                    x_l, x_u, x, z, x_stat)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      bqpb_import(control, data, status, n,
                  "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr)

      bqpb_solve_qp(data, status, n, H_ne, H_val, g, f,
                    x_l, x_u, x, z, x_stat)
    end

    # dense
    if d == 3
      st = 'D'
      H_dense_ne = 6 # number of elements of H
      H_dense = Float64[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
      bqpb_import(control, data, status, n,
                  "dense", H_ne, C_NULL, C_NULL, C_NULL)

      bqpb_solve_qp(data, status, n, H_dense_ne, H_dense, g, f,
                    x_l, x_u, x, z, x_stat)
    end

    # diagonal
    if d == 4
      st = 'L'
      bqpb_import(control, data, status, n,
                  "diagonal", H_ne, C_NULL, C_NULL, C_NULL)

      bqpb_solve_qp(data, status, n, H_ne, H_val, g, f,
                    x_l, x_u, x, z, x_stat)
    end

    # scaled identity
    if d == 5
      st = 'S'
      bqpb_import(control, data, status, n,
                  "scaled_identity", H_ne, C_NULL, C_NULL, C_NULL)

      bqpb_solve_qp(data, status, n, H_ne, H_val, g, f,
                    x_l, x_u, x, z, x_stat)
    end

    # identity
    if d == 6
      st = 'I'
      bqpb_import(control, data, status, n,
                  "identity", H_ne, C_NULL, C_NULL, C_NULL)

      bqpb_solve_qp(data, status, n, H_ne, H_val, g, f,
                    x_l, x_u, x, z, x_stat)
    end

    # zero
    if d == 7
      st = 'Z'
      bqpb_import(control, data, status, n,
                  "zero", H_ne, C_NULL, C_NULL, C_NULL)

      bqpb_solve_qp(data, status, n, H_ne, H_val, g, f,
                    x_l, x_u, x, z, x_stat)
    end

    bqpb_information(data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: BQPB_solve exit status = %1i\n", st, inform[].status)
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
    bqpb_terminate(data, control, inform)
  end

  # test shifted least-distance interface
  for d in 1:1

    # Initialize BQPB
    bqpb_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # Start from 0
    x = Float64[0.0, 0.0, 0.0]
    z = Float64[0.0, 0.0, 0.0]

    # Set shifted least-distance data

    w = Float64[1.0, 1.0, 1.0]
    x_0 = Float64[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = 'W'
      bqpb_import(control, data, status, n,
                  "shifted_least_distance", H_ne, C_NULL, C_NULL, C_NULL)

      bqpb_solve_sldqp(data, status, n, w, x_0, g, f,
                       x_l, x_u, x, z, x_stat)
    end

    bqpb_information(data, inform, status)

    if inform[].status == 0
      @printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
              st, inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%c: BQPB_solve exit status = %1i\n", st, inform[].status)
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
    bqpb_terminate(data, control, inform)
  end

  return 0
end

@testset "BQPB" begin
  @test test_bqpb() == 0
end
