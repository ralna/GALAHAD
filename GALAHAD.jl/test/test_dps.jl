# test_dps.jl
# Simple code to test the Julia interface to DPS

using GALAHAD
using Test
using Printf
using Accessors

function test_dps()
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{dps_control_type{Float64}}()
  inform = Ref{dps_inform_type{Float64}}()

  # Set problem data
  n = 3 # dimension of H
  m = 1 # dimension of A
  H_ne = 4 # number of elements of H
  H_dense_ne = 6 # number of elements of H
  H_row = Cint[1, 2, 3, 3]  # row indices, NB lower triangle
  H_col = Cint[1, 2, 3, 1]
  H_ptr = Cint[1, 2, 3, 5]
  H_val = Float64[1.0, 2.0, 3.0, 4.0]
  H_dense = Float64[1.0, 0.0, 2.0, 4.0, 0.0, 3.0]
  f = 0.96
  radius = 1.0
  half_radius = 0.5
  c = Float64[0.0, 2.0, 0.0]

  st = ' '
  status = Ref{Cint}()
  x = zeros(Float64, n)

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  for storage_type in 1:3
    # Initialize DPS
    dps_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing 
    @reset control[].symmetric_linear_solver = convert(NTuple{31,Int8},
                                                       (115, 121, 116, 114, 32, 32, 32, 32,
                                                        32, 32,
                                                        32, 32, 32, 32, 32, 32, 32, 32, 32,
                                                        32, 32,
                                                        32, 32, 32, 32, 32, 32, 32, 32, 32,
                                                        0))

    # sparse co-ordinate storage
    if storage_type == 1
      st = 'C'
      # import the control parameters and structural data
      dps_import(control, data, status, n,
                 "coordinate", H_ne, H_row, H_col, Cint[])

      # solve the problem
      dps_solve_tr_problem(data, status, n, H_ne, H_val,
                           c, f, radius, x)
    end

    # sparse by rows
    if storage_type == 2
      st = 'R'
      # import the control parameters and structural data
      dps_import(control, data, status, n,
                 "sparse_by_rows", H_ne, Cint[], H_col, H_ptr)

      dps_solve_tr_problem(data, status, n, H_ne, H_val,
                           c, f, radius, x)
    end

    # dense
    if storage_type == 3
      st = 'D'
      # import the control parameters and structural data
      dps_import(control, data, status, n,
                 "dense", H_ne, Cint[], Cint[], Cint[])

      dps_solve_tr_problem(data, status, n, H_dense_ne, H_dense,
                           c, f, radius, x)
    end

    dps_information(data, inform, status)
    @printf("format %c: DPS_solve_problem exit status   = %1i, f = %.2f\n",
            st, inform[].status, inform[].obj)

    # sparse co-ordinate storage
    if storage_type == 1
      st = 'C'
      # solve the problem
      dps_resolve_tr_problem(data, status, n,
                             c, f, half_radius, x)
    end

    # sparse by rows
    if storage_type == 2
      st = 'R'
      dps_resolve_tr_problem(data, status, n,
                             c, f, half_radius, x)
    end

    # dense
    if storage_type == 3
      st = 'D'
      dps_resolve_tr_problem(data, status, n,
                             c, f, half_radius, x)
    end

    dps_information(data, inform, status)
    @printf("format %c: DPS_resolve_problem exit status = %1i, f = %.2f\n",
            st, inform[].status, inform[].obj)

    # @printf("x: ")
    # for i = 1:n+m
    #   @printf("%f ", x[i])
    # end

    # Delete internal workspace
    dps_terminate(data, control, inform)
  end

  return 0
end

@testset "DPS" begin
  @test test_dps() == 0
end
