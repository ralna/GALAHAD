# test_bllsb.jl
# Simple code to test the Julia interface to BLLSB

using GALAHAD
using Test
using Printf
using Accessors

function test_bllsb()
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bllsb_control_type{Float64}}()
  inform = Ref{bllsb_inform_type{Float64}}()

  # Set problem data
  n = 3 # dimension
  o = 4 # number of observations
  sigma = 1.0 # regularization weight
  b = Float64[2.0, 2.0, 3.0, 1.0]  # observations
  x_l = Float64[-1.0, -Inf, -Inf]  # variable lower bound
  x_u = Float64[1.0, Inf, 2.0]  # variable upper bound
  w = Float64[1.0, 1.0, 1.0, 2.0]  # weights

  # Set output storage
  r = zeros(Float64, o) # residual values
  x_stat = zeros(Cint, n) # variable status
  st = ""
  status = Ref{Cint}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of bllsb storage formats\n\n")

  for d in 1:5
    # Initialize BLLSB
    bllsb_initialize(data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    @reset control[].symmetric_linear_solver = galahad_linear_solver("potr")
    @reset control[].fdc_control.symmetric_linear_solver = galahad_linear_solver("potr")
    @reset control[].fdc_control.use_sls = true

    # Start from 0
    x = Float64[0.0, 0.0, 0.0]
    y = Float64[0.0, 0.0]
    z = Float64[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = "CO"
      Ao_ne = 7 # objective Jacobian elements
      Ao_row = Cint[1, 1, 2, 2, 3, 3, 4]  # row indices
      Ao_col = Cint[1, 2, 2, 3, 1, 3, 2]  # column indices
      Ao_val = Float64[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # vals

      bllsb_import(control, data, status, n, o,
                   "coordinate", Ao_ne, Ao_row, Ao_col, 0, Cint[])

      bllsb_solve_blls(data, status, n, o, Ao_ne, Ao_val, b,
                       sigma, x_l, x_u, x, r, z, x_stat, w)
    end

    # sparse by rows
    if d == 2
      st = "SR"

      Ao_ne = 7 # objective Jacobian elements
      Ao_col = Cint[1, 2, 2, 3, 1, 3, 2]  # column indices
      Ao_ptr_ne = o + 1 # number of row pointers
      Ao_ptr = Cint[1, 3, 5, 7, 8]  # row pointers
      Ao_val = Float64[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # vals

      bllsb_import(control, data, status, n, o,
                   "sparse_by_rows", Ao_ne, Cint[], Ao_col,
                   Ao_ptr_ne, Ao_ptr)

      bllsb_solve_blls(data, status, n, o, Ao_ne, Ao_val, b,
                       sigma, x_l, x_u, x, r, z, x_stat, w)
    end

    # sparse by columns
    if d == 3
      st = "SC"
      Ao_ne = 7 # objective Jacobian elements
      Ao_row = Cint[1, 3, 1, 2, 4, 2, 3]  # row indices
      Ao_ptr_ne = n + 1 # number of column pointers
      Ao_ptr = Cint[1, 3, 6, 8]  # column pointers
      Ao_val = Float64[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # vals

      bllsb_import(control, data, status, n, o,
                   "sparse_by_columns", Ao_ne, Ao_row, Cint[],
                   Ao_ptr_ne, Ao_ptr)

      bllsb_solve_blls(data, status, n, o, Ao_ne, Ao_val, b,
                       sigma, x_l, x_u, x, r, z, x_stat, w)
    end

    if d == 4 # dense by rows
      st = "DR"
      Ao_ne = 12 # objective Jacobian elements
      Ao_dense = Float64[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

      bllsb_import(control, data, status, n, o,
                   "dense", Ao_ne, Cint[], Cint[], 0, Cint[])

      bllsb_solve_blls(data, status, n, o, Ao_ne, Ao_dense, b,
                       sigma, x_l, x_u, x, r, z, x_stat, w)
    end

    if d == 5 # dense by cols
      st = "DC"
      Ao_ne = 12 # objective Jacobian elements
      Ao_dense = Float64[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

      bllsb_import(control, data, status, n, o,
                   "dense_by_columns", Ao_ne, Cint[], Cint[], 0, Cint[])

      bllsb_solve_blls(data, status, n, o, Ao_ne, Ao_dense, b,
                       sigma, x_l, x_u, x, r, z, x_stat, w)
    end

    bllsb_information(data, inform, status)

    if inform[].status == 0
      @printf("%s:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%s: BLLSB_solve exit status = %1i\n", st, inform[].status)
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
    bllsb_terminate(data, control, inform)
  end

  return 0
end

@testset "BLLSB" begin
  @test test_bllsb() == 0
end
