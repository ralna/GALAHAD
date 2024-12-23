# test_cro.jl
# Simple code to test the Julia interface to CRO

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_cro(::Type{T}) where T
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{cro_control_type{T}}()
  inform = Ref{cro_inform_type{T}}()

  # Set problem dimensions
  n = 11 # dimension
  m = 3 # number of general constraints
  m_equal = 1 # number of equality constraints

  #  describe the objective function

  H_ne = 21
  H_val = T[1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0,
                  0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
  H_col = Cint[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11]
  H_ptr = Cint[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  g = T[0.5, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]

  #  describe constraints

  A_ne = 30
  A_val = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  A_col = Cint[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5,
               6, 7, 8, 9, 10, 11]
  A_ptr = Cint[1, 12, 21, 31]
  c_l = T[10.0, 9.0, -Inf]
  c_u = T[10.0, Inf, 10.0]
  x_l = T[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  x_u = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf]

  # provide optimal variables, Lagrange multipliers and dual variables
  x = T[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  c = T[10.0, 9.0, 10.0]
  y = T[-1.0, 1.5, -2.0]
  z = T[2.0, 4.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

  # provide interior-point constraint and variable status
  c_stat = Cint[-1, -1, 1]
  x_stat = Cint[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

  # Set output storage
  status = Ref{Cint}()
  @printf(" Fortran sparse matrix indexing\n\n")

  # Initialize CRO
  cro_initialize(T, data, control, status)

  # Set user-defined control options
  @reset control[].f_indexing = true # Fortran sparse matrix indexing

  # crossover the solution
  cro_crossover_solution(T, data, control, inform, n, m, m_equal, H_ne, H_val, H_col, H_ptr,
                         A_ne, A_val, A_col, A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z,
                         x_stat, c_stat)

  @printf(" CRO_crossover exit status = %1i\n", inform[].status)

  # Delete internal workspace
  cro_terminate(T, data, control, inform)

  return 0
end

@testset "CRO" begin
  @test test_cro(Float32) == 0
  @test test_cro(Float64) == 0
  @test test_cro(Float128) == 0
end
