# test_cro.jl
# Simple code to test the Julia interface to CRO

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_cro(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{cro_control_type{T,INT}}()
  inform = Ref{cro_inform_type{T,INT}}()

  # Set problem dimensions
  n = INT(11)  # dimension
  m = INT(3)  # number of general constraints
  m_equal = INT(1)  # number of equality constraints

  #  describe the objective function

  H_ne = INT(21)
  H_val = T[1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0,
                  0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
  H_col = INT[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11]
  H_ptr = INT[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  g = T[0.5, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5]

  #  describe constraints

  A_ne = INT(30)
  A_val = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  A_col = INT[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5,
               6, 7, 8, 9, 10, 11]
  A_ptr = INT[1, 12, 21, 31]
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
  c_stat = INT[-1, -1, 1]
  x_stat = INT[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

  # Set output storage
  status = Ref{INT}()
  @printf(" Fortran sparse matrix indexing\n\n")

  # Initialize CRO
  cro_initialize(T, INT, data, control, status)

  # Set user-defined control options
  @reset control[].f_indexing = true # Fortran sparse matrix indexing

  # crossover the solution
  cro_crossover_solution(T, INT, data, control, inform, n, m, m_equal, H_ne, H_val, H_col, H_ptr,
                         A_ne, A_val, A_col, A_ptr, g, c_l, c_u, x_l, x_u, x, c, y, z,
                         x_stat, c_stat)

  @printf(" CRO_crossover exit status = %1i\n", inform[].status)

  # Delete internal workspace
  cro_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "CRO -- $T -- $INT" begin
      @test test_cro(T, INT) == 0
    end
  end
end
