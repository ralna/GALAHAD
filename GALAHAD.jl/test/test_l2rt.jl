# test_l2rt.jl
# Simple code to test the Julia interface to L2RT

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_l2rt(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{l2rt_control_type{T,INT}}()
  inform = Ref{l2rt_inform_type{T,INT}}()

  # Set problem data
  n = INT(50)  # dimensions
  m = 2 * n

  status = Ref{INT}()
  power = T(3.0)
  weight = one(T)
  shift = one(T)
  x = zeros(T, n)
  u = zeros(T, m)
  v = zeros(T, n)

  # Initialize l2rt
  l2rt_initialize(T, INT, data, control, status)

  status[] = 1
  @reset control[].print_level = INT(0)
  l2rt_import_control(T, INT, control, data, status)

  for i in 1:m
    u[i] = 1.0 # b = 1
  end

  # iteration loop to find the minimizer with A^T = (I:diag(1:n))
  terminated = false
  while !terminated # reverse-communication loop
    l2rt_solve_problem(T, INT, data, status, m, n, power, weight, shift, x, u, v)
    if status[] == 0 # successful termination
      terminated = true
    elseif status[] < 0 # error exit
      terminated = true
    elseif status[] == 2 # form u <- u + A * v
      for i in 1:n
        u[i] = u[i] + v[i]
        u[n + i] = u[n + i] + i * v[i]
      end
    elseif status[] == 3 # form v <- v + A^T * u
      for i in 1:n
        v[i] = v[i] + u[i] + i * u[n + i]
      end
    elseif status[] == 4 # restart
      for i in 1:m
        u[i] = 1.0
      end
    else
      @printf(" the value %1i of status should not occur\n", status)
    end
  end

  l2rt_information(T, INT, data, inform, status)

  @printf("l2rt_solve_problem exit status = %i, f = %.2f\n", inform[].status, inform[].obj)

  # Delete internal workspace
  l2rt_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "L2RT -- $T -- $INT" begin
      @test test_l2rt(T, INT) == 0
    end
  end
end
