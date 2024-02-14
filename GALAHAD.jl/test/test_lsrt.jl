# test_lsrt.jl
# Simple code to test the Julia interface to LSRT

using GALAHAD
using Test
using Printf
using Accessors

function test_lsrt()
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lsrt_control_type{Float64}}()
  inform = Ref{lsrt_inform_type{Float64}}()

  # Set problem data
  n = 50 # dimensions
  m = 2 * n

  status = Ref{Cint}()
  power = 3.0
  weight = 1.0
  x = zeros(Float64, n)
  u = zeros(Float64, m)
  v = zeros(Float64, n)

  # Initialize lsrt
  lsrt_initialize(data, control, status)

  status[] = 1
  @reset control[].print_level = Cint(0)
  lsrt_import_control(control, data, status)

  for i in 1:m
    u[i] = 1.0 # b = 1
  end

  # iteration loop to find the minimizer with A^T = (I:diag(1:n))
  terminated = false
  while !terminated # reverse-communication loop
    lsrt_solve_problem(data, status, m, n, power, weight, x, u, v)
    if status[] == 0 # successful termination
      terminated = true
    elseif status[] < 0 # error exit
      terminated = true
    elseif status[] == 2 # form u <- u + A * v
      for i in 1:n
        u[i] = u[i] + v[i]
        u[n + i] = u[n + i] + (i + 1) * v[i]
      end
    elseif status[] == 3 # form v <- v + A^T * u
      for i in 1:n
        v[i] = v[i] + u[i] + (i + 1) * u[n + i]
      end
    elseif status[] == 4 # restart
      for i in 1:m
        u[i] = 1.0
      end
    else
      @printf(" the value %1i of status should not occur\n",
              status)
    end
  end

  lsrt_information(data, inform, status)
  @printf("lsrt_solve_problem exit status = %i, f = %.2f\n", inform[].status, inform[].obj)

  # Delete internal workspace
  lsrt_terminate(data, control, inform)

  return 0
end

@testset "LSRT" begin
  @test test_lsrt() == 0
end
