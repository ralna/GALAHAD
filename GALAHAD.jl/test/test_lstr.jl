# test_lstr.jl
# Simple code to test the Julia interface to LSTR

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_lstr(::Type{T}) where T
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lstr_control_type{T}}()
  inform = Ref{lstr_inform_type{T}}()

  # Set problem data
  n = 50 # dimensions
  m = 2 * n

  status = Ref{Cint}()
  radius = Ref{T}()
  x = zeros(T, n)
  u = zeros(T, m)
  v = zeros(T, n)

  # Initialize lstr
  lstr_initialize(T, data, control, status)

  # Set user-defined control options
  @reset control[].f_indexing = true # Fortran sparse matrix indexing

  # resolve with a smaller radius ?
  for new_radius in 0:1
    if new_radius == 0 # original radius
      radius[] = 1.0
      status[] = 1
    else # smaller radius
      radius[] = 0.1
      status[] = 5
    end

    @reset control[].print_level = Cint(0)
    lstr_import_control(T, control, data, status)

    for i in 1:m
      u[i] = 1.0 # b = 1
    end

    # iteration loop to find the minimizer with A^T = (I:diag(1:n))
    terminated = false
    while !terminated # reverse-communication loop
      lstr_solve_problem(T, data, status, m, n, radius[], x, u, v)
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

    lstr_information(T, data, inform, status)
    # @printf("%1i lstr_solve_problem exit status = %i, f = %.2f\n", new_radius,
    #         inform[].status, inform[].r_norm)
  end

  # Delete internal workspace
  lstr_terminate(T, data, control, inform)

  return 0
end

@testset "LSTR" begin
  @test test_lstr(Float32) == 0
  @test test_lstr(Float64) == 0
  @test test_lstr(Float128) == 0
end
