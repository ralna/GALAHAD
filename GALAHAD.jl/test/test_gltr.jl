# test_gltr.jl
# Simple code to test the Julia interface to GLTR

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_gltr(::Type{T}) where T
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{gltr_control_type{T}}()
  inform = Ref{gltr_inform_type{T}}()

  # Set problem data
  n = 100 # dimension

  status = Ref{Cint}()
  radius = Ref{T}()
  x = zeros(T, n)
  r = zeros(T, n)
  vector = zeros(T, n)
  h_vector = zeros(T, n)

  # Initialize gltr
  gltr_initialize(T, data, control, status)

  # use a unit M ?
  for unit_m in 0:1
    if unit_m == 0
      @reset control[].unitm = false
    else
      @reset control[].unitm = true
    end

    gltr_import_control(T, control, data, status)

    # resolve with a smaller radius ?
    for new_radius in 0:1
      if new_radius == 0
        radius[] = 1.0
        status[] = 1
      else
        radius[] = 0.1
        status[] = 4
      end

      for i in 1:n
        r[i] = 1.0
      end

      # iteration loop to find the minimizer
      terminated = false
      while !terminated # reverse-communication loop
        gltr_solve_problem(T, data, status, n, radius[], x, r, vector)
        if status[] == 0 # successful termination
          terminated = true
        elseif status[] < 0 # error exit
          terminated = true
        elseif status[] == 2 # form the preconditioned vector
          for i in 1:n
            vector[i] = vector[i] / 2.0
          end
        elseif status[] == 3 # form the Hessian-vector product
          h_vector[1] = 2.0 * vector[1] + vector[2]
          for i in 2:(n - 1)
            h_vector[i] = vector[i - 1] + 2.0 * vector[i] + vector[i + 1]
          end
          h_vector[n] = vector[n - 1] + 2.0 * vector[n]
          for i in 1:n
            vector[i] = h_vector[i]
          end
        elseif status[] == 5 # restart
          for i in 1:n
            r[i] = 1.0
          end
        else
          @printf(" the value %1i of status should not occur\n", status)
        end
      end

      gltr_information(T, data, inform, status)
      @printf("MR = %1i%1i gltr_solve_problem exit status = %i, f = %.2f\n", unit_m,
              new_radius, inform[].status, inform[].obj)
    end
  end

  # Delete internal workspace
  gltr_terminate(T, data, control, inform)

  return 0
end

@testset "GLTR" begin
  @test test_gltr(Float32) == 0
  @test test_gltr(Float64) == 0
  @test test_gltr(Float128) == 0
end
