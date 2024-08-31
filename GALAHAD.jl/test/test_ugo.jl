# test_ugo.jl
# Simple code to test the Julia interface to UGO

using GALAHAD
using Test
using Printf
using Accessors

function test_ugo()
  # Test problem objective
  function objf(x::Float64)
    a = 10.0
    res = x * x * cos(a * x)
    return res
  end

  # Test problem first derivative
  function gradf(x::Float64)
    a = 10.0
    res = -a * x * x * sin(a * x) + 2.0 * x * cos(a * x)
    return res
  end

  # Test problem second derivative
  function hessf(x::Float64)
    a = 10.0
    res = -a * a * x * x * cos(a * x) - 4.0 * a * x * sin(a * x) + 2.0 * cos(a * x)
    return res
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ugo_control_type{Float64}}()
  inform = Ref{ugo_inform_type{Float64}}()

  # Initialize UGO
  status = Ref{Cint}()
  eval_status = Ref{Cint}()
  ugo_initialize(Float64, data, control, status)

  # Set user-defined control options
  @reset control[].print_level = Cint(1)

  # Test problem bounds
  x_l = Ref{Float64}(-1.0)
  x_u = Ref{Float64}(2.0)

  # Test problem objective, gradient, Hessian values
  x = Ref{Float64}(0.0)
  f = Ref{Float64}(objf(x[]))
  g = Ref{Float64}(gradf(x[]))
  h = Ref{Float64}(hessf(x[]))

  # import problem data
  ugo_import(Float64, control, data, status, x_l, x_u)

  # Set for initial entry
  status[] = 1

  # Solve the problem: min f(x), x_l ≤ x ≤ x_u
  terminated = false
  while !terminated
    # Call UGO_solve
    ugo_solve_reverse(Float64, data, status, eval_status, x, f, g, h)

    # Evaluate f(x) and its derivatives as required
    if (status[] ≥ 2)  # need objective
      f[] = objf(x[])
      if (status[] ≥ 3)  # need first derivative
        g[] = gradf(x[])
        if (status[] ≥ 4) # need second derivative
          h[] = hessf(x[])
        end
      end
    else  # the solution has been found (or an error has occured)
      terminated = true
    end
  end

  # Record solution information
  ugo_information(Float64, data, inform, status)

  if inform[].status == 0
    @printf("%i evaluations. Optimal objective value = %5.2f status = %1i\n",
            inform[].f_eval, f[], inform[].status)
  else
    @printf("UGO_solve exit status = %1i\n", inform[].status)
  end

  # Delete internal workspace
  ugo_terminate(Float64, data, control, inform)

  return 0
end

@testset "UGO" begin
  @test test_ugo() == 0
end
