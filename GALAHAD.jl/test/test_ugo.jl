# test_ugo.jl
# Simple code to test the Julia interface to UGO

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_ugo(::Type{T}, ::Type{INT}) where {T,INT}
  # Test problem objective
  function objf(x::T)
    a = 10.0
    res = x * x * cos(a * x)
    return res
  end

  # Test problem first derivative
  function gradf(x::T)
    a = 10.0
    res = -a * x * x * sin(a * x) + 2.0 * x * cos(a * x)
    return res
  end

  # Test problem second derivative
  function hessf(x::T)
    a = 10.0
    res = -a * a * x * x * cos(a * x) - 4.0 * a * x * sin(a * x) + 2.0 * cos(a * x)
    return res
  end

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ugo_control_type{T,INT}}()
  inform = Ref{ugo_inform_type{T,INT}}()

  # Initialize UGO
  status = Ref{INT}()
  eval_status = Ref{INT}()
  ugo_initialize(T, INT, data, control, status)

  # Set user-defined control options
  @reset control[].print_level = INT(1)

  # Test problem bounds
  x_l = Ref{T}(-1.0)
  x_u = Ref{T}(2.0)

  # Test problem objective, gradient, Hessian values
  x = Ref{T}(0.0)
  f = Ref{T}(objf(x[]))
  g = Ref{T}(gradf(x[]))
  h = Ref{T}(hessf(x[]))

  # import problem data
  ugo_import(T, INT, control, data, status, x_l, x_u)

  # Set for initial entry
  status[] = 1

  # Solve the problem: min f(x), x_l ≤ x ≤ x_u
  terminated = false
  while !terminated
    # Call UGO_solve
    ugo_solve_reverse(T, INT, data, status, eval_status, x, f, g, h)

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
  ugo_information(T, INT, data, inform, status)

  if inform[].status == 0
    @printf("%i evaluations. Optimal objective value = %5.2f status = %1i\n",
            inform[].f_eval, f[], inform[].status)
  else
    @printf("UGO_solve exit status = %1i\n", inform[].status)
  end

  # Delete internal workspace
  ugo_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "UGO -- $T -- $INT" begin
      @test test_ugo(T, INT) == 0
    end
  end
end
