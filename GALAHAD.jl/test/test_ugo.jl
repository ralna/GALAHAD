# test_ugo.jl
# Simple code to test the Julia interface to UGO

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

# Custom userdata struct
mutable struct userdata_ugo{T}
  a::T
end

function test_ugo(::Type{T}, ::Type{INT}; mode::String="reverse") where {T,INT}
  # Test problem objective
  function objf(x::T, userdata::userdata_ugo{T})
    a = userdata.a
    res = x * x * cos(a * x)
    return res
  end

  # Test problem first derivative
  function gradf(x::T, userdata::userdata_ugo{T})
    a = userdata.a
    res = -a * x * x * sin(a * x) + 2.0 * x * cos(a * x)
    return res
  end

  # Test problem second derivative
  function hessf(x::T, userdata::userdata_ugo{T})
    a = userdata.a
    res = -a * a * x * x * cos(a * x) - 4.0 * a * x * sin(a * x) + 2.0 * cos(a * x)
    return res
  end

  # Evaluate test problem objective, first and second derivatives
  function eval_fgh(x::T, f::Ptr{T}, g::Ptr{T}, h::Ptr{T}, userdata::Ptr{Cvoid})::INT
    _f = unsafe_wrap(Vector{T}, f, 1)
    _g = unsafe_wrap(Vector{T}, g, 1)
    _h = unsafe_wrap(Vector{T}, h, 1)
    _userdata = unsafe_pointer_to_objref(userdata)::userdata_ugo{T}

    _f[1] = objf(x, _userdata)
    _g[1] = gradf(x, _userdata)
    _h[1] = hessf(x, _userdata)
    return INT(0)
  end

  eval_fgh_ptr = @eval @cfunction($eval_fgh, $INT, ($T, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{Cvoid}))

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ugo_control_type{T,INT}}()
  inform = Ref{ugo_inform_type{T,INT}}()

  # Set user data
  userdata = userdata_ugo{T}(10)
  userdata_ptr = pointer_from_objref(userdata)

  # Initialize UGO
  status = Ref{INT}()
  eval_status = Ref{INT}()
  ugo_initialize(T, INT, data, control, status)

  # Set user-defined control options
  # @reset control[].print_level = INT(1)

  # Test problem bounds
  x_l = Ref{T}(-1.0)
  x_u = Ref{T}(2.0)

  # Test problem objective, gradient, Hessian values
  x = zeros(T, 1)
  f = T[objf(x[1], userdata)]
  g = T[gradf(x[1], userdata)]
  h = T[hessf(x[1], userdata)]

  # import problem data
  ugo_import(T, INT, control, data, status, x_l, x_u)

  # Set for initial entry
  status[] = 1

  # Solve the problem: min f(x), x_l ≤ x ≤ x_u
  if mode == "direct"
    # Call UGO_solve
    ugo_solve_direct(T, INT, data, userdata_ptr, status, x, f, g, h, eval_fgh_ptr)
  end

  if mode == "reverse"
    terminated = false
    while !terminated
      # Call UGO_solve
      ugo_solve_reverse(T, INT, data, status, eval_status, x, f, g, h)

      # Evaluate f(x) and its derivatives as required
      if (status[] ≥ 2)  # need objective
        f[1] = objf(x[1], userdata)
        if (status[] ≥ 3)  # need first derivative
          g[1] = gradf(x[1], userdata)
          if (status[] ≥ 4) # need second derivative
            h[1] = hessf(x[1], userdata)
          end
        end
      else  # the solution has been found (or an error has occured)
        terminated = true
      end
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
      @testset "$mode communication" for mode in ("reverse", "direct")
        @test test_ugo(T, INT; mode) == 0
      end
    end
  end
end
