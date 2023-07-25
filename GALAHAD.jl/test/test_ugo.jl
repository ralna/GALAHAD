# test_ugo.jl
# Simple code to test the Julia interface to UGO

using GALAHAD
using Printf

# Test problem objective
function objf(x)
  a = 10.0
  res = x * x * cos( a*x )
  return Ref{Float64}(res)
end

# Test problem first derivative
function gradf(x)
  a = 10.0
  res = - a * x * x * sin( a*x ) + 2.0 * x * cos( a*x )
  return Ref{Float64}(res)
end

# Test problem second derivative
function hessf(x)
  a = 10.0
  res = - a * a* x * x * cos( a*x ) - 4.0 * a * x * sin( a*x ) + 2.0 * cos( a*x )
  return Ref{Float64}(res)
end

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = ugo_control_type{Float64}()
inform = ugo_inform_type{Float64}()

# Initialize UGO
status = Ref{Cint}()
eval_status = Ref{Cint}()
ugo_initialize( data, control, status )

# Set user-defined control options
control.print_level = 1

# control.prefix = "'ugo: '"
control.prefix = (34, 39, 117, 103, 111, 58, 32, 39, 34, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0)

# Read options from specfile
specfile = "UGO.SPC"
ugo_read_specfile(control, specfile)

# Test problem bounds
x_l = Ref{Float64}(-1.0)
x_u = Ref{Float64}(2.0)

# Test problem objective, gradient, Hessian values
global x = Ref{Float64}(0.0)
global f = Ref{Float64}(0.0)
global g = Ref{Float64}(0.0)
global h = Ref{Float64}(0.0)

# import problem data
ugo_import( control, data, status, x_l, x_u )

# Set for initial entry
status = Ref{Cint}(1)

# Solve the problem: min f(x), x_l ≤ x ≤ x_u
while true

  # Call UGO_solve
  ugo_solve_reverse(data, status, eval_status, x, f, g, h )

  # Evaluate f(x) and its derivatives as required
  if (status[] ≥ 2)  # need objective
    global f = objf(x[])
    if (status[] ≥ 3)  # need first derivative
      global g = gradf(x[])
      if (status[] ≥ 4) # need second derivative
        global h = hessf(x[])
      end
    end
  else  # the solution has been found (or an error has occured)
    break
  end
end

# Record solution information
ugo_information( data, inform, status )

if inform.status == 0
  @printf("%i evaluations. Optimal objective value = %5.2f status = %1i\n", inform.f_eval, f[], inform.status)
else
  @printf("BGO_solve exit status = %1i\n", inform.status)
end

# Delete internal workspace
ugo_terminate( data, control, inform )
