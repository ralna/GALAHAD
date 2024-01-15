# test_lstr.jl
# Simple code to test the Julia interface to LSTR

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = lstr_control_type{Float64}()
inform = lstr_inform_type{Float64}()

# Set problem data
n = 50  # dimensions
m = 2 * n

status = Ref{Cint}()
radius = Ref{Float64}()
x = zeros(Float64, n)
u = zeros(Float64, m)
v = zeros(Float64, n)

# Initialize lstr
lstr_initialize( data, control, status )

# resolve with a smaller radius ?
for new_radius = 0:1
  if new_radius == 0  # original radius
    radius[] = 1.0
    status[] = 1
  else  # smaller radius
    radius[] = 0.1
    status[] = 5
  end

  control.print_level = 0
  lstr_import_control( control, data, status )

  for i = 1:m
    u[i] = 1.0  # b = 1
  end

  # iteration loop to find the minimizer with A^T = (I:diag(1:n))
  while true  # reverse-communication loop
    lstr_solve_problem( data, status, m, n, radius, x, u, v )

    if status[] == 0  # successful termination
      break
    elseif status[] < 0  # error exit
      error("LSTR -- status = $(status[])")
    elseif status[] == 2  # form u <- u + A * v
      for i = 1:n
      u[i] = u[i] + v[i]
      u[n+i] = u[n+i] + i * v[i]
      end
    elseif status[] == 3  # form v <- v + A^T * u
      for i = 1:n
        v[i] = v[i] + u[i] + i * u[n+i]
      end
    elseif status[] == 4  # restart
      for i = 1:m
        u[i] = 1.0
      end
    else
      @printf(" the value %1i of status should not occur\n", status)
    end
  end

  lstr_information( data, inform, status )
  @printf("%1i lstr_solve_problem exit status = %i, f = %.2f\n", new_radius, inform.status, inform.r_norm )
end

# Delete internal workspace
lstr_terminate( data, control, inform )
