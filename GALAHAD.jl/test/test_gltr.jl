# test_gltr.jl
# Simple code to test the Julia interface to GLTR

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = gltr_control_type{Float64}()
inform = gltr_inform_type{Float64}()

# Set problem data
n = 100  # dimension

status = Ref{Cint}()
radius = 0
x = zeros(Float64, n)
r = zeros(Float64, n)
vector = zeros(Float64, n)
h_vector = zeros(Float64, n)

# Initialize gltr
gltr_initialize( data, control, status )

# use a unit M ?
for unit_m = 0:1
  if unit_m == 0
    control.unitm = false
  else
    control.unitm = true
  end
  gltr_import_control( control, data, status )

  # resolve with a smaller radius ?
  for new_radius = 0:1
    if new_radius == 0
      global radius = 1.0
      status[] = 1
    else
      global radius = 0.1
      status[] = 4
    end

    for i = 1:n
      r[i] = 1.0
    end

    # iteration loop to find the minimizer
    while(true) # reverse-communication loop
      gltr_solve_problem( data, status, n, radius, x, r, vector )
      if status[] == 0  # successful termination
        break
      elseif status[] < 0  # error exit
        error("GLTR -- status = $(status[])")
      elseif status[] == 2  # form the preconditioned vector
        for i = 1:n
          vector[i] = vector[i] / 2.0
        end
      elseif status[] == 3  # form the Hessian-vector product
        h_vector[1] = 2.0 * vector[1] + vector[2]
        for i = 2:n-1
          h_vector[i] = vector[i-1] + 2.0 * vector[i] + vector[i+1]
        end
        h_vector[n] = vector[n-1] + 2.0 * vector[n]
        for i = 1:n
          vector[i] = h_vector[i]
        end
      elseif status[] == 5  # restart
        for i = 1:n
          r[i] = 1.0
        end
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
    end

    gltr_information( data, inform, status )
    @printf("MR = %1i%1i gltr_solve_problem exit status = %i, f = %.2f\n", unit_m, new_radius, inform.status, inform.obj )
  end
end

# Delete internal workspace
gltr_terminate( data, control, inform )
