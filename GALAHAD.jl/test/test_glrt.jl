# test_glrt.jl
# Simple code to test the Julia interface to GLRT

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = glrt_control_type{Float64}()
inform = glrt_inform_type{Float64}()

# Set problem data
n = 100  # dimension

status = Ref{Cint}()
weight = 0
power = 3.0
x = zeros(Float64, n)
r = zeros(Float64, n)
vector = zeros(Float64, n)
h_vector = zeros(Float64, n)

# Initialize glrt
glrt_initialize( data, control, status )

# use a unit M ?
for unit_m = 0:1
  if unit_m == 0
    control.unitm = false
  else
    control.unitm = true
  end

  glrt_import_control( control, data, status )

  # resolve with a larger weight ?
  for new_weight = 0:1
    if new_weight == 0
      global weight = 1.0 
      status[] = 1
    else
      global weight = 10.0 
      status[] = 6
    end

    for i = 1:n
      r[i] = 1.0
    end

    # iteration loop to find the minimizer
    while true  # reverse-communication loop
      glrt_solve_problem( data, status, n, power, weight, x, r, vector )
      if status[] == 0  # successful termination
        break
      elseif status[] < 0  # error exit
        error("GLRT -- status = $(status[])")
      elseif status[] == 2  # form the preconditioned vector
        for i = 1:n
          vector[i] = vector[i] / 2.0
        end
      elseif status[] == 3  # form the Hessian-vector product
        h_vector[1] =  2.0 * vector[1] + vector[2]
        for i =2:n-1
          h_vector[i] = vector[i-1] + 2.0 * vector[i] + vector[i+1]
        end
        h_vector[n] = vector[n-1] + 2.0 * vector[n]
        for i = 1:n
          vector[i] = h_vector[i]
        end
      elseif status[] == 4  # restart
        for i = 1:n
          r[i] = 1.0
        end
      else
        @printf(" the value %1i of status should not occur\n", status)
      end
    end

    glrt_information( data, inform, status )
    @printf("MR = %1i%1i glrt_solve_problem exit status = %i, f = %.2f\n", unit_m, new_weight, inform.status, inform.obj_regularized )
  end
end

# Delete internal workspace
glrt_terminate( data, control, inform )
