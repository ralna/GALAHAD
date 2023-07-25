# test_lhs.jl
# Simple code to test the Julia interface to LHS

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = lhs_control_type()
inform = lhs_inform_type()

# Initialize LHS
lhs_initialize(data, control, inform)

# Parameters
n_dimen = 7  # dimension
n_points = 2  # points required
X = [zeros(Cint, n_points) for i=1:n_dimen]  # points
seed = Ref{Cint}()

# Set a random seed
lhs_get_seed(seed) 

# Generate points
lhs_ihs(n_dimen, n_points, seed, X, control, inform, data)
if inform.status == 0  # successful return
  @printf("LHS successful\n")
  for j = 1:n_points
    @printf("Point %d = ", j)
    for i = 1:n_dimen
      @printf("%d ", X[i][j])
    end
  end
  @printf("\n")
else # error returns
    @printf("LHS exit status = %d \n", inform.status)
end

# Delete internal workspace
lhs_terminate(data, control, inform)
