# test_lhs.jl
# Simple code to test the Julia interface to LHS

using GALAHAD
using Test
using Printf
using Accessors

function test_lhs(::Type{T}) where T
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lhs_control_type}()
  inform = Ref{lhs_inform_type}()

  # Initialize LHS
  lhs_initialize(Float64, data, control, inform)

  # Parameters
  n_dimen = Cint(7) # dimension
  n_points = Cint(2) # points required
  X = [zeros(Cint, n_points) for i in 1:n_dimen]  # points
  seed = Ref{Cint}()

  # Set a random seed
  lhs_get_seed(Float64, seed)

  # Generate points
  lhs_ihs(Float64, n_dimen, n_points, seed, X, control, inform, data)

  if inform[].status == 0 # successful return
    @printf("LHS successful\n")
    for j in 1:n_points
      @printf("Point %d = ", j)
      for i in 1:n_dimen
        @printf("%d ", X[i][j])
        @printf("\n")
      end
    end
  else # error returns
    @printf("LHS exit status = %d \n", inform[].status)
  end

  # Delete internal workspace
  lhs_terminate(Float64, data, control, inform)

  return 0
end

@testset "LHS" begin
  @test test_lhs(Float32) == 0
  @test test_lhs(Float64) == 0
end
