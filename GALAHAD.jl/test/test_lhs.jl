# test_lhs.jl
# Simple code to test the Julia interface to LHS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_lhs(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lhs_control_type{INT}}()
  inform = Ref{lhs_inform_type{INT}}()

  # Initialize LHS
  lhs_initialize(T, INT, data, control, inform)

  # Parameters
  n_dimen = INT(7)  # dimension
  n_points = INT(2)  # points required
  X = zeros(INT, n_dimen, n_points)  # points
  seed = Ref{INT}()

  # Set a random seed
  lhs_get_seed(T, INT, seed)

  # Generate points
  lhs_ihs(T, INT, n_dimen, n_points, seed, X, control, inform, data)

  if inform[].status == 0 # successful return
    @printf("LHS successful\n")
    for j in 1:n_points
      @printf("Point %d = [", j)
      for i in 1:n_dimen
        @printf("%d ", X[i,j])
      end
      @printf("]\n")
    end
  else # error returns
    @printf("LHS exit status = %d \n", inform[].status)
  end

  # Delete internal workspace
  lhs_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "LHS -- $T -- $INT" begin
      @test test_lhs(T, INT) == 0
    end
  end
end
