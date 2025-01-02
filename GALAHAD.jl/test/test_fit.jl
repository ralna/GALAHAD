# test_fit.jl
# Simple code to test the Julia interface to FIT

using GALAHAD
using Test
using Quadmath

function test_fit(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{fit_control_type{INT}}()
  inform = Ref{fit_inform_type{INT}}()

  status = Ref{INT}()
  fit_initialize(T, INT, data, control, status)
  fit_information(T, INT, data, inform, status)
  fit_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "FIT -- $T -- $INT" begin
      @test test_fit(T, INT) == 0
    end
  end
end
