# test_roots.jl
# Simple code to test the Julia interface to ROOTS

using GALAHAD
using Test
using Quadmath

function test_roots(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{roots_control_type{T,INT}}()
  inform = Ref{roots_inform_type{INT}}()

  status = Ref{INT}()
  roots_initialize(T, INT, data, control, status)
  roots_information(T, INT, data, inform, status)
  roots_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "ROOTS -- $T -- $INT" begin
      @test test_roots(T, INT) == 0
    end
  end
end
