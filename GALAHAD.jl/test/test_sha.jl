# test_sha.jl
# Simple code to test the Julia interface to SHA

using GALAHAD
using Test
using Quadmath

function test_sha(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sha_control_type{INT}}()
  inform = Ref{sha_inform_type{T,INT}}()

  status = Ref{INT}()
  sha_initialize(T, INT, data, control, status)
  sha_information(T, INT, data, inform, status)
  sha_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "SHA -- $T -- $INT" begin
      @test test_sha(T, INT) == 0
    end
  end
end
