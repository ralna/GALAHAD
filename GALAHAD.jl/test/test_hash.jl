# test_hash.jl
# Simple code to test the Julia interface to HASH

using GALAHAD
using Test
using Quadmath

function test_hash(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{hash_control_type{INT}}()
  inform = Ref{hash_inform_type{INT}}()

  status = Ref{INT}()
  nchar = INT(10)
  length = INT(100)
  hash_initialize(T, INT, nchar, length, data, control, inform)
  hash_information(T, INT, data, inform, status)
  hash_terminate(T, INT, data, control, inform)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "HASH -- $T -- $INT" begin
      @test test_hash(T, INT) == 0
    end
  end
end
