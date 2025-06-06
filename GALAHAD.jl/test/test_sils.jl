# test_sils.jl
# Simple code to test the Julia interface to SILS

using GALAHAD
using Test
using Quadmath

function test_sils(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sils_control_type{T,INT}}()
  ainfo = Ref{sils_ainfo_type{T,INT}}()
  finfo = Ref{sils_finfo_type{T,INT}}()
  sinfo = Ref{sils_sinfo_type{T,INT}}()

  status = Ref{INT}()
  sils_initialize(T, INT, data, control, status)
  sils_information(T, INT, data, ainfo, finfo, sinfo, status)
  sils_finalize(T, INT, data, control, status)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "SILS -- $T -- $INT" begin
      @test test_sils(T, INT) == 0
    end
  end
end
