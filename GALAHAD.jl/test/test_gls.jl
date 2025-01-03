# test_gls.jl
# Simple code to test the Julia interface to GLS

using GALAHAD
using Test
using Quadmath

function test_gls(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{gls_control_type{T,INT}}()
  ainfo = Ref{gls_ainfo_type{T,INT}}()
  finfo = Ref{gls_finfo_type{T,INT}}()
  sinfo = Ref{gls_sinfo_type{INT}}()

  status = Ref{INT}()
  gls_initialize(T, INT, data, control)
  gls_information(T, INT, data, ainfo, finfo, sinfo, status)
  gls_finalize(T, INT, data, control, status)

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "GLS -- $T -- $INT" begin
      @test test_gls(T, INT) == 0
    end
  end
end
