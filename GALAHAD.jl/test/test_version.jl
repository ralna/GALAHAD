# test_version.jl
# Simple code to test the Julia interface to VERSION

using GALAHAD
#using GALAHAD_jll
using Test
using Printf
using Quadmath

function test_version(::Type{T}, ::Type{INT}) where {T,INT}
  version = version_galahad(T, INT)
  # @printf("GALAHAD_VERSION : %s\n", version)
  if GALAHAD.GALAHAD_INSTALLATION == "YGGDRASIL"
    version_jll = pkgversion(GALAHAD_jll)
    @test version.major == version_jll.major
    @test version.minor == version_jll.minor
    @test version.patch == version_jll.patch
  end

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "VERSION -- $T -- $INT" begin
      @test test_version(T, INT) == 0
    end
  end
end
