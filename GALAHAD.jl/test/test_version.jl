# test_version.jl
# Simple code to test the Julia interface to VERSION

using GALAHAD
using Test
using Printf
using Quadmath
using Pkg

# The version of GALAHAD_jll is read from the package environment rather than
# by loading GALAHAD_jll: with a custom installation (JULIA_GALAHAD_LIBRARY_PATH)
# GALAHAD does not use GALAHAD_jll, and loading it would open its prebuilt
# libraries, which may not load on the system (e.g. if they require an
# executable stack)

const GALAHAD_JLL_UUID = Base.UUID("1621ace8-c1f0-5e32-a05d-ca786afa56c2")

function test_version(::Type{T}, ::Type{INT}) where {T,INT}
  version = version_galahad(T, INT)
  # @printf("GALAHAD_VERSION : %s\n", version)
  version_jll = Pkg.dependencies()[GALAHAD_JLL_UUID].version
  # compare (major, minor, patch) only, ignoring any prerelease/build parts
  v = VersionNumber(version.major, version.minor, version.patch)
  v_jll = VersionNumber(version_jll.major, version_jll.minor, version_jll.patch)
  if GALAHAD.GALAHAD_INSTALLATION == "YGGDRASIL"
    @test v == v_jll
  else
    @test v >= v_jll
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
