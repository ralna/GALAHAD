# test_gls.jl
# Simple code to test the Julia interface to GLS

using GALAHAD
using Test

function test_gls(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{gls_control_type{T}}()
  ainfo = Ref{gls_ainfo_type{T}}()
  finfo = Ref{gls_finfo_type{T}}()
  sinfo = Ref{gls_sinfo_type}()

  status = Ref{Cint}()
  gls_initialize(T, data, control)
  gls_information(T, data, ainfo, finfo, sinfo, status)
  gls_finalize(T, data, control, status)

  return 0
end

@testset "GLS" begin
  @test test_gls(Float32) == 0
  @test test_gls(Float64) == 0
end
