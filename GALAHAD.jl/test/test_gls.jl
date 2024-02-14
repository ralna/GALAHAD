# test_gls.jl
# Simple code to test the Julia interface to GLS

using GALAHAD
using Test

function test_gls()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{gls_control_type{Float64}}()
  ainfo = Ref{gls_ainfo_type{Float64}}()
  finfo = Ref{gls_finfo_type{Float64}}()
  sinfo = Ref{gls_sinfo_type}()

  status = Ref{Cint}()
  gls_initialize(data, control)
  gls_information(data, ainfo, finfo, sinfo, status)
  gls_finalize(data, control, status)

  return 0
end

@testset "GLS" begin
  @test test_gls() == 0
end
