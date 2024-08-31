# test_sils.jl
# Simple code to test the Julia interface to SILS

using GALAHAD
using Test

function test_sils(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sils_control_type{Float64}}()
  ainfo = Ref{sils_ainfo_type{Float64}}()
  finfo = Ref{sils_finfo_type{Float64}}()
  sinfo = Ref{sils_sinfo_type{Float64}}()

  status = Ref{Cint}()
  sils_initialize(Float64, data, control, status)
  sils_information(Float64, data, ainfo, finfo, sinfo, status)
  sils_finalize(Float64, data, control, status)

  return 0
end

@testset "SILS" begin
  @test test_sils(Float32) == 0
  @test test_sils(Float64) == 0
end
