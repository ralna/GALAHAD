# test_sils.jl
# Simple code to test the Julia interface to SILS

using GALAHAD
using Test

function test_sils(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sils_control_type{T}}()
  ainfo = Ref{sils_ainfo_type{T}}()
  finfo = Ref{sils_finfo_type{T}}()
  sinfo = Ref{sils_sinfo_type{T}}()

  status = Ref{Cint}()
  sils_initialize(T, data, control, status)
  sils_information(T, data, ainfo, finfo, sinfo, status)
  sils_finalize(T, data, control, status)

  return 0
end

@testset "SILS" begin
  @test test_sils(Float32) == 0
  @test test_sils(Float64) == 0
end
