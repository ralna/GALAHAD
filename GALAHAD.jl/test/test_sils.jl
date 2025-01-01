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

@testset "SILS" begin
  @test test_sils(Float32, Int32) == 0
  @test test_sils(Float64, Int32) == 0
  @test test_sils(Float128, Int32) == 0
end
