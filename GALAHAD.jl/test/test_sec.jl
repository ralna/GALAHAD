# test_sec.jl
# Simple code to test the Julia interface to SEC

using GALAHAD
using Test
using Quadmath

function test_sec(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sec_control_type{T}}()
  inform = Ref{sec_inform_type}()

  status = Ref{Cint}()
  sec_initialize(T, control, status)
  sec_information(T, data, inform, status)
  sec_terminate(T, data, control, inform)

  return 0
end

@testset "SEC" begin
  @test test_sec(Float32) == 0
  @test test_sec(Float64) == 0
  @test test_sec(Float128) == 0
end
