# test_sec.jl
# Simple code to test the Julia interface to SEC

using GALAHAD
using Test
using Quadmath

function test_sec(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sec_control_type{T,INT}}()
  inform = Ref{sec_inform_type{INT}}()

  status = Ref{INT}()
  sec_initialize(T, INT, control, status)
  sec_information(T, INT, data, inform, status)
  sec_terminate(T, INT, data, control, inform)

  return 0
end

@testset "SEC" begin
  @test test_sec(Float32, Int32) == 0
  @test test_sec(Float64, Int32) == 0
  @test test_sec(Float128, Int32) == 0
end
