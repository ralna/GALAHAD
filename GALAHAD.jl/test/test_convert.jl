# test_convert.jl
# Simple code to test the Julia interface to CONVERT

using GALAHAD
using Test
using Quadmath

function test_convert(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{convert_control_type{INT}}()
  inform = Ref{convert_inform_type{T,INT}}()

  status = Ref{INT}()
  convert_initialize(T, INT, data, control, status)
  convert_information(T, INT, data, inform, status)
  convert_terminate(T, INT, data, control, inform)

  return 0
end

@testset "CONVERT" begin
  @test test_convert(Float32, Int32) == 0
  @test test_convert(Float64, Int32) == 0
  @test test_convert(Float128, Int32) == 0
end
