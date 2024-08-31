# test_convert.jl
# Simple code to test the Julia interface to CONVERT

using GALAHAD
using Test

function test_convert(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{convert_control_type}()
  inform = Ref{convert_inform_type{Float64}}()

  status = Ref{Cint}()
  convert_initialize(Float64, data, control, status)
  convert_information(Float64, data, inform, status)
  convert_terminate(Float64, data, control, inform)

  return 0
end

@testset "CONVERT" begin
  @test test_convert(Float32) == 0
  @test test_convert(Float64) == 0
end
