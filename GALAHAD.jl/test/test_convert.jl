# test_convert.jl
# Simple code to test the Julia interface to CONVERT

using GALAHAD
using Test

function test_convert()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{convert_control_type}()
  inform = Ref{convert_inform_type{Float64}}()

  status = Ref{Cint}()
  convert_initialize(data, control, status)
  convert_information(data, inform, status)
  convert_terminate(data, control, inform)

  return 0
end

@testset "CONVERT" begin
  @test test_convert() == 0
end
