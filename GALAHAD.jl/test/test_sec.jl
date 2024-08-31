# test_sec.jl
# Simple code to test the Julia interface to SEC

using GALAHAD
using Test

function test_sec()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sec_control_type{Float64}}()
  inform = Ref{sec_inform_type}()

  status = Ref{Cint}()
  sec_initialize(Float64, control, status)
  sec_information(Float64, data, inform, status)
  sec_terminate(Float64, data, control, inform)

  return 0
end

@testset "SEC" begin
  @test test_sec() == 0
end
