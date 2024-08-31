# test_scu.jl
# Simple code to test the Julia interface to SCU

using GALAHAD
using Test

function test_scu()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{scu_control_type}()
  inform = Ref{scu_inform_type}()

  status = Ref{Cint}()
  scu_initialize(Float64, data, control, status)
  scu_information(Float64, data, inform, status)
  scu_terminate(Float64, data, control, inform)

  return 0
end

@testset "SCU" begin
  @test test_scu() == 0
end
