# test_ir.jl
# Simple code to test the Julia interface to IR

using GALAHAD
using Test

function test_ir()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ir_control_type{Float64}}()
  inform = Ref{ir_inform_type{Float64}}()

  status = Ref{Cint}()
  ir_initialize(data, control, status)
  ir_information(data, inform, status)
  ir_terminate(data, control, inform)

  return 0
end

@testset "IR" begin
  @test test_ir() == 0
end
