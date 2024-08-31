# test_ir.jl
# Simple code to test the Julia interface to IR

using GALAHAD
using Test

function test_ir(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ir_control_type{Float64}}()
  inform = Ref{ir_inform_type{Float64}}()

  status = Ref{Cint}()
  ir_initialize(Float64, data, control, status)
  ir_information(Float64, data, inform, status)
  ir_terminate(Float64, data, control, inform)

  return 0
end

@testset "IR" begin
  @test test_ir(Float32) == 0
  @test test_ir(Float64) == 0
end
