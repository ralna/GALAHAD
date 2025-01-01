# test_ir.jl
# Simple code to test the Julia interface to IR

using GALAHAD
using Test
using Quadmath

function test_ir(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{ir_control_type{T,INT}}()
  inform = Ref{ir_inform_type{T,INT}}()

  status = Ref{INT}()
  ir_initialize(T, INT, data, control, status)
  ir_information(T, INT, data, inform, status)
  ir_terminate(T, INT, data, control, inform)

  return 0
end

@testset "IR" begin
  @test test_ir(Float32, Int32) == 0
  @test test_ir(Float64, Int32) == 0
  @test test_ir(Float128, Int32) == 0
end
