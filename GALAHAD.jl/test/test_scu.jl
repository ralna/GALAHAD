# test_scu.jl
# Simple code to test the Julia interface to SCU

using GALAHAD
using Test
using Quadmath

function test_scu(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{scu_control_type}()
  inform = Ref{scu_inform_type{INT}}()

  status = Ref{INT}()
  scu_initialize(T, INT, data, control, status)
  scu_information(T, INT, data, inform, status)
  scu_terminate(T, INT, data, control, inform)

  return 0
end

@testset "SCU" begin
  @test test_scu(Float32, Int32) == 0
  @test test_scu(Float64, Int32) == 0
  @test test_scu(Float128, Int32) == 0
end
