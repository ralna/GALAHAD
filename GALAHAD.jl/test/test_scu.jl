# test_scu.jl
# Simple code to test the Julia interface to SCU

using GALAHAD
using Test
using Quadmath

function test_scu(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{scu_control_type}()
  inform = Ref{scu_inform_type}()

  status = Ref{Cint}()
  scu_initialize(T, data, control, status)
  scu_information(T, data, inform, status)
  scu_terminate(T, data, control, inform)

  return 0
end

@testset "SCU" begin
  @test test_scu(Float32) == 0
  @test test_scu(Float64) == 0
  @test test_scu(Float128) == 0
end
