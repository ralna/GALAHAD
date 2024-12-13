# test_roots.jl
# Simple code to test the Julia interface to ROOTS

using GALAHAD
using Test
using Quadmath

function test_roots(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{roots_control_type{T}}()
  inform = Ref{roots_inform_type}()

  status = Ref{Cint}()
  roots_initialize(T, data, control, status)
  roots_information(T, data, inform, status)
  roots_terminate(T, data, control, inform)

  return 0
end

@testset "ROOTS" begin
  @test test_roots(Float32) == 0
  @test test_roots(Float64) == 0
  @test test_roots(Float128) == 0
end
