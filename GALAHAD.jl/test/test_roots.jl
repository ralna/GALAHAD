# test_roots.jl
# Simple code to test the Julia interface to ROOTS

using GALAHAD
using Test
using Quadmath

function test_roots(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{roots_control_type{T,INT}}()
  inform = Ref{roots_inform_type{INT}}()

  status = Ref{INT}()
  roots_initialize(T, INT, data, control, status)
  roots_information(T, INT, data, inform, status)
  roots_terminate(T, INT, data, control, inform)

  return 0
end

@testset "ROOTS" begin
  @test test_roots(Float32, Int32) == 0
  @test test_roots(Float64, Int32) == 0
  @test test_roots(Float128, Int32) == 0
end
