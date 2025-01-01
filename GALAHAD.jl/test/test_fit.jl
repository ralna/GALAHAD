# test_fit.jl
# Simple code to test the Julia interface to FIT

using GALAHAD
using Test
using Quadmath

function test_fit(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{fit_control_type{INT}}()
  inform = Ref{fit_inform_type{INT}}()

  status = Ref{INT}()
  fit_initialize(T, INT, data, control, status)
  fit_information(T, INT, data, inform, status)
  fit_terminate(T, INT, data, control, inform)

  return 0
end

@testset "FIT" begin
  @test test_fit(Float32, Int32) == 0
  @test test_fit(Float64, Int32) == 0
  @test test_fit(Float128, Int32) == 0
end
