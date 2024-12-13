# test_fit.jl
# Simple code to test the Julia interface to FIT

using GALAHAD
using Test
using Quadmath

function test_fit(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{fit_control_type}()
  inform = Ref{fit_inform_type}()

  status = Ref{Cint}()
  fit_initialize(T, data, control, status)
  fit_information(T, data, inform, status)
  fit_terminate(T, data, control, inform)

  return 0
end

@testset "FIT" begin
  @test test_fit(Float32) == 0
  @test test_fit(Float64) == 0
  @test test_fit(Float128) == 0
end
