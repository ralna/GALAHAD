# test_fit.jl
# Simple code to test the Julia interface to FIT

using GALAHAD
using Test

function test_fit()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{fit_control_type}()
  inform = Ref{fit_inform_type}()

  status = Ref{Cint}()
  fit_initialize(Float64, data, control, status)
  fit_information(Float64, data, inform, status)
  fit_terminate(Float64, data, control, inform)

  return 0
end

@testset "FIT" begin
  @test test_fit() == 0
end
