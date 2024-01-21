# test_fit.jl
# Simple code to test the Julia interface to FIT

using GALAHAD
using Test

function test_fit()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{fit_control_type}()
  inform = Ref{fit_inform_type}()

  status = Ref{Cint}()
  fit_initialize(data, control, status)
  fit_information(data, inform, status)
  fit_terminate(data, control, inform)

  return 0
end

@testset "FIT" begin
  @test test_fit() == 0
end
