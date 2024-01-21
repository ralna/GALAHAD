# test_bsc.jl
# Simple code to test the Julia interface to BSC

using GALAHAD
using Test

function test_bsc()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bsc_control_type}()
  inform = Ref{bsc_inform_type{Float64}}()

  status = Ref{Cint}()
  bsc_initialize(data, control, status)
  bsc_information(data, inform, status)
  bsc_terminate(data, control, inform)

  return 0
end

@testset "BSC" begin
  @test test_bsc() == 0
end
