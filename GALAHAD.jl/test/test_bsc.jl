# test_bsc.jl
# Simple code to test the Julia interface to BSC

using GALAHAD
using Test

function test_bsc(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bsc_control_type}()
  inform = Ref{bsc_inform_type{Float64}}()

  status = Ref{Cint}()
  bsc_initialize(Float64, data, control, status)
  bsc_information(Float64, data, inform, status)
  bsc_terminate(Float64, data, control, inform)

  return 0
end

@testset "BSC" begin
  @test test_bsc(Float32) == 0
  @test test_bsc(Float64) == 0
end
