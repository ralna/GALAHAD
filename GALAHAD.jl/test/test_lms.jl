# test_lms.jl
# Simple code to test the Julia interface to LMS

using GALAHAD
using Test
using Quadmath

function test_lms(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lms_control_type}()
  inform = Ref{lms_inform_type{T}}()

  status = Ref{Cint}()
  lms_initialize(T, data, control, status)
  lms_information(T, data, inform, status)
  lms_terminate(T, data, control, inform)

  return 0
end

@testset "LMS" begin
  @test test_lms(Float32) == 0
  @test test_lms(Float64) == 0
  @test test_lms(Float128) == 0
end
