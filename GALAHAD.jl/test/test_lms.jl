# test_lms.jl
# Simple code to test the Julia interface to LMS

using GALAHAD
using Test

function test_lms(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lms_control_type}()
  inform = Ref{lms_inform_type{Float64}}()

  status = Ref{Cint}()
  lms_initialize(Float64, data, control, status)
  lms_information(Float64, data, inform, status)
  lms_terminate(Float64, data, control, inform)

  return 0
end

@testset "LMS" begin
  @test test_lms(Float32) == 0
  @test test_lms(Float64) == 0
end
