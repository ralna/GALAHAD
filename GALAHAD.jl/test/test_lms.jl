# test_lms.jl
# Simple code to test the Julia interface to LMS

using GALAHAD
using Test
using Quadmath

function test_lms(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lms_control_type{INT}}()
  inform = Ref{lms_inform_type{T,INT}}()

  status = Ref{INT}()
  lms_initialize(T, INT, data, control, status)
  lms_information(T, INT, data, inform, status)
  lms_terminate(T, INT, data, control, inform)

  return 0
end

@testset "LMS" begin
  @test test_lms(Float32, Int32) == 0
  @test test_lms(Float64, Int32) == 0
  @test test_lms(Float128, Int32) == 0
end
