# test_lms.jl
# Simple code to test the Julia interface to LMS

using GALAHAD
using Test

function test_lms()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{lms_control_type}()
  inform = Ref{lms_inform_type{Float64}}()

  status = Ref{Cint}()
  lms_initialize(data, control, status)
  lms_information(data, inform, status)
  lms_terminate(data, control, inform)

  return 0
end

@testset "LMS" begin
  @test test_lms() == 0
end
