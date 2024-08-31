# test_sha.jl
# Simple code to test the Julia interface to SHA

using GALAHAD
using Test

function test_sha()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sha_control_type}()
  inform = Ref{sha_inform_type}()

  status = Ref{Cint}()
  sha_initialize(Float64, data, control, status)
  sha_information(Float64, data, inform, status)
  sha_terminate(Float64, data, control, inform)

  return 0
end

@testset "SHA" begin
  @test test_sha() == 0
end
