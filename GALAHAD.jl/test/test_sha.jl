# test_sha.jl
# Simple code to test the Julia interface to SHA

using GALAHAD
using Test

function test_sha()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sha_control_type}()
  inform = Ref{sha_inform_type}()

  status = Ref{Cint}()
  sha_initialize(data, control, status)
  sha_information(data, inform, status)
  sha_terminate(data, control, inform)

  return 0
end

@testset "SHA" begin
  @test test_sha() == 0
end
