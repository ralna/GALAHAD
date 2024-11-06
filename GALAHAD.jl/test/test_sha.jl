# test_sha.jl
# Simple code to test the Julia interface to SHA

using GALAHAD
using Test

function test_sha(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sha_control_type}()
  inform = Ref{sha_inform_type{T}}()

  status = Ref{Cint}()
  sha_initialize(T, data, control, status)
  sha_information(T, data, inform, status)
  sha_terminate(T, data, control, inform)

  return 0
end

@testset "SHA" begin
  @test test_sha(Float32) == 0
  @test test_sha(Float64) == 0
end
