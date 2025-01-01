# test_sha.jl
# Simple code to test the Julia interface to SHA

using GALAHAD
using Test
using Quadmath

function test_sha(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sha_control_type{INT}}()
  inform = Ref{sha_inform_type{T,INT}}()

  status = Ref{INT}()
  sha_initialize(T, INT, data, control, status)
  sha_information(T, INT, data, inform, status)
  sha_terminate(T, INT, data, control, inform)

  return 0
end

@testset "SHA" begin
  @test test_sha(Float32, Int32) == 0
  @test test_sha(Float64, Int32) == 0
  @test test_sha(Float128, Int32) == 0
end
