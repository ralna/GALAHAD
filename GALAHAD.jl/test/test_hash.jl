# test_hash.jl
# Simple code to test the Julia interface to HASH

using GALAHAD
using Test
using Quadmath

function test_hash(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{hash_control_type}()
  inform = Ref{hash_inform_type}()

  status = Ref{Cint}()
  nchar = Cint(10)
  length = Cint(100)
  hash_initialize(T, nchar, length, data, control, inform)
  hash_information(T, data, inform, status)
  hash_terminate(T, data, control, inform)

  return 0
end

@testset "HASH" begin
  @test test_hash(Float32) == 0
  @test test_hash(Float64) == 0
  @test test_hash(Float128) == 0
end
