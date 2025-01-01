# test_hash.jl
# Simple code to test the Julia interface to HASH

using GALAHAD
using Test
using Quadmath

function test_hash(::Type{T}, ::Type{INT}) where {T,INT}
  data = Ref{Ptr{Cvoid}}()
  control = Ref{hash_control_type{INT}}()
  inform = Ref{hash_inform_type{INT}}()

  status = Ref{INT}()
  nchar = INT(10)
  length = INT(100)
  hash_initialize(T, INT, nchar, length, data, control, inform)
  hash_information(T, INT, data, inform, status)
  hash_terminate(T, INT, data, control, inform)

  return 0
end

@testset "HASH" begin
  @test test_hash(Float32, Int32) == 0
  @test test_hash(Float64, Int32) == 0
  @test test_hash(Float128, Int32) == 0
end
