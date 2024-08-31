# test_hash.jl
# Simple code to test the Julia interface to HASH

using GALAHAD
using Test

function test_hash(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{hash_control_type}()
  inform = Ref{hash_inform_type}()

  status = Ref{Cint}()
  nchar = Cint(10)
  length = Cint(100)
  hash_initialize(Float64, nchar, length, data, control, inform)
  hash_information(Float64, data, inform, status)
  hash_terminate(Float64, data, control, inform)

  return 0
end

@testset "HASH" begin
  @test test_hash(Float32) == 0
  @test test_hash(Float64) == 0
end
