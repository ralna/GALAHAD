# test_hash.jl
# Simple code to test the Julia interface to HASH

using GALAHAD
using Test

function test_hash()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{hash_control_type}()
  inform = Ref{hash_inform_type}()

  status = Ref{Cint}()
  nchar = Cint(10)
  length = Cint(100)
  hash_initialize(nchar, length, data, control, inform)
  hash_information(data, inform, status)
  hash_terminate(data, control, inform)

  return 0
end

@testset "HASH" begin
  @test test_hash() == 0
end
