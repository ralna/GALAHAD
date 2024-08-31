# test_roots.jl
# Simple code to test the Julia interface to ROOTS

using GALAHAD
using Test

function test_roots(::Type{T}) where T
  data = Ref{Ptr{Cvoid}}()
  control = Ref{roots_control_type{Float64}}()
  inform = Ref{roots_inform_type}()

  status = Ref{Cint}()
  roots_initialize(Float64, data, control, status)
  roots_information(Float64, data, inform, status)
  roots_terminate(Float64, data, control, inform)

  return 0
end

@testset "ROOTS" begin
  @test test_roots(Float32) == 0
  @test test_roots(Float64) == 0
end
