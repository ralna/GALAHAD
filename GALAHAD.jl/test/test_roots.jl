# test_roots.jl
# Simple code to test the Julia interface to ROOTS

using GALAHAD
using Test

function test_roots()
  data = Ref{Ptr{Cvoid}}()
  control = Ref{roots_control_type{Float64}}()
  inform = Ref{roots_inform_type}()

  status = Ref{Cint}()
  roots_initialize(data, control, status)
  roots_information(data, inform, status)
  roots_terminate(data, control, inform)

  return 0
end

@testset "ROOTS" begin
  @test test_roots() == 0
end
