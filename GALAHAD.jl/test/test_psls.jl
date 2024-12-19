# test_psls.jl
# Simple code to test the Julia interface to PSLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_psls(::Type{T}) where T
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{psls_control_type{T}}()
  inform = Ref{psls_inform_type{T}}()

  # Set problem data
  n = 5 # dimension of A
  ne = 7 # number of elements of A
  dense_ne = div(n * (n + 1), 2) # number of elements of dense A

  row = Cint[1, 2, 2, 3, 3, 4, 5]  # A indices  values, NB lower triangle
  col = Cint[1, 1, 5, 2, 3, 3, 5]
  ptr = Cint[1, 2, 4, 6, 7, 8]
  val = T[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]
  dense = T[2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0,
                  0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0]
  st = ' '
  status = Ref{Cint}()
  status_apply = Ref{Cint}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  for d in 1:3
    # Initialize PSLS
    psls_initialize(T, data, control, status)
    @reset control[].preconditioner = Cint(2) # band preconditioner
    @reset control[].semi_bandwidth = Cint(1) # semibandwidth
    @reset control[].definite_linear_solver = galahad_linear_solver("sils")

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      st = 'C'

      psls_import(T, control, data, status, n,
                  "coordinate", ne, row, col, C_NULL)

      psls_form_preconditioner(T, data, status, ne, val)
    end

    # sparse by rows
    if d == 2
      st = 'R'

      psls_import(T, control, data, status, n,
                  "sparse_by_rows", ne, C_NULL, col, ptr)

      psls_form_preconditioner(T, data, status, ne, val)
    end

    # dense
    if d == 3
      st = 'D'

      psls_import(T, control, data, status, n,
                  "dense", ne, C_NULL, C_NULL, C_NULL)

      psls_form_preconditioner(T, data, status, dense_ne, dense)
    end

    # Set right-hand side b in x
    x = T[8.0, 45.0, 31.0, 15.0, 17.0]  # values

    if status == 0
      psls_information(T, data, inform, status)
      psls_apply_preconditioner(T, data, status_apply, n, x)
    else
      status_apply[] = -1
    end

    @printf("%c storage: status from form  factorize = %i apply = %i\n",
            st, status[], status_apply[])

    # @printf("x: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end

    # Delete internal workspace
    psls_terminate(T, data, control, inform)
  end

  return 0
end

@testset "PSLS" begin
  @test test_psls(Float32) == 0
  @test test_psls(Float64) == 0
  @test test_psls(Float128) == 0
end
