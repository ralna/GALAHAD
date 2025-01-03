# test_psls.jl
# Simple code to test the Julia interface to PSLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_psls(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{psls_control_type{T,INT}}()
  inform = Ref{psls_inform_type{T,INT}}()

  # Set problem data
  n = INT(5)  # dimension of A
  ne = INT(7)  # number of elements of A
  dense_ne = div(n * (n + 1), 2) # number of elements of dense A

  row = INT[1, 2, 2, 3, 3, 4, 5]  # A indices  values, NB lower triangle
  col = INT[1, 1, 5, 2, 3, 3, 5]
  ptr = INT[1, 2, 4, 6, 7, 8]
  val = T[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]
  dense = T[2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0,
                  0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0]
  st = ' '
  status = Ref{INT}()
  status_apply = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  for d in 1:3
    # Initialize PSLS
    psls_initialize(T, INT, data, control, status)
    @reset control[].preconditioner = INT(2) # band preconditioner
    @reset control[].semi_bandwidth = INT(1) # semibandwidth
    @reset control[].definite_linear_solver = galahad_linear_solver("sils")

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      st = 'C'

      psls_import(T, INT, control, data, status, n,
                  "coordinate", ne, row, col, C_NULL)

      psls_form_preconditioner(T, INT, data, status, ne, val)
    end

    # sparse by rows
    if d == 2
      st = 'R'

      psls_import(T, INT, control, data, status, n,
                  "sparse_by_rows", ne, C_NULL, col, ptr)

      psls_form_preconditioner(T, INT, data, status, ne, val)
    end

    # dense
    if d == 3
      st = 'D'

      psls_import(T, INT, control, data, status, n,
                  "dense", ne, C_NULL, C_NULL, C_NULL)

      psls_form_preconditioner(T, INT, data, status, dense_ne, dense)
    end

    # Set right-hand side b in x
    x = T[8.0, 45.0, 31.0, 15.0, 17.0]  # values

    if status == 0
      psls_information(T, INT, data, inform, status)
      psls_apply_preconditioner(T, INT, data, status_apply, n, x)
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
    psls_terminate(T, INT, data, control, inform)
  end

  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "PSLS -- $T -- $INT" begin
      @test test_psls(T, INT) == 0
    end
  end
end
