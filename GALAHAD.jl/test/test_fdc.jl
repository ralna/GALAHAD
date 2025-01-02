# test_fdc.jl
# Simple code to test the Julia interface to FDC

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_fdc(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{fdc_control_type{T,INT}}()
  inform = Ref{fdc_inform_type{T,INT}}()

  # Set problem data
  m = INT(3)  # number of rows
  n = INT(4)  # number of columns
  A_ne = INT(10)  # number of nonzeros
  A_col = INT[1, 2, 3, 4, 1, 2, 3, 4, 2, 4]  # column indices
  A_ptr = INT[1, 5, 9, 11]  # row pointers
  A_val = T[1.0, 2.0, 3.0, 4.0, 2.0, -4.0, 6.0, -8.0, 5.0, 10.0]
  b = T[5.0, 10.0, 0.0]

  # Set output storage
  depen = zeros(INT, m) # dependencies, if any
  n_depen = Ref{INT}()
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n")

  # Initialize FDC
  fdc_initialize(T, INT, data, control, status)

  # Set user-defined control options
  @reset control[].f_indexing = true # Fortran sparse matrix indexing

  # Start from 0
  fdc_find_dependent_rows(T, INT, control, data, inform, status, m, n, A_ne, A_col, A_ptr, A_val, b,
                          n_depen, depen)

  if status[] == 0
    if n_depen == 0
      @printf("FDC_find_dependent - no dependent rows, status = %i\n", status[])
    else
      @printf("FDC_find_dependent - dependent rows(s):")
      for i in 1:n_depen[]
        @printf(" %i", depen[i])
      end
      @printf(", status = %i\n", status[])
    end
  else
    @printf("FDC_find_dependent - exit status = %1i\n", status[])
  end

  # Delete internal workspace
  fdc_terminate(T, INT, data, control, inform)
  return 0
end

for (T, INT, libgalahad) in ((Float32 , Int32, GALAHAD.libgalahad_single      ),
                             (Float32 , Int64, GALAHAD.libgalahad_single_64   ),
                             (Float64 , Int32, GALAHAD.libgalahad_double      ),
                             (Float64 , Int64, GALAHAD.libgalahad_double_64   ),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple   ),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "FDC -- $T -- $INT" begin
      @test test_fdc(T, INT) == 0
    end
  end
end
