# test_nodend.jl
# Simple code to test the Julia interface to NODEND

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_nodend(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{nodend_control_type{INT}}()
  inform = Ref{nodend_inform_type{T,INT}}()

  # Set problem data
  n = 10  # dimension
  A_ne = 2 * n - 1   # Hesssian elements, NB lower triangle
  A_dense_ne = div(n * (n + 1),  2)  # dense Hessian elements
  A_row = zeros(INT, A_ne)  # row indices
  A_col = zeros(INT, A_ne)  # column indices
  A_ptr = zeros(INT, n + 1)  # row pointers
  perm = zeros(INT, n)  # permutation
  status = Ref{INT}()

  # Set output storage
  st = ' '

  l = 0
  A_ptr[1] = l + 1
  A_row[1] = 1
  A_col[1] = 1
  for i in 2:n
    l = l + 1
    A_ptr[i] = l + 1
    A_row[l] = i + 1
    A_col[l] = i
    l = l + 1
    A_row[l] = i + 1
    A_col[l] = i + 1
  end
  A_ptr[n + 1] = l + 2

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of nodend storage formats\n\n")

  for d in 1:3
    # Initialize NODEND
    nodend_initialize(T, INT, data, control, status)

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      nodend_order(T, INT, control, data, status, n, perm,
                   "coordinate", A_ne, A_row, A_col, C_NULL)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      nodend_order(T, INT, control, data, status, n, perm,
                   "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)
    end

    # dense
    if d == 3
      st = 'D'
      nodend_order(T, INT, control, data, status, n, perm,
                   "dense", A_dense_ne, C_NULL, C_NULL, C_NULL)
    end
    nodend_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%c: NODEND_order success, perm: ", st)
      for i in 1:n
        @printf("%i ", perm[i])
      end
      @printf("\n")
    else
      @printf("%c: NODEND_order exit status = %i\n", st, inform[].status)
    end

    # Delete internal workspace
    nodend_terminate(T, INT, data)
  end

  return 0
end

for (T, INT, libgalahad) in ((Float32, Int32, GALAHAD.libgalahad_single),
                             (Float32, Int64, GALAHAD.libgalahad_single_64),
                             (Float64, Int32, GALAHAD.libgalahad_double),
                             (Float64, Int64, GALAHAD.libgalahad_double_64),
                             (Float128, Int32, GALAHAD.libgalahad_quadruple),
                             (Float128, Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "NODEND -- $T -- $INT" begin
      @test test_nodend(T, INT) == 0
    end
  end
end
