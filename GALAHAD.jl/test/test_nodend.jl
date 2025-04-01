# test_nodend.jl
# Simple code to test the Julia interface to NODEND

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_nodend(::Type{INT}) where {INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{nodend_control_type{INT}}()
  inform = Ref{nodend_inform_type{INT}}()

  # Set problem data
  n = INT(10)  # dimension
  A_ne = 2 * n - 1 # Hesssian elements, NB lower triangle
  A_dense_ne = div(n * (n + 1), 2) # dense Hessian elements
  A_row = zeros(INT, A_ne) # row indices,
  A_col = zeros(INT, A_ne) # column indices
  A_ptr = zeros(INT, n + 1)  # row pointers

  # Set output storage
  perm = zeros(INT, n) # permutation vector
  st = ' '
  status = Ref{INT}()

  # A = tridiag(2,1)
  l = 1
  A_ptr[1] = l
  A_row[l] = 1
  A_col[l] = 1
  for i in 2:n
    l = l + 1
    A_ptr[i] = l
    A_row[l] = i
    A_col[l] = i - 1
    l = l + 1
    A_row[l] = i
    A_col[l] = i
  end
  A_ptr[n + 1] = l + 1

  @printf(" fortran sparse matrix indexing\n\n")
  @printf(" basic tests of nodend storage formats\n\n")

  for d in 1:3

    # Initialize NODEND
    nodend_initialize(INT, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      nodend_order(INT, control, data, status, n, perm,
                   "coordinate", A_ne, A_row, A_col, C_NULL)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      nodend_order(INT, control, data, status, n, perm,
                   "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)
    end

    # dense
    if d == 3
      st = 'D'
      nodend_order(INT, control, data, status, n, perm,
                   "dense", A_dense_ne, C_NULL, C_NULL, C_NULL)
    end

    nodend_information(INT, data, inform, status)

    if inform[].status == 0
      @printf("%c: NODEND_order success, perm: ", st);
      for i in 1:n
        @printf("%1i ", perm[i]);
      end
      @printf("\n");
    else
      @printf("%c: NODEND_solve exit status = %1i\n", st, inform[].status)
    end

    # Delete internal workspace
    nodend_terminate(INT, data, control, inform)
  end
end

for (INT, libgalahad) in ((Int32, GALAHAD.libgalahad_single      ),
                          (Int64, GALAHAD.libgalahad_single_64   ),
                          (Int32, GALAHAD.libgalahad_double      ),
                          (Int64, GALAHAD.libgalahad_double_64   ),
                          (Int32, GALAHAD.libgalahad_quadruple   ),
                          (Int64, GALAHAD.libgalahad_quadruple_64))
  if isfile(libgalahad)
    @testset "NODEND -- $INT" begin
      @test test_nodend(INT) == 0
    end
  end
end
