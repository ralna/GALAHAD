# test_bsc.jl
# Simple code to test the Julia interface to BSC

using GALAHAD
using Test
using Quadmath

function test_bsc(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{bsc_control_type{INT}}()
  inform = Ref{bsc_inform_type{T,INT}}()

  # Set problem data
  m = INT(3)  # row dimension of A
  n = INT(4)  # column dimension of A
  A_ne = INT(6)  # number of entries in A
  A_row = INT[1, 1, 2, 2, 3, 3]  # row indices
  A_col = INT[1, 2, 3, 4, 1, 4]  # column indices
  A_ptr = INT[1, 3, 5, 7]  # row pointers
  A_val = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # values
  A_dense_ne = INT(12) # number of elements in A as a dense matrix
  A_dense = T[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
  D = T[1.0, 2.0, 3.0, 4.0] # optional diagonal

  # Set output storage
  st = ' '
  status = Ref{INT}()
  @printf(" basic tests of storage formats\n\n")
  S_ne = Ref{INT}()

  # loop over storage types
  for d in 1:3
    # Initialize BSC
    bsc_initialize(T, INT, data, control, status)

    # sparse co-ordinate storage
    if d == 1
      st = 'C'
      bsc_import(T, INT, control, data, status, n, m,
                 "coordinate", A_ne, A_row, A_col, C_NULL, S_ne)
    end

    # sparse by rows
    if d == 2
      st = 'R'
      bsc_import(T, INT, control, data, status, n, m,
                 "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr, S_ne)
    end

    # dense
    if d == 3
      st = 'D'
      A_dense = T[2.0, 1.0, 0.0, 0.0, 1.0, 1.0]

      bsc_import(T, INT, control, data, status, n, m,
                 "dense", A_ne, C_NULL, C_NULL, C_NULL, S_ne)
    end

    S_row = Vector{INT}(undef, S_ne)
    S_col = Vector{INT}(undef, S_ne)
    S_ptr = Vector{INT}(undef, m+1)
    S_val = Vector{T}(undef, S_ne)

    for ptr in 0:1
      if ptr == 0
        if d == 3
          bsc_form_s(T, INT, data, status, n, m, A_dense_ne, A_dense, 
                     S_ne, S_row, S_col, C_NULL, S_val, C_NULL)
        else
          bsc_form_s(T, INT, data, status, n, m, A_ne, A_val, 
                     S_ne, S_row, S_col, C_NULL, S_val, C_NULL)
        end
      else
        if d == 3
          bsc_form_s(T, INT, data, status, n, m, A_dense_ne, A_dense, 
                     S_ne, S_row, S_col, S_ptr, S_val, D)
        else
          bsc_form_s(T, INT, data, status, n, m, A_ne, A_val, 
                     S_ne, S_row, S_col, S_ptr, S_val, D)
        end
      end

      bsc_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf(" format %c: status = %1i\n", st, inform[].status)
      else
        @printf(" format %c: error status = %1i\n", st, inform[].status)
      end

      @printf("S_row: ")
      for i = 1:S_ne
        @printf("%1i ", S_row[i])
      end
      @printf("\n");
      @printf("S_col: ")
      for i = 1:S_ne
        @printf("%1i ", S_col[i])
      end
      @printf("\n")
      @printf("S_val: ")
      for i = 1:S_ne
        @printf("%.2f ", S_val[i])
      end
      printf("\n")
      if ptr == 1
        @printf("S_ptr: ")
        for i = 1:m+1
          @printf("%1i ", S_ptr[i])
        end
        @printf("\n")
      end
    end
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
    @testset "BSC -- $T -- $INT" begin
      @test test_bsc(T, INT) == 0
    end
  end
end
