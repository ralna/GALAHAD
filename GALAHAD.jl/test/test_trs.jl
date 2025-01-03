# test_trs.jl
# Simple code to test the Julia interface to TRS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_trs(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{trs_control_type{T,INT}}()
  inform = Ref{trs_inform_type{T,INT}}()

  # Set problem data
  n = INT(3)  # dimension of H
  m = INT(1)  # dimension of A
  H_ne = INT(4)  # number of elements of H
  M_ne = INT(3)  # number of elements of M
  A_ne = INT(3)  # number of elements of A
  H_dense_ne = INT(6)  # number of elements of H
  M_dense_ne = INT(6)  # number of elements of M
  H_row = INT[1, 2, 3, 3]  # row indices, NB lower triangle
  H_col = INT[1, 2, 3, 1]
  H_ptr = INT[1, 2, 3, 5]
  M_row = INT[1, 2, 3]  # row indices, NB lower triangle
  M_col = INT[1, 2, 3]
  M_ptr = INT[1, 2, 3, 4]
  A_row = INT[1, 1, 1]
  A_col = INT[1, 2, 3]
  A_ptr = INT[1, 4]
  H_val = T[1.0, 2.0, 3.0, 4.0]
  M_val = T[1.0, 2.0, 1.0]
  A_val = T[1.0, 1.0, 1.0]
  H_dense = T[1.0, 0.0, 2.0, 4.0, 0.0, 3.0]
  M_dense = T[1.0, 0.0, 2.0, 0.0, 0.0, 1.0]
  H_diag = T[1.0, 0.0, 2.0]
  M_diag = T[1.0, 2.0, 1.0]
  f = T(0.96)
  radius = one(T)
  c = T[0.0, 2.0, 0.0]

  st = ' '
  status = Ref{INT}()
  x = zeros(T, n)
  ma = ""

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  for a_is in 0:1 # add a linear constraint?
    for m_is in 0:1 # include a scaling matrix?
      if (a_is == 1) && (m_is == 1)
        ma = "MA"
      elseif a_is == 1
        ma = "A "
      elseif m_is == 1
        ma = "M "
      else
        ma = "  "
      end

      for storage_type in 1:4
        # Initialize TRS
        trs_initialize(T, INT, data, control, status)

        # Set user-defined control options
        @reset control[].f_indexing = true # fortran sparse matrix indexing

        # sparse co-ordinate storage
        if storage_type == 1
          st = 'C'
          # import the control parameters and structural data
          trs_import(T, INT, control, data, status, n,
                     "coordinate", H_ne, H_row, H_col, C_NULL)

          if m_is == 1
            trs_import_m(T, INT, data, status, n,
                         "coordinate", M_ne, M_row, M_col, C_NULL)
          end

          if a_is == 1
            trs_import_a(T, INT, data, status, m,
                         "coordinate", A_ne, A_row, A_col, C_NULL)
          end

          # solve the problem
          if (a_is == 1) && (m_is == 1)
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              M_ne, M_val, m, A_ne, A_val, C_NULL)
          elseif a_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              0, C_NULL, m, A_ne, A_val, C_NULL)
          elseif m_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              M_ne, M_val, 0, 0, C_NULL, C_NULL)
          else
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        # sparse by rows
        if storage_type == 2
          st = 'R'
          # import the control parameters and structural data
          trs_import(T, INT, control, data, status, n,
                     "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr)

          if m_is == 1
            trs_import_m(T, INT, data, status, n,
                         "sparse_by_rows", M_ne, C_NULL, M_col, M_ptr)
          end

          if a_is == 1
            trs_import_a(T, INT, data, status, m,
                         "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)
          end

          # solve the problem
          if (a_is == 1) && (m_is == 1)
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              M_ne, M_val, m, A_ne, A_val, C_NULL)
          elseif a_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              0, C_NULL, m, A_ne, A_val, C_NULL)
          elseif m_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              M_ne, M_val, 0, 0, C_NULL, C_NULL)
          else
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        # dense
        if storage_type == 3
          st = 'D'
          # import the control parameters and structural data
          trs_import(T, INT, control, data, status, n,
                     "dense", H_ne, C_NULL, C_NULL, C_NULL)

          if m_is == 1
            trs_import_m(T, INT, data, status, n,
                         "dense", M_ne, C_NULL, C_NULL, C_NULL)
          end

          if a_is == 1
            trs_import_a(T, INT, data, status, m,
                         "dense", A_ne, C_NULL, C_NULL, C_NULL)
          end

          # solve the problem
          if (a_is == 1) && (m_is == 1)
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_dense_ne, H_dense, x,
                              M_dense_ne, M_dense, m, A_ne, A_val,
                              C_NULL)
          elseif a_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_dense_ne, H_dense, x,
                              0, C_NULL, m, A_ne, A_val, C_NULL)
          elseif m_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_dense_ne, H_dense, x,
                              M_dense_ne, M_dense, 0, 0, C_NULL, C_NULL)
          else
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_dense_ne, H_dense, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        # diagonal
        if storage_type == 4
          st = 'L'
          # import the control parameters and structural data
          trs_import(T, INT, control, data, status, n,
                     "diagonal", H_ne, C_NULL, C_NULL, C_NULL)

          if m_is == 1
            trs_import_m(T, INT, data, status, n,
                         "diagonal", M_ne, C_NULL, C_NULL, C_NULL)
          end

          if a_is == 1
            trs_import_a(T, INT, data, status, m,
                         "dense", A_ne, C_NULL, C_NULL, C_NULL)
          end

          # solve the problem
          if (a_is == 1) && (m_is == 1)
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, n, H_diag, x,
                              n, M_diag, m, A_ne, A_val, C_NULL)
          elseif a_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, n, H_diag, x,
                              0, C_NULL, m, A_ne, A_val, C_NULL)
          elseif m_is == 1
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, n, H_diag, x,
                              n, M_diag, 0, 0, C_NULL, C_NULL)
          else
            trs_solve_problem(T, INT, data, status, n,
                              radius, f, c, n, H_diag, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        trs_information(T, INT, data, inform, status)

        @printf("format %c%s: TRS_solve_problem exit status = %1i, f = %.2f\n", st, ma,
                inform[].status, inform[].obj)

        # @printf("x: ")
        # for i = 1:n+m
        #   @printf("%f ", x[i])
        # end

        # Delete internal workspace
        trs_terminate(T, INT, data, control, inform)
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
    @testset "TRS -- $T -- $INT" begin
      @test test_trs(T, INT) == 0
    end
  end
end
