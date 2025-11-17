# test_trek.jl
# Simple code to test the Julia interface to TREK

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_trek(::Type{T}, ::Type{INT}; dls::String="pbtr") where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{trek_control_type{T,INT}}()
  inform = Ref{trek_inform_type{T,INT}}()

  # Set problem data
  n = INT(3)  # dimension of H
  m = INT(1)  # dimension of A
  H_ne = INT(4)  # number of elements of H
  S_ne = INT(3)  # number of elements of M
  H_dense_ne = INT(6)  # number of elements of H
  S_dense_ne = INT(6)  # number of elements of M
  H_row = INT[1, 2, 3, 3]  # row indices, NB lower triangle
  H_col = INT[1, 2, 3, 1]
  H_ptr = INT[1, 2, 3, 5]
  S_row = INT[1, 2, 3]  # row indices, NB lower triangle
  S_col = INT[1, 2, 3]
  S_ptr = INT[1, 2, 3, 4]
  H_val = T[1.0, 2.0, 3.0, 4.0]
  S_val = T[1.0, 2.0, 1.0]
  H_dense = T[1.0, 0.0, 2.0, 4.0, 0.0, 3.0]
  S_dense = T[1.0, 0.0, 2.0, 0.0, 0.0, 1.0]
  H_diag = T[1.0, 0.0, 2.0]
  S_diag = T[1.0, 2.0, 1.0]
  c = T[0.0, 2.0, 0.0]

  st = ' '
  status = Ref{INT}()
  x = zeros(T, n)
  sr = ""

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  for s_is in 0:1 # include a scaling matrix?

    for storage_type in 1:4
      # Initialize TREK
      trek_initialize(T, INT, data, control, status)

      # Linear solvers
      @reset control[].linear_solver = galahad_linear_solver(dls)
      @reset control[].linear_solver_for_s = galahad_linear_solver(dls)

      # sparse co-ordinate storage
      if storage_type == 1
        st = 'C'
        # import the control parameters and structural data
        trek_import(T, INT, control, data, status, n,
                   "coordinate", H_ne, H_row, H_col, C_NULL)
        if s_is == 1
          trek_import_s(T, INT, data, status, n,
                       "coordinate", S_ne, S_row, S_col, C_NULL)
        end
      end

      # sparse by rows
      if storage_type == 2
        st = 'R'
        # import the control parameters and structural data
        trek_import(T, INT, control, data, status, n,
                   "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr)
        if s_is == 1
          trek_import_s(T, INT, data, status, n,
                       "sparse_by_rows", S_ne, C_NULL, S_col, S_ptr)
        end
      end

      # dense
      if storage_type == 3
        st = 'D'
        # import the control parameters and structural data
        trek_import(T, INT, control, data, status, n,
                   "dense", H_ne, C_NULL, C_NULL, C_NULL)
        if s_is == 1
          trek_import_s(T, INT, data, status, n,
                       "dense", S_ne, C_NULL, C_NULL, C_NULL)
        end
      end

      # diagonal
      if storage_type == 4
        st = 'L'
        # import the control parameters and structural data
        trek_import(T, INT, control, data, status, n,
                   "diagonal", H_ne, C_NULL, C_NULL, C_NULL)
        if s_is == 1
          trek_import_s(T, INT, data, status, n,
                       "diagonal", S_ne, C_NULL, C_NULL, C_NULL)
        end
      end

      for r_is in 1:2 # original or smaller radius

        if (r_is == 1)
          radius = one(T)
        else
          radius = inform[].next_radius
          @reset control[].new_radius = true
          trek_reset_control(T, INT, control, data, status)
        end

        if (r_is == 2) && (s_is == 1)
          sr = "S-"
        elseif r_is == 2
          sr = "- "
        elseif s_is == 1
          sr = "S "
        else
          sr = "  "
        end

        # solve the problem

        # sparse co-ordinate storage
        if storage_type == 1
          if s_is == 1
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              S_ne, S_val, 0, 0, C_NULL, C_NULL)
          else
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        # sparse by rows
        if storage_type == 2
          if s_is == 1
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              S_ne, S_val, 0, 0, C_NULL, C_NULL)
          else
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_ne, H_val, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        # dense
        if storage_type == 3
          if s_is == 1
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_dense_ne, H_dense, x,
                              S_dense_ne, S_dense, 0, 0, C_NULL, C_NULL)
          else
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, H_dense_ne, H_dense, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        # diagonal
        if storage_type == 4
          if s_is == 1
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, n, H_diag, x,
                              n, S_diag, 0, 0, C_NULL, C_NULL)
          else
            trek_solve_problem(T, INT, data, status, n,
                              radius, f, c, n, H_diag, x,
                              0, C_NULL, 0, 0, C_NULL, C_NULL)
          end
        end

        trek_information(T, INT, data, inform, status)

        @printf("format %c%s: TREK_solve_problem exit status = %1i, f = %.2f\n", st, sr,
                inform[].status, inform[].obj)

      end

      # Delete internal workspace
      trek_terminate(T, INT, data, control, inform)

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
    @testset "TREK -- $T -- $INT" begin
      @test test_trek(T, INT) == 0
    end
end
