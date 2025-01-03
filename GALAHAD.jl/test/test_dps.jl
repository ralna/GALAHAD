# test_dps.jl
# Simple code to test the Julia interface to DPS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_dps(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{dps_control_type{T,INT}}()
  inform = Ref{dps_inform_type{T,INT}}()

  # Set problem data
  n = INT(3)  # dimension of H
  m = INT(1)  # dimension of A
  H_ne = INT(4)  # number of elements of H
  H_dense_ne = INT(6)  # number of elements of H
  H_row = INT[1, 2, 3, 3]  # row indices, NB lower triangle
  H_col = INT[1, 2, 3, 1]
  H_ptr = INT[1, 2, 3, 5]
  H_val = T[1.0, 2.0, 3.0, 4.0]
  H_dense = T[1.0, 0.0, 2.0, 4.0, 0.0, 3.0]
  f = T(0.96)
  radius = one(T)
  half_radius = T(0.5)
  c = T[0.0, 2.0, 0.0]

  st = ' '
  status = Ref{INT}()
  x = zeros(T, n)

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")

  for storage_type in 1:3
    # Initialize DPS
    dps_initialize(T, INT, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # fortran sparse matrix indexing
    @reset control[].symmetric_linear_solver = galahad_linear_solver("sytr")

    # sparse co-ordinate storage
    if storage_type == 1
      st = 'C'
      # import the control parameters and structural data
      dps_import(T, INT, control, data, status, n,
                 "coordinate", H_ne, H_row, H_col, C_NULL)

      # solve the problem
      dps_solve_tr_problem(T, INT, data, status, n, H_ne, H_val,
                           c, f, radius, x)
    end

    # sparse by rows
    if storage_type == 2
      st = 'R'
      # import the control parameters and structural data
      dps_import(T, INT, control, data, status, n,
                 "sparse_by_rows", H_ne, C_NULL, H_col, H_ptr)

      dps_solve_tr_problem(T, INT, data, status, n, H_ne, H_val,
                           c, f, radius, x)
    end

    # dense
    if storage_type == 3
      st = 'D'
      # import the control parameters and structural data
      dps_import(T, INT, control, data, status, n,
                 "dense", H_ne, C_NULL, C_NULL, C_NULL)

      dps_solve_tr_problem(T, INT, data, status, n, H_dense_ne, H_dense,
                           c, f, radius, x)
    end

    dps_information(T, INT, data, inform, status)
    @printf("format %c: DPS_solve_problem exit status   = %1i, f = %.2f\n",
            st, inform[].status, inform[].obj)

    # sparse co-ordinate storage
    if storage_type == 1
      st = 'C'
      # solve the problem
      dps_resolve_tr_problem(T, INT, data, status, n,
                             c, f, half_radius, x)
    end

    # sparse by rows
    if storage_type == 2
      st = 'R'
      dps_resolve_tr_problem(T, INT, data, status, n,
                             c, f, half_radius, x)
    end

    # dense
    if storage_type == 3
      st = 'D'
      dps_resolve_tr_problem(T, INT, data, status, n,
                             c, f, half_radius, x)
    end

    dps_information(T, INT, data, inform, status)
    @printf("format %c: DPS_resolve_problem exit status = %1i, f = %.2f\n",
            st, inform[].status, inform[].obj)

    # @printf("x: ")
    # for i = 1:n+m
    #   @printf("%f ", x[i])
    # end

    # Delete internal workspace
    dps_terminate(T, INT, data, control, inform)
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
    @testset "DPS -- $T -- $INT" begin
      @test test_dps(T, INT) == 0
    end
  end
end
