# test_clls.jl
# Simple code to test the Julia interface to CLLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_clls(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{clls_control_type{T,INT}}()
  inform = Ref{clls_inform_type{T,INT}}()

  # Set problem data
  n = INT(3)  # dimension
  o = INT(4)  # number of observations
  m = INT(2)  # number of general constraints
  sigma = one(T) # regularization weight
  b = T[2.0, 2.0, 3.0, 1.0]  # observations
  c_l = T[1.0, 2.0]  # constraint lower bound
  c_u = T[2.0, 2.0]  # constraint upper bound
  x_l = T[-1.0, -Inf, -Inf]  # variable lower bound
  x_u = T[1.0, Inf, 2.0]  # variable upper bound
  w = T[1.0, 1.0, 1.0, 2.0]  # weights

  # Set output storage
  r = zeros(T, o) # residual values
  c = zeros(T, m) # constraint values
  x_stat = zeros(INT, n) # variable status
  c_stat = zeros(INT, m) # constraint status
  st = ""
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of clls storage formats\n\n")

  for d in 1:5
    # Initialize CLLS
    clls_initialize(T, INT, data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing
    @reset control[].symmetric_linear_solver = galahad_linear_solver("potr")
    @reset control[].fdc_control.symmetric_linear_solver = galahad_linear_solver("potr")
    @reset control[].fdc_control.use_sls = true

    # Start from 0
    x = T[0.0, 0.0, 0.0]
    y = T[0.0, 0.0]
    z = T[0.0, 0.0, 0.0]

    # sparse co-ordinate storage
    if d == 1
      st = "CO"
      Ao_ne = 7 # objective Jacobian elements
      Ao_row = INT[1, 1, 2, 2, 3, 3, 4]  # row indices
      Ao_col = INT[1, 2, 2, 3, 1, 3, 2]  # column indices
      Ao_val = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # vals
      A_ne = 4 # constraint Jacobian elements
      A_row = INT[1, 1, 2, 2]  # row indices
      A_col = INT[1, 2, 2, 3]  # column indices
      A_val = T[2.0, 1.0, 1.0, 1.0]  # values

      clls_import(T, INT, control, data, status, n, o, m,
                  "coordinate", Ao_ne, Ao_row, Ao_col, 0, C_NULL,
                  "coordinate", A_ne, A_row, A_col, 0, C_NULL)

      clls_solve_clls(T, INT, data, status, n, o, m, Ao_ne, Ao_val, b,
                      sigma, A_ne, A_val, c_l, c_u, x_l, x_u,
                      x, r, c, y, z, x_stat, c_stat, w)
    end

    # sparse by rows
    if d == 2
      st = "SR"
      Ao_ne = 7 # objective Jacobian elements
      Ao_col = INT[1, 2, 2, 3, 1, 3, 2]  # column indices
      Ao_ptr_ne = o + 1 # number of row pointers
      Ao_ptr = INT[1, 3, 5, 7, 8]  # row pointers
      Ao_val = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # vals
      A_ne = 4 # constraint Jacobian elements
      A_col = INT[1, 2, 2, 3]  # column indices
      A_ptr_ne = m + 1 # number of row pointers
      A_ptr = INT[1, 3, 5]  # row pointers
      A_val = T[2.0, 1.0, 1.0, 1.0]  # values

      clls_import(T, INT, control, data, status, n, o, m,
                  "sparse_by_rows", Ao_ne, C_NULL, Ao_col,
                  Ao_ptr_ne, Ao_ptr,
                  "sparse_by_rows", A_ne, C_NULL, A_col,
                  A_ptr_ne, A_ptr)

      clls_solve_clls(T, INT, data, status, n, o, m, Ao_ne, Ao_val, b,
                      sigma, A_ne, A_val, c_l, c_u, x_l, x_u,
                      x, r, c, y, z, x_stat, c_stat, w)
    end

    # sparse by columns
    if d == 3
      st = "SC"
      Ao_ne = 7 # objective Jacobian elements
      Ao_row = INT[1, 3, 1, 2, 4, 2, 3]  # row indices
      Ao_ptr_ne = n + 1 # number of column pointers
      Ao_ptr = INT[1, 3, 6, 8]  # column pointers
      Ao_val = T[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # vals
      A_ne = 4 # constraint Jacobian elements
      A_row = INT[1, 1, 2, 2]  # row indices
      A_ptr_ne = n + 1 # number of column pointers
      A_ptr = INT[1, 2, 4, 5]  # column pointers
      A_val = T[2.0, 1.0, 1.0, 1.0]  # values

      clls_import(T, INT, control, data, status, n, o, m,
                  "sparse_by_columns", Ao_ne, Ao_row, C_NULL,
                  Ao_ptr_ne, Ao_ptr,
                  "sparse_by_columns", A_ne, A_row, C_NULL,
                  A_ptr_ne, A_ptr)

      clls_solve_clls(T, INT, data, status, n, o, m, Ao_ne, Ao_val, b,
                      sigma, A_ne, A_val, c_l, c_u, x_l, x_u,
                      x, r, c, y, z, x_stat, c_stat, w)
    end

    # dense by rows
    if d == 4
      st = "DR"
      Ao_ne = 12 # objective Jacobian elements
      Ao_dense = T[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
      A_ne = 6 # constraint Jacobian elements
      A_dense = T[2.0, 1.0, 0.0, 0.0, 1.0, 1.0]

      clls_import(T, INT, control, data, status, n, o, m,
                  "dense", Ao_ne, C_NULL, C_NULL, 0, C_NULL,
                  "dense", A_ne, C_NULL, C_NULL, 0, C_NULL)

      clls_solve_clls(T, INT, data, status, n, o, m, Ao_ne, Ao_dense, b,
                      sigma, A_ne, A_dense, c_l, c_u, x_l, x_u,
                      x, r, c, y, z, x_stat, c_stat, w)
    end

    # dense by cols
    if d == 5
      st = "DC"
      Ao_ne = 12 # objective Jacobian elements
      Ao_dense = T[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
      A_ne = 6 # constraint Jacobian elements
      A_dense = T[2.0, 0.0, 1.0, 1.0, 0.0, 1.0]

      clls_import(T, INT, control, data, status, n, o, m,
                  "dense_by_columns", Ao_ne, C_NULL, C_NULL, 0, C_NULL,
                  "dense_by_columns", A_ne, C_NULL, C_NULL, 0, C_NULL)

      clls_solve_clls(T, INT, data, status, n, o, m, Ao_ne, Ao_dense, b,
                      sigma, A_ne, A_dense, c_l, c_u, x_l, x_u,
                      x, r, c, y, z, x_stat, c_stat, w)
    end

    clls_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%s:%6i iterations. Optimal objective value = %5.2f status = %1i\n", st,
              inform[].iter, inform[].obj, inform[].status)
    else
      @printf("%s: CLLS_solve exit status = %1i\n", st, inform[].status)
    end

    # @printf("x: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end
    # @printf("\n")
    # @printf("gradient: ")
    # for i = 1:n
    #   @printf("%f ", g[i])
    # end
    # @printf("\n")

    # Delete internal workspace
    clls_terminate(T, INT, data, control, inform)
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
    @testset "CLLS -- $T -- $INT" begin
      @test test_clls(T, INT) == 0
    end
  end
end
