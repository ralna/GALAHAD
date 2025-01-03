# test_llsr.jl
# Simple code to test the Julia interface to LLSR

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_llsr(::Type{T}, ::Type{INT}) where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{llsr_control_type{T,INT}}()
  inform = Ref{llsr_inform_type{T,INT}}()

  # Set problem data
  # set dimensions
  m = INT(100)
  n = 2 * m + 1

  # # A = (I : Diag(1:n) : e)
  A_ne = 3 * m
  A_row = zeros(INT, A_ne)
  A_col = zeros(INT, A_ne)
  A_ptr = zeros(INT, m + 1)
  A_val = zeros(T, A_ne)

  # store A in sparse formats
  l = 1
  for i in 1:m
    A_ptr[i] = l
    A_row[l] = i
    A_col[l] = i
    A_val[l] = 1.0
    l = l + 1
    A_row[l] = i
    A_col[l] = m + i
    A_val[l] = i
    l = l + 1
    A_row[l] = i
    A_col[l] = n
    A_val[l] = 1.0
    l = l + 1
  end
  A_ptr[m + 1] = l

  # store A in dense format
  A_dense_ne = m * n
  A_dense_val = zeros(T, A_dense_ne)
  l = 0
  for i in 1:m
    A_dense_val[l + i] = 1.0
    A_dense_val[l + m + i] = i
    A_dense_val[l + n] = 1.0
    l = l + n
  end

  # S = diag(1:n)**2
  S_ne = n
  S_row = zeros(INT, S_ne)
  S_col = zeros(INT, S_ne)
  S_ptr = zeros(INT, n + 1)
  S_val = zeros(T, S_ne)

  # store S in sparse formats
  for i in 1:n
    S_row[i] = i
    S_col[i] = i
    S_ptr[i] = i
    S_val[i] = i * i
  end
  S_ptr[n + 1] = n + 1

  # store S in dense format
  S_dense_ne = div(n * (n + 1), 2)
  S_dense_val = zeros(T, S_dense_ne)
  l = 0
  for i in 1:n
    S_dense_val[l + i] = i * i
    l = l + i
  end

  # b is a vector of ones
  b = ones(T, m) # observations

  # cubic regularization, weight is one
  power = T(3.0)
  weight = one(T)

  # Set output storage
  x = zeros(T, n) # solution
  st = ' '
  status = Ref{INT}()

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of problem storage formats\n\n")

  # loop over storage formats
  for d in 1:4

    # Initialize LLSR
    llsr_initialize(T, INT, data, control, status)
    @reset control[].definite_linear_solver = galahad_linear_solver("potr")
    @reset control[].sbls_control.symmetric_linear_solver = galahad_linear_solver("sytr")
    @reset control[].sbls_control.definite_linear_solver = galahad_linear_solver("potr")
    # @reset control[].print_level = 1

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # use s or not (1 or 0)
    for use_s in 0:1

      # sparse co-ordinate storage
      if d == 1
        st = 'C'
        llsr_import(T, INT, control, data, status, m, n,
                    "coordinate", A_ne, A_row, A_col, C_NULL)

        if use_s == 0
          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_ne, A_val, b, x, 0, C_NULL)
        else
          llsr_import_scaling(T, INT, control, data, status, n,
                              "coordinate", S_ne, S_row,
                              S_col, C_NULL)

          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_ne, A_val, b, x, S_ne, S_val)
        end
      end

      # sparse by rows
      if d == 2
        st = 'R'
        llsr_import(T, INT, control, data, status, m, n,
                    "sparse_by_rows", A_ne, C_NULL, A_col, A_ptr)

        if use_s == 0
          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_ne, A_val, b, x, 0, C_NULL)
        else
          llsr_import_scaling(T, INT, control, data, status, n,
                              "sparse_by_rows", S_ne, C_NULL,
                              S_col, S_ptr)

          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_ne, A_val, b, x, S_ne, S_val)
        end
      end

      # dense
      if d == 3
        st = 'D'
        llsr_import(T, INT, control, data, status, m, n,
                    "dense", A_dense_ne, C_NULL, C_NULL, C_NULL)
        if use_s == 0
          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_dense_ne, A_dense_val, b, x,
                             0, C_NULL)
        else
          llsr_import_scaling(T, INT, control, data, status, n,
                              "dense", S_dense_ne,
                              C_NULL, C_NULL, C_NULL)

          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_dense_ne, A_dense_val, b, x,
                             S_dense_ne, S_dense_val)
        end
      end

      # diagonal
      if d == 4
        st = 'I'
        llsr_import(T, INT, control, data, status, m, n,
                    "coordinate", A_ne, A_row, A_col, C_NULL)
        if use_s == 0
          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_ne, A_val, b, x, 0, C_NULL)
        else
          llsr_import_scaling(T, INT, control, data, status, n,
                              "diagonal", S_ne, C_NULL, C_NULL, C_NULL)

          llsr_solve_problem(T, INT, data, status, m, n, power, weight,
                             A_ne, A_val, b, x, S_ne, S_val)
        end
      end

      llsr_information(T, INT, data, inform, status)

      if inform[].status == 0
        @printf("storage type %c%1i:  status = %1i, ||r|| = %5.2f\n", st, use_s,
                inform[].status, inform[].r_norm)
      else
        @printf("storage type %c%1i: LLSR_solve exit status = %1i\n", st, use_s,
                inform[].status)
      end
    end

    # @printf("x: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end
    # @printf("\n")

    # Delete internal workspace
    llsr_terminate(T, INT, data, control, inform)
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
    @testset "LLSR -- $T -- $INT" begin
      @test test_llsr(T, INT) == 0
    end
  end
end
