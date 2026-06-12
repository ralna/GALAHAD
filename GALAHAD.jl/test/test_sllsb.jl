# test_sllsb.jl
# Simple code to test the Julia interface to SLLSB

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_sllsb(::Type{T}, ::Type{INT}; sls::String="sytr") where {T,INT}
  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sllsb_control_type{T,INT}}()
  inform = Ref{sllsb_inform_type{T,INT}}()

  # Set problem data
  n = INT(10)  # dimension
  o = n + INT(1)  # number of residuals
  m = INT(1)  # number of cohorts
  sigma = one(T) # regularization weight
  Ao_ne = 2 * n # sparse Jacobian elements
  Ao_dense_ne = o * n # dense Jacobian elements
  # row-wise storage
  Ao_row = zeros(INT, Ao_ne) # row indices,
  Ao_col = zeros(INT, Ao_ne) # column indices
  Ao_ptr_ne = o + INT(1)  # row pointer length
  Ao_ptr = zeros(INT, Ao_ptr_ne) # row pointers
  Ao_val = zeros(T, Ao_ne) # values
  Ao_dense = zeros(T, Ao_dense_ne) # dense values
  # column-wise storage
  Ao_by_col_row = zeros(INT, Ao_ne) # row indices,
  Ao_by_col_ptr_ne = n + INT(1)  # column pointer length
  Ao_by_col_ptr = zeros(INT, Ao_by_col_ptr_ne) # column pointers
  Ao_by_col_val = zeros(T, Ao_ne) # values
  Ao_by_col_dense = zeros(T, Ao_dense_ne) # dense values
  b = zeros(T, o)  # linear term in the objective
  x = zeros(T, n) # variables
  y = zeros(T, m) # Lagrange multipliers
  z = zeros(T, n) # dual variables
  r = zeros(T, o) # residual
  g = zeros(T, n) # gradient

  # Set output storage
  x_stat = zeros(INT, n) # variable status
  st = ""
  status = Ref{INT}()

  #   A = ( I )  and b = (i * e)
  #       (e^T)          (n + 1)
  for i in 1:n
    b[i] = i
  end
  b[n + 1] = n + 1

  # A by rows
  for i in 1:n
    Ao_ptr[i] = i
    Ao_row[i] = i
    Ao_col[i] = i
    Ao_val[i] = 1.0
  end
  Ao_ptr[n + 1] = n + 1
  for i in 1:n
    Ao_row[n + i] = o
    Ao_col[n + i] = i
    Ao_val[n + i] = 1.0
  end
  Ao_ptr[o + 1] = Ao_ne + 1
  l = 0
  for i in 1:n
    for j in 1:n
      l = l + 1
      if i == j
        Ao_dense[l] = 1.0
      else
        Ao_dense[l] = 0.0
      end
    end
  end
  for j in 1:n
    l = l + 1
    Ao_dense[l] = 1.0
  end

  # # A by columns
  l = 0
  for j in 1:n
    l = l + 1
    Ao_by_col_ptr[j] = l
    Ao_by_col_row[l] = j
    Ao_by_col_val[l] = 1.0
    l = l + 1
    Ao_by_col_row[l] = o
    Ao_by_col_val[l] = 1.0
  end
  Ao_by_col_ptr[n + 1] = Ao_ne
  l = 0
  for j in 1:n
    for i in 1:n
      l = l + 1
      if i == j
        Ao_by_col_dense[l] = 1.0
      else
        Ao_by_col_dense[l] = 0.0
      end
    end
    l = l + 1
    Ao_by_col_dense[l] = 1.0
  end

  @printf(" fortran sparse matrix indexing\n\n")

  @printf(" basic tests of sllsb storage formats\n\n")

  for d in 1:5
    # Initialize SLLSB
    sllsb_initialize(T, INT, data, control, status)

    # Linear solvers
    @reset control[].symmetric_linear_solver = galahad_linear_solver(sls)
    @reset control[].fdc_control.symmetric_linear_solver =
      galahad_linear_solver(sls)
    @reset control[].fdc_control.use_sls = true
    # @reset control[].print_level = INT(1)

    # Start from 0
    for i = 1:n
      x[i] = zero(T)
    end

    # sparse co-ordinate storage
    if d == 1
      st = "CO"
      # @printf("stop_abs_p = %.2f\n", control[].stop_abs_p)
      sllsb_import(T, INT, control, data, status, n, o, m, "coordinate", 
                   Ao_ne, Ao_row, Ao_col, INT(0), C_NULL, C_NULL)
      sllsb_solve_given_a(T, INT, data, status, n, o, m, 
                          Ao_ne, Ao_val, b, sigma, x, y, z, r, 
                          x_stat, C_NULL, C_NULL )
    end

    # sparse by rows
    if d == 2
      st = "SR"
      sllsb_import(T, INT, control, data, status, n, o, m, "sparse_by_rows", 
                    Ao_ne, C_NULL, Ao_col, Ao_ptr_ne, Ao_ptr, C_NULL)
      sllsb_solve_given_a(T, INT, data, status, n, o, m, 
                          Ao_ne, Ao_val, b, sigma, x, y, z, r, 
                          x_stat, C_NULL, C_NULL )
    end

    # dense by rows
    if d == 3
      st = "DR"
      sllsb_import(T, INT, control, data, status, n, o, m, "dense_by_rows", 
                   Ao_dense_ne, C_NULL, C_NULL, INT(0), C_NULL, C_NULL)
      sllsb_solve_given_a(T, INT, data, status, n, o, m, 
                          Ao_dense_ne, Ao_dense, b, sigma, x, y, z, r,
                          x_stat, C_NULL, C_NULL )
    end

    # sparse by columns
    if d == 4
      st = "SC"
      sllsb_import(T, INT, control, data, status, n, o, m, "sparse_by_columns", 
                   Ao_ne, Ao_by_col_row, C_NULL, 
                   Ao_by_col_ptr_ne, Ao_by_col_ptr, C_NULL)
      sllsb_solve_given_a(T, INT, data, status, n, o, m,
                         Ao_ne, Ao_by_col_val, b, sigma, x, y, z, r, 
                         x_stat, C_NULL, C_NULL )
    end

    # dense by columns
    if d == 5
      st = "DC"
      sllsb_import(T, INT, control, data, status, n, o, m, "dense_by_columns", 
                   Ao_dense_ne, C_NULL, C_NULL, INT(0), C_NULL, C_NULL)
      sllsb_solve_given_a(T, INT, data, status, n, o, m, 
                         Ao_dense_ne, Ao_by_col_dense, b, sigma, x, y, z, r, 
                         x_stat, C_NULL, C_NULL )
    end

    sllsb_information(T, INT, data, inform, status)

    if inform[].status == 0
      @printf("%s:%6i iterations. Optimal objective value = %.2f, status = %1i\n",
              st, inform[].iter, Float64(inform[].obj), inform[].status)
    else
      @printf("%s: SLLSB_solve exit status = %1i\n", st, inform[].status)
    end

    # Delete internal workspace
    sllsb_terminate(T, INT, data, control, inform)
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
    @testset "SLLSB -- $T -- $INT" begin
        @test test_sllsb(T, INT) == 0
    end
  end
end
