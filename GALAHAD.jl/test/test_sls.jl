# test_sls.jl
# Simple code to test the Julia interface to SLS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_sls(::Type{T}, ::Type{INT}) where {T,INT}
  maxabsarray(a) = abs.(a) |> maximum

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sls_control_type{T,INT}}()
  inform = Ref{sls_inform_type{T,INT}}()

  # Set problem data
  n = INT(5)  # dimension of A
  ne = INT(7)  # number of entries of A
  dense_ne = INT(15)  # number of elements of A as a dense matrix
  row = INT[1, 2, 2, 3, 3, 4, 5]  # row indices, NB lower triangle
  col = INT[1, 1, 5, 2, 3, 3, 5]  # column indices
  ptr = INT[1, 2, 4, 6, 7, 8]  # pointers to indices
  val = T[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]  # values
  dense = T[2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0,
                  0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0]
  rhs = T[8.0, 45.0, 31.0, 15.0, 17.0]
  sol = T[1.0, 2.0, 3.0, 4.0, 5.0]
  status = Ref{INT}()
  x = zeros(T, n)
  error = zeros(T, n)

  norm_residual = Ref{T}()
  good_x = eps(Float64)^(1 / 3)

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")
  @printf(" storage  RHS   refine  partial\n")

  for d in 1:3
    # Initialize SLS - use the sytr solver
    sls_initialize(T, INT, "sytr", data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      @printf(" coordinate ")
      sls_analyse_matrix(T, INT, control, data, status, n,
                         "coordinate", ne, row, col, C_NULL)
      sls_factorize_matrix(T, INT, data, status, ne, val)
    end

    # sparse by rows
    if d == 2
      @printf(" sparse by rows ")
      sls_analyse_matrix(T, INT, control, data, status, n,
                         "sparse_by_rows", ne, C_NULL, col, ptr)
      sls_factorize_matrix(T, INT, data, status, ne, val)
    end

    # dense
    if d == 3
      @printf(" dense  ")
      sls_analyse_matrix(T, INT, control, data, status, n,
                         "dense", ne, C_NULL, C_NULL, C_NULL)
      sls_factorize_matrix(T, INT, data, status, dense_ne, dense)
    end

    # Set right-hand side and solve the system
    for i in 1:n
      x[i] = rhs[i]
    end

    sls_solve_system(T, INT, data, status, n, x)
    sls_information(T, INT, data, inform, status)

    if inform[].status == 0
      for i in 1:n
        error[i] = x[i] - sol[i]
      end

      norm_residual = maxabsarray(error)

      if norm_residual < good_x
        @printf("   ok  ")
      else
        @printf("  fail ")
      end
    else
      @printf(" SLS_solve exit status = %1i\n", inform[].status)
    end

    # @printf("sol: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end

    # resolve, this time using iterative refinement
    @reset control[].max_iterative_refinements = INT(1)
    sls_reset_control(T, INT, control, data, status)

    for i in 1:n
      x[i] = rhs[i]
    end

    sls_solve_system(T, INT, data, status, n, x)
    sls_information(T, INT, data, inform, status)

    if inform[].status == 0
      for i in 1:n
        error[i] = x[i] - sol[i]
      end

      norm_residual = maxabsarray(error)

      if norm_residual < good_x
        @printf("ok  ")
      else
        @printf("   fail ")
      end
    else
      @printf(" SLS_solve exit status = %1i\n", inform[].status)
    end

    # obtain the solution by part solves
    for i in 1:n
      x[i] = rhs[i]
    end

    sls_partial_solve_system(T, INT, "L", data, status, n, x)
    sls_partial_solve_system(T, INT, "D", data, status, n, x)
    sls_partial_solve_system(T, INT, "U", data, status, n, x)
    sls_information(T, INT, data, inform, status)

    if inform[].status == 0
      for i in 1:n
        error[i] = x[i] - sol[i]
      end

      norm_residual = maxabsarray(error)

      if norm_residual < good_x
        @printf("ok  ")
      else
        @printf("   fail ")
      end
    else
      @printf(" SLS_solve exit status = %1i\n", inform[].status)
    end

    # Delete internal workspace
    sls_terminate(T, INT, data, control, inform)
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
    @testset "SLS -- $T -- $INT" begin
      @test test_sls(T, INT) == 0
    end
  end
end
