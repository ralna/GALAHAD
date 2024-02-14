# test_sls.jl
# Simple code to test the Julia interface to SLS

using GALAHAD
using Test
using Printf
using Accessors

function test_sls()
  maxabsarray(a) = abs.(a) |> maximum

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{sls_control_type{Float64}}()
  inform = Ref{sls_inform_type{Float64}}()

  # Set problem data
  n = 5 # dimension of A
  ne = 7 # number of entries of A
  dense_ne = 15 # number of elements of A as a dense matrix
  row = Cint[1, 2, 2, 3, 3, 4, 5]  # row indices, NB lower triangle
  col = Cint[1, 1, 5, 2, 3, 3, 5]  # column indices
  ptr = Cint[1, 2, 4, 6, 7, 8]  # pointers to indices
  val = Float64[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]  # values
  dense = Float64[2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0,
                  0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0]
  rhs = Float64[8.0, 45.0, 31.0, 15.0, 17.0]
  sol = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
  status = Ref{Cint}()
  x = zeros(Float64, n)
  error = zeros(Float64, n)

  norm_residual = Ref{Float64}()
  good_x = eps(Float64)^(1 / 3)

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")
  @printf(" storage  RHS   refine  partial\n")

  for d in 1:3
    # Initialize SLS - use the sytr solver
    sls_initialize("sytr", data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      @printf(" coordinate ")
      sls_analyse_matrix(control, data, status, n,
                         "coordinate", ne, row, col, C_NULL)
      sls_factorize_matrix(data, status, ne, val)
    end

    # sparse by rows
    if d == 2
      @printf(" sparse by rows ")
      sls_analyse_matrix(control, data, status, n,
                         "sparse_by_rows", ne, C_NULL, col, ptr)
      sls_factorize_matrix(data, status, ne, val)
    end

    # dense
    if d == 3
      @printf(" dense  ")
      sls_analyse_matrix(control, data, status, n,
                         "dense", ne, C_NULL, C_NULL, C_NULL)
      sls_factorize_matrix(data, status, dense_ne, dense)
    end

    # Set right-hand side and solve the system
    for i in 1:n
      x[i] = rhs[i]
    end

    sls_solve_system(data, status, n, x)
    sls_information(data, inform, status)

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
    @reset control[].max_iterative_refinements = Cint(1)
    sls_reset_control(control, data, status)

    for i in 1:n
      x[i] = rhs[i]
    end

    sls_solve_system(data, status, n, x)
    sls_information(data, inform, status)

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

    sls_partial_solve_system("L", data, status, n, x)
    sls_partial_solve_system("D", data, status, n, x)
    sls_partial_solve_system("U", data, status, n, x)
    sls_information(data, inform, status)

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
    sls_terminate(data, control, inform)
  end

  return 0
end

@testset "SLS" begin
  @test test_sls() == 0
end
