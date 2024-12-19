# test_uls.jl
# Simple code to test the Julia interface to ULS

using GALAHAD
using Test
using Printf
using Accessors
using Quadmath

function test_uls(::Type{T}) where T
  maxabsarray(a) = maximum(abs.(a))

  # Derived types
  data = Ref{Ptr{Cvoid}}()
  control = Ref{uls_control_type{T}}()
  inform = Ref{uls_inform_type{T}}()

  # Set problem data
  m = 5 # column dimension of A
  n = 5 # column dimension of A
  ne = 7 # number of entries of A
  dense_ne = 25 # number of elements of A as a dense matrix
  row = Cint[1, 2, 2, 3, 3, 4, 5]  # row indices
  col = Cint[1, 1, 5, 2, 3, 3, 4]  # column indices
  ptr = Cint[1, 2, 4, 6, 7, 8]  # pointers to indices
  val = T[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]  # values
  dense = T[2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 6.0, 0.0, 4.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  rhs = T[2.0, 33.0, 11.0, 15.0, 4.0]
  rhst = T[8.0, 12.0, 23.0, 5.0, 12.0]
  sol = T[1.0, 2.0, 3.0, 4.0, 5.0]
  status = Ref{Cint}()
  x = zeros(T, n)
  error = zeros(T, n)
  good_x = eps(Float64)^(1 / 3)

  @printf(" Fortran sparse matrix indexing\n\n")
  @printf(" basic tests of storage formats\n\n")
  @printf(" storage  RHS   refine   RHST  refine\n")

  for d in 1:3
    # Initialize ULS - use the gls solver
    uls_initialize(T, "getr", data, control, status)

    # Set user-defined control options
    @reset control[].f_indexing = true # Fortran sparse matrix indexing

    # sparse co-ordinate storage
    if d == 1
      @printf(" coordinate ")
      uls_factorize_matrix(T, control, data, status, m, n,
                           "coordinate", ne, val, row, col, C_NULL)
    end

    # sparse by rows
    if d == 2
      @printf(" sparse by rows ")
      uls_factorize_matrix(T, control, data, status, m, n,
                           "sparse_by_rows", ne, val, C_NULL, col, ptr)
    end

    # dense
    if d == 3
      @printf(" dense  ")
      uls_factorize_matrix(T, control, data, status, m, n,
                           "dense", dense_ne, dense, C_NULL, C_NULL, C_NULL)
    end

    # Set right-hand side and solve the system A x = b
    for i in 1:n
      x[i] = rhs[i]
    end

    trans = false
    uls_solve_system(T, data, status, m, n, x, trans)
    uls_information(T, data, inform, status)

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
      @printf(" ULS_solve exit status = %1i\n", inform[].status)
    end

    # @printf("sol: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end

    # resolve, this time using iterative refinement
    @reset control[].max_iterative_refinements = Cint(1)
    uls_reset_control(T, control, data, status)

    for i in 1:n
      x[i] = rhs[i]
    end

    uls_solve_system(T, data, status, m, n, x, trans)
    uls_information(T, data, inform, status)

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
      @printf(" ULS_solve exit status = %1i\n", inform[].status)
    end

    # Set right-hand side and solve the system A^T x = b
    for i in 1:n
      x[i] = rhst[i]
    end

    trans = true
    uls_solve_system(T, data, status, m, n, x, trans)
    uls_information(T, data, inform, status)

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
      @printf(" ULS_solve exit status = %1i\n", inform[].status)
    end

    # @printf("sol: ")
    # for i = 1:n
    #   @printf("%f ", x[i])
    # end

    # resolve, this time using iterative refinement
    @reset control[].max_iterative_refinements = Cint(1)

    uls_reset_control(T, control, data, status)
    for i in 1:n
      x[i] = rhst[i]
    end

    uls_solve_system(T, data, status, m, n, x, trans)
    uls_information(T, data, inform, status)

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
      @printf(" ULS_solve exit status = %1i\n", inform[].status)
    end

    # Delete internal workspace
    uls_terminate(T, data, control, inform)
    @printf("\n")
  end

  return 0
end

@testset "ULS" begin
  @test test_uls(Float32) == 0
  @test test_uls(Float64) == 0
  @test test_uls(Float128) == 0
end
