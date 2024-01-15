# test_uls.jl
# Simple code to test the Julia interface to ULS

using GALAHAD
using Printf

# Function maxabsarray
maxabsarray(a) = maximum(abs.(a))

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = uls_control_type{Float64}()
inform = uls_inform_type{Float64}()

# Set problem data
m = 5  # column dimension of A
n = 5  # column dimension of A
ne = 7  # number of entries of A
dense_ne = 25  # number of elements of A as a dense matrix
row = Cint[1, 2, 2, 3, 3, 4, 5]  # row indices
col = Cint[1, 1, 5, 2, 3, 3, 4]  # column indices
ptr = Cint[1, 2, 4, 6, 7, 8]  # pointers to indices
val = Float64[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]  # values
dense = Float64[2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 6.0,
                0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0, 0.0]
rhs = Float64[2.0, 33.0, 11.0, 15.0, 4.0]
rhst = Float64[8.0, 12.0, 23.0, 5.0, 12.0]
sol = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
x = zeros(Float64, n)
error = zeros(Float64, n)

good_x = eps(Float64)^(1/3)

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of storage formats\n\n")
@printf(" storage  RHS   refine   RHST  refine\n")

for d = 1:3
  # Initialize ULS - use the gls solver
  uls_initialize( "getr", data, control, status )

  # Set user-defined control options
  control.f_indexing = true  # Fortran sparse matrix indexing

  # sparse co-ordinate storage
  if d == 1
    @printf(" coordinate ")
    uls_factorize_matrix( control, data, status, m, n,
                          "coordinate", ne, val, row, col, Cint[] )
  end

  # sparse by rows
  if d == 2
    @printf(" sparse by rows ")
    uls_factorize_matrix( control, data, status, m, n,
                          "sparse_by_rows", ne, val, Cint[], col, ptr )
  end

  # dense
  if d == 3
    @printf(" dense  ")
    uls_factorize_matrix( control, data, status, m, n,
                          "dense", dense_ne, dense, Cint[], Cint[], Cint[] )
  end

  # Set right-hand side and solve the system Ax = b
  x .= rhs
  trans = false
  uls_solve_system( data, status, m, n, x, trans )
  uls_information( data, inform, status )

  if inform.status == 0
    error .=  x .- sol
    norm_residual = maxabsarray(error)
    if norm_residual < good_x
      @printf("   ok  ")
    else
      @printf("  fail ")
    end
  else
    @printf(" ULS_solve exit status = %1i\n", inform.status)
  end

  # @printf("sol: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # end

  # resolve, this time using iterative refinement
  control.max_iterative_refinements = 1
  uls_reset_control( control, data, status )
  x .= rhs
  uls_solve_system( data, status, m, n, x, trans )
  uls_information( data, inform, status )

  if inform.status == 0
    error .= x .- sol
    norm_residual = maxabsarray(error)
    if norm_residual < good_x
      @printf("ok  ")
    else
      @printf("   fail ")
    end
  else
    @printf(" ULS_solve exit status = %1i\n", inform.status)
  end

  # Set right-hand side and solve the system Aᵀx = b
  x .= rhst
  trans = true
  uls_solve_system( data, status, m, n, x, trans )
  uls_information( data, inform, status )

  if inform.status == 0
    error .= x .- sol
    norm_residual = maxabsarray(error)
    if norm_residual < good_x
      @printf("   ok  ")
    else
      @printf("  fail ")
    end
  else
    @printf(" ULS_solve exit status = %1i\n", inform.status)
  end

  # @printf("sol: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # end

  # resolve, this time using iterative refinement
  control.max_iterative_refinements = 1
  uls_reset_control( control, data, status )
  x .= rhst
  uls_solve_system( data, status, m, n, x, trans )
  uls_information( data, inform, status )

  if inform.status == 0
    error .= x .- sol
    norm_residual = maxabsarray(error)
    if norm_residual < good_x
      @printf("ok  ")
    else
      @printf("   fail ")
    end
  else
    @printf(" ULS_solve exit status = %1i\n", inform.status)
  end

  # Delete internal workspace
  uls_terminate( data, control, inform )
end
