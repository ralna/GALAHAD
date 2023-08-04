# test_sls.jl
# Full test SLS for the Julia interface

using GALAHAD
using Printf

# Function maxabsarray
maxabsarray(a) = maximum(abs.(a))

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = sls_control_type{T}()
inform = sls_inform_type{T}()

# Set problem data
n = 5  # dimension of A
ne = 7  # number of entries of A
dense_ne = 15  # number of elements of A as a dense matrix
row = Cint[0, 1, 1, 2, 2, 3, 4]  # row indices, NB lower triangle
col = Cint[0, 0, 4, 1, 2, 2, 4]  # column indices
ptr = Cint[0, 1, 3, 5, 6, 7]  # pointers to indices
val = Float64[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]  # values
dense = Float64[2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0]
rhs = Float64[8.0, 45.0, 31.0, 15.0, 17.0]
sol = Float64[1.0, 2.0, 3.0, 4.0, 5.0]

x = zeros(Float64, n)
error = zeros(Float64, n)
good_x = eps(Float64)^(1/3)
status = Ref{Cint}()

@printf(" C sparse matrix indexing\n\n")
@printf(" basic tests of storage formats\n\n")
@printf(" storage          RHS   refine  partial\n")

for d = 1:3
  # Initialize SLS - use the sytr solver
  sls_initialize( "sytr", data, control, status )

  # Set user-defined control options
  control.f_indexing = false  # C sparse matrix indexing

  # import matrix data and factorize
  if d == 1
    # sparse co-ordinate storage
    @printf(" coordinate     ")
    sls_analyse_matrix( control, data, status, n,
                        "coordinate", ne, row, col, Cint[] )
    sls_factorize_matrix( data, status, ne, val )
  end
  if d == 2
    # sparse by rows
    @printf(" sparse by rows ")
    sls_analyse_matrix( control, data, status, n, 
                        "sparse_by_rows", ne, Cint[], col, ptr )
    sls_factorize_matrix( data, status, ne, val )
  end
  if d == 3
    # dense
    @printf(" dense          ")
    sls_analyse_matrix( control, data, status, n,
                        "dense", ne, Cint[], Cint[], Cint[] )
    sls_factorize_matrix( data, status, dense_ne, dense )
  end

  # Set right-hand side and solve the system
  x .= rhs
  sls_solve_system( data, status, n, x )
  sls_information( data, inform, status )

  if inform.status == 0
    error .= x .- sol
    norm_residual = maxabsarray( error )
    if norm_residual < good_x
      @printf("   ok  ")
    else
      @printf("  fail ")
    end
  else
    @printf(" SLS_solve exit status = %1i\n", inform.status)
  end

  # resolve, this time using iterative refinement
  control.max_iterative_refinements = 1
  sls_reset_control( control, data, status )
  x .= rhs
  sls_solve_system( data, status, n, x )
  sls_information( data, inform, status )

  if inform.status == 0
    error .= x .- sol
    norm_residual = maxabsarray( error )
    if norm_residual < good_x
      @printf("    ok  ")
    else
      @printf("   fail ")
    end
  else
    @printf(" SLS_solve exit status = %1i\n", inform.status)
  end

  # obtain the solution by part solves
  x[i] .= rhs
  sls_partial_solve_system( "L", data, status, n, x )
  sls_partial_solve_system( "D", data, status, n, x )
  sls_partial_solve_system( "U", data, status, n, x )
  sls_information( data, inform, status )

  if inform.status == 0
    error .= x .- sol
    norm_residual = maxabsarray( error )
    if norm_residual < good_x
      @printf("    ok  ")
    else
      @printf("   fail ")
    end
  else
    @printf(" SLS_solve exit status = %1i\n", inform.status)
  end
end

# Delete internal workspace
sls_terminate( data, control, inform )
