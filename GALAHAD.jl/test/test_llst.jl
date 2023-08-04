# test_llst.jl
# Simple code to test the Julia interface to LLST

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = llst_control_type{Float64}()
inform = llst_inform_type{Float64}()

# Set problem data
# set dimensions
m = 100
n = 2*m+1
# A = ( I : Diag(1:n) : e )
A_ne = 3*m
A_row[A_ne]
A_col[A_ne]
A_ptr[m+1]
A_val[A_ne]

# store A in sparse formats
l=0
for( i=1 i <= m i++)
 A_ptr[i-1] = l+1
 A_row[l] = i 
 A_col[l] = i 
 A_val[l] = 1.0
 l++
 A_row[l] = i 
 A_col[l] = m+i  
 A_val[l] = i
 l++
 A_row[l] = i 
 A_col[l] = n  
 A_val[l] = 1.0
 l++
]
A_ptr[m] = l+1

# store A in dense format
A_dense_ne = m * n
real_wp_ A_dense_val[A_dense_ne]
for( i=0 i < A_dense_ne i++) A_dense_val[i] = 0.0
l=-1
for( i=1 i <= m i++)
 A_dense_val[l+i] = 1.0
 A_dense_val[l+m+i] = i
 A_dense_val[l+n] = 1.0
 l=l+n
]

# S = diag(1:n)**2
S_ne = n
S_row[S_ne]
S_col[S_ne]
S_ptr[n+1]
S_val = zeros(Float64, S_ne)

# store S in sparse formats
for i = 1:n
  S_row[i] = i+1 
  S_col[i] = i+1 
  S_ptr[i] = i+1
  S_val[i] = (i+1)*(i+1)
end
S_ptr[n+1] = n+2

# store S in dense format
S_dense_ne = n*(n+1)/2
S_dense_val = zeros(Float64, S_dense_ne)
l=-1
for( i=1 i <= n i++)
  S_dense_val[l+i] = i*i
  l=l+i
]

# b is a vector of ones
b = ones(Float64, m)  # observations

# trust-region radius is one
radius = 1.0 

# Set output storage
x = zeros(Float64, n)  # solution
st = ' '
status = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of problem storage formats\n\n")

# loop over storage formats
for d = 1:4
  # Initialize LLST
  llst_initialize( data, control, status )
  strcpy(control.definite_linear_solver, "potr ") 
  strcpy(control.sbls_control.symmetric_linear_solver, "sytr ") 
  strcpy(control.sbls_control.definite_linear_solver, "potr ") 
  # control.print_level = 1

  # Set user-defined control options
  control.f_indexing = true  # Fortran sparse matrix indexing

  # use s or not (1 or 0)
  for use_s = 0:1
    # sparse co-ordinate storage
    if d == 1
      global st = 'C'

      llst_import( control, data, status, m, n,
                    "coordinate", A_ne, A_row, A_col, Cint[] )

      if use_s == 0
        llst_solve_problem( data, status, m, n, radius, A_ne, A_val, b, x, 0, Cint[] )
      else
        llst_import_scaling( control, data, status, n, "coordinate", S_ne, S_row, S_col, Cint[] )
        llst_solve_problem( data, status, m, n, radius, A_ne, A_val, b, x, S_ne, S_val )
      end
    end

    # sparse by rows
    if d == 2
      global st = 'R'
      
      llst_import( control, data, status, m, n,
                   "sparse_by_rows", A_ne, Cint[], A_col, A_ptr )

      if use_s == 0
        llst_solve_problem( data, status, m, n, radius, A_ne, A_val, b, x, 0, Cint[] )
      else
        llst_import_scaling( control, data, status, n, "sparse_by_rows", S_ne, Cint[], S_col, S_ptr )
        llst_solve_problem( data, status, m, n, radius, A_ne, A_val, b, x, S_ne, S_val )
      end
    end
    
    # dense
    if d == 3
      global st = 'D'

      llst_import( control, data, status, m, n, "dense", A_dense_ne, Cint[], Cint[], Cint[] )
      if use_s == 0
        llst_solve_problem( data, status, m, n, radius, A_dense_ne, A_dense_val, b, x, 0, Cint[] )
      else
        llst_import_scaling( control, data, status, n, "dense", S_dense_ne, Cint[], Cint[], Cint[] )
        llst_solve_problem( data, status, m, n, radius, A_dense_ne, A_dense_val, b, x, S_dense_ne, S_dense_val )
      end
    end
    
    # diagonal
    if d == 4
      global st = 'I'

      llst_import( control, data, status, m, n, "coordinate", A_ne, A_row, A_col, Cint[] )
      if use_s == 0
        llst_solve_problem( data, status, m, n, radius, A_ne, A_val, b, x, 0, Cint[] )
      else
        llst_import_scaling( control, data, status, n, "diagonal", S_ne, Cint[], Cint[], Cint[] )
        llst_solve_problem( data, status, m, n, radius, A_ne, A_val, b, x, S_ne, S_val )
      end
    end

    llst_information( data, inform, status )

    if inform.status == 0
      @printf("storage type %c%1i:  status = %1i, ||r|| = %5.2f\n", st, use_s, inform.status, inform.r_norm )
    else
      @printf("storage type %c%1i: LLST_solve exit status = %1i\n", st, use_s, inform.status)
    end
  end

  # @printf("x: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # end
  # @printf("\n")

  # Delete internal workspace
  llst_terminate( data, control, inform )
end
