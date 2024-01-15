# test_psls.jl
# Simple code to test the Julia interface to PSLS

using GALAHAD
using Printf

# Derived types
data = [Ptr{Ptr{Cvoid}}()]
control = psls_control_type{Float64}()
inform = psls_inform_type{Float64}()

# Set problem data
n = 5  # dimension of A
ne = 7  # number of elements of A
dense_ne = n * ( n + 1 ) / 2  # number of elements of dense A

row = Cint[1, 2, 2, 3, 3, 4, 5]  # A indices  values, NB lower triangle
col = Cint[1, 1, 5, 2, 3, 3, 5]
ptr = Cint[1, 2, 4, 6, 7, 8}
val = Float64[2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0]
dense = Float64[2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0]
st = ' '
status = Ref{Cint}()
status_apply = Ref{Cint}()

@printf(" Fortran sparse matrix indexing\n\n")
@printf(" basic tests of storage formats\n\n")

for d = 1:3
  # Initialize PSLS
  psls_initialize( data, control, status )
  control.preconditioner = 2  # band preconditioner
  control.semi_bandwidth = 1  # semibandwidth
  solver = "sils"
  control.definite_linear_solver = (115, 105, 108, 115, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 0)

  # Set user-defined control options
  control.f_indexing = true  # fortran sparse matrix indexing

  # sparse co-ordinate storage
  if d == 1
    global st = 'C'

    psls_import( control, data, status, n,
                 "coordinate", ne, row, col, Cint[] )

    psls_form_preconditioner( data, status, ne, val )
  end

  # sparse by rows
  if d == 2
    global st = 'R'

    psls_import( control, data, status, n,
                 "sparse_by_rows", ne, Cint[], col, ptr )

    psls_form_preconditioner( data, status, ne, val )
  end

  # dense
  if d == 3
    global st = 'D'

    psls_import( control, data, status, n,
                 "dense", ne, Cint[], Cint[], Cint[] )

    psls_form_preconditioner( data, status, dense_ne, dense )
  end

  # Set right-hand side b in x
  x = Float64[8.0, 45.0, 31.0, 15.0, 17.0]  # values

  if status == 0
    psls_information( data, inform, status )
    psls_apply_preconditioner( data, status_apply, n, x )
  else
    status_apply[] = -1
  end

  @printf("%c storage: status from form  factorize = %i apply = %i\n", st, status, status_apply )

  # @printf("x: ")
  # for i = 1:n
  #   @printf("%f ", x[i])
  # end

  # Delete internal workspace
  psls_terminate( data, control, inform )
end
