! THIS VERSION: GALAHAD 2.4 - 4/02/2008 AT 09:00 GMT.
PROGRAM GALAHAD_mop_example
  USE GALAHAD_SMT_double  ! double precision version
  USE GALAHAD_MOP_double  ! double precision version
  IMPLICIT NONE
  integer, parameter :: wp = KIND( 1.0D+0 ) ! Define the working precision
  real (kind = wp), parameter :: one = 1.0_wp, two = 2.0_wp, three = 3.0_wp
  real (kind = wp), dimension(:), allocatable :: X, u, v, R
  real (kind = wp) :: val, alpha, beta
  integer :: m, n, row, col, out, error, print_level, stat
  logical :: symmetric, transpose
  type( SMT_type ) :: A
! Begin problem data.
  A%m = 2 ; A%n = 3 ; A%ne = 6
  allocate( A%row( A%ne ), A%col( A%ne ), A%val( A%ne ), X( A%n ), u( A%m ), v( A%n ), R( A%m ) )
  A%row = (/ 1, 1, 1, 2, 2, 2 /) ;  A%col = (/ 1, 2, 3, 1, 2, 3 /) ; A%val = (/ 1, 2, 3, 4, 5, 6 /)  
  call SMT_put( A%id, 'Toy 2x3 matrix', stat );
  call SMT_put( A%type, 'COORDINATE', stat )
  X = (/ one, one, one /)  ;  R = (/ three, three /)
  u = (/ two, -one /)      ;  v = (/ three, one, two /)
! Compute : R <- 3*A X + 2*R
  alpha = three ;    print_level = 3
  beta  = two   ;    symmetric   = .false. 
  out   = 6     ;    transpose   = .false.
  error = 6
  write(*,*) 'Compute R <- alpha*A*X + beta*R .....'
  call mop_Ax( alpha, A, X, beta, R, out, error, print_level, symmetric, transpose )
! Scale rows of A by u and columns by v.
  write(*,*) 'Scale rows of A by u and columns by v .....'
  call mop_scaleA( A, u, v, out, error, print_level, symmetric )
! Get the (1,2) element of scaled matrix.
  row = 1 ;  col = 2
  write(*,*) 'Obtain the (1,2) element of the scaled matrix A .....'
  call mop_getval( A, row, col, val, symmetric, out, error, print_level )
  write(*,*) 'The value of the (1,2) element of the scaled matrix is', val
END PROGRAM GALAHAD_mop_example
