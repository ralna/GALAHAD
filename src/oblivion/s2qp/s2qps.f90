! THIS VERSION: GALAHAD 2.4 - 4/02/2008 AT 09:00 GMT.
PROGRAM S2QP_example
  USE GALAHAD_SMT_double
  USE GALAHAD_NLPT_double
  USE GALAHAD_S2QP_double
  USE GALAHAD_SPACE_double
  USE GALAHAD_SMT_double
  USE GALAHAD_SYMBOLS
  IMPLICIT NONE
  !*************************************************!
  ! Program for solving the toy problem:            !
  !                                                 !
  !   minimize ( x1 + x2 + x3 )**2 + 3*x3 + 5*x4    !
  !                                                 !
  !   subject to   x1**2 + x2**2 + x3 = 2           !
  !                x2**2 + x4         = 4           !
  !                2*x1 + 4*x2       >= 0           !
  !                        -2 <=  x1 <= 2           !
  !                        -2 <=  x2 <= 2           !
  !                        -2 <=  x3 <= 2           !
  !                        -2 <=  x4 <= 2           !
  !                                                 !
  !*************************************************!
  INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )  ! Set precision.
  REAL ( KIND = wp ), PARAMETER :: zero     = 0.0_wp
  REAL ( KIND = wp ), PARAMETER :: one      = 1.0_wp
  REAL ( KIND = wp ), PARAMETER :: two      = 2.0_wp
  REAL ( KIND = wp ), PARAMETER :: three    = 3.0_wp
  REAL ( KIND = wp ), PARAMETER :: four     = 4.0_wp
  REAL ( KIND = wp ), PARAMETER :: ten      = 10.0_wp
  REAL ( KIND = wp ), PARAMETER :: infinity = 1.0D+18
  INTEGER, PARAMETER :: out   = 6
  INTEGER, PARAMETER :: error = 6
  INTEGER :: m, m_a, n, Ane, Jne, Hne, status, alloc_status
  INTEGER :: num_recursive_calls, max_num_recursive_calls
  LOGICAL :: is_specfile, print_debug
  CHARACTER ( LEN = 16 ) :: spec_name = "RUNS2QP.SPC"
  INTEGER, PARAMETER :: spec_device = 60
  TYPE ( NLPT_problem_type ) :: nlp 
  TYPE ( S2QP_control_type ) :: control
  TYPE ( S2QP_inform_type ) :: inform
  TYPE ( S2QP_data_type ) :: data
  TYPE ( NLPT_userdata_type ) :: userdata
  EXTERNAL fun_FC, fun_GJ, fun_H
  
  ! Set problem dimensions.
  
  nlp%n = 4 ;  nlp%m = 2 ;  nlp%m_a = 1
  
  ! Number of nonzeros in sparse Jacobian matrix J (co-ordinate storage)
  ! and sparse linear constraint matrix A, and number of nonzeros in
  ! lower triangular part of sparse Hessian H (co-ordinate storage).
  
  nlp%H%ne = 6 ;  nlp%J%ne = 5 ;  nlp%A%ne = 2
  
  call SMT_put( nlp%H%type, 'COORDINATE', status )
  call SMT_put( nlp%J%type, 'COORDINATE', status )
  call SMT_put( nlp%A%type, 'COORDINATE', status )
  
  ! for convenience.
  
  n   = nlp%n    ;   m   = nlp%m    ;   m_a = nlp%m_a
  Hne = nlp%H%ne ;   Jne = nlp%J%ne ;   Ane = nlp%A%ne
  
  ! Allocate arrays.
  
  CALL SPACE_resize_array( n, nlp%VNAMES, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( n, nlp%X, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( n, nlp%X_l, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( n, nlp%X_u, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( n, nlp%Z, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( n, nlp%G, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m, nlp%CNAMES, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m, nlp%C, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m, nlp%C_u, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m, nlp%C_l, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m, nlp%Y, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Jne, nlp%J%row, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Jne, nlp%J%col, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Jne, nlp%J%val, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m_a, nlp%ANAMES, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m_a, nlp%Ax, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m_a, nlp%A_u, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m_a, nlp%A_l, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( m_a, nlp%Y_a, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Ane, nlp%A%row, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Ane, nlp%A%col, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Ane, nlp%A%val, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Hne, nlp%H%row, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Hne, nlp%H%col, status, alloc_status )
  IF ( status /= 0 ) GO TO 990
  CALL SPACE_resize_array( Hne, nlp%H%val, status, alloc_status )
  IF ( status /= 0 ) GO TO 990

  ! Dimension, Sparsity, and values for linear constraint.
 
  nlp%A%m   = 1
  nlp%A%n   = 4 
  nlp%A%row = (/ 1, 1 /)
  nlp%A%col = (/ 1, 2 /)
  nlp%A%val = (/ two, four /)
 
  ! Dimension, Sparsity, and values for general constraints.
  
  nlp%J%m   = 2
  nlp%J%n   = 4 
  nlp%J%row = (/ 1, 1, 1, 2, 2 /)
  nlp%J%col = (/ 1, 2, 3, 2, 4 /) 

  ! Dimension, Sparsity, and values for Hessian (lower triangular part only)
  
  nlp%H%m   = 4
  nlp%H%n   = 4
  nlp%H%row = (/ 1, 2, 2, 3, 3, 3 /)
  nlp%H%col = (/ 1, 1, 2, 1, 2, 3 /)
  
  ! Now fill the rest of the problem vectors.
  
  nlp%PNAME    = 'S2QP example'
  nlp%VNAMES   = (/ 'X1', 'X2', 'X3', 'X4' /)
  nlp%CNAMES   = (/ 'C1', 'C2' /)
  nlp%ANAMES   = (/ 'A1' /)
  nlp%A_l      = zero
  nlp%A_u      = infinity
  nlp%C_l      = (/ two, four /)
  nlp%C_u      = (/ two, four /)
  nlp%X_l      = (/ -two, -two, -two, -two /)
  nlp%X_u      = (/  two,  two,  two, two /)
  nlp%X        = (/  0.1_wp, 0.125_wp, 0.666666_wp, 0.142857_wp /)
  nlp%Y        = zero
  nlp%Y_a      = zero
  nlp%Z        = zero
  
  !  Initialize data structure and control structure.

  CALL S2QP_initialize( data, control, inform )
  
  ! Adjust control parameter to agree with the one defined above.

   control%infinity = infinity

   ! IMPORTANT :signify initial entry into s2qp_solve 

   inform%status = 1

   ! Solve the optimization problem.

   CALL S2QP_solve( nlp, control, inform, data, userdata, fun_FC, fun_GJ, fun_H )

   !  Termination

   CALL S2QP_terminate( data, control, inform )

   !  Deallocate everything from nlpt that has been allocated.

   CALL NLPT_cleanup( nlp )

   STOP

   ! Abnormal return.

990 CONTINUE
   WRITE( out, "( ' TOY1: allocation error ', I0, ' status ', I0 )" )   &
          status, alloc_status
   STOP

 END PROGRAM S2QP_example

 ! ------------------------   EXTERNAL FUNCTIONS   --------------------------

 SUBROUTINE fun_FC( status, X, userdata, F, C )
   USE GALAHAD_NLPT_double 
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 ) ! Set precision
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ) :: F
   REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
   INTEGER, INTENT( OUT ) :: status
   TYPE( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

   ! local variables
   
   REAL( KIND = wp ) :: X1, X2, X3, X4

   ! Compute function values.

   X1 = X(1) ;  X2 = X(2) ;  X3 = X(3) ;  X4 = X(4)

   IF ( PRESENT( F ) ) THEN
      F = ( X1 + X2 + X3 )**2 + three*X3 + five*X4
   END IF
   IF ( PRESENT( C ) ) THEN
      C(1) = X1**2 + X2**2 + X3  
      C(2) = X2**2 + X4
   END IF

   status = 0
   
   RETURN

 END SUBROUTINE fun_FC
   
!-*-*-*-*-*-*-*-*-   f u n G J   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

 SUBROUTINE fun_GJ( status, X, userdata, G, Jval )
   USE GALAHAD_NLPT_double 
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )  ! Set precision
   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
   REAL ( KIND = wp), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), intent(out), OPTIONAL, DIMENSION( : ) :: G
   REAL ( KIND = wp ), intent(out), OPTIONAL, DIMENSION( : ) :: Jval
   INTEGER, INTENT( OUT ) :: status
   TYPE( NLPT_userdata_type ), INTENT(INOUT) :: userdata

   ! local variables

   REAL( KIND = wp ) :: X1, X2, X3
   REAL( KIND = wp ) :: J11, J12, J13, J22, J24 

   ! get the gradient function values.
   
   X1 = X(1)
   X2 = X(2)
   X3 = X(3)

   IF ( PRESENT( G ) ) THEN
      G(1) = two * ( X1+ X2 + X3 )
      G(2) = two * ( X1+ X2 + X3 )
      G(3) = two * ( X1+ X2 + X3 ) + three
      G(4) = five
   END IF
   IF ( PRESENT( Jval ) ) THEN
      J11 = two * X1
      J12 = two * X2
      J13 = one
      J22 = two * X2
      J24 = one
      Jval(1) = J11
      Jval(2) = J12
      Jval(3) = J13
      Jval(4) = J22
      Jval(5) = J24
   END IF

   status = 0

   RETURN

 END SUBROUTINE fun_GJ

!-*-*-*-*-*-*-*-*-   f u n H   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

 SUBROUTINE fun_H( status, X, Y, userdata, Hval )
   USE GALAHAD_NLPT_double 
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )  ! Set precision
   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
   REAL ( KIND = wp ), PARAMETER ::  twelve = 12.0_wp   
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
   INTEGER, INTENT( OUT ) :: status
   TYPE( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

   ! local variables

   REAL( KIND = wp ) :: X1, Y1, Y2
   REAL( KIND = wp ) :: H11, H21, H22, H31, H32, H33

   ! to prevent warning message

   !X1 = X(1) ; X1 = -X1

   ! compute the values of the Hessian of the Lagrangian.
   
   Y1 = Y(1) ;  Y2 = Y(2)

   H11 = TWO - TWO*Y1
   H21 = TWO
   H22 = TWO - TWO*Y1 - TWO * Y2
   H31 = TWO
   H32 = TWO
   H33 = TWO

   Hval(1) = H11
   Hval(2) = H21
   Hval(3) = H22
   Hval(4) = H31
   Hval(5) = H32
   Hval(6) = H33

   status = 0

   RETURN

 END SUBROUTINE fun_H
