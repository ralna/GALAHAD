!-*-*-*-*-*-*-*-*-  G A L A H A D   T O Y 1  *-*-*-*-*-*-*-*-
!  Daniel Robinson, for GALAHAD productions
!  Copyright reserved
!  December 22th 2007
!
!  Simple problem created for testing purposes.


   PROGRAM TOY1

   USE GALAHAD_SMT_double
   USE GALAHAD_NLPT_double
   USE GALAHAD_TRIMSQP_double
   USE GALAHAD_SPACE_double
   USE GALAHAD_SMT_double
   USE GALAHAD_SYMBOLS
   !USE USER_toy1_functions

   IMPLICIT NONE

   EXTERNAL funFC, funG, funJ, funGJ, funH

!*************************************************!
! Program for solving the toy problem:            !
!                                                 !
!   minimize ( x1 + x2 + x3 )**2 + 3*x3 + 5*x4    !
!                                                 !
!   subject to   x1**2 + x2**2 + x3 = 2           !
!                x2**4 + x4         = 4           !
!                2*x1 + 4*x2       >= 0.          !
!                                                 !
!*************************************************!

   ! Set precision.

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters.

   REAL ( KIND = wp ), PARAMETER ::  zero    = 0.0_wp
   REAL ( KIND = wp ), PARAMETER ::  one     = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two     = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three   = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four    = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  ten     = 10.0_wp
   REAL ( KIND = wp ), PARAMETER :: infinity = 1.0_wp * 10.0**18

   INTEGER, PARAMETER :: out   = 6
   INTEGER, PARAMETER :: error = 6

   ! Set local variables.

   INTEGER :: m, n, J_ne, H_ne, status, alloc_status
   INTEGER :: num_recursive_calls, max_num_recursive_calls
   LOGICAL :: is_specfile, solved, transpose
   CHARACTER ( LEN = 16 ) :: spec_name = "TOY1.SPC"
   INTEGER, PARAMETER :: spec_device = 60

   ! Derived TRIMSQP data types.
   
   TYPE ( NLPT_problem_type ) :: nlp 
   TYPE ( TRIMSQP_control_type ) :: control
   TYPE ( TRIMSQP_inform_type ) :: inform
   TYPE ( TRIMSQP_data_type ) :: data
   TYPE ( TRIMSQP_userdata_type ) :: userdata

   INTERFACE
      
      SUBROUTINE funFC_reverse(F, C, X, userdata)

         USE GALAHAD_TRIMSQP_double
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!        TYPE :: TRIMSQP_userdata_type
!           INTEGER, ALLOCATABLE, DIMENSION( : ) :: integer_userdata
!           REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: real_userdata
!           CHARACTER ( LEN = 15 ), ALLOCATABLE, DIMENSION( : ) :: character_userdata
!           LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical_userdata
!        END TYPE TRIMSQP_userdata_type
!
        REAL ( kind = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
        REAL ( kind = wp ), INTENT( OUT ) :: F
        REAL ( kind = wp ), POINTER, DIMENSION( : ) :: C
!!       TYPE ( TRIMSQP_userdata_type ), INTENT( INOUT ), OPTIONAL :: userdata
        TYPE ( TRIMSQP_userdata_type ), OPTIONAL :: userdata
        
      END SUBROUTINE funFC_reverse
      
   END INTERFACE

   INTERFACE
      
      SUBROUTINE funG_reverse(G, X, userdata)
        
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        
        TYPE :: TRIMSQP_userdata_type
           INTEGER, ALLOCATABLE, DIMENSION( : ) :: integer_userdata
           REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: real_userdata
           CHARACTER ( LEN = 15 ), ALLOCATABLE, DIMENSION( : ) :: character_userdata
           LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical_userdata
        END TYPE TRIMSQP_userdata_type
        
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
        REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: G
        TYPE ( TRIMSQP_userdata_type ), INTENT( INOUT ), OPTIONAL :: userdata
        
      END SUBROUTINE funG_reverse
      
   END INTERFACE
   
   INTERFACE
      
      SUBROUTINE funJ_reverse(J_val, X, userdata)
        
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        
        TYPE :: TRIMSQP_userdata_type
           INTEGER, ALLOCATABLE, DIMENSION( : ) :: integer_userdata
           REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: real_userdata
           CHARACTER ( LEN = 15 ), ALLOCATABLE, DIMENSION( : ) :: character_userdata
           LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical_userdata
        END TYPE TRIMSQP_userdata_type
        
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
        REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: J_val
        TYPE ( TRIMSQP_userdata_type ), INTENT( INOUT), OPTIONAL :: userdata
        
      END SUBROUTINE funJ_reverse
      
   END INTERFACE
   
   
   INTERFACE
      
      SUBROUTINE funH_reverse(H_val, X, Y, userdata)
        
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        
        TYPE :: TRIMSQP_userdata_type
           INTEGER, ALLOCATABLE, DIMENSION( : ) :: integer_userdata
           REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: real_userdata
           CHARACTER ( LEN = 15 ), ALLOCATABLE, DIMENSION( : ) :: character_userdata
           LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical_userdata
        END TYPE TRIMSQP_userdata_type
        
        REAL ( kind = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
        REAL ( kind = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: Y
        REAL ( kind = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( INOUT ) ::H_val
        TYPE ( TRIMSQP_userdata_type ), INTENT( INOUT ), OPTIONAL :: userdata
        
      END SUBROUTINE funH_reverse
      
   END INTERFACE
   

   ! ***********************************************************
   
   ! Set problem dimensions.
        
   nlp%n = 4;	n = nlp%n
   nlp%m = 3;	m = nlp%m


   ! Set sparse Jacobian/Hessian dimensions and storage type.

   nlp%H_ne  = 6;	H_ne = nlp%H_ne
   nlp%J_ne  = 7;	J_ne = nlp%J_ne

   nlp%J_type = 0  ! co-ordinate
   nlp%H_type = 0  ! co-ordinate

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

   CALL SPACE_resize_array( n, nlp%X_status, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_pointer( n, nlp%G, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( n, nlp%gL, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( m, nlp%CNAMES, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_pointer( m, nlp%C, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( m, nlp%C_u, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( m, nlp%C_l, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( m, nlp%C_status, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( m, nlp%Y, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( J_ne, nlp%J_row, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( J_ne, nlp%J_col, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_pointer( J_ne, nlp%J_val, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( H_ne, nlp%H_row, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( H_ne, nlp%H_col, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( H_ne, nlp%H_val, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( m, nlp%EQUATION, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( m, nlp%LINEAR, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   ! userdata

   CALL SPACE_resize_array( 1, userdata%real_userdata, status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( 1, userdata%integer_userdata , status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( 1, userdata%character_userdata , status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   CALL SPACE_resize_array( 1, userdata%logical_userdata , status, alloc_status )
   IF ( status /= 0 ) GO TO 990

   ! Set remaining problem data.

   nlp%PNAME    = 'Toy1'
   nlp%VNAMES   = (/ 'X1', 'X2', 'X3', 'X4' /)
   nlp%CNAMES   = (/ 'C1', 'C2', 'C2' /)
   nlp%C_l      = (/ two, four, zero /)
   nlp%C_u      = (/ two, four, infinity /)
   nlp%X_l      = (/ -infinity, -infinity, -infinity, -infinity /)
   nlp%X_u      = (/  infinity,  infinity,  infinity, infinity /)
   nlp%X        = (/  0.1_wp, 0.125_wp, 0.666666_wp, 0.142857_wp /)
   nlp%Y        = (/ zero, zero, zero /)
   nlp%Z        = (/ zero, zero, zero, zero /)
   nlp%X_status = (/ 0, 0, 0, 0 /)
   nlp%C_status = (/ 0, 0, 0 /)
   nlp%EQUATION = (/ .FALSE., .FALSE., .FALSE. /)
   nlp%LINEAR   = (/ .FALSE., .FALSE., .FALSE. /)

   nlp%J_col = (/ 1, 2, 3, 2, 4, 1, 2 /) 
   nlp%J_row = (/ 1, 1, 1, 2, 2, 3, 3 /)
   nlp%H_col = (/ 1, 1, 2, 1, 2, 3 /)
   nlp%H_row = (/ 1, 2, 2, 3, 3, 3 /)


!  List of components of nlp not used in this program.

   ! nlp%H_ptr:  'Co-ordinate' used -> not needed (allocated).
   ! nlp%H_val:  Allocated, but not set.
   ! nlp%J_ptr:  'Co-ordinate' used -> not needed (allocated). 
   ! nlp%J_val:  Allocated, but not set.


!  Initialize data structure and control structure.

   CALL TRIMSQP_initialize( data, control )

   WRITE(*,*)
   WRITE(*,*) '** TRIMSQP_initialize: SUCCESSFUL **'
   WRITE(*,*)

! Check for spec file.

   INQUIRE( FILE = spec_name, EXIST = is_specfile )

   IF ( is_specfile ) THEN

      OPEN( spec_device, FILE = spec_name, FORM = 'FORMATTED', STATUS = 'OLD' )

      CALL TRIMSQP_read_specfile( control, spec_device )  ! OPTIONAL: alt_spec_name

      WRITE(*,*)
      WRITE(*,*) '** TRIMSQP_read_specfile: SUCCESSFUL **'
      WRITE(*,*)

   END IF


! Set up for potential recursive calls.

   solved = .FALSE.
   num_recursive_calls = -1;     max_num_recursive_calls = 100

   DO WHILE( (.NOT.solved) .OR. (num_recursive_calls >= max_num_recursive_calls) )

      !CALL TRIMSQP_solve( nlp, control, inform, data, funFC, funG, funJ, &
      !               funGJ, funH, funJv, funHv, userdata )

      CALL TRIMSQP_solve( nlp, control, inform, data,       &
                          funFC=funFC, funGJ=funGJ, funH=funH, userdata=userdata )

      WRITE(*,*) 'TRIMSQP_solve : exit with status = ', inform%status

      SELECT CASE ( inform%status )
      CASE (-1)
         WRITE(*,*) 'Which means : an error has occured within TRIMSQP_solve.'
      CASE (0)
         WRITE(*,*) 'Which means : ** Solution Found **'
         solved = .TRUE.
      CASE (1)
         WRITE(*,*) 'Which means : BUG, forgot to reset status upon initial entry'
      CASE (2)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : f and c are needed.'
         CALL funFC_reverse( nlp%f, nlp%C, nlp%X, userdata = userdata )
      CASE (3)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : g is needed.'
         CALL funG_reverse( nlp%G, nlp%X )
      CASE (4)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : J is needed.'
         CALL funJ_reverse( nlp%J_val, nlp%X )
      CASE (5)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : g and J are needed.'
         CALL funG_reverse( nlp%G, nlp%X )
         CALL funJ_reverse( nlp%J_val, nlp%X )
      CASE (6)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : H is needed.'
         CALL funH_reverse( nlp%H_val, nlp%X, nlp%Y )
      CASE (7)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : u <- J*v+u is needed.'
         !transpose = .FALSE.
         !CALL fun_JV_reverse( data%u, data%v, nlp%X, transpose )
      CASE (8)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : u <- J^T*v+u is needed.'
         !transpose = .TRUE.
         !CALL fun_Jv_reverse( data%u, data%v, nlp%X, transpose )         
      CASE (9)
         WRITE(*,*) 'Which means : REVERSE COMMUNICATION : u <- H*v+u is needed.'
         !CALL fun_Hv_reverse( data%u, data%v, nlp%X )
      CASE DEFAULT
         WRITE(*,*) 'Which means : an in-appropriate value of status has been returned.'
      END SELECT

      num_recursive_calls = num_recursive_calls + 1
      
   END DO


!  Termination

   CALL TRIMSQP_terminate( data, control, inform, userdata )

   WRITE(*,*)
   WRITE(*,*) 'TRIMSQP_terminate : SUCCESSFUL'
   WRITE(*,*)

!  Deallocate everything from nlpt that has been allocated.

   CALL NLPT_cleanup( nlp )

   WRITE(*,*)
   WRITE(*,*) 'NLPT_cleanup : SUCCESSFUL'
   WRITE(*,*)

!  Deallocate userdata.
   CALL SPACE_dealloc_array( userdata%real_userdata, status, alloc_status )
   IF ( status /= 0 ) GO TO 991

   CALL SPACE_dealloc_array( userdata%integer_userdata, status, alloc_status )
   IF ( status /= 0 ) GO TO 991

   CALL SPACE_dealloc_array( userdata%logical_userdata, status, alloc_status )
   IF ( status /= 0 ) GO TO 991

   CALL SPACE_dealloc_array( userdata%character_userdata, status, alloc_status )
   IF ( status /= 0 ) GO TO 991
  
   WRITE(*,*)
   WRITE(*,*) 'deallocation of userdata : SUCCESSFUL'
   WRITE(*,*)

   STOP

! *****************************************************************

! Abnormal return.

 990 CONTINUE
     WRITE( out, "( ' TOY1: allocation error ', I0, ' status ', I0 )" )   &
            status, alloc_status
     STOP

 991 CONTINUE
     WRITE( out, "( ' TOY1: deallocation error ', I0, ' status ', I0 )" )   &
            status, alloc_status
     STOP



   END PROGRAM TOY1


! *****************************************************************
!
!                BEGIN EXTERNAL SUBROUTINES
!
! *****************************************************************


!                  *********************************************
!                     Explicit functions - given to TRIMSQP_sove
!                  *********************************************
   SUBROUTINE funFC( F, C, X, userdata )

   USE GALAHAD_TRIMSQP_double  

   IMPLICIT NONE

    ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters.

   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp

   TYPE( TRIMSQP_userdata_type ), INTENT( INOUT ) :: userdata
   REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), INTENT( OUT ) :: F
   REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: C

   ! local variables
   
   REAL( KIND = wp ) :: X1, X2, X3, X4, C1, C2, C3 
   
   X1 = X(1)
   X2 = X(2)
   X3 = X(3)
   X4 = X(4)

   F = ( X1 + X2 + X3 )**2 + three*X3 + five*X4

   C1 = X1**2 + X2**2 + X3
   C2 = X2**4 + X4
   C3 = two*X1 + four*X2

   C(1) = C1
   C(2) = C2
   C(3) = C3
   
   RETURN

   END SUBROUTINE funFC

!-*-*-*-*-*-*-*-*-   f u n G   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funG( G, X, userdata )

   USE GALAHAD_TRIMSQP_double 

   IMPLICIT NONE

   ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters

   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
 
   TYPE( TRIMSQP_userdata_type ), INTENT(INOUT) :: userdata
   REAL ( KIND = wp), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
   REAL( KIND = wp ), POINTER, DIMENSION( : ) :: G

  ! local variables

   REAL( KIND = wp ) :: X1, X2, X3, X4, G1, G2, G3, G4
   REAL( KIND = wp ) :: J11, J12, J13, J22, J24, J31, J32 


   X1 = X(1)
   X2 = X(2)
   X3 = X(3)
   X4 = X(4)

   G1 = two * ( x1+ X2 + X3 )
   G2 = two * ( x1+ X2 + X3 )
   G3 = two * ( x1+ X2 + X3 ) + three
   G4 = five 

   G(1) = G1
   G(2) = G2
   G(3) = G3
   G(4) = G4

   RETURN

   END SUBROUTINE funG

!-*-*-*-*-*-*-*-*-   f u n J   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funJ( Jval, X, userdata )

   USE GALAHAD_TRIMSQP_double 

   IMPLICIT NONE

   ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters

   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
 
   TYPE( TRIMSQP_userdata_type ), INTENT(INOUT) :: userdata
   REAL ( KIND = wp), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: Jval

  ! local variables

   REAL( KIND = wp ) :: X1, X2, X3, X4, G1, G2, G3, G4
   REAL( KIND = wp ) :: J11, J12, J13, J22, J24, J31, J32 


   X1 = X(1)
   X2 = X(2)
   X3 = X(3)
   X4 = X(4)

   J11 = two * X1
   J12 = two * X2
   J13 = one
   J22 = four * X2**3
   J24 = one
   J31 = two
   J32 = four

   Jval(1) = J11
   Jval(2) = J12
   Jval(3) = J13
   Jval(4) = J22
   Jval(5) = J24
   Jval(6) = J31
   Jval(7) = J32

   RETURN

   END SUBROUTINE funJ


!-*-*-*-*-*-*-*-*-   f u n G J   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funGJ( G, Jval, X, userdata )

   USE GALAHAD_TRIMSQP_double 

   IMPLICIT NONE

   ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters

   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
 
   TYPE( TRIMSQP_userdata_type ), INTENT(INOUT) :: userdata
   REAL ( KIND = wp), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
   REAL( KIND = wp ), POINTER, DIMENSION( : ) :: G
   REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: Jval

  ! local variables

   REAL( KIND = wp ) :: X1, X2, X3, X4, G1, G2, G3, G4
   REAL( KIND = wp ) :: J11, J12, J13, J22, J24, J31, J32 


   X1 = X(1)
   X2 = X(2)
   X3 = X(3)
   X4 = X(4)

   G1 = two * ( x1+ X2 + X3 )
   G2 = two * ( x1+ X2 + X3 )
   G3 = two * ( x1+ X2 + X3 ) + three
   G4 = five 

   G(1) = G1
   G(2) = G2
   G(3) = G3
   G(4) = G4

   J11 = two * X1
   J12 = two * X2
   J13 = one
   J22 = four * X2**3
   J24 = one
   J31 = two
   J32 = four

   Jval(1) = J11
   Jval(2) = J12
   Jval(3) = J13
   Jval(4) = J22
   Jval(5) = J24
   Jval(6) = J31
   Jval(7) = J32

   RETURN

   END SUBROUTINE funGJ

!-*-*-*-*-*-*-*-*-   f u n H   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funH( Hval, X, Y, userdata )

   USE GALAHAD_TRIMSQP_double 
 
   IMPLICIT NONE

   ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters

   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
   REAL ( KIND = wp ), PARAMETER ::  twelve = 12.0_wp   

   TYPE( TRIMSQP_userdata_type ), INTENT( INOUT ) :: userdata
   REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( INOUT ) :: Hval

 ! local variables

   REAL( KIND = wp ) :: X1, X2, Y1, Y2
   REAL( KIND = wp ) :: H11, H21, H22, H31, H32, H33
   
   !X1 = X(1)
   X2 = X(2)
   !X3 = X(3)
   !X4 = X(4)

   Y1 = Y(1)
   Y2 = Y(2)
   !Y3 = Y(3)

   H11 = TWO + Y1
   H21 = TWO
   H22 = TWO + TWO*Y1 + twelve*Y2*X2
   H31 = TWO
   H32 = TWO
   H33 = TWO

   Hval(1) = H11
   Hval(2) = H21
   Hval(3) = H22
   Hval(4) = H31
   Hval(5) = H32
   Hval(6) = H33

   RETURN

   END SUBROUTINE funH

!                  *************************************************
!                     Explicit functions - for reverse communication
!                  *************************************************

!-*-*-*-*-*-*-*-*-   f u n F C _ r e v e r s e   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funFC_reverse( F, C, X, userdata )

   USE GALAHAD_TRIMSQP_double  

   IMPLICIT NONE

    ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters.

   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp

!  TYPE( TRIMSQP_userdata_type ), INTENT( INOUT ) :: userdata
   TYPE( TRIMSQP_userdata_type ), OPTIONAL :: userdata
   REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), INTENT( OUT ) :: F
   REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: C

   ! local variables
   
   REAL( KIND = wp ) :: X1, X2, X3, X4, C1, C2, C3 
   
   X1 = X(1)
   X2 = X(2)
   X3 = X(3)
   X4 = X(4)

   F = ( X1 + X2 + X3 )**2 + three*X3 + five*X4

   C1 = X1**2 + X2**2 + X3
   C2 = X2**4 + X4
   C3 = two*X1 + four*X2

   C(1) = C1
   C(2) = C2
   C(3) = C3
   
   RETURN

   END SUBROUTINE funFC_reverse

!-*-*-*-*-*-*-*-*-   f u n G _ r e v e r s e   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funG_reverse( G, X, userdata )

   USE GALAHAD_TRIMSQP_double 

   IMPLICIT NONE

   ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters

   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
 
   TYPE( TRIMSQP_userdata_type ), INTENT(INOUT), OPTIONAL :: userdata
   REAL ( KIND = wp), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
   REAL( KIND = wp ), POINTER, DIMENSION( : ) :: G

  ! local variables

   REAL( KIND = wp ) :: X1, X2, X3, X4, G1, G2, G3, G4
   REAL( KIND = wp ) :: J11, J12, J13, J22, J24, J31, J32 


   X1 = X(1)
   X2 = X(2)
   X3 = X(3)
   X4 = X(4)

   G1 = two * ( x1+ X2 + X3 )
   G2 = two * ( x1+ X2 + X3 )
   G3 = two * ( x1+ X2 + X3 ) + three
   G4 = five 

   G(1) = G1
   G(2) = G2
   G(3) = G3
   G(4) = G4

   RETURN

   END SUBROUTINE funG_reverse

!-*-*-*-*-*-*-*-*-   f u n J _ r e v e r s e   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funJ_reverse( Jval, X, userdata )

   USE GALAHAD_TRIMSQP_double 

   IMPLICIT NONE

   ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters

   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
 
   TYPE( TRIMSQP_userdata_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = wp), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: Jval

  ! local variables

   REAL( KIND = wp ) :: X1, X2, X3, X4, G1, G2, G3, G4
   REAL( KIND = wp ) :: J11, J12, J13, J22, J24, J31, J32 


   X1 = X(1)
   X2 = X(2)
   X3 = X(3)
   X4 = X(4)

   J11 = two * X1
   J12 = two * X2
   J13 = one
   J22 = four * X2**3
   J24 = one
   J31 = two
   J32 = four

   Jval(1) = J11
   Jval(2) = J12
   Jval(3) = J13
   Jval(4) = J22
   Jval(5) = J24
   Jval(6) = J31
   Jval(7) = J32

   RETURN

   END SUBROUTINE funJ_reverse

!-*-*-*-*-*-*-*-*-   f u n H _ r e v e r s e   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

   SUBROUTINE funH_reverse( Hval, X, Y, userdata )

   USE GALAHAD_TRIMSQP_double 
 
   IMPLICIT NONE

   ! Set precision

   INTEGER, PARAMETER :: wp = KIND(  1.0D+0 )

   ! Set parameters

   REAL ( KIND = wp ), PARAMETER ::  one    = 1.0_wp
   REAL ( KIND = wp ), PARAMETER ::  two    = 2.0_wp
   REAL ( KIND = wp ), PARAMETER ::  three  = 3.0_wp
   REAL ( KIND = wp ), PARAMETER ::  four   = 4.0_wp
   REAL ( KIND = wp ), PARAMETER ::  five   = 5.0_wp
   REAL ( KIND = wp ), PARAMETER ::  twelve = 12.0_wp   

   TYPE( TRIMSQP_userdata_type ), INTENT( INOUT ), OPTIONAL :: userdata
   REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( IN ) :: X, Y
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ), INTENT( INOUT ) :: Hval

 ! local variables

   REAL( KIND = wp ) :: X1, X2, Y1, Y2
   REAL( KIND = wp ) :: H11, H21, H22, H31, H32, H33
   
   !X1 = X(1)
   X2 = X(2)
   !X3 = X(3)
   !X4 = X(4)

   Y1 = Y(1)
   Y2 = Y(2)
   !Y3 = Y(3)

   H11 = TWO + Y1
   H21 = TWO
   H22 = TWO + TWO*Y1 + twelve*Y2*X2
   H31 = TWO
   H32 = TWO
   H33 = TWO

   Hval(1) = H11
   Hval(2) = H21
   Hval(3) = H22
   Hval(4) = H31
   Hval(5) = H32
   Hval(6) = H33

   RETURN

   END SUBROUTINE funH_reverse
