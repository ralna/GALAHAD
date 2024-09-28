! THIS VERSION: GALAHAD 5.1 - 2024-09-10 AT 14:00 GMT.
   PROGRAM GALAHAD_SSLS_EXAMPLE
   USE GALAHAD_SSLS_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SMT_type ) :: H, A, C
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
   TYPE ( SSLS_data_type ) :: data
   TYPE ( SSLS_control_type ) :: control
   TYPE ( SSLS_inform_type ) :: inform
   INTEGER :: s
   INTEGER :: n = 3, m = 2, h_ne = 4, a_ne = 4, c_ne = 1
! start problem data
   ALLOCATE( SOL( n + m ) )
   SOL( 1 : n ) = (/ 7.0_wp, 4.0_wp, 8.0_wp /)  ! RHS a
   SOL( n + 1 : n + m ) = (/ 2.0_wp, 1.0_wp /)  ! RHS b
! sparse co-ordinate storage format
   CALL SMT_put( H%type, 'COORDINATE', s )  ! Specify co-ordinate
   CALL SMT_put( A%type, 'COORDINATE', s )  ! storage for H, A and C
   CALL SMT_put( C%type, 'COORDINATE', s )
   ALLOCATE( H%val( h_ne ), H%row( h_ne ), H%col( h_ne ) )
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   ALLOCATE( C%val( c_ne ), C%row( c_ne ), C%col( c_ne ) )
   H%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /) ! matrix H
   H%row = (/ 1, 2, 3, 3 /)                     ! NB lower triangle
   H%col = (/ 1, 2, 3, 1 /) ; H%ne = h_ne
   A%val = (/ 2.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! matrix A
   A%row = (/ 1, 1, 2, 2 /)
   A%col = (/ 1, 2, 2, 3 /) ; A%ne = a_ne
   C%val = (/ 1.0_wp /) ! matrix C
   C%row = (/ 2 /) ! NB lower triangle
   C%col = (/ 1 /) ; C%ne = c_ne
! problem data complete
   CALL SSLS_initialize( data, control, inform ) ! Initialize control parameters
   control%symmetric_linear_solver = "sytr "
! factorize matrix
   CALL SSLS_analyse( n, m, H, A, C, data, control, inform )
   CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
   IF ( inform%status < 0 ) THEN                 ! Unsuccessful call
     WRITE( 6, "( ' SSLS_analyse exit status = ', I0 )" ) inform%status
     STOP
   END IF
   CALL SSLS_factorize( n, m, H, A, C, data, control, inform )
   IF ( inform%status < 0 ) THEN                 ! Unsuccessful call
     WRITE( 6, "( ' SSLS_factorize exit status = ', I0 )" ) inform%status
     STOP
   END IF
! solve system
   CALL SSLS_solve( n, m, SOL, data, control, inform )
   IF ( inform%status == 0 ) THEN                ! Successful return
     WRITE( 6, "( ' SSLS: Solution = ', /, ( 5ES12.4 ) )" ) SOL
   ELSE                                          !  Error returns
     WRITE( 6, "( ' SSLS_solve exit status = ', I6 )" ) inform%status
   END IF
   CALL SSLS_terminate( data, control, inform )  ! delete internal workspace
   END PROGRAM GALAHAD_SSLS_EXAMPLE

