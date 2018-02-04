! THIS VERSION: GALAHAD 2.6 - 21/10/2013 AT 13:00 GMT.
   PROGRAM GALAHAD_BSC_EXAMPLE
   USE GALAHAD_BSC_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( BSC_data_type ) :: data
   TYPE ( BSC_control_type ) :: control        
   TYPE ( BSC_inform_type ) :: inform
   INTEGER, PARAMETER :: m = 3, n = 4, a_ne = 6
   TYPE ( SMT_type ) :: A, S
   REAL ( KIND = wp ), DIMENSION( n ) :: D
   INTEGER :: i
   D( 1 : n ) = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp /)
!  sparse co-ordinate storage format
   CALL SMT_put( A%type, 'COORDINATE', i )     ! storage for A
   ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
   A%ne = a_ne
   A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /) ! Jacobian A
   A%row = (/ 1, 1, 2, 2, 3, 3 /)
   A%col = (/ 1, 2, 3, 4, 1, 4 /)
! problem data complete
   CALL BSC_initialize( data, control, inform ) ! Initialize control parameters
   CALL BSC_form( m, n, A, S, data, control, inform, D = D ) ! Form S
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' S:', /, ( ' row ', I2, ', column ', I2,                    &
    &  ', value = ', F4.1 ) )" )                                               &
       ( S%row( i ), S%col( i ), S%val( i ),  i = 1, S%ne )
   ELSE                                         !  Error returns
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL BSC_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_BSC_EXAMPLE
