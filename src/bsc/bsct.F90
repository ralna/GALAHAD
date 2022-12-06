! THIS VERSION: GALAHAD 2.6 - 22/10/2013 AT 08:00 GMT.
   PROGRAM GALAHAD_BSC_TEST
   USE GALAHAD_BSC_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( BSC_data_type ) :: data
   TYPE ( BSC_control_type ) :: control        
   TYPE ( BSC_inform_type ) :: inform
   INTEGER, PARAMETER :: m = 3, n = 4, a_ne = 6
   TYPE ( SMT_type ) :: A, S
   REAL ( KIND = wp ), DIMENSION( n ) :: D
   INTEGER :: i, j
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

!  error exit tests

   WRITE( 6, "( ' error exit tests ', / ) " )
   CALL BSC_form( -1, n, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   CALL BSC_form( m, 0, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   CALL SMT_put( A%type, 'BOORDINATE', i )     ! storage for A
   CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   CALL SMT_put( A%type, 'COORDINATE', i )     ! storage for A
   control%max_col = 1
   CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
   WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status

!  tests for normal entry

   control%max_col = - 1
   WRITE( 6, "( /, ' normal exit tests ', / ) " )

   DO j = 1, 3
!  sparse co-ordinate storage format
     IF ( j == 1 ) THEN
       DEALLOCATE( A%row, A%col, A%val )
       CALL SMT_put( A%type, 'COORDINATE', i )
       ALLOCATE( A%val( a_ne ), A%row( a_ne ), A%col( a_ne ) )
       A%ne = a_ne
       A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       A%row = (/ 1, 1, 2, 2, 3, 3 /)
       A%col = (/ 1, 2, 3, 4, 1, 4 /)
! sparse row-wise storage format
     ELSE IF ( j == 2 ) THEN
       DEALLOCATE( A%row, A%col, A%val )
       CALL SMT_put( A%type, 'SPARSE_BY_ROWS', i )
       ALLOCATE( A%val( a_ne ), A%col( a_ne ), A%ptr( m + 1 ) )
       A%val = (/ 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp /)
       A%col = (/ 1, 2, 3, 4, 1, 4 /)
       A%ptr = (/ 1, 3, 5, 7 /)
! dense storage format
     ELSE IF ( j == 3 ) THEN
       DEALLOCATE( A%ptr, A%col, A%val )
       CALL SMT_put( A%type, 'DENSE', i )
       ALLOCATE( A%val( n * m ) )
       A%val = (/ 1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,              &
                  1.0_wp, 1.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 1.0_wp /)
     END IF
     control%new_a = 2
     CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
     control%new_a = 1
     CALL BSC_form( m, n, A, S, data, control, inform ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
     control%new_a = 2
     CALL BSC_form( m, n, A, S, data, control, inform, D = D ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
     control%new_a = 1
     CALL BSC_form( m, n, A, S, data, control, inform, D = D ) ! Form S
     WRITE( 6, "( ' BSC_solve exit status = ', I6 ) " ) inform%status
   END DO
   DEALLOCATE( A%val )

   CALL BSC_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_BSC_TEST
