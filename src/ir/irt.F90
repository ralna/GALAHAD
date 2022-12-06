! THIS VERSION: GALAHAD 3.3 - 29/05/2021 AT 11:30 GMT.
   PROGRAM GALAHAD_IR_TEST  !  further work needed!!
   USE GALAHAD_IR_double                           ! double precision version
   USE GALAHAD_SMT_double
   USE GALAHAD_SLS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )      ! set precision
   TYPE ( SMT_type ) :: matrix
   TYPE ( SLS_data_type ) :: SLS_data
   TYPE ( SLS_control_type ) SLS_control
   TYPE ( SLS_inform_type ) :: SLS_inform
   TYPE ( IR_data_type ) :: data
   TYPE ( IR_control_type ) :: control
   TYPE ( IR_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 5
   INTEGER, PARAMETER :: ne = 7
   REAL ( KIND = wp ) :: B( n ), X( n )
   INTEGER :: i, s
! Read matrix order and number of entries
   matrix%n = n
   matrix%ne = ne
! Allocate and set matrix
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row( : ne ) = (/ 1, 1, 2, 2, 3, 3, 5 /)
   matrix%col( : ne ) = (/ 1, 2, 3, 5, 3, 4, 5 /)
   matrix%val( : ne ) = (/ 2.0_wp, 3.0_wp, 4.0_wp, 6.0_wp, 1.0_wp,             &
                           5.0_wp, 1.0_wp /)
   CALL SMT_put( matrix%type, 'COORDINATE', s )     ! Specify co-ordinate
! Set right-hand side
   B( : n ) = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
! Specify the solver (in this case sils)
   CALL SLS_initialize( 'sils', SLS_data, SLS_control, SLS_inform )
! Analyse
   CALL SLS_analyse( matrix, SLS_data, SLS_control, SLS_inform )
   IF ( SLS_inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_analyse with status = ', SLS_inform%status
     STOP
   END IF
! Factorize
   CALL SLS_factorize( matrix, SLS_data, SLS_control, SLS_inform )
   IF ( SLS_inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_factorize with status = ', SLS_inform%status
     STOP
   END IF
! solve using iterative refinement
   CALL IR_initialize( data, control, inform )    ! initialize IR structures
   control%itref_max = 2                          ! perform 2 iterations
   control%acceptable_residual_relative = 0.1 * EPSILON( 1.0D0 ) ! high accuracy
   X = B
   CALL IR_SOLVE( matrix, X, data, SLS_data, control, SLS_control, inform,     &
                  SLS_inform )
   IF ( inform%status == 0 ) THEN                 ! check for errors
     WRITE( 6, '( A, /, ( 5F10.6 ) )' ) ' Solution after refinement is', X
   ELSE
    WRITE( 6,'( A, I2 )' ) ' Failure of IR_solve with status = ', inform%status
   END IF
   CALL IR_terminate( data, control, inform )     ! delete internal workspace
   CALL SLS_terminate( SLS_data, SLS_control, SLS_inform )
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP
   END PROGRAM GALAHAD_IR_TEST
