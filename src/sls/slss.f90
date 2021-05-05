   PROGRAM SLS_EXAMPLE   !  GALAHAD 3.3 - 05/05/2021 AT 16:00 GMT.
   USE GALAHAD_SLS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( SMT_type ) :: matrix
   TYPE ( SLS_data_type ) :: data
   TYPE ( SLS_control_type ) control
   TYPE ( SLS_inform_type ) :: inform
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
   CALL SLS_initialize( 'sils', data, control, inform )
! Analyse
   CALL SLS_analyse( matrix, data, control, inform )
   IF ( inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_analyse with status = ', inform%status
     STOP
   END IF
! Factorize
   CALL SLS_factorize( matrix, data, control, inform )
   IF ( inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_factorize with status = ', inform%status
     STOP
   END IF
! Solve using iterative refinement and ask for high relative accuracy
   X = B
   control%max_iterative_refinements = 1
   control%acceptable_residual_relative = 0.0_wp
   CALL SLS_solve( matrix, X, data, control, inform )
   IF ( inform%status == 0 ) WRITE( 6, '( A, /, ( 3F20.16 ) )' )               &
     ' Solution is', X
! Clean up
   CALL SLS_terminate( data, control, inform )
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP
   END PROGRAM SLS_EXAMPLE
