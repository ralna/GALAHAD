   PROGRAM SLS_EXAMPLE   !  GALAHAD 4.1 - 2022-11-27 AT 15:15 GMT.
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
   INTEGER :: s
! allocate and set lower triangle of matrix in co-ordinate form
   CALL SMT_put( matrix%type, 'COORDINATE', s )
   matrix%n = n ; matrix%ne = ne
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row( : ne ) = (/ 1, 2, 3, 3, 4, 5, 5 /)
   matrix%col( : ne ) = (/ 1, 1, 2, 3, 3, 2, 5 /)
   matrix%val( : ne ) = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp,             &
                           6.0_wp, 1.0_wp /)
! problem setup complete
! set right-hand side
   B( : n ) = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
! specify the solver (in this case ssids)
   CALL SLS_initialize( 'ssids', data, control, inform, check = .TRUE. )
   WRITE( 6, "( ' solver ', A, ' used' )" ) TRIM( inform%solver )
! analyse
   CALL SLS_analyse( matrix, data, control, inform )
   IF ( inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_analyse with status = ', inform%status
     STOP
   END IF
! factorize
   CALL SLS_factorize( matrix, data, control, inform )
   IF ( inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SLS_factorize with status = ', inform%status
     STOP
   END IF
! solve using iterative refinement and ask for high relative accuracy
   X = B
   control%max_iterative_refinements = 1
   control%acceptable_residual_relative = 0.0_wp
   CALL SLS_solve( matrix, X, data, control, inform )
   IF ( inform%status == 0 ) WRITE( 6, '( A, ( 5F5.2 ) )' )                    &
     ' Solution is', X
! clean up
   CALL SLS_terminate( data, control, inform )
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP
   END PROGRAM SLS_EXAMPLE
