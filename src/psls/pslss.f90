   PROGRAM PSLS_EXAMPLE   !  GALAHAD 4.0 - 2022-01-24 AT 09:15 GMT.
   USE GALAHAD_PSLS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( SMT_type ) :: matrix
   TYPE ( PSLS_data_type ) :: data
   TYPE ( PSLS_control_type ) control
   TYPE ( PSLS_inform_type ) :: inform
   INTEGER, PARAMETER :: n = 5
   INTEGER, PARAMETER :: ne = 7
   REAL ( KIND = wp ) :: X( n )
   INTEGER :: s
! allocate and set lower triangle of matrix in co-ordinate form
   CALL SMT_put( matrix%type, 'COORDINATE', s )
   matrix%n = n ; matrix%ne = ne
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row = (/ 1, 2, 3, 3, 4, 5, 5 /)
   matrix%col = (/ 1, 1, 2, 3, 3, 2, 5 /)
   matrix%val = (/ 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp, 6.0_wp, 1.0_wp /)
! problem setup complete
! specify the solver used by SLS (in this case sils)
   CALL PSLS_initialize( data, control, inform )
   control%preconditioner = 2  ! band preconditioner
   control%semi_bandwidth = 1  ! semi-bandwidth of one
!  control%definite_linear_solver = 'sils'
! form and factorize the preconditioner, P
   CALL PSLS_form_and_factorize( matrix, data, control, inform )
   IF ( inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of PSLS_form_and_factorize with status = ', inform%status
     STOP
   END IF
! use the factors to solve P x = b, with b input in x
   X( : n ) = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
   CALL PSLS_apply( X, data, control, inform )
   IF ( inform%status == 0 ) THEN
     WRITE( 6, "( ' PSLS - Preconditioned solution is ', 5F6.2 )" ) X
   ELSE
     WRITE( 6, "( ' PSLS - exit status = ', I0 )" ) inform%status
   END IF
! clean up
   CALL PSLS_terminate( data, control, inform )
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP

   END PROGRAM PSLS_EXAMPLE
