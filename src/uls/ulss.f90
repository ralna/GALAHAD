   PROGRAM GALAHAD_ULS_example  ! GALAHAD 3.3 - 05/05/2021 AT 16:00 GMT.
   USE GALAHAD_ULS_DOUBLE
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER :: i, info
   INTEGER, PARAMETER :: m = 3
   INTEGER, PARAMETER :: n = 3
   INTEGER, PARAMETER :: ne = 7
   TYPE ( SMT_type ) :: matrix
   TYPE ( ULS_data_type ) :: data
   TYPE ( ULS_control_type ) control
   TYPE ( ULS_inform_type ) :: inform
   INTEGER :: ROWS( m ), COLS( n )
   REAL ( KIND = wp ) :: X( n ), B( m )
!  Record matrix order and number of entries
   matrix%m = m ; matrix%n = n ; matrix%ne = ne
! Allocate and set matrix
   CALL SMT_put( matrix%type, 'COORDINATE', info )   ! Specify co-ordinate
   ALLOCATE( matrix%val( ne ),  matrix%row( ne ),  matrix%col( ne ) )
   matrix%row( : ne ) = (/ 1, 2, 3, 2, 1, 3, 2 /)
   matrix%col( : ne ) = (/ 1, 3, 3, 1, 2, 2, 2 /)
   matrix%val( : ne ) = (/ 11.0_wp, 23.0_wp, 33.0_wp, 21.0_wp, 12.0_wp,        &
                           32.0_wp, 22.0_wp /)
! Specify the solver (in this case gls)
   CALL ULS_initialize( 'gls', data, control, inform )
! Factorize the matrix
   CALL ULS_factorize( matrix, data, control, inform )
   IF ( inform%status < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
       ' Failure of ULS_factorize with inform%status = ',  inform%status
     STOP
   END IF
! Write row and column reorderings
   CALL ULS_enquire( data, inform, ROWS, COLS )
   WRITE( 6, "( A, /, ( 10I5 ) )" ) ' row orderings:', ROWS( : inform%rank )
   WRITE( 6, "( A, /, ( 10I5 ) )" ) ' column orderings:', COLS( : inform%rank )
! set right-hand side and solve system
   B = (/ 23.0_wp, 66.0_wp, 65.0_wp /)
   CALL ULS_solve( matrix, B, X, data, control, inform, .FALSE. )
   IF ( inform%status == 0 ) WRITE( 6, '( A, /,( 6ES11.3 ) )' )                &
     ' Solution of set of equations without refinement is', X
! Clean up
   CALL ULS_terminate( data, control, inform )
   DEALLOCATE( matrix%val, matrix%row, matrix%col )
   STOP
   END PROGRAM GALAHAD_ULS_example
