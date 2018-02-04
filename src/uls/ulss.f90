! THIS VERSION: GALAHAD 2.4 - 30/09/2009 AT 16:00 GMT.
   PROGRAM GALAHAD_ULS_example
   USE GALAHAD_ULS_DOUBLE
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER :: i, m, n, ne, s
   TYPE ( SMT_type ) :: matrix
   TYPE ( ULS_data_type ) :: data
   TYPE ( ULS_control_type ) control
   TYPE ( ULS_inform_type ) :: inform
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROWS, COLS
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, X
!  Read matrix order and number of entries
   READ( 5, * ) m, n, ne
   matrix%m = m ; matrix%n = n ; matrix%ne = ne
! Allocate arrays of appropriate sizes
   ALLOCATE( matrix%val( ne ),  matrix%row( ne ),  matrix%col( ne ) )
   ALLOCATE( B( m ), X( n ), ROWS( m ), COLS( n ) )
! Read matrix
   READ( 5, * ) ( matrix%row( i ), matrix%col( i ), matrix%val( i ), i = 1, ne )
   CALL SMT_put( matrix%type, 'COORDINATE', s )     ! Specify co-ordinate
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
! Read right-hand side and solve system
   READ( 5, * ) B
   CALL ULS_solve( matrix, B, X, data, control, inform, .FALSE. )
   IF ( inform%status == 0 ) WRITE( 6, '( A, /,( 6ES11.3 ) )' )                &
     ' Solution of set of equations without refinement is', X
! Clean up
   CALL ULS_terminate( data, control, inform )
   DEALLOCATE( matrix%val, matrix%row, matrix%col, X, B, ROWS, COLS )
   STOP
   END PROGRAM GALAHAD_ULS_example
