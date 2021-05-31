! THIS VERSION: GALAHAD 3.3 - 29/05/2021 AT 11:30 GMT.
   PROGRAM GALAHAD_GLS_test  ! further work needed!!
   USE GALAHAD_GLS_DOUBLE
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) 
   INTEGER :: i, info, rank
   INTEGER, PARAMETER :: m = 3
   INTEGER, PARAMETER :: n = 3
   INTEGER, PARAMETER :: ne = 7
   TYPE ( SMT_TYPE ) :: matrix
   TYPE ( GLS_CONTROL ) :: control
   TYPE ( GLS_AINFO ) :: ainfo
   TYPE ( GLS_FINFO ) :: finfo
   TYPE ( GLS_SINFO ) :: sinfo
   TYPE ( GLS_FACTORS ) :: factors
   INTEGER :: ROWS( m ), COLS( n )
   REAL ( KIND = wp ) :: X( n ), B( m )
! record matrix order and number of entries
   matrix%m = m ; matrix%n = n ; matrix%ne = ne
! allocate and set matrix
!  CALL SMT_put( matrix%type, 'COORDINATE', info )   ! Specify co-ordinate
   ALLOCATE( matrix%val( ne ),  matrix%row( ne ),  matrix%col( ne ) )
   matrix%row( : ne ) = (/ 1, 2, 3, 2, 1, 3, 2 /)
   matrix%col( : ne ) = (/ 1, 3, 3, 1, 2, 2, 2 /)
   matrix%val( : ne ) = (/ 11.0_wp, 23.0_wp, 33.0_wp, 21.0_wp, 12.0_wp,        &
                           32.0_wp, 22.0_wp /)
! initialize structures
   CALL GLS_initialize( factors, control )
! analyse and factorize
   CALL GLS_analyse( matrix, factors, control, ainfo, finfo )
   IF ( AINFO%flag < 0 ) THEN
     WRITE( 6, "( ' Failure of GLS_ANALYSE with AINFO%flag = ',  I0 )" )       &
       AINFO%flag
     STOP
   END IF
! write row and column reorderings
   CALL GLS_special_rows_and_cols( factors, rank, ROWS, COLS, info )
   WRITE( 6, "( ' row orderings:', /, ( 10I5 ) )" ) ROWS( : rank )
   WRITE( 6, "( ' column orderings:', /, ( 10I5 ) )" ) COLS( : rank )
! set right-hand side and solve system
   B = (/ 23.0_wp, 66.0_wp, 65.0_wp /)
   CALL GLS_solve( matrix, factors, B, X, control, sinfo )
   IF ( SINFO%flag == 0 ) WRITE( 6,                                            &
     "( ' Solution of set of equations without refinement is', /,              &
        & ( 6ES11.3 ) )" ) X
! now solve the transposed system
   B = (/ 32.0_wp, 66.0_wp, 56.0_wp /)
   CALL GLS_solve( matrix, factors, B, X, control, sinfo, trans = 1 )
   IF ( SINFO%flag == 0 ) WRITE( 6,                                            &
     "( ' Solution of set of transposed equations without refinement is', /,   &
        & ( 6ES11.3 ) )" ) X
! clean up
   CALL GLS_FINALIZE( factors, control, info )
   DEALLOCATE( matrix%val, matrix%row, matrix%col )
   STOP
   END PROGRAM GALAHAD_GLS_test
