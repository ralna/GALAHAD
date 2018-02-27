! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
! Specimen test example for HSL_MA48
   PROGRAM GALAHAD_MA48_example
   USE HSL_MA48_SINGLE
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0E+0 )
   INTEGER :: i, info, m, n, ne, rank
   TYPE ( ZD01_TYPE ) :: MATRIX
   TYPE ( MA48_CONTROL ) :: CONTROL
   TYPE ( MA48_AINFO ) :: AINFO
   TYPE ( MA48_FINFO ) :: FINFO
   TYPE ( MA48_SINFO ) :: SINFO
   TYPE ( MA48_FACTORS ) :: FACTORS
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROWS, COLS
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, X

!  Read matrix order and number of entries

   READ( 5, * ) m, n, ne
   MATRIX%m = m ; MATRIX%n = n ; MATRIX%ne = ne

!  Allocate arrays of appropriate sizes

   ALLOCATE( MATRIX%VAL( ne ),  MATRIX%ROW( ne ),  MATRIX%COL( ne ) )
   ALLOCATE( B( m ), X( n ), ROWS( m ), COLS( n ) )

!  Read matrix

   READ( 5, * ) ( MATRIX%ROW( i ), MATRIX%COL( i ), MATRIX%VAL( i ), i = 1, ne )

!  Initialize the structures

   CALL MA48_initialize( FACTORS, CONTROL )

!  Analyse and factorize

   CALL MA48_analyse( MATRIX, FACTORS, CONTROL, AINFO, FINFO )
   IF ( AINFO%flag < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
       ' Failure of MA48_ANALYSE with AINFO%flag = ',  AINFO%flag
     STOP
   END IF

!  Write row and column reorderings

   CALL MA48_special_rows_and_cols( FACTORS, rank, ROWS, COLS, info )
   WRITE( 6, "( A, /, ( 10I5 ) )" ) ' row orderings:', ROWS( : rank )
   WRITE( 6, "( A, /, ( 10I5 ) )" ) ' column orderings:', COLS( : rank )

!  Read right-hand side and solve system

   READ( 5, * ) B
   write(6,*) CONTROL%maxit
   CALL MA48_solve( MATRIX, FACTORS, B, X, CONTROL, SINFO )
   IF ( SINFO%flag == 0 ) WRITE( 6, '( A, /,( 6ES11.3 ) )' )                   &
     ' Solution of set of equations without refinement is', X

!  Clean up

   DEALLOCATE( MATRIX%VAL, MATRIX%ROW, MATRIX%COL )
   CALL MA48_FINALIZE( FACTORS, CONTROL, INFO )

   END PROGRAM GALAHAD_MA48_example
