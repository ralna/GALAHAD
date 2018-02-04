! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
   PROGRAM GALAHAD_GLS_example
   USE GALAHAD_GLS_DOUBLE
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) 
   INTEGER :: i, info, m, n, ne, rank
   TYPE ( SMT_TYPE ) :: MATRIX
   TYPE ( GLS_CONTROL ) :: CONTROL
   TYPE ( GLS_AINFO ) :: AINFO
   TYPE ( GLS_FINFO ) :: FINFO
   TYPE ( GLS_SINFO ) :: SINFO
   TYPE ( GLS_FACTORS ) :: FACTORS
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

   CALL GLS_initialize( FACTORS, CONTROL )

!  Analyse and factorize

   CALL GLS_analyse( MATRIX, FACTORS, CONTROL, AINFO, FINFO )
   IF ( AINFO%flag < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
       ' Failure of GLS_ANALYSE with AINFO%flag = ',  AINFO%flag
     STOP
   END IF

!  Write row and column reorderings

   CALL GLS_special_rows_and_cols( FACTORS, rank, ROWS, COLS, info )
   WRITE( 6, "( A, /, ( 10I5 ) )" ) ' row orderings:', ROWS( : rank )
   WRITE( 6, "( A, /, ( 10I5 ) )" ) ' column orderings:', COLS( : rank )

!  Read right-hand side and solve system

   READ( 5, * ) B( : m )

   CALL GLS_solve( MATRIX, FACTORS, B, X, CONTROL, SINFO )
   IF ( SINFO%flag == 0 ) WRITE( 6, '( A, /,( 6ES11.3 ) )' )                   &
     ' Solution of set of equations without refinement is', X

!  Now solve the transposed system

!  READ( 5, * ) B( : m )

!   CALL GLS_solve( MATRIX, FACTORS, B, X, CONTROL, SINFO, trans = 1 )
!   IF ( SINFO%flag == 0 ) WRITE( 6, '( A, /,( 6ES11.3 ) )' )                  &
!     ' Solution of set of transposed equations without refinement is', X

!  Clean up

   DEALLOCATE( MATRIX%VAL, MATRIX%ROW, MATRIX%COL )
   CALL GLS_FINALIZE( FACTORS, CONTROL, INFO )

   END PROGRAM GALAHAD_GLS_example
