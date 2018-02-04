! THIS VERSION: GALAHAD 2.4 - 19/06/2009 AT 12:00 GMT.
   PROGRAM GALAHAD_ULS_TEST_PROGRAM
   USE GALAHAD_SYMBOLS
   USE GALAHAD_ULS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( SMT_type ) :: matrix
   TYPE ( ULS_data_type ) :: data
   TYPE ( ULS_control_type ) control
   TYPE ( ULS_inform_type ) :: inform
   INTEGER :: i, ordering, solver, type, s
   INTEGER, PARAMETER :: n = 5, ne  = 7
   INTEGER :: ORDER( n )
   REAL ( KIND = wp ) :: B( n ), X( n )
!  REAL ( KIND = wp ) :: B2( n, 2 ), X2( n, 2 )
   INTEGER :: ROWS( n ), COLS( n )
   INTEGER, DIMENSION( ne ) :: row = (/ 1, 2, 2, 3, 3, 4, 5 /)
   INTEGER, DIMENSION( ne ) :: col = (/ 1, 1, 5, 2, 3, 3, 4 /)
   INTEGER, DIMENSION( n + 1 ) :: ptr = (/ 1, 2, 4, 6, 7, 8 /)
   REAL ( KIND = wp ), DIMENSION( ne ) ::                                      &
     val = (/ 2.0_wp, 3.0_wp, 6.0_wp, 4.0_wp,  1.0_wp, 5.0_wp, 1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n * n ) ::                                   &
     dense = (/ 2.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                        &
                3.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 6.0_wp,                        &
                0.0_wp, 4.0_wp, 1.0_wp, 0.0_wp, 0.0_wp,                        &
                0.0_wp, 0.0_wp, 5.0_wp, 0.0_wp, 0.0_wp,                        &
                0.0_wp, 0.0_wp, 0.0_wp, 1.0_wp, 0.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     rhs = (/ 2.0_wp, 33.0_wp, 11.0_wp, 15.0_wp, 4.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     SOL = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /)

! Choose optional ordering
   DO i = 1, n
     ORDER( i ) = n - i + 1
   END DO

! Read matrix order and number of entries
!  DO type = 1, 1
!  DO type = 2, 2
   DO type = 1, 3
! Allocate arrays of appropriate sizes
     SELECT CASE( type )
     CASE ( 1 )
       write(6,"( ' coordinate storage ' )" )
       CALL SMT_put( matrix%type, 'COORDINATE', s )
       ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
     CASE ( 2 )
       write(6,"( ' sparse by rows storage ' )" )
       ALLOCATE( matrix%val( ne ), matrix%ptr( n + 1 ), matrix%col( ne ) )
       CALL SMT_put( matrix%type, 'SPARSE_BY_ROWS', s )
     CASE ( 3 )
       write(6,"( ' dense storage ' )" )
       ALLOCATE( matrix%val( n * n ) )
       CALL SMT_put( matrix%type, 'DENSE', s )
     END SELECT
     matrix%m = n ; matrix%n = n
     DO ordering = 1, 3
!    DO ordering = 1, 1
       IF ( ordering == 1 ) THEN
         write(6,"( '  default ordering' )" )
       ELSE IF ( ordering == 2 ) THEN
         write(6,"( '  computed ordering' )" )
       ELSE
         write(6,"( '  provided ordering' )" )
       END IF
       WRITE( 6,"( '   solver  1 RHS  1 refine')")
!      DO solver = 1, 1
       DO solver = 1, 2
!      DO solver = 1, 5
! assign the matrix and right-hand side
         SELECT CASE( type )
         CASE ( 1 )
           matrix%ne = ne
           matrix%row = row
           matrix%col = col
           matrix%val = val
         CASE ( 2 )
           matrix%col = col
           matrix%val = val
           matrix%ptr = ptr
         CASE ( 3 )
           matrix%val = dense
         END SELECT
         B = rhs
! Initialize the structures
         IF ( solver == 1 ) THEN
           WRITE( 6, "( '     gls  ' )", advance = 'no' )
           CALL ULS_initialize( 'gls ', data, control, inform )
         ELSE IF ( solver == 2 ) THEN
           WRITE( 6, "( '     ma48 ' )", advance = 'no' )
           CALL ULS_initialize( 'ma48', data, control, inform )
         END IF
! Factorize
         CALL ULS_factorize( matrix, data, control, inform )
         IF ( inform%status == GALAHAD_unavailable_option ) THEN
           WRITE( 6, "( '  none ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status < 0 ) THEN
           WRITE( 6, "( '  fail in factorize ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         END IF
! Solve without refinement
         CALL ULS_solve( matrix, B, X, data, control, inform, .FALSE. )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '   ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '  fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         CALL ULS_solve( matrix, B, X, data, control, inform, .FALSE. )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '    ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '  fail  ' )", advance = 'no' )
         END IF
! Solve multiple RHS without refinement
!         B2( : , 1 ) = B ; B2( : , 2 ) = B
!         X2 = B2
!         control%max_iterative_refinements = 0
!         CALL ULS_solve( matrix, B2, X2, data, control, inform, .FALSE. )
!         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                  &
!                 <= EPSILON( 1.0_wp ) ** 0.5 .AND.                            &
!              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                  &
!                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
!           WRITE( 6, "( '      ok  ' )", advance = 'no' )
!         ELSE
!           WRITE( 6, "( '     fail ' )", advance = 'no' )
!         END IF
! Perform one refinement
!         control%max_iterative_refinements = 1
!         CALL ULS_solve( matrix, B2, X2, data, control, inform, .FALSE. )
!         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                  &
!                 <= EPSILON( 1.0_wp ) ** 0.5 .AND.                            &
!              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                  &
!                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
!           WRITE( 6, "( '       ok  ' )", advance = 'no' )
!         ELSE
!           WRITE( 6, "( '      fail ' )", advance = 'no' )
!         END IF
         CALL ULS_enquire( data, inform, ROWS, COLS )
         CALL ULS_terminate( data, control, inform )
         WRITE( 6, "( '' )" )
       END DO
     END DO
     SELECT CASE( type )
     CASE ( 1 )
       DEALLOCATE( matrix%val, matrix%row, matrix%col )
     CASE ( 2 )
       DEALLOCATE( matrix%val, matrix%ptr, matrix%col )
     CASE ( 3 )
       DEALLOCATE( matrix%val )
     CASE ( 4 )
       DEALLOCATE( matrix%val )
     END SELECT
   END DO
!stop
! Test error returns
   WRITE( 6, "( ' error tests' )" )
   WRITE( 6, "( '   solver     -3   -26')" )
   DO solver = 1, 2
! Initialize the structures
     IF ( solver == 1 ) THEN
       WRITE( 6, "( '     gls  ' )", advance = 'no' )
       CALL ULS_initialize( 'gls ', data, control, inform )
     ELSE IF ( solver == 2 ) THEN
       WRITE( 6, "( '     ma48 ' )", advance = 'no')
       CALL ULS_initialize( 'ma48', data, control, inform )
     END IF
     control%error = - 1 ; control%warning = - 1
     control%out = - 1 ; control%print_level = - 1
     matrix%n = 0 ;  matrix%m = 0 ; matrix%ne = 0
     CALL SMT_put( matrix%type, 'COORDINATE', s )
     ALLOCATE( matrix%val( 0 ), matrix%row( 0 ), matrix%col( 0 ) )
! Factorize
     CALL ULS_factorize( matrix, data, control, inform )
     WRITE( 6, "( I6 )" ) inform%status
     DEALLOCATE( matrix%val, matrix%row, matrix%col )
     CALL ULS_terminate( data, control, inform )
   END DO
   WRITE( 6, "( '  unknown ' )", advance = 'no' )
   CALL ULS_initialize( 'unknown_solver', data, control, inform )
   WRITE( 6, "( 6X, I6 )" ) inform%status
   CALL ULS_terminate( data, control, inform )
   STOP
   END PROGRAM GALAHAD_ULS_TEST_PROGRAM
