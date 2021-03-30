! THIS VERSION: GALAHAD 2.4 - 19/06/2009 AT 12:00 GMT.
   PROGRAM GALAHAD_SLS_TEST_PROGRAM
   USE GALAHAD_SYMBOLS
   USE GALAHAD_SLS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE ( SMT_type ) :: matrix
   TYPE ( SLS_data_type ) :: data
   TYPE ( SLS_control_type ) control
   TYPE ( SLS_inform_type ) :: inform
   INTEGER :: i, l, ordering, scaling, solver, type, s
   INTEGER, PARAMETER :: n = 5, ne  = 7
   INTEGER, PARAMETER :: sils = 1
   INTEGER, PARAMETER :: ma57 = 2
   INTEGER, PARAMETER :: ma77 = 3
   INTEGER, PARAMETER :: ma86 = 4
   INTEGER, PARAMETER :: ma87 = 5
   INTEGER, PARAMETER :: ma97 = 6
   INTEGER, PARAMETER :: pardiso = 7
   INTEGER, PARAMETER :: mkl_pardiso = 8
   INTEGER, PARAMETER :: wsmp = 9
   INTEGER, PARAMETER :: potr = 10
   INTEGER, PARAMETER :: sytr = 11
   INTEGER, PARAMETER :: pbtr = 12
   INTEGER, PARAMETER :: ssids = 13
   INTEGER :: ORDER( n )
   REAL ( KIND = wp ) :: B( n ), X( n ), B2( n, 2 ), X2( n, 2 )
   REAL ( KIND = wp ) :: D( 2, n )
   INTEGER, DIMENSION( ne ) :: row = (/ 1, 2, 2, 3, 3, 4, 5 /)
   INTEGER, DIMENSION( ne ) :: col = (/ 1, 1, 5, 2, 3, 3, 5 /)
   INTEGER, DIMENSION( n + 1 ) :: ptr = (/ 1, 2, 4, 6, 7, 8 /)
   REAL ( KIND = wp ), DIMENSION( ne ) ::                                      &
     val = (/ 2.0_wp, 3.0_wp, 6.0_wp, 4.0_wp,  1.0_wp, 5.0_wp, 1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n * ( n + 1 ) / 2 ) ::                       &
     dense = (/ 2.0_wp, 3.0_wp, 0.0_wp, 0.0_wp, 4.0_wp, 1.0_wp, 0.0_wp,        &
                0.0_wp, 5.0_wp, 0.0_wp, 0.0_wp, 6.0_wp, 0.0_wp, 0.0_wp,        &
                1.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     rhs = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     SOL = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /)
   INTEGER, DIMENSION( ne ) :: drow = (/ 1, 2, 3, 3, 4, 5, 5 /)
   INTEGER, DIMENSION( ne ) :: dcol = (/ 1, 2, 2, 3, 4, 1, 5 /)
   INTEGER, DIMENSION( n + 1 ) :: dptr = (/ 1, 2, 3, 5, 6, 8 /)
   REAL ( KIND = wp ), DIMENSION( ne ) ::                                      &
     dval = (/ 6.0_wp, 7.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 1.0_wp, 5.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n * ( n + 1 ) / 2 ) ::                       &
     ddense = (/ 6.0_wp, 0.0_wp, 7.0_wp, 0.0_wp, 2.0_wp, 3.0_wp, 0.0_wp,       &
                 0.0_wp, 0.0_wp, 4.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,       &
                 5.0_wp /)
   REAL ( KIND = wp ), DIMENSION( n ) ::                                       &
     drhs = (/ 11.0_wp, 20.0_wp, 13.0_wp, 16.0_wp, 26.0_wp /)

! Choose optional ordering
   DO i = 1, n
     ORDER( i ) = n - i + 1
   END DO

! Read matrix order and number of entries
!  DO type = 1, 0   ! none
   DO type = 1, 3   ! all
!  DO type = 1, 1   ! coordinate
!  DO type = 2, 2   ! row-wise
!  DO type = 3, 3   ! dense
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
       ALLOCATE( matrix%val( n * ( n + 1 ) / 2 ) )
       CALL SMT_put( matrix%type, 'DENSE', s )
     CASE ( 4 )
       write(6,"( ' diagonal storage ' )" )
       ALLOCATE( matrix%val( n ) )
       CALL SMT_put( matrix%type, 'DIAGONAL', s )
     END SELECT
     matrix%n = n

!  test external ordering strategies

!    DO ordering = 1, 0 ! none
     DO ordering = 1, 3 ! all orderings
!    DO ordering = 1, 1 ! default ordering
!    DO ordering = 2, 2 ! computed ordering
!    DO ordering = 3, 3 ! provided ordering
       IF ( ordering == 1 ) THEN
         write(6,"( '  default ordering' )" )
       ELSE IF ( ordering == 2 ) THEN
         write(6,"( '  computed ordering' )" )
       ELSE
         write(6,"( '  provided ordering' )" )
       END IF
       WRITE( 6,                                                               &
          "( '       solver  1 RHS  1 refine  >1 RHS    >1 refine  partial')" )
       DO l = 1, 13   ! all
!      DO l = 7, 12   ! all hsl
!      DO l = 2, 4    ! all lapack
!      DO l = 13, 13  ! sils
!      DO l = 12, 12  ! ma57
!      DO l = 11, 11  ! ma77
!      DO l = 10, 10  ! ma86
!      DO l = 9, 9    ! ma87
!      DO l = 8, 8    ! ma97
!      DO l = 7, 7    ! pardiso
!      DO l = 6, 6    ! mkl_pardiso
!      DO l = 5, 5    ! wsmp
!      DO l = 4, 4    ! potr
!      DO l = 3, 3    ! sytr
!      DO l = 2, 2    ! pbtr
!      DO l = 1, 1    ! ssids
        solver = l
!       solver = 14 - l
!        IF ( solver == ma57 .OR. solver == ma86 .OR. solver == ma87 ) CYCLE
         SELECT CASE( solver )
         CASE ( sils, ma57, ma77, ma86, ma97, pardiso, mkl_pardiso,            &
                wsmp, sytr, ssids ) ! indefinite
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
           CASE ( 4 )
             matrix%val = rhs
           END SELECT
           B = rhs
         CASE ( ma87, potr, pbtr ) ! positive-definite matrix
           SELECT CASE( type )
           CASE ( 1 )
             matrix%ne = ne
             matrix%row = drow
             matrix%col = dcol
             matrix%val = dval
           CASE ( 2 )
             matrix%col = dcol
             matrix%val = dval
             matrix%ptr = dptr
           CASE ( 3 )
             matrix%val = ddense
           CASE ( 4 )
             matrix%val = drhs
           END SELECT
           B = drhs
         END SELECT
! Initialize the structures
         IF ( solver == sils ) THEN
           WRITE( 6, "( '         sils ' )", advance = 'no' )
           CALL SLS_initialize( 'sils', data, control, inform )
         ELSE IF ( solver == ma57 ) THEN
           WRITE( 6, "( '         ma57 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma57', data, control, inform )
        ELSE IF ( solver == ma77 ) THEN
           WRITE( 6, "( '         ma77 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma77', data, control, inform )
         ELSE IF ( solver == ma86 ) THEN
           WRITE( 6, "( '         ma86 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma86', data, control, inform )
         ELSE IF ( solver == ma87 ) THEN
           WRITE( 6, "( '         ma87 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma87', data, control, inform )
         ELSE IF ( solver == ma97 ) THEN
           WRITE( 6, "( '         ma97 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma97', data, control, inform )
         ELSE IF ( solver == pardiso ) THEN
           WRITE( 6, "( '      pardiso ' )", advance = 'no' )
           CALL SLS_initialize( 'pardiso', data, control, inform )
         ELSE IF ( solver == mkl_pardiso ) THEN
           WRITE( 6, "( '  mkl_pardiso ' )", advance = 'no' )
           CALL SLS_initialize( 'mkl_pardiso', data, control, inform )
         ELSE IF ( solver == wsmp ) THEN
           WRITE( 6, "( '         wsmp ' )", advance = 'no' )
           CALL SLS_initialize( 'wsmp', data, control, inform )
         ELSE IF ( solver == potr ) THEN
           WRITE( 6, "( '         potr ' )", advance = 'no' )
           CALL SLS_initialize( 'potr', data, control, inform )
         ELSE IF ( solver == sytr ) THEN
           WRITE( 6, "( '         sytr ' )", advance = 'no' )
           CALL SLS_initialize( 'sytr', data, control, inform )
         ELSE IF ( solver == pbtr ) THEN
           WRITE( 6, "( '         pbtr ' )", advance = 'no' )
           CALL SLS_initialize( 'pbtr', data, control, inform )
         ELSE IF ( solver == ssids ) THEN
           WRITE( 6, "( '        ssids ' )", advance = 'no' )
           CALL SLS_initialize( 'ssids', data, control, inform )
         END IF
         control%scaling = 0
! Analyse
!write(6,*) ' analyse '
!control%print_level = 11
         IF ( ordering == 3 ) THEN
           CALL SLS_analyse( matrix, data, control, inform, PERM = ORDER )
         ELSE
           IF ( ordering == 2 ) THEN
             IF ( solver == pbtr ) THEN
               control%ordering = 6
             ELSE
               control%ordering = 1
             END IF
           ELSE
             control%ordering = 0
           END IF
           CALL SLS_analyse( matrix, data, control, inform )
         END IF
!write(6,*) ' status - ', inform%status
!stop
         IF ( inform%status == GALAHAD_unavailable_option .OR.                 &
              inform%status == GALAHAD_error_unknown_solver ) THEN
           WRITE( 6, "( '  none ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status == GALAHAD_error_mc64 .OR.                    &
                   inform%status == GALAHAD_error_mc77 ) THEN
           WRITE( 6, "( '  ordering unavailable ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status < 0 ) THEN
           WRITE( 6, "( '  fail in analyse, status = ', i0 )",                 &
                  advance = 'no' ) inform%status
           WRITE( 6, "( '' )" )
           CYCLE
         END IF
! Factorize
!write(6,*) ' factorize '
         CALL SLS_factorize( matrix, data, control, inform )
!write(6,*) ' status - ', inform%status
         IF ( inform%status == GALAHAD_error_unknown_solver ) THEN
           WRITE( 6, "( '  none ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status == GALAHAD_error_mc64 .OR.                    &
                   inform%status == GALAHAD_error_mc77 ) THEN
           WRITE( 6, "( '  ordering unavailable ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status < 0 ) THEN
           WRITE( 6, "( '  fail in factorize, status = ', I0 )",               &
                  advance = 'no' ) inform%status
           WRITE( 6, "( '' )" )
           CYCLE
         END IF
! Solve without refinement
         control%max_iterative_refinements = 0
         X = B
!write(6,*) ' solve 1 RHS'
         CALL SLS_solve( matrix, X, data, control, inform )
!write(6,*) ' status - ', inform%status
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '   ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '  fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X = B
         CALL SLS_solve( matrix, X, data, control, inform )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '    ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '  fail  ' )", advance = 'no' )
         END IF
! Solve multiple RHS without refinement
         B2( : , 1 ) = B ; B2( : , 2 ) = B
         X2 = B2
!write(6,*) ' solve multiple RHS'
         control%max_iterative_refinements = 0
         CALL SLS_solve( matrix, X2, data, control, inform )
!write(6,*) ' status - ', inform%status
         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 .AND.                             &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '      ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '     fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X2 = B2
         CALL SLS_solve( matrix, X2, data, control, inform )
         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 .AND.                             &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '       ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '      fail ' )", advance = 'no' )
         END IF
! Obtain solution by part solves
         X = B
!write(6,*) ' L '
         CALL SLS_part_solve( 'L', X, data, control, inform )
         IF ( inform%status == GALAHAD_unavailable_option ) THEN
           WRITE( 6, "( '    none ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF (inform%status /= GALAHAD_ok ) THEN
           WRITE( 6, "( '    fail ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         END IF
!write(6,*) ' D '
         CALL SLS_part_solve( 'D', X, data, control, inform )
!write(6,*) ' U '
         CALL SLS_part_solve( 'U', X, data, control, inform )
!write(6,*) ' E '
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.333 ) THEN
           WRITE( 6, "( '     ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '    fail ' )", advance = 'no' )
         END IF
!write(6,*) ' enquire '
         CALL SLS_enquire( data, inform, D = D, PERM = ORDER )
!write(6,*) ' alter d '
         CALL SLS_alter_d( data, D, inform )
!write(6,*) ' terminate '
         CALL SLS_terminate( data, control, inform )
!write(6,*) ' end terminate '
         WRITE( 6, "( '' )" )
       END DO
     END DO

!  test external scaling strategies

!    DO scaling = 1, 0  ! none
     DO scaling = 1, 3  ! all
!    DO scaling = 1, 1  ! mc64
!    DO scaling = 2, 2  ! mc77 1-norm
!    DO scaling = 3, 3  ! mc77 inf-norm
       control%scaling = scaling
       IF ( scaling == 1 ) THEN
         write(6,"( '  MC64 scaling' )" )
       ELSE IF ( scaling == 2 ) THEN
         write(6,"( '  MC77 1-norm scaling' )" )
       ELSE
         write(6,"( '  MC77 infinity-norm scaling' )" )
       END IF
       WRITE( 6,                                                               &
          "( '       solver  1 RHS  1 refine  >1 RHS    >1 refine  partial')" )
       DO l = 1, 13   ! all
!      DO l = 7, 12   ! all hsl
!      DO l = 2, 4    ! all lapack
!      DO l = 13, 13  ! sils
!      DO l = 12, 12  ! ma57
!      DO l = 11, 11  ! ma77
!      DO l = 10, 10  ! ma86
!      DO l = 9, 9    ! ma87
!      DO l = 8, 8    ! ma97
!      DO l = 7, 7    ! pardiso
!      DO l = 6, 6    ! mkl_pardiso
!      DO l = 5, 5    ! wsmp
!      DO l = 4, 4    ! potr
!      DO l = 3, 3    ! sytr
!      DO l = 2, 2    ! pbtr
!      DO l = 1, 1    ! ssids
        solver = l
!       solver = 14 - l
!        IF ( solver == ma57 .OR. solver == ma86 .OR. solver == ma87 ) CYCLE
         SELECT CASE( solver )
         CASE ( sils, ma57, ma77, ma86, ma97, pardiso, mkl_pardiso,            &
                wsmp, sytr, ssids ) ! indefinite
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
           CASE ( 4 )
             matrix%val = rhs
           END SELECT
           B = rhs
         CASE ( ma87, potr, pbtr ) ! positive-definite matrix
           SELECT CASE( type )
           CASE ( 1 )
             matrix%ne = ne
             matrix%row = drow
             matrix%col = dcol
             matrix%val = dval
           CASE ( 2 )
             matrix%col = dcol
             matrix%val = dval
             matrix%ptr = dptr
           CASE ( 3 )
             matrix%val = ddense
           CASE ( 4 )
             matrix%val = drhs
           END SELECT
           B = drhs
         END SELECT
! Initialize the structures
         IF ( solver == sils ) THEN
           WRITE( 6, "( '         sils ' )", advance = 'no' )
           CALL SLS_initialize( 'sils', data, control, inform )
         ELSE IF ( solver == ma57 ) THEN
           WRITE( 6, "( '         ma57 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma57', data, control, inform )
        ELSE IF ( solver == ma77 ) THEN
           WRITE( 6, "( '         ma77 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma77', data, control, inform )
         ELSE IF ( solver == ma86 ) THEN
           WRITE( 6, "( '         ma86 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma86', data, control, inform )
         ELSE IF ( solver == ma87 ) THEN
           WRITE( 6, "( '         ma87 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma87', data, control, inform )
         ELSE IF ( solver == ma97 ) THEN
           WRITE( 6, "( '         ma97 ' )", advance = 'no' )
           CALL SLS_initialize( 'ma97', data, control, inform )
         ELSE IF ( solver == pardiso ) THEN
           WRITE( 6, "( '      pardiso ' )", advance = 'no' )
           CALL SLS_initialize( 'pardiso', data, control, inform )
         ELSE IF ( solver == mkl_pardiso ) THEN
           WRITE( 6, "( '  mkl_pardiso ' )", advance = 'no' )
           CALL SLS_initialize( 'mkl_pardiso', data, control, inform )
         ELSE IF ( solver == wsmp ) THEN
           WRITE( 6, "( '         wsmp ' )", advance = 'no' )
           CALL SLS_initialize( 'wsmp', data, control, inform )
         ELSE IF ( solver == potr ) THEN
           WRITE( 6, "( '         potr ' )", advance = 'no' )
           CALL SLS_initialize( 'potr', data, control, inform )
         ELSE IF ( solver == sytr ) THEN
           WRITE( 6, "( '         sytr ' )", advance = 'no' )
           CALL SLS_initialize( 'sytr', data, control, inform )
         ELSE IF ( solver == pbtr ) THEN
           WRITE( 6, "( '         pbtr ' )", advance = 'no' )
           CALL SLS_initialize( 'pbtr', data, control, inform )
         ELSE IF ( solver == ssids ) THEN
           WRITE( 6, "( '        ssids ' )", advance = 'no' )
           CALL SLS_initialize( 'ssids', data, control, inform )
         END IF
         control%ordering = 0
! Analyse
         CALL SLS_analyse( matrix, data, control, inform )
         IF ( inform%status == GALAHAD_unavailable_option .OR.                 &
              inform%status == GALAHAD_error_unknown_solver ) THEN
           WRITE( 6, "( '  none ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status == GALAHAD_error_mc64 .OR.                    &
                   inform%status == GALAHAD_error_mc77 ) THEN
           WRITE( 6, "( '  ordering unavailable ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status < 0 ) THEN
           WRITE( 6, "( '  fail in analyse, status = ', I0 )",                 &
                  advance = 'no' ) inform%status
           WRITE( 6, "( '' )" )
           CYCLE
         END IF
! Factorize
!control%out = 6
!control%print_level = 3
         CALL SLS_factorize( matrix, data, control, inform )
         IF ( inform%status == GALAHAD_error_unknown_solver ) THEN
           WRITE( 6, "( '  none ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status == GALAHAD_error_mc64 .OR.                    &
                   inform%status == GALAHAD_error_mc77 ) THEN
           WRITE( 6, "( '  ordering unavailable ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF ( inform%status < 0 ) THEN
           WRITE( 6, "( '  fail in factorize, status = ', I0 )",               &
                  advance = 'no' ) inform%status
           WRITE( 6, "( '' )" )
           CYCLE
         END IF
! Solve without refinement
         control%max_iterative_refinements = 0
         X = B
         CALL SLS_solve( matrix, X, data, control, inform )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '   ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '  fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X = B
         CALL SLS_solve( matrix, X, data, control, inform )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '    ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '  fail  ' )", advance = 'no' )
         END IF
! Solve multiple RHS without refinement
         B2( : , 1 ) = B ; B2( : , 2 ) = B
         X2 = B2
         control%max_iterative_refinements = 0
         CALL SLS_solve( matrix, X2, data, control, inform )
         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 .AND.                             &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '      ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '     fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X2 = B2
         CALL SLS_solve( matrix, X2, data, control, inform )
         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 .AND.                             &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_wp ) ** 0.5 ) THEN
           WRITE( 6, "( '       ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '      fail ' )", advance = 'no' )
         END IF
! Obtain solution by part solves
         X = B
         CALL SLS_part_solve( 'L', X, data, control, inform )
         IF ( inform%status == GALAHAD_unavailable_option ) THEN
           WRITE( 6, "( '    none ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         ELSE IF (inform%status /= GALAHAD_ok ) THEN
           WRITE( 6, "( '    fail ' )", advance = 'no' )
           WRITE( 6, "( '' )" )
           CYCLE
         END IF
         CALL SLS_part_solve( 'D', X, data, control, inform )
         CALL SLS_part_solve( 'U', X, data, control, inform )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_wp ) ** 0.333 ) THEN
           WRITE( 6, "( '     ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '    fail ' )", advance = 'no' )
         END IF
         CALL SLS_enquire( data, inform, D = D, PERM = ORDER )
         CALL SLS_alter_d( data, D, inform )
         CALL SLS_terminate( data, control, inform )
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
! stop
! Test error returns
   WRITE( 6, "( ' error tests' )" )
   WRITE( 6, "( '       solver     -3   -20   -31   -26')" )
!  DO l = 1, 0    ! none
   DO l = 1, 13   ! all
!  DO l = 7, 12   ! all hsl
!  DO l = 2, 4    ! all lapack
!  DO l = 13, 13  ! sils
!  DO l = 12, 12  ! ma57
!  DO l = 11, 11  ! ma77
!  DO l = 10, 10  ! ma86
!  DO l = 9, 9    ! ma87
!  DO l = 8, 8    ! ma97
!  DO l = 7, 7    ! pardiso
!  DO l = 6, 6    ! mkl_pardiso
!  DO l = 5, 5    ! wsmp
!  DO l = 4, 4    ! potr
!  DO l = 3, 3    ! sytr
!  DO l = 2, 2    ! pbtr
!  DO l = 1, 1    ! ssids
!    solver = 14 - l
     solver = l
! Initialize the structures

! test for error = GALAHAD_error_restrictions

     IF ( solver == sils ) THEN
       WRITE( 6, "( '         sils ' )", advance = 'no' )
       CALL SLS_initialize( 'sils', data, control, inform )
     ELSE IF ( solver == ma57 ) THEN
       WRITE( 6, "( '         ma57 ' )", advance = 'no')
       CALL SLS_initialize( 'ma57', data, control, inform )
     ELSE IF ( solver == ma77 ) THEN
       WRITE( 6, "( '         ma77 ' )", advance = 'no')
       CALL SLS_initialize( 'ma77', data, control, inform )
     ELSE IF ( solver == ma86 ) THEN
       WRITE( 6, "( '         ma86 ' )", advance = 'no' )
       CALL SLS_initialize( 'ma86', data, control, inform )
     ELSE IF ( solver == ma87 ) THEN
       WRITE( 6, "( '         ma87 ' )", advance = 'no' )
     ELSE IF ( solver == ma97 ) THEN
       WRITE( 6, "( '         ma97 ' )", advance = 'no' )
       CALL SLS_initialize( 'ma87', data, control, inform )
     ELSE IF ( solver == pardiso ) THEN
       WRITE( 6, "( '      pardiso ' )", advance = 'no' )
       CALL SLS_initialize( 'pardiso', data, control, inform )
     ELSE IF ( solver == mkl_pardiso ) THEN
       WRITE( 6, "( '  mkl_pardiso ' )", advance = 'no' )
       CALL SLS_initialize( 'mkl_pardiso', data, control, inform )
     ELSE IF ( solver == wsmp ) THEN
       WRITE( 6, "( '         wsmp ' )", advance = 'no' )
       CALL SLS_initialize( 'wsmp', data, control, inform )
     ELSE IF ( solver == potr ) THEN
       WRITE( 6, "( '         potr ' )", advance = 'no' )
       CALL SLS_initialize( 'potr', data, control, inform )
     ELSE IF ( solver == sytr ) THEN
       WRITE( 6, "( '         sytr ' )", advance = 'no' )
       CALL SLS_initialize( 'sytr', data, control, inform )
     ELSE IF ( solver == pbtr ) THEN
       WRITE( 6, "( '         pbtr ' )", advance = 'no' )
       CALL SLS_initialize( 'pbtr', data, control, inform )
     ELSE IF ( solver == ssids ) THEN
       WRITE( 6, "( '        ssids ' )", advance = 'no' )
       CALL SLS_initialize( 'ssids', data, control, inform )
     END IF
     control%error = - 1 ; control%warning = - 1
     control%out = - 1 ; control%print_level = - 1
     matrix%n = 0 ;  matrix%ne = 0
     CALL SMT_put( matrix%type, 'COORDINATE', s )
     ALLOCATE( matrix%val( 0 ), matrix%row( 0 ), matrix%col( 0 ) )
! Analyse
     CALL SLS_analyse( matrix, data, control, inform )
     WRITE( 6, "( I6 )", advance = 'no' ) inform%status
     DEALLOCATE( matrix%val, matrix%row, matrix%col )
     CALL SLS_terminate( data, control, inform )

! test for error = GALAHAD_error_inertia

     IF ( solver == sils ) THEN
       CALL SLS_initialize( 'sils', data, control, inform )
     ELSE IF ( solver == ma57 ) THEN
       CALL SLS_initialize( 'ma57', data, control, inform )
     ELSE IF ( solver == ma77 ) THEN
       CALL SLS_initialize( 'ma77', data, control, inform )
     ELSE IF ( solver == ma86 ) THEN
       CALL SLS_initialize( 'ma86', data, control, inform )
     ELSE IF ( solver == ma87 ) THEN
       CALL SLS_initialize( 'ma87', data, control, inform )
     ELSE IF ( solver == ma97 ) THEN
       CALL SLS_initialize( 'ma97', data, control, inform )
     ELSE IF ( solver == pardiso ) THEN
       CALL SLS_initialize( 'pardiso', data, control, inform )
     ELSE IF ( solver == mkl_pardiso ) THEN
       CALL SLS_initialize( 'mkl_pardiso', data, control, inform )
     ELSE IF ( solver == wsmp ) THEN
       CALL SLS_initialize( 'wsmp', data, control, inform )
     ELSE IF ( solver == potr ) THEN
       CALL SLS_initialize( 'potr', data, control, inform )
     ELSE IF ( solver == sytr ) THEN
       CALL SLS_initialize( 'sytr', data, control, inform )
     ELSE IF ( solver == pbtr ) THEN
       CALL SLS_initialize( 'pbtr', data, control, inform )
     ELSE IF ( solver == ssids ) THEN
       CALL SLS_initialize( 'ssids', data, control, inform )
     END IF
     control%error = - 1 ; control%warning = - 1
     control%out = - 1 ; control%print_level = - 1
     ALLOCATE( matrix%val( 2 ), matrix%row( 2 ), matrix%col( 2 ) )
!IF ( solver == ma86 ) THEN
!     control%error = 6 ; control%warning = 6
!     control%out = 6 ; control%print_level = 5
!ELSE
!     control%error = - 1 ; control%warning = - 1
!    control%out = - 1 ; control%print_level = - 1
! control%print_level = 0
!END IF
! Analyse
     matrix%n = 2 ;  matrix%ne = 2
     matrix%val( 1 ) = 1.0_wp ; matrix%row( 1 ) = 1 ; matrix%col( 1 ) = 1
     matrix%val( 2 ) = - 1.0_wp ; matrix%row( 2 ) = 2 ; matrix%col( 2 ) = 2
     IF ( solver == ma87 .OR. solver == potr .OR. solver == pbtr ) THEN
       matrix%val( 1 ) = 0.0_wp
       matrix%val( 2 ) = 1.0_wp
       control%pivot_control = 3
     ELSE
       control%pivot_control = 2
     END IF
     control%ordering = 0
!write(6,*) ' analyse '
     CALL SLS_analyse( matrix, data, control, inform )
!write(6,*) ' status - ', inform%status
     IF ( inform%status < 0 ) THEN
       WRITE( 6, "( I6 )", advance = 'no' ) inform%status
     ELSE
! Factorize
!write(6,*) ' factorize '
       CALL SLS_factorize( matrix, data, control, inform )
!write(6,*) ' status - ', inform%status
       WRITE( 6, "( I6 )", advance = 'no' ) inform%status
     END IF
!write(6,*) ' terminate '
     CALL SLS_terminate( data, control, inform )
!write(6,*) ' after terminate '
     DEALLOCATE( matrix%val, matrix%row, matrix%col )

! test for error = GALAHAD_error_permutation

     ALLOCATE( matrix%val( 2 ), matrix%row( 2 ), matrix%col( 2 ) )
! Analyse
     matrix%n = 2 ;  matrix%ne = 2
     matrix%val( 1 ) = 1.0_wp ; matrix%row( 1 ) = 1 ; matrix%col( 1 ) = 1
     matrix%val( 2 ) = - 1.0_wp ; matrix%row( 2 ) = 2 ; matrix%col( 2 ) = 2
     ORDER( 1 : 2 ) = (/ 1, 1 /)
!write(6,*) ' analyse '
     CALL SLS_analyse( matrix, data, control, inform, PERM = ORDER( 1 : 2 ) )
     WRITE( 6, "( I6 )" ) inform%status
     CALL SLS_terminate( data, control, inform )
     DEALLOCATE( matrix%val, matrix%row, matrix%col )
   END DO
   WRITE( 6, "( '      unknown ' )", advance = 'no' )
   CALL SLS_initialize( 'unknown_solver', data, control, inform )
   WRITE( 6, "( 18X, I6 )" ) inform%status
   CALL SLS_terminate( data, control, inform )
   DEALLOCATE( matrix%type )
   STOP
   END PROGRAM GALAHAD_SLS_TEST_PROGRAM
