! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

   PROGRAM GALAHAD_SLS_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SYMBOLS
   USE GALAHAD_SLS_precision
   IMPLICIT NONE
   TYPE ( SMT_type ) :: matrix
   TYPE ( SLS_data_type ) :: data
   TYPE ( SLS_control_type ) control
   TYPE ( SLS_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: i, ordering, scaling, solver, type, s
   LOGICAL :: mpi_flag
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5, ne  = 7
   INTEGER ( KIND = ip_ ), PARAMETER :: sils = 1
   INTEGER ( KIND = ip_ ), PARAMETER :: ma57 = 2
   INTEGER ( KIND = ip_ ), PARAMETER :: ma77 = 3
   INTEGER ( KIND = ip_ ), PARAMETER :: ma86 = 4
   INTEGER ( KIND = ip_ ), PARAMETER :: ma87 = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: ma97 = 6
   INTEGER ( KIND = ip_ ), PARAMETER :: pardiso = 7
   INTEGER ( KIND = ip_ ), PARAMETER :: mkl_pardiso = 8
   INTEGER ( KIND = ip_ ), PARAMETER :: wsmp = 9
   INTEGER ( KIND = ip_ ), PARAMETER :: pastix = 10
   INTEGER ( KIND = ip_ ), PARAMETER :: mumps = 11
   INTEGER ( KIND = ip_ ), PARAMETER :: potr = 12
   INTEGER ( KIND = ip_ ), PARAMETER :: sytr = 13
   INTEGER ( KIND = ip_ ), PARAMETER :: pbtr = 14
   INTEGER ( KIND = ip_ ), PARAMETER :: ssids = 15
   INTEGER ( KIND = ip_ ), PARAMETER :: all = ssids
   INTEGER ( KIND = ip_ ) :: ORDER( n )
   REAL ( KIND = rp_ ) :: B( n ), X( n ), B2( n, 2 ), X2( n, 2 ), D( 2, n )
   REAL ( KIND = rp_ ) :: B_diag( n )
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: row = (/ 1, 2, 2, 3, 3, 4, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: col = (/ 1, 1, 5, 2, 3, 3, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: ptr = (/ 1, 2, 4, 6, 7, 8 /)
   REAL ( KIND = rp_ ), DIMENSION( ne ) ::                                     &
     val = (/ 2.0_rp_, 3.0_rp_, 6.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_, 1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n * ( n + 1 ) / 2 ) ::                      &
     dense = (/ 2.0_rp_, 3.0_rp_, 0.0_rp_, 0.0_rp_, 4.0_rp_, 1.0_rp_, 0.0_rp_, &
                0.0_rp_, 5.0_rp_, 0.0_rp_, 0.0_rp_, 6.0_rp_, 0.0_rp_, 0.0_rp_, &
                1.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     rhs = (/ 8.0_rp_,  45.0_rp_,  31.0_rp_,  15.0_rp_,  17.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     SOL = (/ 1.0_rp_,  2.0_rp_,  3.0_rp_,  4.0_rp_,  5.0_rp_ /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: drow = (/ 1, 2, 3, 3, 4, 5, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( ne ) :: dcol = (/ 1, 2, 2, 3, 4, 1, 5 /)
   INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: dptr = (/ 1, 2, 3, 5, 6, 8 /)
   REAL ( KIND = rp_ ), DIMENSION( ne ) ::                                     &
     dval = (/ 6.0_rp_, 7.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 1.0_rp_, 5.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n * ( n + 1 ) / 2 ) ::                      &
     ddense = (/ 6.0_rp_, 0.0_rp_, 7.0_rp_, 0.0_rp_, 2.0_rp_, 3.0_rp_, 0.0_rp_,&
                 0.0_rp_, 0.0_rp_, 4.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_, 0.0_rp_,&
                 5.0_rp_ /)
   REAL ( KIND = rp_ ), DIMENSION( n ) ::                                      &
     drhs = (/ 11.0_rp_,  20.0_rp_,  13.0_rp_,  16.0_rp_,  26.0_rp_ /)

! Choose optional ordering
   DO i = 1, n
     ORDER( i ) = n - i + 1
   END DO
! Read matrix order and number of entries
!  DO type = 1, 3   ! all
   DO type = 1, 1   ! coordinate
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
     END SELECT
     matrix%n = n
!  GO TO 1

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
!      DO solver = 1, 0    ! none
       DO solver = 1, all  ! all
!      DO solver = 2, 6    ! all hsl
!      DO solver = 11, 13  ! all lapack
!      DO solver = 1, 1    ! sils
!      DO solver = 2, 2    ! ma57
!      DO solver = 3, 3    ! ma77
!      DO solver = 4, 4    ! ma86
!      DO solver = 5, 5    ! ma87
!      DO solver = 6, 6    ! ma97
!      DO solver = 7, 7    ! pardiso
!      DO solver = 8, 8    ! mkl_pardiso
!      DO solver = 9, 9    ! wsmp
!      DO solver = 10, 10  ! pastix
!      DO solver = 11, 11  ! mumps
!      DO solver = 12, 12  ! potr
!      DO solver = 13, 13  ! sytr
!      DO solver = 14, 14  ! pbtr
!      DO solver = 15, 15  ! ssids

!        IF ( solver == ma57 .OR. solver == ma86 .OR. solver == ma87 ) CYCLE
         SELECT CASE( solver )
         CASE ( sils, ma57, ma77, ma86, ma97, pardiso, mkl_pardiso,            &
                wsmp, pastix, mumps, sytr, ssids ) ! indefinite
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
         ELSE IF ( solver == pastix ) THEN
           WRITE( 6, "( '       pastix ' )", advance = 'no' )
           CALL SLS_initialize( 'pastix', data, control, inform )
         ELSE IF ( solver == mumps ) THEN
           WRITE( 6, "( '        mumps ' )", advance = 'no' )
           CALL SLS_initialize( 'mumps', data, control, inform )
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
               control%ordering = 7
             END IF
           ELSE
             control%ordering = 0
           END IF
           CALL SLS_analyse( matrix, data, control, inform )
         END IF
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
!stop ! *********
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
!write(6,"( ' X = ', 5ES10.2 )" ) X( 1 : n )
!write(6,*) ' status - ', inform%status
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
           WRITE( 6, "( '   ok  ' )", advance = 'no' )
         ELSE
!write(6,*) MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )
           WRITE( 6, "( '  fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X = B
         CALL SLS_solve( matrix, X, data, control, inform )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
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
                 <= EPSILON( 1.0_rp_ ) ** 0.5 .AND.                            &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
           WRITE( 6, "( '      ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '     fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X2 = B2
         CALL SLS_solve( matrix, X2, data, control, inform )
         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 .AND.                            &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
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
                 <= EPSILON( 1.0_rp_ ) ** 0.333 ) THEN
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

!1 CONTINUE
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
!      DO solver = 1, 0    ! none
       DO solver = 1, all  ! all
!      DO solver = 2, 6    ! all hsl
!      DO solver = 11, 13  ! all lapack
!      DO solver = 1, 1    ! sils
!      DO solver = 2, 2    ! ma57
!      DO solver = 3, 3    ! ma77
!      DO solver = 4, 4    ! ma86
!      DO solver = 5, 5    ! ma87
!      DO solver = 6, 6    ! ma97
!      DO solver = 7, 7    ! pardiso
!      DO solver = 8, 8    ! mkl_pardiso
!      DO solver = 9, 9    ! wsmp
!      DO solver = 10, 10  ! pastix
!      DO solver = 11, 11  ! potr
!      DO solver = 11, 11  ! mumps
!      DO solver = 12, 12  ! potr
!      DO solver = 13, 13  ! sytr
!      DO solver = 14, 14  ! pbtr
!      DO solver = 15, 15  ! ssids
!        IF ( solver == ma57 .OR. solver == ma86 .OR. solver == ma87 ) CYCLE
         SELECT CASE( solver )
         CASE ( sils, ma57, ma77, ma86, ma97, pardiso, mkl_pardiso,            &
                wsmp, pastix, mumps, sytr, ssids ) ! indefinite
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
         ELSE IF ( solver == pastix ) THEN
           WRITE( 6, "( '       pastix ' )", advance = 'no' )
           CALL SLS_initialize( 'pastix', data, control, inform )
         ELSE IF ( solver == mumps ) THEN
           WRITE( 6, "( '        mumps ' )", advance = 'no' )
           CALL SLS_initialize( 'mumps', data, control, inform )
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
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
           WRITE( 6, "( '   ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '  fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X = B
         CALL SLS_solve( matrix, X, data, control, inform )
         IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1: n ) ) )                        &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
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
                 <= EPSILON( 1.0_rp_ ) ** 0.5 .AND.                            &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
           WRITE( 6, "( '      ok  ' )", advance = 'no' )
         ELSE
           WRITE( 6, "( '     fail ' )", advance = 'no' )
         END IF
! Perform one refinement
         control%max_iterative_refinements = 1
         X2 = B2
         CALL SLS_solve( matrix, X2, data, control, inform )
         IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 .AND.                            &
              MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                   &
                 <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
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
                 <= EPSILON( 1.0_rp_ ) ** 0.333 ) THEN
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

   WRITE( 6, "( ' diagonal matrix tests' )" )
   WRITE( 6, "( ' type         1 RHS  >1 RHS    partial' )" )
   DO type = 4, 8   ! all
!  DO type = 4, 4   ! diagonal
!  DO type = 5, 5   ! scaled identity
!  DO type = 6, 6   ! identity
!  DO type = 7, 7   ! zero
!  DO type = 8, 8   ! none
! Allocate arrays of appropriate sizes
     SELECT CASE( type )
     CASE ( 4 )
       write(6,"( ' diagonal ' )", advance = 'no' )
       ALLOCATE( matrix%val( n ) )
       matrix%val( : n ) = rhs( : n ) / sol( : n )
       B_diag = rhs
       CALL SMT_put( matrix%type, 'DIAGONAL', s )
     CASE ( 5 )
       write(6,"( ' scaled I ' )", advance = 'no' )
       ALLOCATE( matrix%val( 1 ) )
       matrix%val( 1 ) = 2.0_rp_
       B_diag = sol * matrix%val( 1 )
       CALL SMT_put( matrix%type, 'SCALED_IDENTITY', s )
     CASE ( 6 )
       write(6,"( ' identity ' )", advance = 'no' )
       B_diag = sol
       CALL SMT_put( matrix%type, 'IDENTITY', s )
     CASE ( 7 )
       write(6,"( ' zero     ' )", advance = 'no' )
       CALL SMT_put( matrix%type, 'ZERO', s )
     CASE ( 8 )
       write(6,"( ' none     ' )", advance = 'no' )
       CALL SMT_put( matrix%type, 'NONE', s )
     END SELECT
     matrix%n = n
     CALL SLS_initialize( 'none', data, control, inform )
     CALL SLS_analyse( matrix, data, control, inform )
     CALL SLS_factorize( matrix, data, control, inform )
     X = B_diag
     CALL SLS_solve( matrix, X, data, control, inform )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1 : n ) ) )                           &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '     ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '    fail ' )", advance = 'no' )
     END IF
     X2( : n, 1 ) = B_diag ; X2( : n, 2 ) = B_diag ;
     CALL SLS_solve( matrix, X2, data, control, inform )
     IF ( MAXVAL( ABS( X2( 1 : n, 1 ) - SOL( 1 : n ) ) )                       &
             <= EPSILON( 1.0_rp_ ) ** 0.5 .AND.                                &
          MAXVAL( ABS( X2( 1 : n, 2 ) - SOL( 1 : n ) ) )                       &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '     ok  ' )", advance = 'no' )
     ELSE
       WRITE( 6, "( '    fail ' )", advance = 'no' )
     END IF
     X = B_diag
     CALL SLS_part_solve( 'L', X, data, control, inform )
     CALL SLS_part_solve( 'D', X, data, control, inform )
     CALL SLS_part_solve( 'U', X, data, control, inform )
     IF ( MAXVAL( ABS( X( 1 : n ) - SOL( 1 : n ) ) )                           &
             <= EPSILON( 1.0_rp_ ) ** 0.5 ) THEN
       WRITE( 6, "( '     ok  ' )" )
     ELSE
       WRITE( 6, "( '    fail ' )" )
     END IF
     CALL SLS_terminate( data, control, inform )
     IF ( type <= 5 ) DEALLOCATE( matrix%val )
   END DO
! stop

! Test error returns
   WRITE( 6, "( ' error tests' )" )
   WRITE( 6, "( '       solver     -3   -20   -31   -26')" )
       DO solver = 1, all  ! all
!      DO solver = 1, 0    ! none
!      DO solver = 2, 6    ! all hsl
!      DO solver = 11, 13  ! all lapack
!      DO solver = 1, 1    ! sils
!      DO solver = 2, 2    ! ma57
!      DO solver = 3, 3    ! ma77
!      DO solver = 4, 4    ! ma86
!      DO solver = 5, 5    ! ma87
!      DO solver = 6, 6    ! ma97
!      DO solver = 7, 7    ! pardiso
!      DO solver = 8, 8    ! mkl_pardiso
!      DO solver = 9, 9    ! wsmp
!      DO solver = 10, 10  ! pastix
!      DO solver = 11, 11  ! mumps
!      DO solver = 12, 12  ! potr
!      DO solver = 13, 13  ! sytr
!      DO solver = 14, 14  ! pbtr
!      DO solver = 15, 15  ! ssids

! Initialize the structures

!go to 1111
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
     ELSE IF ( solver == pastix ) THEN
       WRITE( 6, "( '       pastix ' )", advance = 'no' )
       CALL SLS_initialize( 'pastix', data, control, inform )
     ELSE IF ( solver == mumps ) THEN
       WRITE( 6, "( '        mumps ' )", advance = 'no' )
       CALL SLS_initialize( 'mumps', data, control, inform )
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
     control%out = - 1 ; control%statistics = - 1 ; control%print_level = - 1
     matrix%n = 0 ;  matrix%ne = 0
     CALL SMT_put( matrix%type, 'COORDINATE', s )
     ALLOCATE( matrix%val( 0 ), matrix%row( 0 ), matrix%col( 0 ) )
! Analyse
     CALL SLS_analyse( matrix, data, control, inform )
     WRITE( 6, "( I6 )", advance = 'no' ) inform%status
     DEALLOCATE( matrix%val, matrix%row, matrix%col, matrix%type )
     CALL SLS_terminate( data, control, inform )
     write(6,"('')", advance = 'no')

!1111 continue
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
     ELSE IF ( solver == pastix ) THEN
       CALL SLS_initialize( 'pastix', data, control, inform )
     ELSE IF ( solver == mumps ) THEN
       CALL SLS_initialize( 'mumps', data, control, inform )
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
     control%out = - 1 ; control%statistics = - 1 ; control%print_level = - 1
! Analyse
     CALL SMT_put( matrix%type, 'COORDINATE', s )
     matrix%n = 2 ;  matrix%ne = 3
     ALLOCATE( matrix%val( 3 ), matrix%row( 3 ), matrix%col( 3 ) )
     matrix%val( 1 ) = 1.0_rp_ ; matrix%row( 1 ) = 1 ; matrix%col( 1 ) = 1
     matrix%val( 2 ) = - 1.0_rp_ ; matrix%row( 2 ) = 2 ; matrix%col( 2 ) = 2
     matrix%val( 3 ) = 0.0_rp_ ; matrix%row( 3 ) = 2 ; matrix%col( 3 ) = 1
     IF ( solver == ma87 .OR. solver == potr .OR. solver == pbtr ) THEN
       matrix%val( 1 ) = 0.0_rp_
       matrix%val( 2 ) = 1.0_rp_
       control%pivot_control = 3
     ELSE
       control%pivot_control = 2
     END IF
     control%ordering = 0
     CALL SLS_analyse( matrix, data, control, inform )
     IF ( inform%status < 0 ) THEN
       WRITE( 6, "( I6 )", advance = 'no' ) inform%status
     ELSE
! Factorize
       CALL SLS_factorize( matrix, data, control, inform )
       WRITE( 6, "( I6 )", advance = 'no' ) inform%status
     END IF
     CALL SLS_terminate( data, control, inform )
     DEALLOCATE( matrix%val, matrix%row, matrix%col )
! test for error = GALAHAD_error_permutation
     ALLOCATE( matrix%val( 2 ), matrix%row( 2 ), matrix%col( 2 ) )
! Analyse
     matrix%n = 2 ;  matrix%ne = 2
     matrix%val( 1 ) = 1.0_rp_ ; matrix%row( 1 ) = 1 ; matrix%col( 1 ) = 1
     matrix%val( 2 ) = - 1.0_rp_ ; matrix%row( 2 ) = 2 ; matrix%col( 2 ) = 2
     ORDER( 1 : 2 ) = (/ 1, 1 /)
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
   CALL MPI_INITIALIZED( mpi_flag, i )
   IF ( mpi_flag ) CALL MPI_FINALIZE( i )
   WRITE( 6, "( /, ' tests completed' )" )
   STOP
   END PROGRAM GALAHAD_SLS_test_program
