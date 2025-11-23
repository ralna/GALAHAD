! THIS VERSION: GALAHAD 5.4 - 2025-11-22 AT 09:20 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ N R E K  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  May 20th 2025

   MODULE GALAHAD_USENREK_precision

!  This is the driver program for running NREK for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_KINDS_precision
     USE CUTEST_INTERFACE_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NREK_precision
     USE GALAHAD_RQS_precision
     USE GALAHAD_GLRT_precision
     USE GALAHAD_SLS_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_precision
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_NREK

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ N R E K   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_NREK( input )

!  dummy argument

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( NREK_control_type ) :: control
     TYPE ( NREK_inform_type ) :: inform
     TYPE ( NREK_data_type ) :: data

     TYPE ( RQS_control_type ) :: RQS_control
     TYPE ( RQS_inform_type ) :: RQS_inform
     TYPE ( RQS_data_type ) :: RQS_data

     TYPE ( GLRT_control_type ) :: GLRT_control
     TYPE ( GLRT_inform_type ) :: GLRT_inform
     TYPE ( GLRT_data_type ) :: GLRT_data

      TYPE ( SLS_data_type ) :: SLS_data
      TYPE ( SLS_control_type ) :: SLS_control
      TYPE ( SLS_inform_type ) :: SLS_inform

!------------------------------------
!   L o c a l   P a r a m e t e r s
!------------------------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
!    REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_

!----------------------------------
!   L o c a l   V a r i a b l e s
!----------------------------------

     INTEGER ( KIND = ip_ ) :: iores, i, j, ir, ic, k, l
     INTEGER ( KIND = ip_ ) :: smt_stat, cutest_status
     REAL ( KIND = rp_ ) :: clock_total, clock_start
     LOGICAL :: goth

!  functions

!$   INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  problem characteristics

     INTEGER ( KIND = ip_ ) :: n, nnzh
     INTEGER ( KIND = ip_ ) :: n_threads = 1
     REAL ( KIND = rp_ ) ::  f
     CHARACTER ( LEN = 10 ) :: pname
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X0, X_l, X_u, G
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: R, VECTOR, H_vector
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WEIGHT
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES
     TYPE ( SMT_type ) :: H, S

!  problem input characteristics

     LOGICAL :: filexx, is_specfile

!  default values for specfile-defined parameters

     INTEGER ( KIND = ip_ ) :: nrek_rfiledevice = 47
     INTEGER ( KIND = ip_ ) :: nrek_sfiledevice = 62
     INTEGER ( KIND = ip_ ) :: rqs_rfiledevice = 48
     INTEGER ( KIND = ip_ ) :: rqs_sfiledevice = 63
     INTEGER ( KIND = ip_ ) :: glrt_rfiledevice = 49
     INTEGER ( KIND = ip_ ) :: glrt_sfiledevice = 64
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: nrek_rfilename = 'NREKRES.d'
     CHARACTER ( LEN = 30 ) :: nrek_sfilename = 'NREKSOL.d'
     CHARACTER ( LEN = 30 ) :: rqs_rfilename = 'RQSRES.d'
     CHARACTER ( LEN = 30 ) :: rqs_sfilename = 'RQSSOL.d'
     CHARACTER ( LEN = 30 ) :: glrt_rfilename = 'GLRTRES.d'
     CHARACTER ( LEN = 30 ) :: glrt_sfilename = 'GLRTSOL.d'
     CHARACTER ( LEN = 30 ) :: linear_solver = "ignore" // REPEAT( ' ', 24 )
     REAL ( KIND = rp_ ) ::  power = 3.0_rp_
     REAL ( KIND = rp_ ) ::  weight_initial = 1.0_rp_
     INTEGER ( KIND = ip_ ) :: m = 1
     LOGICAL :: run_nrek = .TRUE.
     LOGICAL :: run_rqs = .FALSE.
     LOGICAL :: run_glrt = .FALSE.
!    LOGICAL :: one_norm = .TRUE.

!  output file characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
     INTEGER ( KIND = ip_ ) :: out  = 6
     INTEGER ( KIND = ip_ ) :: errout = 6
     CHARACTER ( LEN =  6 ) :: solv

!  specfile characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 23
     CHARACTER ( LEN = 16 ) :: specname = 'RUNNREK'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNNREK.SPC'

!  ------------------ Open the specfile for nrek ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   define the keywords

       spec( 1 )%keyword = 'write-problem-data'
       spec( 2 )%keyword = 'write-result-summary'
       spec( 3 )%keyword = 'nrek-result-summary-file-name'
       spec( 4 )%keyword = 'nrek-result-summary-file-device'
       spec( 5 )%keyword = 'rqs-result-summary-file-name'
       spec( 6 )%keyword = 'rqs-result-summary-file-device'
       spec( 7 )%keyword = 'glrt-result-summary-file-name'
       spec( 8 )%keyword = 'glrt-result-summary-file-device'
       spec( 9 )%keyword = 'print-full-solution'
       spec( 10 )%keyword = 'write-solution'
       spec( 11 )%keyword = 'nrek-solution-file-name'
       spec( 12 )%keyword = 'nrek-solution-file-device'
       spec( 13 )%keyword = 'rqs-solution-file-name'
       spec( 14 )%keyword = 'rqs-solution-file-device'
       spec( 15 )%keyword = 'glrt-solution-file-name'
       spec( 16 )%keyword = 'glrt-solution-file-device'
       spec( 17 )%keyword = 'power'
       spec( 18 )%keyword = 'weight'
       spec( 19 )%keyword = 'number-of-weights'
       spec( 20 )%keyword = 'compare-with-rqs'
       spec( 21 )%keyword = 'compare-with-glrt'
       spec( 22 )%keyword = 'compare-with-nrek'
       spec( 23 )%keyword = 'symmetric-linear-equation-solver'

!   read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_logical( spec( 2 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 3 ), nrek_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), nrek_rfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 5 ), rqs_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 6 ), rqs_rfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 7 ), glrt_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 8 ), glrt_rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 9 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 10 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 11 ), nrek_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 12 ), nrek_sfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 13 ), rqs_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 14 ), rqs_sfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 15 ), glrt_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 16 ), glrt_sfiledevice, errout )
       CALL SPECFILE_assign_real( spec( 17 ), power, errout )
       CALL SPECFILE_assign_real( spec( 18 ), weight_initial, errout )
       CALL SPECFILE_assign_integer( spec( 19 ), m, errout )
       CALL SPECFILE_assign_logical( spec( 20 ), run_rqs, errout )
       CALL SPECFILE_assign_logical( spec( 21 ), run_glrt, errout )
       CALL SPECFILE_assign_logical( spec( 22 ), run_nrek, errout )
       CALL SPECFILE_assign_string ( spec( 23 ), linear_solver, errout )
     END IF

!  set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2025' )

!  set up data for next problem

     CALL NREK_initialize( data, control, inform )
     IF ( is_specfile ) CALL NREK_read_specfile( control, input_specfile )

!  read the initial point and bounds

     CALL CUTEST_udimen_r( cutest_status, input, n )
     IF ( cutest_status /= 0 ) GO TO 910

     ALLOCATE( X( n ), X0( n ), X_l( n ), X_u( n ), G( n ), VNAMES( n ) )
     CALL CUTEST_usetup_r( cutest_status, input, control%error, io_buffer,     &
                           n, X0, X_l, X_u )
     IF ( cutest_status /= 0 ) GO TO 910
     DEALLOCATE( X_l, X_u )

!  read the problem and variable names

     CALL CUTEST_unames_r( cutest_status, n, pname, VNAMES )
     IF ( cutest_status /= 0 ) GO TO 910

!  set f to zero

    f = zero

!  evaluate the gradient

     CALL CUTEST_ugr_r( cutest_status, n, X0, G )
     IF ( cutest_status /= 0 ) GO TO 910

     solv = 'NREK  '

!  evaluate the Hessian

     CALL CUTEST_udimsh_r( cutest_status, nnzh )
     IF ( cutest_status /= 0 ) GO TO 910
     H%n = n ; H%ne = nnzh
     CALL SMT_put( H%type, 'COORDINATE', smt_stat )
     ALLOCATE( H%row( nnzh ), H%col( nnzh ), H%val( nnzh ) )
     CALL CUTEST_ush_r( cutest_status, n, X0, H%ne, nnzh, H%val, H%row, H%col )
     IF ( cutest_status /= 0 ) GO TO 910

!  if required, select the linear solver

     IF ( TRIM( linear_solver ) == 'find' ) THEN

!  analyse H by calling sls to find its semi-bandwith

       CALL SLS_initialize( 'pbtr', SLS_data, SLS_control, SLS_inform )
       CALL SLS_analyse( H, SLS_data, SLS_control, SLS_inform )
       WRITE( 6, "( ' n = ', I0, ' semi-bandwidth = ', I0 )" )                 &
         n, SLS_inform%semi_bandwidth
       CALL SLS_terminate( SLS_data, SLS_control, SLS_inform )
       IF ( SLS_inform%semi_bandwidth <= 5 ) THEN
         linear_solver = 'pbtr'
       ELSE
         IF ( TRIM( linear_solver ) /= 'ignore' ) linear_solver = 'ma57'
       END IF
     END IF

!  provide a default scaling matrix

     S%n = n ; S%ne = n
!    CALL SMT_put( S%type, 'COORDINATE', smt_stat ) ! storage for S
     CALL SMT_put( S%type, 'DIAGONAL', smt_stat ) ! storage for S
     ALLOCATE( S%row( n ), S%col( n ), S%val( n ) )
     DO i = 1, n
       S%row( i ) = i ; S%col( i ) = i ; S%val( i ) = 1.0_rp_
     END DO

!    WRITE(6,*) ' H(i,j,val)'
!    DO i = 1, nnzh
!      WRITE(6,*) H%row( i ), H%col( i ), H%val( i )
!    END DO

!  provide default radii

     ALLOCATE( WEIGHT( m ) )
     WEIGHT( 1 ) = weight_initial
     DO k = 2, m
       WEIGHT( k ) = WEIGHT( k - 1 ) * control%increase
     END DO

     IF ( run_nrek ) THEN

!  if required, open a file for the results

     IF ( write_result_summary ) THEN
       INQUIRE( FILE = nrek_rfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( nrek_rfiledevice, FILE = nrek_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
       ELSE
          OPEN( nrek_rfiledevice, FILE = nrek_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( errout, 2030 ) iores, nrek_rfilename
         STOP
       END IF
       WRITE( nrek_rfiledevice, "( A10 )" ) pname
     END IF
     IF ( TRIM( linear_solver ) /= 'ignore' ) THEN
       control%linear_solver = linear_solver
       control%linear_solver_for_S = linear_solver
     END IF

!  loop over m instances

     DO k = 1, m

!  solve the problem

       CALL CLOCK_time( clock_start )
       inform%time%clock_total = 0.0_rp_
       control%new_weight = k > 1
       CALL NREK_solve( n, H, G, power, WEIGHT( k ), X, data, control, inform )
!                       S = S )
       IF ( k < m ) WEIGHT( k + 1 ) = inform%next_weight

       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "( /, ' NREK used ' )" )
       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "(  ' non-zeros and fill-in (H) = ', I0, 1X, I0,  &
        &    ', linear solver: ', A )" ) nnzh,                                 &
           inform%SLS_inform%entries_in_factors,                               &
           TRIM( control%linear_solver )
!$      n_threads = OMP_GET_MAX_THREADS( )
       IF ( k == 1 .OR. ( control%print_level > 0 .AND. control%out > 0 ) )    &
         WRITE( out, "( ' number of threads = ', I0 )" ) n_threads
       CALL CLOCK_time( clock_total )
       clock_total = clock_total - clock_start

!  if required, append results to a file

       IF ( write_result_summary ) THEN
         IF ( k == 1 ) BACKSPACE( nrek_rfiledevice )
         IF ( inform%status == 0 ) THEN
           WRITE( nrek_rfiledevice, 2040 ) pname, n, inform%obj,               &
             inform%multiplier, inform%iter, clock_total, inform%status,       &
             weight( k )
!write(nrek_rfiledevice,*) ' nrek time ', inform%time%clock_total
         ELSE
           WRITE( nrek_rfiledevice, 2040 ) pname, n, inform%obj,               &
             inform%multiplier, inform%iter, - clock_total, inform%status,     &
             weight( k )
         END IF
!write(6,*) ' nrek time ', inform%time%clock_total
       END IF

!  if required, write the solution

       IF ( control%print_level > 0 .AND. control%out > 0 ) THEN
         l = 2
         IF ( fulsol ) l = n
         IF ( control%print_level >= 10 ) l = n

         WRITE( errout, 2000 ) TRIM( solv )
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1 ; ic = MIN( l, n )
           ELSE
             IF ( ic < n - l ) WRITE( errout, 2010 )
             ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
           END IF
           DO i = ir, ic
             WRITE( errout, 2020 ) i, VNAMES( i ), X( i )
           END DO
         END DO
       END IF

       IF ( k == 1 .OR. ( control%print_level > 0 .AND. control%out > 0 ) )    &
         WRITE( errout, 2060 )
       IF ( inform%status == 0 ) THEN
         WRITE( errout, 2050 ) pname, n, inform%obj, inform%multiplier,        &
           inform%iter, inform%time%clock_total, inform%status, solv,          &
           weight( k )
       ELSE
         WRITE( errout, 2050 ) pname, n, inform%obj, inform%multiplier,        &
           inform%iter, - inform%time%clock_total, inform%status, solv,        &
           weight( k )
       END IF

       IF ( write_solution .AND.                                               &
           ( inform%status == 0  .OR. inform%status == - 10 ) ) THEN
         INQUIRE( FILE = nrek_sfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( nrek_sfiledevice, FILE = nrek_sfilename, FORM = 'FORMATTED', &
                STATUS = 'OLD', IOSTAT = iores )
         ELSE
            OPEN( nrek_sfiledevice, FILE = nrek_sfilename, FORM = 'FORMATTED', &
                 STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( out, 2030 ) iores, nrek_sfilename ; STOP ; END IF
         WRITE( nrek_sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :  ',&
        &       A, /, ' Objective:', ES24.16 )" ) pname, solv, inform%obj
         WRITE( nrek_sfiledevice, 2000 ) TRIM( solv )
         DO i = 1, n
           WRITE( nrek_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
         END DO
       END IF
     END DO
     WRITE( 6, "( 1X, A, ' time(anal, fact, sol, nz, fill) =',                 &
    &             3F6.2, 1X, I0, 1X, I0 )" )                                   &
       TRIM( inform%SLS_inform%solver ), inform%time%clock_analyse,            &
       inform%time%clock_factorize, inform%time%clock_solve,                   &
       nnzh, inform%SLS_inform%entries_in_factors
     CALL NREK_terminate( data, control, inform )
     END IF

!  now compare with rqs if desired

     IF ( run_rqs ) THEN
       IF ( write_result_summary ) THEN
         INQUIRE( FILE = rqs_rfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( rqs_rfiledevice, FILE = rqs_rfilename, FORM = 'FORMATTED',   &
                  STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
         ELSE
            OPEN( rqs_rfiledevice, FILE = rqs_rfilename, FORM = 'FORMATTED',   &
                  STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( errout, 2030 ) iores, rqs_rfilename
           STOP
         END IF
         WRITE( rqs_rfiledevice, "( A10 )" ) pname
       END IF

       solv = 'RQS   '
       RQS_inform%time%clock_total = 0.0_rp_
       CALL RQS_initialize( RQS_data, RQS_control, RQS_inform )
       IF ( is_specfile ) CALL RQS_read_specfile( RQS_control, input_specfile )
       IF ( TRIM( linear_solver ) /= 'ignore' ) THEN
         RQS_control%symmetric_linear_solver = linear_solver
         RQS_control%definite_linear_solver = linear_solver
       END IF

!    loop over m instances

       DO k = 1, m

!    solve the problem

         CALL CLOCK_time( clock_start )
         IF ( k > 1 ) THEN
           RQS_control%lower = RQS_inform%multiplier
           RQS_control%initial_multiplier = RQS_control%lower
           RQS_control%use_initial_multiplier = .TRUE.
         END IF
         CALL RQS_solve( n, power, WEIGHT( k ), f, G, H, X, RQS_data,          &
                         RQS_control, RQS_inform )

         IF ( RQS_control%print_level > 0 .AND. RQS_control%out > 0 )          &
           WRITE( RQS_control%out, "( /, ' RQS used ' )" )
         IF ( RQS_control%print_level > 0 .AND. RQS_control%out > 0 )          &
           WRITE( RQS_control%out, "(  ' non-zeros and fill-in (H) = ',        &
          &    I0, 1X, I0, ', linear solver: ', A )" ) nnzh,                   &
           RQS_inform%SLS_inform%entries_in_factors,                           &
           TRIM( RQS_control%definite_linear_solver )

         CALL CLOCK_time( clock_total )
         clock_total = clock_total - clock_start

!    if required, append results to a file

         IF ( write_result_summary ) THEN
           IF ( k == 1 ) BACKSPACE( rqs_rfiledevice )
           IF ( RQS_inform%status == 0 ) THEN
             WRITE( rqs_rfiledevice, 2040 ) pname, n, RQS_inform%obj,          &
               RQS_inform%multiplier, RQS_inform%factorizations,               &
               clock_total, RQS_inform%status, weight( k )
           ELSE
             WRITE( rqs_rfiledevice, 2040 ) pname, n, RQS_inform%obj,          &
               RQS_inform%multiplier, RQS_inform%factorizations,               &
               - RQS_inform%time%clock_total, RQS_inform%status, weight( k )
           END IF
         END IF

!    if required, write the solution

         IF ( RQS_control%print_level > 0 .AND. RQS_control%out > 0 ) THEN
           l = 2
           IF ( fulsol ) l = n
           IF ( RQS_control%print_level >= 10 ) l = n

           WRITE( errout, 2000 ) TRIM( solv )
           DO j = 1, 2
             IF ( j == 1 ) THEN
               ir = 1 ; ic = MIN( l, n )
             ELSE
               IF ( ic < n - l ) WRITE( errout, 2010 )
               ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
             END IF
             DO i = ir, ic
               WRITE( errout, 2020 ) i, VNAMES( i ), X( i )
             END DO
           END DO
         END IF

         IF ( k == 1 .OR. ( RQS_control%print_level > 0 .AND.                  &
                            RQS_control%out > 0 ) ) WRITE( errout, 2070 )
         IF ( RQS_inform%status == 0 ) THEN
           WRITE( errout, 2050 ) pname, n, RQS_inform%obj,                     &
             RQS_inform%multiplier, RQS_inform%factorizations,                 &
             clock_total, RQS_inform%status, solv, weight( k )
         ELSE
           WRITE( errout, 2050 ) pname, n, RQS_inform%obj,                     &
             RQS_inform%multiplier, RQS_inform%factorizations,                 &
             - clock_total, RQS_inform%status, solv, weight( k )
         END IF

         IF ( write_solution .AND.                                             &
             ( RQS_inform%status == 0  .OR. RQS_inform%status == - 10 ) ) THEN
           INQUIRE( FILE = rqs_sfilename, EXIST = filexx )
           IF ( filexx ) THEN
              OPEN( rqs_sfiledevice, FILE = rqs_sfilename, FORM = 'FORMATTED', &
                  STATUS = 'OLD', IOSTAT = iores )
           ELSE
              OPEN( rqs_sfiledevice, FILE = rqs_sfilename, FORM = 'FORMATTED', &
                   STATUS = 'NEW', IOSTAT = iores )
           END IF
           IF ( iores /= 0 ) THEN
             write( out, 2030 ) iores, rqs_sfilename ; STOP ; END IF
           WRITE( rqs_sfiledevice, "( /, ' Problem:    ', A10, /,              &
          &       ' Solver :  ', A, /, ' Objective:', ES24.16 )" )             &
             pname, solv, RQS_inform%obj
           WRITE( rqs_sfiledevice, 2000 ) TRIM( solv )
           DO i = 1, n
             WRITE( rqs_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
           END DO
         END IF
       END DO
       WRITE( 6, "( 1X, A, ' time(anal, fact, sol, nz, fill) =',               &
      &             3F6.2, 1X, I0, 1X, I0 )" )                                 &
         TRIM( inform%SLS_inform%solver ), RQS_inform%time%clock_analyse,      &
         RQS_inform%time%clock_factorize, RQS_inform%time%clock_solve,         &
         nnzh, RQS_inform%SLS_inform%entries_in_factors
       CALL RQS_terminate( RQS_data, RQS_control, RQS_inform )
     END IF

!  now compare with glrt if desired

     IF ( run_glrt ) THEN
       IF ( write_result_summary ) THEN
         INQUIRE( FILE = glrt_rfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( glrt_rfiledevice, FILE = glrt_rfilename, FORM = 'FORMATTED', &
                  STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
         ELSE
            OPEN( glrt_rfiledevice, FILE = glrt_rfilename, FORM = 'FORMATTED', &
                  STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( errout, 2030 ) iores, glrt_rfilename
           STOP
         END IF
         WRITE( glrt_rfiledevice, "( A10 )" ) pname
       END IF

       ALLOCATE( R( n ), VECTOR( n ), H_vector( n ) )
       CALL GLRT_initialize( GLRT_data, GLRT_control, GLRT_inform )
       IF ( is_specfile ) CALL GLRT_read_specfile( GLRT_control, input_specfile)
       solv = 'GLRT  '

!    loop over m instances

       DO k = 1, m

!    solve the problem

         CALL CLOCK_time( clock_start )
         R = G                ! The linear term is the gradient
         goth = .FALSE.
         IF ( k == 1 ) THEN
           GLRT_inform%status = 1
         ELSE
           GLRT_inform%status = 4
         END IF
         DO                  !  Iteration to find the minimizer
           CALL GLRT_solve( n, power, WEIGHT( k ), X, R, VECTOR,               &
                            GLRT_data, GLRT_control, GLRT_inform )
           SELECT CASE( GLRT_inform%status ) ! Branch as a result of status
           CASE( 2 )         ! Form the preconditioned gradient
           CASE( 3 )         ! Form the matrix-vector product
             CALL CUTEST_uhprod_r( cutest_status, n, goth, X0, VECTOR, H_vector)
             VECTOR = H_vector
             goth = .TRUE.
           CASE ( 5 )        !  Restart
             R = G
           CASE ( - 30, 0 )  !  Successful return
             IF ( GLRT_control%print_level > 0 .AND. GLRT_control%out > 0 )    &
              WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier',&
             &  ' = ', 2ES12.4 )" ) GLRT_inform%iter + GLRT_inform%iter_pass2, &
               f, GLRT_inform%multiplier
             EXIT
           CASE DEFAULT      !  Error returns
             WRITE( 6, "( ' GLRT_solve exit status = ', I6 ) " )               &
               GLRT_inform%status
             EXIT
          END SELECT
         END DO
         CALL CLOCK_time( clock_total )
         clock_total = clock_total - clock_start

         IF ( GLRT_control%print_level > 0 .AND. GLRT_control%out > 0 )        &
           WRITE( GLRT_control%out, "( /, ' GLRT used ' )" )

!    If required, append results to a file,

         IF ( write_result_summary ) THEN
           IF ( k == 1 ) BACKSPACE( glrt_rfiledevice )
           IF ( GLRT_inform%status == 0 ) THEN
             WRITE( glrt_rfiledevice, 2080 ) pname, n, f,                      &
               GLRT_inform%multiplier, GLRT_inform%iter +                      &
               GLRT_inform%iter_pass2, clock_total, GLRT_inform%status,        &
               WEIGHT( k )
           ELSE
             WRITE( glrt_rfiledevice, 2080 ) pname, n, f,                      &
               GLRT_inform%multiplier, GLRT_inform%iter +                      &
               GLRT_inform%iter_pass2, - clock_total, GLRT_inform%status,      &
               WEIGHT( k )
           END IF
         END IF

!    if required, write the solution

         IF ( GLRT_control%print_level > 0 .AND. GLRT_control%out > 0 ) THEN
           l = 2
           IF ( fulsol ) l = n
           IF ( GLRT_control%print_level >= 10 ) l = n

           WRITE( errout, 2000 ) TRIM( solv )
           DO j = 1, 2
             IF ( j == 1 ) THEN
               ir = 1 ; ic = MIN( l, n )
             ELSE
               IF ( ic < n - l ) WRITE( errout, 2010 )
               ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
             END IF
             DO i = ir, ic
               WRITE( errout, 2020 ) i, VNAMES( i ), X( i )
             END DO
           END DO
         END IF

         IF ( k == 1 .OR. ( GLRT_control%print_level > 0 .AND.                 &
                            GLRT_control%out > 0 ) ) WRITE( errout, 2100 )
         IF ( GLRT_inform%status == 0 ) THEN
           WRITE( errout, 2090 ) pname, n, f,                                  &
             GLRT_inform%multiplier, GLRT_inform%iter,                         &
             GLRT_inform%iter_pass2, clock_total, GLRT_inform%status,          &
             solv, WEIGHT( k )
         ELSE
           WRITE( errout, 2090 ) pname, n, GLRT_inform%obj,                    &
             GLRT_inform%multiplier, GLRT_inform%iter,                         &
             GLRT_inform%iter_pass2, - clock_total, GLRT_inform%status,        &
             solv, WEIGHT( k )
         END IF

         IF ( write_solution .AND.                                             &
             ( GLRT_inform%status == 0  .OR. GLRT_inform%status == - 10 ) ) THEN
           INQUIRE( FILE = glrt_sfilename, EXIST = filexx )
           IF ( filexx ) THEN
              OPEN( glrt_sfiledevice, FILE = glrt_sfilename,FORM = 'FORMATTED',&
                  STATUS = 'OLD', IOSTAT = iores )
           ELSE
              OPEN( glrt_sfiledevice, FILE = glrt_sfilename,FORM = 'FORMATTED',&
                   STATUS = 'NEW', IOSTAT = iores )
           END IF
           IF ( iores /= 0 ) THEN
             write( out, 2030 ) iores, glrt_sfilename ; STOP ; END IF
           WRITE( glrt_sfiledevice, "( /, ' Problem:    ', A10, /,             &
          &       ' Solver :  ', A, /, ' Objective:', ES24.16 )" )             &
             pname, solv, GLRT_inform%obj
           WRITE( glrt_sfiledevice, 2000 ) TRIM( solv )
           DO i = 1, n
             WRITE( glrt_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
           END DO
         END IF
       END DO
       CALL GLRT_terminate( GLRT_data, GLRT_control, GLRT_inform )
       DEALLOCATE( R, VECTOR, H_vector )
     END IF

     IF ( is_specfile ) CLOSE( input_specfile )
     DEALLOCATE( H%type, H%val, H%row, H%col, X, X0, G, VNAMES, WEIGHT )
     DEALLOCATE( S%type, S%val, S%row, S%col )
     CALL CUTEST_cterminate_r( cutest_status )
     RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )          &
       cutest_status
     inform%status = - 98
     RETURN

!  Non-executable statements

 2000 FORMAT( /, ' Solver: ', A, /, ' Solution: ', /,                          &
               '      # name               value   ' )
 2010 FORMAT( 6X, '. .', 9X, ( 2X, 10( '.' ) ) )
 2020 FORMAT( I7, 1X, A10, ES22.14 )
 2030 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2040 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5, 1X, ES7.1 )
 2050 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5, 1X, A4, 1X, ES7.1 )
 2060 FORMAT( /, 'name           n  f               lambda    ',               &
                 '    iter     time stat alg   weight' )
 2070 FORMAT( /, 'name           n  f               lambda    ',               &
                 '     fac     time stat alg   weight' )
 2080 FORMAT( A10, I6,  ES16.8, ES13.6, I6, F9.2, I5, 1X, ES7.1 )
 2090 FORMAT( A10, I6, ES16.8, ES10.3, 2I5, F9.2, I5, 1X, A4, 1X, ES7.1 )
 2100 FORMAT( /, 'name           n  f              lambda ',                   &
                 '   iter itr2     time stat alg   weight' )

!  End of subroutine USE_NREK

     END SUBROUTINE USE_NREK

!  End of module USENREK

   END MODULE GALAHAD_USENREK_precision
