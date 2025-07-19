! THIS VERSION: GALAHAD 5.3 - 2025-07-03 AT 09:40 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ T R E K  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  May 20th 2025

   MODULE GALAHAD_USETREK_precision

!  This is the driver program for running TREK for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_KINDS_precision
     USE CUTEST_INTERFACE_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_TREK_precision
     USE GALAHAD_TRS_precision
     USE GALAHAD_GLTR_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_precision
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_TREK

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ T R E K   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_TREK( input )

!  Dummy argument

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( TREK_control_type ) :: control
     TYPE ( TREK_inform_type ) :: inform
     TYPE ( TREK_data_type ) :: data

     TYPE ( TRS_control_type ) :: TRS_control
     TYPE ( TRS_inform_type ) :: TRS_inform
     TYPE ( TRS_data_type ) :: TRS_data

     TYPE ( GLTR_control_type ) :: GLTR_control
     TYPE ( GLTR_inform_type ) :: GLTR_inform
     TYPE ( GLTR_data_type ) :: GLTR_data

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

!  Functions

!$   INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  Problem characteristics

     INTEGER ( KIND = ip_ ) :: n, nnzh
     INTEGER ( KIND = ip_ ) :: n_threads = 1
     REAL ( KIND = rp_ ) ::  f
     CHARACTER ( LEN = 10 ) :: pname
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X0, X_l, X_u, G
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: R, VECTOR, H_vector
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RADIUS
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES
     TYPE ( SMT_type ) :: H

!  Problem input characteristics

     LOGICAL :: filexx, is_specfile

!  Default values for specfile-defined parameters

     INTEGER ( KIND = ip_ ) :: trek_rfiledevice = 47
     INTEGER ( KIND = ip_ ) :: trek_sfiledevice = 62
     INTEGER ( KIND = ip_ ) :: trs_rfiledevice = 48
     INTEGER ( KIND = ip_ ) :: trs_sfiledevice = 63
     INTEGER ( KIND = ip_ ) :: gltr_rfiledevice = 49
     INTEGER ( KIND = ip_ ) :: gltr_sfiledevice = 64
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: trek_rfilename = 'TREKRES.d'
     CHARACTER ( LEN = 30 ) :: trek_sfilename = 'TREKSOL.d'
     CHARACTER ( LEN = 30 ) :: trs_rfilename = 'TRSRES.d'
     CHARACTER ( LEN = 30 ) :: trs_sfilename = 'TRSSOL.d'
     CHARACTER ( LEN = 30 ) :: gltr_rfilename = 'GLTRRES.d'
     CHARACTER ( LEN = 30 ) :: gltr_sfilename = 'GLTRSOL.d'
     REAL ( KIND = rp_ ) ::  radius_initial = 1.0_rp_
     INTEGER ( KIND = ip_ ) :: m = 1
     LOGICAL :: run_trs = .FALSE.
     LOGICAL :: run_gltr = .FALSE.
!    LOGICAL :: one_norm = .TRUE.

!  Output file characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
     INTEGER ( KIND = ip_ ) :: out  = 6
     INTEGER ( KIND = ip_ ) :: errout = 6
     CHARACTER ( LEN =  6 ) :: solv

!  Specfile characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 20
     CHARACTER ( LEN = 16 ) :: specname = 'RUNTREK'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNTREK.SPC'

!  ------------------ Open the specfile for trek ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword = 'write-problem-data'
       spec( 2 )%keyword = 'write-result-summary'
       spec( 3 )%keyword = 'trek-result-summary-file-name'
       spec( 4 )%keyword = 'trek-result-summary-file-device'
       spec( 5 )%keyword = 'trs-result-summary-file-name'
       spec( 6 )%keyword = 'trs-result-summary-file-device'
       spec( 7 )%keyword = 'gltr-result-summary-file-name'
       spec( 8 )%keyword = 'gltr-result-summary-file-device'
       spec( 9 )%keyword = 'print-full-solution'
       spec( 10 )%keyword = 'write-solution'
       spec( 11 )%keyword = 'trek-solution-file-name'
       spec( 12 )%keyword = 'trek-solution-file-device'
       spec( 13 )%keyword = 'trs-solution-file-name'
       spec( 14 )%keyword = 'trs-solution-file-device'
       spec( 15 )%keyword = 'gltr-solution-file-name'
       spec( 16 )%keyword = 'gltr-solution-file-device'
       spec( 17 )%keyword = 'radius'
       spec( 18 )%keyword = 'number-of-radii'
       spec( 19 )%keyword = 'compare-with-trs'
       spec( 20 )%keyword = 'compare-with-gltr'

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_logical( spec( 2 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 3 ), trek_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), trek_rfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 5 ), trs_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 6 ), trs_rfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 7 ), gltr_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 8 ), gltr_rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 9 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 10 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 11 ), trek_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 12 ), trek_sfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 13 ), trs_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 14 ), trs_sfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 15 ), gltr_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 16 ), gltr_sfiledevice, errout )
       CALL SPECFILE_assign_real( spec( 17 ), radius_initial, errout )
       CALL SPECFILE_assign_integer( spec( 18 ), m, errout )
       CALL SPECFILE_assign_logical( spec( 19 ), run_trs, errout )
       CALL SPECFILE_assign_logical( spec( 20 ), run_gltr, errout )
     END IF

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2025' )

!  Set up data for next problem

     CALL TREK_initialize( data, control, inform )
     IF ( is_specfile ) CALL TREK_read_specfile( control, input_specfile )

!  Read the initial point and bounds

     CALL CUTEST_udimen_r( cutest_status, input, n )
     IF ( cutest_status /= 0 ) GO TO 910

     ALLOCATE( X( n ), X0( n ), X_l( n ), X_u( n ), G( n ), VNAMES( n ) )
     CALL CUTEST_usetup_r( cutest_status, input, control%error, io_buffer,     &
                           n, X0, X_l, X_u )
     IF ( cutest_status /= 0 ) GO TO 910
     DEALLOCATE( X_l, X_u )

!  Read the problem and variable names

     CALL CUTEST_unames_r( cutest_status, n, pname, VNAMES )
     IF ( cutest_status /= 0 ) GO TO 910

!  Set f to zero

    f = zero

!  Evaluate the gradient

     CALL CUTEST_ugr_r( cutest_status, n, X0, G )
     IF ( cutest_status /= 0 ) GO TO 910

     solv = 'TREK  '

!  Evaluate the Hessian

     CALL CUTEST_udimsh_r( cutest_status, nnzh )
     IF ( cutest_status /= 0 ) GO TO 910
     H%n = n ; H%ne = nnzh
     CALL SMT_put( H%type, 'COORDINATE', smt_stat )
     ALLOCATE( H%row( nnzh ), H%col( nnzh ), H%val( nnzh ) )
     CALL CUTEST_ush_r( cutest_status, n, X0, H%ne, nnzh, H%val, H%row, H%col )
     IF ( cutest_status /= 0 ) GO TO 910

!    WRITE(6,*) ' H(i,j,val)'
!    DO i = 1, nnzh
!      WRITE(6,*) H%row( i ), H%col( i ), H%val( i )
!    END DO

     ALLOCATE( RADIUS( m ) )
     RADIUS( 1 ) = radius_initial

!  If required, open a file for the results

     IF ( write_result_summary ) THEN
       INQUIRE( FILE = trek_rfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( trek_rfiledevice, FILE = trek_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
       ELSE
          OPEN( trek_rfiledevice, FILE = trek_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( errout, 2030 ) iores, trek_rfilename
         STOP
       END IF
       WRITE( trek_rfiledevice, "( A10 )" ) pname
     END IF

!  loop over m instances

     DO k = 1, m

!  Solve the problem

       inform%time%clock_total = 0.0_rp_
       CALL TREK_solve( n, H, G, RADIUS( k ), X, data, control, inform,        &
!                       new_values = .TRUE. )
                        resolve = k > 1 )
       IF ( k < m ) RADIUS( k + 1 ) = inform%next_radius

       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "( /, ' TREK used ' )" )
       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "(  ' non-zeros and fill-in (H) = ', I0, 1X, I0,  &
        &    ', linear solver: ', A )" ) nnzh,                                 &
           inform%SLS_inform%entries_in_factors, TRIM( control%solver )
!$      n_threads = OMP_GET_MAX_THREADS( )
       IF ( k == 1 .OR. ( control%print_level > 0 .AND. control%out > 0 ) )    &
         WRITE( out, "( ' number of threads = ', I0 )" ) n_threads

!  If required, append results to a file,

       IF ( write_result_summary ) THEN
         BACKSPACE( trek_rfiledevice )
         IF ( inform%status == 0 ) THEN
           WRITE( trek_rfiledevice, 2040 ) pname, n, inform%obj,               &
             inform%multiplier,                                                &
             inform%iter, inform%time%clock_total, inform%status
         ELSE
           WRITE( trek_rfiledevice, 2040 ) pname, n, inform%obj,               &
             inform%multiplier,                                                &
             inform%iter, - inform%time%clock_total, inform%status
         END IF
       END IF

!  If required, write the solution

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
           radius( k )
       ELSE
         WRITE( errout, 2050 ) pname, n, inform%obj, inform%multiplier,        &
           inform%iter, - inform%time%clock_total, inform%status, solv,        &
           radius( k )
       END IF

       IF ( write_solution .AND.                                               &
           ( inform%status == 0  .OR. inform%status == - 10 ) ) THEN
         INQUIRE( FILE = trek_sfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( trek_sfiledevice, FILE = trek_sfilename, FORM = 'FORMATTED', &
                STATUS = 'OLD', IOSTAT = iores )
         ELSE
            OPEN( trek_sfiledevice, FILE = trek_sfilename, FORM = 'FORMATTED', &
                 STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( out, 2030 ) iores, trek_sfilename ; STOP ; END IF
         WRITE( trek_sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :  ',&
        &       A, /, ' Objective:', ES24.16 )" ) pname, solv, inform%obj
         WRITE( trek_sfiledevice, 2000 ) TRIM( solv )
         DO i = 1, n
           WRITE( trek_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
         END DO
       END IF
     END DO
     WRITE( 6, "( 1X, A, ' time(analyse, factorize, solve) = ', 3F6.2 )" )     &
       TRIM( inform%SLS_inform%solver ), inform%time%clock_analyse,            &
       inform%time%clock_factorize, inform%time%clock_solve
     CALL TREK_terminate( data, control, inform )

!  now compare with trs if desired

     IF ( run_trs ) THEN
       IF ( write_result_summary ) THEN
         INQUIRE( FILE = trs_rfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( trs_rfiledevice, FILE = trs_rfilename, FORM = 'FORMATTED',   &
                  STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
         ELSE
            OPEN( trs_rfiledevice, FILE = trs_rfilename, FORM = 'FORMATTED',   &
                  STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( errout, 2030 ) iores, trs_rfilename
           STOP
         END IF
         WRITE( trs_rfiledevice, "( A10 )" ) pname
       END IF

       solv = 'TRS   '
       TRS_inform%time%clock_total = 0.0_rp_
       CALL TRS_initialize( TRS_data, TRS_control, TRS_inform )
       IF ( is_specfile ) CALL TRS_read_specfile( TRS_control, input_specfile )

!    loop over m instances

       DO k = 1, m

!    Solve the problem

         IF ( k > 1 ) THEN
           TRS_control%lower = TRS_inform%multiplier
           TRS_control%initial_multiplier = TRS_control%lower
           TRS_control%use_initial_multiplier = .TRUE.
         END IF
         CALL TRS_solve( n, RADIUS( k ), f, G, H, X, TRS_data, TRS_control,    &
                         TRS_inform )

         IF ( TRS_control%print_level > 0 .AND. TRS_control%out > 0 )          &
           WRITE( TRS_control%out, "( /, ' TRS used ' )" )
         IF ( TRS_control%print_level > 0 .AND. TRS_control%out > 0 )          &
           WRITE( TRS_control%out, "(  ' non-zeros and fill-in (H) = ',        &
          &    I0, 1X, I0, ', linear solver: ', A )" ) nnzh,                   &
           TRS_inform%SLS_inform%entries_in_factors,                           &
           TRIM( TRS_control%definite_linear_solver )

!    If required, append results to a file,

         IF ( write_result_summary ) THEN
           BACKSPACE( trs_rfiledevice )
           IF ( TRS_inform%status == 0 ) THEN
             WRITE( trs_rfiledevice, 2040 ) pname, n, TRS_inform%obj,          &
               TRS_inform%multiplier, TRS_inform%factorizations,               &
               TRS_inform%time%clock_total, TRS_inform%status
           ELSE
             WRITE( trs_rfiledevice, 2040 ) pname, n, TRS_inform%obj,          &
               TRS_inform%multiplier, TRS_inform%factorizations,               &
               - TRS_inform%time%clock_total, TRS_inform%status
           END IF
         END IF

!    If required, write the solution

         IF ( TRS_control%print_level > 0 .AND. TRS_control%out > 0 ) THEN
           l = 2
           IF ( fulsol ) l = n
           IF ( TRS_control%print_level >= 10 ) l = n

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

         IF ( k == 1 .OR. ( TRS_control%print_level > 0 .AND.                  &
                            TRS_control%out > 0 ) ) WRITE( errout, 2070 )
         IF ( TRS_inform%status == 0 ) THEN
           WRITE( errout, 2050 ) pname, n, TRS_inform%obj,                     &
             TRS_inform%multiplier, TRS_inform%factorizations,                 &
             TRS_inform%time%clock_total, TRS_inform%status, solv,             &
             radius( k )
         ELSE
           WRITE( errout, 2050 ) pname, n, TRS_inform%obj,                     &
             TRS_inform%multiplier, TRS_inform%factorizations,                 &
             - TRS_inform%time%clock_total, TRS_inform%status, solv,           &
             radius( k )
         END IF

         IF ( write_solution .AND.                                             &
             ( TRS_inform%status == 0  .OR. TRS_inform%status == - 10 ) ) THEN
           INQUIRE( FILE = trs_sfilename, EXIST = filexx )
           IF ( filexx ) THEN
              OPEN( trs_sfiledevice, FILE = trs_sfilename, FORM = 'FORMATTED', &
                  STATUS = 'OLD', IOSTAT = iores )
           ELSE
              OPEN( trs_sfiledevice, FILE = trs_sfilename, FORM = 'FORMATTED', &
                   STATUS = 'NEW', IOSTAT = iores )
           END IF
           IF ( iores /= 0 ) THEN
             write( out, 2030 ) iores, trs_sfilename ; STOP ; END IF
           WRITE( trs_sfiledevice, "( /, ' Problem:    ', A10, /,              &
          &       ' Solver :  ', A, /, ' Objective:', ES24.16 )" )             &
             pname, solv, TRS_inform%obj
           WRITE( trs_sfiledevice, 2000 ) TRIM( solv )
           DO i = 1, n
             WRITE( trs_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
           END DO
         END IF
       END DO
       WRITE( 6, "( 1X, A, ' time(analyse, factorize, solve) = ', 3F6.2 )" )   &
         TRIM( inform%SLS_inform%solver ), TRS_inform%time%clock_analyse,      &
         TRS_inform%time%clock_factorize, TRS_inform%time%clock_solve
       CALL TRS_terminate( TRS_data, TRS_control, TRS_inform )
     END IF

!  now compare with gltr if desired

     IF ( run_gltr ) THEN
       IF ( write_result_summary ) THEN
         INQUIRE( FILE = gltr_rfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( gltr_rfiledevice, FILE = gltr_rfilename, FORM = 'FORMATTED', &
                  STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
         ELSE
            OPEN( gltr_rfiledevice, FILE = gltr_rfilename, FORM = 'FORMATTED', &
                  STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( errout, 2030 ) iores, gltr_rfilename
           STOP
         END IF
         WRITE( gltr_rfiledevice, "( A10 )" ) pname
       END IF

       ALLOCATE( R( n ), VECTOR( n ), H_vector( n ) )
       CALL GLTR_initialize( GLTR_data, GLTR_control, GLTR_inform )
       IF ( is_specfile ) CALL GLTR_read_specfile( GLTR_control, input_specfile)
       solv = 'GLTR  '

!    loop over m instances

       DO k = 1, m

!    Solve the problem

         CALL CLOCK_time( clock_start )
         R = G                ! The linear term is the gradient
         goth = .FALSE.
         IF ( k == 1 ) THEN
           GLTR_inform%status = 1
         ELSE
           GLTR_inform%status = 4
         END IF
         DO                  !  Iteration to find the minimizer
           CALL GLTR_solve( n, RADIUS( k ), f, X, R, VECTOR,                   &
                            GLTR_data, GLTR_control, GLTR_inform )
           SELECT CASE( GLTR_inform%status ) ! Branch as a result of status
           CASE( 2 )         ! Form the preconditioned gradient
           CASE( 3 )         ! Form the matrix-vector product
             CALL CUTEST_uhprod_r( cutest_status, n, goth, X, VECTOR, H_vector )
             VECTOR = H_vector
             goth = .TRUE.
           CASE ( 5 )        !  Restart
             R = G
           CASE ( - 30, 0 )  !  Successful return
             IF ( GLTR_control%print_level > 0 .AND. GLTR_control%out > 0 )    &
              WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier',&
             &  ' = ', 2ES12.4 )" ) GLTR_inform%iter + GLTR_inform%iter_pass2, &
               f, GLTR_inform%multiplier
             EXIT
           CASE DEFAULT      !  Error returns
             WRITE( 6, "( ' GLTR_solve exit status = ', I6 ) " )               &
               GLTR_inform%status
             EXIT
          END SELECT
         END DO
         CALL CLOCK_time( clock_total )
         clock_total = clock_total - clock_start

         IF ( GLTR_control%print_level > 0 .AND. GLTR_control%out > 0 )        &
           WRITE( GLTR_control%out, "( /, ' GLTR used ' )" )

!    If required, append results to a file,

         IF ( write_result_summary ) THEN
           BACKSPACE( gltr_rfiledevice )
           IF ( GLTR_inform%status == 0 ) THEN
             WRITE( gltr_rfiledevice, 2080 ) pname, n, f,                      &
               GLTR_inform%multiplier, GLTR_inform%iter,                       &
               GLTR_inform%iter_pass2, clock_total, GLTR_inform%status
           ELSE
             WRITE( gltr_rfiledevice, 2080 ) pname, n, f,                      &
               GLTR_inform%multiplier, GLTR_inform%iter,                       &
               GLTR_inform%iter_pass2, - clock_total, GLTR_inform%status
           END IF
         END IF

!    If required, write the solution

         IF ( GLTR_control%print_level > 0 .AND. GLTR_control%out > 0 ) THEN
           l = 2
           IF ( fulsol ) l = n
           IF ( GLTR_control%print_level >= 10 ) l = n

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

         IF ( k == 1 .OR. ( GLTR_control%print_level > 0 .AND.                 &
                            GLTR_control%out > 0 ) ) WRITE( errout, 2100 )
         IF ( GLTR_inform%status == 0 ) THEN
           WRITE( errout, 2090 ) pname, n, f,                                  &
             GLTR_inform%multiplier, GLTR_inform%iter,                         &
             GLTR_inform%iter_pass2, clock_total, GLTR_inform%status,          &
             solv, RADIUS( k )
         ELSE
           WRITE( errout, 2090 ) pname, n, GLTR_inform%obj,                    &
             GLTR_inform%multiplier, GLTR_inform%iter,                         &
             GLTR_inform%iter_pass2, - clock_total, GLTR_inform%status,        &
             solv, RADIUS( k )
         END IF

         IF ( write_solution .AND.                                             &
             ( GLTR_inform%status == 0  .OR. GLTR_inform%status == - 10 ) ) THEN
           INQUIRE( FILE = gltr_sfilename, EXIST = filexx )
           IF ( filexx ) THEN
              OPEN( gltr_sfiledevice, FILE = gltr_sfilename,FORM = 'FORMATTED',&
                  STATUS = 'OLD', IOSTAT = iores )
           ELSE
              OPEN( gltr_sfiledevice, FILE = gltr_sfilename,FORM = 'FORMATTED',&
                   STATUS = 'NEW', IOSTAT = iores )
           END IF
           IF ( iores /= 0 ) THEN
             write( out, 2030 ) iores, gltr_sfilename ; STOP ; END IF
           WRITE( gltr_sfiledevice, "( /, ' Problem:    ', A10, /,             &
          &       ' Solver :  ', A, /, ' Objective:', ES24.16 )" )             &
             pname, solv, GLTR_inform%obj
           WRITE( gltr_sfiledevice, 2000 ) TRIM( solv )
           DO i = 1, n
             WRITE( gltr_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
           END DO
         END IF
       END DO
       CALL GLTR_terminate( GLTR_data, GLTR_control, GLTR_inform )
       DEALLOCATE( R, VECTOR, H_vector )
     END IF

     IF ( is_specfile ) CLOSE( input_specfile )
     DEALLOCATE( H%val, H%row, H%col, X, X0, G, VNAMES, RADIUS )
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
 2040 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5 )
 2050 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5, 1X, A4, 1X, ES7.1 )
 2060 FORMAT( /, 'name           n  f               lambda    ',               &
                 '    iter     time stat alg   radius' )
 2070 FORMAT( /, 'name           n  f               lambda    ',               &
                 '     fac     time stat alg   radius' )
 2080 FORMAT( A10, I6, 2ES16.8, 2I6, F9.2, I5 )
 2090 FORMAT( A10, I6, ES16.8, ES10.3, 2I5, F9.2, I5, 1X, A4, 1X, ES7.1 )
 2100 FORMAT( /, 'name           n  f              lambda ',                   &
                 '   iter itr2     time stat alg   radius' )

!  End of subroutine USE_TREK

     END SUBROUTINE USE_TREK

!  End of module USETREK

   END MODULE GALAHAD_USETREK_precision
