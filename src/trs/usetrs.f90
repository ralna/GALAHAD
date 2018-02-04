! THIS VERSION: GALAHAD 2.5 - 09/02/2013 AT 17:50 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ T R S  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  November 24th 2008

   MODULE GALAHAD_USETRS_double

!  This is the driver program for running TRS for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE CUTEst_interface_double
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_TRS_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_double
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_TRS

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ T R S   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_TRS( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( TRS_control_type ) :: control
     TYPE ( TRS_inform_type ) :: inform
     TYPE ( TRS_data_type ) :: data

!------------------------------------
!   L o c a l   P a r a m e t e r s
!------------------------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
!    REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!----------------------------------
!   L o c a l   V a r i a b l e s
!----------------------------------

     INTEGER :: iores, i, j, ir, ic, l, smt_stat, iter, info, cutest_status
     REAL ( KIND = wp ) :: clock_now, clock_record
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W1, W2, W3

!  Functions

!$   INTEGER :: OMP_GET_MAX_THREADS

!  Problem characteristics

     INTEGER :: n, nnzh
     INTEGER :: n_threads = 1
     REAL ( KIND = wp ) ::  f
     CHARACTER ( LEN = 10 ) :: pname
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, X0, X_l, X_u, G
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: H_dense
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES
     TYPE ( SMT_type ) :: H

!  Problem input characteristics

     LOGICAL :: filexx, is_specfile

!  Default values for specfile-defined parameters

     INTEGER :: trs_rfiledevice = 47
     INTEGER :: ms_rfiledevice = 48
     INTEGER :: trs_sfiledevice = 62
     INTEGER :: ms_sfiledevice = 63
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: trs_rfilename = 'TRSRES.d'
     CHARACTER ( LEN = 30 ) :: trs_sfilename = 'TRSSOL.d'
     CHARACTER ( LEN = 30 ) :: ms_rfilename = 'MSRES.d'
     CHARACTER ( LEN = 30 ) :: ms_sfilename = 'MSSOL.d'
     REAL ( KIND = wp ) ::  radius = 1.0_wp
     LOGICAL :: more_sorensen = .FALSE.
     LOGICAL :: trs = .TRUE.
!    LOGICAL :: one_norm = .TRSE.

!  Output file characteristics

     INTEGER, PARAMETER :: io_buffer = 11
     INTEGER :: out  = 6
     INTEGER :: errout = 6
     CHARACTER ( LEN =  6 ) :: solv

!  Specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 15
     CHARACTER ( LEN = 16 ) :: specname = 'RUNTRS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNTRS.SPC'

!  ------------------ Open the specfile for trs ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword  = 'write-problem-data'
       spec( 2 )%keyword  = 'write-result-summary'
       spec( 3 )%keyword  = 'trs-result-summary-file-name'
       spec( 4 )%keyword = 'trs-result-summary-file-device'
       spec( 5 )%keyword  = 'ms-result-summary-file-name'
       spec( 6 )%keyword = 'ms-result-summary-file-device'
       spec( 7 )%keyword  = 'print-full-solution'
       spec( 8 )%keyword  = 'write-solution'
       spec( 9 )%keyword  = 'trs-solution-file-name'
       spec( 10 )%keyword  = 'trs-solution-file-device'
       spec( 11 )%keyword  = 'ms-solution-file-name'
       spec( 12 )%keyword  = 'ms-solution-file-device'
       spec( 13 )%keyword = 'radius'
       spec( 14 )%keyword = 'use-more-sorensen'
       spec( 15 )%keyword = 'use-trs'

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_logical( spec( 2 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 3 ), trs_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), trs_rfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 5 ), ms_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 6 ), ms_rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 7 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 8 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 9 ), trs_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 10 ), trs_sfiledevice, errout )
       CALL SPECFILE_assign_string ( spec( 11 ), ms_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 12 ), ms_sfiledevice, errout )
       CALL SPECFILE_assign_real( spec( 13 ), radius, errout )
       CALL SPECFILE_assign_logical( spec( 14 ), more_sorensen, errout )
       CALL SPECFILE_assign_logical( spec( 15 ), trs, errout )
     END IF

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2008' )

!  Set up data for next problem

     CALL TRS_initialize( data, control, inform )
     IF ( is_specfile ) CALL TRS_read_specfile( control, input_specfile )
     IF ( is_specfile ) CLOSE( input_specfile )

!  Read the initial point and bounds

     CALL CUTEST_udimen( cutest_status, input, n )
     IF ( cutest_status /= 0 ) GO TO 910
     IF ( n > 2000 .AND. more_sorensen ) THEN
       WRITE( errout, "( ' more than 2000 variables ... stopping ' )" )
       STOP
     END IF

     ALLOCATE( X( n ), X0( n ), X_l( n ), X_u( n ), G( n ), VNAMES( n ) )
     CALL CUTEST_usetup( cutest_status, input, control%error, io_buffer,       &
                         n, X0, X_l, X_u )
     IF ( cutest_status /= 0 ) GO TO 910
     DEALLOCATE( X_l, X_u )

!  Read the problem and variable names

     CALL CUTEST_unames( cutest_status, n, pname, VNAMES )
     IF ( cutest_status /= 0 ) GO TO 910

!  Set f to zero

    f = zero

!  Evaluate the gradient

     CALL CUTEST_ugr( cutest_status, n, X0, G )
     IF ( cutest_status /= 0 ) GO TO 910

!  Use TRS

     IF ( trs ) THEN
       solv = 'TRS   '

!  Evaluate the Hessian

       CALL CUTEST_udimsh( cutest_status, nnzh )
       IF ( cutest_status /= 0 ) GO TO 910
       H%ne = nnzh
       CALL SMT_put( H%type, 'COORDINATE', smt_stat )
       ALLOCATE( H%row( nnzh ), H%col( nnzh ), H%val( nnzh ) )
       CALL CUTEST_ush( cutest_status, n, X0, H%ne, nnzh, H%val, H%row, H%col )
       IF ( cutest_status /= 0 ) GO TO 910

!  If required, open a file for the results

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

!  Solve the problem

       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "( ' TRS used ', / )" )
!g = g /  ( ten ** 12 )
!H%val = H%val / ( ten ** 12 )
       CALL TRS_solve( n, radius, f, G, H, X, data, control, inform )
       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "( /, ' TRS used ' )" )
       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "( /, ' non-zeros and fill-in ', I0, 1X, I0,      &
        &    ', solver: ', A )" ) nnzh, inform%SLS_inform%entries_in_factors,  &
           TRIM( control%definite_linear_solver )
!$      n_threads = OMP_GET_MAX_THREADS( )
        WRITE( out, "( ' number of threads = ', I0 )" ) n_threads

!  If required, append results to a file,

       IF ( write_result_summary ) THEN
         BACKSPACE( trs_rfiledevice )
         IF ( inform%status == 0 ) THEN
           WRITE( trs_rfiledevice, 2040 ) pname, n, inform%obj,                &
             inform%multiplier,                                                &
             inform%factorizations, inform%time%clock_total, inform%status
         ELSE
           WRITE( trs_rfiledevice, 2040 ) pname, n, inform%obj,                &
             inform%multiplier,                                                &
             inform%factorizations, - inform%time%clock_total, inform%status
         END IF
       END IF

!  If required, write the solution

       IF ( control%print_level > 0 .AND. control%out > 0 ) THEN
         l = 2
         IF ( fulsol ) l = n
         IF ( control%print_level >= 10 ) l = n

         WRITE( errout, 2000 )
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

       WRITE( errout, 2060 )
       IF ( inform%status == 0 ) THEN
         WRITE( errout, 2050 ) pname, n, inform%obj,                           &
             inform%multiplier,                                                &
           inform%factorizations, inform%time%clock_total, inform%status, solv
       ELSE
         WRITE( errout, 2050 ) pname, n, inform%obj,                           &
             inform%multiplier,                                                &
           inform%factorizations, - inform%time%clock_total, inform%status, solv
       END IF

       IF ( write_solution .AND.                                               &
           ( inform%status == 0  .OR. inform%status == - 10 ) ) THEN
         INQUIRE( FILE = trs_sfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( trs_sfiledevice, FILE = trs_sfilename, FORM = 'FORMATTED',   &
                STATUS = 'OLD', IOSTAT = iores )
         ELSE
            OPEN( trs_sfiledevice, FILE = trs_sfilename, FORM = 'FORMATTED',   &
                 STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( out, 2030 ) iores, trs_sfilename ; STOP ; END IF
         WRITE( trs_sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ',&
        &       A, /, ' Objective:', ES24.16 )" ) pname, solv, inform%obj
         WRITE( trs_sfiledevice, 2000 )
         DO i = 1, n
           WRITE( trs_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
         END DO
       END IF
       CALL TRS_terminate( data, control, inform )
       DEALLOCATE( H%val, H%row, H%col )
     END IF

!  Use More'-Sorensen

     IF ( more_sorensen ) THEN
       solv = 'MS    '

!  Evaluate the Hessian

       ALLOCATE( H_dense( n, n ), W1( n ), W2( n ), W3( n ) )
       CALL CUTEST_udh( cutest_status, n, X0, n, H_dense )
       IF ( cutest_status /= 0 ) GO TO 910

!  If required, open a file for the results

       IF ( write_result_summary ) THEN
         INQUIRE( FILE = ms_rfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( ms_rfiledevice, FILE = ms_rfilename, FORM = 'FORMATTED',     &
                  STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
         ELSE
            OPEN( ms_rfiledevice, FILE = ms_rfilename, FORM = 'FORMATTED',     &
                  STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( errout, 2030 ) iores, ms_rfilename
           STOP
         END IF
         WRITE( ms_rfiledevice, "( A10 )" ) pname
       END IF

!  Solve the problem

       inform%multiplier = zero
       IF ( control%max_factorizations < 0 ) control%max_factorizations = 100
       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "( ' More-Sorensen used ', / )" )
       CALL CLOCK_time( clock_record )
       CALL DGQT( n, H_dense, n, G, radius, control%stop_normal,               &
                  control%stop_normal, control%max_factorizations,             &
                  control%out, control%print_level,                            &
                  inform%multiplier, inform%obj, X, info,                      &
                  iter, inform%factorizations, W1, W2, W3 )
       CALL CLOCK_time( clock_now ) ; clock_now = clock_now - clock_record
       IF ( control%print_level > 0 .AND. control%out > 0 )                    &
         WRITE( control%out, "( /, ' More-Sorensen used ' )" )
       SELECT CASE( info )
       CASE ( 1 )
         inform%status = 0
       CASE ( 2 )
         inform%status = 0
       CASE ( 3 )
         inform%status = GALAHAD_error_tiny_step
       CASE ( 4 )
         inform%status = GALAHAD_error_max_iterations
       CASE DEFAULT
         inform%status = - 99
       END SELECT

!  If required, append results to a file,

       IF ( write_result_summary ) THEN
         BACKSPACE( ms_rfiledevice )
         IF ( inform%status == 0 ) THEN
           WRITE( ms_rfiledevice, 2040 ) pname, n, inform%obj,                 &
             inform%multiplier,                                                &
             inform%factorizations, clock_now, inform%status
         ELSE
           WRITE( ms_rfiledevice, 2040 ) pname, n, inform%obj,                 &
             inform%multiplier,                                                &
             inform%factorizations, - clock_now, inform%status
         END IF
       END IF

!  If required, write the solution

       IF ( control%print_level > 0 .AND. control%out > 0 ) THEN
         l = 2
         IF ( fulsol ) l = n
         IF ( control%print_level >= 10 ) l = n

         WRITE( errout, 2000 )
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

       IF ( .NOT. trs .OR. control%print_level > 0 ) WRITE( errout, 2060 )
       IF ( inform%status == 0 ) THEN
         WRITE( errout, 2050 ) pname, n, inform%obj,                           &
             inform%multiplier,                                                &
           inform%factorizations, clock_now, inform%status, solv
       ELSE
         WRITE( errout, 2050 ) pname, n, inform%obj,                           &
             inform%multiplier,                                                &
           inform%factorizations, - clock_now, inform%status, solv
       END IF

       IF ( write_solution .AND.                                               &
           ( inform%status == 0  .OR. inform%status == - 10 ) ) THEN
         INQUIRE( FILE = ms_sfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( ms_sfiledevice, FILE = ms_sfilename, FORM = 'FORMATTED',     &
                STATUS = 'OLD', IOSTAT = iores )
         ELSE
            OPEN( ms_sfiledevice, FILE = ms_sfilename, FORM = 'FORMATTED',     &
                 STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( out, 2030 ) iores, ms_sfilename ; STOP ; END IF
         WRITE( ms_sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ', &
        &       A, /, ' Objective:', ES24.16 )" ) pname, solv, inform%obj
         WRITE( ms_sfiledevice, 2000 )
         DO i = 1, n
           WRITE( ms_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
         END DO
       END IF
       DEALLOCATE( H_dense )

     END IF
     DEALLOCATE( X, X0, G, VNAMES )

     CALL CUTEST_cterminate( cutest_status )
     RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )          &
       cutest_status
     inform%status = - 98
     RETURN

!  Non-executable statements

 2000 FORMAT( /, ' Solution: ', /, '      # name          value   ' )
 2010 FORMAT( 6X, '. .', 9X, ( 2X, 10( '.' ) ) )
 2020 FORMAT( I7, 1X, A10, ES12.4 )
 2030 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2040 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5 )
 2050 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5, 1X, A )
 2060 FORMAT( /, 'name           n  f               lambda    ',               &
                 '     fac     time stat' )

!  End of subroutine USE_TRS

     END SUBROUTINE USE_TRS

!  End of module USETRS_double

   END MODULE GALAHAD_USETRS_double
