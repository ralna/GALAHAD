! THIS VERSION: GALAHAD 5.2 - 2025-04-30 AT 09:10 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ A R C  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  May 14th 2011

   MODULE GALAHAD_USEARC_precision

!  This is the driver program for running ARC for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_KINDS_precision
     USE GALAHAD_SYMBOLS
     USE GALAHAD_COPYRIGHT
!    USE GALAHAD_CLOCK
     USE GALAHAD_ARC_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type
     USE GALAHAD_CUTEST_precision
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_ARC

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ A R C   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_ARC( input )

!  Dummy argument

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( ARC_control_type ) :: control
     TYPE ( ARC_inform_type ) :: inform
     TYPE ( ARC_data_type ) :: data
     TYPE ( NLPT_problem_type ) :: nlp
     TYPE ( GALAHAD_userdata_type ) :: userdata
     TYPE ( CUTEST_control_type ) :: cutest_control
     TYPE ( CUTEST_inform_type ) :: cutest_inform

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

!  Problem input characteristics

     INTEGER ( KIND = ip_ ) :: iores, i, j, ir, ic, l
     LOGICAL :: filexx, is_specfile
!    REAL :: timeo, timet
!    REAL ( KIND = rp_ ) :: clocko, clockt
     CHARACTER ( LEN = 10 ) :: name

!  Functions

!$    INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  Specfile characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 29
     CHARACTER ( LEN = 16 ) :: specname = 'RUNARC'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNARC.SPC'

!  Default values for specfile-defined parameters

     INTEGER ( KIND = ip_ ) :: dfiledevice = 26
     INTEGER ( KIND = ip_ ) :: rfiledevice = 47
     INTEGER ( KIND = ip_ ) :: sfiledevice = 62
     INTEGER ( KIND = ip_ ) :: wfiledevice = 59
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: dfilename = 'ARC.data'
     CHARACTER ( LEN = 30 ) :: rfilename = 'ARCRES.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'ARCSOL.d'
     CHARACTER ( LEN = 30 ) :: wfilename = 'ARCSAVE.d'
     LOGICAL :: testal = .FALSE.
     LOGICAL :: dechk  = .FALSE.
     LOGICAL :: dechke = .FALSE.
     LOGICAL :: dechkg = .FALSE.
     LOGICAL :: not_fatal  = .FALSE.
     LOGICAL :: not_fatale = .FALSE.
     LOGICAL :: not_fatalg = .FALSE.
     LOGICAL :: getsca = .FALSE.
     INTEGER ( KIND = ip_ ) :: print_level_scaling = 0
     LOGICAL :: scale  = .FALSE.
     LOGICAL :: scaleg = .FALSE.
     LOGICAL :: scalev = .FALSE.
     LOGICAL :: get_max = .FALSE.
     LOGICAL :: warm_start = .FALSE.
     INTEGER ( KIND = ip_ ) :: istore = 0

!  Output file characteristics

     INTEGER ( KIND = ip_ ) :: out  = 6
     INTEGER ( KIND = ip_ ) :: errout = 6
     CHARACTER ( LEN =  6 ) :: solv = 'ARC   '

!  ------------------ Open the specfile for tru ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword  = 'write-problem-data'
       spec( 2 )%keyword  = 'problem-data-file-name'
       spec( 3 )%keyword  = 'problem-data-file-device'
       spec( 4 )%keyword  = 'print-full-solution'
       spec( 5 )%keyword  = 'write-solution'
       spec( 6 )%keyword  = 'solution-file-name'
       spec( 7 )%keyword  = 'solution-file-device'
       spec( 8 )%keyword  = 'write-result-summary'
       spec( 9 )%keyword  = 'result-summary-file-name'
       spec( 10 )%keyword = 'result-summary-file-device'
       spec( 11 )%keyword = 'check-all-derivatives'
       spec( 12 )%keyword = 'check-derivatives'
       spec( 13 )%keyword = 'check-element-derivatives'
       spec( 14 )%keyword = 'check-group-derivatives'
       spec( 15 )%keyword = 'ignore-derivative-bugs'
       spec( 16 )%keyword = 'ignore-element-derivative-bugs'
       spec( 17 )%keyword = 'ignore-group-derivative-bugs'
       spec( 18 )%keyword = 'get-scaling-factors'
       spec( 19 )%keyword = 'scaling-print-level'
       spec( 20 )%keyword = 'use-scaling-factors'
       spec( 21 )%keyword = 'use-constraint-scaling-factors'
       spec( 22 )%keyword = 'use-variable-scaling-factors'
       spec( 23 )%keyword = 'maximizer-sought'
       spec( 24 )%keyword = 'restart-from-previous-point'
       spec( 25 )%keyword = 'restart-data-file-name'
       spec( 26 )%keyword = 'restart-data-file-device'
       spec( 27 )%keyword = 'save-data-for-restart-every'
       spec( 28 )%keyword = ''
       spec( 29 )%keyword = ''

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
       CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 4 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 5 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 6 ), sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 7 ), sfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 8 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 9 ), rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 10 ), rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 11 ), testal, errout )
       CALL SPECFILE_assign_logical( spec( 12 ), dechk, errout )
       CALL SPECFILE_assign_logical( spec( 13 ), dechke, errout )
       CALL SPECFILE_assign_logical( spec( 14 ), dechkg, errout )
       CALL SPECFILE_assign_logical( spec( 15 ), not_fatal, errout )
       CALL SPECFILE_assign_logical( spec( 16 ), not_fatale, errout )
       CALL SPECFILE_assign_logical( spec( 17 ), not_fatalg, errout )
       CALL SPECFILE_assign_logical( spec( 18 ), getsca, errout )
       CALL SPECFILE_assign_integer( spec( 19 ), print_level_scaling, errout )
       CALL SPECFILE_assign_logical( spec( 20 ), scale, errout )
       CALL SPECFILE_assign_logical( spec( 21 ), scaleg, errout )
       CALL SPECFILE_assign_logical( spec( 22 ), scalev, errout )
       CALL SPECFILE_assign_logical( spec( 23 ), get_max, errout )
       CALL SPECFILE_assign_logical( spec( 24 ), warm_start, errout )
       CALL SPECFILE_assign_string ( spec( 25 ), wfilename, errout )
       CALL SPECFILE_assign_integer( spec( 26 ), wfiledevice, errout )
       CALL SPECFILE_assign_integer( spec( 27 ), istore, errout )
     END IF

     IF ( dechk .OR. testal ) THEN ; dechke = .TRUE. ; dechkg = .TRUE. ; END IF
     IF ( not_fatal ) THEN ; not_fatale = .TRUE. ; not_fatalg = .TRUE. ; END IF
     IF ( scale ) THEN ; scaleg = .TRUE. ; scalev = .TRUE. ; END IF

!  If required, open a file for the results

     IF ( write_result_summary ) THEN
       INQUIRE( FILE = rfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',             &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
       ELSE
          OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',             &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( errout, 2030 ) iores, rfilename
         STOP
       END IF
       READ( INPUT, "( /, I2, A8  )" ) iores, nlp%pname
       REWIND( input )
       WRITE( rfiledevice, "( A10 )" ) nlp%pname
     END IF

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2008' )

!  Set up control parameters prior to the next solution

     CALL ARC_initialize( data, control, inform )
     IF ( is_specfile ) CALL ARC_read_specfile( control, input_specfile )

!  Initialize the problem data

     cutest_control%input = input ; cutest_control%error = control%error
     CALL CUTEST_initialize( nlp, cutest_control, cutest_inform, userdata )

!  Read a previous solution file for a re-entry

     IF ( warm_start .AND. wfiledevice > 0 ) THEN
       OPEN( wfiledevice, FILE = wfilename, FORM = 'FORMATTED',                &
             STATUS = 'OLD', IOSTAT = iores )
       IF ( iores /= 0 ) THEN
         WRITE( out, 2030 ) iores, wfilename
         STOP
       END IF

       REWIND( wfiledevice )
       READ( wfiledevice, "( A10 )" ) name
       IF ( name /= nlp%pname ) THEN
         WRITE( out, "( /, ' *** Exit from usearc: re-entry requested with',   &
        &         ' data for problem ', A10, /,                                &
        &         '     but the most recently decoded problem is ', A10 )" )   &
          name, nlp%pname
         STOP
       END IF
       READ( wfiledevice, "( I8 )" )i
       IF ( i /= nlp%n ) THEN
         WRITE( out, "( /, ' *** Exit from usearc: number of variables'      , &
        &        ' changed from ', I0,' to ', I0,' on re-entry ' )" ) nlp%n, i
         STOP
       END IF
       READ( wfiledevice, "( I8 )" )i
       IF ( i /= nlp%m ) THEN
         WRITE( out, "( /, ' *** Exit from usearc: number of residuals',       &
        &        ' changed from ', I0, ' to ', I0, ' on re-entry ' )" ) nlp%m, i
         STOP
       END IF
       DO i = 1, nlp%n
         READ( wfiledevice, "( ES24.16, 2X, A10 )" ) nlp%X( i ), name
         IF ( name /= nlp%VNAMES( i ) ) THEN
           WRITE( out, "( /, ' *** Exit from usearc: variable named ',         &
          &  A, ' out of order on re-entry ' )" ) TRIM( name )
           STOP
         END IF
       END DO
       CLOSE( wfiledevice )
     END IF

!  Solve the problem

     inform%status = 1
!    CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )
     CALL ARC_solve( nlp, control, inform, data, userdata,                     &
                     eval_F = CUTEST_eval_F, eval_G = CUTEST_eval_G,           &
                     eval_H = CUTEST_eval_H, eval_HPROD = CUTEST_eval_HPROD )
!    CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )

!$    WRITE( out, "( ' number of threads = ', I0 )" ) OMP_GET_MAX_THREADS( )

!  If required, append results to a file,

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        IF ( control%subproblem_direct ) THEN
          IF ( inform%status == GALAHAD_ok .OR.                                &
               inform%status == GALAHAD_error_unbounded ) THEN
            WRITE( rfiledevice, 2040 ) nlp%pname, nlp%n, inform%obj,           &
              inform%norm_g, inform%iter, inform%g_eval,                       &
              inform%factorization_average, inform%factorization_max,          &
              inform%time%clock_total, inform%status
          ELSE
            WRITE( rfiledevice, 2040 ) nlp%pname, nlp%n, inform%obj,           &
              inform%norm_g, - inform%iter, - inform%g_eval,                   &
              inform%factorization_average, inform%factorization_max,          &
              - inform%time%clock_total, inform%status
          END IF
        ELSE
          IF ( inform%status == GALAHAD_ok .OR.                                &
               inform%status == GALAHAD_error_unbounded ) THEN
            WRITE( rfiledevice, 2050 ) nlp%pname, nlp%n, inform%obj,           &
              inform%norm_g, inform%iter, inform%g_eval, inform%cg_iter,       &
              inform%time%clock_total, inform%status
          ELSE
            WRITE( rfiledevice, 2050 ) nlp%pname, nlp%n, inform%obj,           &
              inform%norm_g, - inform%iter, - inform%g_eval, inform%cg_iter,   &
              - inform%time%clock_total, inform%status
          END IF
        END IF
      END IF

!  If required, write the solution

      l = 2
      IF ( fulsol ) l = nlp%n
      IF ( control%print_level >= 10 ) l = nlp%n

      WRITE( errout, 2000 ) TRIM( solv )
      DO j = 1, 2
        IF ( j == 1 ) THEN
          ir = 1 ; ic = MIN( l, nlp%n )
        ELSE
          IF ( ic < nlp%n - l ) WRITE( errout, 2010 )
          ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
        END IF
        DO i = ir, ic
          WRITE( errout, 2020 ) i, nlp%VNAMES( i ), nlp%X( i ), nlp%X_l( i ),  &
            nlp%X_u( i ), nlp%G( i )
        END DO
      END DO

      IF ( control%subproblem_direct ) THEN
        WRITE( errout, "( /, 'name           n  f               du-feas ',     &
       &  '   its     #g   av fac     time stat' )" )
        IF ( inform%status == GALAHAD_ok .OR.                                  &
             inform%status == GALAHAD_error_unbounded ) THEN
          WRITE( errout, 2040 ) nlp%pname, nlp%n, inform%obj, inform%norm_g,   &
            inform%iter, inform%g_eval, inform%factorization_average,          &
            inform%factorization_max, inform%time%clock_total, inform%status
        ELSE
          WRITE( errout, 2040 ) nlp%pname, nlp%n, inform%obj, inform%norm_g,   &
            - inform%iter, - inform%g_eval, inform%factorization_average,      &
            inform%factorization_max, - inform%time%clock_total, inform%status
        END IF
      ELSE
        WRITE( errout, "( /, 'name           n  f               du-feas ',     &
       &  '   its     #g      #cg       time stat' )" )
        IF ( inform%status == GALAHAD_ok .OR.                                  &
             inform%status == GALAHAD_error_unbounded ) THEN
          WRITE( errout, 2050 ) nlp%pname, nlp%n, inform%obj,                  &
            inform%norm_g, inform%iter, inform%g_eval,                         &
            inform%cg_iter, inform%time%clock_total, inform%status
        ELSE
          WRITE( errout, 2050 ) nlp%pname, nlp%n, inform%obj,                  &
            inform%norm_g, - inform%iter, - inform%g_eval,                     &
            inform%cg_iter, - inform%time%clock_total, inform%status
        END IF
      END IF

      IF ( write_solution .AND.                                                &
          ( inform%status == 0  .OR. inform%status == - 10 .OR.                &
            inform%status == - 18 ) ) THEN

        INQUIRE( FILE = sfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',            &
               STATUS = 'OLD', IOSTAT = iores )
        ELSE
           OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',            &
                STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2030 ) iores, sfilename
          STOP
        END IF

        WRITE( sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ', A,  &
       &       /, ' Objective:', ES24.16 )" ) nlp%pname, solv, inform%obj

        WRITE( sfiledevice, 2000 ) TRIM( solv )
        DO i = 1, nlp%n
          WRITE( sfiledevice, 2020 ) i, nlp%VNAMES( i ), nlp%X( i ),           &
            nlp%X_l( i ), nlp%X_u( i ), nlp%G( i )
        END DO

     END IF

!  if required, save the solution for a future re-entry

     IF ( wfiledevice > 0 .AND. ( ( istore > 0 .AND.                           &
            ( inform%status == GALAHAD_ok .OR.                                 &
              inform%status == GALAHAD_error_ill_conditioned .OR.              &
              inform%status == GALAHAD_error_tiny_step .OR.                    &
              inform%status == GALAHAD_error_max_iterations .OR.               &
              inform%status == GALAHAD_error_time_limit .OR.                   &
              inform%status == GALAHAD_error_alive ) ) .OR.                    &
            inform%status == GALAHAD_error_alive ) ) THEN
       INQUIRE( FILE = wfilename, OPENED = filexx )
       IF ( .NOT. filexx ) THEN
         INQUIRE( FILE = wfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( wfiledevice, FILE = wfilename, FORM = 'FORMATTED',           &
                  STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
         ELSE
            OPEN( wfiledevice, FILE = wfilename, FORM = 'FORMATTED',           &
                  STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           WRITE( out, 2030 ) iores, wfilename
           STOP
         END IF
       END IF
       REWIND( wfiledevice )
       WRITE( wfiledevice, "( A10, ' problem name ', /,                        &
      &                       I8,  ' number of variables', /,                  &
      &                       I8,  ' number of residuals',                     &
      &                       ' ... and now variables' )" )                    &
       nlp%pname, nlp%n, nlp%m
       DO i = 1, nlp%n
         WRITE( wfiledevice, "( ES24.16, 2X, A10 )" )                          &
          nlp%X( i ), nlp%VNAMES( i )
       END DO
       CLOSE( wfiledevice )
     END IF

!  Close any opened files and deallocate arrays

     IF ( is_specfile ) CLOSE( input_specfile )
     CALL CUTEST_terminate( nlp, cutest_inform, userdata )
     RETURN

!  Non-executable statements

 2000 FORMAT( ' Solver: ', A, /,                                               &
              ' Solution: ', /, '                        ',                    &
              '        <------ Bounds ------> ', /                             &
              '      # name          value   ',                                &
              '    Lower       Upper       Dual ' )
 2010 FORMAT( 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2020 FORMAT( I7, 1X, A10, 4ES12.4 )
 2030 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2040 FORMAT( A10, I6, ES16.8, ES9.1, bn, 2I7, F5.1, I4, F9.2, I5 )
 2050 FORMAT( A10, I6, ES16.8, ES9.1, bn, 2I7, I9, ' :', F9.2, I5 )

!  End of subroutine USE_ARC

     END SUBROUTINE USE_ARC

!  End of module USEARC

   END MODULE GALAHAD_USEARC_precision
