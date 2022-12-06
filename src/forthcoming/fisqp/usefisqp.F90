! THIS VERSION: GALAHAD 2.6 - 24/11/2014 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ F I S Q P  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, Yueling Loh and Daniel P. Robinson, for GALAHAD productions
!  Copyright reserved
!  Started: November 24th 2014

   MODULE GALAHAD_USEFISQP_double

!  This is the driver program for running FISQP for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_FISQP_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_double
     USE GALAHAD_CUTEST_FUNCTIONS_double
     USE GALAHAD_SYMBOLS
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_FISQP

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ F I S Q P   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_FISQP( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( FISQP_control_type ) :: control
     TYPE ( FISQP_inform_type ) :: inform
     TYPE ( FISQP_data_type ) :: data
     TYPE ( NLPT_problem_type ) :: nlp
     TYPE ( GALAHAD_userdata_type ) :: userdata
     TYPE ( CUTEST_FUNCTIONS_control_type ) :: cutest_control
     TYPE ( CUTEST_FUNCTIONS_inform_type ) :: cutest_inform

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

!  Problem input characteristics

     INTEGER :: iores, i
     LOGICAL :: filexx, is_specfile

!  Specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 31
     CHARACTER ( LEN = 16 ) :: specname = 'RUNFISQP'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNFISQP.SPC'

!  Default values for specfile-defined parameters

     INTEGER :: dfiledevice = 26
     INTEGER :: rfiledevice = 47
     INTEGER :: sfiledevice = 62
     INTEGER :: vfiledevice = 63
     INTEGER :: wfiledevice = 59
     LOGICAL :: write_problem_data    = .FALSE.
     LOGICAL :: write_solution        = .FALSE.
     LOGICAL :: write_solution_vector = .FALSE.
!    LOGICAL :: write_result_summary  = .FALSE.
     LOGICAL :: write_result_summary  = .TRUE.
     CHARACTER ( LEN = 30 ) :: dfilename = 'FISQP.data'
     CHARACTER ( LEN = 30 ) :: rfilename = 'FISQPRES.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'FISQPSOL.d'
     CHARACTER ( LEN = 30 ) :: vfilename = 'FISQPSOLVEC.d'
     CHARACTER ( LEN = 30 ) :: wfilename = 'FISQPSAVE.d'
     LOGICAL :: testal = .FALSE.
     LOGICAL :: dechk  = .FALSE.
     LOGICAL :: dechke = .FALSE.
     LOGICAL :: dechkg = .FALSE.
     LOGICAL :: not_fatal  = .FALSE.
     LOGICAL :: not_fatale = .FALSE.
     LOGICAL :: not_fatalg = .FALSE.
     LOGICAL :: getsca = .FALSE.
     INTEGER :: print_level_override = 0
     INTEGER :: print_level_scaling = 0
     LOGICAL :: scale  = .FALSE.
     LOGICAL :: scaleg = .FALSE.
     LOGICAL :: scalev = .FALSE.
     LOGICAL :: get_max = .FALSE.
     LOGICAL :: warm_start = .FALSE.
     INTEGER :: istore = 0

!  Output file characteristics

     INTEGER :: out  = 6
     INTEGER :: errout = 6
     CHARACTER ( LEN =  6 ) :: solv = 'FiSQP'

!  ------------------ Open the specfile for runlpsqp ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword  = 'write-problem-data'
       spec( 2 )%keyword  = 'problem-data-file-name'
       spec( 3 )%keyword  = 'problem-data-file-device'
       spec( 4 )%keyword  = 'print-level-override'
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
       spec( 29 )%keyword = 'write-solution-vector'
       spec( 30 )%keyword = 'solution-vector-file-name'
       spec( 31 )%keyword = 'solution-vector-file-device'

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
       CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), print_level_override, errout )
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
!      CALL SPECFILE_assign_logical( spec( 28 ), equality, errout )
       CALL SPECFILE_assign_logical( spec( 29 ), write_solution_vector, errout )
       CALL SPECFILE_assign_string ( spec( 30 ), vfilename, errout )
       CALL SPECFILE_assign_integer( spec( 31 ), vfiledevice, errout )
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

     cutest_control%input = input ; cutest_control%error = control%error
     CALL CUTEST_initialize( nlp, cutest_control, cutest_inform, userdata )

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2014' )

!  record problem and solver information

     IF ( out > 0 ) WRITE( out, "( ' Problem: ', A, ', solver: ', A, / )" )    &
       TRIM( nlp%pname ) , solv

!  Set up data for next problem

     CALL FISQP_initialize( data, control, inform )
     IF ( is_specfile ) CALL FISQP_read_specfile( control, input_specfile )

!  override print options if required

     SELECT CASE (print_level_override )
     CASE ( 1 )
       control%print_level = 1
     CASE ( 2 )
       control%print_level = 4
       control%QP_steer_control%print_level = 1
       control%QP_steer_control%CQP_control%print_level = 1
       control%QP_steer_control%DLP_control%print_level = 1
       control%QP_steer_control%DQP_control%print_level = 1
       control%QP_pred_control%print_level = 1
       control%QP_pred_control%CQP_control%print_level = 1
       control%QP_pred_control%DLP_control%print_level = 1
       control%QP_pred_control%DQP_control%print_level = 1
       control%QP_accel_control%print_level = 1
     CASE ( 3 : 100 )
       control%print_level = 4
       control%QP_steer_control%print_level = 1
       control%QP_steer_control%CQP_control%print_level = 101
       control%QP_steer_control%DLP_control%print_level = 1
       control%QP_steer_control%DQP_control%print_level = 1
       control%QP_pred_control%print_level = 1
       control%QP_pred_control%CQP_control%print_level = 101
       control%QP_pred_control%DLP_control%print_level = 1
       control%QP_pred_control%DQP_control%print_level = 1
       control%QP_accel_control%print_level = 1
       control%QP_accel_control%SBLS_control%print_level = 1
       control%QP_accel_control%SBLS_control%SLS_control%print_level = 1
       control%QP_accel_control%SBLS_control%SLS_control%print_level_solver = 1
       control%QP_accel_control%GLTR_control%print_level = 1
     CASE ( 101 : )
       control%print_level = 101
       control%QP_steer_control%print_level = 101
       control%QP_steer_control%CQP_control%print_level = 101
       control%QP_steer_control%CQP_control%CRO_control%print_level = 101
       control%QP_steer_control%DLP_control%print_level = 1
       control%QP_steer_control%DQP_control%print_level = 1
       control%QP_pred_control%print_level = 101
       control%QP_pred_control%CQP_control%print_level = 101
       control%QP_pred_control%CQP_control%CRO_control%print_level = 101
       control%QP_pred_control%DLP_control%print_level = 1
       control%QP_pred_control%DQP_control%print_level = 1
       control%QP_accel_control%print_level = 101
       control%QP_accel_control%SBLS_control%print_level = 1
       control%QP_accel_control%SBLS_control%SLS_control%print_level = 1
       control%QP_accel_control%SBLS_control%SLS_control%print_level_solver = 1
       control%QP_accel_control%GLTR_control%print_level = 1
     END SELECT

!  Solve the problem

     inform%status = 1
     CALL FISQP_solve( nlp, control, inform, data, userdata,                   &
                       eval_FC = CUTEST_eval_FC, eval_GJ = CUTEST_eval_GJ,     &
                       eval_HL = CUTEST_eval_HL )

!  If required, append results to a file

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )     &
          nlp%pname, inform%obj, inform%primal_infeasibility,                  &
          inform%dual_infeasibility, inform%complementary_slackness,           &
          inform%iter, inform%time%total, inform%status
      END IF
      WRITE( errout, "( 'name        f               pr-feas  du-feas ',       &
     &                  ' cmp-slk      its        time  stat' )" )
      WRITE( errout, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )            &
        nlp%pname, inform%obj, inform%primal_infeasibility,                    &
        inform%dual_infeasibility, inform%complementary_slackness,             &
        inform%iter, inform%time%total, inform%status

!  If required, write the solution vector to a file

      IF ( inform%status == GALAHAD_ok .OR.                                    &
           inform%status == GALAHAD_error_max_iterations .OR.                  &
           inform%status == GALAHAD_error_cpu_limit ) THEN
        IF ( write_solution_vector ) THEN
          INQUIRE( FILE = vfilename, EXIST = filexx )
          IF ( filexx ) THEN
             OPEN( vfiledevice, FILE = vfilename, FORM = 'FORMATTED',          &
                 STATUS = 'OLD', IOSTAT = iores )
          ELSE
             OPEN( vfiledevice, FILE = vfilename, FORM = 'FORMATTED',          &
                 STATUS = 'NEW', IOSTAT = iores )
          END IF
          IF ( iores /= 0 ) THEN
            write( out, 2030 ) iores, vfilename
            STOP
          END IF

          REWIND( vfiledevice )
          DO i = 1, nlp%n
            WRITE( vfiledevice, "( ES22.15 )" ) nlp%X( i )
          END DO
          CLOSE( vfiledevice )
        END IF
      END IF

!  If required, write the solution

     IF ( write_solution .AND.                                                 &
         ( inform%status == GALAHAD_ok .OR.                                    &
           inform%status == GALAHAD_error_factorization ) ) THEN

       INQUIRE( FILE = sfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
              STATUS = 'OLD', IOSTAT = iores )
       ELSE
          OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
               STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( out, 2030 ) iores, sfilename
         STOP
       END IF

       WRITE( sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ', A,   &
      &       /, ' Objective:', ES24.16 )" ) nlp%pname, solv, inform%obj

       WRITE( sfiledevice, 2000 )
       DO i = 1, nlp%n
         WRITE( sfiledevice, 2020 ) i, nlp%VNAMES( i ), nlp%X( i ),            &
           nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
       END DO

       IF ( nlp%m > 0 ) THEN
         WRITE( sfiledevice, 2010 )
         DO i = 1, nlp%m
           WRITE( sfiledevice, 2020 ) i, nlp%CNAMES( i ), nlp%C( i ),          &
             nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
         END DO
       END IF
    END IF

!  Close any opened files

     IF ( is_specfile ) CLOSE( input_specfile )
     CALL CUTEST_terminate( nlp, cutest_inform, userdata )
     RETURN

!  Non-executable statements

 2000 FORMAT( /,' Solution: ', /,'                        ',                   &
                '        <------ Bounds ------> ', /                           &
                '      # name          value   ',                              &
                '    Lower       Upper       Dual ' )
 2010 FORMAT( /,' Constraints: ', /, '                        ',               &
                '        <------ Bounds ------> ', /                           &
                '      # name           value   ',                             &
                '    Lower       Upper    Multiplier ' )
 2020 FORMAT( I7, 1X, A10, 4ES12.4 )
 2030 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )

!  End of subroutine USE_FISQP

     END SUBROUTINE USE_FISQP

!  End of module USEFISQP_double

   END MODULE GALAHAD_USEFISQP_double
