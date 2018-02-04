! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ S 2 Q P  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould and Daniel Robinson, for GALAHAD productions
!  Copyright reserved
!  Started: December 22th 2007

   MODULE GALAHAD_USES2QP_double

!  This is the driver program for running S2QP for a variety of computing
!  systems. It opens and closes all the files, allocates arrays, reads and
!  checks data, and calls the appropriate package.

     USE GALAHAD_SYMBOLS
     USE GALAHAD_S2QP_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_SMT_double
     USE GALAHAD_CUTEST_FUNCTIONS_double
     USE GALAHAD_CHECK_double
     USE CUTEST_interface_double

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_S2QP

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ S 2 Q P   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_S2QP( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input


!  Set precision

     INTEGER, PARAMETER :: wp    = KIND( 1.0D+0 )
     REAL ( KIND = wp ) :: zero  = 0.0_wp
     REAL ( KIND = wp ) :: one   = 1.0_wp
     REAL ( KIND = wp ) :: two   = 2.0_wp
     REAL ( KIND = wp ) :: three = 3.0_wp

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( S2QP_control_type )            :: control
     TYPE ( S2QP_inform_type )             :: inform
     TYPE ( S2QP_data_type )               :: data
     TYPE ( NLPT_userdata_type )           :: userdata
     TYPE ( NLPT_problem_type )            :: nlp
     TYPE ( CUTEST_FUNCTIONS_inform_type )  :: cutest_inform
     TYPE ( CUTEST_FUNCTIONS_control_type ) :: cutest_control
     TYPE ( CHECK_control_type )           :: CHECK_control
     TYPE ( CHECK_inform_type )            :: CHECK_inform
     TYPE ( CHECK_data_type )              :: CHECK_data

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

!  Problem input characteristics

     INTEGER :: iores, i
     LOGICAL :: filexx, is_specfile
     REAL( kind = wp ) :: dummy_real
     REAL( kind = wp ), dimension(:), allocatable :: X_saved, Y_saved

!  Specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 29
     CHARACTER ( LEN = 16 ) :: specname = 'RUNS2QP'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNS2QP.SPC'

!  Default values for specfile-defined parameters

     INTEGER :: dfiledevice = 26
     INTEGER :: rfiledevice = 47
     INTEGER :: rfiledevice_full = 48
     INTEGER :: sfiledevice = 62
     INTEGER :: wfiledevice = 59
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: dfilename = 'S2QP.data'
     CHARACTER ( LEN = 30 ) :: rfilename = 'S2QPRES.d'
     CHARACTER ( LEN = 30 ) :: rfilename_full = 'S2QPRES_full.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'S2QPSOL.d'
     CHARACTER ( LEN = 30 ) :: wfilename = 'S2QPSAVE.d'
     LOGICAL :: check_derivatives = .FALSE.
     LOGICAL :: derivative_fatal = .FALSE.
     LOGICAL :: getsca = .FALSE.
     INTEGER :: print_level_scaling = 0
     LOGICAL :: scale  = .FALSE.
     LOGICAL :: scaleg = .FALSE.
     LOGICAL :: scalev = .FALSE.
     LOGICAL :: get_max = .FALSE.
     LOGICAL :: warm_start = .FALSE.
     INTEGER :: istore = 0
     LOGICAL :: separate_linear_constraints = .FALSE.

! Output file characteristics

  INTEGER :: out  = 6
  INTEGER :: errout = 6

! ------------------ Open the specfile for runs2qp ----------------

  INQUIRE( FILE = runspec, EXIST = is_specfile )
  IF ( is_specfile ) THEN
     OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD' )

     !   Define the keywords

     spec( 1 )%keyword  = 'write-problem-data'
     spec( 2 )%keyword  = 'problem-data-file-name'
     spec( 3 )%keyword  = 'problem-data-file-device'
     spec( 4 )%keyword  = ''
     spec( 5 )%keyword  = 'write-solution'
     spec( 6 )%keyword  = 'solution-file-name'
     spec( 7 )%keyword  = 'solution-file-device'
     spec( 8 )%keyword  = 'write-result-summary'
     spec( 9 )%keyword  = 'result-summary-file-name'
     spec( 10 )%keyword = 'result-summary-file-device'
     spec( 11 )%keyword = 'call-derivative-checker'
     spec( 15 )%keyword = 'ignore-derivative-bugs'
     spec( 16 )%keyword = 'derivative-print-level'
     spec( 17 )%keyword = 'derivative-verification-level'
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
     spec( 28 )%keyword = 'separate-linear-constraints'
     spec( 29 )%keyword = ''

     ! Read the specfile

     CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

     ! Interpret the result

     CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
     CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
     CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
     CALL SPECFILE_assign_logical( spec( 5 ), write_solution, errout )
     CALL SPECFILE_assign_string ( spec( 6 ), sfilename, errout )
     CALL SPECFILE_assign_integer( spec( 7 ), sfiledevice, errout )
     CALL SPECFILE_assign_logical( spec( 8 ), write_result_summary, errout )
     CALL SPECFILE_assign_string ( spec( 9 ), rfilename, errout )
     CALL SPECFILE_assign_integer( spec( 10 ), rfiledevice, errout )
     CALL SPECFILE_assign_logical( spec( 11 ), check_derivatives, errout )
     CALL SPECFILE_assign_logical( spec( 15 ), derivative_fatal, errout )
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
     CALL SPECFILE_assign_logical( spec( 28 ), separate_linear_constraints, errout )
  END IF

  ! If required, open a file for the results

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
        write( errout, 2160 ) iores, rfilename
        STOP
     END IF
     READ( INPUT, "( /, I2, A10  )" ) iores, nlp%pname
     REWIND( input )
     WRITE( rfiledevice, "( A10 )" ) nlp%pname

     INQUIRE( FILE = rfilename_full, EXIST = filexx )
     IF ( filexx ) THEN
        OPEN( rfiledevice_full, FILE = rfilename_full, FORM = 'FORMATTED',     &
              STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
     ELSE
        OPEN( rfiledevice_full, FILE = rfilename_full, FORM = 'FORMATTED',     &
              STATUS = 'NEW', IOSTAT = iores )
     END IF
     IF ( iores /= 0 ) THEN
        write( errout, 2160 ) iores, rfilename_full
        STOP
     END IF
     READ( INPUT, "( /, I2, A10  )" ) iores, nlp%pname
     REWIND( input )
     WRITE( rfiledevice_full, "( A10 )" ) nlp%pname
  END IF

  ! Copyright

  IF ( out > 0 ) CALL COPYRIGHT( out, '2008' )

!  Initialize S2QP

  CALL S2QP_initialize( data, control, inform )

  IF ( is_specfile ) CALL S2QP_read_specfile( control, input_specfile )

!  Get the problem from CUTEst

  cutest_control%separate_linear_constraints = separate_linear_constraints

  cutest_control%input = input ;    cutest_control%error = control%error
  CALL CUTEST_initialize( nlp, cutest_control, cutest_inform, userdata )

! Define some data that probably should be defined in CUTEST_initialize.

  if ( nlp%m_a > 0 ) then
     nlp%A%m = nlp%m_a
     nlp%A%n = nlp%n
  end if

  if ( nlp%m > 0 ) then
     nlp%J%m  = nlp%m
     nlp%J%n  = nlp%n
     nlp%J%ne = size( nlp%J%val )
  end if

  nlp%H%m = nlp%n
  nlp%H%n = nlp%n

  ! Possibly check derivatives.

  IF ( check_derivatives ) then

     CALL CHECK_initialize( CHECK_control )

     IF ( is_specfile ) CALL CHECK_read_specfile( CHECK_control, input_specfile )

     ! Get nontrial x and y to make the derivative check more reliable.

     allocate( X_saved(1:nlp%n), Y_saved(1:nlp%m), stat=CHECK_inform%status )
     if ( CHECK_inform%status /= 0 ) then
        write( out, "('--ERROR:uses2qp: allocation failed.')")
        return
     end if

     X_saved = nlp%X
     Y_saved = nlp%Y

     do i = 1, nlp%n
        if ( two <= nlp%X_u(i) ) then
           if ( two >= nlp%X_l(i) ) then
              nlp%X(i) = two
           else
              nlp%X(i) = nlp%X_l(i)
           end if
           cycle
        end if
        if ( -two >= nlp%X_u(i) ) then
           nlp%X(i) = nlp%X_u(i)
           cycle
        end if
        if ( nlp%X_u(i) /= zero ) then
           nlp%X(i) = nlp%X_u(i)
           cycle
        end if
        nlp%X(i) = max( -two, nlp%X_l(i) )
     end do

     dummy_real = three
     do i = 1, nlp%m
        nlp%Y(i) = dummy_real
        dummy_real = - dummy_real
     end do

     ! IMPORTANT: signify first call to CHECK_verify

     CHECK_inform%status = 1

     CALL CHECK_verify( nlp, CHECK_data, CHECK_control, CHECK_inform,  &
                        eval_F=CUTEST_eval_F, eval_C=CUTEST_eval_C,      &
                        eval_G=CUTEST_eval_G, eval_J=CUTEST_eval_J,      &
                        eval_HL=CUTEST_eval_HL, userdata=userdata )

     CALL CHECK_terminate( CHECK_data, CHECK_control, CHECK_inform )

     ! Return initial value back to nlp%X

     nlp%X = X_saved
     nlp%Y = Y_saved

     deallocate( X_saved, Y_saved, stat=CHECK_inform%status )
     if ( CHECK_inform%status /= 0 ) then
        write( out, "('--ERROR:uses2qp: deallocation failed.')")
        return
     end if

  END IF

!  Solve the problem

  inform%status = 1
  CALL S2QP_solve( nlp, control, inform, data, userdata,         &
                   CUTEST_eval_FC, CUTEST_eval_GJ, CUTEST_eval_HL )

 !  If required, append results to a file

  IF ( write_result_summary ) THEN
     BACKSPACE( rfiledevice )
     WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )        &
            nlp%pname, inform%obj, inform%primal_vl, inform%dual_vl,           &
            inform%comp_vl, inform%iterate, inform%time%total, inform%status
     BACKSPACE( rfiledevice_full )
     WRITE( rfiledevice_full, "( A10, F20.8, 3F17.9, I5, F10.2, I3, 7I5 )" )   &
            nlp%pname, inform%obj, min( one,inform%primal_vl),                 &
            min( one, inform%dual_vl), min( one, inform%comp_vl),              &
            inform%iterate, inform%time%total, inform%status,                  &
            inform%num_f_eval, inform%num_g_eval, inform%num_c_eval,           &
            inform%num_J_eval, inform%num_H_eval, inform%num_predictors,       &
            inform%num_descent_active
  END IF

  !  If required, write the solution

  IF ( write_solution ) THEN

     INQUIRE( FILE = sfilename, EXIST = filexx )
     IF ( filexx ) THEN
        OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
              STATUS = 'OLD', IOSTAT = iores )
     ELSE
        OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
              STATUS = 'NEW', IOSTAT = iores )
     END IF
     IF ( iores /= 0 ) THEN
        write( out, 2160 ) iores, sfilename
        STOP
     END IF

     !WRITE( sfiledevice, 2250 ) nlp%pname, solv, inform%obj

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

     IF ( nlp%m_a > 0 ) THEN
        WRITE( sfiledevice, 2010 )
        DO i = 1, nlp%m_a
           WRITE( sfiledevice, 2020 ) i, nlp%ANAMES( i ), nlp%Ax( i ),         &
                nlp%A_l( i ), nlp%A_u( i ), nlp%Y_a( i )
        END DO
     END IF


  END IF

  !  Close any opened files

  IF ( is_specfile ) CLOSE( input_specfile )

  !  Deallocate any allocated arrays.

  CALL CUTEST_terminate( nlp, cutest_inform, userdata )
  CALL S2QP_terminate( data, control, inform )

  RETURN

! *****************************************************************************

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
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 !2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                 &
 !             /, ' Objective:', ES24.16 )

!  End of subroutine USE_S2QP

  END SUBROUTINE USE_S2QP

!  End of module USES2QP_double

END MODULE GALAHAD_USES2QP_double
