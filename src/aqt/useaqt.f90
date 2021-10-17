! THIS VERSION: GALAHAD 3.3 - 08/10/2021 AT 09:45 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ A Q T  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  October 8th 2021

   MODULE GALAHAD_USEAQT_double

!  This is the driver program for running AQT for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE CUTEst_interface_double
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_AQT_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_double
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_AQT

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ A Q T   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_AQT( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( AQT_control_type ) :: control
     TYPE ( AQT_inform_type ) :: inform
     TYPE ( AQT_data_type ) :: data

!------------------------------------
!   L o c a l   P a r a m e t e r s
!------------------------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
!    REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!----------------------------------
!   L o c a l   V a r i a b l e s
!----------------------------------

     INTEGER :: iores, i, j, ir, ic, l, status, cutest_status
     REAL :: time_now, time_start
     REAL ( KIND = wp ) :: clock_now, clock_start
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R, VECTOR, H_vector
     LOGICAL :: goth

!  Problem characteristics

     INTEGER :: n
     REAL ( KIND = wp ) ::  f
     CHARACTER ( LEN = 10 ) :: pname
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, X0, X_l, X_u, G
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES

!  Problem input characteristics

     LOGICAL :: filexx, is_specfile

!  Default values for specfile-defined parameters

     INTEGER :: aqt_rfiledevice = 47
     INTEGER :: aqt_sfiledevice = 62
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: aqt_rfilename = 'AQTRES.d'
     CHARACTER ( LEN = 30 ) :: aqt_sfilename = 'AQTSOL.d'
     REAL ( KIND = wp ) ::  radius = 1.0_wp

!  Output file characteristics

     INTEGER, PARAMETER :: io_buffer = 11
     INTEGER :: out  = 6
     INTEGER :: errout = 6
     CHARACTER ( LEN = 4 ) :: solv

!  Specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 15
     CHARACTER ( LEN = 16 ) :: specname = 'RUNAQT'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNAQT.SPC'

!  ------------------ Open the specfile for aqt ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword  = 'write-problem-data'
       spec( 2 )%keyword  = 'write-result-summary'
       spec( 3 )%keyword  = 'aqt-result-summary-file-name'
       spec( 4 )%keyword = 'aqt-result-summary-file-device'
       spec( 5 )%keyword  = ''
       spec( 6 )%keyword = ''
       spec( 7 )%keyword  = 'print-full-solution'
       spec( 8 )%keyword  = 'write-solution'
       spec( 9 )%keyword  = 'aqt-solution-file-name'
       spec( 10 )%keyword  = 'aqt-solution-file-device'
       spec( 11 )%keyword  = ''
       spec( 12 )%keyword  = ''
       spec( 13 )%keyword = 'radius'
       spec( 14 )%keyword = ''
       spec( 15 )%keyword = ''

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_logical( spec( 2 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 3 ), aqt_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), aqt_rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 7 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 8 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 9 ), aqt_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 10 ), aqt_sfiledevice, errout )
       CALL SPECFILE_assign_real( spec( 13 ), radius, errout )
     END IF

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2008' )

!  Set up data for next problem

     CALL AQT_initialize( data, control, inform )
     IF ( is_specfile ) CALL AQT_read_specfile( control, input_specfile )
     IF ( is_specfile ) CLOSE( input_specfile )

!  Read the initial point and bounds

     CALL CUTEST_udimen( cutest_status, input, n )
     IF ( cutest_status /= 0 ) GO TO 910

     ALLOCATE( X( n ), X0( n ), X_l( n ), X_u( n ), G( n ), VNAMES( n ),       &
               R( n ), VECTOR( n ), H_vector( n ) )
     CALL CUTEST_usetup( cutest_status, input, control%error, io_buffer,       &
                         n, X0, X_l, X_u )
     IF ( cutest_status /= 0 ) GO TO 910
     DEALLOCATE( X_l, X_u )

!  Read the problem and variable names

     CALL CUTEST_unames( cutest_status, n, pname, VNAMES )
     IF ( cutest_status /= 0 ) GO TO 910

!  Set f to zero

     control%f_0 = zero

!  Evaluate the gradient

     CALL CUTEST_ugr( cutest_status, n, X0, G )
     IF ( cutest_status /= 0 ) GO TO 910

!  Use AQT

     solv = 'AQT'

!  If required, open a file for the results

     IF ( write_result_summary ) THEN
       INQUIRE( FILE = aqt_rfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( aqt_rfiledevice, FILE = aqt_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
       ELSE
          OPEN( aqt_rfiledevice, FILE = aqt_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( errout, 2030 ) iores, aqt_rfilename
         STOP
       END IF
       WRITE( aqt_rfiledevice, "( A10 )" ) pname
     END IF

!  Solve the problem

     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( ' AQT used ', / )" )

 ! Initialize control parameters

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )
     R = G                ! The linear term is the gradient
     goth = .FALSE.
     inform%status = 1
     DO                   !  Iteration to find the minimizer
       CALL AQT_solve( n, radius, f, X, R, VECTOR, data, control, inform )
       SELECT CASE( inform%status ) ! Branch as a result of inform%status
       CASE( 2 )         ! Form the preconditioned gradient
       CASE( 3 )         ! Form the matrix-vector product
         CALL CUTEST_uhprod( status, n, goth, X, VECTOR, H_vector )
         VECTOR = H_vector
         goth = .TRUE.
       CASE ( 5 )        !  Restart
         R = G
       CASE ( - 30, 0 )  !  Successful return
         WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier = ',  &
        &    2ES12.4 )" ) inform%iter + inform%iter_pass2, f, inform%multiplier
         EXIT
       CASE DEFAULT      !  Error returns
         WRITE( 6, "( ' AQT_solve exit status = ', I6 ) " ) inform%status
         EXIT
      END SELECT
     END DO
     CALL AQT_terminate( data, control, inform ) ! delete internal workspace

     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     time_now = time_now - time_start
     clock_now = clock_now - clock_start
     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( /, ' AQT used ' )" )

!  If required, append results to a file,

     IF ( write_result_summary ) THEN
       BACKSPACE( aqt_rfiledevice )
       IF ( inform%status == 0 ) THEN
         WRITE( aqt_rfiledevice, 2040 ) pname, n, f, inform%multiplier,       &
          inform%iter, inform%iter_pass2, clock_now, inform%status
       ELSE
         WRITE( aqt_rfiledevice, 2040 ) pname, n, f, inform%multiplier,       &
          - inform%iter, - inform%iter_pass2, - clock_now, inform%status
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
       WRITE( errout, 2050 ) pname, n, f, inform%multiplier,                   &
           inform%iter, inform%iter_pass2, clock_now, inform%status, solv
     ELSE
       WRITE( errout, 2050 ) pname, n, f, inform%multiplier,                   &
           - inform%iter, - inform%iter_pass2, - clock_now, inform%status, solv
     END IF

     IF ( write_solution .AND.                                                 &
         ( inform%status == 0  .OR. inform%status == - 10 ) ) THEN
       INQUIRE( FILE = aqt_sfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( aqt_sfiledevice, FILE = aqt_sfilename, FORM = 'FORMATTED',   &
              STATUS = 'OLD', IOSTAT = iores )
       ELSE
          OPEN( aqt_sfiledevice, FILE = aqt_sfilename, FORM = 'FORMATTED',   &
               STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( out, 2030 ) iores, aqt_sfilename ; STOP ; END IF
       WRITE( aqt_sfiledevice, "( /, ' Problem:    ', A10, /,' Solver :   ',  &
      &       A, /, ' Objective:', ES24.16 )" ) pname, solv, f
       WRITE( aqt_sfiledevice, 2000 )
       DO i = 1, n
         WRITE( aqt_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
       END DO
     END IF
     DEALLOCATE( X, X0, G, R, VECTOR, VNAMES )

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
 2040 FORMAT( A10, I6, 2ES16.8, 2I6, F9.2, I5 )
 2050 FORMAT( A10, I6, 2ES16.8, 2I6, F9.2, I5, 1X, A )
 2060 FORMAT( /, 'name           n  f               lambda    ',               &
                 '      iter iter2     time stat' )

!  End of subroutine USE_AQT

     END SUBROUTINE USE_AQT

!  End of module USEAQT_double

   END MODULE GALAHAD_USEAQT_double
