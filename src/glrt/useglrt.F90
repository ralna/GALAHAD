! THIS VERSION: GALAHAD 5.3 - 2025-07-08 AT 12:35 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ G L R T  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  October 24th 2016

   MODULE GALAHAD_USEGLRT_precision

!  This is the driver program for running GLRT for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_KINDS_precision
     USE CUTEST_INTERFACE_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_GLRT_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_precision
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_GLRT

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ G L R T   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_GLRT( input )

!  Dummy argument

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( GLRT_control_type ) :: control
     TYPE ( GLRT_inform_type ) :: inform
     TYPE ( GLRT_data_type ) :: data

!------------------------------------
!   L o c a l   P a r a m e t e r s
!------------------------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
!    REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_

!----------------------------------
!   L o c a l   V a r i a b l e s
!----------------------------------

     INTEGER ( KIND = ip_ ) :: iores, i, j, ir, ic, l, cutest_status
     REAL :: time_now, time_start
     REAL ( KIND = rp_ ) :: clock_now, clock_start
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: R, VECTOR, H_vector
     LOGICAL :: goth

!  Problem characteristics

     INTEGER ( KIND = ip_ ) :: n
     REAL ( KIND = rp_ ) ::  f
     CHARACTER ( LEN = 10 ) :: pname
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X0, X_l, X_u, G
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES

!  Problem input characteristics

     LOGICAL :: filexx, is_specfile

!  Default values for specfile-defined parameters

     INTEGER ( KIND = ip_ ) :: glrt_rfiledevice = 47
     INTEGER ( KIND = ip_ ) :: glrt_sfiledevice = 62
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: glrt_rfilename = 'GLRTRES.d'
     CHARACTER ( LEN = 30 ) :: glrt_sfilename = 'GLRTSOL.d'
     REAL ( KIND = rp_ ) ::  order = 3.0_rp_
     REAL ( KIND = rp_ ) ::  weight = 1.0_rp_

!  Output file characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
     INTEGER ( KIND = ip_ ) :: out  = 6
     INTEGER ( KIND = ip_ ) :: errout = 6
     CHARACTER ( LEN = 4 ) :: solv

!  Specfile characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 15
     CHARACTER ( LEN = 16 ) :: specname = 'RUNGLRT'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNGLRT.SPC'

!  ------------------ Open the specfile for glrt ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword  = 'write-problem-data'
       spec( 2 )%keyword  = 'write-result-summary'
       spec( 3 )%keyword  = 'glrt-result-summary-file-name'
       spec( 4 )%keyword = 'glrt-result-summary-file-device'
       spec( 5 )%keyword  = ''
       spec( 6 )%keyword = ''
       spec( 7 )%keyword  = 'print-full-solution'
       spec( 8 )%keyword  = 'write-solution'
       spec( 9 )%keyword  = 'glrt-solution-file-name'
       spec( 10 )%keyword  = 'glrt-solution-file-device'
       spec( 11 )%keyword  = ''
       spec( 12 )%keyword = 'order'
       spec( 13 )%keyword = 'weight'
       spec( 14 )%keyword = ''
       spec( 15 )%keyword = ''

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_logical( spec( 2 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 3 ), glrt_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), glrt_rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 7 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 8 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 9 ), glrt_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 10 ), glrt_sfiledevice, errout )
       CALL SPECFILE_assign_real( spec( 12 ), order, errout )
       CALL SPECFILE_assign_real( spec( 13 ), weight, errout )
     END IF

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2008' )

!  Set up data for next problem

     CALL GLRT_initialize( data, control, inform )
     IF ( is_specfile ) CALL GLRT_read_specfile( control, input_specfile )
     IF ( is_specfile ) CLOSE( input_specfile )

!  Read the initial point and bounds

     CALL CUTEST_udimen_r( cutest_status, input, n )
     IF ( cutest_status /= 0 ) GO TO 910

     ALLOCATE( X( n ), X0( n ), X_l( n ), X_u( n ), G( n ), VNAMES( n ),       &
               R( n ), VECTOR( n ), H_vector( n ) )
     CALL CUTEST_usetup_r( cutest_status, input, control%error, io_buffer,     &
                           n, X0, X_l, X_u )
     IF ( cutest_status /= 0 ) GO TO 910
     DEALLOCATE( X_l, X_u )

!  Read the problem and variable names

     CALL CUTEST_unames_r( cutest_status, n, pname, VNAMES )
     IF ( cutest_status /= 0 ) GO TO 910

!  Set f to zero

     control%f_0 = zero

!  Evaluate the gradient

     CALL CUTEST_ugr_r( cutest_status, n, X0, G )
     IF ( cutest_status /= 0 ) GO TO 910

!  Use GLRT

     solv = 'GLRT'

!  If required, open a file for the results

     IF ( write_result_summary ) THEN
       INQUIRE( FILE = glrt_rfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( glrt_rfiledevice, FILE = glrt_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
       ELSE
          OPEN( glrt_rfiledevice, FILE = glrt_rfilename, FORM = 'FORMATTED',   &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( errout, 2030 ) iores, glrt_rfilename
         STOP
       END IF
       WRITE( glrt_rfiledevice, "( A10 )" ) pname
     END IF

!  Solve the problem

     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( ' GLRT used ', / )" )

 ! Initialize control parameters

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )
     R = G                ! The linear term is the gradient
     goth = .FALSE.
     inform%status = 1
     DO                   !  Iteration to find the minimizer
       CALL GLRT_solve( n, order, weight, X, R, VECTOR, data, control, inform )
       SELECT CASE( inform%status ) ! Branch as a result of inform%status
       CASE( 2 )         ! Form the preconditioned gradient
       CASE( 3 )         ! Form the matrix-vector product
         CALL CUTEST_uhprod_r( cutest_status, n, goth, X, VECTOR, H_vector )
         VECTOR = H_vector
         goth = .TRUE.
       CASE ( 4 )        !  Restart
         R = G
       CASE ( - 30, 0 )  !  Successful return
         WRITE( 6, "( I6, ' iterations. Solution and Lagrange multiplier = ',  &
        &    2ES12.4 )" ) inform%iter + inform%iter_pass2,                     &
                          inform%obj_regularized, inform%multiplier
         EXIT
       CASE DEFAULT      !  Error returns
         WRITE( 6, "( ' GLRT_solve exit status = ', I6 ) " ) inform%status
         EXIT
      END SELECT
     END DO
     CALL GLRT_terminate( data, control, inform ) ! delete internal workspace

     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     time_now = time_now - time_start
     clock_now = clock_now - clock_start
     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( /, ' GLRT used ' )" )

!  If required, append results to a file,

     IF ( write_result_summary ) THEN
       BACKSPACE( glrt_rfiledevice )
       IF ( inform%status == 0 ) THEN
         WRITE( glrt_rfiledevice, 2040 ) pname, n, inform%obj_regularized,     &
           inform%multiplier, inform%iter, inform%iter_pass2, clock_now,       &
           inform%status
       ELSE
         WRITE( glrt_rfiledevice, 2040 ) pname, n, inform%obj_regularized,     &
          inform%multiplier, - inform%iter, - inform%iter_pass2, - clock_now,  &
          inform%status
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

     WRITE( errout, 2060 )
     IF ( inform%status == 0 ) THEN
       WRITE( errout, 2050 ) pname, n, inform%obj_regularized,                 &
           inform%multiplier, inform%iter, inform%iter_pass2, clock_now,       &
           inform%status, solv
     ELSE
       WRITE( errout, 2050 ) pname, n, inform%obj_regularized,                 &
           inform%multiplier, - inform%iter, - inform%iter_pass2, - clock_now, &
           inform%status, solv
     END IF

     IF ( write_solution .AND.                                                 &
         ( inform%status == 0  .OR. inform%status == - 10 ) ) THEN
       INQUIRE( FILE = glrt_sfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( glrt_sfiledevice, FILE = glrt_sfilename, FORM = 'FORMATTED',   &
              STATUS = 'OLD', IOSTAT = iores )
       ELSE
          OPEN( glrt_sfiledevice, FILE = glrt_sfilename, FORM = 'FORMATTED',   &
               STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( out, 2030 ) iores, glrt_sfilename ; STOP ; END IF
       WRITE( glrt_sfiledevice, "( /, ' Problem:    ', A10, /,' Solver :   ',  &
      &       A, /, ' Objective:', ES24.16 )" ) pname, solv, f
       WRITE( glrt_sfiledevice, 2000 ) TRIM( solv )
       DO i = 1, n
         WRITE( glrt_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
       END DO
     END IF
     DEALLOCATE( X, X0, G, R, VECTOR, H_vector, VNAMES )

     CALL CUTEST_cterminate_r( cutest_status )
     RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )          &
       cutest_status
     inform%status = - 98
     RETURN

!  Non-executable statements

 2000 FORMAT( ' Solver: ', A, /, ' Solution: ', /,                             &
               '      # name          value   ' )
 2010 FORMAT( 6X, '. .', 9X, ( 2X, 10( '.' ) ) )
 2020 FORMAT( I7, 1X, A10, ES12.4 )
 2030 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2040 FORMAT( A10, I6, 2ES16.8, 2I6, F9.2, I5 )
 2050 FORMAT( A10, I6, 2ES16.8, 2I6, F9.2, I5, 1X, A )
 2060 FORMAT( /, 'name           n  f               lambda    ',               &
                 '      iter iter2     time stat' )

!  End of subroutine USE_GLRT

     END SUBROUTINE USE_GLRT

!  End of module USEGLRT

   END MODULE GALAHAD_USEGLRT_precision
