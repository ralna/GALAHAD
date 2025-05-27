! THIS VERSION: GALAHAD 5.3 - 2025-05-20 AT 10:05 GMT.

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

!------------------------------------
!   L o c a l   P a r a m e t e r s
!------------------------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
!    REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_

!----------------------------------
!   L o c a l   V a r i a b l e s
!----------------------------------

     INTEGER ( KIND = ip_ ) :: iores, i, j, ir, ic, l, smt_stat, cutest_status

!  Functions

!$   INTEGER ( KIND = ip_ ) :: OMP_GET_MAX_THREADS

!  Problem characteristics

     INTEGER ( KIND = ip_ ) :: n, nnzh
     INTEGER ( KIND = ip_ ) :: n_threads = 1
     REAL ( KIND = rp_ ) ::  f
     CHARACTER ( LEN = 10 ) :: pname
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X0, X_l, X_u, G
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RADIUS
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES
     TYPE ( SMT_type ) :: H

!  Problem input characteristics

     LOGICAL :: filexx, is_specfile

!  Default values for specfile-defined parameters

     INTEGER ( KIND = ip_ ) :: trek_rfiledevice = 47
     INTEGER ( KIND = ip_ ) :: trek_sfiledevice = 62
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: trek_rfilename = 'TREKRES.d'
     CHARACTER ( LEN = 30 ) :: trek_sfilename = 'TREKSOL.d'
     REAL ( KIND = rp_ ) ::  radius_1 = 1.0_rp_
     INTEGER ( KIND = ip_ ) :: m = 1
!    LOGICAL :: one_norm = .TREKE.

!  Output file characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
     INTEGER ( KIND = ip_ ) :: out  = 6
     INTEGER ( KIND = ip_ ) :: errout = 6
     CHARACTER ( LEN =  6 ) :: solv

!  Specfile characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 10
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
       spec( 5 )%keyword = 'print-full-solution'
       spec( 6 )%keyword = 'write-solution'
       spec( 7 )%keyword = 'trek-solution-file-name'
       spec( 8 )%keyword = 'trek-solution-file-device'
       spec( 9 )%keyword = 'radius'
       spec( 10 )%keyword = 'number-of-radii'

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_logical( spec( 2 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 3 ), trek_rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), trek_rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 5 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 6 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 7 ), trek_sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 8 ), trek_sfiledevice, errout )
       CALL SPECFILE_assign_real( spec( 9 ), radius_1, errout )
       CALL SPECFILE_assign_integer( spec( 10 ), m, errout )
     END IF

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2008' )

!  Set up data for next problem

     CALL TREK_initialize( data, control, inform )
     IF ( is_specfile ) CALL TREK_read_specfile( control, input_specfile )
     IF ( is_specfile ) CLOSE( input_specfile )

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

     solv = 'TREK   '

!  Evaluate the Hessian

     CALL CUTEST_udimsh_r( cutest_status, nnzh )
     IF ( cutest_status /= 0 ) GO TO 910
     H%ne = nnzh
     CALL SMT_put( H%type, 'COORDINATE', smt_stat )
     ALLOCATE( H%row( nnzh ), H%col( nnzh ), H%val( nnzh ) )
     CALL CUTEST_ush_r( cutest_status, n, X0, H%ne, nnzh, H%val, H%row, H%col )
     IF ( cutest_status /= 0 ) GO TO 910

     ALLOCATE( RADIUS( m ) )
     RADIUS( 1 ) = radius_1

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

!  Solve the problem

     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( ' TREK used ', / )" )

     CALL TREK_solve( n, m, f, G, H, RADIUS, X, data, control, inform )

     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( /, ' TREK used ' )" )
     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( /, ' non-zeros and fill-in ', I0, 1X, I0,        &
      &    ', solver: ', A )" ) nnzh, inform%SLS_inform%entries_in_factors,    &
         TRIM( control%definite_linear_solver )
!$    n_threads = OMP_GET_MAX_THREADS( )
      WRITE( out, "( ' number of threads = ', I0 )" ) n_threads

!  If required, append results to a file,

     IF ( write_result_summary ) THEN
       BACKSPACE( trek_rfiledevice )
       IF ( inform%status == 0 ) THEN
         WRITE( trek_rfiledevice, 2040 ) pname, n, inform%obj,                 &
           inform%multiplier,                                                  &
           inform%factorizations, inform%time%clock_total, inform%status
       ELSE
         WRITE( trek_rfiledevice, 2040 ) pname, n, inform%obj,                 &
           inform%multiplier,                                                  &
           inform%factorizations, - inform%time%clock_total, inform%status
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
       INQUIRE( FILE = trek_sfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( trek_sfiledevice, FILE = trek_sfilename, FORM = 'FORMATTED',   &
              STATUS = 'OLD', IOSTAT = iores )
       ELSE
          OPEN( trek_sfiledevice, FILE = trek_sfilename, FORM = 'FORMATTED',   &
               STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( out, 2030 ) iores, trek_sfilename ; STOP ; END IF
       WRITE( trek_sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ',&
      &       A, /, ' Objective:', ES24.16 )" ) pname, solv, inform%obj
       WRITE( trek_sfiledevice, 2000 ) TRIM( solv )
       DO i = 1, n
         WRITE( trek_sfiledevice, 2020 ) i, VNAMES( i ), X( i )
       END DO
     END IF
     CALL TREK_terminate( data, control, inform )
     DEALLOCATE( H%val, H%row, H%col )
     DEALLOCATE( X, X0, G, VNAMES )
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
 2040 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5 )
 2050 FORMAT( A10, I6, 2ES16.8, I4, F9.2, I5, 1X, A )
 2060 FORMAT( /, 'name           n  f               lambda    ',               &
                 '     fac     time stat' )

!  End of subroutine USE_TREK

     END SUBROUTINE USE_TREK

!  End of module USETREK

   END MODULE GALAHAD_USETREK_precision
