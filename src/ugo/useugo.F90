! THIS VERSION: GALAHAD 4.3 - 2024-02-01 AT 16:30 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ U G O  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  June 3rd 2016

   MODULE GALAHAD_USEUGO_precision

!  This is the driver program for running UGO for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE GALAHAD_KINDS_precision
!    USE GALAHAD_CLOCK
     USE GALAHAD_UGO_precision
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_precision
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type
     USE GALAHAD_CUTEST_precision
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_UGO

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ U G O   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_UGO( input )

!  Dummy argument

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( UGO_control_type ) :: control
     TYPE ( UGO_inform_type ) :: inform
     TYPE ( UGO_data_type ) :: data
     TYPE ( NLPT_problem_type ) :: nlp
     TYPE ( GALAHAD_userdata_type ) :: userdata
     TYPE ( CUTEST_control_type ) :: cutest_control
     TYPE ( CUTEST_inform_type ) :: cutest_inform

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

!  Problem input characteristics

     INTEGER ( KIND = ip_ ) :: iores, i, status
     LOGICAL :: filexx, is_specfile
!    REAL :: timeo, timet
!    REAL ( KIND = rp_ ) :: clocko, clockt
     REAL ( KIND = rp_ ) :: x, f, g, h
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S, V

!  Functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  Specfile characteristics

     INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 29
     CHARACTER ( LEN = 16 ) :: specname = 'RUNUGO'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNUGO.SPC'

!  Default values for specfile-defined parameters

     INTEGER ( KIND = ip_ ) :: dfiledevice = 26
     INTEGER ( KIND = ip_ ) :: rfiledevice = 47
     INTEGER ( KIND = ip_ ) :: sfiledevice = 62
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: dfilename = 'UGO.data'
     CHARACTER ( LEN = 30 ) :: rfilename = 'UGORES.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'UGOSOL.d'
     LOGICAL :: testal = .FALSE.
     LOGICAL :: dechk  = .FALSE.
     LOGICAL :: dechke = .FALSE.
     LOGICAL :: dechkg = .FALSE.
     LOGICAL :: not_fatal  = .FALSE.
     LOGICAL :: not_fatale = .FALSE.
     LOGICAL :: not_fatalg = .FALSE.

     REAL ( KIND = rp_ ) :: x_l = - 1.0_rp_
     REAL ( KIND = rp_ ) :: x_u = 1.0_rp_

!  Output file characteristics

     INTEGER ( KIND = ip_ ) :: out  = 6
     INTEGER ( KIND = ip_ ) :: errout = 6

!  ------------------ Open the specfile for ugo ----------------

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
       spec( 18 )%keyword = 'lower-bound-on-x'
       spec( 19 )%keyword = 'upper-bound-on-x'
       spec( 20 : 29 )%keyword = ''

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
       CALL SPECFILE_assign_real( spec( 18 ), x_l, errout )
       CALL SPECFILE_assign_real( spec( 19 ), x_u, errout )
     END IF
     IF ( dechk .OR. testal ) THEN ; dechke = .TRUE. ; dechkg = .TRUE. ; END IF
     IF ( not_fatal ) THEN ; not_fatale = .TRUE. ; not_fatalg = .TRUE. ; END IF

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

     IF ( out > 0 ) CALL COPYRIGHT( out, '2016' )

!  Set up control parameters prior to the next solution

     CALL UGO_initialize( data, control, inform )
     IF ( is_specfile ) CALL UGO_read_specfile( control, input_specfile )

!  Initialize the problem data

     cutest_control%input = input ; cutest_control%error = control%error
     CALL CUTEST_initialize( nlp, cutest_control, cutest_inform, userdata )

!  initialize workspace

     ALLOCATE( S( nlp%n ), V( nlp%n ), STAT = i )

!  ===================================================
!  Solve the problem: min f( x * S ), -x_l <= x <= x_l
!  ===================================================

     S = 1.0_rp_
     inform%status = 1
!    CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )
     DO
       CALL UGO_solve( x_l, x_u, x, f, g, h, control, inform, data, userdata )

!  evaluate f( x * S ) and its derivatives as required

       IF ( inform%status >= 2 ) THEN
         nlp%X = x * S
         CALL CUTEST_ufn_r( status, nlp%n, nlp%X, f )
         IF ( inform%status >= 3 ) THEN
           CALL CUTEST_ugr_r( status, nlp%n, nlp%X, V )
           g = DOT_PRODUCT( S, V )
           IF ( inform%status >= 4 ) THEN
             CALL CUTEST_uhprod_r( status, nlp%n, .FALSE., nlp%X, S, V )
             h = DOT_PRODUCT( S, V )
           END IF
         END IF

!  the solution has been found (or an error has occured)

       ELSE
         EXIT
       END IF
     END DO

!    CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )

!$    WRITE( out, "( ' number of threads = ', I0 )" ) OMP_GET_MAX_THREADS( )

!  ================
!  Solution details
!  ================

!  If required, append results to a file,

     IF ( write_result_summary ) THEN
       BACKSPACE( rfiledevice )
       IF ( inform%status == GALAHAD_ok .OR.                                   &
            inform%status == GALAHAD_error_unbounded ) THEN
         WRITE( rfiledevice, 2000 ) nlp%pname, x, f, g,                        &
           inform%iter, inform%f_eval, inform%g_eval,                          &
           inform%time%clock_total, inform%status
       ELSE
         WRITE( rfiledevice, 2000 ) nlp%pname, x, f, g,                        &
           - inform%iter, - inform%f_eval, - inform%g_eval,                    &
           inform%time%clock_total, inform%status
       END IF
     END IF

!  If required, write the solution

     WRITE( errout, "( /, 'name              x             f             g ',  &
    &  '         #f     time stat' )" )

     IF ( inform%status == GALAHAD_ok .OR.                                     &
          inform%status == GALAHAD_error_unbounded ) THEN
       WRITE( errout, 2020 ) nlp%pname, x, f, g,                               &
         inform%f_eval, inform%time%clock_total, inform%status
     ELSE
       WRITE( errout, 2020 ) nlp%pname, x, f, g,                               &
         - inform%f_eval, inform%time%clock_total, inform%status
     END IF

!  Close any opened files and deallocate arrays

     IF ( is_specfile ) CLOSE( input_specfile )
     CALL CUTEST_terminate( nlp, cutest_inform, userdata )
     DEALLOCATE( S, V, STAT = i )
     RETURN

!  Non-executable statements

 2000 FORMAT( A10, 3ES16.8, bn, 3I7, F9.2, I5 )
 2020 FORMAT( A10, 3ES14.6, bn, I7, F9.2, I5 )
 2030 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )

!  End of subroutine USE_UGO

     END SUBROUTINE USE_UGO

!  End of module USEUGO_precision

   END MODULE GALAHAD_USEUGO_precision
