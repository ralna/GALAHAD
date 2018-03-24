! THIS VERSION: GALAHAD 3.0 - 15/03/2018 AT 15:40 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E _ D P S  -*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  March 15th 2018

   MODULE GALAHAD_USEDPS_double

!  This is the driver program for running DPS for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

     USE CUTEst_interface_double
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_DPS_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_COPYRIGHT
     USE GALAHAD_SPACE_double
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_DPS

   CONTAINS

!-*-*-*-*-*-*-*-*-*-  U S E _ D P S   S U B R O U T I N E  -*-*-*-*-*-*-*-

     SUBROUTINE USE_DPS( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( DPS_control_type ) :: control
     TYPE ( DPS_inform_type ) :: inform
     TYPE ( DPS_data_type ) :: data

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

     INTEGER :: rfiledevice = 47
     INTEGER :: sfiledevice = 62
     LOGICAL :: fulsol = .FALSE.
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
!    LOGICAL :: write_result_summary = .FALSE.
     LOGICAL :: write_result_summary = .TRUE.
     CHARACTER ( LEN = 30 ) :: rfilename = 'DPSRES.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'DPSSOL.d'
     REAL ( KIND = wp ) ::  radius = 1.0_wp
     REAL ( KIND = wp ) ::  order = 3.0_wp
     REAL ( KIND = wp ) ::  weight = 1.0_wp
     LOGICAL :: trs = .TRUE.

!  Output file characteristics

     INTEGER, PARAMETER :: io_buffer = 11
     INTEGER :: out  = 6
     INTEGER :: errout = 6
     CHARACTER ( LEN =  6 ) :: solv

!  Specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 15
     CHARACTER ( LEN = 16 ) :: specname = 'RUNDPS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNDPS.SPC'

!  ------------------ Open the specfile for dps ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword  = 'write-problem-data'
       spec( 2 )%keyword  = 'write-result-summary'
       spec( 3 )%keyword  = 'dps-result-summary-file-name'
       spec( 4 )%keyword  = 'dps-result-summary-file-device'
       spec( 7 )%keyword  = 'print-full-solution'
       spec( 8 )%keyword  = 'write-solution'
       spec( 9 )%keyword  = 'dps-solution-file-name'
       spec( 10 )%keyword = 'dps-solution-file-device'
       spec( 11 )%keyword = 'solve-trust-region-subproblem'
       spec( 13 )%keyword = 'trust-region-radius'
       spec( 14 )%keyword = 'regularization-order'
       spec( 15 )%keyword = 'regularization-weight'

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_logical( spec( 2 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 3 ), rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 4 ), rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 7 ), fulsol, errout )
       CALL SPECFILE_assign_logical( spec( 8 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 9 ), sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 10 ), sfiledevice, errout )
       CALL SPECFILE_assign_logical ( spec( 11 ), trs, errout )
       CALL SPECFILE_assign_real( spec( 13 ), radius, errout )
       CALL SPECFILE_assign_real( spec( 14 ), order, errout )
       CALL SPECFILE_assign_real( spec( 15 ), weight, errout )
     END IF

!  Set copyright

     IF ( out > 0 ) CALL COPYRIGHT( out, '2008' )

!  Set up data for next problem

     CALL DPS_initialize( data, control, inform )
     IF ( is_specfile ) CALL DPS_read_specfile( control, input_specfile )
     IF ( is_specfile ) CLOSE( input_specfile )

!  Read the initial point and bounds

     CALL CUTEST_udimen( cutest_status, input, n )
     IF ( cutest_status /= 0 ) GO TO 910

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

     solv = 'DPS   '

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
       WRITE( rfiledevice, "( A10 )" ) pname
     END IF

!  Solve the problem

     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( ' DPS used ', / )" )
     IF ( trs ) THEN
       CALL DPS_solve( n, H, G, f, X, data, control, inform, delta = radius )
     ELSE
       CALL DPS_solve( n, H, G, f, X, data, control, inform, sigma = weight,   &
                       p = order )
     END IF
     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( /, ' DPS used ' )" )
     IF ( control%print_level > 0 .AND. control%out > 0 )                      &
       WRITE( control%out, "( /, ' non-zeros and fill-in ', I0, 1X, I0,        &
      &    ', solver: ', A )" ) nnzh, inform%SLS_inform%entries_in_factors,    &
         TRIM( control%symmetric_linear_solver )
!$    n_threads = OMP_GET_MAX_THREADS( )
      WRITE( out, "( ' number of threads = ', I0 )" ) n_threads

!  If required, append results to a file,

     IF ( write_result_summary ) THEN
       BACKSPACE( rfiledevice )
       IF ( trs ) THEN
         IF ( inform%status == 0 ) THEN
           WRITE( rfiledevice, 2040 ) pname, n, inform%obj,                    &
             inform%multiplier, inform%time%clock_total, inform%status
         ELSE
           WRITE( rfiledevice, 2040 ) pname, n, inform%obj,                    &
             inform%multiplier, - inform%time%clock_total, inform%status
         END IF
       ELSE
         IF ( inform%status == 0 ) THEN
           WRITE( rfiledevice, 2040 ) pname, n, inform%obj_regularized,        &
             inform%multiplier, inform%time%clock_total, inform%status
         ELSE
           WRITE( rfiledevice, 2040 ) pname, n, inform%obj_regularized,        &
             inform%multiplier, - inform%time%clock_total, inform%status
         END IF
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

     IF ( trs ) THEN
       WRITE( errout, 2060 )
       IF ( inform%status == 0 ) THEN
         WRITE( errout, 2050 ) pname, n, inform%obj, inform%multiplier,        &
           inform%time%clock_total, inform%status, solv
       ELSE
         WRITE( errout, 2050 ) pname, n, inform%obj, inform%multiplier,        &
           - inform%time%clock_total, inform%status, solv
       END IF
     ELSE
       WRITE( errout, 2070 )
       IF ( inform%status == 0 ) THEN
         WRITE( errout, 2050 )                                                 &
           pname, n, inform%obj_regularized, inform%multiplier,                &
           inform%time%clock_total, inform%status, solv
       ELSE
         WRITE( errout, 2050 )                                                 &
           pname, n, inform%obj_regularized, inform%multiplier,                &
           - inform%time%clock_total, inform%status, solv
       END IF
     END IF

     IF ( write_solution .AND.                                                 &
         ( inform%status == 0  .OR. inform%status == - 10 ) ) THEN
       INQUIRE( FILE = sfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
              STATUS = 'OLD', IOSTAT = iores )
       ELSE
          OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',             &
               STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( out, 2030 ) iores, sfilename ; STOP ; END IF
       IF ( trs ) THEN
         WRITE( sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ',    &
        &       A, /, ' TRS Objective:', ES24.16 )" ) pname, solv, inform%obj
       ELSE
         WRITE( sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ',    &
        &       A, /, ' RQS Objective:', ES24.16 )" ) pname, solv,             &
           inform%obj_regularized
       END IF
       WRITE( sfiledevice, 2000 )
       DO i = 1, n
         WRITE( sfiledevice, 2020 ) i, VNAMES( i ), X( i )
       END DO
     END IF
     CALL DPS_terminate( data, control, inform )
     DEALLOCATE( H%val, H%row, H%col )

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
 2040 FORMAT( A10, I6, 2ES16.8, F9.2, I5 )
 2050 FORMAT( A10, I6, 2ES16.8, F9.2, I5, 1X, A )
 2060 FORMAT( /, 'name           n  obj             lambda    ',               &
                 '         time stat' )
 2070 FORMAT( /, 'name           n  obj_reg         lambda    ',               &
                 '         time stat' )

!  End of subroutine USE_DPS

     END SUBROUTINE USE_DPS

!  End of module USEDPS_double

   END MODULE GALAHAD_USEDPS_double
