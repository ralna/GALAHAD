! THIS VERSION: GALAHAD 4.3 - 2023-12-31 AT 10:40 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*-*-*-*-  G A L A H A D   U S E S L L S   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. March 26th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USESLLS_precision

!     -------------------------------------------------------
!    | CUTEst/AMPL interface to SLLS, a preconditioned       |
!    | projected conjugate-gradient algorithm for            |
!    | simplex-constrained linear least-squares minimization |
!     -------------------------------------------------------

      USE GALAHAD_KINDS_precision
      USE CUTEST_INTERFACE_precision
      USE GALAHAD_QPT_precision
      USE GALAHAD_SLLS_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_SLLS

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ S L L S  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_SLLS( input )

!  --------------------------------------------------------------------
!
!  Solve the simplex-constrained linear-least squares problem
!
!      minimize     1/2 || A_o x - b ||_2^2
!
!     subject to      e^T x = 1, x >= 0
!
!  using the GALAHAD package GALAHAD_SLLS
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: input

!  Parameters

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19

!  Scalars

      INTEGER ( KIND = ip_ ) :: n, o, ir, ic, iores, smt_stat, cutest_status
      INTEGER ( KIND = ip_ ) :: i, j, k, l, nfixed, alloc_stat, nnzj
      INTEGER ( KIND = ip_ ) :: n_s, slls_status
      INTEGER ( KIND = ip_ ) :: e_order = 0
      INTEGER ( KIND = ip_ ) :: l_order = 0
      INTEGER ( KIND = ip_ ) :: v_order = 0
!     INTEGER ( KIND = ip_ ) :: np1, npm
!     INTEGER ( KIND = ip_ ) :: factorization_integer, factorization_real
      REAL :: time, timeo, times, timet
      REAL ( KIND = rp_ ) :: stopr
      LOGICAL :: filexx, printo, printe, is_specfile
!     LOGICAL :: ldummy

!  Specfile characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = 21
      CHARACTER ( LEN = 16 ) :: specname = 'RUNSLLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNSLLS.SPC'

!  The default values for SLLS could have been set as:

! BEGIN RUNSLLS SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            SLLS.data
!  problem-data-file-device                          26
!  write-initial-sif                                 NO
!  initial-sif-file-name                             INITIAL.SIF
!  initial-sif-file-device                           51
!  scale-problem                                     0
!  solve-problem                                     YES
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                SLLSSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          SLLSRES.d
!  result-summary-file-device                        47
! END RUNSLLS SPECIFICATIONS

!  Default values for specfile-defined parameters

!     INTEGER ( KIND = ip_ ) :: scale = 0
      INTEGER ( KIND = ip_ ) :: dfiledevice = 26
      INTEGER ( KIND = ip_ ) :: ifiledevice = 51
      INTEGER ( KIND = ip_ ) :: rfiledevice = 47
      INTEGER ( KIND = ip_ ) :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'SLLS.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'SLLSRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'SLLSSOL.d'
!     LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.

!  Output file characteristics

      INTEGER ( KIND = ip_ ), PARAMETER :: out  = 6
      INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
      INTEGER ( KIND = ip_ ) :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( SLLS_data_type ) :: data
      TYPE ( SLLS_control_type ) :: SLLS_control
      TYPE ( SLLS_inform_type ) :: SLLS_inform
      TYPE ( GALAHAD_userdata_type ) :: userdata
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
!     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: SH, SA
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u
      REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: X_stat

      CALL CPU_TIME( time )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen_r( cutest_status, input, n, o )
      IF ( cutest_status /= 0 ) GO TO 910

!  allocate temporary arrays

      ALLOCATE( X( n ), X_l( o ), X_u( o ), Y( o ), C_l( o ), C_u( o ),        &
                EQUATN( o ), LINEAR( o ), STAT = alloc_stat )

      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X etc', alloc_stat ; STOP
      END IF

!  set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup_r( cutest_status, input, out,                         &
                            io_buffer, n, o, X, X_l, X_u, Y, C_l, C_u,         &
                            EQUATN, LINEAR, e_order, l_order, v_order )
      DEALLOCATE( X_l, X_u, C_l, C_u, Y, LINEAR, STAT = alloc_stat )

!  count the number of slack variables, and set problem dimensions

      n_s = o - COUNT( EQUATN )
      prob%o = o ; prob%n = n + n_s

!  Determine the names of the problem, variables and constraints.

      ALLOCATE( VNAME( prob%n ), CNAME( o ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'VNAME etc', alloc_stat ; STOP
      END IF

      CALL CUTEST_cnames_r( cutest_status, n, o, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( /, ' Problem: ', A )" ) pname

!  allocate problem arrays

      ALLOCATE( prob%X( prob%n ), prob%B( prob%o ), prob%C( prob%o ),          &
                prob%Z( prob%n ), X_stat( prob%n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'prob%X etc', alloc_stat ; STOP
      END IF

!  transfer data to problem

      prob%X( : n ) = X( : n )

!  determine the number of entries in the Jacobian, and set its dimensions

      CALL CUTEST_cdimsj_r( cutest_status, nnzj )
      IF ( cutest_status /= 0 ) GO TO 910
      prob%Ao%m = prob%o ; prob%Ao%n = prob%n ; prob%Ao%ne = nnzj + n_s
      CALL SMT_put( prob%Ao%type, 'COORDINATE', smt_stat )

!  allocate problem arrays

      ALLOCATE( prob%Ao%val( prob%Ao%ne ), prob%Ao%row( prob%Ao%ne ),          &
                prob%Ao%col( prob%Ao%ne ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'prob%Ao%val etc', alloc_stat ; STOP
      END IF

!  compute the values of the constraints and Jacobian

      CALL CUTEST_ccfsg_r( cutest_status, n, o, X, prob%C, nnzj, prob%Ao%ne,   &
                           prob%Ao%val, prob%Ao%col, prob%Ao%row, .TRUE. )
      prob%B = - prob%C

!  deal with slack variables

      prob%Ao%ne = nnzj
      IF ( n_s > 0 ) THEN
        l = n
        DO i = 1, o
          IF ( .NOT. EQUATN( i ) ) THEN
            l = l + 1
            prob%X( l ) = zero
            VNAME( l ) = CNAME( i )
            prob%Ao%ne = prob%Ao%ne + 1
            prob%Ao%row( prob%Ao%ne ) = i
            prob%Ao%col( prob%Ao%ne ) = l
            prob%Ao%val( prob%Ao%ne ) = - one
          END IF
        END DO
      END IF
      DEALLOCATE( X, CNAME, EQUATN, prob%C, STAT = alloc_stat )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )

!  ------------------ Open the specfile for runslls ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec( 1 )%keyword = 'write-problem-data'
        spec( 2 )%keyword = 'problem-data-file-name'
        spec( 3 )%keyword = 'problem-data-file-device'
        spec( 4 )%keyword = 'write-initial-sif'
        spec( 5 )%keyword = 'initial-sif-file-name'
        spec( 6 )%keyword = 'initial-sif-file-device'
!       spec( 8 )%keyword = 'scale-problem'
!       spec( 13 )%keyword = 'solve-problem'
        spec( 14 )%keyword = 'print-full-solution'
        spec( 15 )%keyword = 'write-solution'
        spec( 16 )%keyword = 'solution-file-name'
        spec( 17 )%keyword = 'solution-file-device'
        spec( 18 )%keyword = 'write-result-summary'
        spec( 19 )%keyword = 'result-summary-file-name'
        spec( 20 )%keyword = 'result-summary-file-device'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
!       CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
!       CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
        CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
      END IF

!  If required, print out the (raw) problem data

      IF ( write_problem_data ) THEN
        INQUIRE( FILE = dfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', IOSTAT = iores )
        ELSE
           OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',            &
                  STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2160 ) iores, dfilename
          STOP
        END IF

        n = prob%n ; k = prob%Ao%ne
        WRITE( dfiledevice, "( 'o, n = ', I0, 1X, I0 )" ) o, n
        WRITE( dfiledevice, "( ' b ', /, ( 5ES12.4 ) )" ) prob%B( : o )
        WRITE( dfiledevice, "( ' Ao_row ', /, ( 10I6 ) )" ) prob%Ao%row( : k )
        WRITE( dfiledevice, "( ' Ao_col ', /, ( 10I6 ) )" ) prob%Ao%col( : k )
        WRITE( dfiledevice, "( ' Ao_val ', /, ( 5ES12.4 ) )" ) prob%Ao%val( : k)

        CLOSE( dfiledevice )
      END IF

!  If required, append results to a file

      IF ( write_result_summary ) THEN
        INQUIRE( FILE = rfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
        ELSE
           OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',            &
                 STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2160 ) iores, rfilename
          STOP
        END IF
        WRITE( rfiledevice, "( A )" ) pname
      END IF

!  Set all default values, and override defaults if requested

      CALL SLLS_initialize( data, SLLS_control, SLLS_inform )
      IF ( is_specfile ) CALL SLLS_read_specfile( SLLS_control, input_specfile )

      printo = out > 0 .AND. SLLS_control%print_level > 0
      printe = out > 0 .AND. SLLS_control%print_level >= 0
      WRITE( out, "( /, ' problem dimensions:  o = ', I0, ', n = ', I0,        &
     &  ', Ao_ne = ', I0 )" ) prob%o, prob%n, prob%Ao%ne

      IF ( printo ) CALL COPYRIGHT( out, '2020' )
      X_stat = 0

!  Solve the problem

      CALL CPU_TIME( timeo )

      solv = ' SLLS'
      IF ( printo ) WRITE( out, " ( ' ** SLLS solver used ** ' ) " )
      SLLS_inform%status = 1
      CALL SLLS_solve( prob, X_stat, data, SLLS_control, SLLS_inform,          &
                       userdata )
      slls_status = SLLS_inform%status

      IF ( printo ) WRITE( out, " ( /, ' ** SLLS solver used ** ' ) " )
      CALL SYMBOLS_status( slls_status, out, ' ', 'SLLS_solve' )
      CALL CPU_TIME( timet )

!  Deallocate arrays from the minimization

      stopr = SLLS_control%stop_d
      CALL SLLS_terminate( data, SLLS_control, SLLS_inform )

!  Print details of the solution obtained

      WRITE( out, 2010 ) slls_status
      IF ( slls_status == 0 .OR. slls_status == - 8 .OR.                       &
           slls_status == - 9 .OR. slls_status == - 10 ) THEN
        l = 4
        IF ( fulsol ) l = prob%n
        ir = 1 ; ic = MIN( l, prob%n )

!  Print details of the primal and dual variables

        WRITE( out, 2090 )
        DO j = 1, 2
          IF ( j == 2 ) THEN
            IF ( ic < prob%n - l ) WRITE( out, 2000 )
            ir = MAX( ic + 1, prob%n - ic + 1 ) ; ic = prob%n
          END IF
          DO i = ir, ic
            state = ' FREE'
            IF ( ABS( prob%X( i ) ) < ten * stopr ) state = 'LOWER'
            WRITE( out, 2050 ) i, VNAME( i ), state, prob%X( i ),              &
                               prob%Z( i )
          END DO
        END DO

!  Compute the number of fixed and degenerate variables.

        nfixed = COUNT( prob%X( : n ) < ten * stopr )

        WRITE( out, "( /, ' Of the ', I0, ' variables, ', I0,                  &
       &                  ' are at zero' )" ) prob%n, nfixed

!  If required, write the solution to a file

        IF ( write_solution ) THEN
          INQUIRE( FILE = sfilename, EXIST = filexx )
          IF ( filexx ) THEN
             OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                 STATUS = 'OLD', IOSTAT = iores )
          ELSE
             OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                  STATUS = 'NEW', IOSTAT = iores )
          END IF
          IF ( iores /= 0 ) THEN
            write( out, 2160 ) iores, sfilename
            STOP
          END IF

          WRITE( sfiledevice, "( /, ' Problem:    ', A10, /, ' Solver :   ',   &
         &       A5, /, ' Objective:', ES24.16 )" ) pname, solv, SLLS_inform%obj
          WRITE( sfiledevice, 2090 )

          DO i = 1, n
            state = ' FREE'
            IF ( prob%X( i ) < ten * stopr ) state = 'LOWER'
            WRITE( sfiledevice, 2050 ) i, VNAME( i ), STATE, prob%X( i ),      &
              prob%Z( i )
          END DO

          CLOSE( sfiledevice )
        END IF
      END IF

      times = times - time ; timet = timet - timeo
      WRITE( out, "( /, ' Total time = ', 0P, F12.2 )" ) times + timet
      WRITE( out, 2070 ) pname

!  Compare the variants used so far

      WRITE( out, "( A5, I7, 6X, ES12.4, I6, 0P, 3F8.2 )" ) solv,              &
                     SLLS_inform%iter, SLLS_inform%obj, slls_status,           &
                     times, timet, times + timet

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        WRITE( rfiledevice, "( A10, 2I7, I6, ES13.4, I6, 0P, F8.2 )" ) pname,  &
           o, n, SLLS_inform%iter, SLLS_inform%obj, slls_status, timet
      END IF

      DEALLOCATE( prob%X, prob%B, prob%R, prob%G, X_stat,                      &
                  prob%Ao%val, prob%Ao%row, prob%Ao%col, prob%Ao%type,         &
                  VNAME, STAT = alloc_stat )
      IF ( is_specfile ) CLOSE( input_specfile )

      CALL CUTEST_cterminate_r( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      RETURN

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' )
 2010 FORMAT( /,' Stopping with inform%status = ', I3 )
 2050 FORMAT( I7, 1X, A10, A6, 2ES12.4 )
 2070 FORMAT( /, ' Problem: ', A, //,                                          &
                 '                     objective',                             &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  iterations    value  ',                             &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------   ',                           &
                 ' ------ -----    ----   -----  ' )
 2090 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )

!  End of subroutine USE_SLLS

     END SUBROUTINE USE_SLLS

!  End of module USESLLS

   END MODULE GALAHAD_USESLLS_precision
