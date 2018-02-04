! THIS VERSION: GALAHAD 2.5 - 09/02/2013 AT 16:25 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E E Q P  *-*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Ph. L. Toint, for GALAHAD productions
!  Copyright reserved
!  October 20th 2007

    MODULE GALAHAD_USELLS_double

!  CUTEst/AMPL interface to GALAHAD_LLS, an algorithm for solving 
!  linear least-squares problems using  a conjugate gradient method

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_QPT_double
      USE GALAHAD_LLS_double
      USE GALAHAD_SPECFILE_double 
      USE GALAHAD_COPYRIGHT

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_LLS

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ L L S  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_LLS( input )

!  --------------------------------------------------------------------
!
!  Solve the linear least-squares problem from CUTEst
!
!     minimize     || A x - b ||_2
!
!  using the GALAHAD package GALAHAD_LLS
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: n, m, la, iores
      INTEGER :: i, ir, ic, j, l, nea, A_ne, smt_stat, natotal
!     INTEGER :: np1, npm
!     INTEGER :: factorization_integer, factorization_real
      INTEGER :: status, alloc_stat, cutest_status
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: objf
      LOGICAL :: filexx, is_specfile
            
!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 13
      CHARACTER ( LEN = 16 ) :: specname = 'RUNLLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNLLS.SPC'

!  The default values for LLS could have been set as:

! BEGIN RUNLLS SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    LLS.data
!  problem-data-file-device                  26
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        LLSSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  LLSRES.d
!  result-summary-file-device                47
!  write-oneline-result-summary              NO
!  result-oneline-summary-file-name          LLSRES_1line.d
!  result-oneline-summary-file-device        47
! END RUNLLS SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: rfiledevice = 47
      INTEGER :: lfiledevice = 48
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_1line_summary  = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'LLS.data'
!     CHARACTER ( LEN = 30 ) :: rfilename = 'LLSRES.d'
      CHARACTER ( LEN = 34 ) :: rfilename = '../results/LLS_IMPLICIT_fact.d'
!     CHARACTER ( LEN = 30 ) :: lfilename = 'LLSRES_1line.d'
      CHARACTER ( LEN = 36 ) :: lfilename ='../results/LLS_IMPLICIT_fact_1line.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'LLSSOL.d'
      LOGICAL :: fulsol = .FALSE. 
      LOGICAL :: printo = .TRUE.

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( LLS_data_type ) :: data
      TYPE ( LLS_control_type ) :: LLS_control        
      TYPE ( LLS_inform_type ) :: LLS_inform
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR

      CALL CPU_TIME( time )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( X0( n ), prob%X_l( n ), prob%X_u( n ),                     &
                VNAME( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X', alloc_stat ; STOP
      END IF

      ALLOCATE( prob%C_l( m ), prob%C_u( m ), prob%Y( m ), CNAME( m ),         &
                EQUATN( m ), LINEAR( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat ; STOP
      END IF

!  Set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup( cutest_status, input, out, io_buffer,                &
                          n, m, X0, prob%X_l, prob%X_u,                        &
                          prob%Y, prob%C_l, prob%C_u, EQUATN, LINEAR, 0, 0, 0 )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( LINEAR )

!  Allocate derived types

      ALLOCATE( prob%X( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X0', alloc_stat
        STOP
      END IF

      ALLOCATE( prob%C( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat
        STOP
      END IF

!  Determine the names of the problem, variables and constraints.

      CALL CUTEST_cnames( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      ALLOCATE( prob%name( 10 ) )
      prob%name = TRANSFER( pname, prob%name )
      WRITE( out, "( /, ' Problem: ', A10 )" ) pname 

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ), MAX( prob%X_l( : n ), X0( : n ) ) )

!  Set X0 to zero to determine the constant terms for the problem functions

      X0 = zero 

!  Evaluate the constant terms of the objective (objf) and constraint 
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, X0, objf, prob%C( : m ) )
      IF ( cutest_status /= 0 ) GO TO 910

!  Determine the number of nonzeros in the Jacobian

      CALL CUTEST_cdimsj( cutest_status, la )
      IF ( cutest_status /= 0 ) GO TO 910
      natotal = la
      la = MAX( la, 1 )

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( natotal ), prob%A%col( natotal ),                  &
                prob%A%val( natotal ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, X0, prob%Y, .FALSE.,              &
                        nea, la, prob%A%val( : la ), prob%A%col( : la ),       &
                        prob%A%row( : la ) )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( X0 )
      
!  Exclude zeros; set the linear term for the objective function

      A_ne = 0
      DO l = 1, nea
        IF ( prob%A%val( l ) /= zero ) THEN
          IF ( prob%A%row( l ) > 0 ) THEN
            A_ne = A_ne + 1
            prob%A%row( A_ne ) = prob%A%row( l ) 
            prob%A%col( A_ne ) = prob%A%col( l )
            prob%A%val( A_ne ) = prob%A%val( l )
          END IF  
        END IF
      END DO

!     WRITE( out, "( ' maximum element of A = ', ES12.4 )                      &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) )

!  Store the problem dimensions

      prob%n    = n
      prob%m    = m
      prob%A%ne = A_ne
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'COORDINATE', smt_stat )

!  Print details
        
!     WRITE( out, "( /, ' m    = ', I10, '  n    = ', I10, /,                  &
!    &               ' A_ne = ', I10 )" ) m, n, A_ne
!     WRITE( out, "( ' maximum element of A = ', ES12.4  )" )                  &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) )
!   END IF  

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times )

!  ------------------ Open the specfile for runlls ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec( 1 )%keyword = 'write-problem-data'
        spec( 2 )%keyword = 'problem-data-file-name'
        spec( 3 )%keyword = 'problem-data-file-device'
        spec( 4 )%keyword = 'write-oneline-result-summary'
        spec( 5 )%keyword = 'result-oneline-summary-file-name'
        spec( 6 )%keyword = 'result-oneline-summary-file-device'
        spec( 7 )%keyword = 'print-full-solution'
        spec( 8 )%keyword = 'write-solution'
        spec( 9 )%keyword = 'solution-file-name'
        spec( 10 )%keyword = 'solution-file-device'
        spec( 11 )%keyword = 'write-result-summary'
        spec( 12 )%keyword = 'result-summary-file-name'
        spec( 13 )%keyword = 'result-summary-file-device'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_1line_summary, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), lfilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), lfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 7 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 8 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 9 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 10 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 11 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 12 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 13 ), rfiledevice, errout )
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

        WRITE( dfiledevice, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" )          &
          n, m, prob%f
        WRITE( dfiledevice, "( ' c ', /, ( 5ES12.4 ) )" ) prob%C( : m )
        WRITE( dfiledevice, "( ' A_row ', /, ( 10I6 ) )" ) prob%A%row( : A_ne )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )

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
          WRITE( out,                                                          &
            "( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )" ) &
            iores, rfilename
          STOP
        END IF
        WRITE( rfiledevice, "( /, ' Problem ', ( 20A ) )" ) pname
      END IF

!  If required, open files for the results

      IF ( write_1line_summary ) THEN
        INQUIRE( FILE = lfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( lfiledevice, FILE = lfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
        ELSE
           OPEN( lfiledevice, FILE = lfilename, FORM = 'FORMATTED',            &
                 STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN 
          WRITE( out,                                                          &
            "( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )" ) &
            iores, lfilename
          STOP
        END IF
        WRITE( lfiledevice, "( 8A )" ) pname
      END IF

!  Set all default values, and override defaults if requested
 
      CALL LLS_initialize( data, LLS_control, LLS_inform )
      IF ( is_specfile )                                                       &
        CALL LLS_read_specfile( LLS_control, input_specfile )

      WRITE( out, "( /, ' problem dimensions:  n = ', I7, ' m = ', I7,         &
     &            ' a_ne = ', I9 )" ) n, m, A_ne

      IF ( printo ) CALL COPYRIGHT( out, '2007' )

!  Call the optimizer

      CALL CPU_TIME( timeo )
  
      IF ( printo ) WRITE( out, " ( ' ** LLS solver used ** ' ) " )
      CALL LLS_solve( prob, data, LLS_control, LLS_inform )

      IF ( printo ) WRITE( out, " ( /, ' Exit from LLS solver' ) " )
  
      CALL CPU_TIME( timet )
  
      status = LLS_inform%status
!     factorization_integer = LLS_inform%factorization_integer 
!     factorization_real = LLS_inform%factorization_real
      CALL LLS_terminate( data, LLS_control, LLS_inform )

!  Print details of the solution obtained

      WRITE( out, 2060 ) LLS_inform%status
      IF ( status == 0 .OR. status == - 8 .OR. status == - 9 .OR.              &
           status == - 10 ) THEN
        l = 4
        IF ( fulsol ) l = n 

!  Print details of the primal and dual variables

        WRITE( out, 2050 ) 
        DO j = 1, 2 
          IF ( j == 1 ) THEN 
            ir = 1 ; ic = MIN( l, n ) 
          ELSE 
            IF ( ic < n - l ) WRITE( out, 2040 ) 
            ir = MAX( ic + 1, n - ic + 1 ) ; ic = n 
          END IF 
          DO i = ir, ic 
            WRITE( out, 2030 ) i, VNAME( i ), prob%X( i )
          END DO 
        END DO 

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

          WRITE( sfiledevice, 2000 )                                           &
            pname, LLS_inform%obj, SQRT( DOT_PRODUCT(  prob%X, prob%X ) ),     &
           LLS_control%preconditioner, LLS_inform%time%total, LLS_inform%cg_iter

          WRITE( sfiledevice, 2050 ) 

          DO i = 1, n 
            WRITE( sfiledevice, 2030 ) i, VNAME( i ), prob%X( i )
          END DO 

          CLOSE( sfiledevice ) 
        END IF 
      END IF 

!  Print details of the solution obtained

      DEALLOCATE( VNAME, CNAME )

      WRITE( out, 2000 )                                                       &
         pname, LLS_inform%obj, SQRT( DOT_PRODUCT(  prob%X, prob%X ) ),        &
         LLS_control%preconditioner, LLS_inform%time%total, LLS_inform%cg_iter

!  If required, write results to  the appropriate files

     IF ( write_1line_summary ) THEN
       BACKSPACE( lfiledevice )
       IF ( status >= 0 ) THEN
         WRITE( lfiledevice, 2010 )                                            &
           pname, LLS_inform%time%total, LLS_inform%time%factorize,            &
           LLS_inform%cg_iter,                                                 &
           LLS_inform%time%factorize + LLS_inform%time%solve,                  &
           LLS_control%GLTR_control%stop_relative, LLS_control%preconditioner
        ELSE
          WRITE( lfiledevice, 2020 ) pname, status
        END IF
      END IF

     IF ( write_result_summary ) THEN
       WRITE( rfiledevice, "( ' Stopping with inform%status = ', I3 )" ) status
       WRITE( rfiledevice, 2000 )                                              &
         pname, LLS_inform%obj, SQRT( DOT_PRODUCT(  prob%X, prob%X ) ),        &
         LLS_control%preconditioner, LLS_inform%time%total, LLS_inform%cg_iter
      END IF
      IF ( is_specfile ) CLOSE( input_specfile )
      CALL CUTEST_cterminate( cutest_status )
      RETURN

  910 CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2000 FORMAT( /, ' Problem: ', A10, /,                                         &
                 ' Optimal value  = ', ES10.4, /,                              &
                 ' Solution norm  = ', ES10.4, /,                              &
                 ' Preconditioner = ', I0, /,                                  &
                 ' Total time     = ', F0.2, /,                                &
                 ' Iterations     = ', I0 )
 2010 FORMAT( A10, 2( 0P, F8.2 ), I6, 0P, F8.2, I6, 2ES8.1, I3 )
 2020 FORMAT( A10, '       -       -', 2( '     -       -' ), I6 )

 2030 FORMAT( I7, 1X, A10, ES12.4 ) 
 2040 FORMAT( '      . .           ..........' )
 2050 FORMAT( /, ' Solution : ', /, '      # name          value   ' )
 2060 FORMAT( ' Exit status = ', I0 )

 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )

!  End of subroutine USE_LLS

     END SUBROUTINE USE_LLS

!  End of module USELLS_double

   END MODULE GALAHAD_USELLS_double


