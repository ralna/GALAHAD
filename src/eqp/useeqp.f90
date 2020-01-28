! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D   U S E E Q P  *-*-*-*-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Ph. L. Toint, for GALAHAD productions
!  Copyright reserved
!  March 25th 2004

    MODULE GALAHAD_USEEQP_double

!  CUTEst/AMPL interface to GALAHAD_EQP, an algorithm for solving
!  equality-constrained quadratic program using a projected conjugate
!  gradient method

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_double
      USE GALAHAD_EQP_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_EQP

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ E Q P  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_EQP( input )

!  --------------------------------------------------------------------
!
!  Solve the equality-constrained quadratic program from CUTEst
!
!     minimize     1/2 x(T) H x + g(T) x
!
!     subject to     A x + c = 0
!
!  using the GALAHAD package GALAHAD_EQP
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

      INTEGER :: n, m, la, lh, iores, i, j, k, l, neh, nea, ir, ic
!     INTEGER :: np1, npm
!     INTEGER :: factorization_integer, factorization_real
      INTEGER :: status, cutest_status, alloc_stat, A_ne, H_ne, smt_stat
      INTEGER :: ntotal, natotal, nhtotal, nb, ni
      REAL ( KIND = wp ) :: objf, h_max
      LOGICAL :: filexx, is_specfile

!  Functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 13
      CHARACTER ( LEN = 16 ) :: specname = 'RUNEQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNEQP.SPC'

!  The default values for EQP could have been set as:

! BEGIN RUNEQP SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    EQP.data
!  problem-data-file-device                  26
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        EQPSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  EQPRES.d
!  result-summary-file-device                47
!  write-oneline-result-summary              NO
!  result-oneline-summary-file-name          EQPRES_1line.d
!  result-oneline-summary-file-device        47
! END RUNEQP SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: rfiledevice = 47
      INTEGER :: lfiledevice = 48
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_1line_summary  = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'EQP.data'
!     CHARACTER ( LEN = 30 ) :: rfilename = 'EQPRES.d'
      CHARACTER ( LEN = 34 ) :: rfilename = '../results/EQP_IMPLICIT_fact.d'
!     CHARACTER ( LEN = 30 ) :: lfilename = 'EQPRES_1line.d'
      CHARACTER ( LEN = 36 ) :: lfilename ='../results/EQP_IMPLICIT_fact_1line.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'EQPSOL.d'
      LOGICAL :: fulsol = .FALSE.
      LOGICAL :: printo = .TRUE.

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN = 10 ) :: pname
      CHARACTER ( LEN = 30 ) :: sls_solv

!  Arrays

      TYPE ( EQP_data_type ) :: data
      TYPE ( EQP_control_type ) :: EQP_control
      TYPE ( EQP_inform_type ) :: EQP_inform
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( X0( n ), prob%X_l( n ), prob%X_u( n ),                         &
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

      nb = COUNT( prob%X_l > - infinity .OR. prob%X_u < infinity )
      ni = COUNT( .NOT. EQUATN )
      ntotal = n + ni

!  Allocate derived types

      ALLOCATE( prob%X( ntotal ), prob%G( ntotal ), STAT = alloc_stat )
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
      WRITE( out, "( /, ' Problem: ', A )" ) pname

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ), MAX( prob%X_l( : n ), X0( : n ) ) )
      prob%X( n + 1 : ntotal ) = zero

!  Set X0 to zero to determine the constant terms for the problem functions

      X0 = zero

!  Evaluate the constant terms of the objective (objf) and constraint
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, X0, objf, prob%C( : m ) )
      IF ( cutest_status /= 0 ) GO TO 910

!  Determine the number of nonzeros in the Jacobian

      CALL CUTEST_cdimsj( cutest_status, la )
      IF ( cutest_status /= 0 ) GO TO 910
      natotal = la + ni
      la = MAX( la, 1 )

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( natotal ), prob%A%col( natotal ),                  &
                prob%A%val( natotal ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, X0, prob%Y, .FALSE., nea, la,     &
                        prob%A%val( : la ), prob%A%col( : la ),                &
                        prob%A%row( : la ) )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( X0 )

!  Exclude zeros; set the linear term for the objective function

      A_ne = 0
      prob%G( : ntotal ) = zero
      DO l = 1, nea
        IF ( prob%A%val( l ) /= zero ) THEN
          IF ( prob%A%row( l ) > 0 ) THEN
            A_ne = A_ne + 1
            prob%A%row( A_ne ) = prob%A%row( l )
            prob%A%col( A_ne ) = prob%A%col( l )
            prob%A%val( A_ne ) = prob%A%val( l )
          ELSE
            prob%G( prob%A%col( l ) ) = prob%A%val( l )
          END IF
        END IF
      END DO

!  Determine the number of nonzeros in the Hessian

      CALL CUTEST_cdimsh( cutest_status, lh )
      IF ( cutest_status /= 0 ) GO TO 910
      nhtotal = lh + nb + ni
      lh = MAX( lh, 1 )

!  Allocate arrays to hold the Hessian

      ALLOCATE( prob%H%row( nhtotal ), prob%H%col( nhtotal ),                  &
                prob%H%val( nhtotal ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
!       WRITE( out, "( ' nea = ', i8, ' la   = ', i8 )" ) nea, la
        WRITE( out, 2150 ) 'H', alloc_stat
        STOP
      END IF

!  Evaluate the Hessian of the Lagrangian function at the initial point.

      CALL CUTEST_csh( cutest_status, n, m, prob%X( : n ), prob%Y, neh, lh,    &
                       prob%H%val( : lh ), prob%H%row( : lh ),                 &
                       prob%H%col( : lh ) )
      IF ( cutest_status /= 0 ) GO TO 910
!      WRITE( out, "( ' nea = ', i8, ' la   = ', i8,                           &
!     &               ' neh  = ', i8, ' lh   = ', i8 )" ) nea, la, neh, lh

!  Remove Hessian out of range

      h_max = one
      H_ne = 0
      DO l = 1, neh
        IF ( prob%H%val( l ) == zero ) CYCLE
        i = prob%H%row( l ) ; j = prob%H%col( l )
        IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) CYCLE
        H_ne = H_ne + 1 ; prob%H%val( H_ne ) = prob%H%val( l )
        h_max = MAX( h_max, ABS( prob%H%val( l ) ) )
        IF ( i >= j ) THEN
          prob%H%row( H_ne ) = i
          prob%H%col( H_ne ) = j
        ELSE
          prob%H%row( H_ne ) = j
          prob%H%col( H_ne ) = i
        END IF
      END DO

!  Add penalty terms for bounded variables

      h_max = ten * h_max
      DO i = 1, n
        IF ( prob%X_l( i ) > - infinity .OR. prob%X_u( i ) < infinity ) THEN
          H_ne = H_ne + 1
          prob%H%row( H_ne ) = i
          prob%H%col( H_ne ) = i
          prob%H%val( H_ne ) = h_max
        END IF
      END DO

!  Add penalty terms for inequality constraints

      ni = 0
      DO i = 1, m
        IF ( .NOT. EQUATN( i ) ) THEN
          ni = ni + 1
          A_ne = A_ne + 1
          prob%A%row( A_ne ) = i
          prob%A%col( A_ne ) = n + ni
          prob%A%val( A_ne ) = - one
          H_ne = H_ne + 1
          prob%H%row( H_ne ) = n + ni
          prob%H%col( H_ne ) = n + ni
          prob%H%val( H_ne ) = h_max
        END IF
      END DO

!   ldummy = .TRUE.
!   IF ( .not. ldummy ) THEN

!     WRITE( out, "( ' maximum element of A = ', ES12.4,                       &
!    &                ' maximum element of H = ', ES12.4 )" )                  &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) ),                                  &
!      MAXVAL( ABS( prob%H%val( : H_ne ) ) )

!  Store the problem dimensions

      prob%n    = ntotal
      prob%m    = m
      prob%A%ne = A_ne
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'COORDINATE', smt_stat )
      prob%H%ne = H_ne
      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'COORDINATE', smt_stat )
      prob%f    = objf

!  Print details

      WRITE( out, "( /, ' m    = ', I10, '  n    = ', I10,                     &
     &               ' (n_slack = ', I7, ')', /,                               &
     &               ' A_ne = ', I10, '  H_ne = ', I10 )" )                    &
                       m, ntotal, ni, A_ne, H_ne
!     WRITE( out, "( ' maximum element of A = ', ES12.4,                       &
!    &                ' maximum element of H = ', ES12.4 )" )                  &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) ),                                  &
!      MAXVAL( ABS( prob%H%val( : H_ne ) ) )
!   END IF

!  ------------------- problem set-up complete ----------------------

!  ------------------ Open the specfile for runeqp ----------------

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
          ntotal, m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : ntotal )
        WRITE( dfiledevice, "( ' c ', /, ( 5ES12.4 ) )" ) prob%C( : m )
        WRITE( dfiledevice, "( ' A_row ', /, ( 10I6 ) )" ) prob%A%row( : A_ne )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )
        WRITE( dfiledevice, "( ' H_row ', /, ( 10I6 ) )" ) prob%H%row( : H_ne )
        WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
        WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                  &
          prob%H%val( : H_ne )

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

      CALL EQP_initialize( data, EQP_control, EQP_inform )
      IF ( is_specfile )                                                       &
        CALL EQP_read_specfile( EQP_control, input_specfile )

      WRITE( out, "( /, ' problem dimensions:  n = ', I7, ' m = ', I7,         &
     &            ' a_ne = ', I9, ' h_ne = ', I9 )" ) ntotal, m, A_ne, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2004' )

!  Call the optimizer

      IF ( printo ) WRITE( out, " ( ' ** EQP solver used ** ' ) " )
      CALL EQP_solve( prob, data, EQP_control, EQP_inform )

      IF ( printo ) WRITE( out, " ( /, ' Exit from EQP solver' ) " )

      status = EQP_inform%status
!     factorization_integer = EQP_inform%factorization_integer
!     factorization_real = EQP_inform%factorization_real
      CALL EQP_terminate( data, EQP_control, EQP_inform )

!  Print details of the solution obtained

      WRITE( out, "( ' Stopping with inform%status = ', I3 )" ) status

      IF ( status == 0 .AND. write_solution ) THEN
        l = 4
        IF ( fulsol ) l = n

!  Print details of the primal and dual variables

        WRITE( out, 2090 )
        DO j = 1, 2
          IF ( j == 1 ) THEN
            ir = 1 ; ic = MIN( l, n )
          ELSE
            IF ( ic < n - l ) WRITE( out, 2060 )
            ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
          END IF
          DO i = ir, ic
            WRITE( out, 2030 ) i, VNAME( i ), prob%X( i )
          END DO
        END DO

!  Print details of the constraints.

        prob%C_l( : prob%m ) = zero
        DO k = 1, prob%A%ne
          i = prob%A%row( k )
          prob%C_l( i ) =                                                      &
            prob%C_l( i ) + prob%A%val( k ) * prob%X( prob%A%col( k ) )
        END DO

        IF ( m > 0 ) THEN
          WRITE( out, 2040 )
          l = 2  ; IF ( fulsol ) l = m
          DO j = 1, 2
            IF ( j == 1 ) THEN
              ir = 1 ; ic = MIN( l, m )
            ELSE
              IF ( ic < m - l ) WRITE( out, 2010 )
              ir = MAX( ic + 1, m - ic + 1 ) ; ic = m
            END IF
            DO i = ir, ic
              WRITE( out, 2050 ) i, CNAME( i ), prob%C_l( i ), prob%Y( i )
            END DO
          END DO
        END IF
      END IF

      DEALLOCATE( VNAME, CNAME )

      WRITE( out, "( /, ' objective function =', ES22.14 )" ) EQP_inform%obj

      sls_solv = EQP_control%SBLS_control%symmetric_linear_solver
      CALL STRING_upper_word( sls_solv )
      WRITE( out, "( /, 1X, A, ' symmetric equation solver used' )" )          &
        TRIM( sls_solv )
      WRITE( out, "( ' Typically ', I0, ', ', I0,                              &
    &                ' entries in matrix, factors' )" )                        &
        EQP_inform%SBLS_inform%SLS_inform%entries,                             &
        EQP_inform%SBLS_inform%SLS_inform%entries_in_factors
      WRITE( out, "( ' Factorize & solve CPU   times =',                       &
     &  2( 1X, F8.3 ), /, ' Factorize & solve clock times =',                  &
     &  2( 1X, F8.3 ) )" ) EQP_inform%time%factorize, EQP_inform%time%solve,   &
        EQP_inform%time%clock_factorize, EQP_inform%time%clock_solve

!$    WRITE( out, "( ' number of threads = ', I0 )" ) OMP_GET_MAX_THREADS( )

      WRITE( out, 2000 ) pname,                                                &
        EQP_inform%SBLS_inform%preconditioner,                                 &
        EQP_inform%SBLS_inform%factorization,                                  &
        EQP_inform%time%clock_total, EQP_inform%time%clock_factorize,          &
!       EQP_inform%cg_iter_inter,                                              &
!       EQP_inform%time%factorize + EQP_inform%time%solve_inter,               &
!       EQP_control%inner_stop_inter,                                          &
        EQP_inform%cg_iter,                                                    &
        EQP_inform%time%clock_factorize + EQP_inform%time%clock_solve,         &
        EQP_control%inner_stop_relative

!  If required, write results to  the appropriate files

     IF ( write_1line_summary ) THEN
       BACKSPACE( lfiledevice )
       IF ( status >= 0 ) THEN
         WRITE( lfiledevice, 2110 )                                            &
           pname, EQP_inform%time%clock_total, EQP_inform%time%clock_factorize,&
!          EQP_inform%cg_iter_inter,                                           &
!          EQP_inform%time%factorize + EQP_inform%time%solve_inter,            &
           EQP_inform%cg_iter,                                                 &
           EQP_inform%time%clock_factorize + EQP_inform%time%clock_solve,      &
           status, EQP_control%inner_stop_inter,                               &
           EQP_control%inner_stop_relative, EQP_control%preconditioner
        ELSE
          WRITE( lfiledevice, 2020 ) pname, status
        END IF
      END IF

     IF ( write_result_summary ) THEN
       WRITE( rfiledevice, "( ' Stopping with inform%status = ', I3 )" ) status
       WRITE( rfiledevice, 2000 ) pname,                                       &
         EQP_inform%SBLS_inform%preconditioner,                                &
         EQP_inform%SBLS_inform%factorization,                                 &
         EQP_inform%time%clock_total, EQP_inform%time%clock_factorize,         &
!        EQP_inform%cg_iter_inter,                                             &
!        EQP_inform%time%factorize + EQP_inform%time%solve_inter,              &
!        EQP_control%inner_stop_inter,                                         &
         EQP_inform%cg_iter,                                                   &
         EQP_inform%time%clock_factorize + EQP_inform%time%clocK_solve,        &
         EQP_control%inner_stop_relative
      END IF
      IF ( is_specfile ) CLOSE( input_specfile )

      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2000 FORMAT( /, ' Problem: ', A, /,                                           &
                 ' Preconditioner type = ', I0,                                &
                 ', factorization type = ', I0, /,                             &
                 ' Total time = ', 0P, F0.2,                                  &
                 ', factorization time = ', 0P, F0.2, /,                       &
                 '              cg its    time   reduction', /,                &
!                ' intermediate ', I6, 0P, F8.2, ES12.4, /,                    &
                 ' total        ', I6, 0P, F8.2, ES12.4 )
 2010 FORMAT( '      . .           ..........  .......... ' )
 2020 FORMAT( A10, '       -       -', 2( '     -       -' ), I6 )
 2030 FORMAT( I7, 1X, A10, ES12.4 )
 2040 FORMAT( /, ' Constraints: ', /,                                          &
                 '      # name          value     Multiplier ' )
 2050 FORMAT( I7, 1X, A10, 2ES12.4 )
 2060 FORMAT( '      . .           .......... ' )
 2090 FORMAT( /, ' Solution: ', /,                                             &
                 '      # name          value   ' )
 2110 FORMAT( A10, 2( 0P, F8.2 ), 1( I6, 0P, F8.2 ), I6, 2ES8.1, I3 )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )

!  End of subroutine USE_EQP

     END SUBROUTINE USE_EQP

!  End of module USEEQP_double

   END MODULE GALAHAD_USEEQP_double


