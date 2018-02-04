! THIS VERSION: GALAHAD 2.5 - 08/02/2013 AT 12:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E S L S   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.4. January 19th 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USESLS_double

!    -------------------------------------------------
!    | CUTEst/AMPL interface to SLS, a method for    |
!    | solving symmetric systems of linear equations |
!    -------------------------------------------------

      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double, only: SORT_reorder_by_rows
      USE GALAHAD_SLS_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING_double, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
            GENERAL => GALAHAD_GENERAL, ALL_ZEROS => GALAHAD_ALL_ZEROS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_SLS

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ S L S  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_SLS( input )

!  --------------------------------------------------------------------
!
!  Solve the linear system from CUTEst
!
!     ( H  A^T ) ( x ) = ( g )
!     ( A   0  ) ( y )   ( c )
!
!  using the symmetric linear solver SLS
!
!  --------------------------------------------------------------------

!  Dummy argument

      INTEGER, INTENT( IN ) :: input

!  Parameters

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
      REAL ( KIND = wp ), PARAMETER :: K22 = ten ** 6

!     INTEGER, PARAMETER :: n_k = 100, k_k = 3, in = 28
!     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: k_val
!     CHARACTER ( len = 10 ) :: filename = 'k.val'

!  Scalars

      INTEGER :: n, m, ir, ic, la, lh, liw, iores, smt_stat
!     INTEGER :: np1, npm
      INTEGER :: i, j, l, neh, nea
      INTEGER :: status, alloc_stat, cutest_status, A_ne, H_ne
      REAL :: time, timeo, times, timet
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt
      REAL ( KIND = wp ) :: objf
      REAL ( KIND = wp ) :: res_c, res_k
      LOGICAL :: filexx, printo, is_specfile

!  Functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 14
      CHARACTER ( LEN = 16 ) :: specname = 'RUNSLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNSLS.SPC'
      CHARACTER ( LEN = 30 ) :: solver = "sils" // REPEAT( ' ', 26 )

!  The default values for SLS could have been set as:

! BEGIN RUNSLS SPECIFICATIONS (DEFAULT)
!  write-problem-data                                NO
!  problem-data-file-name                            SLS.data
!  problem-data-file-device                          26
!  kkt-system                                        YES
!  symmetric-linear-equation-solver                  sils
!  print-full-solution                               NO
!  write-solution                                    NO
!  solution-file-name                                SLSSOL.d
!  solution-file-device                              62
!  write-result-summary                              NO
!  result-summary-file-name                          SLSRES.d
!  result-summary-file-device                        47
!  barrier-perturbation                              1.0
!  solve                                             YES
! END RUNSLS SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: dfiledevice = 26
      INTEGER :: sfiledevice = 62
      INTEGER :: rfiledevice = 47
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      LOGICAL :: kkt_system = .TRUE.
      LOGICAL :: solve = .TRUE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'SLS.data'
      CHARACTER ( LEN = 30 ) :: sfilename = 'SLSSOL.d'
      CHARACTER ( LEN = 30 ) :: rfilename = 'SLSRES.d'
      LOGICAL :: fulsol = .FALSE.
      REAL ( KIND = wp ) :: barrier_pert = 1.0_wp

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN = 10 ) :: pname

!  Arrays

      TYPE ( SMT_type ) :: K
      TYPE ( SLS_data_type ) :: data
      TYPE ( SLS_control_type ) :: SLS_control
      TYPE ( SLS_inform_type ) :: SLS_inform
      TYPE ( QPT_problem_type ) :: prob

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: AY, HX, SOL
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW

!  ------------------ Open the specfile for runSLS ----------------

      INQUIRE( FILE = runspec, EXIST = is_specfile )
      IF ( is_specfile ) THEN
        OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',              &
              STATUS = 'OLD' )

!   Define the keywords

        spec( 1 )%keyword = 'write-problem-data'
        spec( 2 )%keyword = 'problem-data-file-name'
        spec( 3 )%keyword = 'problem-data-file-device'
        spec( 4 )%keyword = 'print-full-solution'
        spec( 5 )%keyword = 'write-solution'
        spec( 6 )%keyword = 'solution-file-name'
        spec( 7 )%keyword = 'solution-file-device'
        spec( 8 )%keyword = 'symmetric-linear-equation-solver'
        spec( 9 )%keyword = 'barrier-perturbation'
        spec( 10 )%keyword = 'kkt-system'
        spec( 11 )%keyword = 'solve'
        spec( 12 )%keyword = 'write-result-summary'
        spec( 13 )%keyword = 'result-summary-file-name'
        spec( 14 )%keyword = 'result-summary-file-device'

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
        CALL SPECFILE_assign_value( spec( 8 ), solver, errout )
        CALL SPECFILE_assign_real( spec( 9 ), barrier_pert, errout )
        CALL SPECFILE_assign_logical( spec( 10 ), kkt_system, errout )
        CALL SPECFILE_assign_logical( spec( 11 ), solve, errout )
        CALL SPECFILE_assign_logical( spec( 12 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 13 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 14 ), rfiledevice, errout )
      END IF

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ),                     &
                prob%G( n ), VNAME( n ), STAT = alloc_stat )
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
                          n, m, prob%X, prob%X_l, prob%X_u,                    &
                          prob%Y, prob%C_l, prob%C_u, EQUATN, LINEAR, 0, 0, 0 )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( LINEAR )

!  Allocate derived types

      ALLOCATE( prob%X0( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'X0', alloc_stat
        STOP
      END IF

      ALLOCATE( prob%C( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'prob%C', alloc_stat
        STOP
      END IF

!  Determine the names of the problem, variables and constraints.

      CALL CUTEST_cnames( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, "( /, ' Problem: ', A10 )" ) pname

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Set X0 to zero to determine the constant terms for the problem functions

      prob%X0 = zero

!  Evaluate the constant terms of the objective (objf) and constraint
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, prob%X0, objf, prob%C( : m ) )
      IF ( cutest_status /= 0 ) GO TO 910
      DO i = 1, m
        IF ( EQUATN( i ) ) THEN
          prob%C_l( i ) = prob%C_l( i ) - prob%C( i )
          prob%C_u( i ) = prob%C_l( i )
        ELSE
          prob%C_l( i ) = prob%C_l( i ) - prob%C( i )
          prob%C_u( i ) = prob%C_u( i ) - prob%C( i )
        END IF
      END DO

!  Determine the number of nonzeros in the Jacobian

      CALL CUTEST_cdimsj( cutest_status, la )
      IF ( cutest_status /= 0 ) GO TO 910
      la = MAX( la, 1 )

!  Allocate arrays to hold the Jacobian

      ALLOCATE( prob%A%row( la ), prob%A%col( la ), prob%A%val( la ),          &
                STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y, .FALSE.,         &
                        nea, la, prob%A%val, prob%A%col, prob%A%row )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( prob%X0 )

!  Exclude zeros; set the linear term for the objective function

      A_ne = 0
      prob%G( : n ) = zero
      prob%gradient_kind = ALL_ZEROS
      DO i = 1, nea
        IF ( prob%A%val( i ) /= zero ) THEN
          IF ( prob%A%row( i ) > 0 ) THEN
            A_ne = A_ne + 1
            prob%A%row( A_ne ) = prob%A%row( i )
            prob%A%col( A_ne ) = prob%A%col( i )
            prob%A%val( A_ne ) = prob%A%val( i )
          ELSE
            prob%G( prob%A%col( i ) ) = prob%A%val( i )
            prob%gradient_kind = GENERAL
          END IF
        END IF
      END DO

!  Determine the number of nonzeros in the Hessian

      CALL CUTEST_cdimsh( cutest_status, lh )
      IF ( cutest_status /= 0 ) GO TO 910
      lh = MAX( lh, 1 )

!  Allocate arrays to hold the Hessian

      ALLOCATE( prob%H%row( lh + n ), prob%H%col( lh + n ),                    &
                prob%H%val( lh + n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'H', alloc_stat
        STOP
      END IF

!  Evaluate the Hessian of the Lagrangian function at the initial point.

      CALL CUTEST_csh( cutest_status, n, m, prob%X, prob%Y,                    &
                       neh, lh, prob%H%val, prob%H%row, prob%H%col )
      IF ( cutest_status /= 0 ) GO TO 910
!      WRITE( out, "( ' nea = ', i8, ' la   = ', i8,                           &
!     &               ' neh  = ', i8, ' lh   = ', i8 )" ) nea, la, neh, lh

!  Remove Hessian out of range

      H_ne = 0
      DO l = 1, neh
        i = prob%H%row( l ) ; j = prob%H%col( l )
        IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) CYCLE
        H_ne = H_ne + 1 ; prob%H%val( H_ne ) = prob%H%val( l )
        IF ( i >= j ) THEN
          prob%H%row( H_ne ) = i
          prob%H%col( H_ne ) = j
        ELSE
          prob%H%row( H_ne ) = j
          prob%H%col( H_ne ) = i
        END IF
      END DO

!  Add barrier terms

      IF ( barrier_pert > zero ) THEN
        DO i = 1, n
          IF ( prob%X_l( i ) > - infinity ) THEN
            H_ne = H_ne + 1
            prob%H%row( H_ne ) = i
            prob%H%col( H_ne ) = i
            IF ( prob%X_u( i ) < infinity ) THEN
              prob%H%val( H_ne ) = two * barrier_pert
            ELSE
              prob%H%val( H_ne ) = barrier_pert
            END IF
          ELSE IF ( prob%X_u( i ) < infinity ) THEN
            H_ne = H_ne + 1
            prob%H%row( H_ne ) = i
            prob%H%col( H_ne ) = i
            prob%H%val( H_ne ) = barrier_pert
          END IF
        END DO
      END IF

!  Allocate and initialize dual variables.

      ALLOCATE( prob%Z( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2050 ) 'Z', alloc_stat
        STOP
      END IF
      prob%Z( : n ) = one

      liw = MAX( m, n ) + 1
      ALLOCATE( IW( liw ) )

!     WRITE( 6, "( ( 3( 2I6, ES12.4 ) ) )" )                                  &
!        ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, H_ne )
!     WRITE( 26, "( ' H_row ', /, ( 10I6 ) )" ) prob%H%row( : H_ne )
!     WRITE( 26, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
!     WRITE( 26, "( ' H_val ', /, ( 5ES12.4 ) )" ) prob%H%val( : H_ne )

!  Deallocate arrays holding matrix row indices

      DEALLOCATE( IW )
      ALLOCATE( SOL( n + m ), STAT = alloc_stat )

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n ; prob%m = m

!  set up the matrix

      CALL SMT_put( K%type, 'COORDINATE', smt_stat )
      IF ( kkt_system ) THEN
!       K%n = n + m ; K%ne = A_ne + H_ne + m
        K%n = n + m ; K%ne = A_ne + H_ne
        ALLOCATE( K%val( K%ne ), K%row( K%ne ), K%col( K%ne ) )

        K%row( : H_ne ) = prob%H%row( : H_ne )
        K%col( : H_ne ) = prob%H%col( : H_ne )
        K%val( : H_ne ) = prob%H%val( : H_ne )
        K%row( H_ne + 1 : A_ne + H_ne ) = prob%A%row( : A_ne ) + n
        K%col( H_ne + 1 : A_ne + H_ne ) = prob%A%col( : A_ne )
        K%val( H_ne + 1 : A_ne + H_ne ) = prob%A%val( : A_ne )

!       DO i = 1, m
!         K%row( A_ne + H_ne + i ) = n + i
!         K%col( A_ne + H_ne + i ) = n + i
!         K%val( A_ne + H_ne + i ) = K22
!       END DO
        write(6,*) ' nnz(A,H) = ', A_ne, H_ne
      ELSE
        K%n = n ; K%ne = H_ne
        ALLOCATE( K%val( K%ne ), K%row( K%ne ), K%col( K%ne ) )

        K%row( : H_ne ) = prob%H%row( : H_ne )
        K%col( : H_ne ) = prob%H%col( : H_ne )
        K%val( : H_ne ) = prob%H%val( : H_ne )
        write(6,*) ' nnz(H) = ', H_ne
      END IF

!     WRITE( 6, "( /, 'K', /, ( 3( 2I6, ES12.4 ) ) )" )                        &
!        ( K%row( i ), K%col( i ), K%val( i ), i = 1, k%ne )

!     WRITE( 96, "( 2I8 )" ) K%n, K%ne
!     WRITE( 96, "( ( 3( 2I6, ES24.16 ) ) )" )                                 &
!        ( K%row( i ), K%col( i ), K%val( i ), i = 1, k%ne )

!  set up the right-hand side

      SOL( : n ) = - prob%G( : n )
      IF ( kkt_system ) SOL( n + 1 : n + m ) = zero

!     WRITE( 6, "( 'rhs', /, ( 6ES12.4 ) )" ) SOL( : n + m )

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )

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
          write( out, 2060 ) iores, dfilename
          STOP
        END IF

        WRITE( dfiledevice, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" )          &
          n, m, prob%f
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : m )
        WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" ) prob%A%ptr( : m + 1 )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )
        WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" ) prob%H%ptr( : n + 1 )
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
          write( out, 2060 ) iores, rfilename
          STOP
        END IF
        WRITE( rfiledevice, "( A10 )" ) pname
      END IF

      CALL SLS_initialize( solver, data, SLS_control, SLS_inform )
      IF ( is_specfile )                                                       &
        CALL SLS_read_specfile( SLS_control, input_specfile )

      printo = out > 0 .AND. SLS_control%print_level > 0
      WRITE( out, "( /, ' problem dimensions:  n = ', I0, ', m = ', I0,        &
     &       ', a_ne = ', I0, ', h_ne = ', I0 )" ) n, m, A_ne, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2011' )

!  Call the solver

      IF ( prob%n > 0 ) THEN
        CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )

!  =================
!  solve the problem
!  =================

        IF ( printo ) WRITE( out, " ( ' ** SLS solver used ** ' ) " )

!  analyse

        CALL SLS_analyse( K, data, SLS_control, SLS_inform )
        WRITE( 6, "( /, ' analyse   time = ', F8.3, ' clock = ', F8.3,         &
       &  ' ststus = ', I0 )" ) SLS_inform%time%analyse,                       &
          SLS_inform%time%clock_analyse, SLS_inform%status
        WRITE( 6, "( ' external  time = ', F8.3, ' clock = ', F8.3 )" )        &
          SLS_inform%time%analyse_external,                                    &
          SLS_inform%time%clock_analyse_external
        WRITE( 6, "( ' K n = ', I0,                                            &
         &  ', nnz(prec,predicted factors) = ', I0, ', ', I0 )" )              &
               K%n, K%ne, SLS_inform%entries_in_factors

!  factorize

        IF ( SLS_inform%status >= 0 ) THEN
          CALL SLS_factorize( K, data, SLS_control, SLS_inform )
          WRITE( 6, "( ' factorize time = ', F8.3, ' clock = ', F8.3,          &
       &    ' ststus = ', I0 )" ) SLS_inform%time%factorize,                   &
            SLS_inform%time%clock_factorize, SLS_inform%status
          WRITE( 6, "( ' external  time = ', F8.3, ' clock = ', F8.3 )" )      &
            SLS_inform%time%factorize_external,                                &
            SLS_inform%time%clock_factorize_external

!  solve

          IF ( SLS_inform%status >= 0 .AND. solve ) THEN
            CALL SLS_solve( K, SOL, data, SLS_control, SLS_inform )
            WRITE( 6, "( ' solve status = ', I0 )" ) SLS_inform%status
            prob%X( : prob%n ) = SOL( : prob%n )
            prob%Y( : prob%m ) = SOL( prob%n + 1 : prob%n + prob%m )
          END IF
        END IF
        IF ( printo ) WRITE( out, " ( /, ' ** SLS solver used ** ' ) " )

        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )
        timet = timet - timeo ; clockt = clockt - clocko

        WRITE( 6, "( ' nullity, # -ve eigenvalues = ', I0, ', ', I0 )" )       &
              K%n - SLS_inform%rank, SLS_inform%negative_eigenvalues

!  Deallocate arrays from the minimization

        status = SLS_inform%status
        CALL SLS_terminate( data, SLS_control, SLS_inform )
      ELSE
        status = 0
      END IF
      WRITE( out, "( /, ' Solver: ', A, ' with ordering = ', I0 )" )           &
        TRIM( solver ), SLS_control%ordering
      WRITE( out, "(  ' Stopping with inform%status = ', I0 )" ) status

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
        IF ( SLS_inform%status >= 0 ) THEN
          WRITE( rfiledevice, "( A10, I8, A10, 2I8, F10.2, I4 )" )             &
            pname, K%n, TRIM( solver ),                                        &
            K%n - SLS_inform%rank, SLS_inform%negative_eigenvalues,            &
            clockt, SLS_inform%status
        ELSE
          WRITE( rfiledevice, "( A10, I8, A10, 2I8, F10.2, I4 )" )             &
            pname, K%n, TRIM( solver ),                                        &
            K%n - SLS_inform%rank, SLS_inform%negative_eigenvalues,            &
            - clockt, SLS_inform%status
        END IF
      END IF
      IF ( .NOT. solve ) RETURN

!  Compute maximum contraint residual

      IF ( status >= 0 ) THEN
        IF ( kkt_system ) THEN
          ALLOCATE( AY( m ), STAT = alloc_stat )
          AY = zero
          DO l = 1, A_ne
            i = prob%A%row( l ) ; j = prob%A%col( l )
            AY( i ) = AY( i ) +  prob%A%val( l ) * prob%X( j )
          END DO
          res_c = MAX( zero, MAXVAL( AY + K22 * prob%Y( : prob%m ) ) )
          DEALLOCATE( AY )

!  Compute maximum KKT residual

          ALLOCATE( AY( n ), HX( n ), STAT = alloc_stat )
          AY = zero ; HX = prob%G( : n )
          DO l = 1, A_ne
            i = prob%A%row( l ) ; j = prob%A%col( l )
            AY( j ) = AY( j ) + prob%A%val( l ) * prob%Y( i )
          END DO
          DO l = 1, H_ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
            IF ( j /= i )                                                      &
              HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
          END DO
          res_k = MAXVAL( ABS( HX( : n ) + AY( : n ) ) )
        ELSE
          ALLOCATE( HX( n ), STAT = alloc_stat )
          HX = prob%G( : n )
          DO l = 1, H_ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
            IF ( j /= i )                                                      &
              HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
          END DO
          res_k = MAXVAL( ABS( HX( : n ) ) )
        END IF

!  Print details of the solution obtained

        IF ( status == GALAHAD_ok .OR.                                         &
             status == GALAHAD_error_cpu_limit .OR.                            &
             status == GALAHAD_error_max_iterations  .OR.                      &
             status == GALAHAD_error_tiny_step .OR.                            &
             status == GALAHAD_error_ill_conditioned ) THEN
          l = 4
          IF ( fulsol ) l = n

!  Print details of the primal and dual variables

          WRITE( out, 2000 )
          DO j = 1, 2
            IF ( j == 1 ) THEN
              ir = 1 ; ic = MIN( l, n )
            ELSE
              IF ( ic < n - l ) WRITE( out, 2010 )
              ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
            END IF
            DO i = ir, ic
              WRITE( out, 2040 ) i, VNAME( i ), prob%X( i )
            END DO
          END DO

!  Print details of the constraints.

          IF ( kkt_system .AND. m > 0 ) THEN
            WRITE( out, 2020 )
            l = 2  ; IF ( fulsol ) l = m
            DO j = 1, 2
              IF ( j == 1 ) THEN
                ir = 1 ; ic = MIN( l, m )
              ELSE
                IF ( ic < m - l ) WRITE( out, 2010 )
                ir = MAX( ic + 1, m - ic + 1 ) ; ic = m
              END IF
              DO i = ir, ic
                WRITE( out, 2040 ) i, CNAME( i ), prob%C( i )
              END DO
            END DO
          END IF
          IF ( kkt_system ) THEN
            WRITE( out, 2030 ) res_c, res_k
          ELSE
            WRITE( out, 2070 ) res_k
          END IF

!  If required, write the solution to a file

          IF ( write_solution ) THEN
            INQUIRE( FILE = sfilename, EXIST = filexx )
            IF ( filexx ) THEN
               OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',        &
                   STATUS = 'OLD', IOSTAT = iores )
            ELSE
               OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',        &
                    STATUS = 'NEW', IOSTAT = iores )
            END IF
            IF ( iores /= 0 ) THEN
              write( out, 2060 ) iores, sfilename
              STOP
            END IF

            WRITE( sfiledevice, "( /, ' Problem:    ', A10 )" ) pname
            WRITE( sfiledevice, 2000 )

            DO i = 1, n
              WRITE( sfiledevice, 2040 ) i, VNAME( i ), prob%X( i )
            END DO

            IF ( kkt_system .AND. m > 0 ) THEN
              WRITE( sfiledevice, 2020 )
              DO i = 1, m
                WRITE( sfiledevice, 2040 ) i, CNAME( i ), prob%C( i )
              END DO
            END IF

            IF ( kkt_system ) THEN
              WRITE( sfiledevice, 2030 ) res_c, res_k
            ELSE
              WRITE( sfiledevice, 2070 ) res_k
            END IF
            CLOSE( sfiledevice )
          END IF
        END IF
      END IF

      times = times - time ; timet = timet - timeo
      clocks = clocks - clock ; clockt = clockt - clocko
      WRITE( out, "( /, ' Total time, clock = ', F0.2, ', ', F0.2 )" )         &
        times + timet, clocks + clockt
!$    WRITE( out, "( ' number of threads = ', I0 )" ) OMP_GET_MAX_THREADS( )
      WRITE( out, "( /, ' Problem: ', A10, //,                                 &
     &                  '          < ------ time ----- > ',                    &
     &                  '  < ----- clock ---- > ', /,                          &
     &                  '   status setup   solve   total',                     &
     &                  '   setup   solve   total', /,                         &
     &                  '   ------ -----    ----   -----',                     &
     &                  '   -----   -----   -----  ' )" ) pname

!  Compare the variants used so far

      WRITE( out, "( 1X, I6, 0P, 6F8.2 )" )                                    &
        status, times, timet, times + timet , clocks, clockt, clocks + clockt

      DEALLOCATE( VNAME, CNAME, K%row, K%col, K%val )
      IF ( is_specfile ) CLOSE( input_specfile )
      CALL CUTEST_cterminate( cutest_status )

      RETURN

 910 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")          &
       cutest_status
     status = - 98
     RETURN

!  Non-executable statements

 2000 FORMAT( /, ' Solution : ', /,                                            &
                 '      # name          value   ' )
 2010 FORMAT( '      . .           ..........' )
 2020 FORMAT( /, ' Constraints : ', /,                                         &
                 '      # name          value   ' )
 2030 FORMAT( /, ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14 )
 2040 FORMAT( I7, 1X, A10, ES12.4 )
 2050 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2060 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2070 FORMAT( /, ' Maximum dual infeasibility      ', ES22.14 )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I6 )

!  End of subroutine USE_SLS

     END SUBROUTINE USE_SLS

!  End of module USESLS_double

   END MODULE GALAHAD_USESLS_double


