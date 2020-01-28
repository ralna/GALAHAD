! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   U S E D Q P   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.5. August 1st 2012

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_USEDQP_double

!    ---------------------------------------------------------------
!    | CUTEst/AMPL interface to DQP, a dual gradient-projection     |
!    |  algorithm for convex quadratic & least-distance programming |
!    ---------------------------------------------------------------

!$    USE omp_lib
      USE CUTEst_interface_double
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_RAND_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SORT_double, ONLY: SORT_reorder_by_rows
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_DQP_double
      USE GALAHAD_SLS_double
      USE GALAHAD_PRESOLVE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS
      USE GALAHAD_SCALE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_DQP

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ D Q P  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_DQP( input )

!  --------------------------------------------------------------------
!
!  Solve the quadratic program from CUTEst
!
!     minimize     1/2 x(T) H x + g(T) x
!
!     subject to     c_l <= A x <= c_u
!                    x_l <=  x <= x_u
!
!  using the GALAHAD package GALAHAD_DQP
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

      INTEGER :: n, m, ir, ic, la, lh, liw, iores, smt_stat
!     INTEGER :: np1, npm
      INTEGER :: i, j, l, neh, nea
      INTEGER :: status, mfixed, mdegen, nfixed, ndegen, mequal, mredun
      INTEGER :: alloc_stat, cutest_status, A_ne, H_ne, iter
      REAL :: time, timeo, times, timet, timep1, timep2, timep3, timep4
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt
      REAL ( KIND = wp ) :: objf, qfval, stopr, dummy, wnorm, wnorm_old
      REAL ( KIND = wp ) :: res_c, res_k, max_cs, max_d, lambda_lower
      LOGICAL :: filexx, printo, printe, is_specfile
!     LOGICAL :: ldummy
      TYPE ( RAND_seed ) :: seed

!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 30
      CHARACTER ( LEN = 6 ) :: specname = 'RUNDQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 10 ) :: runspec = 'RUNDQP.SPC'

!  The default values for DQP could have been set as:

! BEGIN RUNDQP SPECIFICATIONS (DEFAULT)
!  write-problem-data                        NO
!  problem-data-file-name                    DQP.data
!  problem-data-file-device                  26
!  write-initial-sif                         NO
!  initial-sif-file-name                     INITIAL.SIF
!  initial-sif-file-device                   51
!  least-squares-qp                          NO
!  scale-problem                             0
!  pre-solve-problem                         NO
!  write-presolved-sif                       NO
!  presolved-sif-file-name                   PRESOLVE.SIF
!  presolved-sif-file-device                 50
!  write-scaled-sif                          NO
!  scaled-sif-file-name                      SCALED.SIF
!  scaled-sif-file-device                    58
!  solve-problem                             YES
!  print-full-solution                       NO
!  write-solution                            NO
!  solution-file-name                        DQPSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  DQPRES.d
!  result-summary-file-device                47
!  perturb-hessian-diagonals-by              0.0
!  perturb-bounds-by                         0.0
!  convexify                                 NO
!  objective-type                            0
!  save-checkpoint-info                      NO
!  checkpoint-fine-name                      DQP.checkpoints
!  checkpoint-fine-device                    65
! END RUNDQP SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: pfiledevice = 50
      INTEGER :: qfiledevice = 58
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      INTEGER :: cfiledevice = 65
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_presolved_sif  = .FALSE.
      LOGICAL :: write_scaled_sif     = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      LOGICAL :: write_checkpoints = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'DQP.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: pfilename = 'PRESOLVE.SIF'
      CHARACTER ( LEN = 30 ) :: qfilename = 'SCALED.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'DQPRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'DQPSOL.d'
      CHARACTER ( LEN = 30 ) :: cfilename = 'DQP.checkpoints'
      LOGICAL :: do_presolve = .FALSE.
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE.
      INTEGER :: objective_type = 0
      REAL ( KIND = wp ) :: pert_bnd = zero
      REAL ( KIND = wp ) :: H_pert = zero
      REAL ( KIND = wp ) :: wnorm_stop = 0.0000000000001_wp
      LOGICAL :: convexify = .FALSE.

!  Output file characteristics

      INTEGER, PARAMETER :: out  = 6
      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) :: pname
      CHARACTER ( LEN = 30 ) :: sls_solv

!  Arrays

      TYPE ( DQP_data_type ) :: data
      TYPE ( DQP_control_type ) :: DQP_control
      TYPE ( DQP_inform_type ) :: DQP_inform
      TYPE ( QPT_problem_type ) :: prob
      TYPE ( PRESOLVE_control_type ) :: PRE_control
      TYPE ( PRESOLVE_inform_type )  :: PRE_inform
      TYPE ( PRESOLVE_data_type )    :: PRE_data
      TYPE ( SCALE_trans_type ) :: SCALE_trans
      TYPE ( SCALE_data_type ) :: SCALE_data
      TYPE ( SCALE_control_type ) :: SCALE_control
      TYPE ( SCALE_inform_type ) :: SCALE_inform
      TYPE ( sls_data_type ) :: sls_data
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( sls_inform_type ) :: sls_inform

!  Allocatable arrays

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAME, CNAME
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, AY, HX, D, O
      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, C_stat, B_stat

      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )

!  ------------------ Open the specfile for rundqp ----------------

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
        spec( 7 )%keyword = ''
        spec( 8 )%keyword = 'scale-problem'
        spec( 9 )%keyword = 'pre-solve-problem'
        spec( 10 )%keyword = 'write-presolved-sif'
        spec( 11 )%keyword = 'presolved-sif-file-name'
        spec( 12 )%keyword = 'presolved-sif-file-device'
        spec( 13 )%keyword = 'solve-problem'
        spec( 14 )%keyword = 'print-full-solution'
        spec( 15 )%keyword = 'write-solution'
        spec( 16 )%keyword = 'solution-file-name'
        spec( 17 )%keyword = 'solution-file-device'
        spec( 18 )%keyword = 'write-result-summary'
        spec( 19 )%keyword = 'result-summary-file-name'
        spec( 20 )%keyword = 'result-summary-file-device'
        spec( 21 )%keyword = 'perturb-bounds-by'
        spec( 22 )%keyword = 'write-scaled-sif'
        spec( 23 )%keyword = 'scaled-sif-file-name'
        spec( 24 )%keyword = 'scaled-sif-file-device'
        spec( 25 )%keyword = 'perturb-hessian-diagonals-by'
        spec( 26 )%keyword = 'convexify'
        spec( 27 )%keyword = 'objective-type'
        spec( 28 )%keyword = 'save-checkpoint-info'
        spec( 29 )%keyword = 'checkpoint-fine-name'
        spec( 30 )%keyword = 'checkpoint-fine-device'

!   Read the specfile

        CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

        CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
        CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
        CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 4 ), write_initial_sif, errout )
        CALL SPECFILE_assign_string ( spec( 5 ), ifilename, errout )
        CALL SPECFILE_assign_integer( spec( 6 ), ifiledevice, errout )
        CALL SPECFILE_assign_integer( spec( 8 ), scale, errout )
        CALL SPECFILE_assign_logical( spec( 9 ), do_presolve, errout )
        CALL SPECFILE_assign_logical( spec( 10 ), write_presolved_sif, errout )
        CALL SPECFILE_assign_string ( spec( 11 ), pfilename, errout )
        CALL SPECFILE_assign_integer( spec( 12 ), pfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 13 ), do_solve, errout )
        CALL SPECFILE_assign_logical( spec( 14 ), fulsol, errout )
        CALL SPECFILE_assign_logical( spec( 15 ), write_solution, errout )
        CALL SPECFILE_assign_string ( spec( 16 ), sfilename, errout )
        CALL SPECFILE_assign_integer( spec( 17 ), sfiledevice, errout )
        CALL SPECFILE_assign_logical( spec( 18 ), write_result_summary, errout )
        CALL SPECFILE_assign_string ( spec( 19 ), rfilename, errout )
        CALL SPECFILE_assign_integer( spec( 20 ), rfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 21 ), pert_bnd, errout )
        CALL SPECFILE_assign_logical( spec( 22 ), write_scaled_sif, errout )
        CALL SPECFILE_assign_string ( spec( 23 ), qfilename, errout )
        CALL SPECFILE_assign_integer( spec( 24 ), qfiledevice, errout )
        CALL SPECFILE_assign_real( spec( 25 ), H_pert, errout )
        CALL SPECFILE_assign_logical( spec( 26 ), convexify, errout )
        CALL SPECFILE_assign_integer( spec( 27 ), objective_type, errout )
        CALL SPECFILE_assign_logical( spec( 28 ), write_checkpoints, errout )
        CALL SPECFILE_assign_string ( spec( 29 ), cfilename, errout )
        CALL SPECFILE_assign_integer ( spec( 30 ), cfiledevice, errout )
      END IF

!  Set all default values, and override defaults if requested

      CALL DQP_initialize( data, DQP_control, DQP_inform )
      IF ( is_specfile )                                                       &
        CALL DQP_read_specfile( DQP_control, input_specfile )
      IF ( scale /= 0 )                                                        &
        CALL SCALE_read_specfile( SCALE_control, input_specfile )

!  Determine the number of variables and constraints

      CALL CUTEST_cdimen( cutest_status, input, n, m )
      IF ( cutest_status /= 0 ) GO TO 910

!  Allocate suitable arrays

      ALLOCATE( prob%X( n ), prob%X_l( n ), prob%X_u( n ),                     &
                prob%G( n ), VNAME( n ), B_stat( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X', alloc_stat ; STOP
      END IF

      ALLOCATE( prob%C_l( m ), prob%C_u( m ), prob%Y( m ), CNAME( m ),         &
                EQUATN( m ), LINEAR( m ), C_stat( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat ; STOP
      END IF

!  Set up the data structures necessary to hold the group partially
!  separable function.

      CALL CUTEST_csetup( cutest_status, input, out, io_buffer,                &
                          n, m, prob%X, prob%X_l, prob%X_u,                    &
                          prob%Y, prob%C_l, prob%C_u,EQUATN, LINEAR, 0, 0, 0 )
      IF ( cutest_status /= 0 ) GO TO 910
      DEALLOCATE( LINEAR )

!  Allocate derived types

      ALLOCATE( prob%X0( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'X0', alloc_stat
        STOP
      END IF

      ALLOCATE( prob%C( m ), C( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'C', alloc_stat
        STOP
      END IF

!  Determine the names of the problem, variables and constraints.

      CALL CUTEST_cnames( cutest_status, n, m, pname, VNAME, CNAME )
      IF ( cutest_status /= 0 ) GO TO 910
      WRITE( out, 2020 ) pname

!  Set up the initial estimate of the solution and
!  right-hand-side of the Kuhn-Tucker system.

!  Determine the constant terms for the problem functions.

      prob%X( : n ) = MIN( prob%X_u( : n ),                                    &
                           MAX( prob%X_l( : n ), prob%X( : n ) ) )

!  Set X0 to zero to determine the constant terms for the problem functions

      prob%X0 = zero

!  Evaluate the constant terms of the objective (objf) and constraint
!  functions (C)

      CALL CUTEST_cfn( cutest_status, n, m, prob%X0, objf, C( : m ) )
      IF ( cutest_status /= 0 ) GO TO 910
      DO i = 1, m
        IF ( EQUATN( i ) ) THEN
          prob%C_l( i ) = prob%C_l( i ) - C( i )
          prob%C_u( i ) = prob%C_l( i )
        ELSE
          prob%C_l( i ) = prob%C_l( i ) - C( i )
          prob%C_u( i ) = prob%C_u( i ) - C( i )
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
        WRITE( out, 2150 ) 'A', alloc_stat ; STOP
      END IF

!  Evaluate the linear terms of the constraint functions

      CALL CUTEST_csgr( cutest_status, n, m, prob%X0, prob%Y, .FALSE.,         &
                        nea, la, prob%A%val, prob%A%col, prob%A%row )
      IF ( cutest_status /= 0 ) GO TO 910

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

!  The objective specified by the SIF file will be replaced by the
!  least-distance problem 1/2 sum_i=1^n x_i^2

      IF ( objective_type == 1 ) THEN
        prob%gradient_kind = 0
        IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
        CALL SMT_put( prob%H%type, 'IDENTITY', smt_stat )
        H_pert = zero
        WRITE( out, "( /, ' ** objective replaced by 1/2 ||x||^2' )" )

!  The objective specified by the SIF file will be used

      ELSE

!  Determine the number of nonzeros in the Hessian

        CALL CUTEST_cdimsh( cutest_status, lh )
        IF ( cutest_status /= 0 ) GO TO 910
        lh = MAX( lh, 1 )
        IF ( convexify .OR. H_pert > 0.0_wp ) lh = lh + n

!  Allocate arrays to hold the Hessian

        ALLOCATE( prob%H%row( lh ), prob%H%col( lh ), prob%H%val( lh ),        &
                  STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
!         WRITE( out, "( ' nea = ', i8, ' la   = ', i8 )" ) nea, la
          WRITE( out, 2150 ) 'H', alloc_stat
          STOP
        END IF

!  Evaluate the Hessian of the Lagrangian function at the initial point.

        CALL CUTEST_csh( cutest_status, n, m, prob%X, prob%Y,                  &
                         neh, lh, prob%H%val, prob%H%row, prob%H%col )
        IF ( cutest_status /= 0 ) GO TO 910
!        WRITE( out, "( ' nea = ', i8, ' la   = ', i8,                         &
!       &               ' neh  = ', i8, ' lh   = ', i8 )" ) nea, la, neh, lh

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

        IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
        CALL SMT_put( prob%H%type, 'COORDINATE', smt_stat )
        prob%H%ne = H_ne ; prob%H%n = n
      END IF

!  convexify?

      IF ( convexify ) THEN

!  find the leftmost eigenvalue of H by minimizing x^T H x : || x ||_2 = 1

        CALL SLS_initialize( DQP_control%SBLS_control%symmetric_linear_solver, &
                             sls_data, sls_control, sls_inform )
!sls_control%print_level = 2

        CALL SLS_analyse( prob%H, sls_data, sls_control, sls_inform )
        IF ( sls_inform%status < 0 ) THEN
          WRITE( 6, '( A, I0 )' )                                              &
               ' Failure of SLS_analyse with status = ', sls_inform%status
          STOP
        END IF
        CALL SLS_factorize( prob%H, sls_data, sls_control, sls_inform )

!  the Hessian is not positive definite. Compute the Gershgorin lower bound
!  on the leftmost eigenvalue

!write(6,*) n, sls_inform%rank, sls_inform%negative_eigenvalues, &
! sls_inform%status
        IF ( n > sls_inform%rank .OR. sls_inform%negative_eigenvalues > 0 .OR. &
            sls_inform%status == GALAHAD_error_inertia ) THEN
          ALLOCATE( D( n ), O( n ), STAT = alloc_stat )
          D = 0.0_wp ; O = 0.0_wp
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( i == j ) THEN
              D( i ) = D( i ) + prob%H%val( l )
            ELSE
              O( i ) = O( i ) + ABS( prob%H%val( l ) )
              O( j ) = O( j ) + ABS( prob%H%val( l ) )
            END IF
          END DO
!write(6,*) MAXVAL( ABS( D ) ), MAXVAL( ABS( O ) )
          IF ( MAXVAL( ABS( O ) ) > zero ) THEN
            lambda_lower = 0.0_wp
            DO i = 1, n
              lambda_lower = MIN( lambda_lower, D( i ) - O( i ) )
            END DO

!  add - the Gershgorin lower bound (plus a tiny bit) to the diagonals of H

            lambda_lower = - ( 1.000001_wp * lambda_lower ) + 0.000001_wp
!write(6,*)  lambda_lower

            DO i = 1, n
              H_ne = H_ne + 1
              prob%H%row( H_ne ) = i ; prob%H%col( H_ne ) = i
              prob%H%val( H_ne ) = lambda_lower
            END DO
            prob%H%ne = H_ne

!  refactorize H

            CALL SLS_terminate( sls_data, sls_control, sls_inform )
            CALL SLS_initialize(                                               &
              DQP_control%SBLS_control%definite_linear_solver,                 &
              sls_data, sls_control, sls_inform )
            CALL SLS_analyse( prob%H, sls_data, sls_control, sls_inform )
            IF ( sls_inform%status < 0 ) THEN
              WRITE( 6, '( A, I0 )' )                                          &
                   ' Failure of SLS_analyse with status = ', sls_inform%status
              STOP
            END IF
            CALL SLS_factorize( prob%H, sls_data, sls_control, sls_inform )
            IF ( sls_inform%status < 0 ) THEN
              WRITE( 6, '( A, I0 )' )                                          &
                   ' Failure of SLS_factorize with status = ', sls_inform%status
              STOP
            END IF
!write(6,*) SLS_inform%entries, SLS_inform%entries_in_factors,&
!DQP_control%SBLS_control%definite_linear_solver

!  compute a random vector

            wnorm_old = - 1.0_wp
            DO i = 1, n
              CALL RAND_random_real( seed, .TRUE., D( i ) )
            END DO

!  inverse iteration

            DO iter = 1, 100

!  solve ( H + lambda I ) w = d, overwriting d with the solution

              sls_control%max_iterative_refinements = 1
!           control%acceptable_residual_relative = 0.0_wp
!write(6,*) ' d ', TWO_NORM( D )
              CALL SLS_solve( prob%H, D, sls_data, sls_control, sls_inform )
!write(6,*) ' w ', TWO_NORM( D )

!  Normalize w

              wnorm = one / TWO_NORM( D )
!write(6,*) iter, wnorm
              IF ( ABS( wnorm_old - wnorm ) <= wnorm_stop * lambda_lower ) EXIT
              D = D * wnorm
              wnorm_old = wnorm
            END DO

!  compute the leftmost eigenvalue

            lambda_lower = wnorm - lambda_lower

!  perturb it a bit

            lambda_lower = ABS( lambda_lower ) + MAX( H_pert, wnorm_stop )

!  special case for diagonal H

          ELSE
            DO i = 1, n
              H_ne = H_ne + 1
              prob%H%row( H_ne ) = i ; prob%H%col( H_ne ) = i
            END DO
            prob%H%ne = H_ne
            lambda_lower = MAXVAL( ABS( D ) ) + MAX( H_pert, wnorm_stop )
          END IF
          WRITE( out, "( /, ' -- Hessian perturbed by', ES11.4,                &
         &  ' to ensure positive definiteness' )" ) lambda_lower

!  this ensures that the diagonal perturbation to H is large enough

          prob%H%val( H_ne - n + 1 : H_ne ) = lambda_lower
          DEALLOCATE( D, O, STAT = alloc_stat )

        ELSE IF ( sls_inform%status < 0 ) THEN
          WRITE( 6, '( A, I0 )' )                                              &
               ' Failure of SLS_factorize with status = ', sls_inform%status
          STOP
        END IF

!  add ddiagonal perturbations, if any

      ELSE IF ( H_pert > 0.0_wp ) THEN
        DO i = 1, n
          H_ne = H_ne + 1 ; prob%H%val( H_ne ) = H_pert
          prob%H%row( H_ne ) = i ; prob%H%col( H_ne ) = i
        END DO
      END IF

!  Allocate and initialize dual variables.

      ALLOCATE( prob%Z( n ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        WRITE( out, 2150 ) 'Z', alloc_stat
        STOP
      END IF
      prob%Z( : n ) = one
!     prob%Z( : n ) = zero

!   ldummy = .TRUE.
!   IF ( .not. ldummy ) THEN

!     WRITE( out, "( ' maximum element of A = ', ES12.4,                       &
!    &                ' maximum element of H = ', ES12.4 )" )                  &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) ),                                  &
!      MAXVAL( ABS( prob%H%val( : H_ne ) ) )

      liw = MAX( m, n ) + 1
      ALLOCATE( prob%A%ptr( m + 1 ), prob%H%ptr( n + 1 ) )
      ALLOCATE( IW( liw ) )

!     WRITE( 27, "( ( 3( 2I6, ES12.4 ) ) )" )                                  &
!        ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, H_ne )
!     WRITE( 26, "( ' H_row ', /, ( 10I6 ) )" ) prob%H%row( : H_ne )
!     WRITE( 26, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( : H_ne )
!     WRITE( 26, "( ' H_val ', /, ( 5ES12.4 ) )" ) prob%H%val( : H_ne )

!     ALLOCATE( k_val( n_k, n_k ) )
!     OPEN( in, FILE = filename, FORM = 'FORMATTED', STATUS = 'OLD' )
!     REWIND in
!     DO j = 1, n_k
!       DO i = 1, n_k
!          READ( in, "( ES24.16 )" ) k_val( i, j )
!       END DO
!     END DO
!     CLOSE( in )
!     DO l = 1, H_ne
!       i = MOD( prob%H%row( l ), n_k ) ; IF ( i == 0 ) i = n_k
!       j = MOD( prob%H%col( l ), n_k ) ; IF ( j == 0 ) j = n_k
!       IF ( prob%H%row( l ) <= k_k * n_k .AND.                                &
!            prob%H%col( l ) <= k_k * n_k ) THEN
!         IF ( ABS( prob%H%val( l ) - k_val( i, j ) ) > 0.0001 )               &
!           WRITE( 6, "( 2I6, 2ES22.14 )" )                                    &
!             prob%H%row( l ), prob%H%col( l ), prob%H%val( l ),  k_val( i, j )
!          prob%H%val( l ) = k_val( i, j )
!       ELSE
!         IF ( ABS( prob%H%val( l ) + k_val( i, j ) / k_k ) > 0.0001 )         &
!           WRITE( 6, "( 2I6, 2ES22.14 )" ) prob%H%row( l ), prob%H%col( l ),  &
!             prob%H%val( l ), - k_val( i, j ) / k_k
!          prob%H%val( l ) = - k_val( i, j ) / k_k
!       END IF
!     END DO
!     DEALLOCATE( k_val )

!  Transform A to row storage format

      IF ( A_ne /= 0 ) THEN
        CALL SORT_reorder_by_rows( m, n, A_ne, prob%A%row, prob%A%col, A_ne,   &
                                   prob%A%val, prob%A%ptr, m + 1, IW, liw,     &
                                   out, out, i )
      ELSE
        prob%A%ptr = 0
      END IF
      DEALLOCATE( prob%A%row )
      ALLOCATE( prob%A%row( 0 ) )
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )

!  Same for H

      IF ( objective_type /= 1 .AND. objective_type /= 2 ) THEN
        IF ( H_ne /= 0 ) THEN
          CALL SORT_reorder_by_rows( n, n, H_ne, prob%H%row, prob%H%col, H_ne, &
                                     prob%H%val, prob%H%ptr, n + 1, IW, liw,   &
                                     out, out, i )
        ELSE
          prob%H%ptr = 0
        END IF
        DEALLOCATE( prob%H%row )
        ALLOCATE( prob%H%row( 0 ) )
        IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
        CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
      END IF

! try negating the constraints
!prob%A%val( : A_ne ) = - prob%A%val( : A_ne )
!do i = 1, m
! dummy = - prob%C_l( i )
! prob%C_l( i ) = - prob%C_u( i )
! prob%C_u( i )  = dummy
!end do

      DEALLOCATE( IW )

      prob%new_problem_structure = .TRUE.

!  Store the problem dimensions

      prob%n = n ; prob%m = m
      prob%f    = objf

!     WRITE( out, "( ' maximum element of A = ', ES12.4,                       &
!    &                ' maximum element of H = ', ES12.4 )" )                  &
!      MAXVAL( ABS( prob%A%val( : A_ne ) ) ),                                  &
!      MAXVAL( ABS( prob%H%val( : H_ne ) ) )
!   END IF

!    prob%H%val( : H_ne ) = 100000.0 * prob%H%val( : H_ne )
!    prob%g = 100000.0 * prob%g
!    prob%f = 100000.0 * prob%f

!  ------------------- problem set-up complete ----------------------

      CALL CPU_TIME( times ) ; CALL CLOCK_time( clocks )

!  Perturb bounds if required

      IF ( pert_bnd /= zero ) THEN
        DO i = 1, n
          IF (  prob%X_l( i ) /= prob%X_u( i ) ) THEN
            IF ( prob%X_l( i ) > - infinity )                                  &
              prob%X_l( i ) = prob%X_l( i ) - pert_bnd
            IF ( prob%X_u( i ) < infinity )                                    &
              prob%X_u( i ) = prob%X_u( i ) + pert_bnd
          END IF
        END DO

        DO i = 1, m
          IF (  prob%C_l( i ) /= prob%C_u( i ) ) THEN
            IF ( prob%C_l( i ) > - infinity )                                  &
              prob%C_l( i ) = prob%C_l( i ) - pert_bnd
            IF ( prob%C_u( i ) < infinity )                                    &
              prob%C_u( i ) = prob%C_u( i ) + pert_bnd
          END IF
        END DO
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
        WRITE( dfiledevice, "( ' g ', /, ( 5ES12.4 ) )" ) prob%G( : n )
        WRITE( dfiledevice, "( ' x_l ', /, ( 5ES12.4 ) )" ) prob%X_l( : n )
        WRITE( dfiledevice, "( ' x_u ', /, ( 5ES12.4 ) )" ) prob%X_u( : n )
        WRITE( dfiledevice, "( ' c_l ', /, ( 5ES12.4 ) )" ) prob%C_l( : m )
        WRITE( dfiledevice, "( ' c_u ', /, ( 5ES12.4 ) )" ) prob%C_u( : m )
        WRITE( dfiledevice, "( ' A_ptr ', /, ( 10I6 ) )" ) prob%A%ptr( : m + 1 )
        WRITE( dfiledevice, "( ' A_col ', /, ( 10I6 ) )" ) prob%A%col( : A_ne )
        WRITE( dfiledevice, "( ' A_val ', /, ( 5ES12.4 ) )" )                  &
          prob%A%val( : A_ne )
        IF ( objective_type == 1 ) THEN
          WRITE( dfiledevice, "( ' H = I' )" )
        ELSE
          WRITE( dfiledevice, "( ' H_ptr ', /, ( 10I6 ) )" ) prob%H%ptr(: n + 1)
          WRITE( dfiledevice, "( ' H_col ', /, ( 10I6 ) )" ) prob%H%col( :H_ne )
          WRITE( dfiledevice, "( ' H_val ', /, ( 5ES12.4 ) )" )                &
            prob%H%val( : H_ne )
        END IF
        CLOSE( dfiledevice )
      END IF

!write(6,*) ' row 227: ', CNAME( 227 ), prob%A%col( prob%A%ptr( 227 ) : prob%A%ptr( 228 ) - 1 )
!write(6,*) ' val 227: ', CNAME( 227 ), prob%A%val( prob%A%ptr( 227 ) : prob%A%ptr( 228 ) - 1 )

!  If required, write the checkpoint data to a file

      IF ( write_checkpoints ) THEN
        INQUIRE( FILE = cfilename, EXIST = filexx )
        IF ( filexx ) THEN
           OPEN( cfiledevice, FILE = cfilename, FORM = 'FORMATTED',            &
                 STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
        ELSE
           OPEN( cfiledevice, FILE = cfilename, FORM = 'FORMATTED',            &
                 STATUS = 'NEW', IOSTAT = iores )
        END IF
        IF ( iores /= 0 ) THEN
          write( out, 2160 ) iores, cfilename
          STOP
        END IF
        WRITE( cfiledevice, 2180 ) pname
!WRITE( rfiledevice, "(A10, I8, 1X, I8 )" ) pname, n, m
!CLOSE( rfiledevice )
!STOP
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
        WRITE( rfiledevice, 2180 ) pname
!WRITE( rfiledevice, "(A10, I8, 1X, I8 )" ) pname, n, m
!CLOSE( rfiledevice )
!STOP
      END IF

!     SCALE_control%print_level = DQP_control%print_level

      printo = out > 0 .AND. DQP_control%print_level > 0
      printe = out > 0 .AND. DQP_control%print_level >= 0
      WRITE( out, 2200 ) n, m, A_ne, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2010' )

      C_stat = 0 ; B_stat = 0 ; prob%C = zero

!  If required, scale the problem

      IF ( scale < 0 ) THEN
        CALL SCALE_get( prob, - scale, SCALE_trans, SCALE_data,                &
                        SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
        CALL SCALE_apply( prob, SCALE_trans, SCALE_data,                       &
                          SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
        IF ( write_scaled_sif )                                                &
          CALL QPT_write_to_sif( prob, pname, qfilename, qfiledevice,          &
                                 .FALSE., .FALSE., infinity )
      END IF

!  If the preprocessor is to be used, or the problem to be output,
!  allocate sufficient space

      IF ( write_initial_sif .OR. do_presolve ) THEN

        ALLOCATE( prob%X_status( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%X_status = ACTIVE

        ALLOCATE( prob%C_status( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
          STOP
        END IF
        prob%C_status = ACTIVE

        ALLOCATE( prob%Z_l( n ), prob%Z_u( n ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'Z_lu', alloc_stat
          STOP
        END IF
        prob%Z_l( : n ) = - infinity
        prob%Z_u( : n ) =   infinity

        ALLOCATE( prob%Y_l( m ), prob%Y_u( m ), STAT = alloc_stat )
        IF ( alloc_stat /= 0 ) THEN
          IF ( printe ) WRITE( out, 2150 ) 'C_lu', alloc_stat
          STOP
        END IF
        prob%Y_l( : m ) = - infinity
        prob%Y_u( : m ) =   infinity

!  Writes the initial SIF file, if needed

        IF ( write_initial_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, ifilename, ifiledevice,          &
                                 .FALSE., .FALSE., infinity )
          IF ( .NOT. ( do_presolve .OR. do_solve ) ) STOP
        END IF
      END IF

!  Presolve

      IF ( do_presolve ) THEN

        CALL CPU_TIME( timep1 )

!       set the control variables

        CALL PRESOLVE_initialize( PRE_control, PRE_inform, PRE_data )
        IF ( is_specfile )                                                     &
          CALL PRESOLVE_read_specfile( input_specfile, PRE_control, PRE_inform )

        IF ( PRE_inform%status /= 0 ) STOP

!       Overide some defaults

        PRE_control%infinity = DQP_control%infinity
        PRE_control%c_accuracy = ten * DQP_control%stop_abs_p
        PRE_control%z_accuracy = ten * DQP_control%stop_abs_d

!  Call the presolver

        CALL PRESOLVE_apply( prob, PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from PRESOLVE (exitc =', I6, ')' )" ) &
            PRE_inform%status
          STOP
        END IF

        CALL CPU_TIME( timep2 )

        A_ne = MAX( 0, prob%A%ptr( prob%m + 1 ) - 1 )
        H_ne = MAX( 0, prob%H%ptr( prob%n + 1 ) - 1 )
        IF ( printo ) WRITE( out, 2300 ) prob%n, prob%m, A_ne, H_ne,           &
           timep2 - timep1, PRE_inform%nbr_transforms

!  If required, write a SIF file containing the presolved problem

        IF ( write_presolved_sif ) THEN
          CALL QPT_write_to_sif( prob, pname, pfilename, pfiledevice,          &
                                 .FALSE., .FALSE.,                             &
                                 DQP_control%infinity )
        END IF
      END IF

!  Call the optimizer

      qfval = objf

      IF ( do_solve .AND. prob%n > 0 ) THEN

!  If required, scale the problem

        IF ( scale > 0 ) THEN
          CALL SCALE_get( prob, scale, SCALE_trans, SCALE_data,                &
                          SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF
          CALL SCALE_apply( prob, SCALE_trans, SCALE_data,                     &
                            SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF

          IF ( write_scaled_sif )                                              &
            CALL QPT_write_to_sif( prob, pname, qfilename, qfiledevice,        &
                                   .FALSE., .FALSE., infinity )
        END IF

        CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko )

        IF ( .not. do_presolve ) THEN
          prob%m = m ; prob%n = n
        END IF

        DEALLOCATE( prob%X0 )

!       prob%m = m
!       prob%n = n

!       WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!       WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!       WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z

!       prob%Z = 100.0_wp

!  =================
!  solve the problem
!  =================

        solv = ' DQP'
        IF ( printo ) WRITE( out, " ( ' ** DQP solver used ** ' ) " )
        CALL DQP_solve( prob, data, DQP_control, DQP_inform, C_stat, B_stat )

        IF ( printo ) WRITE( out, " ( /, ' ** DQP solver used ** ' ) " )
        qfval = DQP_inform%obj

        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt )

!  Deallocate arrays from the minimization

        status = DQP_inform%status
        iter = DQP_inform%iter
        stopr = DQP_control%stop_abs_d
        CALL DQP_terminate( data, DQP_control, DQP_inform )

!  If the problem was scaled, unscale it

        IF ( scale > 0 ) THEN
          CALL SCALE_recover( prob, SCALE_trans, SCALE_data,                   &
                              SCALE_control, SCALE_inform )
          IF ( SCALE_inform%status < 0 ) THEN
            WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" ) &
              SCALE_inform%status
            STOP
          END IF
        END IF
      ELSE
        timeo  = 0.0
        timet  = 0.0
        iter  = 0
        solv   = ' NONE'
        status = 0
        stopr  = DQP_control%stop_abs_d
        qfval  = prob%f
      END IF

!  Restore from presolve

      IF ( do_presolve ) THEN
        IF ( PRE_control%print_level >= DEBUG )                                &
          CALL QPT_write_problem( out, prob )

        CALL CPU_TIME( timep3 )
        CALL PRESOLVE_restore( prob, PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status /= 0 .AND. printo )                             &
          WRITE( out, " ( /, ' Warning: info%status following',                &
       &  ' PRESOLVE_restore is ', I5, / ) " ) PRE_inform%status
!       IF ( PRE_inform%status /= 0 ) STOP
        qfval = prob%q
        CALL PRESOLVE_terminate( PRE_control, PRE_inform, PRE_data )
        IF ( PRE_inform%status /= 0 .AND. printo )                             &
          WRITE( out, " ( /, ' Warning: info%status following',                &
       &    ' PRESOLVE_terminate is ', I5, / ) " ) PRE_inform%status
!       IF ( PRE_inform%status /= 0 ) STOP
        IF ( .NOT. do_solve ) STOP
        CALL CPU_TIME( timep4 )
        IF ( printo ) WRITE( out, 2210 )                                       &
          timep4 - timep3, timep2 - timep1 + timep4 - timep3
      ELSE
        PRE_control%print_level = TRACE
      END IF

!  If the problem was scaled, unscale it

      IF ( scale < 0 ) THEN
        CALL SCALE_recover( prob, SCALE_trans, SCALE_data,                     &
                            SCALE_control, SCALE_inform )
        IF ( SCALE_inform%status < 0 ) THEN
          WRITE( out, "( '  ERROR return from SCALE (status =', I0, ')' )" )   &
            SCALE_inform%status
          STOP
        END IF
      END IF

!  Compute maximum contraint residual and complementary slackness

      res_c = zero ; max_cs = zero
      max_d = MAX( MAXVAL( ABS( prob%Y( : m ) ) ),                             &
                   MAXVAL( ABS( prob%Z( : n ) ) ) )
      DO i = 1, m
        dummy = zero
        DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          dummy = dummy +  prob%A%val( j ) * prob%X( prob%A%col( j ) )
        END DO
        res_c = MAX( res_c, MAX( zero, prob%C_l( i ) - dummy,                  &
                                       dummy - prob%C_u( i ) ) )
        IF ( prob%C_l( i ) > - infinity ) THEN
          IF ( prob%C_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ),          &
                      ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%C_l( i ) - dummy ) * prob%Y( i ) ) )
          END IF
        ELSE IF ( prob%C_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs, ABS( ( prob%C_u( i ) - dummy ) * prob%Y( i ) ) )
        END IF
      END DO

      DO i = 1, n
        dummy = prob%X( i )
        IF ( prob%X_l( i ) > - infinity ) THEN
          IF ( prob%X_u( i ) < infinity ) THEN
            max_cs = MAX( max_cs,                                              &
                 MIN( ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ),          &
                      ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) ) )
          ELSE
            max_cs = MAX( max_cs,                                              &
                          ABS( ( prob%X_l( i ) - dummy ) * prob%Z( i ) ) )
          END IF
        ELSE IF ( prob%X_u( i ) < infinity ) THEN
          max_cs = MAX( max_cs, ABS( ( prob%X_u( i ) - dummy ) * prob%Z( i ) ) )
        END IF
      END DO

!  Compute maximum KKT residual

      ALLOCATE( AY( n ), HX( n ), STAT = alloc_stat )
      AY = zero ; HX = prob%G( : n )
!     prob%G( : n ) = prob%G( : n ) - prob%Z( : n )
      DO i = 1, m
        DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          j = prob%A%col( l )
!         prob%G( j ) = prob%G( j ) - prob%A%val( l ) * prob%Y( i )
          AY( j ) = AY( j ) - prob%A%val( l ) * prob%Y( i )
!if(j == 10228) WRITE(6,*) i, prob%A%val( l ), prob%Y( i )
        END DO
      END DO
      DO i = 1, n
        DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
          j = prob%H%col( l )
!         prob%G( i ) = prob%G( i ) + prob%H%val( l ) * prob%X( j )
!         IF ( j /= i )                                                        &
!           prob%G( j ) = prob%G( j ) + prob%H%val( l ) * prob%X( i )
          HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
          IF ( j /= i )                                                        &
            HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
        END DO
      END DO
!     DO i = 1, n
!       WRITE(6,"( i6, 4ES12.4 )" ) i, HX( i ), prob%Z( i ), AY( i ),          &
!                                   HX( i ) - prob%Z( i ) + AY( i )
!     END DO
!     WRITE(6,"( ( 5ES12.4 ) ) " ) MAXVAL( ABS( prob%Z ) )
!     WRITE(6,"( ' G ', /, ( 5ES12.4 ) )" ) prob%G( : n )
      res_k = MAXVAL( ABS( HX( : n ) - prob%Z( : n ) + AY( : n ) ) )
      DEALLOCATE( AY, HX )

!  Print details of the solution obtained

      WRITE( out, 2010 ) status
      IF ( status == GALAHAD_ok .OR.                                           &
           status == GALAHAD_error_cpu_limit .OR.                              &
           status == GALAHAD_error_max_iterations  .OR.                        &
           status == GALAHAD_error_tiny_step .OR.                              &
           status == GALAHAD_error_ill_conditioned ) THEN
        l = 4
        IF ( fulsol ) l = n
        IF ( do_presolve ) THEN
          IF ( PRE_control%print_level >= DEBUG ) l = n
        END IF

!  Print details of the primal and dual variables

        WRITE( out, 2090 )
        DO j = 1, 2
          IF ( j == 1 ) THEN
            ir = 1 ; ic = MIN( l, n )
          ELSE
            IF ( ic < n - l ) WRITE( out, 2000 )
            ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
          END IF
          DO i = ir, ic
            state = ' FREE'
            IF ( ABS( prob%X  ( i ) - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X  ( i ) - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) <     1.0D-10 )          &
              state = 'FIXED'
            WRITE( out, 2050 ) i, VNAME( i ), state, prob%X( i ),              &
                               prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO
        END DO

!  Compute the number of fixed and degenerate variables.

        nfixed = 0 ; ndegen = 0
        DO i = 1, n
          IF ( ABS( prob%X_u( i ) - prob%X_l( i ) ) < stopr ) THEN
            nfixed = nfixed + 1
            IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1
          ELSE IF ( MIN( ABS( prob%X( i ) - prob%X_l( i ) ),                   &
                    ABS( prob%X( i ) - prob%X_u( i ) ) ) <=                    &
                    MAX( ten * stopr, ABS( prob%Z( i ) ) ) ) THEN
            nfixed = nfixed + 1
            IF ( ABS( prob%Z( i ) ) < ten * stopr ) ndegen = ndegen + 1
          END IF
        END DO

!  Print details of the constraints.

        IF ( m > 0 ) THEN
          WRITE( out, 2040 )
          l = 2  ; IF ( fulsol ) l = m
          IF ( do_presolve ) THEN
            IF ( PRE_control%print_level >= DEBUG ) l = m
          END IF
          DO j = 1, 2
            IF ( j == 1 ) THEN
              ir = 1 ; ic = MIN( l, m )
            ELSE
              IF ( ic < m - l ) WRITE( out, 2000 )
              ir = MAX( ic + 1, m - ic + 1 ) ; ic = m
            END IF
            DO i = ir, ic
              state = ' FREE'
              IF ( ABS( prob%C( I )   - prob%C_l( i ) ) < ten * stopr )        &
                state = 'LOWER'
              IF ( ABS( prob%C( I )   - prob%C_u( i ) ) < ten * stopr )        &
                state = 'UPPER'
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) <       stopr )        &
                state = 'EQUAL'
              IF ( prob%C( I )   - prob%C_l( i ) <= - ten * stopr )            &
                state = 'VIOL8'
              IF ( prob%C( I )   - prob%C_u( i ) >=   ten * stopr )            &
                state = 'VIOL8'
              WRITE( out, 2130 ) i, CNAME( i ), STATE, prob%C( i ),            &
                                 prob%C_l( i ), prob%C_u( i ), prob%Y( i )
            END DO
          END DO

!  Compute the number of equality, fixed inequality and degenerate constraints

          mequal = 0 ; mfixed = 0 ; mdegen = 0 ; mredun = 0
          DO i = 1, m
           IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr ) THEN
              mequal = mequal + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mredun = mredun + 1
            ELSE IF ( MIN( ABS( prob%C( i ) - prob%C_l( i ) ),                 &
                      ABS( prob%C( i ) - prob%C_u( i ) ) ) <=                  &
                 MAX( ten * stopr, ABS( prob%Y( i ) ) ) ) THEN
              mfixed = mfixed + 1
              IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1
            END IF
!           IF ( ABS( prob%C( i ) - prob%C_l( i ) ) < ten * stopr .OR.         &
!                ABS( prob%C( i ) - prob%C_u( i ) ) < ten * stopr ) THEN
!             IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < ten * stopr ) THEN
!                mequal = mequal + 1
!             ELSE
!                mfixed = mfixed + 1
!             END IF
!             IF ( ABS( prob%Y( i ) ) < stopr ) mdegen = mdegen + 1
!           END IF
          END DO
        END IF
        WRITE( out, 2100 ) n, nfixed, ndegen
        IF ( m > 0 ) THEN
           WRITE( out, 2110 ) m, mequal, mredun
           IF ( m /= mequal ) WRITE( out, 2120 ) m - mequal, mfixed, mdegen
        END IF
        WRITE( out, 2030 ) qfval, max_d, res_c, res_k, max_cs, iter

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

          WRITE( sfiledevice, 2250 ) pname, solv, qfval
          WRITE( sfiledevice, 2090 )

          DO i = 1, n
            state = ' FREE'
            IF ( ABS( prob%X( i )   - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X( i )   - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( I ) - prob%X_u( I ) ) < stopr )                &
              state = 'FIXED'
            WRITE( sfiledevice, 2050 ) i, VNAME( i ), STATE, prob%X( i ),      &
              prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO

          IF ( m > 0 ) THEN
            WRITE( sfiledevice, 2040 )
            DO i = 1, m
              state = ' FREE'
              IF ( ABS( prob%C( I ) - prob%C_l( i ) ) < ten * stopr )          &
                state = 'LOWER'
              IF ( ABS( prob%C( I ) - prob%C_u( i ) ) < ten * stopr )          &
                state = 'UPPER'
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) < stopr )              &
                state = 'EQUAL'
              IF ( prob%C( I )   - prob%C_l( i ) <= - ten * stopr )            &
                state = 'VIOL8'
              IF ( prob%C( I )   - prob%C_u( i ) >=   ten * stopr )            &
                state = 'VIOL8'
              WRITE( sfiledevice, 2130 ) i, CNAME( i ), STATE, prob%C( i ),    &
                prob%C_l( i ), prob%C_u( i ), prob%Y( i )
            END DO
          END IF

          WRITE( sfiledevice, 2030 ) qfval, max_d, res_c, res_k, max_cs, iter
          CLOSE( sfiledevice )
        END IF
     END IF

!write(6,*) ' ma77_info ', DQP_inform%SBLS_inform%SLS_inform%ma77_info%flag
      sls_solv = DQP_control%SBLS_control%symmetric_linear_solver
      CALL STRING_upper_word( sls_solv )
      WRITE( out, "( /, 1X, A, ' symmetric equation solver used' )" )          &
        TRIM( sls_solv )
      WRITE( out, "( ' - typically ', I0, ', ', I0,                            &
    &                ' entries in matrix, factors' )" )                        &
        DQP_inform%SBLS_inform%SLS_inform%entries,                             &
        DQP_inform%SBLS_inform%SLS_inform%entries_in_factors
      sls_solv = DQP_control%definite_linear_solver
      CALL STRING_upper_word( sls_solv )
      WRITE( out, "( 1X, A, ' definite equation solver used' )" )              &
        TRIM( sls_solv )
      WRITE( out, "( ' - typically ', I0, ', ', I0,                            &
    &                ' entries in matrix, factors' )" )                        &
        DQP_inform%SLS_inform%entries,                                         &
        DQP_inform%SLS_inform%entries_in_factors
      WRITE( out, "( ' Analyse, factorize & solve CPU   times =',              &
     &  3( 1X, F8.3 ), /, ' Analyse, factorize & solve clock times =',         &
     &  3( 1X, F8.3 ) )" ) DQP_inform%time%analyse, DQP_inform%time%factorize, &
        DQP_inform%time%solve, DQP_inform%time%clock_analyse,                  &
        DQP_inform%time%clock_factorize, DQP_inform%time%clock_solve

      times = times - time ; timet = timet - timeo
      clocks = clocks - clock ; clockt = clockt - clocko
      WRITE( out, "( /, ' Total CPU, clock times = ', F8.3, ', ', F8.3 )" )    &
        times + timet, clocks + clockt
      WRITE( out, "( ' number of threads = ', I0 )" ) DQP_inform%threads
      WRITE( out, 2070 ) pname

!  Compare the variants used so far

      WRITE( out, 2080 ) solv, iter, qfval, status, clocks,                    &
                         clockt, clocks + clockt

      IF ( write_checkpoints ) THEN
        BACKSPACE( cfiledevice )
        !WRITE( cfiledevice, "( A10, 16( 1X, I0 ) )" )                         &
        !  pname, DQP_inform%checkpoints( 1 : 16 )
        WRITE( cfiledevice, "( A10, 16( 1X, I0 ), 16( 1X, F0.2 ) )" )          &
          pname, DQP_inform%checkpointsIter( 1 : 16 ),                         &
         DQP_inform%checkpointsTime( 1 : 16 )
     END IF

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
!       WRITE( rfiledevice, 2190 )                                             &
!          pname, n, m, iter, qfval, status, clockt
        IF ( status >= 0 ) THEN
          WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )   &
            pname, qfval, res_c, res_k, max_cs, iter, clockt, status
        ELSE
          WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )   &
            pname, qfval, res_c, res_k, max_cs, - iter, - clockt, status
        END IF
      END IF

!  Print the first and last few components of the solution.

!     WRITE( out, 2070 )
!     WRITE( out, 2120 )
!     j = MIN( npm, 12 )
!     RES( : n ) = prob%X( : n ) ; RES( np1 : npm ) = prob%Y( : m )
!     WRITE( out, 2100 ) ( RES( i ), i = 1, j )
!     IF ( j < npm ) THEN
!        IF ( j + 1 < npm - 11 ) WRITE( out, 2110 )
!        WRITE( out, 2100 ) ( RES( i ), i = MAX( j + 1, npm - 11 ), npm )
!     END IF
!     DEALLOCATE( RES )
      DEALLOCATE( VNAME, CNAME, C )
      IF ( is_specfile ) CLOSE( input_specfile )

      CALL CUTEST_cterminate( cutest_status )
      RETURN

 910  CONTINUE
      WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )         &
        cutest_status
      status = - 98
      RETURN

!  Non-executable statements

 2000 FORMAT( '      . .          .....  ..........',                          &
              '  ..........  ..........  .......... ' )
 2010 FORMAT( /,' Stopping with inform%status = ', I0 )
 2020 FORMAT( /, ' Problem: ', A )
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum dual variable           ', ES22.14, /,              &
                 ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, //,             &
                 ' Number of DQP iterations = ', I0 )
 2040 FORMAT( /, ' Constraints : ', /, '                             ',        &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper     Multiplier ' )
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 )
 2070 FORMAT( /, ' Problem: ', A, //,                                          &
                 '                     objective',                             &
                 '          < ------ time ----- > ', /,                        &
                 ' Method  iterations    value  ',                             &
                 '   status setup   solve   total', /,                         &
                 ' ------  ----------   -------   ',                           &
                 ' ------ -----    ----   -----  ' )
 2080 FORMAT( A5, I7, 6X, ES12.4, 1X, I6, 0P, 3F8.2 )
 2090 FORMAT( /, ' Solution : ', /, '                              ',          &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper       Dual ' )
 2100 FORMAT( /, ' Of the ', I0, ' variables, ', I0,                           &
              ' are on bounds & ', I0, ' are dual degenerate' )
 2110 FORMAT( ' Of the ', I0, ' constraints, ', I0,' are equations, & ',       &
              I0, ' are redundant' )
 2120 FORMAT( ' Of the ', I0, ' inequalities, ', I0, ' are on bounds, & ',     &
              I0, ' are degenerate' )
 2130 FORMAT( I7, 1X, A10, A6, 4ES12.4 )
 2150 FORMAT( ' Allocation error, variable ', A8, ' status = ', I0 )
 2160 FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2180 FORMAT( A10 )
!2190 FORMAT( A10, I7, 2I6, ES13.4, I6, 0P, F8.2 )
 2200 FORMAT( /, ' problem dimensions:  n = ', I0, ', m = ', I0,               &
              ', a_ne = ', I0, ', h_ne = ', I0 )
 2300 FORMAT( ' updated dimensions:  n = ', I0, ', m = ', I0,                  &
              ', a_ne = ', I0, ', h_ne = ', I0, /,                             &
              ' preprocessing time = ', F0.2,                                  &
              ', number of transformations = ', I0 )
 2210 FORMAT( ' postprocessing time = ', F0.2,                                 &
              ', processing time = ', F0.2 )
 2250 FORMAT( /, ' Problem:    ', A10, /, ' Solver :   ', A5,                  &
              /, ' Objective:', ES24.16 )

!  End of subroutine USE_DQP

     END SUBROUTINE USE_DQP

!  End of module USEDQP_double

   END MODULE GALAHAD_USEDQP_double


