! THIS VERSION: GALAHAD 2.5 - 09/02/2013 AT 17:30 GMT

!-*-*-*-*-*-*-*-  G A L A H A D   U S E W A R M   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould and Dominique Orban

!  History -
!   originally released with GALAHAD Version 2.5. March 9th 2012

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_usewarm_double

!    --------------------------------------------------------
!    | CUTEst/AMPL interface to WARMSTART, a test of the    |
!    |  warmstart capabilities of the GALAHAD qp solver DQP |
!    --------------------------------------------------------

!$    USE omp_lib
      USE CUTEST_LQP_double, ONLY: CUTEST_lqp_create
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_QPT_double
      USE GALAHAD_QPD_double, ONLY: QPD_SIF
      USE GALAHAD_RAND_double
      USE GALAHAD_SORT_double, ONLY: SORT_reorder_by_rows
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_SLS_double
      USE GALAHAD_QP_double
      USE GALAHAD_DQP_double
      USE GALAHAD_PRESOLVE_double
      USE GALAHAD_SPECFILE_double 
      USE GALAHAD_STRING_double, ONLY: STRING_upper_word
      USE GALAHAD_COPYRIGHT
      USE GALAHAD_SYMBOLS,                                                     &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          DEBUG                 => GALAHAD_DEBUG,                              &
          GENERAL               => GALAHAD_GENERAL,                            &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS
      USE GALAHAD_SCALE_double
      USE GALAHAD_SPACE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: USE_warm

    CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ W A R M  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_warm( input )

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

      INTEGER :: n, m, ir, ic, iores, smt_stat, i, j, l, change, change_pert
!     INTEGER :: np1, npm
      INTEGER :: status, mfixed, mdegen, nfixed, ndegen, mequal, mredun
      INTEGER :: alloc_stat, cutest_status, A_ne, H_ne, iter
      INTEGER :: new_length, min_length, iter_pert, status_pert
      REAL :: time, timeo, times, timet, timep1, timep2, timep3, timep4
      REAL ( KIND = wp ) :: clock, clocko, clocks, clockt, clockt1, clockt2
      REAL ( KIND = wp ) :: clockt3, clocko1, clocko2, clocko3 
      REAL ( KIND = wp ) :: qfval, stopr, dummy, lambda_lower
      REAL ( KIND = wp ) :: res_c, res_k, max_cs, wnorm, wnorm_old
      LOGICAL :: filexx, printo, printe, printm, printd, is_specfile
!     LOGICAL :: ldummy
      TYPE ( RAND_seed ) :: seed
            
!  Specfile characteristics

      INTEGER, PARAMETER :: input_specfile = 34
      INTEGER, PARAMETER :: lspec = 29
      CHARACTER ( LEN = 16 ) :: specname = 'RUNWARM'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
      CHARACTER ( LEN = 16 ) :: runspec = 'RUNWARM.SPC'

!  The default values for DQP could have been set as:

! BEGIN RUNWARM SPECIFICATIONS (DEFAULT)
!  printout-device                           6
!  print-level                               1
!  write-problem-data                        NO
!  problem-data-file-name                    WARM.data
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
!  solution-file-name                        WARMSOL.d
!  solution-file-device                      62
!  write-result-summary                      NO
!  result-summary-file-name                  WARMRES.d
!  result-summary-file-device                47
!  perturb-hessian-diagonals-by              0.0
!  perturb-bounds-by                         0.0
!  perturb-problem-by                        0.0
!  convexify                                 YES
! END RUNWARM SPECIFICATIONS

!  Default values for specfile-defined parameters

      INTEGER :: out  = 6
      INTEGER :: print_level = 1
      INTEGER :: scale = 0
      INTEGER :: dfiledevice = 26
      INTEGER :: ifiledevice = 51
      INTEGER :: pfiledevice = 50
      INTEGER :: qfiledevice = 58
      INTEGER :: rfiledevice = 47
      INTEGER :: sfiledevice = 62
      LOGICAL :: write_problem_data   = .FALSE.
      LOGICAL :: write_initial_sif    = .FALSE.
      LOGICAL :: write_presolved_sif  = .FALSE.
      LOGICAL :: write_scaled_sif     = .FALSE.
      LOGICAL :: write_solution       = .FALSE.
      LOGICAL :: write_result_summary = .FALSE.
      CHARACTER ( LEN = 30 ) :: dfilename = 'WARM.data'
      CHARACTER ( LEN = 30 ) :: ifilename = 'INITIAL.SIF'
      CHARACTER ( LEN = 30 ) :: pfilename = 'PRESOLVE.SIF'
      CHARACTER ( LEN = 30 ) :: qfilename = 'SCALED.SIF'
      CHARACTER ( LEN = 30 ) :: rfilename = 'WARMRES.d'
      CHARACTER ( LEN = 30 ) :: sfilename = 'WARMSOL.d'
      LOGICAL :: do_presolve = .FALSE.
      LOGICAL :: do_solve = .TRUE.
      LOGICAL :: fulsol = .FALSE. 
      REAL ( KIND = wp ) :: prob_pert = zero
      REAL ( KIND = wp ) :: pert_bnd = zero
      REAL ( KIND = wp ) :: wnorm_stop = 0.0000000000001_wp
      REAL ( KIND = wp ) :: H_pert = zero
      LOGICAL :: convexify = .TRUE.

!  Output file characteristics

      INTEGER, PARAMETER :: io_buffer = 11
      INTEGER :: errout = 6
      CHARACTER ( LEN =  5 ) :: state, solv
      CHARACTER ( LEN = 10 ) :: p_name
      CHARACTER ( LEN = 30 ) :: sls_solv

!  Arrays

      TYPE ( DQP_data_type ) :: data
      TYPE ( DQP_control_type ) :: DQP_control        
      TYPE ( DQP_inform_type ) :: DQP_inform
      TYPE ( QP_data_type ) :: QP_data
      TYPE ( QP_control_type ) :: QP_control        
      TYPE ( QP_inform_type ) :: QP_inform
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

      CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: X_names, C_names
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, AY, HX, D, O
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_status, X_status
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_status_1, X_status_1

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
        spec( 27 )%keyword = 'perturb-problem-by'
        spec( 28 )%keyword = 'printout-device'
        spec( 29 )%keyword = 'print-level'

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
        CALL SPECFILE_assign_real( spec( 27 ), prob_pert, errout )
        CALL SPECFILE_assign_integer( spec( 28 ), out, errout )
        CALL SPECFILE_assign_integer( spec( 29 ), print_level, errout )
      END IF

      printe = out > 0 .AND. print_level >= 0
      printo = out > 0 .AND. print_level > 0
      printm = out > 0 .AND. print_level > 1
      printd = out > 0 .AND. print_level > 2

!  ----------------------- problem set-up -------------------------

      CALL CUTEST_lqp_create( cutest_status, input, io_buffer, out, prob%n,    &
                              prob%m, prob%f, prob%G, prob%X, prob%X_l,        &
                              prob%X_u, prob%Z, prob%Y, prob%C_l, prob%C_u,    &
                              p_name, X_names, C_names,                        &
                              A_ne = prob%A%ne, A_row = prob%A%row,            &
                              A_col = prob%A%col, A_val = prob%A%val,          &
                              H_ne = prob%H%ne, H_row = prob%H%row,            &
                              H_col = prob%H%col, H_val = prob%H%val )
!                             H_col = prob%H%col, H_val = prob%H%val,          &
!                             H_pert = H_pert )
      IF ( cutest_status /= 0 ) GO TO 910

      IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
      CALL SMT_put( prob%H%type, 'COORDINATE', smt_stat )
      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'COORDINATE', smt_stat )
      n = prob%n ; m = prob%m
      prob%H%n = n
      WRITE( out, 2060 ) p_name 

!  convexify?

      lambda_lower = 0.0_wp
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

        IF ( n > sls_inform%rank .OR. sls_inform%negative_eigenvalues > 0 .OR. &
            sls_inform%status == GALAHAD_error_inertia ) THEN           
          ALLOCATE( D( n ), O( n ), STAT = alloc_stat )
          D = 0.0_wp ; O = 0.0_wp       
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( I == j ) THEN
              D( i ) = D( i ) + prob%H%val( l )
            ELSE
              O( i ) = O( i ) + ABS( prob%H%val( l ) )
              O( j ) = O( j ) + ABS( prob%H%val( l ) )
            END IF
          END DO

          new_length = prob%H%ne + n ; min_length = new_length
          CALL SPACE_extend_array( prob%H%row, prob%H%ne, prob%H%ne,           &
                                   new_length, min_length, io_buffer,          &
                                   status, alloc_stat )
          new_length = prob%H%ne + n ; min_length = new_length
          CALL SPACE_extend_array( prob%H%col, prob%H%ne, prob%H%ne,           &
                                   new_length, min_length, io_buffer,          &
                                   status, alloc_stat )
          new_length = prob%H%ne + n ; min_length = new_length
          CALL SPACE_extend_array( prob%H%val, prob%H%ne, prob%H%ne,           &
                                   new_length, min_length, io_buffer,          &
                                   status, alloc_stat )

          IF ( MAXVAL( ABS( O ) ) > zero ) THEN

!  add - the Gershgorin lower bound (plus a tiny bit) to the diagonals of H

            lambda_lower = 0.0_wp
            DO i = 1, n
              lambda_lower = MIN( lambda_lower, D( i ) - O( i ) )
            END DO
            lambda_lower = - ( 1.000001_wp * lambda_lower ) + 0.000001_wp 
            DO i = 1, n
              prob%H%ne = prob%H%ne + 1
              prob%H%row( prob%H%ne ) = i ; prob%H%col( prob%H%ne ) = i
              prob%H%val( prob%H%ne ) = lambda_lower
            END DO

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

!  compute a random vector

            wnorm_old = - 1.0_wp
            DO i = 1, n
              CALL RAND_random_real( seed, .TRUE., D( i ) )
            END DO

!  inverse iteration

            DO iter = 1, 100

!  solve ( H + lambda I ) w = d, overwriting d with the solution

              sls_control%max_iterative_refinements = 1
!             control%acceptable_residual_relative = 0.0_wp
              CALL SLS_solve( prob%H, D, sls_data, sls_control, sls_inform )

!  Normalize w

              wnorm = one / TWO_NORM( D )
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

          prob%H%val( prob%H%ne - n + 1 : prob%H%ne ) = lambda_lower
          DEALLOCATE( D, O, STAT = alloc_stat )

        ELSE IF ( sls_inform%status < 0 ) THEN
          WRITE( 6, '( A, I0 )' )                                              &
               ' Failure of SLS_factorize with status = ', sls_inform%status
          STOP
        END IF
      END IF

!  setup other required scalars and arrays

      A_ne = prob%A%ne ; H_ne = prob%H%ne
      prob%new_problem_structure = .TRUE.
      prob%gradient_kind = - 1
      prob%hessian_kind = - 1

!  ------------------- problem set-up complete ----------------------

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

!  Allocate suitable arrays

      ALLOCATE( prob%C( m ), X_status( n ), prob%X_status( n ),                &
                C_status( m ), prob%C_status( m ),                             &
                X_status_1( n ), C_status_1( m ), STAT = alloc_stat )
      IF ( alloc_stat /= 0 ) THEN
        IF ( printe ) WRITE( out, 2150 ) 'X_status', alloc_stat
        STOP
      END IF

      prob%C = zero
      X_status = 0 ; prob%X_status = ACTIVE
      C_status = 0 ; prob%C_status = ACTIVE

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
          write( out, 2160 ) iores, rfilename
          STOP
        END IF
        WRITE( rfiledevice, 2180 ) p_name
      END IF

!  Set all default values, and override defaults if requested
 
      CALL QP_initialize( QP_data, QP_control, QP_inform )
      IF ( is_specfile )                                                       &
        CALL QP_read_specfile( QP_control, input_specfile )
      CALL DQP_initialize( data, DQP_control, DQP_inform )
      IF ( is_specfile )                                                       &
        CALL DQP_read_specfile( DQP_control, input_specfile )
      IF ( scale /= 0 )                                                        &
        CALL SCALE_read_specfile( SCALE_control, input_specfile )
!     SCALE_control%print_level = DQP_control%print_level

      WRITE( out, 2200 ) n, m, A_ne, H_ne

      IF ( printo ) CALL COPYRIGHT( out, '2013' )

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
          CALL QPD_SIF( prob, qfilename, qfiledevice,                          &
                        DQP_control%infinity, .TRUE. )
!         CALL QPT_write_to_sif( prob, p_name, qfilename, qfiledevice,         &
!                                .FALSE., .FALSE., infinity )
      END IF

!  If the preprocessor is to be used, or the problem to be output,
!  allocate sufficient space

      IF ( write_initial_sif .OR. do_presolve ) THEN

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
          CALL QPD_SIF( prob, ifilename, ifiledevice,                          &
                        DQP_control%infinity, .TRUE. )
!         CALL QPT_write_to_sif( prob, p_name, ifilename, ifiledevice,         &
!                                .FALSE., .FALSE., infinity )
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
        
        A_ne = prob%A%ne ; H_ne = prob%H%ne
        IF ( printo ) WRITE( out, 2300 ) prob%n, prob%m, A_ne, H_ne,           &
           timep2 - timep1, PRE_inform%nbr_transforms

!  If required, write a SIF file containing the presolved problem

        IF ( write_presolved_sif ) THEN
          CALL QPD_SIF( prob, pfilename, pfiledevice,                          &
                        DQP_control%infinity, .TRUE. )
!         CALL QPT_write_to_sif( prob, p_name, pfilename, pfiledevice,         &
!                                .FALSE., .FALSE.,                             &
!                                DQP_control%infinity )
        END IF
      END IF

!  Call the optimizer

      qfval = prob%f 

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
            CALL QPD_SIF( prob, qfilename, qfiledevice,                        &
                          DQP_control%infinity, .TRUE. )
!           CALL QPT_write_to_sif( prob, p_name, qfilename, qfiledevice,       &
!                                  .FALSE., .FALSE., infinity )
        END IF

        CALL CPU_TIME( timeo ) ; CALL CLOCK_time( clocko1 )
  
        IF ( .not. do_presolve ) THEN
          prob%m = m ; prob%n = n
        END IF
  
        stopr  = DQP_control%stop_abs_d

!       prob%m = m
!       prob%n = n
  
!       WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!       WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!       WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z

!       prob%Z = 100.0_wp
  
!  =================
!  solve the problem
!  =================

        solv = ' QP'
        IF ( printo ) WRITE( out, " ( ' ** QP solver used ** ' ) " )
        CALL QP_solve( prob, QP_data, QP_control, QP_inform,                   &
                       C_status, X_status )
        C_status_1 = C_status ; X_status_1 = X_status

!  Print details of the primal and dual variables

        IF ( printm ) THEN
!       IF ( .TRUE. ) THEN
          WRITE( out, 2090 ) 
          DO i = 1, n
            state = ' FREE' 
            IF ( ABS( prob%X  ( i ) - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER'
            IF ( ABS( prob%X  ( i ) - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER'
            IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) <     1.0D-10 )          &
              state = 'FIXED'
            WRITE( out, 2050 ) i, X_names( i ), state, prob%X( i ),            &
                               prob%X_l( i ), prob%X_u( i ), prob%Z( i )
          END DO 

!  Print details of the constraints

          IF ( m > 0 ) THEN 
            WRITE( out, 2040 ) 
            DO i = 1, m
              state = ' FREE' 
              IF ( ABS( prob%C( I )   - prob%C_l( i ) ) < ten * stopr )        &
                state = 'LOWER' 
              IF ( ABS( prob%C( I )   - prob%C_u( i ) ) < ten * stopr )        &
                state = 'UPPER' 
              IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) <       stopr )        &
                state = 'EQUAL' 
              WRITE( out, 2130 ) i, C_names( i ), STATE, prob%C( i ),          &
                                 prob%C_l( i ), prob%C_u( i ), prob%Y( i ) 
            END DO 
          END IF
        END IF

!       WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!       WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!       WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z

        IF ( printo ) WRITE( out, " ( /, ' ** QP solver return, status = ',    &
     &     I0, ' ** ' ) " ) QP_inform%status
        qfval = QP_inform%obj

        status = QP_inform%status
        IF ( QP_control%quadratic_programming_solver == 'qpa' ) THEN
          iter = QP_inform%QPA_inform%iter
        ELSE IF ( QP_control%quadratic_programming_solver == 'qpb' ) THEN
          iter = QP_inform%QPB_inform%iter
        ELSE IF ( QP_control%quadratic_programming_solver == 'cqp' ) THEN
          iter = QP_inform%CQP_inform%iter
        ELSE IF ( QP_control%quadratic_programming_solver == 'dqp' ) THEN
          iter = QP_inform%DQP_inform%iter
        ELSE
          iter = - 1
        END IF
        stopr = DQP_control%stop_abs_d

        CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt1 )
        clockt1 = clockt1 - clocko1
        CALL QP_terminate( QP_data, QP_control, QP_inform )

!  ================================
!  solve the problem again with DQP
!  ================================

!       IF ( .FALSE. ) THEN
        CALL CLOCK_time( clocko2 )
        IF ( .TRUE. ) THEN
          DQP_control%dual_starting_point = 0
          solv = ' DQP'
          IF ( printo ) WRITE( out, " ( ' ** DQP solver warmstart used **' ) " )
          CALL DQP_solve( prob, data, DQP_control, DQP_inform,                 &
                          C_status, X_status )

          IF ( printo ) WRITE( out, " ( /, ' ** DQP solver return, status = ', &
       &     I0, ' ** ' ) " ) DQP_inform%status
          qfval = DQP_inform%obj

!         WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!         WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!         WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z

!  Print details of the primal and dual variables

          IF ( printm ) THEN
!         IF ( .TRUE. ) THEN
            WRITE( out, 2090 ) 
            DO i = 1, n
              state = ' FREE' 
              IF ( ABS( prob%X  ( i ) - prob%X_l( i ) ) < ten * stopr )        &
                state = 'LOWER'
              IF ( ABS( prob%X  ( i ) - prob%X_u( i ) ) < ten * stopr )        &
                state = 'UPPER'
              IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) <     1.0D-10 )        &
                state = 'FIXED'
              WRITE( out, 2050 ) i, X_names( i ), state, prob%X( i ),          &
                                 prob%X_l( i ), prob%X_u( i ), prob%Z( i )
            END DO 

!  Print details of the constraints

            IF ( m > 0 ) THEN 
              WRITE( out, 2040 ) 
              DO i = 1, m
                state = ' FREE' 
                IF ( ABS( prob%C( I )   - prob%C_l( i ) ) < ten * stopr )      &
                  state = 'LOWER' 
                IF ( ABS( prob%C( I )   - prob%C_u( i ) ) < ten * stopr )      &
                  state = 'UPPER' 
                IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) <       stopr )      &
                  state = 'EQUAL' 
                WRITE( out, 2130 ) i, C_names( i ), STATE, prob%C( i ),        &
                                   prob%C_l( i ), prob%C_u( i ), prob%Y( i ) 
              END DO 
            END IF
          END IF

!  print the ststus arrays before and after the solve

          IF ( printd ) THEN
            WRITE( out, "( ' C_status     old' )" ) 
            DO i = 1, m
              WRITE( out,"( 2I7 )" )  C_status( i ), C_status_1( i )
            END DO
            WRITE( out, "( ' X_status     old' )" ) 
            DO i = 1, n
              WRITE( out,"( 2I7 )" )  X_status( i ), X_status_1( i )
            END DO
          END IF

          CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt2 )
          clockt2 = clockt2 - clocko2
          change = COUNT( C_status /= C_status_1 )                             &
                     + COUNT( X_status /= X_status_1 )

          IF ( printo ) WRITE( out, " ( /, ' -- change in active set = ',      &
         &  I0, / ) " ) change

          C_status = C_status_1
          X_status = X_status_1

!  Deallocate arrays from the minimization
  
          status = DQP_inform%status
          iter = DQP_inform%iter
          stopr = DQP_control%stop_abs_d

!  ============================================
!  Perturb the problem and solve again with DQP
!  ============================================

          prob%g = prob%g + prob_pert
          prob%A%val = prob%A%val + prob_pert

          CALL CLOCK_time( clocko3 )

          DQP_control%dual_starting_point = 0
          solv = ' DQP'
          IF ( printo ) WRITE( out, " ( ' ** DQP solver perturbed warmstart',  &
         &   ' used **' ) " )
          CALL DQP_solve( prob, data, DQP_control, DQP_inform,                 &
                          C_status, X_status )

          IF ( printo ) WRITE( out, " ( /, ' ** DQP solver return, status = ', &
       &     I0, ' ** ' ) " ) DQP_inform%status
          qfval = DQP_inform%obj

          CALL CPU_TIME( timet ) ; CALL CLOCK_time( clockt3 )
  
!         WRITE( 6, "( ' x ', /, (5ES12.4) )" ) prob%X
!         WRITE( 6, "( ' y ', /, (5ES12.4) )" ) prob%Y
!         WRITE( 6, "( ' z ', /, (5ES12.4) )" ) prob%Z

!  Print details of the primal and dual variables

          IF ( printm ) THEN
!         IF ( .TRUE. ) THEN
            WRITE( out, 2090 ) 
            DO i = 1, n
              state = ' FREE' 
              IF ( ABS( prob%X  ( i ) - prob%X_l( i ) ) < ten * stopr )        &
                state = 'LOWER'
              IF ( ABS( prob%X  ( i ) - prob%X_u( i ) ) < ten * stopr )        &
                state = 'UPPER'
              IF ( ABS( prob%X_l( i ) - prob%X_u( i ) ) <     1.0D-10 )        &
                state = 'FIXED'
              WRITE( out, 2050 ) i, X_names( i ), state, prob%X( i ),          &
                                 prob%X_l( i ), prob%X_u( i ), prob%Z( i )
            END DO 

!  Print details of the constraints

            IF ( m > 0 ) THEN 
              WRITE( out, 2040 ) 
              DO i = 1, m
                state = ' FREE' 
                IF ( ABS( prob%C( I )   - prob%C_l( i ) ) < ten * stopr )      &
                  state = 'LOWER' 
                IF ( ABS( prob%C( I )   - prob%C_u( i ) ) < ten * stopr )      &
                  state = 'UPPER' 
                IF ( ABS( prob%C_l( i ) - prob%C_u( i ) ) <       stopr )      &
                  state = 'EQUAL' 
                WRITE( out, 2130 ) i, C_names( i ), STATE, prob%C( i ),        &
                                   prob%C_l( i ), prob%C_u( i ), prob%Y( i ) 
              END DO 
            END IF
          END IF

!  print the ststus arrays before and after the solve

          IF ( printd ) THEN
            WRITE( out, "( ' C_status     old' )" ) 
            DO i = 1, m
              WRITE( out,"( 2I7 )" )  C_status( i ), C_status_1( i )
            END DO
            WRITE( out, "( ' X_status     old' )" ) 
            DO i = 1, n
              WRITE( out,"( 2I7 )" )  X_status( i ), X_status_1( i )
            END DO
          END IF

          change_pert = COUNT( C_status /= C_status_1 )                        &
                          + COUNT( X_status /= X_status_1 )
          IF ( printo ) WRITE( out, " ( /, ' -- change in active set = ',      &
         &  I0 ) " ) change_pert

!  Deallocate arrays from the minimization
  
          status_pert = DQP_inform%status
          iter_pert = DQP_inform%iter
          CALL DQP_terminate( data, DQP_control, DQP_inform )
        END IF

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

      ALLOCATE( C( m ), STAT = alloc_stat )
      C = zero
      DO l = 1, prob%A%ne
        i =  prob%A%row( l )
        C( i ) = C( i ) +  prob%A%val( l ) * prob%X( prob%A%col( l ) )
      END DO
      res_c = zero ; max_cs = zero
      DO i = 1, m
        dummy = C( i )
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
      DEALLOCATE( C )

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

      DO l = 1, prob%A%ne
        i = prob%A%row( l ) ; j = prob%A%col( l )
        AY( j ) = AY( j ) - prob%A%val( l ) * prob%Y( i )
      END DO
      DO l = 1, prob%H%ne
        i = prob%H%row( l ) ; j = prob%H%col( l )
        HX( i ) = HX( i ) + prob%H%val( l ) * prob%X( j )
        IF ( j /= i ) HX( j ) = HX( j ) + prob%H%val( l ) * prob%X( i )
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
            WRITE( out, 2050 ) i, X_names( i ), state, prob%X( i ),            &
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
              WRITE( out, 2130 ) i, C_names( i ), STATE, prob%C( i ),          &
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
          END DO 
        END IF 
        WRITE( out, 2100 ) n, nfixed, ndegen 
        IF ( m > 0 ) THEN 
           WRITE( out, 2110 ) m, mequal, mredun
           IF ( m /= mequal ) WRITE( out, 2120 ) m - mequal, mfixed, mdegen
        END IF 
        WRITE( out, 2030 ) qfval, res_c, res_k, max_cs, iter

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

          WRITE( sfiledevice, 2250 ) p_name, solv, qfval
          WRITE( sfiledevice, 2090 ) 

          DO i = 1, n 
            state = ' FREE' 
            IF ( ABS( prob%X( i )   - prob%X_l( i ) ) < ten * stopr )          &
              state = 'LOWER' 
            IF ( ABS( prob%X( i )   - prob%X_u( i ) ) < ten * stopr )          &
              state = 'UPPER' 
            IF ( ABS( prob%X_l( I ) - prob%X_u( I ) ) < stopr )                &
              state = 'FIXED' 
            WRITE( sfiledevice, 2050 ) i, X_names( i ), STATE, prob%X( i ),    &
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
              WRITE( sfiledevice, 2130 ) i, C_names( i ), STATE, prob%C( i ),  &
                prob%C_l( i ), prob%C_u( i ), prob%Y( i )   
            END DO 
          END IF 
  
          WRITE( sfiledevice, 2030 ) qfval, res_c, res_k, max_cs, iter
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
      clocks = clocks - clock ; clockt3 = clockt3 - clocko3
      WRITE( out, "( /, ' Total CPU, clock times = ', F8.3, ', ', F8.3 )" )    &
        times + timet, clocks + clockt1 + clockt2 + clockt3 
      WRITE( out, "( ' number of threads = ', I0 )" ) DQP_inform%threads
      WRITE( out, " ( /, ' -- changes in active set = ', I0, 1X, I0 ) " )      &
        change, change_pert
      WRITE( out, "( ' perturbation = ', ES12.4 )" ) prob_pert
      WRITE( out, 2070 ) p_name 

!  Compare the variants used so far

      WRITE( out, 2080 ) p_name, lambda_lower, qfval, QP_inform%status,        &
        clockt1, status, change, clockt2, status_pert, change_pert, clockt3

      IF ( write_result_summary ) THEN
        BACKSPACE( rfiledevice )
!       WRITE( rfiledevice, 2190 )                                             &
!          p_name, n, m, iter, qfval, status, clockt
          WRITE( rfiledevice, 2080 ) p_name, lambda_lower, qfval,              &
            QP_inform%status, clockt1,                                         &
           status, change, clockt2, status_pert, change_pert, clockt3

!        IF ( status >= 0 ) THEN
!          WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )  &
!            p_name, qfval, res_c, res_k, max_cs, iter, clockt, status
!        ELSE
!          WRITE( rfiledevice, "( A10, ES16.8, 3ES9.1, bn, I9, F12.2, I6 )" )  &
!            p_name, qfval, res_c, res_k, max_cs, - iter, - clockt, status
        END IF
!      END IF

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
      DEALLOCATE( X_names, C_names )
      IF ( is_specfile ) CLOSE( input_specfile )
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
 2030 FORMAT( /, ' Final objective function value  ', ES22.14, /,              &
                 ' Maximum constraint violation    ', ES22.14, /,              &
                 ' Maximum dual infeasibility      ', ES22.14, /,              &
                 ' Maximum complementary slackness ', ES22.14, //,             &
                 ' Number of DQP iterations = ', I0 )
 2040 FORMAT( /, ' Constraints : ', /, '                             ',        &
                 '        <------ Bounds ------> ', /                          &
                 '      # name       state    value   ',                       &
                 '    Lower       Upper     Multiplier ' ) 
 2050 FORMAT( I7, 1X, A10, A6, 4ES12.4 ) 
 2060 FORMAT( /, ' Problem: ', A )
 2070 FORMAT( /, ' Problem: ', A, //,                                          &
                 '                     objective  <-  CQP ->',                 &
                 ' <----- DQP ---->  <--- DQP_pert --> ', /,                 &
                 ' Problem     H pert    value    st     CPU',                 &
                 2( ' st  change    CPU' ), /,                                 &
                 ' -------     ------  ---------- --     ---',                 &
                 2( ' --  ------    ---' ) )
 2080 FORMAT( A10, ES10.2, ES11.3, I3, 0P, F8.2, 2( I3, I7, F8.2 ) )
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

!  End of subroutine USE_warm

     END SUBROUTINE USE_warm

!  End of module USEDQP_double

   END MODULE GALAHAD_usewarm_double

