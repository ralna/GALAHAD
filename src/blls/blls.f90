! THIS VERSION: GALAHAD 3.3 - 05/04/2021 AT 11:30 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ B L L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 3.3. October 30th 2019

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BLLS_double

!        ----------------------------------------------------------------
!        |                                                              |
!        | Solve the bound-constrained linear-least-squares problem     |
!        |                                                              |
!        |    minimize   1/2 || A x - b ||_2^2 + weight / 2 || x ||^2   |
!        |    subject to     x_l <= x <= x_u                            |
!        |                                                              |
!        | using a preconditioned projected conjugate-gradient approach |
!        |                                                              |
!        ----------------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_STRING
     USE GALAHAD_SPACE_double
     USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                       &
                                    SORT_heapsort_smallest, SORT_partition
     USE GALAHAD_SBLS_double
     USE GALAHAD_NORMS_double
     USE GALAHAD_QPT_double
     USE GALAHAD_QPP_double
     USE GALAHAD_QPD_double, ONLY: QPD_SIF
     USE GALAHAD_USERDATA_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_CONVERT_double, ONLY: CONVERT_control_type,                   &
                                       CONVERT_inform_type,                    &
                                       CONVERT_to_sparse_column_format
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: BLLS_initialize, BLLS_read_specfile, BLLS_solve,                &
               BLLS_terminate,  BLLS_reverse_type, BLLS_data_type,             &
               BLLS_subproblem_data_type,                                      &
               BLLS_exact_arc_search, BLLS_inexact_arc_search,                 &
               GALAHAD_userdata_type, QPT_problem_type,                        &
               SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

     REAL ( KIND = wp ), PARAMETER :: g_zero = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: h_zero = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: epstl2 = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
     REAL ( KIND = wp ), PARAMETER :: alpha_search = one
     REAL ( KIND = wp ), PARAMETER :: beta_search = half
     REAL ( KIND = wp ), PARAMETER :: mu_search = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: fixed_tol = ten ** ( -15 )
     REAL ( KIND = wp ), PARAMETER :: eta = 0.01_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BLLS_control_type

!  unit number for error and warning diagnostics

       INTEGER :: error = 6

!  general output unit number

       INTEGER :: out  = 6

!  the level of output required

       INTEGER :: print_level = 0

!  on which iteration to start printing

       INTEGER :: start_print = - 1

!  on which iteration to stop printing

       INTEGER :: stop_print = - 1

!  how many iterations between printing

       INTEGER :: print_gap = 1

!  how many iterations to perform (-ve reverts to HUGE(1)-1)

       INTEGER :: maxit = 1000

!  cold_start should be set to 0 if a warm start is required (with variables
!   assigned according to X_stat, see below), and to any other value if the
!   values given in prob%X suffice

       INTEGER :: cold_start = 1

!  the preconditioner (scaling) used (0=none,1=diagonal,anything else=user)

       INTEGER :: preconditioner = 1

!  the ratio of how many iterations use CGLS rather than steepest descent

       INTEGER :: ratio_cg_vs_sd = 1

!  the maximum number of per-iteration changes in the working set permitted
!   when allowing CGLS rather than steepest descent

       INTEGER :: change_max = 2

!  how many CG iterations to perform per BLLS iteration (-ve reverts to n+1)

       INTEGER :: cg_maxit = 1000

!  the maximum number of steps allowed in a piecewise arcsearch (-ve=infinite)

       INTEGER :: arcsearch_max_steps = - 1

!  the unit number to write generated SIF file describing the current problem

       INTEGER :: sif_file_device = 52

!  the objective function will be regularized by adding 1/2 weight ||x||^2

       REAL ( KIND = wp ) :: weight = zero

!  any bound larger than infinity in modulus will be regarded as infinite

       REAL ( KIND = wp ) :: infinity = ten ** 19

!  the required accuracy for the dual infeasibility

       REAL ( KIND = wp ) :: stop_d = ten ** ( - 6 )

!  any pair of constraint bounds (x_l,x_u) that are closer than
!   identical_bounds_tol will be reset to the average of their values
!
       REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!  the CG iteration will be stopped as soon as the current norm of the
!   preconditioned gradient is smaller than
!    max( stop_cg_relative * initial preconditioned gradient, stop_cg_absolute )

       REAL ( KIND = wp ) :: stop_cg_relative = ten ** ( - 2 )
       REAL ( KIND = wp ) :: stop_cg_absolute = epsmch

!  the largest permitted arc length during the piecewise line search

       REAL ( KIND = wp ) :: alpha_max = ten ** 20

!  the initial arc length during the inexact piecewise line search

       REAL ( KIND = wp ) :: alpha_initial = one

!  the arc length reduction factor for the inexact piecewise line search

       REAL ( KIND = wp ) :: alpha_reduction = half

!  the required relative reduction during the inexact piecewise line search

       REAL ( KIND = wp ) :: arcsearch_acceptance_tol = ten ** ( - 2 )

!  the stabilisation weight added to the search-direction subproblem

       REAL ( KIND = wp ) :: stabilisation_weight = ten ** ( - 12 )

!  the maximum CPU time allowed (-ve = no limit)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!  direct_subproblem_solve is true if the least-squares subproblem is to be
!   solved using a matrix factorization, and false if conjugate gradients
!   are to be preferred

       LOGICAL :: direct_subproblem_solve = .TRUE.

!  exact_arc_search is true if an exact arc_search is required, and false if an
!   approximation suffices

       LOGICAL :: exact_arc_search = .TRUE.

!  advance is true if an inexact exact arc_search can increase steps as well
!   as decrease them

       LOGICAL :: advance = .TRUE.

!  if space_critical is true, every effort will be made to use as little
!   space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!  if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  if generate_sif_file is true, a SIF file describing the current problem
!  will be generated

       LOGICAL :: generate_sif_file = .FALSE.

!  name (max 30 characters) of generated SIF file containing input problem

       CHARACTER ( LEN = 30 ) :: sif_file_name =                               &
         "BLLSPROB.SIF"  // REPEAT( ' ', 18 )

!  all output lines will be prefixed by a string (max 30 characters)
!    prefix(2:LEN(TRIM(%prefix))-1)
!   where prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'
!
       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SBLS

       TYPE ( SBLS_control_type ) :: SBLS_control

!  control parameters for CONVERT

       TYPE ( CONVERT_control_type ) :: CONVERT_control
     END TYPE BLLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BLLS_time_type

!  total time

       REAL :: total = 0.0

!  time for the analysis phase

       REAL :: analyse = 0.0

!  time for the factorization phase

       REAL :: factorize = 0.0

!  time for the linear solution phase

       REAL :: solve = 0.0
     END TYPE BLLS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BLLS_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)
!   -20  alegedly +ve definite matrix is not

       INTEGER :: status = 1

!  STAT value after allocate failure

       INTEGER :: alloc_status = 0

!  status return from factorization

       INTEGER :: factorization_status = 0

!  number of iterations required

       INTEGER :: iter = - 1

!  number of CG iterations required

       INTEGER :: cg_iter = 0

!  current value of the objective function

       REAL ( KIND = wp ) :: obj = infinity

!  current value of the projected gradient

       REAL ( KIND = wp ) :: norm_pg = infinity

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  times for various stages

       TYPE ( BLLS_time_type ) :: time

!  inform values from SBLS

       TYPE ( SBLS_inform_type ) :: SBLS_inform

!  inform values for CONVERT

       TYPE ( CONVERT_inform_type ) :: CONVERT_inform
     END TYPE BLLS_inform_type

!  - - - - - - - - - - - - - - -
!   arc_search data derived type
!  - - - - - - - - - - - - - - -

     TYPE :: BLLS_subproblem_data_type
       INTEGER :: branch, n_break, n_free, n_fixed, n_a0, preconditioner
       INTEGER :: nz_d_start, nz_d_end, nz_out_end, base_free, step
       REAL ( KIND = wp ) :: f_alpha_dash, f_alpha_dashdash, stop_cg
       REAL ( KIND = wp ) :: rho_alpha, rho_alpha_dash, rho_alpha_dashdash
       REAL ( KIND = wp ) :: phi_alpha_dash, phi_alpha_dashdash
       REAL ( KIND = wp ) :: alpha_i, alpha_next, delta_alpha, alpha, target
       REAL ( KIND = wp ) :: f_s, f_c, f_i, f_l, f_q, gamma, gamma_a, gamma_f
       REAL ( KIND = wp ) :: rho_c, rho_i, rho_l, rho_q, phi_s, phi_i
       REAL ( KIND = wp ) :: mu, mu_a, mu_f
       LOGICAL :: printp, printw, printd, printdd, debug
       LOGICAL :: present_a, present_asprod, reverse_asprod, present_afprod
       LOGICAL :: reverse_afprod, reverse_prec, present_prec, present_dprec
       LOGICAL :: recompute, regularization, preconditioned
       CHARACTER ( LEN = 1 ) :: direction
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: FREE, NZ_d, NZ_out, P_used
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G, P, Q, R, S, U, W
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R_a, R_f, X_a, D_f
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BREAK_points
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_debug, R_debug
     END TYPE BLLS_subproblem_data_type

!  - - - - - - - - - - -
!   reverse derived type
!  - - - - - - - - - - -

     TYPE :: BLLS_reverse_type
       INTEGER :: nz_in_start, nz_in_end, nz_out_end
       INTEGER :: eval_status = GALAHAD_ok
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_in, NZ_out, FIXED
       LOGICAL :: transpose
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V, P
     END TYPE BLLS_reverse_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE :: BLLS_data_type
       INTEGER :: out, error, print_level, start_print, stop_print, print_gap
       INTEGER :: arc_search_status, cgls_status, change_status
       INTEGER :: n_free, branch, cg_iter, preconditioner
       INTEGER :: nz_in_start, nz_in_end, nz_out_end, maxit, cg_maxit
       INTEGER :: segments, max_segments, steps, max_steps, eval_status
       REAL :: time_start
       REAL ( KIND = wp ) :: norm_step, step, stop_cg, old_gnrmsq, pnrmsq
       REAL ( KIND = wp ) :: alpha_0, alpha_max, alpha_new, f_new, phi_new
       REAL ( KIND = wp ) :: weight, stabilisation_weight, regularization_weight
       LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
       LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe
       LOGICAL :: reverse, reverse_prod, explicit_a, use_aprod, header
       LOGICAL :: direct_subproblem_solve, steepest_descent
       CHARACTER ( LEN = 6 ) :: string_cg_iter
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_status, OLD_status
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_in, NZ_out
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_new, G, R, V, P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIAG, S, SBLS_sol
       TYPE ( SMT_type ) :: A, H_sbls, AT_sbls, C_sbls
       TYPE ( BLLS_subproblem_data_type ) :: subproblem_data
       TYPE ( SBLS_data_type ) :: SBLS_data
     END TYPE BLLS_data_type

   CONTAINS

!-*-*-*-*-*-   B L L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE BLLS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  default control data for BLLS. This routine should be called before
!  BLLS_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE ( BLLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( BLLS_control_type ), INTENT( OUT ) :: control
     TYPE ( BLLS_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  initialize control parameters for SBLS (see GALAHAD_SBLS for details)

     CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
     control%SBLS_control%prefix = '" - SBLS:"                    '

!  added here to prevent for compiler bugs

     control%stop_d = epsmch ** 0.33_wp
     control%stop_cg_absolute = SQRT( epsmch )

     RETURN

!  end of BLLS_initialize

     END SUBROUTINE BLLS_initialize

!-*-*-*-   B L L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE BLLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BLLS_initialize could (roughly)
!  have been set as:

! BEGIN BLLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  iterations-between-printing                       1
!  maximum-number-of-iterations                      1000
!  cold-start                                        1
!  preconditioner                                    0
!  ratio-of-cg-iterations-to-steepest-descent        1
!  max-change-to-working-set-for-subspace-solution   2
!  maximum-number-of-cg-iterations-per-iteration     1000
!  maximum-number-of-arcsearch-steps                -1
!  sif-file-device                                   52
!  regularization-weight                             0.0D+0
!  infinity-value                                    1.0D+19
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  identical-bounds-tolerance                        1.0D-15
!  cg-relative-accuracy-required                     0.01
!  cg-absolute-accuracy-required                     1.0D-8
!  maximum-arcsearch-stepsize                       1.0D+20
!  initial-arcsearch-stepsize                       1.0
!  arcsearch-reduction-factor                       5.0D-1
!  arcsearch-acceptance-tolerance                   1.0D-2
!  stabilisation-weight                             0.0
!  maximum-cpu-time-limit                            -1.0
!  direct-subproblem-solve                           F
!  exact-arc-search-used                             T
!  inexact-arc-search-can-advance                    T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  sif-file-name                                     BLLSPROB.SIF
!  output-line-prefix                                ""
! END BLLS SPECIFICATIONS

!  dummy arguments

     TYPE ( BLLS_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  programming: Nick Gould and Ph. Toint, January 2002.

!  local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: start_print = print_level + 1
     INTEGER, PARAMETER :: stop_print = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print + 1
     INTEGER, PARAMETER :: sif_file_device = print_gap + 1
     INTEGER, PARAMETER :: maxit = sif_file_device + 1
     INTEGER, PARAMETER :: cold_start = maxit + 1
     INTEGER, PARAMETER :: preconditioner = maxit + 1
     INTEGER, PARAMETER :: ratio_cg_vs_sd = preconditioner + 1
     INTEGER, PARAMETER :: change_max = ratio_cg_vs_sd + 1
     INTEGER, PARAMETER :: cg_maxit = change_max + 1
     INTEGER, PARAMETER :: arcsearch_max_steps = cg_maxit + 1
     INTEGER, PARAMETER :: weight = arcsearch_max_steps + 1
     INTEGER, PARAMETER :: infinity = weight + 1
     INTEGER, PARAMETER :: stop_d = infinity + 1
     INTEGER, PARAMETER :: identical_bounds_tol = stop_d + 1
     INTEGER, PARAMETER :: stop_cg_relative = identical_bounds_tol + 1
     INTEGER, PARAMETER :: stop_cg_absolute = stop_cg_relative + 1
     INTEGER, PARAMETER :: alpha_max = stop_cg_absolute + 1
     INTEGER, PARAMETER :: alpha_initial = alpha_max + 1
     INTEGER, PARAMETER :: alpha_reduction = alpha_initial + 1
     INTEGER, PARAMETER :: arcsearch_acceptance_tol = alpha_reduction + 1
     INTEGER, PARAMETER :: stabilisation_weight = arcsearch_acceptance_tol + 1
     INTEGER, PARAMETER :: cpu_time_limit = stabilisation_weight + 1
     INTEGER, PARAMETER :: direct_subproblem_solve = cpu_time_limit + 1
     INTEGER, PARAMETER :: exact_arc_search = direct_subproblem_solve + 1
     INTEGER, PARAMETER :: advance = exact_arc_search + 1
     INTEGER, PARAMETER :: space_critical = advance + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: sif_file_name = generate_sif_file + 1
     INTEGER, PARAMETER :: prefix = sif_file_name + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'BLLS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  define the keywords

     spec%keyword = ''

!  integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( cold_start )%keyword = 'cold-start'
     spec( preconditioner )%keyword = 'preconditioner'
     spec( ratio_cg_vs_sd )%keyword =                                          &
       'ratio-of-cg-iterations-to-steepest-descent'
     spec( change_max )%keyword =                                              &
       'max-change-to-working-set-for-subspace-solution'
     spec( cg_maxit )%keyword = 'maximum-number-of-cg-iterations-per-iteration'
     spec( arcsearch_max_steps )%keyword = 'maximum-number-of-arcsearch-steps'
     spec( sif_file_device )%keyword = 'sif-file-device'

!  real key-words

     spec( weight )%keyword = 'regularization-weight'
     spec( infinity )%keyword = 'infinity-value'
     spec( stop_d )%keyword = 'dual-accuracy-required'
     spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
     spec( stop_cg_relative )%keyword = 'cg-relative-accuracy-required'
     spec( alpha_max )%keyword = 'maximum-arcsearch-stepsize'
     spec( alpha_initial )%keyword = 'initial-arcsearch-stepsize'
     spec( alpha_reduction )%keyword = 'arcsearch-reduction-factor'
     spec( arcsearch_acceptance_tol )%keyword =                                &
       'arcsearch-acceptance-tolerance'
     spec( stabilisation_weight )%keyword = 'stabilisation-weight'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'

!  logical key-words

     spec( exact_arc_search )%keyword = 'exact-arc-search-used'
     spec( direct_subproblem_solve )%keyword = 'direct-subproblem-solve'
     spec( advance )%keyword = 'inexact-arc-search-can-advance'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
     spec( generate_sif_file )%keyword = 'generate-sif-file'

!  character key-words

     spec( sif_file_name )%keyword = 'sif-file-name'
     spec( prefix )%keyword = 'output-line-prefix'

!  read the specfile

    IF ( PRESENT( alt_specname ) ) THEN
      CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
    ELSE
      CALL SPECFILE_read( device, specname, spec, lspec, control%error )
    END IF

!  interpret the result

!  set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print ),                           &
                                 control%stop_print,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_gap ),                            &
                                 control%print_gap,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cold_start ),                           &
                                 control%cold_start,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( preconditioner ),                       &
                                 control%preconditioner,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( ratio_cg_vs_sd ),                       &
                                 control%ratio_cg_vs_sd,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( change_max ),                           &
                                 control%change_max,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cg_maxit ),                             &
                                 control%cg_maxit,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( arcsearch_max_steps ),                  &
                                 control%arcsearch_max_steps,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )

!  set real value

     CALL SPECFILE_assign_value( spec( weight ),                               &
                                 control%weight,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_d ),                               &
                                 control%stop_d,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_relative ),                     &
                                 control%stop_cg_relative,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_cg_absolute ),                     &
                                 control%stop_cg_absolute,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alpha_max ),                            &
                                 control%alpha_max,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alpha_initial ),                        &
                                 control%alpha_initial,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alpha_reduction ),                      &
                                 control%alpha_reduction,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( arcsearch_acceptance_tol ),             &
                                 control%arcsearch_acceptance_tol,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stabilisation_weight ),                &
                                 control%stabilisation_weight,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )

!  set logical values

     CALL SPECFILE_assign_value( spec( direct_subproblem_solve ),              &
                                 control%direct_subproblem_solve,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( exact_arc_search ),                     &
                                 control%exact_arc_search,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( advance ),                              &
                                 control%advance,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_sif_file ),                    &
                                 control%generate_sif_file,                    &
                                 control%error )

!  set character value

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  read the specfiles for SBLS

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SBLS_read_specfile( control%SBLS_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-SBLS' )
     ELSE
       CALL SBLS_read_specfile( control%SBLS_control, device )
     END IF

     RETURN

     END SUBROUTINE BLLS_read_specfile

!-*-*-*-*-*-*-*-   B L L S _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE BLLS_solve( prob, X_stat, data, control, inform, userdata,     &
                            reverse, eval_APROD, eval_ASPROD, eval_AFPROD,     &
                            eval_PREC )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the linear least-squares problem
!
!     minimize     q(x) = 1/2 || A x - b ||_2^2 + weight / 2 || x ||^2
!
!     subject to   (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), b is an m-vector,
!  A is an m by n matrix, and any of the bounds (x_l)_i, (x_u)_i may be
!  infinite, using a preconditioned projected CG method.
!
!  The subroutine is particularly appropriate when A is sparse, or if it
!  not availble explicitly (but its action may be found by subroutine call 
!  or reverse communication)
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of residuals, m. RESTRICTION: %m >= 1
!
!   %A is a structure of type SMT_type used to hold A if available).
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       A%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', A%type )
!       A%val( : )   the values of the components of A, stored column by column
!       A%row( : )   the row indices of the components of A
!       A%ptr( : )   pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       A%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', A%type )
!       A%val( : )   the values of the components of A, stored column by column,
!                    with each the entries in each column in order of
!                    increasing row indicies.
!
!    If A is not available explicitly, matrix-vector products must be
!      provided by the user using either reverse communication
!      (see reverse below) or a provided subroutine (see eval_APROD
!       and eval_ASPROD below).
!
!   %B is a REAL array of length %m, which must be set by the user to the
!    value of b, the linear term of the residuals, in the least-squares
!    objective function. The i-th component of B, i = 1, ...., m,
!    should contain the value of b_i.
!
!   %X_l, %X_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %X_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %X_u to a value larger than
!    control%infinity. On exit, %X_l and %X_u will most likely have been
!    reordered.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %C is a REAL array of length %m, which need not be set on input. On
!    successful exit, it will contain the residual vector c(x) = A x - b. The
!    i-th component of C, i = 1, ...., m, will contain the value of c_i(x).
!
!   %Z is a REAL array of length %n, which need not be set on input. On
!    successful exit, it will contain estimates of the values of the dual
!    variables, i.e., Lagrange multipliers corresponding to the simple bound
!    constraints x_l <= x <= x_u.
!
!  X_stat is a INTEGER array of length n, which may be set by the user
!   on entry to BLLS_solve to indicate which of the simple bound constraints
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; X_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   X_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   X_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  data is a structure of type BLLS_data_type which holds private internal data
!
!  control is a structure of type BLLS_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to BLLS_initialize. See BLLS_initialize
!   for details
!
!  inform is a structure of type BLLS_inform_type that provides
!    information on exit from BLLS_solve. The component %status
!    must be set to 1 on initial entry, and on exit has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!     2 The product A * v of the matrix A with a given vector v is required
!       from the user. The vector v will be provided in reverse%V and the
!       required product must be returned in reverse%P. BLLS_solve must then
!       be re-entered with reverse%eval_status set to 0, and any remaining
!       arguments unchanged. Should the user be unable to form the product,
!       this should be flagged by setting reverse%eval_status to a nonzero value
!
!     3 The product A^T * v of the transpose of the matrix A with a given
!       vector v is required from the user. The vector v will be provided in
!       reverse%V and the required product must be returned in reverse%P.
!       BLLS_solve must then be re-entered with reverse%eval_status set to 0,
!       and any remaining arguments unchanged. Should the user be unable to
!       form the product, this should be flagged by setting reverse%eval_status
!       to a nonzero value
!
!     4 The product A * v of the matrix A with a given sparse vector v is
!       required from the user. Only components
!         reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end )
!       of the vector v stored in reverse%V are nonzero. The required product
!       should be returned in reverse%P. BLLS_solve must then be re-entered
!       with all other arguments unchanged. Typically v will be very sparse
!       (i.e., reverse%nz_in_end-reverse%NZ_in_start will be small).
!       reverse%eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     5 The product A * v of the matrix A with a given sparse vector v
!       is required from the user. Only components
!         reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end )
!       of the vector v stored in reverse%V are nonzero. The resulting
!       NONZEROS in the product A * v must be placed in their appropriate
!       comnpinents of reverse%P, while a list of indices of the nonzeos
!       placed in reverse%NZ_out( 1 : reverse%nz_out_end ). BLLS_solve should
!       then be re-entered with all other arguments unchanged. Typically
!       v will be very sparse (i.e., reverse%nz_in_end-reverse%NZ_in_start
!       will be small). Once again reverse%eval_status should be set to zero
!       unless the product cannot be form, in which case a nonzero value
!       should be returned.
!
!     6 Specified components of the product A^T * v of the transpose of the
!       matrix A with a given vector v stored in reverse%V are required from
!       the user. Only components indexed by
!         reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end )
!       of the product should be computed, and these should be recorded in
!         reverse%P( reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end ) )
!       and BLLS_solve then re-entered with all other arguments unchanged.
!       reverse%eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     7 The product P^-1 * v involving the preconditioner P with a specified 
!       vector v is required from the user. Here P should be a symmtric, 
!       postive-definite approximation of A^T A. The vector v will be provided 
!       in reverse%V and the required product must be returned in reverse%P. 
!       BLLS_solve must then be re-entered with reverse%eval_status set to 0, 
!       and any remaining arguments unchanged. Should the user be unable
!       to form the product, this should be flagged by setting 
!       reverse%eval_status to a nonzero value. This return can only happen 
!       when control%preciditioner is not 0 or 1.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%A%type in { 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',
!                         'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS', 'COORDINATE' }
!       (if prob%A is provided) has been violated.
!
!    -4 The bound constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -9 The factorization failed; the return status from the factorization
!       package is given in the component factorization_status.
!
!    -13 The problem is so ill-conditoned that further progress is impossible.
!
!    -16 The step is too small to make further impact.
!
!    -17 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!    -18 Too much CPU time has passed. This may happen if
!        control%cpu_time_limit is too small, but may also be
!        symptomatic of a badly scaled problem.
!
!    -23 an entry from the strict upper triangle of H has been input.
!
!  On exit from BLLS_solve, other components of inform give the
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation
!     factorization_integer = The total integer workspace required for the
!       factorization.
!     factorization_real = The total real workspace required for the
!       factorization.
!     nfacts = The total number of factorizations performed.
!     nmods = The total number of factorizations which were modified to
!       ensure that the matrix was an appropriate preconditioner.
!     factorization_status = the return status from the matrix factorization
!       package.
!     obj = the value of the objective function at the best estimate of the
!       solution determined by BLLS_solve.
!     non_negligible_pivot = the smallest pivot which was not judged to be
!       zero when detecting linearly dependent constraints
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package.
!     time%analyse = the time spent analysing the required matrices prior to
!       factorization.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be used
!   to pass user data to and from the eval_* subroutines (see below)
!   Available coomponents which may be allocated as required are:
!
!    integer is a rank-one allocatable array of type default integer.
!    real is a rank-one allocatable array of type default real
!    complex is a rank-one allocatable array of type default comple.
!    character is a rank-one allocatable array of type default character.
!    logical is a rank-one allocatable array of type default logical.
!    integer_pointer is a rank-one pointer array of type default integer.
!    real_pointer is a rank-one pointer array of type default  real
!    complex_pointer is a rank-one pointer array of type default complex.
!    character_pointer is a rank-one pointer array of type default character.
!    logical_pointer is a rank-one pointer array of type default logical.
!
!  reverse is an OPTIONAL structure of type BLLS_reverse_type which is used to
!   pass intermediate data to and from BLLS_solve. This will only be necessary
!   if reverse-communication is to be used to form matrix-vector products
!   of the form H * v or preconditioning steps of the form P^{-1} * v. If
!   reverse is present (and eval_APROD or eval_PREC is absent), reverse 
!   communication will be used and the user must monitor the value of 
!   inform%status(see above) to await instructions about required 
!   matrix-vector products.
!
!  eval_APROD is an OPTIONAL subroutine which if present must have the
!   arguments given below (see the interface blocks). The sum p + A * v
!   (if transpose is .FALSE.) or p + A^T v (if transpose is .TRUE.)
!   involving the given matrix A and vectors p and v stored in
!   P and V must be returned in P. The status variable should be set
!   to 0 unless the product is impossible in which case status should
!   be set to a nonzero value. If eval_APROD is not present, BLLS_solve
!   will either return to the user each time an evaluation is required
!   (see reverse above) or form the product directly from user-provided %A.
!
!  eval_ASPROD is an OPTIONAL subroutine which if present must have the
!   arguments given below (see the interface blocks). The product A * v of
!   the given matrix A and vector v stored in V must be returned in P; only the
!   components NZ_in( nz_in_start : nz_in_end ) of V are nonzero. If either of
!   the optional argeuments NZ_out or nz_out_end are absent, the WHOLE of A v
!   including zeros should be returned in P. If NZ_out and nz_out_end are
!   present, the NONZEROS in the product A * v must be placed in their
!   appropriate comnponents of reverse%P, while a list of indices of the
!   nonzeos placed in NZ_out( 1 : nz_out_end ). In both cases, the status
!   variable should be set to 0 unless the product is impossible in which
!   case status should be set to a nonzero value. If eval_ASPROD is not
!   present, BLLS_solve will either return to the user each time an evaluation
!   is required (see reverse above) or form the product directly from
!   user-provided %A.
!
!  eval_AFPROD is an OPTIONAL subroutine which if present must have the
!   arguments given below (see the interface blocks). The product A * v
!   (if transpose is .FALSE.) or A^T v (if transpose is .TRUE.) involving
!   the given matrix A and the vector v stored in V must be returned
!   in P. If transpose is .FALSE., only the components of V with
!   indices FREE(:n_free) should be used, the remaining components should be
!   treated as zero. If transpose is .TRUE., all of V should be used, but
!   only the components P(IFREE(:nfree) need be computed, the remainder will
!   be ignored. The status variable should be set to 0 unless the product
!   is impossible in which case status should be set to a nonzero value.
!   If eval_AFPROD is not present, BLLS_solve will either return to the user
!   each time an evaluation is required (see reverse above) or form the
!   product directly from user-provided %A.
!
!  eval_PREC is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product P^{-1} * v of the given
!   preconditioner P and vector v stored in V must be returned in P. 
!   The intention is that P is an approximation to A^T A. The status variable 
!   should be set to 0 unless the product is impossible in which case status 
!   should be set to a nonzero value. If eval_PREC is not present, BLLS_solve 
!   will return to the user each time a preconditioning operation is required 
!   (see reverse above) when control%preconditioner is not 0 or 1.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER, INTENT( INOUT ), DIMENSION( prob%n ) :: X_stat
     TYPE ( BLLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( BLLS_control_type ), INTENT( IN ) :: control
     TYPE ( BLLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( BLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse
     OPTIONAL :: eval_APROD, eval_ASPROD, eval_AFPROD, eval_PREC

!  interface blocks

     INTERFACE
       SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, INTENT( IN ) :: transpose
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P
       END SUBROUTINE eval_APROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_ASPROD( status, userdata, V, P, NZ_in, nz_in_start,     &
                               nz_in_end, NZ_out, nz_out_end )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
       INTEGER, OPTIONAL, INTENT( IN ) :: nz_in_start, nz_in_end
       INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_out_end
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_in
       INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_out
       END SUBROUTINE eval_ASPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_AFPROD( status, userdata, transpose, V, P, FREE, n_free )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, INTENT( IN ) :: transpose
       INTEGER, INTENT( IN ) :: n_free
       INTEGER, INTENT( IN ), DIMENSION( : ) :: FREE
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
       END SUBROUTINE eval_AFPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_PREC( status, userdata, V, P )
       USE GALAHAD_USERDATA_double
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
       END SUBROUTINE eval_PREC
     END INTERFACE

!  local variables

     INTEGER :: i, j, k, nap
     REAL :: time
     REAL ( KIND = wp ) :: val, av_bnd, x_j, g_j
     LOGICAL :: reset_bnd
     CHARACTER ( LEN = 6 ) :: string_iter
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  enter or re-enter the package and jump to appropriate re-entry point

     IF ( inform%status == 1 ) data%branch = 100
     IF ( inform%status == 10 ) data%branch = 200

     SELECT CASE ( data%branch )
     CASE ( 100 ) ; GO TO 100
     CASE ( 210 ) ; GO TO 210  ! re-entry with p = p + Av
     CASE ( 220 ) ; GO TO 220  ! re-entry with p = p + A^T v
     CASE ( 300 ) ; GO TO 300  ! re-entry with p = A v (dense or sparse)
     CASE ( 400 ) ; GO TO 400  ! re-entry with p = A v (dense or sparse)
     END SELECT

 100 CONTINUE

     IF ( control%out > 0 .AND. control%print_level >= 5 )                     &
       WRITE( control%out, 2000 ) prefix, ' entering '

! -------------------------------------------------------------------
!  if desired, generate a SIF file for problem passed

     IF ( control%generate_sif_file ) THEN
       CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,     &
                     control%infinity, .TRUE., no_linear = .TRUE. )
     END IF

!  SIF file generated
! -------------------------------------------------------------------

!  initialize time

     CALL CPU_TIME( data%time_start )

!  set initial timing breakdowns

     inform%time%total = 0.0 ; inform%time%analyse = 0.0
     inform%time%factorize = 0.0 ; inform%time%solve = 0.0

!  check that optional arguments are consistent -

!  operations with A

     data%use_aprod = PRESENT( eval_APROD ) .AND. PRESENT( eval_ASPROD ) .AND. &
                      PRESENT( eval_AFPROD )
     data%reverse = PRESENT( reverse )
     data%explicit_a = ALLOCATED( prob%A%type )
     data%reverse_prod = .NOT. ( data%explicit_a .OR. data%use_aprod )
     IF ( data%reverse_prod .AND. .NOT. data%reverse ) THEN
       inform%status = GALAHAD_error_optional ; GO TO 910
     END IF

!  operations with a preconditioner

     IF ( control%preconditioner == 0 ) THEN
       data%preconditioner = 0
     ELSE IF ( control%preconditioner == 1 ) THEN
       IF ( .NOT. data%use_aprod ) THEN
         data%preconditioner = 1
       ELSE
         data%preconditioner = 0
       END IF
     ELSE
       IF ( PRESENT( eval_PREC ) ) THEN
         data%preconditioner = 2
       ELSE IF ( PRESENT( reverse ) ) THEN
         data%preconditioner = 2
       ELSE IF ( .NOT. data%use_aprod ) THEN
         data%preconditioner = 1
       ELSE
         data%preconditioner = 0
       END IF
     END IF

     IF ( data%preconditioner == 2  .AND. .NOT. PRESENT( eval_prec ) .AND.     &
          .NOT. data%reverse ) THEN
       inform%status = GALAHAD_error_optional ; GO TO 900
     END IF

     IF ( control%maxit < 0 ) THEN
       data%maxit = HUGE( 1 ) - 1
     ELSE
       data%maxit = control%maxit
     END IF

     IF ( control%cg_maxit < 0 ) THEN
       data%cg_maxit = prob%n + 1
     ELSE
       data%cg_maxit = control%cg_maxit
     END IF

     data%steepest_descent = .TRUE.
     data%direct_subproblem_solve = data%explicit_a .AND.                      &
                                    control%direct_subproblem_solve
     data%header = .TRUE.
     data%weight = MAX( control%weight, zero )

     inform%iter = - 1

!  ===========================
!  control the output printing
!  ===========================

     data%out = control%out ; data%error = control%error
     data%print_level = 0
     IF ( control%start_print <= 0 ) THEN
       data%start_print = 0
     ELSE
       data%start_print = control%start_print
     END IF

     IF ( control%stop_print < 0 ) THEN
       data%stop_print = data%maxit + 2
     ELSE
       data%stop_print = control%stop_print
     END IF

     IF ( control%print_gap < 2 ) THEN
       data%print_gap = 1
     ELSE
       data%print_gap = control%print_gap
     END IF

!  error output

     data%set_printe = data%error > 0 .AND. control%print_level >= 1

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. control%print_level >= 2

!  as per printt, but with checking of residuals, etc

     data%set_printm = data%out > 0 .AND. control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. control%print_level >= 4

!  full debugging printing with significant arrays printed

     data%set_printd = data%out > 0 .AND. control%print_level >= 5

!  start setting control parameters

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%print_level = control%print_level
       data%printe = data%set_printe ; data%printi = data%set_printi
       data%printt = data%set_printt ; data%printm = data%set_printm
       data%printw = data%set_printw ; data%printd = data%set_printd
     ELSE
       data%print_level = 0
       data%printe = .FALSE. ; data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
     END IF

!  ensure that input parameters are within allowed ranges

     IF ( prob%n <= 0 .OR. prob%m <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 910
     ELSE IF ( data%explicit_a ) THEN
       IF ( .NOT. QPT_keyword_A( prob%A%type ) ) THEN
         inform%status = GALAHAD_error_restrictions
         GO TO 910
       END IF
     END IF

!  if required, write out problem

     IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
       WRITE( control%out, "( ' m, n = ', I0, 1X, I0 )" ) prob%m, prob%n
       WRITE( control%out, "( ' B = ', /, ( 5ES12.4 ) )" ) prob%B( : prob%m )
       SELECT CASE ( SMT_get( prob%A%type ) )
       CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS'  )
         WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )           &
           prob%A%val( : prob%m * prob%n )
       CASE ( 'SPARSE_BY_ROWS' )
         WRITE( control%out, "( ' A (row-wise) = ' )" )
         DO i = 1, prob%m
           WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                    &
             ( i, prob%A%col( j ), prob%A%val( j ),                            &
               j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
         END DO
       CASE ( 'SPARSE_BY_COLUMNS' )
         WRITE( control%out, "( ' A (column-wise) = ' )" )
         DO j = 1, prob%n
           WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                    &
             ( prob%A%row( i ), j, prob%A%val( i ),                            &
               i = prob%A%ptr( j ), prob%A%ptr( j + 1 ) - 1 )
         END DO
       CASE ( 'COORDINATE' )
         WRITE( control%out, "( ' A (co-ordinate) = ' )" )
         WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                      &
        ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne  )
       END SELECT
       WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" )                   &
         prob%X_l( : prob%n )
       WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" )                   &
         prob%X_u( : prob%n )
     END IF

!  check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

     reset_bnd = .FALSE.
     DO i = 1, prob%n
       IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 910
       ELSE IF ( prob%X_u( i ) == prob%X_l( i ) ) THEN
       ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                 &
                 <= control%identical_bounds_tol ) THEN
         av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
         prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
         reset_bnd = .TRUE.
       ELSE IF ( control%cold_start == 0 ) THEN
         IF ( X_stat( i ) < 0 ) THEN
            prob%X_l( i ) =  prob%X_l( i )
           reset_bnd = .TRUE.
         ELSE IF ( X_stat( i ) > 0 ) THEN
            prob%X_l( i ) =  prob%X_u( i )
           reset_bnd = .TRUE.
         END IF
       END IF
     END DO
     IF ( reset_bnd .AND. data%printi ) WRITE( control%out,                    &
       "( ' ', /, A, '   **  Warning: one or more variable bounds reset ' )" ) &
         prefix

!  allocate workspace arrays

     array_name = 'blls: prob%C'
     CALL SPACE_resize_array( prob%m, prob%C, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'blls: prob%G'
     CALL SPACE_resize_array( prob%n, prob%G, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'blls: prob%Z'
     CALL SPACE_resize_array( prob%n, prob%Z, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'blls: data%R'
     CALL SPACE_resize_array( prob%m, data%R, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'blls: data%S'
     CALL SPACE_resize_array( prob%n, data%S, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( data%preconditioner /= 0 ) THEN
       array_name = 'blls: data%DIAG'
       CALL SPACE_resize_array( prob%n, data%DIAG, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     array_name = 'blls: data%X_new'
     CALL SPACE_resize_array( prob%n, data%X_new, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'blls: data%X_status'
     CALL SPACE_resize_array( prob%n, data%X_status, inform%status,     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'blls: data%OLD_status'
     CALL SPACE_resize_array( prob%n, data%OLD_status, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( data%reverse ) THEN
       array_name = 'blls: reverse%NZ_in'
       CALL SPACE_resize_array( prob%n, reverse%NZ_in, inform%status,          &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'blls: reverse%NZ_out'
       CALL SPACE_resize_array( prob%n, reverse%NZ_out, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'blls: reverse%V'
       CALL SPACE_resize_array( MAX( prob%m, prob%n ), reverse%V,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'blls: reverse%P'
       CALL SPACE_resize_array( MAX( prob%m, prob%n ), reverse%P,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     ELSE
       array_name = 'blls: data%NZ_in'
       CALL SPACE_resize_array( prob%n, data%NZ_in, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'blls: data%V'
       CALL SPACE_resize_array( prob%n, data%V, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'blls: data%P'
       CALL SPACE_resize_array( prob%n, data%P, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

!  build a copy of A stored by columns

     IF ( data%explicit_a ) CALL CONVERT_to_sparse_column_format( prob%A,      &
                      data%A, control%CONVERT_control, inform%CONVERT_inform )

!  compute the diagonal preconditioner if required

     IF ( data%preconditioner /= 0 ) THEN
       IF ( data%explicit_a ) THEN
         DO j = 1, prob%n
           val = zero
           DO k = data%A%ptr( j ), data%A%ptr( j + 1 ) - 1
             val = val + data%A%val( k ) ** 2
           END DO
           data%DIAG( j ) = val
         END DO
       ELSE
         data%DIAG( : prob%n ) = one
       END IF
       IF ( control%weight > zero ) data%DIAG( : prob%n ) =                    &
          data%DIAG( : prob%n ) + control%weight
!write(6,"( ' diag ', 4ES12.4 )" )  data%DIAG( : prob%n )
       IF ( data%set_printm ) WRITE( data%out,                                 &
         "( /, A, ' diagonal preconditioner, min, max =', 2ES11.4 )" ) prefix, &
           MINVAL( data%DIAG( : prob%n ) ), MAXVAL( data%DIAG( : prob%n ) )
     END IF

!  if  a block matrix factorization is possible and to be used to solve the
!  subproblem, set as many components as possible

     IF ( data%direct_subproblem_solve ) THEN

!  the 1,1 block

       CALL SMT_put( data%H_sbls%type, 'IDENTITY', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate
         GO TO 910
       END IF
       data%H_sbls%m = prob%m ; data%H_sbls%n = prob%m

!  the 2,1 block

       CALL SMT_put( data%AT_sbls%type, 'SPARSE_BY_ROWS', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate
         GO TO 910
       END IF
       data%AT_sbls%n = prob%m

       array_name = 'blls: data%AT_sbls%val'
       CALL SPACE_resize_array( data%A%ne, data%AT_sbls%val, inform%status,    &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'blls: data%AT_sbls%col'
       CALL SPACE_resize_array( data%A%ne, data%AT_sbls%col, inform%status,    &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'blls: data%AT_sbls%ptr'
       CALL SPACE_resize_array( prob%n + 1, data%AT_sbls%ptr, inform%status,   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  the 2,2 block

!  regularized case

       data%stabilisation_weight                                               &
         = MAX( control%weight, control%stabilisation_weight, zero )
       IF ( data%stabilisation_weight > zero ) THEN
         CALL SMT_put( data%C_sbls%type, 'SCALED_IDENTITY', inform%alloc_status)
         IF ( inform%alloc_status /= 0 ) THEN
           inform%status = GALAHAD_error_allocate
           GO TO 910
         END IF

         array_name = 'blls: data%C_sbls%val'
         CALL SPACE_resize_array( 1, data%C_sbls%val, inform%status,           &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910
         data%C_sbls%val( 1 ) = data%stabilisation_weight

!  unregularized case

       ELSE
         CALL SMT_put( data%C_sbls%type, 'ZERO', inform%alloc_status )
         IF ( inform%alloc_status /= 0 ) THEN
           inform%status = GALAHAD_error_allocate
           GO TO 910
         END IF
       END IF

!  solution vector

       array_name = 'blls: data%SBLS_sol'
       CALL SPACE_resize_array( prob%n + prob%m, data%SBLS_sol, inform%status, &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

     END IF

!  ------------------------
!  start the main iteration
!  ------------------------

     data%change_status = prob%n
     data%X_status = 0

     IF ( data%set_printi ) WRITE( data%out,                                   &
       "( /, A, 9X, 'S=steepest descent, F=factorization used' )" ) prefix

 110 CONTINUE ! mock iteration loop
       CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start

!  set the print levels for the iteration

       inform%iter = inform%iter + 1
       IF ( ( inform%iter >= data%start_print .AND.                            &
              inform%iter < data%stop_print ) .AND.                            &
            MOD( inform%iter - data%start_print, data%print_gap ) == 0 ) THEN
         data%print_level = control%print_level
         data%printe = data%set_printe ; data%printi = data%set_printi
         data%printt = data%set_printt ; data%printm = data%set_printm
         data%printw = data%set_printw ; data%printd = data%set_printd
       ELSE
         data%print_level = 0
         data%printe = .FALSE. ; data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
       END IF

!  compute the residual

       IF ( data%explicit_a ) THEN
         prob%C( : prob%m ) = - prob%B( : prob%m )
         DO j = 1, prob%n
           x_j = prob%X( j )
           DO k = data%A%ptr( j ), data%A%ptr( j + 1 ) - 1
             prob%C( data%A%row( k ) )                                         &
               = prob%C( data%A%row( k ) ) + data%A%val( k ) * x_j
           END DO
         END DO
       ELSE IF ( data%use_aprod ) THEN
         prob%C( : prob%m ) = - prob%B( : prob%m )
         CALL eval_APROD( data%eval_status, userdata, .FALSE., prob%X, prob%C )
         IF ( data%eval_status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 910
         END IF
       ELSE
!        reverse%P( : prob%m ) = - prob%B
         reverse%V( : prob%n ) = prob%X
         reverse%transpose = .FALSE.
         data%branch = 210 ; inform%status = 2 ; RETURN
       END IF

!  re-entry point after the jacobian-vector product

 210   CONTINUE

!  compute the gradient

       IF ( data%explicit_a ) THEN
         prob%G( : prob%n ) = zero
         DO j = 1, prob%n
           g_j = zero
           DO k = data%A%ptr( j ), data%A%ptr( j + 1 ) - 1
             g_j = g_j + data%A%val( k ) * prob%C( data%A%row( k ) )
           END DO
           prob%G( j ) = g_j
         END DO
!write(6,"(' g', 4ES12.4)" ) prob%G( : prob%n )
       ELSE IF ( data%use_aprod ) THEN
         prob%G( : prob%n ) = zero
         CALL eval_APROD( data%eval_status, userdata, .TRUE., prob%C, prob%G )
         IF ( data%eval_status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 910
         END IF
       ELSE
         prob%C( : prob%m ) = reverse%P( : prob%m ) - prob%B( : prob%m )
!        reverse%P( : prob%n ) = zero
         reverse%V( : prob%m ) = prob%C( : prob%m )
         reverse%transpose = .TRUE.
         data%branch = 220 ; inform%status = 3 ; RETURN
       END IF

!  re-entry point after the Jacobian-transpose vector product

 220   CONTINUE
       IF ( data%reverse_prod ) prob%G( : prob%n ) = reverse%P( : prob%n )

!  compute the objective function

       inform%obj = half * DOT_PRODUCT( prob%C( : prob%m ), prob%C( : prob%m ) )

!  adjust the value and gradient to account for any regularization term

       IF ( control%weight > zero ) THEN
         inform%obj = inform%obj + half * control%weight *                     &
           DOT_PRODUCT( prob%X( : prob%n ), prob%X( : prob%n ) )
         prob%G( : prob%n )                                                    &
           = prob%G( : prob%n ) + control%weight * prob%X( : prob%n )
       END IF
!write(6,"(' g', 4ES12.4)" ) prob%G( : prob%n )

!  record the dual variables

       prob%Z( : prob%n ) = prob%G( : prob%n )

!  record the dual variables

!  compute the norm of the projected gradient

       inform%norm_pg =                                                        &
         MAXVAL( ABS( MAX( prob%X_l( : prob%n ),                               &
                           MIN( prob%X( : prob%n ) - prob%G( : prob%n ),       &
                                prob%X_u( : prob%n ) ) ) - prob%X( : prob%n ) ))

!  print details of the current iteration

       IF ( ( data%printi .AND. data%header ) .OR. data%printt )               &
         WRITE( data%out, "( /, A, '  #its   #cg            f          ',      &
        &        ' proj gr     step    #free change   time' )" ) prefix
       data%header = .FALSE.
       IF ( data%printi ) THEN
         string_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
         IF ( inform%iter > 1 ) THEN
           WRITE( data%out, "( A, 2A6, ES22.14, 2ES10.3, 2I7, A7 )" )          &
             prefix, string_iter, data%string_cg_iter,                         &
             inform%obj, inform%norm_pg, data%norm_step,                       &
             data%n_free, data%change_status,                                  &
             STRING_real_7( inform%time%total )
         ELSE IF ( inform%iter == 1 ) THEN
           WRITE( data%out, "( A, 2A6, ES22.14, 2ES10.3, I7, 6X, '-',  A7 )" ) &
             prefix, string_iter, data%string_cg_iter,                         &
             inform%obj, inform%norm_pg, data%norm_step, data%n_free,          &
             STRING_real_7( inform%time%total )
         ELSE
           WRITE( data%out, "( A, I6, '     -', ES22.14, ES10.3,               &
          & '      -         -      -', A7 )" )                                &
             prefix, inform%iter, inform%obj, inform%norm_pg,                  &
             STRING_real_7( inform%time%total )
         END IF
       END IF

!  test for an approximate first-order critical point

       IF ( inform%norm_pg <= control%stop_d ) THEN
         inform%status = GALAHAD_ok ; GO TO 910
       END IF

!  test to see if more than maxit iterations have been performed

       IF ( inform%iter > data%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  check that the CPU time limit has not been reached

       IF ( control%cpu_time_limit >= zero .AND.                               &
            inform%time%total > control%cpu_time_limit ) THEN
         inform%status = GALAHAD_error_cpu_limit ; GO TO 900
       END IF

!  ----------------------------------------------------------------------------
!                      compute the search direction
!  ----------------------------------------------------------------------------

!  - - - - - - - - - - - - - - - negative gradient - - - - - - - - - - - - - -

!  compute the search direction as the steepest-descent direction

       IF ( data%steepest_descent ) THEN
         IF ( data%printm )  WRITE( data%out,                                  &
           "( /, A, ' steepest descent search direction' )" ) prefix

!  assign the search direction

         IF ( data%preconditioner /= 0 ) THEN
           data%S( : prob%n ) =                                                &
             - prob%G( : prob%n ) / SQRT( data%DIAG( : prob%n ) )
         ELSE
           data%S( : prob%n ) = - prob%G( : prob%n )
         END IF
         data%string_cg_iter = '     S'

!  initialize the status of the variables

         DO i = 1, prob%n
           IF ( prob%X_l( i ) == prob%X_u( i ) ) THEN
             data%X_status( i ) = 3
           ELSE
             data%X_status( i ) = 0
           END IF
         END DO
         GO TO 310
       END IF

!  - - - - - - - - - - - - - - - augmented system - - - - - - - - - - - - - - -

!  compute the search direction by minimizing the objective over
!  the free subspace by solving the related augmented system

!    (    I        A_F   ) (  y  ) = (   b - A x  )
!    ( A_F^T  - weight I ) ( s_F )   ( weight x_F )

       IF ( data%direct_subproblem_solve ) THEN

!  set up the block matrices. Copy the free columns of A into the rows
!  of AT

         data%n_free = 0 ; nap = 0
         DO j = 1, prob%n
           IF ( data%X_status( j ) == 0 ) THEN
             data%n_free = data%n_free + 1
             data%AT_sbls%ptr( data%n_free ) = nap + 1
             DO k = data%A%ptr( j ), data%A%ptr( j + 1 ) - 1
               nap = nap + 1
               data%AT_sbls%col( nap ) = data%A%row( k )
               data%AT_sbls%val( nap ) = data%A%val( k )
             END DO

!  include components of the right-hand side vector

             data%SBLS_sol( prob%m + data%n_free ) =                           &
               data%stabilisation_weight * prob%X( j )
           END IF
         END DO
         data%AT_sbls%ptr( data%n_free + 1 ) = nap + 1
         data%AT_sbls%m = data%n_free

!  make sure that the 2,2 block has the correct dimension

         data%C_sbls%m = data%n_free ; data%C_sbls%n = data%n_free

!  complete the right-hand side vector

         data%SBLS_sol( : prob%m ) = - prob%C( : prob%m )

!  form and factorize the augmented matrix

         CALL SBLS_form_and_factorize( prob%m, data%n_free,                    &
                 data%H_sbls, data%AT_sbls, data%C_sbls, data%SBLS_data,       &
                 control%SBLS_control, inform%SBLS_inform )

!  test for factorization failure

         IF ( inform%SBLS_inform%status < 0 ) THEN
           IF ( data%printe )                                                  &
             WRITE( control%error, 2010 ) prefix, inform%SBLS_inform%status,   &
               'SBSL_form_and_factorize'
           CALL SYMBOLS_status( inform%SBLS_inform%status, control%out,        &
                                prefix, 'SBSL_form_and_factorize' )
           inform%status = GALAHAD_error_factorization ; GO TO 910
         END IF

!  solve the system

         CALL SBLS_solve( prob%m, data%n_free, data%AT_sbls, data%C_sbls,      &
                 data%SBLS_data, control%SBLS_control, inform%SBLS_inform,     &
                 data%SBLS_sol )

!  extract the search direction

         data%n_free = 0
         DO j = 1, prob%n
           IF ( data%X_status( j ) == 0 ) THEN
             data%n_free = data%n_free + 1
             data%S( j ) = data%SBLS_sol( prob%m + data%n_free )
           ELSE
             data%S( j ) = zero
           END IF
         END DO
         IF ( data%printm ) WRITE( data%out, "( ' stg, sts = ', 2ES12.4 )" )   &
           DOT_PRODUCT( data%S( : prob%n ), prob%G( : prob%n ) ),              &
           DOT_PRODUCT( data%S( : prob%n ), data%S( : prob%n ) )
         data%string_cg_iter = '     F'

         GO TO 310
       END IF

!  - - - - - - - - - - - - - - - - - - CGLS - - - - - - - - - - - - - - - - - -

!  compute the search direction by minimizing the objective over the
!  free subspace using CGLS

       IF ( data%printm )  WRITE( data%out,                                    &
        "( /, A, ' search direction from CGLS over free variables' )" ) prefix

!  record the current estimate of the solution and its residual,
!  and prepare for minimization

       data%X_new( : prob%n ) = prob%X( : prob%n )
       data%R( : prob%m ) = prob%C( : prob%m )
       data%cgls_status = 1

!  if reverse communication is to be used, store the list of free variables

       IF ( data%reverse ) THEN
         reverse%nz_in_start = 1
         reverse%nz_in_end = 0
         DO i = 1, prob%n
           IF ( data%X_status( i ) == 0 ) THEN
             reverse%nz_in_end = reverse%nz_in_end + 1
             reverse%NZ_in( reverse%nz_in_end ) = i
           END IF
         END DO
       END IF

!  minimization loop

 300   CONTINUE ! mock CGLS loop

!  find an improved point, X_new, by conjugate-gradient least-squares ...

!  ... using the available Jacobian ...

         IF (  data%explicit_a ) THEN
           IF ( data%preconditioner == 0 ) THEN
             CALL BLLS_cgls( prob%m, prob%n, data%weight,                      &
                             data%f_new, data%X_new, data%R,                   &
                             data%X_status, control%stop_cg_relative,          &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, control%out,                    &
                             control%print_level, prefix,                      &
                             data%subproblem_data, userdata,                   &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, A_ptr = data%A%ptr,             &
                             A_row = data%A%row, A_val = data%A%val )
           ELSE IF ( data%preconditioner == 1 ) THEN
             CALL BLLS_cgls( prob%m, prob%n, data%weight,                      &
                             data%f_new, data%X_new, data%R,                   &
                             data%X_status, control%stop_cg_relative,          &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, control%out,                    &
                             control%print_level, prefix,                      &
                             data%subproblem_data, userdata,                   &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, A_ptr = data%A%ptr,             &
                             A_row = data%A%row, A_val = data%A%val,           &
                             preconditioned = .TRUE.,                          &
                             DPREC = data%DIAG )
           ELSE
             CALL BLLS_cgls( prob%m, prob%n, data%weight,                      &
                             data%f_new, data%X_new, data%R,                   &
                             data%X_status, control%stop_cg_relative,          &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, control%out,                    &
                             control%print_level, prefix,                      &
                             data%subproblem_data, userdata,                   &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, A_ptr = data%A%ptr,             &
                             A_row = data%A%row, A_val = data%A%val,           &
                             reverse = reverse, preconditioned = .TRUE.,       &
                             eval_PREC = eval_PREC )
           END IF

!  ... or products via the user's subroutine or reverse communication ...

         ELSE
           IF ( data%preconditioner == 0 ) THEN
             CALL BLLS_cgls( prob%m, prob%n, data%weight,                      &
                             data%f_new, data%X_new, data%R,                   &
                             data%X_status, control%stop_cg_relative,          &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, control%out,                    &
                             control%print_level, prefix,                      &
                             data%subproblem_data, userdata,                   &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, eval_AFPROD = eval_AFPROD,      &
                             reverse = reverse )
           ELSE IF ( data%preconditioner == 1 ) THEN
             CALL BLLS_cgls( prob%m, prob%n, data%weight,                      &
                             data%f_new, data%X_new, data%R,                   &
                             data%X_status, control%stop_cg_relative,          &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, control%out,                    &
                             control%print_level, prefix,                      &
                             data%subproblem_data, userdata,                   &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, eval_AFPROD = eval_AFPROD,      &
                             reverse = reverse, preconditioned = .TRUE.,       &
                             DPREC = data%DIAG )
           ELSE
             CALL BLLS_cgls( prob%m, prob%n, data%weight,                      &
                             data%f_new, data%X_new, data%R,                   &
                             data%X_status, control%stop_cg_relative,          &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, control%out,                    &
                             control%print_level, prefix,                      &
                             data%subproblem_data, userdata,                   &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, eval_AFPROD = eval_AFPROD,      &
                             reverse = reverse, preconditioned = .TRUE.,       &
                             eval_PREC = eval_PREC )
           END IF
         END IF

!  check the output status

         SELECT CASE ( data%cgls_status )

!  successful exit with the new point, record the resulting step

         CASE ( GALAHAD_ok, GALAHAD_error_max_iterations )
           data%S( : prob%n ) = data%X_new( : prob%n ) - prob%X( : prob%n )
           data%norm_step = MAXVAL( ABS( data%S( : prob%n ) ) )
           data%n_free = COUNT( data%X_status( : prob%n ) == 0 )
           data%string_cg_iter = ADJUSTR( STRING_integer_6( data%cg_iter ) )
           inform%cg_iter = inform%cg_iter + data%cg_iter
           GO TO 310

!  form the matrix-vector product A * v with sparse v

         CASE ( 2 )
           data%branch = 300 ; inform%status = 4 ; RETURN

!  form the sparse matrix-vector product A * v

         CASE ( 3 )
           data%branch = 300 ; inform%status = 6 ; RETURN

!  form the preconditioned product P^-1 * v

         CASE ( 4 )
           data%branch = 300 ; inform%status = 7 ; RETURN

!  error exit without the new point

         CASE DEFAULT
           IF ( data%printe )                                                  &
             WRITE( control%error, 2010 ) prefix, data%cgls_status, 'BLLS_cgls'
           inform%status = data%cgls_status
           GO TO 910
         END SELECT

         GO TO 300  ! end of minimization loop

!  a search-direction had been computed

 310   CONTINUE ! end of mock CGLS loop

!  ----------------------------------------------------------------------------
!           perform a projected arc search along the search direction
!  ----------------------------------------------------------------------------

!  set parameters for arc search

       IF ( .NOT. control%exact_arc_search ) THEN
         data%alpha_0 = alpha_search
       END IF

!  arc search loop

       data%arc_search_status = 1
 400   CONTINUE ! mock arc search loop

!  find an improved point, X_new, by exact arc_search ...

         IF ( control%exact_arc_search ) THEN

!  ... using the available Jacobian ...

           IF (  data%explicit_a ) THEN
             CALL BLLS_exact_arc_search( prob%m, prob%n, data%weight,          &
                                         prob%X_l, prob%X_u,                   &
                                         control%infinity, prob%X, prob%C,     &
                                         data%S, data%X_status,                &
                                         control%identical_bounds_tol,         &
                                         control%alpha_max,                    &
                                         control%arcsearch_max_steps,          &
                                         control%out, control%print_level,     &
                                         prefix, data%X_new, data%f_new,       &
                                         data%phi_new,                         &
                                         data%alpha_new, data%segments,        &
                                         data%subproblem_data, userdata,       &
                                         data%arc_search_status,               &
                                         inform%alloc_status,                  &
                                         inform%bad_alloc,                     &
                                         A_ptr = data%A%ptr,                   &
                                         A_row = data%A%row,                   &
                                         A_val = data%A%val )

!  ... or products via the user's subroutine or reverse communication ...

           ELSE
             CALL BLLS_exact_arc_search( prob%m, prob%n, data%weight,          &
                                         prob%X_l, prob%X_u,                   &
                                         control%infinity, prob%X, prob%C,     &
                                         data%S, data%X_status,                &
                                         control%identical_bounds_tol,         &
                                         control%alpha_max,                    &
                                         control%arcsearch_max_steps,          &
                                         control%out, control%print_level,     &
                                         prefix, data%X_new, data%f_new,       &
                                         data%phi_new,                         &
                                         data%alpha_new, data%segments,        &
                                         data%subproblem_data, userdata,       &
                                         data%arc_search_status,               &
                                         inform%alloc_status,                  &
                                         inform%bad_alloc,                     &
                                         eval_ASPROD = eval_ASPROD,            &
                                         reverse = reverse )
           END IF

!   ... or inexact arc_search ...

         ELSE

!  ... using the available Jacobian ...

           IF (  data%explicit_a ) THEN
             CALL BLLS_inexact_arc_search( prob%m, prob%n, data%weight,        &
                                           prob%X_l, prob%X_u,                 &
                                           control%infinity,                   &
                                           prob%X, prob%C, data%S,             &
                                           data%X_status, fixed_tol,           &
                                           control%alpha_max,                  &
                                           control%alpha_initial,              &
                                           control%alpha_reduction,            &
                                           control%arcsearch_acceptance_tol,   &
                                           control%arcsearch_max_steps,        &
                                           control%advance, control%out,       &
                                           control%print_level, prefix,        &
                                           data%X_new, data%f_new,             &
                                           data%phi_new,                       &
                                           data%alpha_new, data%steps,         &
                                           data%subproblem_data, userdata,     &
                                           data%arc_search_status,             &
                                           inform%alloc_status,                &
                                           inform%bad_alloc,                   &
                                           A_ptr = data%A%ptr,                 &
                                           A_row = data%A%row,                 &
                                           A_val = data%A%val,                 &
                                           B = prob%B )

!  ... or products via the user's subroutine or reverse communication

           ELSE
             CALL BLLS_inexact_arc_search( prob%m, prob%n, data%weight,        &
                                           prob%X_l, prob%X_u,                 &
                                           control%infinity,                   &
                                           prob%X, prob%C, data%S,             &
                                           data%X_status, fixed_tol,           &
                                           control%alpha_max,                  &
                                           control%alpha_initial,              &
                                           control%alpha_reduction,            &
                                           control%arcsearch_acceptance_tol,   &
                                           control%arcsearch_max_steps,        &
                                           control%advance, control%out,       &
                                           control%print_level, prefix,        &
                                           data%X_new, data%f_new,             &
                                           data%phi_new,                       &
                                           data%alpha_new, data%steps,         &
                                           data%subproblem_data, userdata,     &
                                           data%arc_search_status,             &
                                           inform%alloc_status,                &
                                           inform%bad_alloc,                   &
                                           eval_ASPROD = eval_ASPROD,          &
                                           reverse = reverse,                  &
                                           B = prob%B )
           END IF
         END IF

!  check the output status

         SELECT CASE ( data%arc_search_status )

!  successful exit with the new point

         CASE ( 0 )
           data%norm_step =                                                    &
             MAXVAL( ABS( data%X_new( : prob%n ) - prob%X( : prob%n ) ) )
           data%n_free = COUNT( data%X_status( : prob%n ) == 0 )
           GO TO 410

!  error exit without the new point

         CASE ( : - 1 )
           IF ( data%printe ) THEN
             IF ( control%exact_arc_search ) THEN
               WRITE( control%error, 2010 )                                    &
                 prefix, data%arc_search_status, 'BLLS_exact_arc_search'
             ELSE
               WRITE( control%error, 2010 )                                    &
                 prefix, data%arc_search_status, 'BLLS_inexact_arc_search'
             END IF
           END IF
           inform%status = data%arc_search_status
           GO TO 910

!  form the matrix-vector product A * v

         CASE ( 2 )
           data%branch = 400 ; inform%status = 2 ; RETURN

!  form the sparse matrix-vector product A * v

         CASE ( 3 )
           data%branch = 400 ; inform%status = 4 ; RETURN

!  form the sparse matrix-vector product A * v and return a sparse result

         CASE ( 4 )
           data%branch = 400 ; inform%status = 5 ; RETURN
         END SELECT
         GO TO 400 ! end of arc length loop

!  the arc length has been computed

 410   CONTINUE  ! end of mock arc search loop

!  record the new point in x

       prob%X( : prob%n ) = data%X_new( : prob%n )
       inform%obj = data%phi_new

!  record the number of variables that have changed status

       IF ( inform%iter > 0 ) data%change_status                               &
         = COUNT( data%X_status( : prob%n ) /= data%OLD_status( : prob%n ) )
       data%OLD_status( : prob%n ) = data%X_status( : prob%n )

!  decide whether the next iteration uses steepest descent or CGLS

       IF ( data%steepest_descent ) THEN
         data%steepest_descent = data%change_status > control%change_max
!        data%steepest_descent = data%change_status > control%change_max .OR.  &
!          MOD( inform%iter, control%ratio_cg_vs_sd + 1 ) == 0
       ELSE
         data%steepest_descent = data%change_status == 0
!        data%steepest_descent = data%change_status == 0 .OR.                  &
!          MOD( inform%iter, control%ratio_cg_vs_sd + 1 ) == 0
       END IF
       GO TO 110 ! end of mock iteration loop

!  ----------------------
!  end the main iteration
!  ----------------------

!  successful return

 900 CONTINUE
     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  error returns

 910 CONTINUE
     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start

     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, control%out, prefix, 'BLLS_solve' )
!      SELECT CASE ( inform%status )
!      CASE ( GALAHAD_ok )
!      CASE ( GALAHAD_error_allocate )
!        WRITE( control%out, 2020 ) prefix, 'allocation error'
!        WRITE( control%out, 2030 ) prefix, inform%alloc_status,               &
!                                           inform%bad_alloc
!      CASE ( GALAHAD_error_deallocate )
!        WRITE( control%out, 2020 ) prefix, 'de-allocation error'
!        WRITE( control%out, 2030 ) prefix, inform%alloc_status,               &
!                                           inform%bad_alloc
!      CASE ( GALAHAD_error_restrictions )
!        WRITE( control%out, 2020 ) prefix, 'input restriction violated'
!      CASE ( GALAHAD_error_dual_infeasible )
!        WRITE( control%out, 2020 ) prefix, 'no feasible point'
!      CASE ( GALAHAD_error_unbounded )
!        WRITE( control%out, 2020 ) prefix, 'problem unbounded'
!      CASE ( GALAHAD_error_max_iterations )
!        WRITE( control%out, 2020 ) prefix, 'iteration limit exceeded'
!      CASE ( GALAHAD_error_max_inner_its )
!        WRITE( control%out, 2020 ) prefix, 'inner iteration limit exceeded'
!      CASE ( GALAHAD_error_cpu_limit )
!        WRITE( control%out, 2020 ) prefix, 'CPU time limit exceeded'
!      CASE ( GALAHAD_error_inertia )
!        WRITE( control%out, 2020 ) prefix, 'problem is not strictly convex'
!      CASE DEFAULT
!        WRITE( control%out, 2020 ) prefix, 'undefined error'
!      END SELECT
     ELSE IF ( data%printe ) THEN
       WRITE( control%error, 2010 ) prefix, inform%status, 'BLLS_solve'
     END IF
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  Non-executable statements

2000 FORMAT( /, A, ' --', A, ' BLLS_solve' )
2010 FORMAT( A, '   **  Error return, status = ', I0, ', from ', A )
!2020 FORMAT( /, A, ' BLLS error exit: ', A )
!2030 FORMAT( /, A, ' allocation error, status = ', I0, ', for ', A )

!  End of BLLS_solve

      END SUBROUTINE BLLS_solve

!-*-*-*-*-*-   B L L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE BLLS_terminate( data, control, inform, reverse )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine BLLS_initialize
!   control see Subroutine BLLS_initialize
!   inform  see Subroutine BLLS_solve
!   reverse see Subroutine BLLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( BLLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( BLLS_control_type ), INTENT( IN ) :: control
     TYPE ( BLLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( BLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by SBLS

     CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,                &
                          inform%SBLS_inform )
     IF ( inform%SBLS_inform%status /= GALAHAD_ok ) THEN
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = inform%SBLS_inform%alloc_status
!      inform%bad_alloc = inform%SBLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all those required for reverse communication

     IF ( PRESENT( reverse ) ) THEN
       CALL BLLS_reverse_terminate( reverse, control, inform )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN
     END IF

!  Deallocate all those required when solving the subproblems

     CALL BLLS_subproblem_terminate( data%subproblem_data, control, inform )

!  Deallocate all remaining allocated arrays

     array_name = 'blls: data%A%ptr'
     CALL SPACE_dealloc_array( data%A%ptr,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%A%col'
     CALL SPACE_dealloc_array( data%A%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%A%val'
     CALL SPACE_dealloc_array( data%A%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%R'
     CALL SPACE_dealloc_array( data%R,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%SBLS_sol'
     CALL SPACE_dealloc_array( data%SBLS_sol,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%X_new'
     CALL SPACE_dealloc_array( data%X_new,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%X_status'
     CALL SPACE_dealloc_array( data%X_status,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%OLD_status'
     CALL SPACE_dealloc_array( data%OLD_status,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%P'
     CALL SPACE_dealloc_array( data%P,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%DIAG'
     CALL SPACE_dealloc_array( data%DIAG,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%NZ_in'
     CALL SPACE_dealloc_array( data%NZ_in,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%NZ_out'
     CALL SPACE_dealloc_array( data%NZ_out,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%AT_sbls%val'
     CALL SPACE_dealloc_array( data%AT_sbls%val,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%AT_sbls%col'
     CALL SPACE_dealloc_array( data%AT_sbls%col,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%AT_sbls%ptr'
     CALL SPACE_dealloc_array( data%AT_sbls%ptr,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%SBLS_sol'
     CALL SPACE_dealloc_array( data%SBLS_sol,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%C_sbls%val'
     CALL SPACE_dealloc_array( data%C_sbls%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine BLLS_terminate

     END SUBROUTINE BLLS_terminate

!-*-   B L L S _ R E V E R S E _ T E R M I N A T E   S U B R O U T I N E   -*-

     SUBROUTINE BLLS_reverse_terminate( reverse, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   reverse see Subroutine BLLS_solve
!   control see Subroutine BLLS_initialize
!   inform  see Subroutine BLLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( BLLS_reverse_type ), INTENT( INOUT ) :: reverse
     TYPE ( BLLS_control_type ), INTENT( IN ) :: control
     TYPE ( BLLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

     array_name = 'blls: reverse%V'
     CALL SPACE_dealloc_array( reverse%V,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: reverse%P'
     CALL SPACE_dealloc_array( reverse%P,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: reverse%NZ_in'
     CALL SPACE_dealloc_array( reverse%NZ_in,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: reverse%NZ_out'
     CALL SPACE_dealloc_array( reverse%NZ_out,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine BLLS_reverse_terminate

     END SUBROUTINE BLLS_reverse_terminate

!-   B L L S _ S U B P R O B L E M _ T E R M I N A T E   S U B R O U T I N E   -

     SUBROUTINE BLLS_subproblem_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine BLLS_initialize
!   control see Subroutine BLLS_initialize
!   inform  see Subroutine BLLS_solve
!   reverse see Subroutine BLLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( BLLS_subproblem_data_type ), INTENT( INOUT ) :: data
     TYPE ( BLLS_control_type ), INTENT( IN ) :: control
     TYPE ( BLLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'blls: data%FREE'
     CALL SPACE_dealloc_array( data%FREE,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%P_used'
     CALL SPACE_dealloc_array( data%P_used,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%NZ_d'
     CALL SPACE_dealloc_array( data%NZ_d,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%NZ_out'
     CALL SPACE_dealloc_array( data%NZ_out,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%BREAK_points'
     CALL SPACE_dealloc_array( data%BREAK_points,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%P'
     CALL SPACE_dealloc_array( data%P,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%Q'
     CALL SPACE_dealloc_array( data%Q,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%R'
     CALL SPACE_dealloc_array( data%R,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%G'
     CALL SPACE_dealloc_array( data%G,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%R_a'
     CALL SPACE_dealloc_array( data%R_a,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%R_f'
     CALL SPACE_dealloc_array( data%R_f,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%X_a'
     CALL SPACE_dealloc_array( data%X_a,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%D_f'
     CALL SPACE_dealloc_array( data%D_f,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%W'
     CALL SPACE_dealloc_array( data%W,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%X_debug'
     CALL SPACE_dealloc_array( data%X_debug,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'blls: data%R_debug'
     CALL SPACE_dealloc_array( data%R_debug,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine BLLS_subproblem_terminate

     END SUBROUTINE BLLS_subproblem_terminate

! -*-*-  B L L S _ E X A C T _ A R C _ S E A R C H   S U B R O U T I N E  -*-*-

     SUBROUTINE BLLS_exact_arc_search( m, n, weight, X_l, X_u, bnd_inf,        &
                                       X_s, R_s, D_s, X_status,                &
                                       feas_tol, alpha_max, max_segments,      &
                                       out, print_level, prefix,               &
                                       X_alpha, f_alpha, phi_alpha, alpha,     &
                                       segment, data, userdata, status,        &
                                       alloc_status, bad_alloc,                &
                                       A_ptr, A_row, A_val, eval_ASPROD,       &
                                       reverse )

!  Find the arc minimizer in the direction d_s from x_s of the regularized 
!  least-squares objective function

!    phi(x) = f(x) + weight * rho(x), where
!      f(x) = 1/2 || A x - b ||_2^2 and 
!      rho(x) = 1/2 || x ||^2,

!  within the feasible "box" x_l <= x <= x_u

!  Define the arc x(alpha) = projection of x_s + alpha * d_s into the
!  feasible box. The arc minimizer is the first minimizer of the objective
!  function for points lying on x(alpha), with 0 <= alpha <= alpha_max

!  The value of the array X_status gives the status of the variables

!  IF X_status( I ) = 0, the I-th variable is free
!  IF X_status( I ) = 1, the I-th variable is fixed on its lower bound
!  IF X_status( I ) = 2, the I-th variable is fixed on its upper bound
!  IF X_status( I ) = 3, the I-th variable is permanently fixed
!  IF X_status( I ) = 4, the I-th variable is fixed at some other value

!  The addresses of the free variables are given in the first n_free entries
!  of the array NZ_d

!  At the initial point, variables within feas_tol of their bounds and
!  for which the search direction d_s points out of the box will be fixed

!  Based on CAUCHY_get_exact_gcp from LANCELOT B

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  m      (INTEGER) the number of rows of A
!  n      (INTEGER) the number of columns of A = number of independent variables
!  weight  (REAL) the positive regularization weight (<= zero is zero)
!  X_l, X_u (REAL arrays) the lower and upper bounds on the variables
!  bnd_inf (REAL) any BND larger than bnd_inf in modulus is infinite
!          ** this variable is not altered by the subroutine
!  X_s     (REAL array of length at least n) the point x_s from which
!           the search arc commences
!  R_s     (REAL array of length at least m) the residual A x_s - b
!  D_s     (REAL array of length at least n) the arc vector d_s
!  feas_tol (REAL) a tolerance on allowed infeasibility of x_s
!  alpha_max (REAL) the largest arc length permitted
!  max_segments  (INTEGER) the maximum number of segments to be investigated
!  out    (INTEGER) the fortran output channel number to be used
!  print_level (INTEGER) allows detailed printing. If print_level is larger
!          than 4, detailed output from the routine will be given. Otherwise,
!          no output occurs
!  prefix (CHARACTER string) that is used to prefix any output line
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  X_alpha (REAL array of length at least n) the arc minimizer
!          ** this variable need not be sent on initial entry
!  f_alpha (REAL) the value of the piecewise least-squares function f(x)
!           at the arc minimizer
!          ** this variable need not be sent on initial entry
!  phi_alpha (REAL) the value of the piecewise regularized least-squares 
!           function phi(x) at the arc minimizer
!          ** this variable need not be sent on initial entry
!  alpha   (REAL) the optimal arc length
!          ** this variable need not be sent on initial entry
!  X_status (INTEGER array of length at least n) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          X_status should be set as follows:
!          If X_status( i ) = 0, the i-th variable is free
!          If X_status( i ) = 1, the i-th variable is on its lower bound
!          If X_status( i ) = 2, the i-th variable is on its upper bound
!          If X_status( i ) = 3, 4, the i-th variable is fixed at X_alpha(i)
!          ** this variable is not altered by the subroutine.
!  segment (INTEGER) the number of segments investigated
!  userdata (structure of type GALAHAD_userdata_type) that may be used to pass
!          data to and from the optional eval_* subroutines
!  alloc_status  (INTEGER) status of the most recent array (de)allocation
!  bad_alloc (CHARACTER string of length 80) that provides information
!          following an unsuccesful call
!
!  INPUT/OUTPUT arguments
!
!  status  (INTEGER) that should be set to 1 on initial input, and
!          signifies the return status from the package on output.
!          Possible output values are:
!          0 a successful exit
!          > 0 exit requiring extra information (reverse communication).
!              The requested information must be provided in the variable
!              reverse (see below) and the subroutine re-entered with
!              other variables unchanged, Requirements are
!            2 [Dense in, dense out] The components of the product p = A * v,
!              where v is stored in reverse%V, should be returned in reverse%P.
!              The argument reverse%eval_status should be set to 0 if the
!              calculation succeeds, and to a nonzero value otherwise.
!            3 [Sparse in, dense out] The components of the product
!              p = A * v, where v is stored in reverse%V, should be
!              returned in reverse%P. Only the components
!              reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end ) of
!              reverse%V are nonzero, and the remeinder may not have
!              been set. All components of reverse%P should be set.
!              The argument reverse%eval_status should be set to 0 if the
!              calculation succeeds, and to a nonzero value otherwise.
!            4 [Sparse in, sparse out] The nonzero components of the
!              product p = A * v, where v is stored in reverse%V,
!              should be returned in reverse%P. Only the components
!              reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end ) of
!              reverse%V are nonzero, and the remeinder may not have
!              been set. On return, the positions of the nonzero components of
!              p should be indicated as reverse%NZ_out( : reverse%nz_out_end ),
!              and only these components of reverse%p need be set.
!              The argument reverse%eval_status should be set to 0 if the
!              calculation succeeds, and to a nonzero value otherwise.
!          < 0 an error exit
!
!  WORKSPACE
!
!  data (structure of type BLLS_subproblem_data_type)
!
!  OPTIONAL ARGUMENTS
!
!  A_val   (REAL array of length A_ptr( n + 1 ) - 1) the values of the
!          nonzeros of A, stored by consecutive columns
!  A_row   (INTEGER array of length A_ptr( n + 1 ) - 1) the row indices
!          of the nonzeros of A, stored by consecutive columns
!  A_ptr   (INTEGER array of length n + 1) the starting positions in
!          A_val and A_row of the i-th column of A, for i = 1,...,n,
!          with A_ptr(n+1) pointin to the storage location 1 beyond
!          the last entry in A
!  eval_ASPROD subroutine that performs products with A, see the argument
!           list for BLLS_solve
!  reverse (structure of type BLLS_reverse_type) used to communicate
!           reverse communication data to and from the subroutine.
!
!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER, INTENT( IN ):: m, n, max_segments, out, print_level
      INTEGER, INTENT( INOUT ) :: segment, status
      INTEGER, INTENT( OUT ) :: alloc_status
      REAL ( KIND = wp ), INTENT( IN ):: weight, alpha_max, feas_tol, bnd_inf
      REAL ( KIND = wp ), INTENT( INOUT ):: f_alpha, phi_alpha, alpha
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
      INTEGER, DIMENSION( n ), INTENT( INOUT ) :: X_status
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_s, X_l, X_u, D_s
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: R_s
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X_alpha
      TYPE ( BLLS_subproblem_data_type ), INTENT( INOUT ) :: data
      TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( n + 1 ) :: A_ptr
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( : ) :: A_row
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: A_val
      OPTIONAL :: eval_ASPROD
      TYPE ( BLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse

!  interface blocks

      INTERFACE
        SUBROUTINE eval_ASPROD( status, userdata, V, P, NZ_in, nz_in_start,    &
                                nz_in_end, NZ_out, nz_out_end )
        USE GALAHAD_USERDATA_double
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        INTEGER, INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
        INTEGER, OPTIONAL, INTENT( IN ) :: nz_in_start, nz_in_end
        INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_out_end
        INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_in
        INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_out
        END SUBROUTINE eval_ASPROD
      END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, H, X_l, X_u, X_s, R_s, D_s, alpha_max, feas_tol, max_segments,
!      out, print_level, prefix

!  If the i-th variable is required to be fixed at its initial value, X_s(i),
!   X_status(i) must be set to 3 or more

!  local variables

      INTEGER :: i, j, k, l, ll, lu, ibreak, inform_sort, n_freed, n_zero
      REAL ( KIND = wp ) :: alpha_star, alpha_current, phi_alpha_dash_old
      REAL ( KIND = wp ) :: rts, sts, vtv, vtx, dx, s, feasep
!     REAL ( KIND = wp ) :: ptr, ptu, arth, phi_alpha_dash_old
      LOGICAL :: printi, xlower, xupper
      CHARACTER ( LEN = 80 ) :: array_name
!     REAL ( KIND = wp ), DIMENSION( m ) :: R

!  enter or re-enter the package and jump to appropriate re-entry point

      IF ( status <= 1 ) data%branch = 100

      SELECT CASE ( data%branch )
      CASE ( 100 ) ; GO TO 100  ! status = 1
      CASE ( 180 ) ; GO TO 180  ! status = 3
      CASE ( 220 ) ; GO TO 220  ! status = 4
      CASE ( 240 ) ; GO TO 240  ! status = 3
      CASE ( 260 ) ; GO TO 260  ! status = 2
      END SELECT

!  initial entry

  100 CONTINUE

!  check input parameters

      printi = print_level > 0 .AND. out > 0
      IF ( n <= 0 ) THEN
        IF ( printi ) WRITE( out, "( A, 'error - BLLS_exact_arc_search: ',     &
       &  'n = ', I0, '<= 0' )" ) prefix, n
        status = GALAHAD_error_restrictions ; GO TO 900
      END IF

!  check that it is possible to access A in some way

      data%present_a = PRESENT( A_ptr ) .AND. PRESENT( A_row ) .AND.           &
                       PRESENT( A_val )
      data%present_asprod = PRESENT( eval_ASPROD )
      data%reverse_asprod = .NOT. ( data%present_a .OR. data%present_asprod )
      IF ( data%reverse_asprod .AND. .NOT. PRESENT( reverse ) ) THEN
        status = GALAHAD_error_optional ; GO TO 900
      END IF

!  check if regularization is necessary

      data%regularization = weight > zero

!  set printing controls

      data%printp = print_level >= 3 .AND. out > 0
      data%printw = print_level >= 4 .AND. out > 0
      data%printd = print_level >= 5 .AND. out > 0
      data%printdd = print_level > 10 .AND. out > 0
!     data%printd = .TRUE.

      IF ( data%printp ) WRITE( out, "( /, A, ' ** exact arc search entered',  &
     &    ' (m = ', I0, ', n = ', I0, ') ** ' )" ) prefix, m, n

!  if required, record the numbers of free and fixed variables and  the path

!     IF ( data%printp ) THEN
      IF ( .FALSE. ) THEN
        i = COUNT( X_status == 0 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( /, A, 1X, I0, ' variables are free' )" ) prefix, i
        ELSE
          WRITE( out, "( /, A, ' 1 variable is free' )" ) prefix
        END IF
        i = COUNT( X_status == 1 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( A, 1X, I0, ' variables are on lower bounds' )" )      &
            prefix, i
        ELSE
          WRITE( out, "( A, 1X, ' 1 variable is on its lower bound' )" ) prefix
        END IF
        i = COUNT( X_status == 2 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( A, 1X, I0, ' variables are on upper bounds' )" )      &
            prefix, i
        ELSE
          WRITE( out, "( A, 1X, ' 1 variable is on its upper bound' )" ) prefix
        END IF
        i = COUNT( X_status >= 3 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( /, A, 1X, I0, ' variables are fixed' )" ) prefix, i
        ELSE
          WRITE( out, "( /, A, 1X, ' 1 variable is fixed' )" ) prefix
        END IF
      END IF

!     IF ( print_level >= 100 ) THEN
!       DO i = 1, n
!         WRITE( out, "( A, ' Var low V up D_s', I6, 4ES12.4 )" )              &
!           prefix, i, X_l( i ), X_s( i ), X_u( i ), D_s( i )
!       END DO
!     END IF

!  allocate workspace array NZ_d

      array_name = 'blls_exact_arc_search: data%NZ_d'
      CALL SPACE_resize_array( n, data%NZ_d, status, alloc_status,             &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

!  record the status of the variables

      IF ( data%printdd ) WRITE( out, "( A, '     i     X_l         X_s',      &
     & '         X_u         D_s    stat' )" ) prefix

!  count the number of free (in nbreak) and fixed variables at the base point

      data%n_break = 0 ; data%n_fixed = 0 ; n_zero = n + 1 ; n_freed = 0
      DO i = 1, n
        IF ( data%printdd ) WRITE( out, "( A, I6, 4ES12.4, I3 )" )             &
          prefix, i, X_l( i ), X_s( i ), X_u( i ), D_s( i ), X_status( i )

!  check to see whether the variable is fixed

        IF ( X_status( i ) <= 2 ) THEN
          X_status( i ) = 0
          xupper = X_u( i ) - X_s( i ) <= feas_tol
          xlower = X_s( i ) - X_l( i ) <= feas_tol

!  the variable lies between its bounds. Check to see if the search
!  direction is zero

          IF ( .NOT. ( xupper .OR. xlower ) ) THEN
            IF ( ABS( D_s( i ) ) > epsmch ) GO TO 110
            n_zero = n_zero - 1
            data%NZ_d( n_zero ) = i

!  the variable lies close to its lower bound

          ELSE
            IF ( xlower ) THEN
              IF ( D_s( i ) > epsmch ) THEN
                n_freed = n_freed + 1
                GO TO 110
              END IF
              X_status( i ) = 1

!  the variable lies close to its upper bound

            ELSE
              IF ( D_s( i ) < - epsmch ) THEN
                n_freed = n_freed + 1
                GO TO 110
              END IF
              X_status( i ) = 2
            END IF
          END IF
        END IF

!  set the search direction to zero

        data%n_fixed = data%n_fixed + 1
        CYCLE
  110   CONTINUE

!  if the variable is free, set up the pointers to the nonzeros in the vector
!  d_s ready for calculating q = H * p

        data%n_break = data%n_break + 1
        data%NZ_d( data%n_break ) = i
      END DO

!  record the number of break points

      data%nz_d_start = 1 ; data%nz_d_end = data%n_break

      IF ( data%printp ) WRITE( out, "( /, A, 1X, I0, ' variable', A,          &
     &  ' freed from ', A, ' bound', A, ', ', I0, ' variable', A, ' remain',   &
     &  A, ' fixed,', /, A, ' of which ', I0, 1X, A, ' between bounds' )" )    &
        prefix, n_freed, TRIM( STRING_pleural( n_freed ) ),                    &
        TRIM( STRING_their( n_freed ) ), TRIM( STRING_pleural( n_freed ) ),    &
        n - data%n_break, TRIM( STRING_pleural( n - data%n_break ) ),          &
        TRIM( STRING_verb_pleural( n - data%n_break ) ),                       &
        prefix, n - n_zero + 1, TRIM( STRING_are( n - n_zero + 1 ) )

!  record the values of f(x) and rho(x) at the starting point

      f_alpha = half * DOT_PRODUCT( R_s( : m ), R_s( : m ) )
      IF ( data%regularization ) THEN
        data%rho_alpha = half * DOT_PRODUCT( X_s( : n ), X_s( : n ) )
        phi_alpha = f_alpha + weight * data%rho_alpha
      ELSE
        phi_alpha = f_alpha
      END IF

!  if all of the variables are fixed, exit

      IF ( data%n_break == 0 ) THEN
        alpha = zero
        status = GALAHAD_ok ; GO TO 800
      END IF

!  allocate further workspace arrays

      array_name = 'blls_exact_arc_search: data%NZ_out'
      CALL SPACE_resize_array( m, data%NZ_out, status, alloc_status,           &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_exact_arc_search: data%P_used'
      CALL SPACE_resize_array( m, data%P_used, status, alloc_status,           &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_exact_arc_search: data%S'
      CALL SPACE_resize_array( m, data%S, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_exact_arc_search: data%P'
      CALL SPACE_resize_array( m, data%P, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_exact_arc_search: data%U'
      CALL SPACE_resize_array( m, data%U, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_exact_arc_search: data%BREAK_points'
      CALL SPACE_resize_array( n, data%BREAK_points, status, alloc_status,     &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      IF ( data%regularization ) THEN
        array_name = 'blls_exact_arc_search: data%W'
        CALL SPACE_resize_array( m, data%W, status, alloc_status,              &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( data%present_asprod ) THEN
        array_name = 'blls_exact_arc_search: data%R'
        CALL SPACE_resize_array( m, data%R, status, alloc_status,              &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( data%reverse_asprod ) THEN
        array_name = 'blls_exact_arc_search: reverse%V'
        CALL SPACE_resize_array( n, reverse%V, status, alloc_status,           &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_exact_arc_search: reverse%P'
        CALL SPACE_resize_array( m, reverse%P, status, alloc_status,           &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_exact_arc_search: reverse%NZ_in'
        CALL SPACE_resize_array( n, reverse%NZ_in, status, alloc_status,       &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_exact_arc_search: reverse%NZ_out'
        CALL SPACE_resize_array( m, reverse%NZ_out, status, alloc_status,      &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

!  find the breakpoints for the piecewise linear arc (the distances
!  to the boundary)

      DO j = data%nz_d_start, data%nz_d_end
        i = data%NZ_d( j )
        IF ( D_s( i ) > epsmch ) THEN
          IF ( X_u( i ) >= bnd_inf ) THEN
            alpha = alpha_max
          ELSE
            alpha = ( X_u( i ) - X_s( i ) ) / D_s( i )
          END IF
        ELSE
          IF ( X_l( i ) <= - bnd_inf ) THEN
            alpha = alpha_max
          ELSE
            alpha = ( X_l( i ) - X_s( i ) ) / D_s( i )
          END IF
        END IF
        data%BREAK_points( j ) = alpha
      END DO

!  order the breakpoints in increasing size using a heapsort. Build the heap

      CALL SORT_heapsort_build( data%n_break, data%BREAK_points,               &
                                inform_sort, ix = data%NZ_d )

!  compute p = A v, where v contains the non-fixed components of d_s

!  a) evaluation directly via A

      IF ( data%present_a ) THEN
        data%P( : m ) = zero
        DO k = data%nz_d_start, data%nz_d_end
          j = data%NZ_d( k )
          IF ( j <= 0 .OR. j > n ) THEN
            IF ( data%printdd ) WRITE( out, "( ' index ', I0,                  &
           &   ' out of range [1,n=', I0, '] = ', I0 )" ) j, n
          ELSE
            s = D_s( j )
            DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
              i = A_row( l )
              data%P( i ) = data%P( i ) + A_val( l ) * s
            END DO
          END IF
        END DO

!  b) evaluation via matrix-vector product call

      ELSE IF ( data%present_asprod ) THEN
        CALL eval_ASPROD( status, userdata, V = D_s, P = data%P,               &
                          NZ_in = data%NZ_d, nz_in_start = data%nz_d_start,    &
                          nz_in_end = data%nz_d_end )
        IF ( status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_evaluation ; GO TO 900
        END IF

!  c) evaluation via reverse communication

      ELSE
        reverse%nz_in_start = data%nz_d_start
        reverse%nz_in_end = data%nz_d_end
        DO k = data%nz_d_start, data%nz_d_end
          j = data%NZ_d( k )
          reverse%NZ_in( k ) = j
          reverse%V( j ) = D_s( j )
        END DO
        data%branch = 180 ; status = 3
        RETURN
      END IF

!  return following reverse communication

  180 CONTINUE
      data%nz_out_end = m
      IF ( data%reverse_asprod ) THEN
        IF ( reverse%eval_status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_evaluation ; GO TO 900
        END IF
        data%P( : m ) = reverse%P( : m )
      END IF

!  calculate the first derivative (f_alpha_dash) and second derivative 
!  (f_alpha_dashdash) of the univariate piecewise quadratic function at 
!  the start of the piecewise linear arc

      data%f_alpha_dash = DOT_PRODUCT( R_s( : m ), data%P( : m ) )
      data%f_alpha_dashdash = DOT_PRODUCT( data%P( : m ), data%P( : m ) )
      IF ( data%f_alpha_dashdash < h_zero ) data%f_alpha_dashdash = zero

!  calculate the first derivative (rho_alpha_dash) and second derivative 
!  (rho_alpha_dashdash) of the regularization function rho(alpha) = 
!  1/2||x(alpha)||^2 at the start of the piecewise linear arc

      IF ( data%regularization ) THEN
        data%rho_alpha_dash =                                                  &
          DOT_PRODUCT( X_s( data%NZ_d( data%nz_d_start : data%nz_d_end ) ),    &
                       D_s( data%NZ_d( data%nz_d_start : data%nz_d_end ) ) )
        data%rho_alpha_dashdash =                                              &
          DOT_PRODUCT( D_s( data%NZ_d( data%nz_d_start : data%nz_d_end ) ),    &
                       D_s( data%NZ_d( data%nz_d_start : data%nz_d_end ) ) )

!  record the derivatives of phi

        data%phi_alpha_dash = data%f_alpha_dash + weight * data%rho_alpha_dash
        data%phi_alpha_dashdash =                                              &
          data%f_alpha_dashdash + weight * data%rho_alpha_dashdash
      ELSE
        data%phi_alpha_dash = data%f_alpha_dash
        data%phi_alpha_dashdash = data%f_alpha_dashdash
      END IF

!  print the search direction, if required

      IF ( data%printdd ) WRITE( out,                                          &
        "( A, ' Current search direction ', /, ( 6X, 4( I6, ES12.4 ) ) )" )    &
         prefix, ( data%NZ_d( i ), D_s( data%NZ_d( i ) ),                      &
                     i = data%nz_d_start, data%nz_d_end )

!  initialize alpha, u and s

      data%alpha_next = zero
      data%U( : m ) = zero
      data%S( : m ) = data%P( : m )

!  flag nonzero P components in P_used

      data%P_used( : m ) = 0

!  ---------
!  main loop
!  ---------

      segment = 0

!  start the main loop to find the first local minimizer of the piecewise
!  quadratic function. Consider the problem over successive pieces

  200 CONTINUE ! mock breakpoint loop

!  print details of the piecewise quadratic in the next interval

        IF ( data%printw .OR. ( data%printp .AND. segment == 0 ) )             &
          WRITE( out, 2000 ) prefix
        IF ( data%printp ) WRITE( out, "( A, 2I7, ES16.8, 3ES12.4 )" ) prefix, &
            segment, data%n_fixed, phi_alpha, data%phi_alpha_dash,             &
            data%phi_alpha_dashdash, data%alpha_next
        IF ( data%printw ) WRITE( out,                                         &
          "( /, A, ' Piece', I5, ': f, G & H at start point', 4ES11.3 )" )     &
              prefix, segment, phi_alpha, data%phi_alpha_dash,                 &
              data%phi_alpha_dashdash, data%alpha_next

        data%n_fixed = 0
        segment = segment + 1

!  if the gradient of the univariate function increases, exit

        IF ( data%phi_alpha_dash > - g_zero ) THEN
          alpha = data%alpha_next
          status = GALAHAD_ok ; GO TO 800
        END IF

!  exit if the segment limit has been exceeded

        IF ( max_segments >= 0 .AND. segment > max_segments ) THEN
          alpha = data%alpha_next
          status = GALAHAD_error_max_inner_its ; GO TO 800
        END IF

!  record the value of the last breakpoint

        alpha_current = data%alpha_next

!  find the next breakpoint ( end of the segment )

        data%alpha_next = data%BREAK_points( 1 )
        CALL SORT_heapsort_smallest( data%n_break, data%BREAK_points,          &
                                     inform_sort, ix = data%NZ_d )

!  compute the length of the current segment

        data%delta_alpha = MIN( data%alpha_next, alpha_max ) - alpha_current

!  print details of the breakpoint

        IF ( data%printw ) THEN
          WRITE( out, "( /, A, ' Next break point =', ES11.4, /, A,            &
         &  ' Maximum step     =', ES11.4 )" ) prefix, data%alpha_next,        &
            prefix, alpha_max
        END IF

!  if the gradient of the univariate function is small and its curvature
!  is positive, exit

        IF ( ABS( data%phi_alpha_dash ) <= g_zero ) THEN
          IF ( data%delta_alpha >= HUGE( one ) ) THEN
            alpha = alpha_current
            status = GALAHAD_ok ; GO TO 800
          ELSE
            alpha = data%alpha_next
          END IF

!  if the gradient of the univariate function is nonzero and its curvature is
!  positive, compute the line minimum

        ELSE
          IF ( data%phi_alpha_dashdash > zero ) THEN
            alpha_star = - data%phi_alpha_dash / data%phi_alpha_dashdash
            IF ( data%printw ) WRITE( out, "( A, ' Stationary point =',        &
           &      ES11.4 )" ) prefix, alpha_current + alpha_star

!  if the line minimum occurs before the breakpoint, the line minimum gives
!  the arc minimizer. Exit

            alpha = MIN( alpha_current + alpha_star, data%alpha_next )
            IF ( alpha_star < data%delta_alpha ) THEN
              data%delta_alpha = alpha_star
              status = GALAHAD_ok ; GO TO 700
            END IF
          ELSE
            alpha = data%alpha_next
          END IF
        END IF

!  if the arc minimizer occurs at alpha_max, exit.

        IF ( alpha_current >= alpha_max ) THEN
          alpha = alpha_max
          data%delta_alpha = alpha_max - alpha_current
          status = GALAHAD_error_primal_infeasible ; GO TO 700
        END IF

!  update the univariate function values f, rho and phi

        f_alpha = f_alpha + data%delta_alpha *                                 &
         ( data%f_alpha_dash + half * data%delta_alpha * data%f_alpha_dashdash )
        IF ( data%regularization ) THEN
          data%rho_alpha = data%rho_alpha +                                    &
                             data%delta_alpha * ( data%rho_alpha_dash +        &
                             half * data%delta_alpha * data%rho_alpha_dashdash )
          phi_alpha = f_alpha + weight * data%rho_alpha
        ELSE
          phi_alpha = f_alpha
        END IF

!  update w

        IF ( data%regularization ) THEN
          IF ( segment > 1 ) THEN
            DO k = data%nz_d_start, data%nz_d_end
              j = data%NZ_d( k )
              data%W( j ) = data%W( j ) + alpha_current * D_s( j )
            END DO
          ELSE
            data%W( : n ) = zero
          END IF
        END IF

!  update u and reset p to zero

        IF ( segment > 1 ) THEN
          DO j = 1, data%nz_out_end
            i = data%NZ_out( j )
            data%U( i ) = data%U( i ) + alpha_current * data%P( i )
            data%P( i ) = zero
          END DO
        ELSE
!write(6,*) ' segment = 1, alpha_current = ', alpha_current
!          data%U( : m ) = alpha_current * data%P( : m )
          data%U( : m ) = zero
          data%P( : m ) = zero
        END IF

!  record the new breakpoint and the amount by which other breakpoints
!  are allowed to vary from this one and still be considered to be
!  within the same cluster

        feasep = data%alpha_next + epstl2

!  move the appropriate variable(s) to their bound(s)

        DO
          data%n_fixed = data%n_fixed + 1
          ibreak = data%NZ_d( data%n_break )
          IF ( data%printw )                                                   &
            write( out, "( ' variable ', I0, ' reaches a bound' )" ) ibreak
          IF ( data%printd ) WRITE( out, "( A, ' Variable ', I0,               &
         &  ' is fixed, step =', ES12.4 )" ) prefix, ibreak, data%alpha_next

!  indicate the status of the newly-fixed variable

          IF ( D_s( ibreak ) < zero ) THEN
            X_status( ibreak ) = 1
          ELSE
            X_status( ibreak ) = 2
          END IF

!  if all of the remaining search direction is zero, return

          data%n_break = data%n_break - 1
          IF ( data%n_break == 0 ) THEN
            alpha = alpha_current
            status = GALAHAD_ok ; GO TO 800
          END IF

!  determine if other variables hit their bounds at the breakpoint

          IF (  data%BREAK_points( 1 ) >= feasep  ) EXIT
          CALL SORT_heapsort_smallest( data%n_break, data%BREAK_points,        &
                                       inform_sort, ix = data%NZ_d )
        END DO

!  update alpha

        alpha = alpha_current + data%delta_alpha

!  compute p = A v, where v contains the components of d_s that have been
!  fixed in the current segment. The nonzeros of p are in positions
!  NZ_out(1:nz_out_end). p_used(i) = segment if and only if variable i
!  is fixed in that segment

!  a) evaluation directly via A

        data%nz_d_start = data%n_break + 1
        IF ( data%present_a ) THEN
          data%nz_out_end = 0
          DO k = data%nz_d_start, data%nz_d_end
            j = data%NZ_d( k )
            IF ( j <= 0 .OR. j > n ) THEN
              IF ( data%printdd ) WRITE( out, "( ' index ', I0,                &
             &   ' out of range [1,n=', I0, '] = ', I0 )" ) j, n
            ELSE
              s = D_s( j )
              DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
                i = A_row( l )
                IF ( data%P_used( i ) < segment ) THEN
                  data%nz_out_end = data%nz_out_end + 1
                  data%NZ_out( data%nz_out_end ) = i
                  data%P_used( i ) = segment
                END IF
                data%P( i ) = data%P( i ) + A_val( l ) * s
              END DO
            END IF
          END DO

!  b) evaluation via matrix-vector product call

        ELSE IF ( data%present_asprod ) THEN
          CALL eval_ASPROD( status, userdata, V = D_s, P = data%P,             &
                            NZ_in = data%NZ_d, nz_in_start = data%nz_d_start,  &
                            nz_in_end = data%nz_d_end, NZ_out = data%NZ_out,   &
                            nz_out_end = data%nz_out_end )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%nz_in_start = data%nz_d_start
          reverse%nz_in_end = data%nz_d_end
          DO k = data%nz_d_start, data%nz_d_end
            j = data%NZ_d( k )
            reverse%NZ_in( k ) = j
            reverse%V( j ) = D_s( j )
          END DO
          data%branch = 220 ; status = 4
          RETURN
        END IF

!  return following reverse communication

  220   CONTINUE
        IF ( data%reverse_asprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%nz_out_end = reverse%nz_out_end
          data%NZ_out( : data%nz_out_end )                                     &
            = reverse%NZ_out( : reverse%nz_out_end )
          DO k = 1, data%nz_out_end
            i = data%NZ_out( k )
            data%P( i ) = reverse%P( i )
          END DO
        END IF

!  update the first and second derivatives of f(x(alpha))

        phi_alpha_dash_old = data%phi_alpha_dash
        data%f_alpha_dash = data%f_alpha_dash                                  &
          + data%delta_alpha * data%f_alpha_dashdash
        DO j = 1, data%nz_out_end
          i = data%NZ_out( j )
          data%f_alpha_dash = data%f_alpha_dash - data%P( i )                  &
            * ( R_s( i ) + data%U( i ) + data%alpha_next * data%S( i ) )
          data%f_alpha_dashdash = data%f_alpha_dashdash                        &
            + data%P( i ) * ( data%P( i ) - two * data%S( i ) )

!  update s

          data%S( i ) = data%S( i ) - data%P( i )
        END DO

!  update the first and second derivatives of rho(x(alpha))

        IF ( data%regularization ) THEN
          data%rho_alpha_dash = data%rho_alpha_dash                            &
            + data%delta_alpha * data%rho_alpha_dashdash
          DO k = data%nz_d_start, data%nz_d_end
            j = data%NZ_d( k )
            data%rho_alpha_dash = data%rho_alpha_dash - D_s( j )               &
              * ( X_s( j ) + data%W( j ) + data%alpha_next * D_s( j ) )
            data%rho_alpha_dashdash = data%rho_alpha_dashdash - D_s( j ) ** 2
          END DO

!  compute the first and second derivatives of phi(x(alpha))

          data%phi_alpha_dash = data%f_alpha_dash +                            &
                                  weight * data%rho_alpha_dash
          data%phi_alpha_dashdash = data%f_alpha_dashdash +                    &
                                      weight * data%rho_alpha_dashdash
        ELSE
          data%phi_alpha_dash = data%f_alpha_dash
          data%phi_alpha_dashdash = data%f_alpha_dashdash
        END IF

!  reset the number of free variables

        data%nz_d_start = 1 ; data%nz_d_end = data%n_break

!  check that the size of the line gradient has not shrunk significantly in
!  the current segment of the piecewise arc. If it has, there may be a loss
!  of accuracy, so the line derivatives will be recomputed

        data%recompute = ABS( data%phi_alpha_dash )                            &
                             < - SQRT( epsmch ) * phi_alpha_dash_old .OR.      &
                               data%phi_alpha_dashdash <= zero .OR. data%printw

!  Compute s = A v, where v are the components of d_s that have not been fixed

        IF ( data%recompute ) THEN

!  a) evaluation directly via A

          IF ( data%present_a ) THEN
            data%S( : m ) = zero
            DO k = data%nz_d_start, data%nz_d_end
              j = data%NZ_d( k )
              IF ( j <= 0 .OR. j > n ) THEN
                IF ( data%printdd ) WRITE( out, "( ' index ', I0,              &
               &   ' out of range [1,n=', I0, '] = ', I0 )" ) j, n
              ELSE
                s = D_s( j )
                DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
                  i = A_row( l )
                  data%S( i ) = data%S( i ) + A_val( l ) * s
                END DO
              END IF
            END DO

!  b) evaluation via matrix-vector product call

          ELSE IF ( data%present_asprod ) THEN
            CALL eval_ASPROD( status, userdata, V = D_s, P = data%S,           &
                              NZ_in = data%NZ_d, nz_in_start = data%nz_d_start,     &
                              nz_in_end = data%nz_d_end )
            IF ( status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF

!  c) evaluation via reverse communication

          ELSE
            reverse%nz_in_start = data%nz_d_start
            reverse%nz_in_end = data%nz_d_end
            DO k = data%nz_d_start, data%nz_d_end
              j = data%NZ_d( k )
              reverse%NZ_in( k ) = j
              reverse%V( j ) = D_s( j )
            END DO
            data%branch = 240 ; status = 3
            RETURN
          END IF
        END IF

!  return following reverse communication

  240   CONTINUE
        IF ( data%recompute ) THEN
          IF ( data%reverse_asprod ) THEN
            IF ( reverse%eval_status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%S( : m ) = reverse%P( : m )
          END IF

!  compute sts = s^T s

          sts = DOT_PRODUCT( data%S( : m ), data%S( : m ) )

!  compute rts = r^T s, where r = A ( x_i+1 - x_s ) + r_s, and
!  x_i+1 = Proj(x_s + alpha_i+1 d_s)

!  a) evaluation directly via A

          IF ( data%present_a ) THEN
            rts = DOT_PRODUCT( R_s, data%S( : m ) )
            DO j = 1, n
              ll = A_ptr( j ) ; lu = A_ptr( j + 1 ) - 1
              IF ( ll <= lu ) THEN
                s = DOT_PRODUCT( data%S( A_row( ll : lu ) ), A_val( ll : lu ) )
                dx = MAX( X_l( j ), MIN( X_u( j ),                             &
                        X_s( j ) + data%alpha_next * D_s( j ) ) ) - X_s( j )
                rts = rts + s * dx
              END IF
            END DO

!  b) evaluation via matrix-vector product call

          ELSE IF ( data%present_asprod ) THEN
            X_alpha = MAX( X_l, MIN( X_u, X_s + data%alpha_next * D_s ) ) - X_s
!           data%R = R_s
!           DO j = 1, n
!             dx = MAX( X_l( j ), MIN( X_u( j ),                               &
!                     X_s( j ) + data%alpha_next * D_s( j ) ) ) - X_s( j )
!             DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
!               i = A_row( l )
!               R( i ) = R( i ) + A_val( l ) * dx
!             END DO
!           END DO
            CALL eval_ASPROD( status, userdata, V = X_alpha, P = data%R )
            IF ( status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%R = data%R + R_s
            rts = DOT_PRODUCT( data%R( : m ), data%S( : m ) )

!  c) evaluation via reverse communication

          ELSE
            reverse%V( : n )                                                   &
              = MAX( X_l, MIN( X_u, X_s + data%alpha_next * D_s ) ) - X_s
            data%branch = 260 ; status = 2
            RETURN
          END IF
        END IF

!  return following reverse communication

  260   CONTINUE
        IF ( data%recompute ) THEN
          IF ( data%reverse_asprod ) THEN
            IF ( reverse%eval_status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            reverse%P( : m ) = reverse%P( : m ) + R_s
            rts = DOT_PRODUCT( reverse%P( : m ), data%S( : m ) )
          END IF

!  restore p to zero

!         data%P( : m ) = zero

          IF ( data%printw ) WRITE( out, "(                                    &
         &  /, A, ' Calculated f'' and f''''(alpha) =', 2ES22.14,              &
         &  /, A, ' Recurred   f'' and f''''(alpha) =', 2ES22.14 )" ) prefix,  &
             rts, sts, prefix, data%f_alpha_dash, data%f_alpha_dashdash

!  use the newly-computed derivatives of f

          data%f_alpha_dash = rts ; data%f_alpha_dashdash = sts

!  compute vtv = v^T v and vtx = v^T x_i+1, where v are the components of d_s 
!  that have not been fixed and x_i+1 = Proj(x_s + alpha_i+1 d_s)

          IF ( data%regularization ) THEN
            vtv = zero ; vtx = zero
            DO l =  data%nz_d_start, data%nz_d_end
             j =  data%NZ_d( l )
             s =  D_s( j )
             vtv = vtv + s ** 2
             vtx = vtx + s * MAX( X_l( j ), MIN( X_u( j ),                     &
                                  X_s( j ) + data%alpha_next * D_s( j ) ) )
            END DO

            IF ( data%printw ) WRITE( out, "(                                  &
           &  /, A, ' Calculated rho'' and rho''''(alpha) =', 2ES22.14,        &
           &  /, A, ' Recurred   rho'' and rho''''(alpha) =', 2ES22.14 )" )    &
              prefix, vtx, vtv, prefix, data%rho_alpha_dash,                   &
              data%rho_alpha_dashdash

!  use the newly-computed derivatives of rho

            data%rho_alpha_dash = vtx ; data%rho_alpha_dashdash = vtv

!  record the derivatives of phi

            data%phi_alpha_dash =                                              &
              data%f_alpha_dash + weight * data%rho_alpha_dash
            data%phi_alpha_dashdash =                                          &
              data%f_alpha_dashdash + weight * data%rho_alpha_dashdash
          ELSE
            data%phi_alpha_dash = data%f_alpha_dash
            data%phi_alpha_dashdash = data%f_alpha_dashdash
          END IF
        END IF

!  jump back to calculate the next breakpoint

      GO TO 200 ! end of mock breakpoint loop

!  ----------------
!  end of main loop
!  ----------------

!  the arc minimizer has been found

  700 CONTINUE

!  calculate the function value for the piecewise quadratic

      f_alpha = f_alpha + data%delta_alpha * ( data%f_alpha_dash               &
                    + half * data%delta_alpha * data%f_alpha_dashdash )
      IF ( data%regularization ) THEN
        data%rho_alpha =                                                       &
          data%rho_alpha + data%delta_alpha * ( data%rho_alpha_dash            &
                           + half * data%delta_alpha * data%rho_alpha_dashdash )
        phi_alpha = f_alpha + weight * data%rho_alpha
      ELSE
        phi_alpha = f_alpha
      END IF
      IF ( data%printw ) WRITE( out, 2000 ) prefix
      IF ( data%printp ) WRITE( out, "( A, 2I7, ES16.8, 3ES12.4 )" ) prefix,   &
        segment, data%n_fixed, phi_alpha, zero, data%phi_alpha_dashdash, alpha
      IF ( data%printp ) WRITE( out, "( /, A,                                  &
     &  ' Function value at the arc minimizer', ES22.14 )" ) prefix, phi_alpha

!  record the value of the arc minimizer

  800 CONTINUE
      X_alpha = MAX( X_l, MIN( X_u, X_s + alpha * D_s ) )
      RETURN

!  error returns

  900 CONTINUE
      RETURN

!  non-executable statement

 2000 FORMAT( /, A, ' segment fixed    objective       gradient   curvature',  &
              '     step' )

!  End of subroutine BLLS_exact_arc_search

      END SUBROUTINE BLLS_exact_arc_search

! -*-  B L L S _ I N E X A C T _ A R C _ S E A R C H   S U B R O U T I N E  -*-

      SUBROUTINE BLLS_inexact_arc_search( m, n, weight, X_l, X_u, bnd_inf,     &
                                          X_s, R_s, D_s, X_status,             &
                                          feas_tol, alpha_max, alpha_0,        &
                                          beta, eta, max_steps, advance,       &
                                          out, print_level, prefix,            &
                                          X_alpha, f_alpha, phi_alpha, alpha,  &
                                          steps, data, userdata, status,       &
                                          alloc_status, bad_alloc,             &
                                          A_ptr, A_row, A_val, eval_ASPROD,    &
                                          reverse, B )

!  Find an approximation to the arc minimizer in the direction d_s from x_s
!  of the regularized least-squares objective function

!    phi(x) = f(x) + weight * rho(x), where
!      f(x) = 1/2 || A x - b ||_2^2 and 
!      rho(x) = 1/2 || x ||^2,

!  within the feasible "box" x_l <= x <= x_u

!  Define the arc x(alpha) = projection of x_s + alpha * d_s into the
!  feasible box. The approximation to the arc minimizer we seek is a
!  point x(alpha_i) for which the Armijo condition

!      phi(x(alpha_i)) <= linear(x(alpha_i),eta)
!                      = f(x_s) + eta * nabla f(x_s)^T (x(alpha_i) - x_s)   (*)

!  where alpha_i = \alpha_0 * beta^i for some integer i is satisfied

!  Proceed as follows:

!  1) if the minimizer of phi(x) along x_s + alpha * d_s lies on the search arc,
!     this is the required point. Otherwise,

!  2) from some specified alpha_0, check whether (*) is satisfied with i = 0.

!  If so (optionally - alternatively simply pick x(\alpha_0))

!  2a) construct an increasing sequence alpha_i = \alpha_0 * beta^i for i < 0
!     and pick the one before (*) is violated

!  Otherwise

!  2b) construct a decreasing sequence alpha_i = \alpha_0 * beta^i for i > 0
!     and pick the first for which (*) is satified

!  Progress through the routine is controlled by the parameter status

!  If status = 0, the approximate minimizer has been found
!  If status = 1, an initial entry has been made
!  If status = 2 the vector HP = H * P is required

!  The value of the array X_status gives the status of the variables

!  IF X_status( I ) = 0, the I-th variable is free
!  IF X_status( I ) = 1, the I-th variable is fixed on its lower bound
!  IF X_status( I ) = 2, the I-th variable is fixed on its upper bound
!  IF X_status( I ) = 3, the I-th variable is permanently fixed
!  IF X_status( I ) = 4, the I-th variable is fixed at some other value

!  The addresses of the free variables are given in the first n_free entries
!  of the array NZ_d

!  At the initial point, variables within feas_tol of their bounds and
!  for which the search direction d_s points out of the box will be fixed

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  m           (INTEGER) the number of rows of A
!  n           (INTEGER) the number of columns of A=# of independent variables
!  weight      (REAL) the positive regularization weight (<= zero is zero)
!  X_l, X_u    (REAL arrays) the lower and upper bounds on the variables
!  bnd_inf     (REAL) any BND larger than bnd_inf in modulus is infinite
!  X_s         (REAL array of length at least n) the point x_s from which
!                the search arc commences
!  R_s         (REAL array of length at least m) the residual A x_s - b
!  D_s         (REAL array of length at least n) the arc vector d_s
!  feas_tol    (REAL) a tolerance on allowed infeasibility of x_s
!  alpha_0     (REAL) initial arc length
!  alpha_max   (REAL) the largest arc length permitted (alpha_max >= alpha_0)
!  beta        (REAL) arc length reduction factor in (0,1)
!  eta         (REAL) decrease tolerance in (0,1/2)
!  max_steps   (INTEGER) the maximum number of steps allowed
!  advance     (LOGICAL) allow alpha to increase as well as decrease?
!  out         (INTEGER) the fortran output channel number to be used
!  print_level (INTEGER) allows detailed printing. If print_level is larger
!                than 4, detailed output from the routine will be given.
!                Otherwise, no output occurs
!  prefix      (CHARACTER string) that is used to prefix any output line
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  X_alpha     (REAL array of length at least n) the arc minimizer
!  f_alpha     (REAL) the value of the piecewise least-squares function
!                 at the arc minimizer
!  alpha       (REAL) the optimal arc length
!  X_status    (INTEGER array of length at least n) specifies which of the
!               variables are to be fixed at the start of the minimization.
!            X_status( i ) should be set as follows:
!              = 0, the i-th variable is free
!              = 1, the i-th variable is on its lower bound
!              = 2, the i-th variable is on its upper bound
!              = 3, 4, the i-th variable is fixed at X_alpha(i)
!  steps       (INTEGER) the number of steps taken
!  userdata (structure of type GALAHAD_userdata_type) that may be used to pass
!          data to and from the optional eval_* subroutines
!  alloc_status  (INTEGER) status of the most recent array (de)allocation
!  bad_alloc (CHARACTER string of length 80) that provides information
!          following an unsuccesful call
!
!  INPUT/OUTPUT arguments
!
!  status (INTEGER) that should be set to 1 on initial input, and
!           signifies the return status from the package on output.
!           Possible output values are:
!            0 a successful exit
!          > 0 exit requiring extra information (reverse communication).
!                The requested information must be provided in the variable
!                reverse (see below) and the subroutine re-entered with
!                other variables unchanged, Requirements are
!            2 [Dense in, dense out] The components of the product p = A * v,
!              where v is stored in reverse%V, should be returned in reverse%P.
!              The argument reverse%eval_status should be set to 0 if the
!              calculation succeeds, and to a nonzero value otherwise.
!            3 [Sparse in, dense out] The components of the product
!              p = A * v, where v is stored in reverse%V, should be
!              returned in reverse%P. Only the components
!              reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end ) of
!              reverse%V are nonzero, and the remeinder may not have
!              been set. All components of reverse%P should be set.
!              The argument reverse%eval_status should be set to 0 if the
!              calculation succeeds, and to a nonzero value otherwise.
!            4 [Sparse in, sparse out] The nonzero components of the
!              product p = A * v, where v is stored in reverse%V,
!              should be returned in reverse%P. Only the components
!              reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end ) of
!              reverse%V are nonzero, and the remeinder may not have
!              been set. On return, the positions of the nonzero components of
!              p should be indicated as reverse%NZ_out( : reverse%nz_out_end ),
!              and only these components of reverse%p need be set.
!              The argument reverse%eval_status should be set to 0 if the
!              calculation succeeds, and to a nonzero value otherwise.
!          < 0 an error exit
!
!  WORKSPACE
!
!  data (structure of type BLLS_subproblem_data_type)
!
!  OPTIONAL ARGUMENTS
!
!  A_val   (REAL array of length A_ptr( n + 1 ) - 1) the values of the
!          nonzeros of A, stored by consecutive columns
!  A_row   (INTEGER array of length A_ptr( n + 1 ) - 1) the row indices
!          of the nonzeros of A, stored by consecutive columns
!  A_ptr   (INTEGER array of length n + 1) the starting positions in
!           A_val and A_row of the i-th column of A, for i = 1,...,n,
!           with A_ptr(n+1) pointin to the storage location 1 beyond
!           the last entry in A
!  weight  (REAL) the positive regularization weight (absent = zero)
!  eval_ASPROD subroutine that performs products with A
!           and its transpose, see the argument list for BLLS_solve
!  reverse (structure of type BLLS_reverse_type) used to communicate
!           reverse communication data to and from the subroutine.
!
!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER, INTENT( IN ):: m, n, out, print_level, max_steps
      INTEGER, INTENT( INOUT ) :: status
      INTEGER, INTENT( OUT ) :: alloc_status, steps
      REAL ( KIND = wp ), INTENT( IN ):: weight, alpha_0, alpha_max, eta, beta
      REAL ( KIND = wp ), INTENT( IN ):: feas_tol, bnd_inf
      REAL ( KIND = wp ), INTENT( INOUT ):: f_alpha, phi_alpha, alpha
      LOGICAL, INTENT( IN ) :: advance
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
      INTEGER, DIMENSION( n ), INTENT( INOUT ) :: X_status
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_s, X_l, X_u, D_s
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: R_s
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X_alpha
      TYPE ( BLLS_subproblem_data_type ), INTENT( INOUT ) :: data
      TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( n + 1 ) :: A_ptr
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( : ) :: A_row
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: A_val
      OPTIONAL :: eval_ASPROD
      TYPE ( BLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( m ) :: B

!  interface blocks

      INTERFACE
        SUBROUTINE eval_ASPROD( status, userdata, V, P, NZ_in, nz_in_start,    &
                                nz_in_end, NZ_out, nz_out_end )
        USE GALAHAD_USERDATA_double
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        INTEGER, INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
        INTEGER, OPTIONAL, INTENT( IN ) :: nz_in_start, nz_in_end
        INTEGER, OPTIONAL, INTENT( INOUT ) :: nz_out_end
        INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_in
        INTEGER, DIMENSION( : ), OPTIONAL, INTENT( INOUT ) :: NZ_out
        END SUBROUTINE eval_ASPROD
      END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, H, X_l, X_u, X_s, R_s, D_s, feas_tol, alpha_0, beta, eta, advance
!      out, print_level, prefix

!  If the i-th variable is required to be fixed at its initial value, X_s(i),
!   X_status(i) must be set to 3 or more

!  local variables

      INTEGER :: i, j, jj, k, l, inform_sort, base_fixed
      REAL ( KIND = wp ) :: pi, qi, rai, rfi, rsi, s, alpha_b, ds, xb, xs
      REAL ( KIND = wp ) :: xaj, dfj, rho_alpha
      LOGICAL :: printi, xlower, xupper
      CHARACTER ( LEN = 80 ) :: array_name

!  enter or re-enter the package and jump to appropriate re-entry point

      IF ( status <= 1 ) data%branch = 10

      SELECT CASE ( data%branch )
      CASE ( 10 ) ; GO TO 10       ! status = 1
      CASE ( 80 ) ; GO TO 80       ! status = 3
      CASE ( 90 ) ; GO TO 90       ! status = 3
      CASE ( 110 ) ; GO TO 110     ! status = 2
      CASE ( 310 ) ; GO TO 310     ! status = 4
      CASE ( 320 ) ; GO TO 320     ! status = 4
      CASE ( 410 ) ; GO TO 410     ! status = 4
      CASE ( 420 ) ; GO TO 420     ! status = 4
      CASE ( 610 ) ; GO TO 610     ! status = 2
      END SELECT

!  initial entry

   10 CONTINUE

!  check input parameters

      printi = print_level > 0 .AND. out > 0
      IF ( beta <= zero .OR. beta >= one ) THEN
        IF ( printi ) WRITE( out, "( A, 'error - BLLS_inexact_arc_search: ',   &
       &  'beta = ', ES10.2, ' not in (0,1)' )" ) prefix, beta
        status = GALAHAD_error_restrictions ; GO TO 900
      ELSE IF ( eta <= zero .OR. eta > half ) THEN
        IF ( printi ) WRITE( out, "( A, 'error - BLLS_inexact_arc_search: ',   &
       &  'eta = ', ES10.2, ' not in (0,1/2)]' )" ) prefix, eta
        status = GALAHAD_error_restrictions ; GO TO 900
      END IF

!  check that it is possible to access A in some way

      data%present_a = PRESENT( A_ptr ) .AND. PRESENT( A_row ) .AND.           &
                       PRESENT( A_val )
      data%present_asprod = PRESENT( eval_ASPROD )
      data%reverse_asprod = .NOT. ( data%present_a .OR. data%present_asprod )

      IF ( data%reverse_asprod .AND. .NOT. PRESENT( reverse ) ) THEN
        status = GALAHAD_error_optional ; GO TO 900
      END IF

!  check if regularization is necessary

      data%regularization = weight > zero

!  check for other optional arguments

      data%debug = PRESENT( B )

!  set printing controls

      data%printp = print_level >= 3 .AND. out > 0
      data%printw = print_level >= 4 .AND. out > 0
      data%printd = print_level >= 5 .AND. out > 0
      data%printdd = print_level > 10 .AND. out > 0
!     data%printd = .TRUE.

      IF ( data%printp ) WRITE( out, "( /, A, ' ** inexact arc search',        &
     &    ' entered (m = ', I0, ', n = ', I0, ') ** ' )" ) prefix, m, n

!  if required, record the numbers of free and fixed variables and  the path

!     IF ( data%printp ) THEN
      IF ( .FALSE. ) THEN
        i = COUNT( X_status == 0 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( /, A, 1X, I0, ' variables are free' )" ) prefix, i
        ELSE
          WRITE( out, "( /, A, ' 1 variable is free' )" ) prefix
        END IF
        i = COUNT( X_status == 1 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( A, 1X, I0, ' variables are on lower bounds' )" )      &
            prefix, i
        ELSE
          WRITE( out, "( A, 1X, ' 1 variable is on its lower bound' )" ) prefix
        END IF
        i = COUNT( X_status == 2 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( A, 1X, I0, ' variables are on upper bounds' )" )      &
            prefix, i
        ELSE
          WRITE( out, "( A, 1X, ' 1 variable is on its upper bound' )" ) prefix
        END IF
        i = COUNT( X_status >= 3 )
        IF ( i /= 1 ) THEN
          WRITE( out, "( /, A, 1X, I0, ' variables are fixed' )" ) prefix, i
        ELSE
          WRITE( out, "( /, A, 1X, ' 1 variable is fixed' )" ) prefix
        END IF
      END IF

      IF ( print_level >= 100 ) THEN
        DO i = 1, n
          WRITE( out, "( A, ' Var low V up D_s', I6, 4ES12.4 )" )              &
            prefix, i, X_l( i ), X_s( i ), X_u( i ), D_s( i )
        END DO
      END IF

!  compute the initial objective value from the input residual r_s = A x_s - b

!    f(x_s) = 1/2 || r_s||^2,

      data%f_s = half * DOT_PRODUCT( R_s, R_s )

!    phi(x_s) = 1/2 || r_s||^2 + 1/2 weight || x_s||^2,

      IF ( data%regularization ) THEN
        data%phi_s = data%f_s + half * weight * DOT_PRODUCT( X_s, X_s )
      ELSE
        data%phi_s = data%f_s
      END IF

!  allocate workspace array NZ_d that holds the components of the base-free
!  and -fixed components of d_s

      array_name = 'blls_exact_arc_search: data%NZ_d'
      CALL SPACE_resize_array( n, data%NZ_d, status, alloc_status,             &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

!  record the status of the variables

      IF ( data%printdd ) WRITE( out,                                          &
        "( A, '    j      X_l        X_s          X_u         D_s' )" ) prefix

!  count the number of free (base_free) variables at the base point,
!  and the base-fixed set

!    set_As = {j: [d]_j = 0 or
!                 [x_s]_j = [x_l]_j and [d]_j < 0 or
!                 [x_s]_j = [x_u]_j and [d]_j > 0 }

      data%base_free = 0 ; base_fixed = n + 1
      DO j = 1, n
        IF ( data%printdd ) WRITE( out, "( A, I6, 5ES12.4 )" )                 &
          prefix, j, X_l( j ), X_s( j ), X_u( j ), D_s( j )

!  check to see whether the variable is fixed

        IF ( X_status( j ) <= 2 ) THEN
          xupper = X_u( j ) - X_s( j ) <= feas_tol
          xlower = X_s( j ) - X_l( j ) <= feas_tol

!  the variable lies between its bounds. Check to see if the search
!  direction is zero

          IF ( .NOT. ( xupper .OR. xlower ) ) THEN
            IF ( ABS( D_s( j ) ) > epsmch ) GO TO 20

!  the variable lies close to its lower bound. Check if the search direction
!  points into the feasible region

          ELSE
            IF ( xlower ) THEN
              IF ( D_s( j ) > epsmch ) GO TO 20

!  the variable lies close to its upper bound. Check if the search direction
!  points into the feasible region

            ELSE
              IF ( D_s( j ) < - epsmch ) GO TO 20
            END IF
          END IF
        END IF

!  the variable is base fixed; set_As( base_fixed : n ) gives the
!  indices of the base-fixed set

        base_fixed = base_fixed - 1
        data%NZ_d( base_fixed ) = i
        X_alpha( j ) = X_s( j )
        CYCLE

!  the variable is base free; set_As( 1 : base_free ) gives the
!  indices of the base-free set.

   20   CONTINUE
        data%base_free = data%base_free + 1
        data%NZ_d( data%base_free ) = j
      END DO

!  report the number of base-free and base-fixed variables

      IF ( data%printp ) WRITE( out, "( /, A, 1X, I0, ' variable', A,          &
     &  ' free from ', A, ' bound', A, ' and ', I0, ' variable', A,            &
     &  ' remain', A, ' fixed,' )" ) prefix, data%base_free,                   &
        TRIM( STRING_pleural( data%base_free ) ),                              &
        TRIM( STRING_their( data%base_free ) ),                                &
        TRIM( STRING_pleural( data%base_free ) ), n - base_fixed + 1,          &
        TRIM( STRING_pleural( n - base_fixed + 1 ) ),                          &
        TRIM( STRING_verb_pleural( n - base_fixed + 1 ) )

!  initialise printing

      IF ( data%printp )THEN
        WRITE( out, 2000 ) prefix
        WRITE( out, "( A, I7, 1X, I7, ES16.8, ES22.14 )" )                     &
          prefix, 0, n - base_fixed + 1, zero, data%phi_s
      END IF

!  if all of the variables are fixed, exit

      IF ( data%base_free == 0 ) THEN
        alpha = zero
        status = GALAHAD_ok ; GO TO 800
      END IF

!  allocate further workspace arrays

      array_name = 'blls_inexact_arc_search: data%NZ_out'
      CALL SPACE_resize_array( m, data%NZ_out, status, alloc_status,           &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_inexact_arc_search: data%P_used'
      CALL SPACE_resize_array( m, data%P_used, status, alloc_status,           &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_inexact_arc_search: data%P'
      CALL SPACE_resize_array( m, data%P, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_inexact_arc_search: data%Q'
      CALL SPACE_resize_array( m, data%Q, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_inexact_arc_search: data%S'
      CALL SPACE_resize_array( n, data%S, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_inexact_arc_search: data%R_f'
      CALL SPACE_resize_array( m, data%R_f, status, alloc_status,              &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_inexact_arc_search: data%R_a'
      CALL SPACE_resize_array( m, data%R_a, status, alloc_status,              &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_inexact_arc_search: data%BREAK_points'
      CALL SPACE_resize_array( n, data%BREAK_points, status, alloc_status,     &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      IF ( data%regularization ) THEN
        array_name = 'blls_inexact_arc_search: data%X_a'
        CALL SPACE_resize_array( n, data%X_a, status, alloc_status,            &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_inexact_arc_search: data%D_f'
        CALL SPACE_resize_array( n, data%D_f, status, alloc_status,            &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( data%present_asprod ) THEN
        array_name = 'blls_inexact_arc_search: data%R'
        CALL SPACE_resize_array( m, data%R, status, alloc_status,              &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( data%reverse_asprod ) THEN
        array_name = 'blls_inexact_arc_search: reverse%V'
        CALL SPACE_resize_array( n, reverse%V, status, alloc_status,           &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_inexact_arc_search: reverse%P'
        CALL SPACE_resize_array( m, reverse%P, status, alloc_status,           &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_inexact_arc_search: reverse%NZ_in'
        CALL SPACE_resize_array( n, reverse%NZ_in, status, alloc_status,       &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_inexact_arc_search: reverse%NZ_out'
        CALL SPACE_resize_array( m, reverse%NZ_out, status, alloc_status,      &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( data%debug ) THEN
        array_name = 'blls_inexact_arc_search: data%X_debug'
        CALL SPACE_resize_array( n, data%X_debug, status, alloc_status,        &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900

        array_name = 'blls_inexact_arc_search: data%R_debug'
        CALL SPACE_resize_array( m, data%R_debug, status, alloc_status,        &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

!  initialize the step

      alpha = alpha_0
      data%P( : m ) = zero ; data%Q( : m ) = zero

!  find the breakpoints for the piecewise linear arc (the distances
!  to the boundary)

!                { ([x_u]_j - [x_s]_j)/[d]_j if [d_s]_j > 0
!    alpha_b_j = { ([x_l]_j - [x_s]_j)/[d]_j if [d_s]_j < 0 for j = 1 , ... , n
!                {          0                if [d_s]_j = 0

!  compute the initial search point x_0 = P_x[x_s + alpha_0 d_s]
!  and the end of the arc

!              { [x_s]_j if j in set_As
!    [x_b]_j = { [x_l]_j if j notin set_As and [d_s]_j < 0
!              { [x_u]_j if j notin set_As and [d_s]_j > 0

      DO jj = 1, data%base_free
        j = data%NZ_d( jj )
        xs = X_s( j ) ; ds = D_s( j )
        IF ( ds > epsmch ) THEN
          xb = X_u( j )
          IF ( xb >= bnd_inf ) THEN
            alpha_b = alpha_max
          ELSE
            alpha_b = ( xb - xs ) / ds
          END IF
          X_alpha( j ) = MIN( xb, xs + alpha * ds )
        ELSE
          xb = X_l( j )
          IF ( xb <= - bnd_inf ) THEN
            alpha_b = alpha_max
          ELSE
            alpha_b = ( xb - xs ) / ds
          END IF
          X_alpha( j ) = MAX( xb, xs + alpha * ds )
        END IF
        data%BREAK_points( jj ) = alpha_b

!  compute the direction to the end of the arc, s = x_b - x_s,

        data%S( j ) = xb - xs
      END DO

!  re-arrange the base-free variables so that the components (j) of those
!  whose breakpoints appear before alpha_0 (i.e., j in set_A_0) occur before
!  those whose breakpoints appear after alpha_0 (i.e., j in set_F_0).
!  calA_0 has n_a0 components

     CALL SORT_partition( data%base_free, data%BREAK_points, alpha, data%n_a0, &
                          IX = data%NZ_d )

!  determine the active and free components at x_0,

!    set_A_0 = {j : alpha_0 > alpha_b_j}
!    set_F_0 = {j not in set_A_0 : alpha_0 <= alpha_b_j }

!  initialize the active and free vectors

!    x^A_0 = x_s + s^A_0 with s^A_0 = s_{set_A_0}
!    and d^F_0 = d_s_{set_F_0}

!  as well as the scalars,

!    rhoc_0 = ||x^A_0||^2, rhol_0 = <x_s,d^F_0>, rhoq_0 = ||d^F_0||^2
!    gamma_a_0 = <s^A_0,x_s> and gamma_f_0 = <d^F_0,x_s>

      IF ( data%regularization ) THEN
        data%mu_a = zero
        data%X_a( : n ) = X_s( : n )
        DO jj = 1, data%n_a0
          j = data%NZ_d( jj )
          xs = X_s( j )  ; s = data%S( j )
          data%X_a( j ) = xs + s
          data%mu_a = data%mu_a + xs * s
        END DO
        data%rho_c = DOT_PRODUCT( data%X_a( : n ), data%X_a( : n ) )

        data%rho_l = zero ; data%rho_q = zero ; data%mu_f = zero
        data%D_f( : n ) = zero
        DO jj = data%n_a0 + 1, data%base_free
          j = data%NZ_d( jj )
          xs = X_s( j )  ; ds = D_s( j )
          data%D_f( j ) = ds
          data%rho_l = data%rho_l + xs * ds
          data%rho_q = data%rho_q + ds ** 2
          data%mu_f = data%mu_f + xs * ds
        END DO
      END IF

!  and compute the matrix-vector products

!    p_0 = A_{set_A_0} s_{set_A_0}
!    q_0 = A_{set_F_0} d_s_{set_F_0}

!  a) evaluation directly via A

      IF ( data%present_a ) THEN

! indices in set_A_0

        DO jj = 1, data%n_a0
          j = data%NZ_d( jj )
          s = data%S( j )
          DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
            i = A_row( l )
            data%P( i ) = data%P( i ) + A_val( l ) * s
          END DO
        END DO

! indices in set_F_0

        DO jj = data%n_a0 + 1, data%base_free
          j = data%NZ_d( jj )
          ds = D_s( j )
          DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
            i = A_row( l )
            data%Q( i ) = data%Q( i ) + A_val( l ) * ds
          END DO
        END DO

!  b) evaluation via matrix-vector product call

      ELSE IF ( data%present_asprod ) THEN

! indices in set_A_0

        IF ( data%n_a0 >= 1 ) THEN
          CALL eval_ASPROD( status, userdata, V = data%S, P = data%P,          &
                            NZ_in = data%NZ_d, nz_in_start = 1,                &
                            nz_in_end = data%n_a0 )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
        END IF

! indices in set_F_0

        IF ( data%base_free >= data%n_a0 + 1 ) THEN
          CALL eval_ASPROD( status, userdata, V = D_s, P = data%Q,             &
                            NZ_in = data%NZ_d, nz_in_start = data%n_a0 + 1,    &
                            nz_in_end = data%base_free )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
        END IF

!  c) evaluation via reverse communication

      ELSE IF ( data%reverse_asprod ) THEN

! indices in set_A_0

        IF ( data%n_a0 >= 1 ) THEN
          reverse%nz_in_start = 1 ; reverse%nz_in_end = data%n_a0
          DO k = reverse%nz_in_start, reverse%nz_in_end
            j = data%NZ_d( k )
            reverse%NZ_in( k ) = j
            reverse%V( j ) = data%S( j )
            END DO
          data%branch = 80 ; status = 3
          RETURN
        END IF
      END IF

!  return from reverse communication

   80 CONTINUE
      IF ( data%reverse_asprod ) THEN
        IF ( data%n_a0 >= 1 ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%P( : m ) = reverse%P( : m )
        ELSE
          data%P( : m ) = zero
        END IF

! indices in set_F_0

        IF ( data%base_free >= data%n_a0 + 1 ) THEN
          reverse%nz_in_start = data%n_a0 + 1
          reverse%nz_in_end = data%base_free
          DO k = reverse%nz_in_start, reverse%nz_in_end
            j = data%NZ_d( k )
            reverse%NZ_in( k ) = j
            reverse%V( j ) = D_s( j )
            END DO
          data%branch = 90 ; status = 3
          RETURN
        END IF
      END IF

!  return from reverse communication

   90 CONTINUE
      IF ( data%reverse_asprod ) THEN
        IF ( data%base_free >= data%n_a0 + 1 ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%Q( : m ) = reverse%P( : m )
        ELSE
          data%Q( : m ) = zero
        END IF
      END IF

!  compute the corresponding residuals

!    rA_0 = r_s + p_0 and rF_0 = q_0

      data%R_a( : m ) = R_s( : m ) + data%P( : m )
      data%R_f( : m ) = data%Q( : m )

!  initialize

!    fc_0 = ||rA_0||^2, fl_0 = <rA_0,rF_0> and fq_0 = ||rF_0||^2

      data%f_c = DOT_PRODUCT( data%R_a( : m ), data%R_a( : m ) )
      data%f_l = DOT_PRODUCT( data%R_a( : m ), data%R_f( : m ) )
      data%f_q = DOT_PRODUCT( data%R_f( : m ), data%R_f( : m ) )

!  as well as

!    gamma_a_0 = <p_0,r_s> and  gamma_f_0 = <q_0,r_s>

      data%gamma_a = DOT_PRODUCT( data%P( : m ), R_s( : m ) )
      data%gamma_f = DOT_PRODUCT( data%Q( : m ), R_s( : m ) )

!  set i = 0 (record i in step)

      data%step = 1
      data%n_break = data%n_a0

!  re-initialize p and q

      data%P( : m ) = zero ; data%Q( : m ) = zero

!  flag nonzero p components in P_used

      data%P_used( : m ) = 0

!  ---------
!  main loop
!  ---------

      data%direction = ' '

!  --------------------------- backtracking steps ----------------------------

!  ** Step 1. Check for an approximate arc minimizer

  100 CONTINUE  ! mock backtracking loop

!  compute

!    f_i = 1/2 fc_i + alpha_i fl_i + 1/2 alpha_i^2 fq_i
!    gamma_i = gamma_a_i + alpha_i gamma_f_1.

        f_alpha =                                                              &
          half * data%f_c + alpha * ( data%f_l + half * alpha * data%f_q )
        data%gamma = data%gamma_a + alpha * data%gamma_f

!  additionally, if there is a regularization term, compute

!    rho_{i+1} = 1/2 rhoc_{i+1} + alpha_{i+1} rhol_{i+1} + 
!                1/2 alpha_{i+1}^2 rhq_{i+1}
!    mu_{i+1} = mu_a_{i+1} + alpha_{i+1} mu_f_{1+1}.

        IF ( data%regularization ) THEN
          rho_alpha = half * data%rho_c +                                      &
             alpha * ( data%rho_l + half * alpha * data%rho_q )
          phi_alpha = f_alpha + weight * rho_alpha
          data%target = data%phi_s + eta * ( data%gamma +                      &
                          weight * ( data%mu_a + alpha * data%mu_f ) )
        ELSE
          phi_alpha = f_alpha
          data%target = data%f_s + eta * data%gamma
        END IF

!  for debugging, recompute the objective value

        IF ( data%debug .AND. data%printp ) THEN
          data%X_debug( : n ) = MAX( X_l, MIN( X_u, X_s + alpha * D_s ) )

!  a) evaluation directly via A

          IF ( data%present_a ) THEN
            data%R_debug( : m ) = - B
            DO j = 1, n
              s = data%X_debug( j )
              DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
                i = A_row( l )
                data%R_debug( i ) = data%R_debug( i ) + A_val( l ) * s
              END DO
            END DO

!  b) evaluation via matrix-vector product call

          ELSE IF ( data%present_asprod ) THEN
            CALL eval_ASPROD( status, userdata, V = data%X_debug,              &
                              P = data%R_debug )
            IF ( status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%R_debug( : m ) = data%R_debug( : m ) - B

!  c) evaluation via reverse communication

          ELSE
            reverse%V( : n ) = data%X_debug( : n )
            data%branch = 110 ; status = 2
            RETURN
          END IF
        END IF

!  return from reverse communication

  110   CONTINUE
        IF ( data%debug .AND. data%printp ) THEN
          IF ( data%reverse_asprod ) THEN
            IF ( reverse%eval_status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%R_debug( : m ) = reverse%P( : m ) - B
          END IF
!         WRITE(  out, "( A, 25X, ' true ', 2ES22.14 ) ")                      &
!           prefix, half * DOT_PRODUCT( data%R_debug, data%R_debug ), data%gamma
        END IF

!  print details of the current step

        IF ( data%printp ) WRITE( out, "( A, I7, A1, I7, ES16.8, 2ES22.14 )" ) &
!         prefix, data%step, data%n_a0, alpha, f_alpha, data%target
          prefix, data%step, data%direction, data%n_break, alpha,              &
          phi_alpha, data%target

!  if f_i <= f(x_s) + eta gamma_i, the stepsize alpha is acceptable

        IF ( phi_alpha <= data%target ) THEN

!  if this occurs for the initial step and advancing steps are to be
!  investigated, order the breakpoints beyond alpha_0 by increasing size
!  using a heapsort and continue with Step 4

          IF ( data%step == 1 .AND. advance ) THEN
            data%n_break = data%base_free - data%n_a0
            CALL SORT_heapsort_build( data%n_break,                            &
                data%BREAK_points( data%n_a0 + 1 : data%base_free ) ,          &
                inform_sort,                                                   &
                ix = data%NZ_d( data%n_a0 + 1 : data%base_free ) )
            data%direction = 'F'
            GO TO 400

!  otherwise, i.e., if i > 0 or advancing steps are not permitted, set

!    alpha_c = alpha_i, x_c = P_X[x_s + alpha_c d], f(x_c) = f_i and
!    phi(x_c) = phi_i

!  and stop

          ELSE
            status = GALAHAD_ok ; GO TO 800
          END IF

!  if f_i > f(x_s) + eta gamma_i, the stepsize alpha is not acceptable

        ELSE

!  if this is for the initial step, order the breakpoints before alpha_0
!  by decreasing size using a heapsort

          IF ( data%step == 1 ) THEN
            data%n_break = data%n_a0
            CALL SORT_heapsort_build( data%n_break, data%BREAK_points,         &
                                      inform_sort, ix = data%NZ_d,             &
                                      largest = .TRUE. )
            data%direction = 'B'
          END IF
        END IF

!  exit if the step limit has been exceeded

        IF ( max_steps >= 0 .AND. data%step > max_steps ) THEN
          status = GALAHAD_error_max_inner_its ; GO TO 800
        END IF

!  ** Step 2. Find the next set of indices that change status

!  set alpha_{i+1} = beta alpha_i

        alpha = beta * alpha

!  compute

!    set_I_{i+1} = { j: alpha_{i+1} < alpha_b_j <= alpha_i}

!  using the Heapsort algorithm

        data%nz_out_end = 0 ; data%nz_d_end = data%n_break
        DO
          data%nz_d_start = data%n_break + 1
          IF ( data%n_break == 0 .OR. data%BREAK_points( 1 ) <= alpha ) EXIT
          CALL SORT_heapsort_smallest( data%n_break, data%BREAK_points,        &
                                       inform_sort, ix = data%NZ_d,            &
                                       largest = .TRUE. )
          j = data%NZ_d( data%n_break )
          data%n_break = data%n_break - 1

!  ** Step 3. Update the components of the objective and its slope

!  Compute

!    p_{i+1} = A_{set_I_{i+1}} s_{set_I_{i+1}}
!    q_{i+1} = A_{set_I_{i+1}} d_s_{set_I_{}+1}

!  a) evaluation directly via A

          IF ( data%present_a ) THEN
            s = data%S( j ) ; ds = D_s( j )
            DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
              i = A_row( l )
              IF ( data%P_used( i ) < data%step ) THEN
                data%nz_out_end = data%nz_out_end + 1
                data%NZ_out( data%nz_out_end ) = i
                data%P_used( i ) = data%step
                data%P( i ) = zero ; data%Q( i ) = zero
              END IF
              data%P( i ) = data%P( i ) + A_val( l ) * s
              data%Q( i ) = data%Q( i ) + A_val( l ) * ds
            END DO
          END IF
        END DO

!  b) evaluation via matrix-vector product call

        IF ( data%present_asprod ) THEN
          CALL eval_ASPROD( status, userdata, V = data%S, P = data%P,          &
                            NZ_in = data%NZ_d, nz_in_start = data%nz_d_start,  &
                            nz_in_end = data%nz_d_end, NZ_out = data%NZ_out,   &
                            nz_out_end = data%nz_out_end )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          CALL eval_ASPROD( status, userdata, V = D_s, P = data%Q,             &
                            NZ_in = data%NZ_d, nz_in_start = data%nz_d_start,  &
                            nz_in_end = data%nz_d_end, NZ_out = data%NZ_out,   &
                            nz_out_end = data%nz_out_end )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE IF ( data%reverse_asprod ) THEN
          reverse%nz_in_start = data%nz_d_start
          reverse%nz_in_end = data%nz_d_end
          DO k = reverse%nz_in_start, reverse%nz_in_end
            j = data%NZ_d( k )
            reverse%NZ_in( k ) = j
            reverse%V( j ) = data%S( j )
          END DO
          data%branch = 310 ; status = 4
          RETURN
        END IF

!  return from reverse communication

  310   CONTINUE
        IF ( data%reverse_asprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          DO j = 1, reverse%nz_out_end
            i = reverse%NZ_out( j )
            data%P( i ) = reverse%P( i )
          END DO

          DO k = reverse%nz_in_start, reverse%nz_in_end
            j = data%NZ_d( k )
            reverse%V( j ) = D_s( j )
          END DO
          data%branch = 320 ; status = 4
          RETURN
        END IF

!  return from reverse communication

  320   CONTINUE
        IF ( data%reverse_asprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%nz_out_end = reverse%nz_out_end
          DO j = 1, reverse%nz_out_end
            i = reverse%NZ_out( j ) ; data%NZ_out( j ) = i
            data%Q( i ) = reverse%P( i )
          END DO
        END IF

!  loop over the nonzero components of p (and q)

        DO j = 1, data%nz_out_end
          i = data%NZ_out( j )
          pi = data%P( i ) ; qi = data%Q( i )
!write(6,"('P, Q', 2ES12.4 )" ) pi, qi
          rai = data%R_a( i ) ; rfi = data%R_f( i ) ; rsi = R_s( i )

!  update

!    fc_{i+1} = fc_i - 2 <p_{i+1},rA_i> + ||p_{i+1}||^2,
!    fl_{i+1} = fl_i + <q_{i+1},rA_i> - <p_{i+1},rF_i> - <q_{i+1},p_{i+1>}, 
!    and fq_{i+1} = fq_i + 2 <q_{i+1},rF_i> + ||q_{i+1}||^2

          data%f_c = data%f_c - two * pi * rai + pi ** 2
          data%f_l = data%f_l + qi * rai - pi * rfi - qi * pi
          data%f_q = data%f_q + two * qi * rfi + qi ** 2

!    gamma_a_{i+1} = gamma_a_i - <p_{i+1},r_s> and
!    gamma_f_{i+1} = gamma_f_i + <q_{i+1},r_s>

          data%gamma_a = data%gamma_a - pi * rsi
          data%gamma_f = data%gamma_f + qi * rsi

!    rA_{i+1} = rA_i - p_{i+1} and rF_{i+1} = rF_i + q_{i+1}

          data%R_a( i ) = rai - pi ; data%R_f( i ) = rfi + qi
        END DO

!  now deal with the regularization term, if any

        IF ( data%regularization ) THEN

!  loop over the indices that have changed status

          DO k = data%nz_d_start, data%nz_d_end
            j = data%NZ_d( k )
            s = data%S( j ) ; ds = D_s( j )
            xaj = data%X_a( j ) ; dfj = data%D_f( j ) ; xs = X_s( j )

!  update

!    rhoc_{i+1} = rhoc_i - 2 <s_{i+1},xA_i> + ||s_{i+1}||^2,
!    rhol_{i+1} = rhol_i + <d_{i+1},xA_i> - <s_{i+1},dF_i> - <d_{i+1},s_{i+1>}
!    and rhoq_{i+1} = rhoq_i + 2 <d_{i+1},dF_i> + ||d_{i+1}||^2

            data%rho_c = data%rho_c - two * s * xaj + s ** 2
            data%rho_l = data%rho_l + ds * xaj - s * dfj - ds * s
            data%rho_q = data%rho_q + two * ds * dfj + ds ** 2

!    mu_a_{i+1} = mu_a_i - <s_{i+1},x_s> and mu_f_{i+1} = mu_f_i + <d_{i+1},x_s>

            data%mu_a = data%mu_a - s * xs
            data%mu_f = data%mu_f + ds * xs

!    xA_{i+1} = xA_i - s_{i+1} and  dF_{i+1} = dF_i + d_{i+1}

            data%X_a( j ) = xaj - s ; data%D_f( j ) = dfj + ds
          END DO
        END IF

!  increment i by 1 and return to Step 1
!
        data%step = data%step + 1
        GO TO 100  ! end of mock backtracking loop

!  ---------------------------- extending steps ----------------------------

!  Step 4. Find the next set of indices that change status

  400 CONTINUE  ! mock advancing loop

!  record the current step and function value

        data%alpha_i = alpha
        data%f_i = f_alpha
        data%phi_i = phi_alpha

!  set alpha_{i+1} = beta^{-1} alpha_i

        alpha = alpha / beta

!  compute

!    set_J_{i+1} = { j: alpha_i < alpha_b_j <= alpha_{i+1}}

!  using the Heapsort algorithm; store set_I_{i+1} in NZ_out

        data%nz_out_end = 0 ; data%nz_d_end = data%n_a0 + data%n_break
        DO
          data%nz_d_start = data%n_a0 + data%n_break + 1
          IF ( data%n_break == 0 ) EXIT
          IF ( data%BREAK_points( data%n_a0 + 1 ) > alpha ) EXIT
          CALL SORT_heapsort_smallest( data%n_break,                           &
              data%BREAK_points( data%n_a0 + 1 : data%n_a0 + data%n_break ),   &
              inform_sort,                                                     &
              ix = data%NZ_d( data%n_a0 + 1 : data%n_a0 + data%n_break ) )
          j = data%NZ_d( data%n_a0 + data%n_break )
          data%n_break = data%n_break - 1

!  ** Step 5. Update the components of the objective and its slope

!  Compute

!    p_{i+1} = A_{set_J_{i+1}} s_{set_J_{i+1}}
!    q_{i+1} = A_{set_J_{i+1}} d_s_{set_J_{}+1}

!  a) evaluation directly via A

          IF ( data%present_a ) THEN
            s = data%S( j ) ; ds = D_s( j )
            DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
              i = A_row( l )
              IF ( data%P_used( i ) < data%step ) THEN
                data%nz_out_end = data%nz_out_end + 1
                data%NZ_out( data%nz_out_end ) = i
                data%P_used( i ) = data%step
                data%P( i ) = zero ; data%Q( i ) = zero
              END IF
              data%P( i ) = data%P( i ) + A_val( l ) * s
              data%Q( i ) = data%Q( i ) + A_val( l ) * ds
            END DO
          END IF
        END DO

!  b) evaluation via matrix-vector product call

        IF ( data%present_asprod ) THEN
          CALL eval_ASPROD( status, userdata, V = data%S, P = data%P,          &
                            NZ_in = data%NZ_d, nz_in_start = data%nz_d_start,  &
                            nz_in_end = data%nz_d_end, NZ_out = data%NZ_out,   &
                            nz_out_end = data%nz_out_end )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          CALL eval_ASPROD( status, userdata, V = D_s, P = data%Q,             &
                            NZ_in = data%NZ_d, nz_in_start = data%nz_d_start,  &
                            nz_in_end = data%nz_d_end, NZ_out = data%NZ_out,   &
                            nz_out_end = data%nz_out_end )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE IF ( data%reverse_asprod ) THEN
          reverse%nz_in_start = data%nz_d_start
          reverse%nz_in_end = data%nz_d_end
          DO k = reverse%nz_in_start, reverse%nz_in_end
            j = data%NZ_d( k )
            reverse%NZ_in( k ) = j
            reverse%V( j ) = data%S( j )
          END DO
          data%branch = 410 ; status = 4
          RETURN
        END IF

!  return from reverse communication

  410   CONTINUE
        IF ( data%reverse_asprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          DO j = 1, reverse%nz_out_end
            i = reverse%NZ_out( j )
            data%P( i ) = reverse%P( i )
          END DO

          DO k = reverse%nz_in_start, reverse%nz_in_end
            j = data%NZ_d( k )
            reverse%V( j ) = D_s( j )
          END DO
          data%branch = 420 ; status = 4
          RETURN
        END IF

!  return from reverse communication

  420   CONTINUE
        IF ( data%reverse_asprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%nz_out_end = reverse%nz_out_end
          DO j = 1, reverse%nz_out_end
            i = reverse%NZ_out( j ) ; data%NZ_out( j ) = i
            data%Q( i ) = reverse%P( i )
          END DO
        END IF

!  loop over the nonzero components of p (and q)

        DO j = 1, data%nz_out_end
          i = data%NZ_out( j )
          pi = data%P( i ) ; qi = data%Q( i )
          rai = data%R_a( i ) ; rfi = data%R_f( i ) ; rsi = R_s( i )

!  update

!    fc_{i+1} = fc_i + 2 <p_{i+1},rA_i> + ||p_{i+1}||^2,
!    fl_{i+1} = fl_i  - <q_{i+1},rA_i>
!                     + <p_{i+1},rF_i> - <q_{i+1},p_{i+1>}, and
!    fq_{i+1} = fq_i - 2 <q_{i+1},rF_i> + ||q_{i+1}||^2

          data%f_c = data%f_c + two * pi * rai + pi ** 2
          data%f_l = data%f_l - qi * rai + pi * rfi - qi * pi
          data%f_q = data%f_q - two * qi * rfi + qi ** 2

!    gamma_a_{i+1} = gamma_a_i + <p_{i+1},r_s> and
!    gamma_f_{i+1} = gamma_f_i - <q_{i+1},r_s>

          data%gamma_a = data%gamma_a + pi * rsi
          data%gamma_f = data%gamma_f - qi * rsi

!    rA_{i+1} = rA_i + p_{i+1}
!    rF_{i+1} = rF_i - q_{i+1}

          data%R_a( i ) = rai + pi ; data%R_f( i ) = rfi - qi
        END DO

!  now deal with the regularization term, if any

        IF ( data%regularization ) THEN

!  loop over the indices that have changed status

          DO k = data%nz_d_start, data%nz_d_end
            j = data%NZ_d( k )
            s = data%S( j ) ; ds = D_s( j )
            xaj = data%X_a( j ) ; dfj = data%D_f( j ) ; xs = X_s( j )

!  update

!    rhoc_{i+1} = rhoc_i + 2 <s_{i+1},xA_i> + ||s_{i+1}||^2,
!    rhol_{i+1} = rhol_i - <d_{i+1},xA_i> + <s_{i+1},dF_i> - <d_{i+1},s_{i+1>}
!    and rhoq_{i+1} = rhoq_i - 2 <d_{i+1},dF_i> + ||d_{i+1}||^2

            data%rho_c = data%rho_c + two * s * xaj + s ** 2
            data%rho_l = data%rho_l - ds * xaj + s * dfj - ds * s
            data%rho_q = data%rho_q - two * ds * dfj + ds ** 2

!    mu_a_{i+1} = mu_a_i + <s_{i+1},x_s> and mu_f_{i+1} = mu_f_i - <d_{i+1},x_s>

            data%mu_a = data%mu_a + s * xs
            data%mu_f = data%mu_f - ds * xs

!    xA_{i+1} = xA_i + s_{i+1} and  dF_{i+1} = dF_i - d_{i+1}

            data%X_a( j ) = xaj + s ; data%D_f( j ) = dfj - ds
          END DO
        END IF

!  ensure that end of the arc is not exceeded

        IF ( data%n_break == 0 ) THEN
          IF ( data%n_a0 == data%base_free ) THEN
            alpha = alpha_max
          ELSE
            alpha = data%BREAK_points( data%n_a0 + 1 )
          END IF
        END IF

!  ** Step 6. Check for an approximate extended arc minimizer

!  compute

!    f_{i+1} = 1/2 fc_{i+1} + alpha_{i+1} fl_{i+1} + 1/2 alpha_{i+1}^2 fq_{i+1}
!    gamma_{i+1} = gamma_a_{i+1} + alpha_{i+1} gamma_f_{1+1}.

        f_alpha = half * data%f_c +                                            &
          alpha * ( data%f_l + half * alpha * data%f_q )
        data%gamma = data%gamma_a + alpha * data%gamma_f

!  additionally, if there is a regularization term, compute

!    rho_{i+1} = 1/2 rhoc_{i+1} + alpha_{i+1} rhol_{i+1} + 
!                1/2 alpha_{i+1}^2 rhq_{i+1}
!    mu_{i+1} = mu_a_{i+1} + alpha_{i+1} mu_f_{1+1}.

        IF ( data%regularization ) THEN
          rho_alpha = half * data%rho_c +                                      &
             alpha * ( data%rho_l + half * alpha * data%rho_q )
          phi_alpha = f_alpha + weight * rho_alpha
          data%target = data%phi_s + eta * ( data%gamma +                      &
                          weight * ( data%mu_a + alpha * data%mu_f ) )
        ELSE
          phi_alpha = f_alpha
          data%target = data%f_s + eta * data%gamma
        END IF

!  If f_{i+1} > f(x_s) + eta gamma_{i+1} or alpha_{i+1} >= max_j alpha_b_j, set

!    alpha_c = alpha_i, x_c = P_X[x_s + alpha_c d], f(x_c) = f_i and
!    phi(x_c) = phi_i

!  and stop

        IF ( phi_alpha > data%target ) THEN
          alpha = data%alpha_i ; f_alpha = data%f_i ; phi_alpha = data%phi_i
          status = GALAHAD_ok ; GO TO 800

!  Otherwise, increment i by 1 and return to Step 4

        ELSE

!  for debugging, recompute the objective value

          IF ( data%debug .AND. data%printp ) THEN
            data%X_debug( : n ) = MAX( X_l, MIN( X_u, X_s + alpha * D_s ) )

!  a) evaluation directly via A

            IF ( data%present_a ) THEN
              data%R_debug( : m ) = - B
              DO j = 1, n
                s = data%X_debug( j )
                DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
                  i = A_row( l )
                  data%R_debug( i ) = data%R_debug( i ) + A_val( l ) * s
                END DO
              END DO

!  b) evaluation via matrix-vector product call

            ELSE IF ( data%present_asprod ) THEN
              CALL eval_ASPROD( status, userdata, V = data%X_debug,            &
                                P = data%R_debug )
              IF ( status /= GALAHAD_ok ) THEN
                status = GALAHAD_error_evaluation ; GO TO 900
              END IF
              data%R_debug( : m ) = data%R_debug( : m ) - B

!  c) evaluation via reverse communication

            ELSE
              reverse%V( : n ) = data%X_debug( : n )
              data%branch = 610 ; status = 2
              RETURN
            END IF
          END IF
        END IF

!  return from reverse communication

  610   CONTINUE
        IF ( data%debug .AND. data%printp ) THEN
          IF ( data%reverse_asprod ) THEN
            IF ( reverse%eval_status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%R_debug( : m ) = reverse%P( : m ) - B
          END IF
        END IF

!  print details of the current step

        data%step = data%step + 1
!       IF ( data%printw .OR. ( data%printp .AND. data%steps == 1 ) )          &
        IF ( data%printw )                                                     &
          WRITE( out, 2000 ) prefix
        IF ( data%printp ) WRITE( out, "( A, I7, A1, I7, ES16.8, 2ES22.14 )")  &
          prefix, data%step, data%direction, data%base_free - data%n_break,    &
          alpha, phi_alpha, data%target

!  exit if the end of the arc is reached

        IF ( data%n_break == 0 ) THEN
          status = GALAHAD_ok ; GO TO 800
        END IF

!  exit if the step limit has been exceeded

        IF ( max_steps >= 0 .AND. data%step > max_steps ) THEN
          status = GALAHAD_error_max_inner_its ; GO TO 800
        END IF

        GO TO 400  ! end of mock advancing loop

!  ----------------
!  end of main loop
!  ----------------

!  record the value of the arc minimizer

  800 CONTINUE
      X_alpha = MAX( X_l, MIN( X_u, X_s + alpha * D_s ) )

!  record the status of the minimizer

      IF ( data%printdd ) WRITE( out,                                          &
        "( A, '    j      X_l        X_s          X_u' )" ) prefix
      DO j = 1, n
        IF ( data%printdd ) WRITE( out, "( A, I6, 5ES12.4 )" )                 &
          prefix, j, X_l( j ), X_alpha( j ), X_u( j ), D_s( j )

!  leave fixed variables alone

        IF ( X_status( j ) <= 2 ) THEN

!  the variable lies close to its lower bound

          IF ( X_alpha( j ) - X_l( j ) <= feas_tol ) THEN
            X_status( j ) = 1

!  the variable lies close to its upper bound

          ELSE IF ( X_u( j ) - X_alpha( j ) <= feas_tol ) THEN
            X_status( j ) = 2

!  the variable lies between its bounds

          ELSE
            X_status( j ) = 0
          END IF
        END IF
      END DO

!  record the number of steps taken

      steps = data%step
      IF ( data%printp ) WRITE( out, "( /, A,                                  &
     &  ' Function value at the arc minimizer', ES22.14 )" ) prefix, phi_alpha

      RETURN

!  error returns

  900 CONTINUE
      RETURN

!  non-executable statement

 2000 FORMAT( /, A, '   step  fixed         length        objective',          &
                    '              target' )

!  End of subroutine BLLS_inexact_arc_search

      END SUBROUTINE BLLS_inexact_arc_search

! -*-*-*-*-*-*-*-*-  B L L S _ C G L S   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

      SUBROUTINE BLLS_cgls( m, n, weight, f, X, R, FIXED,                      &
                            stop_cg_relative, stop_cg_absolute,                &
                            iter, maxit, out, print_level, prefix,             &
                            data, userdata, status, alloc_status, bad_alloc,   &
                            A_ptr, A_row, A_val, eval_AFPROD, eval_PREC,       &
                            DPREC, reverse, preconditioned, B )

!  Find the minimizer of the constrained (regularized) least-squares 
!  objective function

!    f(x) =  1/2 || A x - b ||_2^2 + 1/2 weight * ||x||_2^2

!  for which certain components of x are fixed at their input values

!  IF FIXED( I ) /= 0, the I-th variable is fixed at the input X

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  m      (INTEGER) the number of rows of A.
!  n      (INTEGER) the number of columns of A = number of independent variables
!  weight (REAL) the positive regularization weight (<= zero is zero)
!  FIXED  (INTEGER array of length at least n) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          FIXED should be set as follows:
!          If FIXED( i ) = 0, the i-th variable is free
!          IF FIXED( I ) /= 0, the I-th variable is fixed at the input X
!  stop_cg_relative, stop_cg_absolute (REAL) the iteration will stop as
!          soon as the gradient of the objective is smaller than
!          MAX( stop_cg_relative * norm initial gradient, stop_cg_absolute)
!  maxit  (INTEGER) the maximum number of iterations allowed (<0 = infinite)
!  out    (INTEGER) the fortran output channel number to be used
!  print_level (INTEGER) allows detailed printing. If print_level is larger
!          than 4, detailed output from the routine will be given. Otherwise,
!          no output occurs
!  prefix (CHARACTER string) that is used to prefix any output line
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  iter   (INTEGER) the number of iterations performed
!  userdata (structure of type GALAHAD_userdata_type) that may be used to pass
!          data to and from the optional eval_* subroutines
!  alloc_status  (INTEGER) status of the most recent array (de)allocation
!  bad_alloc (CHARACTER string of length 80) that provides information
!          following an unsuccesful call
!
!  INPUT/OUTPUT arguments
!
!  f       (REAL) the value of the objective function at X. It need not be
!           set on input, on output it will contain the objective value at X
!  X       (REAL array of length at least n) the minimizer. On input, X should
!           be set to an estimate of the minimizer, on output it will
!           contain an improved estimate
!  R       (REAL array of length at least m) the residual A x - b. On input,
!           R should be set to the residual at the input X, on output it will
!           contain the residual at the improved estimate X
!  status  (INTEGER) that should be set to 1 on initial input, and
!          signifies the return status from the package on output.
!          Possible output values are:
!          0 a successful exit
!          > 0 exit requiring extra information (reverse communication).
!              The requested information must be provided in the variable
!              reverse (see below) and the subroutine re-entered with
!              other variables unchanged, Requirements are
!            2 [Sparse in, dense out] The components of the product p = A * v,
!              where the i-th component of v is stored in reverse%V(i)
!              if FIXED(i)=0 and zero if FIXED(i)/=0, should be returned
!              in reverse%P. The argument reverse%eval_status should be set to
!              0 if the calculation succeeds, and to a nonzero value otherwise.
!            3 [Dense in, sparse out] The components of the product
!              p = A^T * v, where v is stored in reverse%V, should be
!              returned in reverse%P. Only components p_i whose for
!              which FIXED(i)=0 need be assigned, the remainder will be
!              ignored. The argument reverse%eval_status should be set to 0
!              if the calculation succeeds, and to a nonzero value otherwise.
!            4 the product p = P^-1 v between the inverse of the preconditionr
!              P and the vector v, where v is stored in reverse%V, should be
!              returned in reverse%P. Only the components of v with indices i
!              for which FIXED(i)/=0 are nonzero, and only the components of
!              p with indices i for which FIXED(i)/=0 are needed. The argument 
!              reverse%eval_status should  be set to 0 if the calculation 
!              succeeds, and to a nonzero value otherwise.
!          < 0 an error exit
!
!  WORKSPACE
!
!  data (structure of type BLLS_subproblem_data_type)
!
!  OPTIONAL ARGUMENTS
!
!  A_val   (REAL array of length A_ptr( n + 1 ) - 1) the values of the
!          nonzeros of A, stored by consecutive columns
!  A_row   (INTEGER array of length A_ptr( n + 1 ) - 1) the row indices
!          of the nonzeros of A, stored by consecutive columns
!  A_ptr   (INTEGER array of length n + 1) the starting positions in
!          A_val and A_row of the i-th column of A, for i = 1,...,n,
!          with A_ptr(n+1) pointin to the storage location 1 beyond
!          the last entry in A
!  weight  (REAL) the positive regularization weight (absent = zero)
!  eval_AFPROD subroutine that performs products with A, see the argument
!           list for BLLS_solve
!  eval_PREC subroutine that performs the preconditioning operation p = P v
!            see the argument list for BLLS_solve
!  DPREC   (REAL array of length n) the values of a diagonal preconditioner
!           that aims to approximate A^T A
!  preconditioned (LOGICAL) prsent and set true is there a preconditioner
!  reverse (structure of type BLLS_reverse_type) used to communicate
!           reverse communication data to and from the subroutine

!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER, INTENT( IN ):: m, n, maxit, out, print_level
      INTEGER, INTENT( INOUT ) :: iter, status
      INTEGER, INTENT( OUT ) :: alloc_status
      REAL ( KIND = wp ), INTENT( IN ) :: weight
      REAL ( KIND = wp ), INTENT( INOUT ) :: f
      REAL ( KIND = wp ), INTENT( IN ) :: stop_cg_relative, stop_cg_absolute
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
      INTEGER, DIMENSION( n ), INTENT( INOUT ) :: FIXED
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: R
      TYPE ( BLLS_subproblem_data_type ), INTENT( INOUT ) :: data
      TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( n + 1 ) :: A_ptr
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( : ) :: A_row
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: A_val
      OPTIONAL :: eval_AFPROD, eval_PREC
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: DPREC
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( m ) :: B
      LOGICAL, OPTIONAL, INTENT( IN ) :: preconditioned
      TYPE ( BLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse

!  interface blocks

      INTERFACE
        SUBROUTINE eval_AFPROD( status, userdata, transpose, V, P,             &
                                FREE, n_free )
        USE GALAHAD_USERDATA_double
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        INTEGER, INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        LOGICAL, INTENT( IN ) :: transpose
        INTEGER, INTENT( IN ) :: n_free
        INTEGER, INTENT( IN ), DIMENSION( : ) :: FREE
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
        END SUBROUTINE eval_AFPROD
      END INTERFACE

      INTERFACE
        SUBROUTINE eval_PREC( status, userdata, V, P )
        USE GALAHAD_USERDATA_double
        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
        INTEGER, INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: P
        END SUBROUTINE eval_PREC
      END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, H, X_l, X_u, X_s, R_s, D_s, alpha_max, feas_tol, max_segments,
!      out, print_level, prefix

!  If the i-th variable is required to be fixed at its initial value, X_s(i),
!   FIXED(i) must be set to 3 or more

!  local variables

      INTEGER :: i, j, k, l
      REAL ( KIND = wp ) :: val, alpha, beta, gamma_old, curv, norm_r, norm_g
      LOGICAL :: printi
      CHARACTER ( LEN = 80 ) :: array_name

!  enter or re-enter the package and jump to appropriate re-entry point

      IF ( status <= 1 ) data%branch = 10

      SELECT CASE ( data%branch )
      CASE ( 10 ) ; GO TO 10       ! status = 1
      CASE ( 20 ) ; GO TO 20       ! status = 3
      CASE ( 90 ) ; GO TO 90       ! status = 4
      CASE ( 110 ) ; GO TO 110     ! status = 2
      CASE ( 120 ) ; GO TO 120     ! status = 3
      CASE ( 140 ) ; GO TO 140     ! status = 4
      END SELECT

!  initial entry

   10 CONTINUE
      iter = 0

!  check input parameters

      printi = print_level > 0 .AND. out > 0

!  check that it is possible to access A in some way

      data%present_a = PRESENT( A_ptr ) .AND. PRESENT( A_row ) .AND.           &
                       PRESENT( A_val )
      data%present_afprod = PRESENT( eval_AFPROD )
      data%reverse_afprod = .NOT. ( data%present_a .OR. data%present_afprod )
      IF ( data%reverse_afprod .AND. .NOT. PRESENT( reverse ) ) THEN
        status = GALAHAD_error_optional ; GO TO 900
      END IF

!  check for the means of preconditioning, if any

      IF ( PRESENT( preconditioned ) ) THEN
        data%preconditioned = preconditioned
      ELSE
        data%preconditioned = .FALSE.
      END IF
      data%present_prec = PRESENT( eval_prec )
      data%present_dprec = PRESENT( DPREC )
      data%reverse_prec = .NOT. ( data%present_dprec .OR. data%present_prec )
      IF ( data%preconditioned .AND.  data%reverse_prec .AND.                  &
           .NOT. PRESENT( reverse ) ) THEN
        status = GALAHAD_error_optional ; GO TO 900
      END IF

!  check for other optional arguments

      data%debug = PRESENT( B )

!  check if regularization is necessary

      data%regularization = weight > zero

!  set printing controls

      data%printp = print_level >= 3 .AND. out > 0
      data%printw = print_level >= 4 .AND. out > 0
      data%printd = print_level >= 5 .AND. out > 0
      data%printdd = print_level > 10 .AND. out > 0
!     data%printd = .TRUE.

!  compute the number of free variables

      data%n_free = COUNT( FIXED( : n ) == 0 )

!  allocate workspace

      array_name = 'blls_cgls: data%FREE'
      CALL SPACE_resize_array( data%n_free, data%FREE, status, alloc_status,   &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_cgls: data%P'
      CALL SPACE_resize_array( n, data%P, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_cgls: data%Q'
      CALL SPACE_resize_array( m, data%Q, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'blls_cgls: data%G'
      CALL SPACE_resize_array( n, data%G, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      IF ( data%preconditioned ) THEN
        array_name = 'blls_cgls: data%S'
        CALL SPACE_resize_array( n, data%S, status, alloc_status,              &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

!  record the free variables

      data%n_free = 0
      DO j = 1, n
        IF ( FIXED( j ) == 0 ) THEN
          data%n_free = data%n_free + 1
          data%FREE( data%n_free ) = j
        END IF
      END DO

      IF ( data%printp ) WRITE( out, "( /, A, ' ** cgls entered',              &
     &    ' (n = ', I0, ', free = ', I0, ') ** ' )" ) prefix, n, data%n_free

!  compute the initial function value f = 1/2 ||r||^2

      norm_r = TWO_NORM( R )
      f = half * norm_r ** 2

!  exit if there are no free variables

      IF ( data%n_free == 0 ) THEN
        status = GALAHAD_ok
        GO TO 800
      END IF

!  compute the gradient g = A^T r at the initial point, and its norm

!  a) evaluation directly via A

      IF ( data%present_a ) THEN
        DO k = 1, data%n_free
          j = data%FREE( k )
          data%G( j ) = zero
          DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
            i = A_row( l )
            data%G( j ) = data%G( j ) + A_val( l ) * R( i )
          END DO
        END DO

!  b) evaluation via matrix-vector product call

      ELSE IF ( data%present_afprod ) THEN
        CALL eval_AFPROD( status, userdata, transpose = .TRUE., V = R,         &
                          P = data%G, FREE = data%FREE, n_free = data%n_free )
        IF ( status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_evaluation ; GO TO 900
        END IF

!  c) evaluation via reverse communication

      ELSE
        reverse%V = R
        data%branch = 20 ; status = 3
        RETURN
      END IF

!  return from reverse communication

  20  CONTINUE
      IF ( data%reverse_afprod ) THEN
        IF ( reverse%eval_status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_evaluation ; GO TO 900
        END IF
        IF ( data%n_free < n ) THEN
          data%G( data%FREE( : data%n_free ) )                                &
            = reverse%P( data%FREE( : data%n_free ) )
        ELSE
          data%G( : n ) = reverse%P( : n )
        END IF
      END IF

!  include the gradient of the regularization term if present

      IF ( data%regularization ) THEN
        IF ( data%n_free < n ) THEN
          data%G( data%FREE( : data%n_free ) ) =                               &
            data%G( data%FREE( : data%n_free ) ) +                             &
              weight * X( data%FREE( : data%n_free ) )
        ELSE
          data%G( : n ) = data%G( : n ) + weight * X
        END IF
        f = f + half * weight * TWO_NORM( X ) ** 2
      END IF

!  set the initial preconditioned gradient

!  a) evaluation via preconditioner-inverse-vector product call

      IF ( data%preconditioned ) THEN
!write(6,"(' g', 4ES12.4)" ) data%G( : n )
        IF ( data%present_dprec ) THEN
          IF ( data%n_free < n ) THEN
            data%S( data%FREE( : data%n_free ) )                               &
              = data%G( data%FREE( : data%n_free ) )                           &
                  / DPREC( data%FREE( : data%n_free ) )
          ELSE
            data%S( : n ) = data%G( : n ) / DPREC( : n )
          END IF

!  b) evaluation via preconditioner-inverse-vector product call

        ELSE IF ( data%present_prec ) THEN
          CALL eval_PREC( status, userdata, V = data%G, P = data%S )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%V( : n ) = data%G( : n )
          data%branch = 90 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   90 CONTINUE

!  set the initial preconditioned search direction from the preconditioned
!  gradient

      IF ( data%preconditioned ) THEN
        IF (  data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%S( : n ) = reverse%P( : n )
        END IF
!write(6,"(' s', 4ES12.4)" ) data%S( : n )
        IF ( data%n_free < n ) THEN
          data%P( data%FREE( : data%n_free ) )                                 &
            = - data%S( data%FREE( : data%n_free ) )
        ELSE
          data%P( : n ) = - data%S( : n )
        END IF

!  and compute its length

        IF ( data%n_free < n ) THEN
          data%gamma = DOT_PRODUCT( data%G( data%FREE( : data%n_free ) ),      &
                                    data%S( data%FREE( : data%n_free ) ) )
        ELSE
          data%gamma = DOT_PRODUCT( data%G( : n ), data%S( : n ) )
        END IF
        norm_g = SQRT( data%gamma )

!  or set the initial unpreconditioned search direction

      ELSE
        IF ( data%n_free < n ) THEN
          data%P( data%FREE( : data%n_free ) )                                 &
            = - data%G( data%FREE( : data%n_free ) )
        ELSE
          data%P( : n ) = - data%G( : n )
        END IF

!  and compute its length

        IF ( data%n_free < n ) THEN
          norm_g = TWO_NORM( data%G( data%FREE( : data%n_free ) ) )
        ELSE
          norm_g = TWO_NORM( data%G( : n ) )
        END IF
        data%gamma = norm_g ** 2
      END IF

!  set the cg stopping tolerance

      data%stop_cg = MAX( stop_cg_relative * norm_g, stop_cg_absolute )
      IF ( data%printp ) WRITE( out, "( A, '    stopping tolerance =',         &
     &   ES11.4, / )" ) prefix, data%stop_cg

!  print details of the intial point

      IF ( data%printp ) THEN
        WRITE( out, 2000 )
        WRITE( out, 2010 ) iter, f, norm_g
      END IF

!  ---------
!  main loop
!  ---------

  100 CONTINUE  ! mock iteration loop
        iter = iter + 1

!  check that the iteration limit has not been reached

        IF ( iter > maxit ) THEN
          status = GALAHAD_error_max_iterations ; GO TO 800
        END IF

! form q = A p

!  a) evaluation directly via A

        IF ( data%present_a ) THEN
          data%Q( : m ) = zero
          DO k = 1, data%n_free
            j = data%FREE( k )
            val = data%P( j )
            DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
              i = A_row( l )
              data%Q( i ) = data%Q( i ) + A_val( l ) * val
            END DO
          END DO

!  b) evaluation via matrix-vector product call

        ELSE IF ( data%present_afprod ) THEN
          CALL eval_AFPROD( status, userdata, transpose = .FALSE., V = data%P, &
                            P = data%Q, FREE = data%FREE, n_free = data%n_free )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%V( : n ) = data%P( : n )
          data%branch = 110 ; status = 2
          RETURN
        END IF

!  return from reverse communication

  110   CONTINUE
        IF ( data%reverse_afprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%Q( : m ) = reverse%P( : m )
        END IF

!  compute the step length to the minimizer along the line x + alpha p

        curv = TWO_NORM( data%Q( : m ) ) ** 2
        IF ( data%regularization ) THEN
          IF ( data%n_free < n ) THEN
            curv = curv + weight *                                             &
              TWO_NORM( data%P( data%FREE( : data%n_free ) ) ) ** 2
          ELSE
            curv = curv + weight * TWO_NORM( data%P( : n ) ) ** 2
          END IF
        END IF
        IF ( curv == zero ) curv = epsmch
        alpha = data%gamma / curv

!  update the estimate of the minimizer, x, and its residual, r

        IF ( data%n_free < n ) THEN
          X( data%FREE( : data%n_free ) )                                      &
            = X( data%FREE( : data%n_free ) )                                  &
                + alpha * data%P( data%FREE( : data%n_free ) )
        ELSE
          X = X + alpha * data%P( : n )
        END IF
        R = R + alpha * data%Q( : m )

!  update the value of the objective function

        f = f - half * alpha * alpha * curv
        norm_r = SQRT( two * f )

!  compute the gradient g = A^T r at x and its norm

!  a) evaluation directly via A

        IF ( data%present_a ) THEN
          DO k = 1, data%n_free
            j = data%FREE( k )
            data%G( j ) = zero
            DO l = A_ptr( j ) , A_ptr( j + 1 ) - 1
              i = A_row( l )
              data%G( j ) = data%G( j ) + A_val( l ) * R( i )
            END DO
          END DO

!  b) evaluation via matrix-vector product call

        ELSE IF ( data%present_afprod ) THEN
          CALL eval_AFPROD( status, userdata, transpose = .TRUE., V = R,       &
                            P = data%G, FREE = data%FREE, n_free = data%n_free )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%V( : m ) = R
          data%branch = 120 ; status = 3
          RETURN
        END IF

!  return from reverse communication

  120   CONTINUE
        IF ( data%reverse_afprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          IF ( data%n_free < n ) THEN
            data%G( data%FREE( : data%n_free ) )                               &
              = reverse%P( data%FREE( : data%n_free ) )
          ELSE
            data%G( : n ) = reverse%P( : n )
          END IF
        END IF

!  include the gradient of the regularization term if present

        IF ( data%regularization ) THEN
          IF ( data%n_free < n ) THEN
            data%G( data%FREE( : data%n_free ) ) =                             &
              data%G( data%FREE( : data%n_free ) ) +                           &
                weight * X( data%FREE( : data%n_free ) )
          ELSE
            data%G( : n ) = data%G( : n ) + weight * X
          END IF
        END IF

!  compute the preconditioned gradient

!  a) evaluation via preconditioner-inverse-vector product call

        IF ( data%preconditioned ) THEN
          IF ( data%present_dprec ) THEN
            IF ( data%n_free < n ) THEN
              data%S( data%FREE( : data%n_free ) )                             &
                = data%G( data%FREE( : data%n_free ) )                         &
                    / DPREC( data%FREE( : data%n_free ) )
            ELSE
              data%S( : n ) = data%G( : n ) / DPREC( : n )
            END IF

!  b) evaluation via preconditioner-inverse-vector product call

          ELSE IF ( data%present_prec ) THEN
            CALL eval_PREC( status, userdata, V = data%G, P = data%S )
            IF ( status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF

!  c) evaluation via reverse communication

          ELSE
            reverse%V( : n ) = data%G( : n )
            data%branch = 140 ; status = 4
            RETURN
          END IF
        END IF

!  return from reverse communication

  140   CONTINUE
        gamma_old = data%gamma

 !  compute the length of the preconditioned gradient

        IF ( data%preconditioned ) THEN
          IF ( data%reverse_prec ) THEN
            IF ( reverse%eval_status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%S( : n ) = reverse%P( : n )
          END IF
          IF ( data%n_free < n ) THEN
            data%gamma = DOT_PRODUCT( data%G( data%FREE( : data%n_free ) ),    &
                                      data%S( data%FREE( : data%n_free ) ) )
          ELSE
            data%gamma = DOT_PRODUCT( data%G( : n ), data%S( : n ) )
          END IF
          norm_g = SQRT( data%gamma )

!  compute the length of the unpreconditioned gradient

        ELSE
          IF ( data%n_free < n ) THEN
            norm_g = TWO_NORM( data%G( data%FREE( : data%n_free ) ) )
          ELSE
            norm_g = TWO_NORM( data%G( : n ) )
          END IF
          data%gamma = norm_g ** 2
        END IF

!  print details of the current step

!       IF ( data%printw ) WRITE( out, 2000 )
        IF ( data%printp ) WRITE( out, 2010 ) iter, f, norm_g

! test for convergence

        IF ( norm_g <= data%stop_cg ) THEN
          status = GALAHAD_ok ; GO TO 800
        END IF

!  compute the next preconditioned search direction, p

        beta = data%gamma / gamma_old

        IF ( data%preconditioned ) THEN
          IF ( data%n_free < n ) THEN
            data%P( data%FREE( : data%n_free ) )                               &
              = - data%S( data%FREE( : data%n_free ) )                         &
                + beta * data%P( data%FREE( : data%n_free ) )
          ELSE
            data%P( : n ) = - data%S( : n ) + beta * data%P( : n )
          END IF

!  or the next unpreconditioned one

        ELSE
          IF ( data%n_free < n ) THEN
            data%P( data%FREE( : data%n_free ) )                               &
              = - data%G( data%FREE( : data%n_free ) )                         &
                + beta * data%P( data%FREE( : data%n_free ) )
          ELSE
            data%P( : n ) = - data%G( : n ) + beta * data%P( : n )
          END IF
        END IF

        GO TO 100  ! end of mock iteration loop

!  ----------------
!  end of main loop
!  ----------------

  800 CONTINUE
      RETURN

!  error returns

  900 CONTINUE
      RETURN

!  non-executable statement

 2000 FORMAT( '   iter           f                     g')
 2010 FORMAT( I7, 2ES22.14 )

!  End of subroutine BLLS_cgls

      END SUBROUTINE BLLS_cgls

!  End of module BLLS

   END MODULE GALAHAD_BLLS_double













