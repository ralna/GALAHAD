! THIS VERSION: GALAHAD 4.3 - 2024-01-17 AT 16:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ S L L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. June 3rd 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SLLS_precision

!        --------------------------------------------------------------------
!        |                                                                  |
!        | Solve the simplex-constrained linear-least-squares problem       |
!        |                                                                  |
!        |    minimize   1/2 || A_o x - b ||_W^2 + weight / 2 || x ||^2     |
!        |    subject to      e^T x = 1, x >= 0                             |
!        |                                                                  |
!        | where ||v|| and ||r||_W^2 are the Euclidean & weighted Euclidean |
!        | norms defined by ||v||^2 = v^T v and ||r||_W^2 = r^T W r, using  |
!        | a preconditioned projected conjugate-gradient approach           |
!        |                                                                  |
!        --------------------------------------------------------------------

     USE GALAHAD_KINDS_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_STRING
     USE GALAHAD_SPACE_precision
     USE GALAHAD_SORT_precision, ONLY: SORT_heapsort_build,                    &
                                    SORT_heapsort_smallest, SORT_quicksort
     USE GALAHAD_SBLS_precision
     USE GALAHAD_NORMS_precision
     USE GALAHAD_QPT_precision
     USE GALAHAD_QPD_precision, ONLY: QPD_SIF
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_CONVERT_precision, ONLY: CONVERT_control_type,                &
                                          CONVERT_inform_type,                 &
                                          CONVERT_to_sparse_column_format
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SLLS_initialize, SLLS_read_specfile, SLLS_solve,                &
               SLLS_terminate,  SLLS_reverse_type, SLLS_data_type,             &
               SLLS_full_initialize, SLLS_full_terminate,                      &
               SLLS_search_data_type,                                          &
               SLLS_subproblem_data_type, SLLS_exact_arc_search,               &
               SLLS_inexact_arc_search, SLLS_import, SLLS_import_without_a,    &
               SLLS_solve_given_a, SLLS_solve_reverse_a_prod,                  &
               SLLS_reset_control, SLLS_information, GALAHAD_userdata_type,    &
               QPT_problem_type, SMT_type, SMT_put, SMT_get,                   &
               SLLS_project_onto_simplex, SLLS_simplex_projection_path,        &
               SLLS_cgls


!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE SLLS_initialize
       MODULE PROCEDURE SLLS_initialize, SLLS_full_initialize
     END INTERFACE SLLS_initialize

     INTERFACE SLLS_terminate
       MODULE PROCEDURE SLLS_terminate, SLLS_full_terminate
     END INTERFACE SLLS_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

     REAL ( KIND = rp_ ), PARAMETER :: x_zero = ten * epsmch
     REAL ( KIND = rp_ ), PARAMETER :: g_zero = ten * epsmch
     REAL ( KIND = rp_ ), PARAMETER :: h_zero = ten * epsmch
     REAL ( KIND = rp_ ), PARAMETER :: epstl2 = ten * epsmch
     REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )
     REAL ( KIND = rp_ ), PARAMETER :: alpha_search = one
     REAL ( KIND = rp_ ), PARAMETER :: beta_search = half
     REAL ( KIND = rp_ ), PARAMETER :: mu_search = 0.1_rp_
     REAL ( KIND = rp_ ), PARAMETER :: fixed_tol = ten ** ( -15 )
     REAL ( KIND = rp_ ), PARAMETER :: eta = 0.01_rp_

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLLS_control_type

!  unit number for error and warning diagnostics

       INTEGER ( KIND = ip_ ) :: error = 6

!  general output unit number

       INTEGER ( KIND = ip_ ) :: out  = 6

!  the level of output required

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  on which iteration to start printing

       INTEGER ( KIND = ip_ ) :: start_print = - 1

!  on which iteration to stop printing

       INTEGER ( KIND = ip_ ) :: stop_print = - 1

!  how many iterations between printing

       INTEGER ( KIND = ip_ ) :: print_gap = 1

!  how many iterations to perform (-ve reverts to HUGE(1)-1)

       INTEGER ( KIND = ip_ ) :: maxit = 1000

!  cold_start should be set to 0 if a warm start is required (with variables
!   assigned according to X_stat, see below), and to any other value if the
!   values given in prob%X suffice

       INTEGER ( KIND = ip_ ) :: cold_start = 1

!  the preconditioner (scaling) used (0=none,1=diagonal,anything else=user)

       INTEGER ( KIND = ip_ ) :: preconditioner = 1

!  the ratio of how many iterations use CGLS rather than steepest descent

       INTEGER ( KIND = ip_ ) :: ratio_cg_vs_sd = 1

!  the maximum number of per-iteration changes in the working set permitted
!   when allowing CGLS rather than steepest descent

       INTEGER ( KIND = ip_ ) :: change_max = 2

!  how many CG iterations to perform per SLLS iteration (-ve reverts to n+1)

       INTEGER ( KIND = ip_ ) :: cg_maxit = 1000

!  the maximum number of steps allowed in a piecewise arcsearch (-ve=infinite)

       INTEGER ( KIND = ip_ ) :: arcsearch_max_steps = - 1

!  the unit number to write generated SIF file describing the current problem

       INTEGER ( KIND = ip_ ) :: sif_file_device = 52

!  the objective function will be regularized by adding 1/2 weight ||x||^2

       REAL ( KIND = rp_ ) :: weight = zero

!  the required accuracy for the dual infeasibility

       REAL ( KIND = rp_ ) :: stop_d = ten ** ( - 6 )

!  the CG iteration will be stopped as soon as the current norm of the
!   preconditioned gradient is smaller than
!    max( stop_cg_relative * initial preconditioned gradient, stop_cg_absolute )

       REAL ( KIND = rp_ ) :: stop_cg_relative = ten ** ( - 2 )
       REAL ( KIND = rp_ ) :: stop_cg_absolute = epsmch

!  the largest permitted arc length during the piecewise line search

       REAL ( KIND = rp_ ) :: alpha_max = ten ** 20

!  the initial arc length during the inexact piecewise line search

       REAL ( KIND = rp_ ) :: alpha_initial = one

!  the arc length reduction factor for the inexact piecewise line search

       REAL ( KIND = rp_ ) :: alpha_reduction = half

!  the required relative reduction during the inexact piecewise line search

       REAL ( KIND = rp_ ) :: arcsearch_acceptance_tol = ten ** ( - 2 )

!  the stabilisation weight added to the search-direction subproblem

       REAL ( KIND = rp_ ) :: stabilisation_weight = ten ** ( - 12 )

!  the maximum CPU time allowed (-ve = no limit)

       REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!  direct_subproblem_solve is true if the least-squares subproblem is to be
!   solved using a matrix factorization, and false if conjugate gradients
!   are to be preferred

       LOGICAL :: direct_subproblem_solve = .TRUE.

!  exact_arc_search is true if an exact arc_search is required, and false if an
!   approximation suffices

       LOGICAL :: exact_arc_search = .TRUE.

!  advance is true if an inexact exact arc_search can increase steps as well
!   as decrease them (currently not implemented)

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
         "SLLSPROB.SIF"  // REPEAT( ' ', 18 )

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
     END TYPE SLLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLLS_time_type

!  total time

       REAL :: total = 0.0

!  time for the analysis phase

       REAL :: analyse = 0.0

!  time for the factorization phase

       REAL :: factorize = 0.0

!  time for the linear solution phase

       REAL :: solve = 0.0
     END TYPE SLLS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLLS_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)
!   -20  alegedly +ve definite matrix is not

       INTEGER ( KIND = ip_ ) :: status = 1

!  STAT value after allocate failure

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  status return from factorization

       INTEGER ( KIND = ip_ ) :: factorization_status = 0

!  number of iterations required

       INTEGER ( KIND = ip_ ) :: iter = - 1

!  number of CG iterations required

       INTEGER ( KIND = ip_ ) :: cg_iter = 0

!  current value of the objective function

       REAL ( KIND = rp_ ) :: obj = infinity

!  current value of the projected gradient

       REAL ( KIND = rp_ ) :: norm_pg = infinity

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  times for various stages

       TYPE ( SLLS_time_type ) :: time

!  inform values from SBLS

       TYPE ( SBLS_inform_type ) :: SBLS_inform

!  inform values for CONVERT

       TYPE ( CONVERT_inform_type ) :: CONVERT_inform
     END TYPE SLLS_inform_type

!  - - - - - - - - - - - - - - -
!   arc_search data derived type
!  - - - - - - - - - - - - - - -

     TYPE :: SLLS_search_data_type
       INTEGER ( KIND = ip_ ) :: step
       REAL ( KIND = rp_ ) :: ete, f_0, f_1_stop, gamma, rtr, xtx
       REAL ( KIND = rp_ ) :: rho_0, rho_1, rho_2, s_fixed, t, t_break, t_total
       LOGICAL :: present_a, reverse, backwards
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: NZ_out
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V, P
     END TYPE SLLS_search_data_type
!  - - - - - - - - - - - - - - -
!   arc_search data derived type
!  - - - - - - - - - - - - - - -

     TYPE :: SLLS_subproblem_data_type
       INTEGER ( KIND = ip_ ) :: branch, n_preconditioner, step
       REAL ( KIND = rp_ ) :: stop_cg, gamma, gamma_old, etpe
       LOGICAL :: printp, printw, printd, printdd, debug
       LOGICAL :: present_a, present_asprod, reverse_asprod, present_afprod
       LOGICAL :: reverse_afprod, reverse_prec, present_prec, present_dprec
       LOGICAL :: recompute, regularization, preconditioned
       CHARACTER ( LEN = 1 ) :: direction
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, P, Q, PG, PE
     END TYPE SLLS_subproblem_data_type

!  - - - - - - - - - - -
!   reverse derived type
!  - - - - - - - - - - -

     TYPE :: SLLS_reverse_type
       INTEGER ( KIND = ip_ ) :: nz_in_start, nz_in_end, nz_out_end
       INTEGER ( KIND = ip_ ) :: eval_status = GALAHAD_ok
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: NZ_in, NZ_out
       LOGICAL :: transpose
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V, P
     END TYPE SLLS_reverse_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE :: SLLS_data_type
       INTEGER ( KIND = ip_ ) :: out, error, print_level, eval_status
       INTEGER ( KIND = ip_ ) :: start_print, stop_print, print_gap
       INTEGER ( KIND = ip_ ) :: arc_search_status, cgls_status, change_status
       INTEGER ( KIND = ip_ ) :: n_free, branch, cg_iter, preconditioner
       INTEGER ( KIND = ip_ ) :: maxit, cg_maxit, segment, steps, max_steps
       REAL :: time_start
       REAL ( KIND = rp_ ) :: norm_step, step, stop_cg, old_gnrmsq, pnrmsq
       REAL ( KIND = rp_ ) :: alpha_0, alpha_max, alpha_new, f_new, phi_new
       REAL ( KIND = rp_ ) :: weight, stabilisation_weight
       REAL ( KIND = rp_ ) :: regularization_weight
       LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
       LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe
       LOGICAL :: reverse, reverse_prod, explicit_a, use_aprod, header
       LOGICAL :: direct_subproblem_solve, steepest_descent, w_eq_identity
       CHARACTER ( LEN = 6 ) :: string_cg_iter
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: FREE
       LOGICAL ( KIND = lp_ ), ALLOCATABLE, DIMENSION( : ) :: FIXED, FIXED_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_new, G, R, SBLS_sol
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: D, AD, S, AE, DIAG
       TYPE ( SMT_type ) :: Ao, H_sbls, AT_sbls, C_sbls
       TYPE ( SLLS_subproblem_data_type ) :: subproblem_data
       TYPE ( SLLS_search_data_type ) :: search_data
       TYPE ( SBLS_data_type ) :: SBLS_data
     END TYPE SLLS_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: SLLS_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        LOGICAL :: explicit_a
        TYPE ( SLLS_data_type ) :: SLLS_data
        TYPE ( SLLS_control_type ) :: SLLS_control
        TYPE ( SLLS_inform_type ) :: SLLS_inform
        TYPE ( QPT_problem_type ) :: prob
        TYPE ( GALAHAD_userdata_type ) :: userdata
        TYPE ( SLLS_reverse_type ) :: reverse
      END TYPE SLLS_full_data_type

   CONTAINS

!-*-*-*-*-*-   S L L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE SLLS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  default control data for SLLS. This routine should be called before
!  SLLS_solve
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

     TYPE ( SLLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_control_type ), INTENT( OUT ) :: control
     TYPE ( SLLS_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  initialize control parameters for SBLS (see GALAHAD_SBLS for details)

     CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
     control%SBLS_control%prefix = '" - SBLS:"                    '

!  added here to prevent for compiler bugs

     control%stop_d = epsmch ** 0.33_rp_
     control%stop_cg_absolute = SQRT( epsmch )

     RETURN

!  end of SLLS_initialize

     END SUBROUTINE SLLS_initialize

!- G A L A H A D -  S L L S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE SLLS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SLLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_control_type ), INTENT( OUT ) :: control
     TYPE ( SLLS_inform_type ), INTENT( OUT ) :: inform

     data%explicit_a = .FALSE.
     CALL SLLS_initialize( data%slls_data, control, inform )

     RETURN

!  End of subroutine SLLS_full_initialize

     END SUBROUTINE SLLS_full_initialize

!-*-*-*-   S L L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE SLLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by SLLS_initialize could (roughly)
!  have been set as:

! BEGIN SLLS SPECIFICATIONS (DEFAULT)
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
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  cg-relative-accuracy-required                     0.01
!  cg-absolute-accuracy-required                     1.0D-8
!  maximum-arcsearch-stepsize                        1.0D+20
!  initial-arcsearch-stepsize                        1.0
!  arcsearch-reduction-factor                        5.0D-1
!  arcsearch-acceptance-tolerance                    1.0D-2
!  stabilisation-weight                              0.0
!  maximum-cpu-time-limit                            -1.0
!  direct-subproblem-solve                           F
!  exact-arc-search-used                             T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  sif-file-name                                     SLLSPROB.SIF
!  output-line-prefix                                ""
! END SLLS SPECIFICATIONS

!  dummy arguments

     TYPE ( SLLS_control_type ), INTENT( INOUT ) :: control
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  programming: Nick Gould and Ph. Toint, January 2002.

!  local variables

     INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: start_print = print_level + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_print = start_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_gap = stop_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_device = print_gap + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: maxit = sif_file_device + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cold_start = maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: preconditioner = maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: ratio_cg_vs_sd = preconditioner + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: change_max = ratio_cg_vs_sd + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cg_maxit = change_max + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: arcsearch_max_steps = cg_maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: weight = arcsearch_max_steps + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_d = weight + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_cg_relative = stop_d + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_cg_absolute                     &
                                            = stop_cg_relative + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alpha_max = stop_cg_absolute + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alpha_initial = alpha_max + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alpha_reduction = alpha_initial + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: arcsearch_acceptance_tol             &
                                            = alpha_reduction + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stabilisation_weight                 &
                                            = arcsearch_acceptance_tol + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit                       &
                                            = stabilisation_weight + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: direct_subproblem_solve              &
                                            = cpu_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: exact_arc_search                     &
                                            = direct_subproblem_solve + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: advance = exact_arc_search + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = advance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal               &
                                            = space_critical + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: generate_sif_file                    &
                                            = deallocate_error_fatal + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: sif_file_name = generate_sif_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = sif_file_name + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'SLLS'
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
     spec( stop_d )%keyword = 'dual-accuracy-required'
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
     CALL SPECFILE_assign_value( spec( stop_d ),                               &
                                 control%stop_d,                               &
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

     END SUBROUTINE SLLS_read_specfile

!-*-*-*-*-*-*-*-   S L L S _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE SLLS_solve( prob, X_stat, data, control, inform, userdata,     &
                            W, reverse, eval_APROD, eval_ASPROD, eval_AFPROD,  &
                            eval_PREC )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the linear least-squares problem
!
!     minimize     q(x) = 1/2 || Ao x - b ||_2^2 + weight / 2 || x ||^2
!
!     subject to    e^T x = 1, x >= 0
!
!  where x is a vector of n components ( x_1, .... , x_n ), b is an m-vector,
!  and Ao is an o by n matrix, using a preconditioned projected CG method.
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
!   %o is an INTEGER variable, which must be set by the user to the
!    number of residuals, o. RESTRICTION: %o >= 1
!
!   %Ao is a structure of type SMT_type used to hold Ao if available).
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 10 ) = TRANSFER( 'COORDINATE', Ao%type )
!       Ao%val( : )   the values of the components of A
!       Ao%row( : )   the row indices of the components of A
!       Ao%col( : )   the column indices of the components of A
!       Ao%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', Ao%type )
!       Ao%val( : )   the values of the components of A, stored row by row
!       Ao%col( : )   the column indices of the components of A
!       Ao%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', Ao%type )
!       Ao%val( : )   the values of the components of A, stored column by column
!       Ao%row( : )   the row indices of the components of A
!       Ao%ptr( : )   pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', Ao%type )
!       Ao%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', Ao%type )
!       Ao%val( : )   the values of the components of A, stored column by column,
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
!    objective function. The i-th component of B, i = 1, ...., o,
!    should contain the value of b_i.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %C is a REAL array of length %m, which need not be set on input. On
!    successful exit, it will contain the residual vector c(x) = A x - b. The
!    i-th component of C, i = 1, ...., o, will contain the value of c_i(x).
!
!   %Z is a REAL array of length %n, which need not be set on input. On
!    successful exit, it will contain estimates of the values of the dual
!    variables, i.e., Lagrange multipliers corresponding to the simple bound
!    constraints x >= 0.
!
!  X_stat is a INTEGER array of length n, which may be set by the user
!   on entry to SLLS_solve to indicate which of the simple bound constraints
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; X_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   X_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   X_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound, zero,
!               = 0, the i-th bound constraint is not in the working set
!
!  data is a structure of type SLLS_data_type which holds private internal data
!
!  control is a structure of type SLLS_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to SLLS_initialize. See SLLS_initialize
!   for details
!
!  inform is a structure of type SLLS_inform_type that provides
!    information on exit from SLLS_solve. The component %status
!    must be set to 1 on initial entry, and on exit has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!     2 The product A * v of the matrix A with a given vector v is required
!       from the user. The vector v will be provided in reverse%V and the
!       required product must be returned in reverse%P. SLLS_solve must then
!       be re-entered with reverse%eval_status set to 0, and any remaining
!       arguments unchanged. Should the user be unable to form the product,
!       this should be flagged by setting reverse%eval_status to a nonzero value
!
!     3 The product A^T * v of the transpose of the matrix A with a given
!       vector v is required from the user. The vector v will be provided in
!       reverse%V and the required product must be returned in reverse%P.
!       SLLS_solve must then be re-entered with reverse%eval_status set to 0,
!       and any remaining arguments unchanged. Should the user be unable to
!       form the product, this should be flagged by setting reverse%eval_status
!       to a nonzero value
!
!     4 The product A * v of the matrix A with a given sparse vector v is
!       required from the user. Only components
!         reverse%NZ_in( reverse%nz_in_start : reverse%nz_in_end )
!       of the vector v stored in reverse%V are nonzero. The required product
!       should be returned in reverse%P. SLLS_solve must then be re-entered
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
!       placed in reverse%NZ_out( 1 : reverse%nz_out_end ). SLLS_solve should
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
!       and SLLS_solve then re-entered with all other arguments unchanged.
!       reverse%eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     7 The product P^-1 * v involving the preconditioner P with a specified
!       vector v is required from the user. Here P should be a symmtric,
!       postive-definite approximation of A^T A. The vector v will be provided
!       in reverse%V and the required product must be returned in reverse%P.
!       SLLS_solve must then be re-entered with reverse%eval_status set to 0,
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
!        prob%o     >=  0
!        prob%Ao%type in { 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',
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
!  On exit from SLLS_solve, other components of inform give the
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
!       solution determined by SLLS_solve.
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
!  W is an optional rank-one array of type default real that if present
!   must be of length prob%o and filled with the weights w_i > 0. If W is
!   absent, weights of one will be used.
!
!  reverse is an OPTIONAL structure of type SLLS_reverse_type which is used to
!   pass intermediate data to and from SLLS_solve. This will only be necessary
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
!   be set to a nonzero value. If eval_APROD is not present, SLLS_solve
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
!   present, SLLS_solve will either return to the user each time an evaluation
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
!   If eval_AFPROD is not present, SLLS_solve will either return to the user
!   each time an evaluation is required (see reverse above) or form the
!   product directly from user-provided %A.
!
!  eval_PREC is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product P^{-1} * v of the given
!   preconditioner P and vector v stored in V must be returned in P.
!   The intention is that P is an approximation to A^T A. The status variable
!   should be set to 0 unless the product is impossible in which case status
!   should be set to a nonzero value. If eval_PREC is not present, SLLS_solve
!   will return to the user each time a preconditioning operation is required
!   (see reverse above) when control%preconditioner is not 0 or 1.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( prob%n ) :: X_stat
     TYPE ( SLLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: W
     TYPE ( SLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse
     OPTIONAL :: eval_APROD, eval_ASPROD, eval_AFPROD, eval_PREC

!  interface blocks

     INTERFACE
       SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, INTENT( IN ) :: transpose
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
       END SUBROUTINE eval_APROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_ASPROD( status, userdata, V, P, NZ_in, nz_in_start,     &
                               nz_in_end, NZ_out, nz_out_end )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: nz_in_start, nz_in_end
       INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ) :: nz_out_end
       INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_in
       INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL,                       &
                                               INTENT( INOUT ) :: NZ_out
       END SUBROUTINE eval_ASPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_AFPROD( status, userdata, transpose, V, P, FREE, n_free )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, INTENT( IN ) :: transpose
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_free
       INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: FREE
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       END SUBROUTINE eval_AFPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_PREC( status, userdata, V, P )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       END SUBROUTINE eval_PREC
     END INTERFACE

!  local variables

     INTEGER ( KIND = ip_ ) :: i, j, k, l, nap
     INTEGER ( KIND = ip_ ) :: minloc_g( 1 )
     REAL :: time
     REAL ( KIND = rp_ ) :: val, x_j, g_j, d_j, lambda
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
     CASE ( 110 ) ; GO TO 110  ! re-entry with p = p + Av
     CASE ( 210 ) ; GO TO 210  ! re-entry with p = p + Av
     CASE ( 220 ) ; GO TO 220  ! re-entry with p = p + A^T v
     CASE ( 300 ) ; GO TO 300  ! re-entry with p = A v (dense or sparse)
     CASE ( 400 ) ; GO TO 400  ! re-entry with p = p + A v
     CASE ( 410 ) ; GO TO 410  ! re-entry with p = A v (dense or sparse)
     CASE ( 450 ) ; GO TO 450  ! re-entry with p = A v (dense or sparse)
     END SELECT

 100 CONTINUE

     IF ( control%out > 0 .AND. control%print_level >= 5 )                     &
       WRITE( control%out, 2000 ) prefix, ' entering '

! -------------------------------------------------------------------
!  if desired, generate a SIF file for problem passed

     IF ( control%generate_sif_file ) THEN
       CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,     &
                     infinity = infinity, qp = .FALSE. )
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
     data%explicit_a = ALLOCATED( prob%Ao%type )
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
       inform%status = GALAHAD_error_optional ; GO TO 910
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

     IF ( prob%n <= 0 .OR. prob%o <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 910
     ELSE IF ( data%explicit_a ) THEN
       IF ( .NOT. QPT_keyword_A( prob%Ao%type ) ) THEN
         inform%status = GALAHAD_error_restrictions
         GO TO 910
       END IF
     END IF

!  see if W = I

       data%w_eq_identity = .NOT. PRESENT( W )
       IF ( .NOT. data%w_eq_identity ) THEN
         IF ( COUNT( W( : prob%o ) <= zero ) > 0 ) THEN
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( A, ' error: input entries of W must be strictly positive' )" ) &
             prefix
           inform%status = GALAHAD_error_restrictions
           GO TO 910
         ELSE IF ( COUNT( W( : prob%o ) == one ) == prob%o ) THEN
           data%w_eq_identity = .TRUE.
         END IF
       END IF

!  if required, write out problem

     IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
       WRITE( control%out, "( ' o, n = ', I0, 1X, I0 )" ) prob%o, prob%n
       WRITE( control%out, "( ' B = ', /, ( 5ES12.4 ) )" ) prob%B( : prob%o )
       IF ( data%w_eq_identity ) THEN
         WRITE( control%out, "( ' W = identity' )" )
       ELSE
         WRITE( control%out, "( ' W = ', /, ( 5ES12.4 ) )" ) W( : prob%o )
       END IF
       IF ( data%explicit_a ) THEN
         SELECT CASE ( SMT_get( prob%Ao%type ) )
         CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS'  )
           WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )         &
             prob%Ao%val( : prob%o * prob%n )
         CASE ( 'SPARSE_BY_ROWS' )
           WRITE( control%out, "( ' A (row-wise) = ' )" )
           DO i = 1, prob%o
             WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                  &
               ( i, prob%Ao%col( j ), prob%Ao%val( j ),                        &
                 j = prob%Ao%ptr( i ), prob%Ao%ptr( i + 1 ) - 1 )
           END DO
         CASE ( 'SPARSE_BY_COLUMNS' )
           WRITE( control%out, "( ' A (column-wise) = ' )" )
           DO j = 1, prob%n
             WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                  &
               ( prob%Ao%row( i ), j, prob%Ao%val( i ),                        &
                 i = prob%Ao%ptr( j ), prob%Ao%ptr( j + 1 ) - 1 )
           END DO
         CASE ( 'COORDINATE' )
           WRITE( control%out, "( ' A (co-ordinate) = ' )" )
           WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                    &
               ( prob%Ao%row( i ), prob%Ao%col( i ), prob%Ao%val( i ),         &
                 i = 1, prob%Ao%ne )
         END SELECT
       ELSE
         WRITE( control%out, "( ' A available via function reverse call' )" )
       END IF
     END IF

!  if required, use the initial variable status implied by X_stat

     IF ( control%cold_start == 0 ) THEN
       DO i = 1, prob%n
         IF ( X_stat( i ) < 0 ) prob%X( i ) = zero
       END DO
     END IF

!  check that input estimate of the solution is in the simplex, and if not
!  project it so that it is

     array_name = 'slls: data%X_new'
     CALL SPACE_resize_array( prob%n, data%X_new, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     CALL SLLS_project_onto_simplex( prob%n, prob%X, data%X_new, i )

     IF ( i < 0 ) THEN
       inform%status = GALAHAD_error_sort
       GO TO 910
     ELSE IF ( i > 0 ) THEN
       prob%X( : prob%n ) = data%X_new( : prob%n )
       IF ( data%printi ) WRITE( control%out,                                  &
       "( ' ', /, A, '   **  Warning: input point projected onto simplex' )" ) &
         prefix
     END IF

!  allocate workspace arrays

     array_name = 'slls: prob%R'
     CALL SPACE_resize_array( prob%o, prob%R, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: prob%G'
     CALL SPACE_resize_array( prob%n, prob%G, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: prob%Z'
     CALL SPACE_resize_array( prob%n, prob%Z, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: data%R'
     CALL SPACE_resize_array( prob%o, data%R, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: data%D'
     CALL SPACE_resize_array( prob%n, data%D, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: data%S'
     CALL SPACE_resize_array( prob%n, data%S, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: data%AD'
     CALL SPACE_resize_array( prob%o, data%AD, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: data%AE'
     CALL SPACE_resize_array( prob%o, data%AE, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( data%preconditioner /= 0 ) THEN
       array_name = 'slls: data%DIAG'
       CALL SPACE_resize_array( prob%n, data%DIAG, inform%status,              &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     array_name = 'slls: data%FREE'
     CALL SPACE_resize_array( prob%n, data%FREE, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: data%FIXED'
     CALL SPACE_resize_array( prob%n, data%FIXED, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'slls: data%FIXED_old'
     CALL SPACE_resize_array( prob%n, data%FIXED_old, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     IF ( data%reverse ) THEN
       array_name = 'slls: reverse%NZ_in'
       CALL SPACE_resize_array( prob%n, reverse%NZ_in, inform%status,          &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: reverse%NZ_out'
       CALL SPACE_resize_array( prob%o, reverse%NZ_out, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: reverse%V'
       CALL SPACE_resize_array( MAX( prob%o, prob%n ), reverse%V,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: reverse%P'
       CALL SPACE_resize_array( MAX( prob%o, prob%n ), reverse%P,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     IF ( control%exact_arc_search .AND. PRESENT( eval_ASPROD ) .AND.          &
          .NOT. ( data%explicit_a .OR. data%reverse ) ) THEN
       array_name = 'slls: data%search_data%NZ_out'
       CALL SPACE_resize_array( prob%o, data%search_data%NZ_out, inform%status,&
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%search_data%V'
       CALL SPACE_resize_array( prob%n, data%search_data%V,                    &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%search_data%P'
       CALL SPACE_resize_array( prob%o, data%search_data%P,                    &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

!  build a copy of A stored by columns

     IF ( data%explicit_a ) THEN
       CALL CONVERT_to_sparse_column_format( prob%Ao, data%Ao,                 &
                                             control%CONVERT_control,          &
                                             inform%CONVERT_inform )

!  weight by W if required

       IF ( .NOT. data%w_eq_identity ) THEN
         DO j = 1, prob%n
           DO k = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             data%Ao%val( k ) = data%Ao%val( k ) * SQRT( W( data%Ao%row( k ) ) )
           END DO
         END DO
       END IF
     END IF

!  set A e if needed later

     IF ( control%exact_arc_search ) THEN
       data%AE( : prob%o ) = zero
       IF (  data%explicit_a ) THEN
         DO j = 1, prob%n
           DO l = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             i = data%Ao%row( l )
             data%AE( i ) = data%AE( i ) + data%Ao%val( l )
           END DO
         END DO
       ELSE IF ( data%use_aprod ) THEN
         data%D( : prob%n ) = one
         data%AE( : prob%o ) = zero
         CALL eval_APROD( data%eval_status, userdata, .FALSE., data%D, data%AE )
         IF ( data%eval_status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 910
         END IF
       ELSE
!        reverse%P( : prob%o ) = zero
         reverse%V( : prob%n ) = one
         reverse%transpose = .FALSE.
         data%branch = 110 ; inform%status = 2 ; RETURN
       END IF
     END IF

!  re-entry point after the A e product

 110   CONTINUE
       IF ( control%exact_arc_search .AND. data%reverse_prod )                 &
         data%AE( : prob%o ) = reverse%P( : prob%o )

!  compute the diagonal preconditioner if required

     IF ( data%preconditioner /= 0 ) THEN
       IF ( data%explicit_a ) THEN
         DO j = 1, prob%n
           val = zero
           DO k = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             val = val + data%Ao%val( k ) ** 2
           END DO
           IF ( val > zero ) THEN
             data%DIAG( j ) = val
           ELSE
             data%DIAG( j ) = one
           END IF
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
       data%H_sbls%m = prob%o ; data%H_sbls%n = prob%o

!  the 2,1 block

       CALL SMT_put( data%AT_sbls%type, 'SPARSE_BY_ROWS', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate
         GO TO 910
       END IF
       data%AT_sbls%n = prob%o

       array_name = 'slls: data%AT_sbls%val'
       CALL SPACE_resize_array( data%Ao%ne, data%AT_sbls%val, inform%status,   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%AT_sbls%col'
       CALL SPACE_resize_array( data%Ao%ne, data%AT_sbls%col, inform%status,   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%AT_sbls%ptr'
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

         array_name = 'slls: data%C_sbls%val'
         CALL SPACE_resize_array( 1_ip_, data%C_sbls%val, inform%status,       &
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

     END IF

!  solution vector (used as workspace elsewhere)

     array_name = 'slls: data%SBLS_sol'
     CALL SPACE_resize_array( prob%n + prob%o, data%SBLS_sol, inform%status,   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  ------------------------
!  start the main iteration
!  ------------------------

     data%change_status = prob%n

     IF ( data%set_printi ) WRITE( data%out,                                   &
       "( /, A, 9X, 'S=steepest descent, F=factorization used' )" ) prefix

 200 CONTINUE ! mock iteration loop
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
         prob%R( : prob%o ) = - prob%B( : prob%o )
         DO j = 1, prob%n
           x_j = prob%X( j )
           DO k = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             prob%R( data%Ao%row( k ) )                                        &
               = prob%R( data%Ao%row( k ) ) + data%Ao%val( k ) * x_j
           END DO
         END DO
       ELSE IF ( data%use_aprod ) THEN
         prob%R( : prob%o ) = - prob%B( : prob%o )
         CALL eval_APROD( data%eval_status, userdata, .FALSE., prob%X, prob%R )
         IF ( data%eval_status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 910
         END IF
       ELSE
!        reverse%P( : prob%o ) = - prob%B
!        reverse%P( : prob%o ) = zero
         reverse%V( : prob%n ) = prob%X
!write(6,"(' v', /, ( 5ES12.4 ) )" ) reverse%V( : prob%n )
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
           DO k = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             g_j = g_j + data%Ao%val( k ) * prob%R( data%Ao%row( k ) )
           END DO
           prob%G( j ) = g_j
         END DO
       ELSE IF ( data%use_aprod ) THEN
         prob%G( : prob%n ) = zero
         CALL eval_APROD( data%eval_status, userdata, .TRUE., prob%R, prob%G )
         IF ( data%eval_status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 910
         END IF
       ELSE
         prob%R( : prob%o ) = reverse%P( : prob%o ) - prob%B( : prob%o )
!        prob%C( : prob%o ) = reverse%P( : prob%o )
!        reverse%P( : prob%n ) = zero
         reverse%V( : prob%o ) = prob%R( : prob%o )
         reverse%transpose = .TRUE.
         data%branch = 220 ; inform%status = 3 ; RETURN
       END IF

!  re-entry point after the Jacobian-transpose vector product

 220   CONTINUE
       IF ( data%reverse_prod ) prob%G( : prob%n ) = reverse%P( : prob%n )
!write(6,"( ' G =', / ( 5ES12.4 ) )" ) prob%G

!  compute the objective function

       inform%obj = half * DOT_PRODUCT( prob%R( : prob%o ), prob%R( : prob%o ) )

!  adjust the value and gradient to account for any regularization term

       IF ( control%weight > zero ) THEN
         inform%obj = inform%obj + half * control%weight *                     &
           DOT_PRODUCT( prob%X( : prob%n ), prob%X( : prob%n ) )
         prob%G( : prob%n )                                                    &
           = prob%G( : prob%n ) + control%weight * prob%X( : prob%n )
       END IF

!  record the dual variables

       prob%Z( : prob%n ) = prob%G( : prob%n )

!  record the dual variables

!  compute the norm of the projected gradient

       val = MIN( one, one / TWO_NORM( prob%G( : prob%n ) ) )
       CALL SLLS_project_onto_simplex( prob%n,                                 &
                                       prob%X - val * prob%G( : prob%n ),      &
                                       data%X_new, i )
       inform%norm_pg = MAXVAL( ABS( data%X_new -prob%X ) )

!write(6,"( ' etx etxnew = ', 2ES12.4 )" ) SUM( prob%X ), SUM( data%X_new )
!write(6,"( ' minval x xnew = ', 2ES12.4 )" ) &
! MINVAL( prob%X ), MINVAL( data%X_new )
!write(6,"( ' ||g|| = ', ES12.4 )" ) MAXVAL( prob%G )

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
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  test to see if more than maxit iterations have been performed

       IF ( inform%iter > data%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 910
       END IF

!  check that the CPU time limit has not been reached

       IF ( control%cpu_time_limit >= zero .AND.                               &
            inform%time%total > control%cpu_time_limit ) THEN
         inform%status = GALAHAD_error_cpu_limit ; GO TO 910
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
           data%D( : prob%n ) =                                                &
             - prob%G( : prob%n ) / SQRT( data%DIAG( : prob%n ) )
         ELSE
           data%D( : prob%n ) = - prob%G( : prob%n )
         END IF
         data%string_cg_iter = '     S'

!  compute the Cauchy direction d^C = - argmin g^Ts : e^s = 0, s >= - x

!  find the largest component of g

         minloc_g = MINLOC( prob%G( : prob%n ) )
         i = minloc_g( 1 )
!write(6,"('c', /, ( 5ES12.4) )" ) prob%R( : prob%o )
!write(6,"('g', /, ( 5ES12.4) )" ) prob%G( : prob%n )
!write(6,*) ' minloc g ', minloc_g( 1 )
         data%D( : prob%n ) = - prob%X( : prob%n )
         data%D( i ) = one - prob%X( i )
         GO TO 310
       END IF

!  - - - - - - - - - - - - - - - augmented system - - - - - - - - - - - - - - -

!  compute the search direction d by minimizing the objective over
!  the free subspace by solving the related augmented systems

!    (    I        Ao_F   ) (  p  ) = (   b - A x  )  (a)
!    ( Ao_F^T  - weight I ) ( q_F )   ( weight x_F )

!  and

!    (    I        Ao_F   ) (  u  ) = (  0  )         (b)
!    ( Ao_F^T  - weight I ) ( v_F )   ( e_F )

!  and then forming

!   (  r  ) = (  p  ) - lambda (  u  ), where lambda = q_F^T e_F / v_F^T e_f (c)
!   ( d_F )   ( q_F )          ( v_F )

       IF ( data%direct_subproblem_solve ) THEN

!  set up the block matrices. Copy the free columns of A into the rows
!  of AT

         nap = 0
         DO i = 1, data%n_free
           j = data%FREE( i )
           data%AT_sbls%ptr( i ) = nap + 1
           DO k = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             nap = nap + 1
             data%AT_sbls%col( nap ) = data%Ao%row( k )
             data%AT_sbls%val( nap ) = data%Ao%val( k )
           END DO

!  include components of the right-hand side vector for system (a)

           data%SBLS_sol( prob%o + i ) = data%stabilisation_weight * prob%X( j )
         END DO
         data%AT_sbls%ptr( data%n_free + 1 ) = nap + 1
         data%AT_sbls%m = data%n_free

!  make sure that the 2,2 block has the correct dimension

         data%C_sbls%m = data%n_free ; data%C_sbls%n = data%n_free

!  complete the right-hand side vector for system (a)

         data%SBLS_sol( : prob%o ) = - prob%R( : prob%o )

!  form and factorize the augmented matrix

         CALL SBLS_form_and_factorize( prob%o, data%n_free,                    &
                 data%H_sbls, data%AT_sbls, data%C_sbls, data%SBLS_data,       &
                 control%SBLS_control, inform%SBLS_inform )

!  test for factorization failure

         IF ( inform%SBLS_inform%status < 0 ) THEN
           IF ( data%printe )                                                  &
             WRITE( control%error, 2010 ) prefix, inform%SBLS_inform%status,   &
               'SBLS_form_and_factorize'
           CALL SYMBOLS_status( inform%SBLS_inform%status, control%out,        &
                                prefix, 'SBLS_form_and_factorize' )
           inform%status = GALAHAD_error_factorization ; GO TO 910
         END IF

!  solve system (a)

         CALL SBLS_solve( prob%o, data%n_free, data%AT_sbls, data%C_sbls,      &
                 data%SBLS_data, control%SBLS_control, inform%SBLS_inform,     &
                 data%SBLS_sol )

!  record the components q_F in data%S

         data%S( : data%n_free )                                               &
           = data%SBLS_sol( prob%o + 1 : prob%o + data%n_free )

!  compute q_F^T e_F

         lambda = SUM( data%S( : data%n_free ) )

!  set up the right-hand side vector for system (b)

         data%SBLS_sol( : prob%o ) = zero
         data%SBLS_sol( prob%o + 1 : prob%o + data%n_free ) = one

!  solve system (b)

         CALL SBLS_solve( prob%o, data%n_free, data%AT_sbls, data%C_sbls,      &
                 data%SBLS_data, control%SBLS_control, inform%SBLS_inform,     &
                 data%SBLS_sol )

!  compute q_F^T e_F / v_F^T e_f

         lambda                                                                &
           = lambda / SUM( data%SBLS_sol( prob%o + 1 : prob%o + data%n_free ) )

!  extract the search direction d_f from (c)

         data%D( : prob%n ) = zero
         DO i = 1, data%n_free
           j = data%FREE( i )
           data%D( j ) =  data%S( i ) - lambda * data%SBLS_sol( prob%o + i )
         END DO
!write(6,*) ' ----------- nfree ', data%n_free
         IF ( data%printm ) WRITE( data%out, "( ' dtg, dtd = ', 2ES12.4 )" )   &
           DOT_PRODUCT( data%D( : prob%n ), prob%G( : prob%n ) ),              &
           DOT_PRODUCT( data%D( : prob%n ), data%D( : prob%n ) )
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
       data%R( : prob%o ) = prob%R( : prob%o )
       data%cgls_status = 1

!  if reverse communication is to be used, store the list of free variables

       IF ( data%reverse ) THEN
         reverse%nz_in_start = 1
         reverse%nz_in_end = data%n_free
         reverse%NZ_in( : data%n_free ) = data%FREE( : data%n_free )
       END IF

!  minimization loop

!write(6,"( ' nfree = ', I0, ' FREE = ', /, ( 10I8 ) )" ) &
! data%n_free, data%FREE( : data%n_free )
!write(6,"( ' x = ', /, ( 5ES12.4 ) )" ) data%X_new( : prob%n )

 300   CONTINUE ! mock CGLS loop

!  find an improved point, X_new, by conjugate-gradient least-squares ...

!  ... using the available Jacobian ...

         IF (  data%explicit_a ) THEN
           IF ( data%preconditioner == 0 ) THEN
             CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,         &
                             data%out, data%printm, data%printd, prefix,       &
!                            data%out, .TRUE., .FALSE., prefix,                &
                             data%f_new, data%X_new, data%R,                   &
                             data%FREE, control%stop_cg_relative,              &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, data%subproblem_data, userdata, &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, Ao_ptr = data%Ao%ptr,           &
                             Ao_row = data%Ao%row, Ao_val = data%Ao%val )
           ELSE IF ( data%preconditioner == 1 ) THEN
             CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,         &
                             data%out, data%printm, data%printd, prefix,       &
!                            data%out, .TRUE., .FALSE., prefix,                &
                             data%f_new, data%X_new, data%R,                   &
                             data%FREE, control%stop_cg_relative,              &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, data%subproblem_data, userdata, &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, Ao_ptr = data%Ao%ptr,           &
                             Ao_row = data%Ao%row, Ao_val = data%Ao%val,       &
                             preconditioned = .TRUE.,                          &
                             DPREC = data%DIAG )
           ELSE
             CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,         &
                             data%out, data%printm, data%printd, prefix,       &
                             data%f_new, data%X_new, data%R,                   &
                             data%FREE, control%stop_cg_relative,              &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, data%subproblem_data, userdata, &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, Ao_ptr = data%Ao%ptr,           &
                             Ao_row = data%Ao%row, Ao_val = data%Ao%val,       &
                             reverse = reverse, preconditioned = .TRUE.,       &
                             eval_PREC = eval_PREC )
           END IF

!  ... or products via the user's subroutine or reverse communication ...

         ELSE
           IF ( data%preconditioner == 0 ) THEN
             CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,         &
                             data%out, data%printm, data%printd, prefix,       &
                             data%f_new, data%X_new, data%R,                   &
                             data%FREE, control%stop_cg_relative,              &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, data%subproblem_data, userdata, &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, eval_AFPROD = eval_AFPROD,      &
                             reverse = reverse )
           ELSE IF ( data%preconditioner == 1 ) THEN
             CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,         &
                             data%out, data%printm, data%printd, prefix,       &
                             data%f_new, data%X_new, data%R,                   &
                             data%FREE, control%stop_cg_relative,              &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, data%subproblem_data, userdata, &
                             data%cgls_status, inform%alloc_status,            &
                             inform%bad_alloc, eval_AFPROD = eval_AFPROD,      &
                             reverse = reverse, preconditioned = .TRUE.,       &
                             DPREC = data%DIAG )
           ELSE
             CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,         &
                             data%out, data%printm, data%printd, prefix,       &
                             data%f_new, data%X_new, data%R,                   &
                             data%FREE, control%stop_cg_relative,              &
                             control%stop_cg_absolute, data%cg_iter,           &
                             control%cg_maxit, data%subproblem_data, userdata, &
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
           data%D( : prob%n ) = data%X_new( : prob%n ) - prob%X( : prob%n )
!write(6,"( ' d = ', /, ( 5ES12.4 ) )" ) data%D( : prob%n )
           data%norm_step = MAXVAL( ABS( data%D( : prob%n ) ) )
!write(6,*) ' ----------- nfree ', data%n_free
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
             WRITE( control%error, 2010 ) prefix, data%cgls_status, 'SLLS_cgls'
           inform%status = data%cgls_status
           GO TO 910
         END SELECT

         GO TO 300  ! end of minimization loop

!  a search-direction had been computed

 310   CONTINUE ! end of mock CGLS loop

!  ----------------------------------------------------------------------------
!           perform a projected arc search along the search direction
!  ----------------------------------------------------------------------------

!  initialize x and r = A x - b

       data%arc_search_status = 0
       data%X_new( : prob%n ) = prob%X( : prob%n )
!write(6,"( ' x = ', /, ( 5ES12.4 ) )" ) prob%X( : prob%n )
!write(6,"( ' d = ', /, ( 5ES12.4 ) )" ) data%D( : prob%n )
       data%R( : prob%o ) = prob%R( : prob%o )

!  perform an exact arc search

       IF ( .NOT. control%exact_arc_search ) GO TO 440
!      GO TO 440

!  set A d and A e

       data%AD( : prob%o ) = zero
       IF (  data%explicit_a ) THEN
         DO j = 1, prob%n
           d_j = data%D( j )
           DO l = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             i = data%Ao%row( l )
             data%AD( i ) = data%AD( i ) + data%Ao%val( l ) * d_j
           END DO
         END DO
       ELSE IF ( data%use_aprod ) THEN
         data%AD( : prob%o ) = zero
         CALL eval_APROD( data%eval_status, userdata, .FALSE., data%D, data%AD )
         IF ( data%eval_status /= GALAHAD_ok ) THEN
           inform%status = GALAHAD_error_evaluation ; GO TO 910
         END IF
       ELSE
!        reverse%P( : prob%o ) = zero
         reverse%V( : prob%n ) = data%D( : prob%n )
         reverse%transpose = .FALSE.
         data%branch = 400 ; inform%status = 2 ; RETURN
       END IF

!  re-entry point after the A d product

 400   CONTINUE
       IF ( data%reverse_prod ) data%AD( : prob%o ) = reverse%P( : prob%o )
       data%SBLS_sol( : prob%o ) = data%AE( : prob%o ) ! used as surrogate AE

 410   CONTINUE ! mock arc search loop

!  find an improved point, X_new, by exact arc_search using the available
!  Jacobian ...

         IF (  data%explicit_a ) THEN
           CALL SLLS_exact_arc_search( prob%n, prob%o, data%weight, data%out,  &
                                       data%printm, data%printd, prefix,       &
!                                        .TRUE., .FALSE., prefix, &
                                       data%arc_search_status, data%X_new,     &
                                       data%R, data%D, data%AD, data%sbls_sol, &
                                       data%segment, data%n_free, data%FREE,   &
                                       data%search_data, userdata,             &
                                       data%f_new, data%alpha_new,             &
                                       Ao_ptr = data%Ao%ptr,                   &
                                       Ao_row = data%Ao%row,                   &
                                       Ao_val = data%Ao%val )

!  ... or products via the user's subroutine or reverse communication ...

         ELSE
           CALL SLLS_exact_arc_search( prob%n, prob%o, data%weight, data%out,  &
                                       data%printm, data%printd, prefix,       &
                                       data%arc_search_status, data%X_new,     &
                                       data%R, data%D, data%AD, data%sbls_sol, &
                                       data%segment, data%n_free, data%FREE,   &
                                       data%search_data, userdata,             &
                                       data%f_new, data%alpha_new,             &
                                       eval_ASPROD = eval_ASPROD,              &
                                       reverse = reverse )
         END IF

!  check the output status

         SELECT CASE ( data%arc_search_status )

!  successful exit with the new point

!        CASE ( 0 )
         CASE ( - 1 : 0 )
           data%norm_step =                                                    &
             MAXVAL( ABS( data%X_new( : prob%n ) - prob%X( : prob%n ) ) )
           GO TO 500

!  error exit without the new point

!        CASE ( : - 1 )
         CASE ( : - 2 )
           IF ( data%printe ) WRITE( control%error, 2010 )                     &
               prefix, data%arc_search_status, 'SLLS_exact_arc_search'
           inform%status = data%arc_search_status
           GO TO 910

!  compute the status-th column of A

         CASE ( 1 : )
           reverse%nz_in_start = 1
           reverse%nz_in_end = 1
           reverse%NZ_in( 1 ) =  data%arc_search_status
           reverse%V( data%arc_search_status ) = one
           data%branch = 410 ; inform%status = 5 ; RETURN
         END SELECT
         GO TO 410 ! end of arc length loop


!  find an improved point, X_new, by inexact arc_search, using the available
!  Jacobian ...

 440   CONTINUE
       data%alpha_0 = alpha_search

 450   CONTINUE ! mock arc search loop

         IF (  data%explicit_a ) THEN
           CALL SLLS_inexact_arc_search( prob%n, prob%o, data%weight, data%out,&
                                         data%printm, data%printd, prefix,     &
!                                        .TRUE., .FALSE., prefix, &
                                         data%arc_search_status, data%X_new,   &
                                         prob%R, data%D, data%sbls_sol,        &
                                         data%R, control%alpha_initial,        &
                                         control%alpha_reduction,              &
                                         control%arcsearch_acceptance_tol,     &
                                         control%arcsearch_max_steps,          &
!                                        control%alpha_max,                    &
!                                        control%advance,                      &
                                         data%n_free, data%FREE,               &
                                         data%search_data, userdata,           &
                                         data%f_new, data%alpha_new,           &
                                         Ao_ptr = data%Ao%ptr,                 &
                                         Ao_row = data%Ao%row,                 &
                                         Ao_val = data%Ao%val )

!  ... or products via the user's subroutine or reverse communication

         ELSE
           CALL SLLS_inexact_arc_search( prob%n, prob%o, data%weight, data%out,&
                                         data%printm, data%printd, prefix,     &
                                         data%arc_search_status, data%X_new,   &
                                         prob%R, data%D, data%sbls_sol,        &
                                         data%R, control%alpha_initial,        &
                                         control%alpha_reduction,              &
                                         control%arcsearch_acceptance_tol,     &
                                         control%arcsearch_max_steps,          &
!                                        control%alpha_max,                    &
!                                        control%advance,                      &
                                         data%n_free, data%FREE,               &
                                         data%search_data, userdata,           &
                                         data%f_new, data%alpha_new,           &
                                         eval_APROD = eval_APROD,              &
                                         reverse = reverse )
         END IF

!  check the output status

         SELECT CASE ( data%arc_search_status )

!  successful exit with the new point

!        CASE ( 0 )
         CASE ( - 1 : 0 )
           data%norm_step =                                                    &
             MAXVAL( ABS( data%X_new( : prob%n ) - prob%X( : prob%n ) ) )
           GO TO 500

!  error exit without the new point

!        CASE ( : - 1 )
         CASE ( : - 2 )
           IF ( data%printe ) WRITE( control%error, 2010 )                     &
                 prefix, data%arc_search_status, 'SLLS_inexact_arc_search'
           inform%status = data%arc_search_status
           GO TO 910

!  form the matrix-vector product A * v

         CASE ( 2 )
           data%branch = 450 ; inform%status = 2 ; RETURN

         END SELECT
         GO TO 450 ! end of arc length loop

!  the arc length has been computed

 500   CONTINUE  ! end of mock arc search loop

!  record the new point in x

       prob%X( : prob%n ) = data%X_new( : prob%n )
       inform%obj = data%f_new
!      write(6,"( ' x = ', /, ( 5ES12.4 ) )" ) prob%X( : prob%n )
!      inform%obj = data%phi_new

!  record the number of variables that have changed status

       data%FIXED( : prob%n ) = .TRUE.
       data%FIXED( data%FREE( : data%n_free ) ) = .FALSE.
       IF ( inform%iter > 0 ) data%change_status                               &
         = COUNT( data%FIXED( : prob%n ) .NEQV. data%FIXED_old( : prob%n ) )
       data%FIXED_old( : prob%n ) = data%FIXED( : prob%n )

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
       GO TO 200 ! end of mock iteration loop

!  ----------------------
!  end the main iteration
!  ----------------------

!  successful return

 900 CONTINUE

!  set the optimal variable status

     X_stat = - 1
     X_stat( data%FREE( : data%n_free ) ) = 0
!    WRITE( 6, "( ' X_stat = ', /, ( 20I3 ) )" ) X_stat
     DO i = 1, prob%n
       IF ( prob%X( i ) > epsmch ) THEN
         X_stat( i ) = 0
       ELSE
         X_stat( i ) = - 1
       END IF
     END DO

     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  error returns

 910 CONTINUE
     CALL CPU_TIME( time ) ; inform%time%total = time - data%time_start

     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, control%out, prefix, 'SLLS_solve' )
     ELSE IF ( data%printe ) THEN
       WRITE( control%error, 2010 ) prefix, inform%status, 'SLLS_solve'
     END IF
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  Non-executable statements

2000 FORMAT( /, A, ' --', A, ' SLLS_solve' )
2010 FORMAT( A, '   **  Error return, status = ', I0, ', from ', A )

!  End of SLLS_solve

      END SUBROUTINE SLLS_solve

!-*-*-*-*-*-   S L L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE SLLS_terminate( data, control, inform, reverse )

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
!   data    see Subroutine SLLS_initialize
!   control see Subroutine SLLS_initialize
!   inform  see Subroutine SLLS_solve
!   reverse see Subroutine SLLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SLLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( SLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse

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
       CALL SLLS_reverse_terminate( reverse, control, inform )
       IF ( control%deallocate_error_fatal .AND.                               &
            inform%status /= GALAHAD_ok ) RETURN
     END IF

!  Deallocate all those required when solving the subproblems

     CALL SLLS_subproblem_terminate( data%subproblem_data, control, inform )

!  Deallocate all remaining allocated arrays

     array_name = 'slls: data%Ao%ptr'
     CALL SPACE_dealloc_array( data%Ao%ptr,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%Ao%col'
     CALL SPACE_dealloc_array( data%Ao%col,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%Ao%val'
     CALL SPACE_dealloc_array( data%Ao%val,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%R'
     CALL SPACE_dealloc_array( data%R,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%D'
     CALL SPACE_dealloc_array( data%D,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%AD'
     CALL SPACE_dealloc_array( data%AD,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%AE'
     CALL SPACE_dealloc_array( data%AE,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%SBLS_sol'
     CALL SPACE_dealloc_array( data%SBLS_sol,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%X_new'
     CALL SPACE_dealloc_array( data%X_new,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%FREE'
     CALL SPACE_dealloc_array( data%FREE,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%FIXED'
     CALL SPACE_dealloc_array( data%FIXED,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%FIXED_old'
     CALL SPACE_dealloc_array( data%FIXED_old,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%DIAG'
     CALL SPACE_dealloc_array( data%DIAG,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%AT_sbls%val'
     CALL SPACE_dealloc_array( data%AT_sbls%val,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%AT_sbls%col'
     CALL SPACE_dealloc_array( data%AT_sbls%col,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%AT_sbls%ptr'
     CALL SPACE_dealloc_array( data%AT_sbls%ptr,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%SBLS_sol'
     CALL SPACE_dealloc_array( data%SBLS_sol,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%C_sbls%val'
     CALL SPACE_dealloc_array( data%C_sbls%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%V'
     CALL SPACE_dealloc_array( data%search_data%V,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%P'
     CALL SPACE_dealloc_array( data%search_data%P,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%NZ_out'
     CALL SPACE_dealloc_array( data%search_data%NZ_out,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine SLLS_terminate

     END SUBROUTINE SLLS_terminate

! -  G A L A H A D -  S L L S _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE SLLS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLLS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

     data%explicit_a = .FALSE.

!  deallocate workspace

     CALL SLLS_terminate( data%slls_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'slls: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%G'
     CALL SPACE_dealloc_array( data%prob%G,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%B'
     CALL SPACE_dealloc_array( data%prob%B,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%R'
     CALL SPACE_dealloc_array( data%prob%R,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Ao%ptr'
     CALL SPACE_dealloc_array( data%prob%Ao%ptr,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Ao%row'
     CALL SPACE_dealloc_array( data%prob%Ao%row,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Ao%col'
     CALL SPACE_dealloc_array( data%prob%Ao%col,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Ao%val'
     CALL SPACE_dealloc_array( data%prob%Ao%val,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Ao%type'
     CALL SPACE_dealloc_array( data%prob%Ao%type,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     CALL SLLS_reverse_terminate( data%reverse, control, inform )

     RETURN

!  End of subroutine SLLS_full_terminate

     END SUBROUTINE SLLS_full_terminate

!-*-   S L L S _ R E V E R S E _ T E R M I N A T E   S U B R O U T I N E   -*-

     SUBROUTINE SLLS_reverse_terminate( reverse, control, inform )

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
!   reverse see Subroutine SLLS_solve
!   control see Subroutine SLLS_initialize
!   inform  see Subroutine SLLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SLLS_reverse_type ), INTENT( INOUT ) :: reverse
     TYPE ( SLLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

     array_name = 'slls: reverse%V'
     CALL SPACE_dealloc_array( reverse%V,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: reverse%P'
     CALL SPACE_dealloc_array( reverse%P,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: reverse%NZ_in'
     CALL SPACE_dealloc_array( reverse%NZ_in,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: reverse%NZ_out'
     CALL SPACE_dealloc_array( reverse%NZ_out,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine SLLS_reverse_terminate

     END SUBROUTINE SLLS_reverse_terminate

!-   S L L S _ S U B P R O B L E M _ T E R M I N A T E   S U B R O U T I N E   -

     SUBROUTINE SLLS_subproblem_terminate( data, control, inform )

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
!   data    see Subroutine SLLS_initialize
!   control see Subroutine SLLS_initialize
!   inform  see Subroutine SLLS_solve
!   reverse see Subroutine SLLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SLLS_subproblem_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'slls: data%P'
     CALL SPACE_dealloc_array( data%P,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%PG'
     CALL SPACE_dealloc_array( data%PG,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%Q'
     CALL SPACE_dealloc_array( data%Q,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%G'
     CALL SPACE_dealloc_array( data%G,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%PE'
     CALL SPACE_dealloc_array( data%PE,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     RETURN

!  End of subroutine SLLS_subproblem_terminate

     END SUBROUTINE SLLS_subproblem_terminate

! -  S L L S _ P R O J E C T _ O N T O _ S I M P L E X    S U B R O U T I N E  -

      SUBROUTINE SLLS_project_onto_simplex( n, X, X_proj, status )

!  Find the projection of a given x onto the unit simplex {x | e^Tx = 1, x >= 0}

!  The algorithm is essentially that from
!   E. van den Berg and M. P. Friedlander.
!   Probing the Pareto frontier for basis pursuit solutions.
!   SIAM Journal on Scientific Computing, 31(2):890—-912, 2008.

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  n      (INTEGER) the number of variables
!  X      (REAL array of length at least n) the point to be projected
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  X_proj (REAL array of length at least n) the projected point
!  status (INTEGER) the return status from the package.
!          Possible output values are:
!          > 0 a successful exit and the input x has been projected
!          0 a successful exit and the input x already lies on the simplex
!          < 0 an error exit

!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X_proj

!  local variables

      INTEGER ( KIND = ip_ ) :: j, j_max, n_heap
      REAL ( KIND = rp_ ) :: h, sum, tau

!  build a heap with largest entry h from the components of x

      X_proj = X
      CALL SORT_heapsort_build ( n, X_proj, status, largest = .TRUE. )
      IF ( status < 0 ) RETURN
      h = X_proj( 1 )

!  record the number of entries in the heap in n_heap

      n_heap = n

!  store the sum of the j largest entries of x minus one in sum

      sum = - 1.0_rp_

!  loop to find how many components will be set to zero

      DO j = 1, n

!  check for termination

!       write(6,"( ' tau, h ', I10, 2ES12.4) ") j, sum / REAL( j, KIND = rp_ ), h
        IF ( ( sum + h ) / REAL( j, KIND = rp_ ) >= h ) EXIT
        sum = sum + h
        j_max = j

!  remove h from the heap, and restore the head with a new largest entry h

        CALL SORT_heapsort_smallest( n_heap, X_proj, status, largest = .TRUE. )
        IF ( status < 0 ) RETURN
        n_heap = n_heap - 1
        h = X_proj( 1 )

!  end loop

      END DO

!  compute the projection

      tau = sum / REAL( j_max, KIND = rp_ )
!     write(6,"( ' tau = ', ES12.4, ' j_max = ', I0 )" ) tau, j_max
      IF ( ABS( tau ) > REAL( n, KIND = rp_ ) * epsmch ) THEN
        status = 1
        X_proj = MAX( X - tau, zero )
      ELSE
        status = 0
        X_proj = X
      END IF

      RETURN

!  End of subroutine SLLS_project_onto_simplex

      END SUBROUTINE SLLS_project_onto_simplex

! - S L L S _ S I M P L E X _ P R O J E C T I O N _ P A T H  S U B R O U T I N E

      SUBROUTINE SLLS_simplex_projection_path( n, X, D, status )

!  Let Delta^n = { s | e^T s = 1, s >= 0 } be the unit simplex. Follow the
!  projection path P( x + t d ) from a given x and direction d as t increases
!  from zero, and P(v) projects v onto Delta^n

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  n      (INTEGER) the number of variables
!  X      (REAL array of length at least n) the initial point x
!  D      (REAL array of length at least n) the direction d
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  status (INTEGER) the return status from the package.
!          Possible output values are:
!          0 a successful exit
!          < 0 an error exit

!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: D

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, i_free, n_free
      REAL ( KIND = rp_ ) :: ete, etd, gamma, t, t_min, t_total
      REAL ( KIND = rp_ ) :: t_max = ten ** 20
      INTEGER ( KIND = ip_ ), DIMENSION( n ) :: FREE
      REAL ( KIND = rp_ ), DIMENSION( n ) :: V, S, PROJ

!  ensure that the path goes somewhere

      IF ( MAXVAL( ABS( D ) ) == zero ) THEN
        status = - 1
        RETURN
      END IF

!  project the initial point x onto Delta^n

      CALL SLLS_project_onto_simplex( n, X, V, status )

!  project d onto the manifold v^T s = 0 to compute the search direction
!  s = d - ( e^T d / e^T e ) e

      t_total = zero
      ete = REAL( n, KIND = rp_ )
      etd = SUM( D( : n ) )
      S = D - etd / ete

      DO j = 1, n
        FREE( j ) = j
      END DO
      PROJ = - V / S
      CALL SORT_quicksort( n, PROJ, status, ix = FREE )
      DO i = 1, n
        IF ( PROJ( i ) > zero ) WRITE( 6, "( I8, ES12.4 )" ) FREE( i ), PROJ( i)
      END DO

!  FREE(:n_free) are indices of variables that are free

      n_free = n
      DO j = 1, n
        FREE( j ) = j
      END DO

!  main loop

      DO k = 1, n - 1

!  compute the step along s to the boundary of Delta^n, as well as the
!  index i_free of the position in FREE of the variable that meets its bound

        WRITE( 6, "( 8X, '        t           v           s' )" )
        t_min = t_max
        DO i = 1, n_free
          j = FREE( i )
          IF ( S( j ) < zero ) THEN
            t = - V( j ) / S( j )
            WRITE( 6, "( I8, 3ES12.4 )" ) j, t, V( j ), S( j )
            IF ( t < t_min ) THEN
              i_free = i
              t_min = t
            END IF
          END IF
        END DO

!  update the total steplength

        t_total = t_total + t_min

!  fix the i-th variable and adjust FREE

        i = FREE( i_free )
        FREE( i_free ) = FREE( n_free )
        FREE( n_free ) = i
        n_free = n_free - 1
write( 6, "( ' fix variable ', I0 )" ) i

!  step to the boundary

        V( i ) = zero
        DO l = 1, n_free
          j = FREE( l )
          V( j ) = V( j ) + t_min * S( j )
write(6,"( ' x ', I8, ES12.4 )" ) j, V( j )
        END DO

!  update the search direction

        ete = ete - one
        gamma = S( i ) / ete
write(6,"( ' gamma = ', ES12.4 )" ) gamma
        DO i_free = 1, n_free
          j = FREE( i_free )
          IF ( S( j ) >= zero .AND. S( j ) + gamma < zero ) &
            write( 6, "( ' variable ', I6, ' now a candidate, t = ', ES12.4 )")&
               j, - V( j ) / ( S( j ) + gamma )
          S( j ) = S( j ) + gamma
write(6,"( ' s ', I8, ES12.4 )" ) j, S( j )
        END DO

!  compare calculated and recurred breakpoint

!       CALL SLLS_project_onto_simplex( n, V, PROJ, status )
!       WRITE( 6, "( ' status = ', I0 )" ) status
!        CALL SLLS_project_onto_simplex( n, X + t_total * D, PROJ, status )
!write(6, "(' status proj = ', I0 )" ) status
!        WRITE( 6, "( 8X, '        v          proj' )" )
!        DO j = 1, n
!          WRITE(6, "( I8, 2ES12.4 )" ) j, V( j ), PROJ( j )
!        END DO

!  end of main loop

      END DO

      RETURN

!  End of subroutine SLLS_simplex_projection_path

      END SUBROUTINE SLLS_simplex_projection_path

! -*-*-*- S L L S _ E X A C T _ A R C _ S E A R C H  S U B R O U T I N E -*-*-*-

      SUBROUTINE SLLS_exact_arc_search( n, o, weight, out, summary, debug,     &
                                        prefix, status, X, R, D, AD, AE,       &
                                        segment, n_free, FREE, data, userdata, &
                                        f_opt, t_opt, Ao_val, Ao_row, Ao_ptr,  &
                                        reverse, eval_ASPROD )

!  Let Delta^n = { s | e^T s = 1, s >= 0 } be the unit simplex. Follow the
!  projection path P( x + t d ) from a given x in Delta^n and direction d as
!  t increases from zero, and P(v) projects v onto Delta^n, and stop at the
!  first (local) minimizer of
!    1/2 || A P( x + t d ) - b ||^2 + 1/2 weight || P( x + t d ) ||^2

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  n        (INTEGER) the number of variables
!  o        (INTEGER) the number of observations
!  weight   (REAL) the regularization weight
!  out      (INTEGER) the output unit for printing
!  summary  (LOGICAL) one line of output per segment if true
!  debug    (LOGICAL) lots of output per segment if true
!  prefix   (CHARACTER string) that is used to prefix any output line
!
!  INPUT/OUTPUT arguments
!
!  status   (INTEGER) the input and return status for/from the package.
!            On initial entry, status must be set to 0.
!            Possible output values are:
!            0 a successful exit
!            < 0 an error exit
!            > 0 the user should provide data for the status-th column of A
!                and re-enter the subroutine with all non-optional arguments
!                unchanged. Row indices and values for the nonzero components
!                of the column should be provided in reverse%NZ_out(:nz) and
!                reverse%P(:nz), where nz is stored in reverse%nz_out_end;
!                these arrays should be allocated before use, and lengths of
!                at most o suffice. This value of status will only occur if
!                Ao_val, Ao_row and Ao_ptr are absent
!  X        (REAL array of length at least n) the initial point x
!  R        (REAL array of length at least o) the residual A x - b
!  D        (REAL array of length at least n) the direction d
!  AD       (REAL array of length at least o) the vectror A d, but
!           subsequently used as workspace
!  AE       (REAL array of length at least o) the vector A e where e is the
!           vector of ones, but subsequently used as workspace
!  segment  (INTEGER) the number of segments searched
!  n_free   (INTEGER) the number of free variables (i.e., variables not at zero)
!  FREE     (INTEGER array of length at least n) FREE(:n_free) are the indices
!            of the free variables
!  data     (structure of type slls_search_data_type) private data that is
!            preserved between calls to the subroutine
!  userdata  (structure of type GALAHAD_userdata_type ) data that may be passed
!             between calls to the evaluation subroutine eval_asprod
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  f_opt    (REAL) the optimal objective value
!  t_opt    (REAL) the optimal step length
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!          nonzeros of A, stored by consecutive columns. N.B. If present,
!          Ao_row and Ao_ptr must also be present
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!          of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!           Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!           with Ao_ptr(n+1) pointin to the storage location 1 beyond
!           the last entry in A
!  reverse (structure of type slls_reverse_type) data that is provided by the
!           user when prompted by status > 0
!  eval_asprod (subroutine) that provides sparse products with A(see slls_solve)
!
!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n, o, out
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status, segment, n_free
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      LOGICAL, INTENT( IN ):: summary, debug
      REAL ( KIND = rp_ ), INTENT( OUT ) :: f_opt, t_opt
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: FREE
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, D
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R, AD, AE
      TYPE ( SLLS_search_data_type ), INTENT( INOUT ) :: data
      TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      TYPE ( SLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse
      OPTIONAL :: eval_ASPROD

!  interface blocks

      INTERFACE
        SUBROUTINE eval_ASPROD( status, userdata, V, P, NZ_in, nz_in_start,    &
                                nz_in_end, NZ_out, nz_out_end )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: nz_in_start, nz_in_end
        INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ) :: nz_out_end
        INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: NZ_in
        INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL,                      &
                                                INTENT( INOUT ) :: NZ_out
        END SUBROUTINE eval_ASPROD
      END INTERFACE

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, l, i_fixed, now_fixed, nz_out_end
      INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: NZ_in
!     REAL ( KIND = rp_ ) :: s_j
      REAL ( KIND = rp_ ) :: a, f_1, f_2, t
      REAL ( KIND = rp_ ) :: t_max = ten ** 20
!     REAL ( KIND = rp_ ), DIMENSION( n ) :: V, PROJ
!     REAL ( KIND = rp_ ), DIMENSION( o ) :: AE_tmp, AD_tmp

!  if a re-entry occurs, branch to the appropriate place in the code

      IF ( status > 0 ) GO TO 200

!  check to see if A has been provided

      IF ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) .AND.                     &
           PRESENT( Ao_val ) ) THEN
        data%present_a = .TRUE. ; data%reverse = .FALSE.
      ELSE
        data%present_a = .FALSE.

!  check to see if access to products with A by reverse communication has
!  been provided

        IF ( PRESENT( reverse ) ) THEN
          data%reverse = .TRUE.

!  check to see if access to products with A by subroutine call has been
!  provided

        ELSE IF ( PRESENT( eval_ASPROD ) ) THEN
          data%reverse = .FALSE.

!  if none of thses option is available, exit

        ELSE
          status = - 2 ; RETURN
        END IF
      END IF

!  ensure that the path goes somewhere

      IF ( MAXVAL( ABS( D ) ) == zero ) THEN
        t_opt = zero ; f_opt = data%f_0 ; status = - 1
        RETURN
      END IF
      data%f_1_stop = 100.0_rp_ * epsmch * REAL( n, KIND = rp_ )

!  project the initial point x onto Delta^n

!     IF ( debug ) THEN
!       CALL SLLS_project_onto_simplex( n, X, V, status )
!       DO j = 1, n
!         FREE( j ) = j
!       END DO
!       PROJ = - V / D
!       CALL SORT_quicksort( n, PROJ, status, ix = FREE )
!       IF ( summary ) THEN
!         DO i = 1, n
!           IF ( PROJ( i ) > zero )                                            &
!             WRITE( out,  "( I8, ES12.4)" ) FREE( i ), PROJ( i )
!         END DO
!       END IF
!     END IF

      IF ( debug ) WRITE( out,  "( A, ' d = ', /, ( 5ES12.4 ) )" ) prefix, D

!  project d onto the manifold v^T s = 0 to compute the search direction
!  s = d - ( e^T d / e^T e ) e and product As = Ad - ( e^T d / e^T e ) Ae.
!  NB: s and As will be stored in D and AD respectively

      data%t_total = zero
      data%ete = REAL( n, KIND = rp_ )
      data%gamma = SUM( D( : n ) ) / data%ete
      IF ( data%gamma /= zero ) THEN
        D = D - data%gamma ; AD = AD - data%gamma * AE
      END IF

!  initialize f_0 = 1/2 || A x - b ||^2 + 1/2 weight || x ||^2

      IF ( weight > zero ) THEN
        data%rho_0 = DOT_PRODUCT( X, X )
        data%f_0 = half * ( DOT_PRODUCT( R, R ) + weight * data%rho_0 )
      ELSE
        data%f_0 = half * DOT_PRODUCT( R, R )
      END IF

!  if there is a regularization term, initialize rho_1 = x^T s and
!  rho_2 = ||s||^2

      IF ( weight > zero ) THEN
        data%rho_1 = DOT_PRODUCT( X, D ) ; data%rho_2 = DOT_PRODUCT( D, D )
      END IF

!  FREE(:n_free) are indices of variables that are free

      n_free = n
      FREE = (/ ( j, j = 1, n ) /)

!  main loop (mock do loop to allow reverse communication)

      segment = 1
      IF ( summary ) WRITE( out,  "( A, ' segment    f_0         f_1      ',   &
     &             '   f_2        t_break       t_opt' )" ) prefix
  100 CONTINUE
!       IF ( debug ) THEN
!         WRITE( out,  "( ' s = ', /, ( 5ES12.4 ) )" ) S
!         WRITE( out,  "( ' Ad = ', /, ( 5ES12.4 ) )" ) AD
!         AD_tmp = zero ; AE_tmp = zero
!         DO i_fixed = 1, n_free
!           j = FREE( i_fixed )
!           s_j = D( j )
!           DO l = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
!             i = Ao_row( l ) ; a = Ao_val( l )
!             AE_tmp( i ) = AE_tmp( i ) + a
!             AD_tmp( i ) = AD_tmp( i ) + a * s_j
!           END DO
!         END DO
!         WRITE( out,  "( ' D = ', /, ( 5ES12.4 ) )" ) D
!         WRITE( out,  "( ' Ad_tmp = ', /, ( 5ES12.4 ) )" ) AD_tmp
!         WRITE( out,  "( ' Ae = ', /, ( 5ES12.4 ) )" ) AE
!         WRITE( out,  "( ' Ae_tmp = ', /, ( 5ES12.4 ) )" ) AE_tmp
!       END IF

!  compute the slope f_1 and curvature f_2 along the current segment

        IF ( weight > zero ) THEN
          f_1 = DOT_PRODUCT( R, AD ) + weight * data%rho_1
          f_2 = DOT_PRODUCT( AD, AD ) + weight * data%rho_2
        ELSE
          f_1 = DOT_PRODUCT( R, AD ) ; f_2 = DOT_PRODUCT( AD, AD )
        END IF

!  stop if the slope is positive

        IF ( f_1 > - data%f_1_stop ) THEN
          t_opt = data%t_total ; f_opt = data%f_0
          IF ( summary ) WRITE( out,  "( A, ' f_opt =', ES12.4, ' at t =',     &
         &    ES11.4, ' at start of segment ', I0 )" )                         &
                prefix, f_opt, t_opt, segment
          GO TO 900
        END IF

!  compute the step to the minimizer along the segment

        t_opt = - f_1 / f_2

!  compute the step along s to the boundary of Delta^n, as well as the
!  index i_fixed of the position in FREE of the variable that meets its bound

        IF ( debug )                                                           &
          WRITE( out,  "( A, 14X, '  t           x           s' )" ) prefix
        data%t_break = t_max
        DO i = 1, n_free
          j = FREE( i )
          IF ( D( j ) < zero ) THEN
            t = - X( j ) / D( j )
            IF ( debug )                                                       &
              WRITE( out,  "( A, I8, 3ES12.4 )" ) prefix, j, t, X( j ), D( j )
            IF ( t < data%t_break ) THEN
              i_fixed = i ; data%t_break = t
            END IF
          END IF
        END DO
        IF ( summary ) WRITE( out,  "( A, I8, 5ES12.4 )" ) prefix, segment,    &
          data%f_0, f_1, f_2, data%t_total + data%t_break, data%t_total + t_opt
!       IF ( data%t_break == t_max .AND. debug )                               &
!         WRITE( out,  "( ' s = ', /, ( 5ES12.4 ) )" ) S

!  stop if the minimizer on the segment occurs before the end of the segment

        IF ( t_opt > zero .AND. t_opt <= data%t_break ) THEN
          f_opt = data%f_0 + f_1 * t_opt + half * f_2 * t_opt ** 2
          DO l = 1, n_free
            j = FREE( l )
            X( j ) = X( j ) + t_opt * D( j )
!           IF ( debug ) WRITE( out, "( ' x ', I8, ES12.4 )" ) j, X( j )
          END DO
          t_opt = data%t_total + t_opt
          IF ( summary ) WRITE( out,  "( A, ' f_opt =', ES12.4, ' at t =',     &
         &   ES11.4, ' in segment ', I0 )" ) prefix, f_opt, t_opt, segment
          GO TO 900
        END IF

!  update the total steplength

        data%t_total = data%t_total + data%t_break

!  fix the variable that encounters its bound and adjust FREE

        now_fixed = FREE( i_fixed )
        FREE( i_fixed ) = FREE( n_free )
        FREE( n_free ) = now_fixed
        n_free = n_free - 1
        IF ( debug )                                                           &
          WRITE( out,  "( A, ' fix variable ', I0 )" ) prefix, now_fixed

!  step to the boundary

        X( now_fixed ) = zero
        DO l = 1, n_free
          j = FREE( l )
          X( j ) = X( j ) + data%t_break * D( j )
!         IF ( debug ) WRITE( out, "( ' x ', I8, ES12.4 )" ) j, X( j )
        END DO

!  update the search direction

        data%ete = data%ete - one
        data%s_fixed = D( now_fixed ) ; data%gamma = data%s_fixed / data%ete
!       IF ( debug ) WRITE( out, "( ' data%gamma = ', ES12.4 )" ) data%gamma
        DO i = 1, n_free
          j = FREE( i )
!         IF ( D( j ) >= zero .AND. D( j ) + data%gamma < zero .AND. debug )   &
!           WRITE( out,                                                        &
!              "( ' variable ', I6, ' now a candidate, t = ', ES12.4 )")       &
!              j, - X( j ) / ( D( j ) + data%gamma )
          D( j ) = D( j ) + data%gamma
!         IF ( debug ) WRITE( out,"( ' s ', I8, ES12.4 )" ) j, D( j )
        END DO
        D( now_fixed ) = zero

!  compare calculated and recurred breakpoint

!       IF ( debug ) THEN
!         CALL SLLS_project_onto_simplex( n, V, PROJ, status )
!         WRITE( out,  "( ' status = ', I0 )" ) status
!         CALL SLLS_project_onto_simplex( n, X + data%t_total * D, PROJ, status)
!         WRITE( out, "(' status proj = ', I0 )" ) status
!         WRITE( out,  "( 8X, '        v          proj' )" )
!         DO j = 1, n
!           WRITE( out, "( I8, 2ES12.4 )" ) j, X( j ), PROJ( j )
!         END DO
!       END IF

!  update f_0

        data%f_0 = data%f_0 + f_1 * data%t_break                               &
                     + half * f_2 * data%t_break ** 2

!  if the fixed column of A is only availble by reverse communication, get it

        IF ( data%reverse ) THEN
          status = now_fixed ; RETURN
        END IF

!  re-enter with the required column

  200   CONTINUE

!  update r, Ae and Ad

        IF ( data%present_a ) THEN
          DO l = Ao_ptr( now_fixed ), Ao_ptr( now_fixed + 1 ) - 1
            i = Ao_row( l ) ; a = Ao_val( l )
            AE( i ) = AE( i ) - a
            AD( i ) = AD( i ) - data%s_fixed * a
          END DO
        ELSE IF ( data%reverse ) THEN
          DO l = 1, reverse%nz_out_end
            i = reverse%NZ_out( l ) ; a = reverse%P( l )
            AE( i ) = AE( i ) - a
            AD( i ) = AD( i ) - data%s_fixed * a
          END DO
        ELSE
          data%V( now_fixed ) = one ; NZ_in( 1 ) = now_fixed
          CALL eval_ASPROD( status, userdata, data%V, data%P, NZ_in,           &
                            1_ip_, 1_ip_, data%NZ_out, nz_out_end )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; RETURN
          END IF

          DO l = 1, nz_out_end
            i = data%NZ_out( l ) ; a = data%P( l )
            AE( i ) = AE( i ) - a
            AD( i ) = AD( i ) - data%s_fixed * a
          END DO
        END IF

        R = R + data%t_break * AD
        AD = AD + data%gamma * AE

!  update rho_0, rho_1 and rho_2 if required

        IF ( weight > zero ) THEN
          data%rho_0 = data%rho_0 +                                            &
            data%t_break * ( two * data%rho_1 +  data%t_break * data%rho_2 )
          data%rho_1 = data%rho_1 + data%t_break * data%rho_2 + data%gamma
          data%rho_2 = data%rho_2 - data%s_fixed * ( data%s_fixed + data%gamma )
        END IF

        segment = segment + 1
        IF ( segment < n ) GO TO 100

!  end of main (mock) loop

  900 CONTINUE

!  record free variables

      n_free = 0
      DO j = 1, n
        IF ( X( j ) >= x_zero ) THEN
          n_free = n_free + 1
          FREE( n_free ) = j
        END IF
      END DO

      status = 0

      RETURN

!  End of subroutine SLLS_exact_arc_search

      END SUBROUTINE SLLS_exact_arc_search

! -*-*- S L L S _ I N E X A C T _ A R C _ S E A R C H  S U B R O U T I N E -*-*-

      SUBROUTINE SLLS_inexact_arc_search( n, o, weight, out, summary, debug,   &
                                          prefix, status, X, R, D, S, R_t,     &
                                          t_0, beta, eta, max_steps,           &
!                                         t_max, advance,                      &
                                          n_free, FREE, data, userdata,        &
                                          f_opt, t_opt, Ao_val, Ao_row, Ao_ptr,&
                                          reverse , eval_APROD )

!  Let Delta^n = { s | e^T s = 1, s >= 0 } be the unit simplex. Follow the
!  projection path x(t) = P( x + t d ) from a given x and direction d for
!  a sequence of decreasing/increasing values of t, from an initial value
!  t_0 > 0, to find an approximate local minimizer of the regularized
!  least-squares objective
!
!    f(x) = 1/2 || A_o x - b ||^2 + 1/2 weight || x ||^2 for x = P(x(t))
!
!  The approximation to the arc minimizer we seek is a point x(t_i) for
!  which the Armijo condition
!
!      f(x(t_i)) <= linear(x(t_i),eta)
!                 = f(x) + eta * nabla f(x)^T (x(t_i) - x)   (*)
!
!  where t_i = t_0 * beta^i for some integer i is satisfied
!
!  Proceed as follows:
!
!  1) if the minimizer of f(x) along x + t * d lies on the search arc,
!     this is the required point. Otherwise,
!
!  2) from some specified t_0, check whether (*) is satisfied with i = 0.
!
!  If so (optionally - alternatively simply pick x(t_0))

!  2a) construct an increasing sequence t_i = t_0 * beta^i for i < 0
!     and pick the one before (*) is violated

!  Otherwise

!  2b) construct a decreasing sequence t_i = t_0 * beta^i for i > 0
!     and pick the first for which (*) is satified

!  Progress through the routine is controlled by the parameter status

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  n         (INTEGER) the number of variables
!  o         (INTEGER) the number of observations
!  weight    (REAL) the regularization weight
!  R         (REAL array of length at least o) the residual A x - b
!  D         (REAL array of length at least n) the direction d
!  out       (INTEGER) the output unit for printing
!  summary   (LOGICAL) one line of output per segment if true
!  debug     (LOGICAL) lots of output per segment if true
!  prefix    (CHARACTER string) that is used to prefix any output line
!
!  INPUT/OUTPUT arguments
!
!  status    (INTEGER) the input and return status for/from the package.
!             On initial entry, status must be set to 0.
!             Possible output values are:
!             0 a successful exit
!             < 0 an error exit
!             > 0 the user should form the product p + A v and re-enter the
!                 subroutine with all non-optional arguments unchanged.
!                 the vectors p and v will be provided in reverse%p(:m)
!                 and reverse%v(:n), and the result should overwrite
!                 reverse%p(:m). These arrays should be allocated before use.
!                 This value of status will only occur if Ao_val, Ao_row and
!                 Ao_ptr are absent
!  X         (REAL array of length at least n) the initial point x
!  S         (REAL array of length at least o) the direction p(x+td)-x
!  R_t       (REAL array of length at least o) the residual A p(x+td) - b
!  t_0       (REAL) initial arc length
!  beta      (REAL) arc length reduction factor in (0,1)
!  eta       (REAL) decrease tolerance in (0,1/2)
!  max_steps (INTEGER) the maximum number of steps allowed
!  t_max     (REAL) the largest arc length permitted (t_max >= t_0)
!            (** not used and commented out at present **)
!  advance   (LOGICAL) allow alpha to increase as well as decrease?
!            (** not used and commented out at present **)
!  n_free    (INTEGER) the number of free variables (i.e. variables not at zero)
!  FREE      (INTEGER array of length at least n) FREE(:n_free) are the indices
!             of the free variables
!  data      (structure of type slls_search_data_type) private data that is
!             preserved between calls to the subroutine
!  userdata  (structure of type GALAHAD_userdata_type ) data that may be passed
!             between calls to the evaluation subroutine eval_prod
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  f_opt   (REAL) the optimal objective value
!  t_opt   (REAL) the optimal step length
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!          nonzeros of A, stored by consecutive columns. N.B. If present,
!          Ao_row and Ao_ptr must also be present
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!          of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!           Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!           with Ao_ptr(n+1) pointin to the storage location 1 beyond
!           the last entry in A
!  reverse (structure of type slls_reverse_type) data that is provided by the
!           user when prompted by status > 0
!  eval_aprod (subroutine) that provides products with A (see slls_solve)
!
!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n, o, out, max_steps
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status, n_free
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      LOGICAL, INTENT( IN ):: summary, debug
!     LOGICAL, INTENT( IN ):: advance
      REAL ( KIND = rp_ ), INTENT( OUT ) :: f_opt, t_opt
      REAL ( KIND = rp_ ), INTENT( IN ) :: t_0, beta, eta
!     REAL ( KIND = rp_ ), INTENT( IN ) :: t_max
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: FREE
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: D
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( o ) :: R
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, S
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R_t
      TYPE ( SLLS_search_data_type ), INTENT( INOUT ) :: data
      TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      TYPE ( SLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse
      OPTIONAL :: eval_APROD

!  interface blocks

      INTERFACE
        SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        LOGICAL, INTENT( IN ) :: transpose
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
        END SUBROUTINE eval_APROD
      END INTERFACE

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, l
      REAL ( KIND = rp_ ) :: f_t, gts_t, s_j

!  if a re-entry occurs, branch to the appropriate place in the code

      IF ( status > 0 ) GO TO 200

!  check to see if A has been provided

      IF ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) .AND.                     &
           PRESENT( Ao_val ) ) THEN
        data%present_a = .TRUE. ; data%reverse = .FALSE.
      ELSE
        data%present_a = .FALSE.

!  check to see if access to products with A by reverse communication has
!  been provided

        IF ( PRESENT( reverse ) ) THEN
          data%reverse = .TRUE.

!  check to see if access to products with A by subroutine call has been
!  provided

        ELSE IF ( PRESENT( eval_APROD ) ) THEN
          data%reverse = .FALSE.

!  if none of thses option is available, exit

        ELSE
          status = - 2 ; RETURN
        END IF
      END IF

!  compute r^T r and f = 1/2 || r ||^2 + 1/2 weight || x ||^2

      data%rtr = DOT_PRODUCT( R, R )
      IF ( weight > zero ) THEN
        data%xtx = DOT_PRODUCT( X, X )
        data%f_0 = half * ( data%rtr + weight * data%xtx )
      ELSE
        data%f_0 = half * data%rtr
      END IF

!  ensure that the path goes somewhere

      IF ( MAXVAL( ABS( D ) ) == zero ) THEN
        t_opt = zero ; f_opt = data%f_0 ; status = - 1
        RETURN
      END IF

      IF ( summary ) THEN
        WRITE( out, 2000 ) prefix
        WRITE( out, 2010 ) prefix, 0, zero, zero, data%f_0
      END IF

!  main loop (mock do loop to allow reverse communication)

      data%step = 1 ; data%t = t_0 ; data%backwards = .TRUE.
  100 CONTINUE

!  store the projection P(x + t d) of x + t d onto Delta^n in s

        CALL SLLS_project_onto_simplex( n, X + data%t * D, S, status )
        IF ( weight > zero ) data%xtx = DOT_PRODUCT( S, S )

!  compute the step s_t from x to this point

        S = S - X

!  compute r_t = r + A s_t

!write(6,"( ' R =', / ( 5ES12.4 ) )" ) R
!write(6,"( ' S =', / ( 5ES12.4 ) )" ) S
!  if A is explicit, form the sum directly

        IF ( data%present_a ) THEN
          R_t = R
          DO j = 1, n
            s_j = S( j )
            DO l = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
              i = Ao_row( l )
              R_t( i ) = R_t( i ) + Ao_val( l ) * s_j
            END DO
          END DO

!  if A is only availble by reverse communication, obtain the sum in that way

        ELSE IF ( data%reverse ) THEN
!         reverse%P( : o ) = R
          reverse%V( : n ) = S
          status = 2
          RETURN

!  otherwise form the sum by a subroutine call

        ELSE
          R_t = R
          CALL eval_APROD( status, userdata, .FALSE., S, R_t )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; RETURN
          END IF
        END IF

!  re-enter with the required sum

  200   CONTINUE
        IF ( data%reverse ) R_t = R + reverse%P( : o )
        IF ( debug ) WRITE( out, "( ' R_t =', / ( 5ES12.4 ) )" ) R

!  compute f_t = 1/2 || r_t ||^2 + 1/2 weight || x_t ||^2

        f_t = half * DOT_PRODUCT( R_t, R_t )
        IF ( weight > zero ) f_t = f_t + half * weight * data%xtx

!  compute g^T s_t = r^T A s_t = r^T r_t - r^T r + weight * x^T s_t

        gts_t = DOT_PRODUCT( R_t, R ) - data%rtr
        IF ( weight > zero ) gts_t = gts_t + weight * DOT_PRODUCT( X, S )

        IF ( debug ) WRITE( out, 2000 ) prefix
        IF ( summary ) WRITE( out, 2010 )                                      &
          prefix, data%step, data%t, TWO_NORM( S ), f_t

!  test for sufficient decrease

        IF ( data%backwards ) THEN ! this is irrelevant at present

!  exit if the decrease is sufficient with the optimal x, f and status arrays

          IF ( f_t <= data%f_0 + eta * gts_t ) THEN
            X = X + S ; f_opt = f_t ; t_opt = data%t
            GO TO 900
          END IF

!  insufficient decrease; reduce t

          data%t = data%t * beta
        ELSE
        END IF

        data%step = data%step + 1
        IF ( max_steps >= 0 .AND. data%step > max_steps ) THEN
          status = - 2
          RETURN
        END IF
        GO TO 100

!  end of main (mock) loop

  900 CONTINUE

!  record free variables

      n_free = 0
      DO j = 1, n
        IF ( X( j ) >= x_zero ) THEN
          n_free = n_free + 1
          FREE( n_free ) = j
        END IF
      END DO

      status = 0

      RETURN

!  non-executable statements

 2000 FORMAT( A, '       k      t           s           f' )
 2010 FORMAT( A, I8, 3ES12.4 )

!  End of subroutine SLLS_inexact_arc_search

      END SUBROUTINE SLLS_inexact_arc_search

! -*-*-*-*-*-*-*-*-  S L L S _ C G L S   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

      SUBROUTINE SLLS_cgls( o, n, n_free, weight, out, summary, debug, prefix, &
                            f, X, R, FREE, stop_cg_relative, stop_cg_absolute, &
                            iter, maxit, data, userdata, status, alloc_status, &
                            bad_alloc, Ao_ptr, Ao_row, Ao_val, eval_AFPROD,    &
                            eval_PREC, DPREC, reverse, preconditioned, B )

!  Find the minimizer of the constrained (regularized) least-squares
!  objective function

!    f(x) =  1/2 || A_o x - b ||_2^2 + 1/2 weight * ||x||_2^2

!  for which certain components of x are fixed at zero, and the remainder
!  sum to zero

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  o       (INTEGER) the number of rows of A = number of observations
!  n       (INTEGER) the number of columns of A = number of variables
!  n_free  (INTEGER) the number of components of x that are not fixed at zero
!  weight  (REAL) the positive regularization weight (<= zero is zero)
!  out     (INTEGER) the output unit for printing
!  summary (LOGICAL) one line of output per segment if true
!  debug   (LOGICAL) lots of output per segment if true
!  prefix  (CHARACTER string) that is used to prefix any output line
!  FREE    (INTEGER array of length at least n_free) specifies which gives
!         the indices of variables that are not fixed at zero
!  stop_cg_relative, stop_cg_absolute (REAL) the iteration will stop as
!          soon as the gradient of the objective is smaller than
!          MAX( stop_cg_relative * norm initial gradient, stop_cg_absolute)
!  maxit   (INTEGER) the maximum number of iterations allowed (<0 = infinite)
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
!  R       (REAL array of length at least o) the residual A x - b. On input,
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
!              for i = FREE( : n_free ), should be returned in reverse%P.
!              The argument reverse%eval_status should be set to
!              0 if the calculation succeeds, and to a nonzero value otherwise.
!            3 [Dense in, sparse out] The components of the product
!              p = A^T * v, where v is stored in reverse%V, should be
!              returned in reverse%P. Only components p_i with indices
!              i = FREE( : n_free ) need be assigned, the remainder will be
!              ignored. The argument reverse%eval_status should be set to 0
!              if the calculation succeeds, and to a nonzero value otherwise.
!            4 the product p = P^-1 v between the inverse of the preconditionr
!              P and the vector v, where v is stored in reverse%V, should be
!              returned in reverse%P. Only the components of v with indices
!              i = FREE( : n_free ) are nonzero, and only the components of
!              p with indices i = FREE( : n_free ) are needed. The argument
!              reverse%eval_status should  be set to 0 if the calculation
!              succeeds, and to a nonzero value otherwise.
!          < 0 an error exit
!
!  WORKSPACE
!
!  data (structure of type SLLS_subproblem_data_type)
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!          nonzeros of A, stored by consecutive columns
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!          of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!          Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!          with Ao_ptr(n+1) pointin to the storage location 1 beyond
!          the last entry in A
!  weight  (REAL) the positive regularization weight (absent = zero)
!  eval_AFPROD subroutine that performs products with A, see the argument
!           list for SLLS_solve
!  eval_PREC subroutine that performs the preconditioning operation p = P v
!            see the argument list for SLLS_solve
!  DPREC   (REAL array of length n) the values of a diagonal preconditioner
!           that aims to approximate A^T A
!  preconditioned (LOGICAL) prsent and set true is there a preconditioner
!  reverse (structure of type SLLS_reverse_type) used to communicate
!           reverse communication data to and from the subroutine

!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: o, n, n_free, maxit, out
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iter, status
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: alloc_status
      LOGICAL, INTENT( IN ) :: summary, debug
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: f
      REAL ( KIND = rp_ ), INTENT( IN ) :: stop_cg_relative, stop_cg_absolute
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
      INTEGER ( KIND = ip_ ), DIMENSION( n_free ), INTENT( IN ) :: FREE
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R
      TYPE ( SLLS_subproblem_data_type ), INTENT( INOUT ) :: data
      TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      OPTIONAL :: eval_AFPROD, eval_PREC
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: DPREC
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( o ) :: B
      LOGICAL, OPTIONAL, INTENT( IN ) :: preconditioned
      TYPE ( SLLS_reverse_type ), OPTIONAL, INTENT( INOUT ) :: reverse

!  interface blocks

      INTERFACE
        SUBROUTINE eval_AFPROD( status, userdata, transpose, V, P,             &
                                FREE, n_free )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        LOGICAL, INTENT( IN ) :: transpose
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_free
        INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: FREE
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
        END SUBROUTINE eval_AFPROD
      END INTERFACE

      INTERFACE
        SUBROUTINE eval_PREC( status, userdata, V, P )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
        END SUBROUTINE eval_PREC
      END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, H, X_l, X_u, X_s, R_s, D_s, alpha_max, feas_tol, max_segments,
!      out, print_level, prefix

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l
      REAL ( KIND = rp_ ) :: val, alpha, beta, curv, norm_r, norm_g
      CHARACTER ( LEN = 80 ) :: array_name

!  enter or re-enter the package and jump to appropriate re-entry point

      IF ( status <= 1 ) data%branch = 10

      SELECT CASE ( data%branch )
      CASE ( 10 ) ; GO TO 10       ! status = 1
      CASE ( 20 ) ; GO TO 20       ! status = 3
      CASE ( 70 ) ; GO TO 70       ! status = 4
      CASE ( 80 ) ; GO TO 80       ! status = 4
      CASE ( 90 ) ; GO TO 90       ! status = 4
      CASE ( 110 ) ; GO TO 110     ! status = 2
      CASE ( 120 ) ; GO TO 120     ! status = 3
      CASE ( 140 ) ; GO TO 140     ! status = 4
      CASE ( 150 ) ; GO TO 150     ! status = 4
      END SELECT

!  initial entry

   10 CONTINUE
      iter = 0

!  check that it is possible to access A in some way

      data%present_a = PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) .AND.         &
                       PRESENT( Ao_val )
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

!  allocate workspace

      array_name = 'slls_cgls: data%P'
      CALL SPACE_resize_array( n, data%P, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'slls_cgls: data%Q'
      CALL SPACE_resize_array( o, data%Q, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'slls_cgls: data%G'
      CALL SPACE_resize_array( n, data%G, status, alloc_status,                &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'slls_cgls: data%PG'
      CALL SPACE_resize_array( n, data%PG, status, alloc_status,               &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      IF ( data%preconditioned ) THEN
        array_name = 'slls_cgls: data%PE'
        CALL SPACE_resize_array( n, data%PE, status, alloc_status,             &
               array_name = array_name, bad_alloc = bad_alloc, out = out )
        IF ( status /= GALAHAD_ok ) GO TO 900
      END IF

      IF ( summary ) WRITE( out, "( /, A, ' ** cgls entered',                  &
     &    ' (n = ', I0, ', free = ', I0, ') ** ' )" ) prefix, n, n_free

!  compute the initial function value f = 1/2 ||r||^2

      norm_r = TWO_NORM( R )
      f = half * norm_r ** 2

!  exit if there are no free variables

      IF ( n_free == 0 ) THEN
        status = GALAHAD_ok
        GO TO 800
      END IF

!  compute the gradient g = A^T r at the initial point, and its norm

!  a) evaluation directly via A

      IF ( data%present_a ) THEN
        DO k = 1, n_free
          j = FREE( k )
          data%G( j ) = zero
          DO l = Ao_ptr( j ) , Ao_ptr( j + 1 ) - 1
            i = Ao_row( l )
            data%G( j ) = data%G( j ) + Ao_val( l ) * R( i )
          END DO
        END DO

!  b) evaluation via matrix-vector product call

      ELSE IF ( data%present_afprod ) THEN
        CALL eval_AFPROD( status, userdata, transpose = .TRUE., V = R,         &
                          P = data%G, FREE = FREE, n_free = n_free )
        IF ( status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_evaluation ; GO TO 900
        END IF

!  c) evaluation via reverse communication

      ELSE
        reverse%V( : o ) = R( : o )
        data%branch = 20 ; status = 3
        RETURN
      END IF

!  return from reverse communication

  20  CONTINUE
      IF ( data%reverse_afprod ) THEN
        IF ( reverse%eval_status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_evaluation ; GO TO 900
        END IF
        IF ( n_free < n ) THEN
          data%G( FREE( : n_free ) ) = reverse%P( FREE( : n_free ) )
        ELSE
          data%G( : n ) = reverse%P( : n )
        END IF
      END IF

!  include the gradient of the regularization term if present

      IF ( data%regularization ) THEN
        IF ( n_free < n ) THEN
          data%G( FREE( : n_free ) )                                           &
            = data%G( FREE( : n_free ) ) + weight * X( FREE( : n_free ) )
        ELSE
          data%G( : n ) = data%G( : n ) + weight * X
        END IF
        f = f + half * weight * TWO_NORM( X ) ** 2
      END IF

!  set preconditioned vector of ones P^{-1} e

!  a) evaluation via preconditioner-inverse-vector product call

      IF ( data%preconditioned ) THEN
!write(6,"(' g', 4ES12.4)" ) data%G( : n )
        IF ( data%present_dprec ) THEN
          IF ( n_free < n ) THEN
            data%PE( FREE( : n_free ) ) = one / DPREC( FREE( : n_free ) )
          ELSE
            data%PE( : n ) = one / DPREC( : n )
          END IF

!  b) evaluation via preconditioner-inverse-vector product call

        ELSE IF ( data%present_prec ) THEN
          data%PG( : n ) = one
          CALL eval_PREC( status, userdata, V = data%PG, P = data%PE )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%V( : n ) = one
          data%branch = 70 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   70 CONTINUE
      IF ( data%preconditioned ) THEN
        IF (  data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PE( : n ) = reverse%P( : n )
        END IF

!  compute the product e^T P^{-1} e

        IF ( n_free < n ) THEN
          data%etpe = SUM( data%PE( FREE( : n_free ) ) )
        ELSE
          data%etpe = SUM( data%PE( : n ) )
        END IF
      ELSE
        data%etpe = REAL( n_free, KIND = rp_ )
      END IF

!  set the initial preconditioned gradient

!  a) evaluation via preconditioner-inverse-vector product call

      IF ( data%preconditioned ) THEN
!write(6,"(' g', 4ES12.4)" ) data%G( : n )
        IF ( data%present_dprec ) THEN
          IF ( n_free < n ) THEN
            data%PG( FREE( : n_free ) )                                        &
              = data%G( FREE( : n_free ) ) / DPREC( FREE( : n_free ) )
          ELSE
            data%PG( : n ) = data%G( : n ) / DPREC( : n )
          END IF

!  b) evaluation via preconditioner-inverse-vector product call

        ELSE IF ( data%present_prec ) THEN
          CALL eval_PREC( status, userdata, V = data%G, P = data%PG )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%V( : n ) = data%G( : n )
          data%branch = 80 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   80 CONTINUE
      IF ( data%preconditioned ) THEN
        IF (  data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PG( : n ) = reverse%P( : n )
        END IF
!write(6,"(' s', 4ES12.4)" ) data%PG( : n )

!  find its projection onto e^T s = 0,

        IF ( n_free < n ) THEN
          val = SUM( data%PG( FREE( : n_free ) ) ) / data%etpe
          DO i = 1, n_free
            j = FREE( i )
            data%PG( j ) = data%PG( j ) - val * data%PE( j )
          END DO
        ELSE
          val = SUM( data%PG ) / data%etpe
          data%PG = data%PG - val * data%PE
        END IF

!  or set the initial unpreconditioned search direction

      ELSE
        IF ( n_free < n ) THEN
          val = SUM( data%G( FREE( : n_free ) ) ) / data%etpe
          data%PG( FREE( : n_free ) ) = data%G( FREE( : n_free ) ) - val
        ELSE
          val = SUM( data%G ) / data%etpe
          data%PG = data%G - val
        END IF
      END IF

!write(6,"(' pg^Te', ES12.4)" ) SUM( data%PG( FREE( : n_free ) ) )

!  as a precaution, re-project

!  a) evaluation via preconditioner-inverse-vector product call

      IF ( data%preconditioned ) THEN
!write(6,"(' g', 4ES12.4)" ) data%G( : n )
        IF ( data%present_dprec ) THEN
          IF ( n_free < n ) THEN
            data%PG( FREE( : n_free ) )                                        &
              = data%PG( FREE( : n_free ) ) / DPREC( FREE( : n_free ) )
          ELSE
            data%PG( : n ) = data%PG( : n ) / DPREC( : n )
          END IF

!  b) evaluation via preconditioner-inverse-vector product call

        ELSE IF ( data%present_prec ) THEN
          CALL eval_PREC( status, userdata, V = data%PG, P = data%PG )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%V( : n ) = data%PG( : n )
          data%branch = 90 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   90 CONTINUE

!  form the re-projection

      IF ( data%preconditioned ) THEN
        IF (  data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PG( : n ) = reverse%P( : n )
        END IF

        IF ( n_free < n ) THEN
          val = SUM( data%PG( FREE( : n_free ) ) ) / data%etpe
          DO i = 1, n_free
            j = FREE( i )
            data%PG( j ) = data%PG( j ) - val * data%PE( j )
          END DO
        ELSE
          val = SUM( data%PG ) / data%etpe
          data%PG = data%PG - val * data%PE
        END IF
      ELSE
        IF ( n_free < n ) THEN
          val = SUM( data%PG( FREE( : n_free ) ) ) / data%etpe
          data%PG( FREE( : n_free ) ) = data%PG( FREE( : n_free ) ) - val
        ELSE
          val = SUM( data%PG ) / data%etpe
          data%PG = data%PG - val
        END IF
      END IF

!write(6,"(' pg^Te re-projected', ES12.4)" ) SUM( data%PG( FREE( : n_free ) ) )

!  set the initial preconditioned search direction from the projected
!  preconditioned gradient

      data%P( FREE( : n_free ) ) = - data%PG( FREE( : n_free ) )

!  compute the length of the projected preconditioned gradient

      IF ( n_free < n ) THEN
        data%gamma = DOT_PRODUCT( data%G( FREE( : n_free ) ),                  &
                                  data%PG( FREE( : n_free ) ) )
      ELSE
        data%gamma = DOT_PRODUCT( data%G( : n ), data%PG( : n ) )
      END IF
      norm_g = SQRT( MAX( data%gamma, zero ) )

!  set the cg stopping tolerance

      data%stop_cg = MAX( stop_cg_relative * norm_g, stop_cg_absolute )
      IF ( summary ) WRITE( out, "( A, '    stopping tolerance =',             &
     &   ES11.4, / )" ) prefix, data%stop_cg

!  print details of the intial point

      IF ( summary ) THEN
        WRITE( out, 2000 )
        WRITE( out, 2010 ) iter, f, norm_g
!write(6,*) ' ||pg|| = ', SQRT( DOT_PRODUCT( data%PG( FREE( : n_free )  ), &
!                                            data%PG( FREE( : n_free )  ) ) )
      END IF

! test for convergence

      IF ( norm_g <= data%stop_cg ) THEN
        status = GALAHAD_ok ; GO TO 800
      END IF

!  ---------
!  main loop
!  ---------

  100 CONTINUE  ! mock iteration loop
        iter = iter + 1

!       IF ( iter > 3 ) stop

!  check that the iteration limit has not been reached

        IF ( iter > maxit ) THEN
          status = GALAHAD_error_max_iterations ; GO TO 800
        END IF

! form q = A p

!  a) evaluation directly via A

!write(6,"( ' p ', /, ( 5ES12.4 ) )" ) data%P( : n )
        IF ( data%present_a ) THEN
          data%Q( : o ) = zero
          DO k = 1, n_free
            j = FREE( k ) ; val = data%P( j )
            DO l = Ao_ptr( j ) , Ao_ptr( j + 1 ) - 1
              i = Ao_row( l )
              data%Q( i ) = data%Q( i ) + Ao_val( l ) * val
            END DO
          END DO

!  b) evaluation via matrix-vector product call

        ELSE IF ( data%present_afprod ) THEN
          CALL eval_AFPROD( status, userdata, transpose = .FALSE., V = data%P, &
                            P = data%Q, FREE = FREE, n_free = n_free )
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
          data%Q( : o ) = reverse%P( : o )
        END IF

!  compute the step length to the minimizer along the line x + alpha p

        curv = TWO_NORM( data%Q( : o ) ) ** 2
        IF ( data%regularization ) THEN
          IF ( n_free < n ) THEN
            curv = curv + weight *                                             &
              TWO_NORM( data%P( FREE( : n_free ) ) ) ** 2
          ELSE
            curv = curv + weight * TWO_NORM( data%P( : n ) ) ** 2
          END IF
        END IF
        IF ( curv == zero ) curv = epsmch
        alpha = data%gamma / curv

!  update the estimate of the minimizer, x, and its residual, r

        IF ( n_free < n ) THEN
          X( FREE( : n_free ) )                                                &
            = X( FREE( : n_free ) ) + alpha * data%P( FREE( : n_free ) )
        ELSE
          X = X + alpha * data%P( : n )
        END IF
        R = R + alpha * data%Q( : o )
!write(6,"( ' q ', /, ( 5ES12.4 ) )" ) data%Q( : o )

!  update the value of the objective function

        f = f - half * alpha * alpha * curv
        norm_r = SQRT( two * f )

!  compute the gradient g = A^T r at x and its norm

!  a) evaluation directly via A

!write(6,"( ' r ', /, ( 5ES12.4 ) )" ) R( : o )
        IF ( data%present_a ) THEN
          DO k = 1, n_free
            j = FREE( k )
            data%G( j ) = zero
            DO l = Ao_ptr( j ) , Ao_ptr( j + 1 ) - 1
              i = Ao_row( l )
              data%G( j ) = data%G( j ) + Ao_val( l ) * R( i )
            END DO
          END DO

!  b) evaluation via matrix-vector product call

        ELSE IF ( data%present_afprod ) THEN
          CALL eval_AFPROD( status, userdata, transpose = .TRUE., V = R,       &
                            P = data%G, FREE = FREE, n_free = n_free )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%P( : n ) = zero
          reverse%V( : o ) = R
          data%branch = 120 ; status = 3
          RETURN
        END IF

!  return from reverse communication

  120   CONTINUE
        IF ( data%reverse_afprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          IF ( n_free < n ) THEN
            data%G( FREE( : n_free ) ) = reverse%P( FREE( : n_free ) )
          ELSE
            data%G( : n ) = reverse%P( : n )
          END IF
        END IF
!write(6,"( ' g ', /, ( 5ES12.4 ) )" ) data%G( : n )

!  include the gradient of the regularization term if present

        IF ( data%regularization ) THEN
          IF ( n_free < n ) THEN
            data%G( FREE( : n_free ) ) =                                       &
              data%G( FREE( : n_free ) ) + weight * X( FREE( : n_free ) )
          ELSE
            data%G( : n ) = data%G( : n ) + weight * X
          END IF
        END IF

!  compute the preconditioned gradient

!  a) evaluation via preconditioner-inverse-vector product call

        IF ( data%preconditioned ) THEN
          IF ( data%present_dprec ) THEN
            IF ( n_free < n ) THEN
              data%PG( FREE( : n_free ) )                                      &
                = data%G( FREE( : n_free ) ) / DPREC( FREE( : n_free ) )
            ELSE
              data%PG( : n ) = data%G( : n ) / DPREC( : n )
            END IF

!  b) evaluation via preconditioner-inverse-vector product call

          ELSE IF ( data%present_prec ) THEN
            CALL eval_PREC( status, userdata, V = data%G, P = data%PG )
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
        data%gamma_old = data%gamma

 !  compute the length of the preconditioned gradient

        IF ( data%preconditioned ) THEN
          IF ( data%reverse_prec ) THEN
            IF ( reverse%eval_status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%PG( : n ) = reverse%P( : n )
          END IF

!  find the projected preconditioned gradient by projection onto e^T s = 0

          IF ( n_free < n ) THEN
            val = SUM( data%PG( FREE( : n_free ) ) ) / data%etpe
            DO i = 1, n_free
              j = FREE( i )
              data%PG( j ) = data%PG( j ) - val * data%PE( j )
            END DO
          ELSE
            val = SUM( data%PG ) / data%etpe
            data%PG = data%PG - val * data%PE
          END IF
        ELSE
          IF ( n_free < n ) THEN
            val = SUM( data%G( FREE( : n_free ) ) ) / data%etpe
            data%PG( FREE( : n_free ) ) = data%G( FREE( : n_free ) ) - val
          ELSE
            val = SUM( data%G ) / data%etpe
            data%PG = data%G - val
          END IF
        END IF

!write(6,"(' pg^Te', ES12.4)" ) SUM( data%PG( FREE( : n_free ) ) )

!  as a precaution, re-project

!  a) evaluation via preconditioner-inverse-vector product call

        IF ( data%preconditioned ) THEN
          IF ( data%present_dprec ) THEN
            IF ( n_free < n ) THEN
              data%PG( FREE( : n_free ) )                                      &
                = data%PG( FREE( : n_free ) ) / DPREC( FREE( : n_free ) )
            ELSE
              data%PG( : n ) = data%PG( : n ) / DPREC( : n )
            END IF

!  b) evaluation via preconditioner-inverse-vector product call

          ELSE IF ( data%present_prec ) THEN
            CALL eval_PREC( status, userdata, V = data%PG, P = data%PG )
            IF ( status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF

!  c) evaluation via reverse communication

          ELSE
            reverse%V( : n ) = data%PG( : n )
            data%branch = 150 ; status = 4
            RETURN
          END IF
        END IF

!  return from reverse communication

  150   CONTINUE

!  form the re-projection

        IF ( data%preconditioned ) THEN
          IF ( data%reverse_prec ) THEN
            IF ( reverse%eval_status /= GALAHAD_ok ) THEN
              status = GALAHAD_error_evaluation ; GO TO 900
            END IF
            data%PG( : n ) = reverse%P( : n )
          END IF

          IF ( n_free < n ) THEN
            val = SUM( data%PG( FREE( : n_free ) ) ) / data%etpe
            DO i = 1, n_free
              j = FREE( i )
              data%PG( j ) = data%PG( j ) - val * data%PE( j )
            END DO
          ELSE
            val = SUM( data%PG ) / data%etpe
            data%PG = data%PG - val * data%PE
          END IF
        ELSE
          IF ( n_free < n ) THEN
            val = SUM( data%PG( FREE( : n_free ) ) ) / data%etpe
            data%PG( FREE( : n_free ) ) = data%PG( FREE( : n_free ) ) - val
          ELSE
            val = SUM( data%PG ) / data%etpe
            data%PG = data%PG - val
          END IF
        END IF
!write(6,"(' pg^Te re-projected', ES12.4)" ) SUM( data%PG( FREE( : n_free ) ) )

!  compute the length of the projected preconditioned gradient

        IF ( n_free < n ) THEN
          data%gamma = DOT_PRODUCT( data%G( FREE( : n_free ) ),                &
                                    data%PG( FREE( : n_free ) ) )
        ELSE
          data%gamma = DOT_PRODUCT( data%G( : n ), data%PG( : n ) )
        END IF
        norm_g = SQRT( MAX( zero, data%gamma ) )

!  print details of the current step

        IF ( debug ) WRITE( out, 2000 )
        IF ( summary ) WRITE( out, 2010 ) iter, f, norm_g
!write(6,*) ' ||pg||, ||g|| = ', &
!  SQRT( DOT_PRODUCT( data%PG( FREE( : n_free )  ), &
!                     data%PG( FREE( : n_free )  ) ) ), &
!  SQRT( DOT_PRODUCT( data%G( FREE( : n_free )  ), &
!                     data%G( FREE( : n_free )  ) ) )

! test for convergence

        IF ( norm_g <= data%stop_cg ) THEN
          status = GALAHAD_ok ; GO TO 800
        END IF

!  compute the next preconditioned search direction, p

        beta = data%gamma / data%gamma_old

        IF ( n_free < n ) THEN
          data%P( FREE( : n_free ) )                                           &
            = - data%PG( FREE( : n_free ) ) + beta * data%P( FREE( : n_free ) )
        ELSE
          data%P( : n ) = - data%PG( : n ) + beta * data%P( : n )
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

!  End of subroutine SLLS_cgls

      END SUBROUTINE SLLS_cgls

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!- G A L A H A D -  S L L S _ i m p o r t _ w i t h o u t _ a  S U B R O U TINE

     SUBROUTINE SLLS_import_without_a( control, data, status, n, o )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SLLS_solve
!
!  data is a scalar variable of type SLLS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0 has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables (columns of Ao)
!
!  o is a scalar variable of type default integer, that holds the number of
!   residuals (rows of Ao)
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug
     data%slls_control = control

     error = data%slls_control%error
     space_critical = data%slls_control%space_critical
     deallocate_error_fatal = data%slls_control%space_critical

!  record that the Jacobian is not explicitly available

     data%explicit_a = .FALSE.

!  allocate vector space if required

     array_name = 'slls: data%prob%X'
     CALL SPACE_resize_array( n, data%prob%X,                                  &
            data%slls_inform%status, data%slls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
     IF ( data%slls_inform%status /= 0 ) GO TO 900

     array_name = 'slls: data%prob%G'
     CALL SPACE_resize_array( n, data%prob%G,                                  &
            data%slls_inform%status, data%slls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
     IF ( data%slls_inform%status /= 0 ) GO TO 900

     array_name = 'slls: data%prob%B'
     CALL SPACE_resize_array( o, data%prob%B,                                  &
            data%slls_inform%status, data%slls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
     IF ( data%slls_inform%status /= 0 ) GO TO 900

     array_name = 'slls: data%prob%R'
     CALL SPACE_resize_array( o, data%prob%R,                                  &
            data%slls_inform%status, data%slls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
     IF ( data%slls_inform%status /= 0 ) GO TO 900

     array_name = 'slls: data%prob%Z'
     CALL SPACE_resize_array( n, data%prob%Z,                                  &
            data%slls_inform%status, data%slls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
     IF ( data%slls_inform%status /= 0 ) GO TO 900

!  put data into the required components of the qpt storage type

     data%prob%n = n ; data%prob%o = o

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%slls_inform%status
     RETURN

!  End of subroutine SLLS_import_without_a

     END SUBROUTINE SLLS_import_without_a

!-*-*-*-  G A L A H A D -  S L L S _ i m p o r t _ S U B R O U T I N E -*-*-*-

     SUBROUTINE SLLS_import( control, data, status, n, o,                      &
                             Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SLLS_solve
!
!  data is a scalar variable of type SLLS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, o >= 0 or requirement that Ao_type contains
!       its relevant string 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS',
!       'COORDINATE', 'SPARSE_BY_ROWS', or 'SPARSE_BY_COLUMNS'
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables (columns of Ao)
!
!  o is a scalar variable of type default integer, that holds the number of
!   residuals (rows of Ao)
!
!  Ao_type is a character string that specifies the design matrix storage
!   scheme used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   or 'absent', the latter if m = 0; lower or upper case variants are allowed
!
!  Ao_ne is a scalar variable of type default integer, that holds the number of
!   entries in Ao in the sparse co-ordinate storage scheme. It need not be set
!  for any of the other schemes.
!
!  Ao_row is a rank-one array of type default integer, that holds the row
!   indices Ao in the sparse co-ordinate storage scheme. It need not be set
!   for any of the other schemes, and in this case can be of length 0
!
!  Ao_col is a rank-one array of type default integer, that holds the column
!   indices of Ao in either the sparse co-ordinate, or the sparse row-wise
!   storage scheme. It need not be set when the dense scheme is used, and
!   in this case can be of length 0
!
!  Ao_ptr is a rank-one array of dimension max(o+1,n+1) and type default
!   integer, that holds the starting position of each row of J, as well as the
!   total number of entries plus one, in the sparse row-wise storage scheme,
!   or the starting position of each column of Ao, as well as the total
!   number of entries plus one, in the sparse column-wise storage scheme.
!   It need not be set when the other schemes are used, and in this case
!   can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, Ao_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: Ao_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug

!  assign space for vector data

     CALL SLLS_import_without_a( control, data, status, n, o )
     IF ( status /= GALAHAD_ready_to_solve ) GO TO 900

     error = data%slls_control%error
     space_critical = data%slls_control%space_critical
     deallocate_error_fatal = data%slls_control%space_critical

!  record that the Jacobian is explicitly available

     data%explicit_a = .TRUE.

!  set A appropriately in the qpt storage type

     SELECT CASE ( Ao_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( Ao_row ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%slls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'COORDINATE',                          &
                     data%slls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = Ao_ne

       array_name = 'slls: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_col ) ) ) THEN
         data%slls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_ROWS',                      &
                     data%slls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( o + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( o + 1 )
       END IF

       array_name = 'slls: data%prob%Ao%ptr'
       CALL SPACE_resize_array( o + 1, data%prob%Ao%ptr,                       &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%prob%Ao%col'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%col,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 )
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : o + 1 ) = Ao_ptr( : o + 1 ) + 1
         data%prob%Ao%col( : data%prob%Ao%ne ) = Ao_col( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'sparse_by_columns', 'SPARSE_BY_COLUMNS' )
       IF ( .NOT. ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) ) ) THEN
         data%slls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%prob%Ao%type, 'SPARSE_BY_COLUMNS',                   &
                     data%slls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       IF ( data%f_indexing ) THEN
         data%prob%Ao%ne = Ao_ptr( n + 1 ) - 1
       ELSE
         data%prob%Ao%ne = Ao_ptr( n + 1 )
       END IF
       array_name = 'slls: data%prob%Ao%ptr'
       CALL SPACE_resize_array( n + 1, data%prob%Ao%ptr,                       &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%prob%Ao%row'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%row,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 )
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne )
       ELSE
         data%prob%Ao%ptr( : n + 1 ) = Ao_ptr( : n + 1 ) + 1
         data%prob%Ao%row( : data%prob%Ao%ne ) = Ao_row( : data%prob%Ao%ne ) + 1
       END IF

     CASE ( 'dense_by_rows', 'DENSE_BY_ROWS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE_BY_ROWS',                       &
                     data%slls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'slls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

     CASE ( 'dense_by_columns', 'DENSE_BY_COLUMNS' )
       CALL SMT_put( data%prob%Ao%type, 'DENSE_BY_COLUMNS',                    &
                     data%slls_inform%alloc_status )
       data%prob%Ao%n = n ; data%prob%Ao%m = o
       data%prob%Ao%ne = o * n

       array_name = 'slls: data%prob%Ao%val'
       CALL SPACE_resize_array( data%prob%Ao%ne, data%prob%Ao%val,             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%slls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%slls_inform%status
     RETURN

!  End of subroutine SLLS_import

     END SUBROUTINE SLLS_import

!-  G A L A H A D -  S L L S _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE SLLS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See SLLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%slls_control = control

!  flag a successful call

     status = GALAHAD_ready_to_solve
     RETURN

!  end of subroutine SLLS_reset_control

     END SUBROUTINE SLLS_reset_control

!-  G A L A H A D -  S L L S _ s o l v e _ g i v e n _ a  S U B R O U T I N E  -

     SUBROUTINE SLLS_solve_given_a( data, userdata, status, Ao_val, B,         &
                                    X, Z, R, G, X_stat, eval_PREC )

!  solve the bound-constrained linear least-squares problem whose structure
!  was previously imported. See SLLS_solve for a description of the required
!  arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type SLLS_full_data_type used for internal data
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be
!   used to pass user data to and from the eval_PREC subroutine (see below).
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
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, slls_solve above.
!
!  Ao_val is a rank-one array of type default real, that holds the values of
!   the constraint Jacobian A in the storage scheme specified in slls_import.
!
!  B is a rank-one array of dimension o and type default
!   real, that holds the vector of observations, b.
!   The i-th component of B, i = 1, ... , o, contains (b)_i.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  Z is a rank-one array of dimension n and type default
!   real, that holds the vector of the dual variables, z.
!   The j-th component of Z, j = 1, ... , n, contains (z)_j.
!
!  R is a rank-one array of dimension o and type default
!   real, that holds the vector of residuals, r = A x - b on exit.
!   The i-th component of R, i = 1, ... , o, contains (r)_i.
!
!  G is a rank-one array of dimension n and type default
!   real, that holds the gradient, g = A^T c on exit.
!   The j-th component of G, j = 1, ... , n, contains (g)_j.
!
!  X_stat is a rank-one array of dimension n and type default integer,
!   that may be set by the user on entry to indicate which of the variables
!   are to be included in the initial working set. If this facility is
!   required, the component control%cold_start must be set to 0 on entry;
!   X_stat need not be set if control%cold_start is nonzero. On exit,
!   X_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   X_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  eval_PREC is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product P^{-1} * v of the given
!   preconditioner P and vector v stored in V must be returned in P.
!   The intention is that P is an approximation to A^T A. The status variable
!   should be set to 0 unless the product is impossible in which case status
!   should be set to a nonzero value. If eval_PREC is not present, SLLS_solve
!   will return to the user each time a preconditioning operation is required
!   (see reverse above) when control%preconditioner is not 0 or 1.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: Ao_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R, G
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : ) :: X_stat
     OPTIONAL :: eval_PREC

!  interface blocks

     INTERFACE
       SUBROUTINE eval_PREC( status, userdata, V, P )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       END SUBROUTINE eval_PREC
     END INTERFACE

!  local variables

     INTEGER ( KIND = ip_ ) :: n, o

!  check that space for the constraint Jacobian has been provided

     IF ( .NOT. data%explicit_a ) GO TO 900

!  recover the dimensions

     n = data%prob%n ; o = data%prob%o

!  save the observations

     data%prob%B( : o ) = B( : o )

!  save the initial primal and dual variables

     data%prob%X( : n ) = X( : n )
     data%prob%Z( : n ) = Z( : n )

!  save the Jacobian entries

     IF ( data%prob%Ao%ne > 0 )                                                &
       data%prob%Ao%val( : data%prob%Ao%ne ) = Ao_val( : data%prob%Ao%ne )

!  call the solver

     CALL SLLS_solve( data%prob, X_stat, data%slls_data, data%slls_control,    &
                      data%slls_inform, userdata, eval_PREC = eval_PREC )
     status = data%slls_inform%status

!  recover the optimal primal and dual variables

     X( : n ) = data%prob%X( : n )
     Z( : n ) = data%prob%Z( : n )

!  recover the residual value and gradient

     R( : o ) = data%prob%R( : o )
     G( : n ) = data%prob%G( : n )

     RETURN

!  error returns

 900 CONTINUE
     status = GALAHAD_error_h_not_permitted
     RETURN

!  End of subroutine SLLS_solve_given_a

     END SUBROUTINE SLLS_solve_given_a

!- G A L A H A D -  S L L S _ s o l v e _ r e v e r s e _ a _ p r o d SUBROUTINE

     SUBROUTINE SLLS_solve_reverse_a_prod( data, status, eval_status,          &
                                           B, X, Z, R, G, X_stat,              &
                                           V, P, NZ_in, nz_in_start,           &
                                           nz_in_end, NZ_out, nz_out_end )

!  solve the bound-constrained linear least-squares problem whose structure
!  was previously imported, and for which the action of A and its traspose
!  on a given vector are obtained by reverse communication. See SLLS_solve
!  for a description of the required arguments.
!
!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------
!
!  data is a scalar variable of type SLLS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the solve. status must be set to 1 on initial
!   entry, and on exit has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!     2 The product A * v of the matrix A with a given vector v is required
!       from the user. The vector v will be provided in V and the
!       required product must be returned in P. SLLS_solve must then
!       be re-entered with eval_status set to 0, and any remaining
!       arguments unchanged. Should the user be unable to form the product,
!       this should be flagged by setting eval_status to a nonzero value
!
!     3 The product A^T * v of the transpose of the matrix A with a given
!       vector v is required from the user. The vector v will be provided in
!       V and the required product must be returned in P.
!       SLLS_solve must then be re-entered with eval_status set to 0,
!       and any remaining arguments unchanged. Should the user be unable to
!       form the product, this should be flagged by setting eval_status
!       to a nonzero value
!
!     4 The product A * v of the matrix A with a given sparse vector v is
!       required from the user. Only components
!         NZ_in( nz_in_start : nz_in_end )
!       of the vector v stored in V are nonzero. The required product
!       should be returned in P. SLLS_solve must then be re-entered
!       with all other arguments unchanged. Typically v will be very sparse
!       (i.e., nz_in_end-NZ_in_start will be small).
!       eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     5 The product A * v of the matrix A with a given sparse vector v
!       is required from the user. Only components
!         NZ_in( nz_in_start : nz_in_end )
!       of the vector v stored in V are nonzero. The resulting
!       NONZEROS in the product A * v must be placed in their appropriate
!       comnpinents of P, while a list of indices of the nonzeos
!       placed in NZ_out( 1 : nz_out_end ). SLLS_solve should
!       then be re-entered with all other arguments unchanged. Typically
!       v will be very sparse (i.e., nz_in_end-NZ_in_start
!       will be small). Once again eval_status should be set to zero
!       unless the product cannot be form, in which case a nonzero value
!       should be returned.
!
!     6 Specified components of the product A^T * v of the transpose of the
!       matrix A with a given vector v stored in V are required from
!       the user. Only components indexed by
!         NZ_in( nz_in_start : nz_in_end )
!       of the product should be computed, and these should be recorded in
!         P( NZ_in( nz_in_start : nz_in_end ) )
!       and SLLS_solve then re-entered with all other arguments unchanged.
!       eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     7 The product P^-1 * v involving the preconditioner P with a specified
!       vector v is required from the user. Here P should be a symmtric,
!       postive-definite approximation of A^T A. The vector v will be provided
!       in V and the required product must be returned in P.
!       SLLS_solve must then be re-entered with eval_status set to 0,
!       and any remaining arguments unchanged. Should the user be unable
!       to form the product, this should be flagged by setting
!       eval_status to a nonzero value. This return can only happen
!       when control%preciditioner is not 0 or 1.
!
!   For other, negative, values see, slls_solve above.
!
!  eval_status is a scalar variable of type default intege that indicates the
!   success or otherwise of the reverse comunication operation. It must
!   be set to 0 if the opertation was successful, and to a nonzero value
!   if the operation failed for any reason.
!
!  B is a rank-one array of dimension o and type default
!   real, that holds the vector of observations, b.
!   The i-th component of B, i = 1, ... , o, contains (b)_i.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  Z is a rank-one array of dimension n and type default
!   real, that holds the vector of the dual variables, z.
!   The j-th component of Z, j = 1, ... , n, contains (z)_j.
!
!  R is a rank-one array of dimension o and type default
!   real, that holds the vector of residuals, r = A x - b on exit.
!   The i-th component of R, i = 1, ... , o, contains (r)_i.
!
!  G is a rank-one array of dimension n and type default
!   real, that holds the gradient, g = A^T c on exit.
!   The j-th component of G, j = 1, ... , n, contains (g)_j.
!
!  X_stat is a rank-one array of dimension n and type default integer,
!   that may be set by the user on entry to indicate which of the variables
!   are to be included in the initial working set. If this facility is
!   required, the component control%cold_start must be set to 0 on entry;
!   X_stat need not be set if control%cold_start is nonzero. On exit,
!   X_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   X_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  The remaining components V, ... , nz_out_end need not be set
!  on initial entry, but must be set as instructed by status as above.

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: eval_status
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R, G
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : ) :: X_stat
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: nz_out_end
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nz_in_start, nz_in_end
     INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: NZ_out
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : ) :: NZ_in
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: P
     REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( : ) :: V

!  local variables

     INTEGER ( KIND = ip_ ) :: n, o, error
     CHARACTER ( LEN = 80 ) :: array_name
     LOGICAL :: deallocate_error_fatal, space_critical

!  recover the dimensions

     n = data%prob%n ; o = data%prob%o

     SELECT CASE ( status )

!  initial entry

     CASE( 1 )

!  save the observations

       data%prob%B( : o ) = B( : o )

!  save the initial primal and dual variables

       data%prob%X( : n ) = X( : n )
       data%prob%Z( : n ) = Z( : n )

!  allocate space for reverse-communication data

       error = data%slls_control%error
       space_critical = data%slls_control%space_critical
       deallocate_error_fatal = data%slls_control%space_critical

       array_name = 'slls: data%reverse%NZ_out'
       CALL SPACE_resize_array( n, data%reverse%NZ_out,                        &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%reverse%NZ_in'
       CALL SPACE_resize_array( n, data%reverse%NZ_in,                         &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%reverse%P'
       CALL SPACE_resize_array( n, data%reverse%P,                             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%reverse%V'
       CALL SPACE_resize_array( n, data%reverse%V,                             &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

!  make sure that Ao%type is not allocated

       array_name = 'slls: data%prob%Ao%type'
       CALL SPACE_dealloc_array( data%prob%Ao%type,                            &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

!  save Jacobian-vector product information on re-entries

     CASE( 2, 4 )
       data%reverse%eval_status = eval_status
       data%reverse%P( : o ) = P( : o )
     CASE( 3, 7 )
       data%reverse%eval_status = eval_status
       data%reverse%P( : n ) = P( : n )
     CASE( 5 )
       data%reverse%eval_status = eval_status
       data%reverse%P( NZ_out( 1 : nz_out_end ) )                              &
         = P( NZ_out( 1 : nz_out_end ) )
       data%reverse%NZ_out( 1 : nz_out_end ) = NZ_out( 1 : nz_out_end )
       data%reverse%nz_out_end = nz_out_end
     CASE( 6 )
       data%reverse%eval_status = eval_status
       data%reverse%P( data%reverse%NZ_in( nz_in_start : nz_in_end ) )         &
         = P( data%reverse%NZ_in( nz_in_start : nz_in_end ) )
     CASE DEFAULT
       data%slls_inform%status = GALAHAD_error_input_status
       GO TO 900
     END SELECT

!  call the solver

     CALL SLLS_solve( data%prob, X_stat, data%slls_data, data%slls_control,    &
                      data%slls_inform, data%userdata, reverse = data%reverse )
     status = data%slls_inform%status

!  recover the optimal primal and dual variables

     X( : n ) = data%prob%X( : n )
     Z( : n ) = data%prob%Z( : n )

!  recover the residual value and gradient

     R( : o ) = data%prob%R( : o )
     G( : n ) = data%prob%G( : n )

!  record Jacobian-vector product information for reverse communication

     SELECT CASE ( status )
     CASE( 2, 7 )
       V( : n ) = data%reverse%V( : n )
     CASE( 3 )
       V( : o ) = data%reverse%V( : o )
     CASE( 4, 5 )
       nz_in_start = data%reverse%nz_in_start
       nz_in_end = data%reverse%nz_in_end
       NZ_in( nz_in_start : nz_in_end )                                        &
         = data%reverse%NZ_in( nz_in_start : nz_in_end )
       V( NZ_in( nz_in_start : nz_in_end ) )                                   &
         = data%reverse%V( NZ_in( nz_in_start : nz_in_end ) )
     CASE( 6 )
       nz_in_start = data%reverse%nz_in_start
       nz_in_end = data%reverse%nz_in_end
       NZ_in( nz_in_start : nz_in_end )                                        &
         = data%reverse%NZ_in( nz_in_start : nz_in_end )
       V( : o ) = data%reverse%V( : o )
     END SELECT

     RETURN

!  error returns

 900 CONTINUE
     status = data%slls_inform%status
     RETURN

!  End of subroutine SLLS_solve_reverse_a_prod

     END SUBROUTINE SLLS_solve_reverse_a_prod

!-  G A L A H A D -  S L L S _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE SLLS_information( data, inform, status )

!  return solver information during or after solution by SLLS
!  See SLLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%slls_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SLLS_information

     END SUBROUTINE SLLS_information

!  End of module SLLS

   END MODULE GALAHAD_SLLS_precision
