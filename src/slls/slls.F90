! THIS VERSION: GALAHAD 5.5 - 2026-02-26 AT 09:50 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ S L L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Fowkes/Gould/Montoison/Orban, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 4.1. June 3rd 2022
!   extended from the single- to multiple-simplex case, December 19th 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SLLS_precision

!     ---------------------------------------------------------------------
!    |                                                                     |
!    | Solve the multiply-simplex-constrained linear-least-squares problem |
!    |                                                                     |
!    |   minimize   1/2 || A_o x - b ||_W^2 + weight / 2 || x - x_s ||^2   |
!    |   subject to    e_Ci^T x_Ci = 1, x_Ci >= 0, i = 1,..., m,           |
!    |                                                                     |
!    | where the Ci are non-overlapping index subsets of {1,...,n}, ||v||  |
!    | and ||r||_W^2 are the Euclidean & weighted Euclidean norms defined  |
!    | by ||v||^2 = v^T v and ||r||_W^2 = r^T W r, using a preconditioned  |
!    ! projected conjugate-gradient approach                               |
!    |                                                                     |
!     ---------------------------------------------------------------------

     USE GALAHAD_KINDS_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_STRING
     USE GALAHAD_SPACE_precision
     USE GALAHAD_SORT_precision, ONLY: SORT_heapsort_build,                    &
                                       SORT_heapsort_smallest, SORT_quicksort
     USE GALAHAD_LAPACK_inter_precision, ONLY : POTRF, POTRS
     USE GALAHAD_SBLS_precision
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_QPT_precision, ONLY: QPT_problem_type, QPT_keyword_A
     USE GALAHAD_QPD_precision, ONLY: QPD_SIF
     USE GALAHAD_USERDATA_precision, ONLY: USERDATA_type
     USE GALAHAD_REVERSE_precision, ONLY: REVERSE_type, REVERSE_terminate
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_CONVERT_precision, ONLY: CONVERT_control_type,                &
                                          CONVERT_inform_type,                 &
                                          CONVERT_to_sparse_column_format
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SLLS_initialize, SLLS_read_specfile, SLLS_solve,                &
               SLLS_terminate,  SLLS_data_type, REVERSE_type,                  &
               SLLS_full_initialize, SLLS_full_terminate,                      &
               SLLS_search_data_type,                                          &
               SLLS_subproblem_data_type, SLLS_exact_arc_search,               &
               SLLS_inexact_arc_search, SLLS_import, SLLS_import_without_a,    &
               SLLS_solve_given_a, SLLS_solve_reverse_a_prod,                  &
               SLLS_reset_control, SLLS_information, USERDATA_type,            &
               QPT_problem_type, SMT_type, SMT_put, SMT_get,                   &
               SLLS_project_onto_simplex, SLLS_project_onto_simplices,         &
               SLLS_simplex_projection_path, SLLS_cgls


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
!   assigned according to X_status, see below), and to any other value if the
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

       REAL ( KIND = rp_ ) :: total = 0.0

!  total clock time

       REAL ( KIND = rp_ ) :: clock_total = 0.0

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

!  current value of the  least-squares function, 1/2 || A_o x - b ||_W^2

       REAL ( KIND = rp_ ) :: ls_obj = infinity

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

!  the output scalar from LAPACK routines

       INTEGER ( KIND = ip_ ) :: lapack_error = 0

     END TYPE SLLS_inform_type

!  - - - - - - - - - - - - - - -
!   arc_search data derived type
!  - - - - - - - - - - - - - - -

     TYPE :: SLLS_search_data_type
       INTEGER ( KIND = ip_ ) :: ic, m, step
       REAL ( KIND = rp_ ) :: ete, f_0, phi_0, phi_1_stop, gamma, rtr, xtx
       REAL ( KIND = rp_ ) :: s_fixed, x_s_fixed, t, t_break, t_total
       REAL ( KIND = rp_ ) :: rho_0, rho_1, rho_2, xi
       LOGICAL :: shifts, present_a, reverse_a, reverse_as, backwards
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: IP, I_len
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: P, rhom_0
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: rhom_1, rhom_2, xim
     END TYPE SLLS_search_data_type

!  - - - - - - - - - - - - - - -
!   arc_search data derived type
!  - - - - - - - - - - - - - - -

     TYPE :: SLLS_subproblem_data_type
       INTEGER ( KIND = ip_ ) :: branch, n_preconditioner, step
       REAL ( KIND = rp_ ) :: stop_cg, gamma, gamma_old, etpe
       LOGICAL :: printp, printw, printd, printdd, debug
       LOGICAL :: present_a, present_ascol, reverse_ascol, present_afprod
       LOGICAL :: reverse_afprod, reverse_prec, present_prec, present_dprec
       LOGICAL :: recompute, regularization, shifts, preconditioned
       CHARACTER ( LEN = 1 ) :: direction
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, P, Q, PG, PE
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ETPEM, Y
     END TYPE SLLS_subproblem_data_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE :: SLLS_data_type
       INTEGER ( KIND = ip_ ) :: out, error, print_level, eval_status
       INTEGER ( KIND = ip_ ) :: start_print, stop_print, print_gap
       INTEGER ( KIND = ip_ ) :: arc_search_status, cgls_status, change_status
       INTEGER ( KIND = ip_ ) :: n_free, n_c, branch, cg_iter, preconditioner
       INTEGER ( KIND = ip_ ) :: maxit, cg_maxit, segment, steps, max_steps
       REAL ( KIND = rp_ ) :: time_start, clock_start
       REAL ( KIND = rp_ ) :: norm_step, step, stop_cg, old_gnrmsq, pnrmsq
       REAL ( KIND = rp_ ) :: alpha_0, alpha_max, alpha_new, f_new, phi_new
       REAL ( KIND = rp_ ) :: weight, stabilisation_weight
       REAL ( KIND = rp_ ) :: regularization_weight
       LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
       LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe
       LOGICAL :: reverse, reverse_prod, explicit_a, use_aprod, header
       LOGICAL :: direct_subproblem_solve, steepest_descent
       LOGICAL :: multiple_simplices, w_eq_identity, shifts
       CHARACTER ( LEN = 6 ) :: string_cg_iter
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: FREE, S_ind, S_ptr
       LOGICAL ( KIND = lp_ ), ALLOCATABLE, DIMENSION( : ) :: FIXED, FIXED_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_new, G, R, SBLS_sol
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: D, AD, S, AE, DIAG
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_c, X_c_proj
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: VT, EVT, Y
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
        LOGICAL :: explicit_a = .FALSE.
        TYPE ( SLLS_data_type ) :: SLLS_data
        TYPE ( SLLS_control_type ) :: SLLS_control
        TYPE ( SLLS_inform_type ) :: SLLS_inform
        TYPE ( QPT_problem_type ) :: prob
        TYPE ( USERDATA_type ) :: userdata
        TYPE ( REVERSE_type ) :: reverse
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
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_d = arcsearch_max_steps + 1
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

     SUBROUTINE SLLS_solve( prob, data, control, inform, userdata, reverse,    &
                            eval_APROD, eval_ASCOL, eval_AFPROD, eval_PREC )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the linear least-squares problem
!
!     minimize     q(x) = 1/2 || Ao x - b ||_W^2 + weight / 2 || x - x_s ||^2
!
!     subject to    e_Ci^T x_Ci = 1, x_Ci >= 0, i = 1,..., m,
!
!  where x is a vector of n components (x_1, ...., x_n), b is an m-vector,
!  Ao is an o by n matrix, weight is scalar weight, W and x_s are vectors 
!  of diagonal-scaling weights and shifts, and the cohorts Ci are 
!  non-overlapping index subsets of {1, ..., n}, using a preconditioned 
!  projected CG method.
!
!  The subroutine is particularly appropriate when Ao is sparse, or if it
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
!   %m is an INTEGER variable, that should be set by the user to the
!    number of simplex constraints, m, invoved. RESTRICTION: %m >= 0
!
!   %regularization_weight is a REAL variable, that may be set by the user
!    to the value of the non-negative regularization weight. It takes the 
!    default value of zero
!
!   %Ao is a structure of type SMT_type used to hold Ao if available).
!    Five storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 10 ) = TRANSFER( 'COORDINATE', Ao%type )
!       Ao%val( : )  the values of the components of A
!       Ao%row( : )  the row indices of the components of A
!       Ao%col( : )  the column indices of the components of A
!       Ao%ne        the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', Ao%type )
!       Ao%val( : )  the values of the components of A, stored row by row
!       Ao%col( : )  the column indices of the components of A
!       Ao%ptr( : )  pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) sparse, by columns
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 17 ) = TRANSFER( 'SPARSE_BY_COLUMNS', Ao%type )
!       Ao%val( : )  the values of the components of A, stored column by column
!       Ao%row( : )  the row indices of the components of A
!       Ao%ptr( : )  pointers to the start of each column, and past the end of
!                    the last column
!
!    iv) dense, by rows
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 13 ) = TRANSFER( 'DENSE_BY_ROWS', Ao%type )
!       Ao%val( : )  the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    v) dense, by columns
!
!       In this case, the following must be set:
!
!       Ao%type( 1 : 16 ) = TRANSFER( 'DENSE_BY_COLUMNS', Ao%type )
!       Ao%val( : )  the values of the components of A, stored column by column,
!                    with each the entries in each column in order of
!                    increasing row indicies.
!
!    If Ao is not available explicitly, matrix-vector products must be
!      provided by the user using either reverse communication
!      (see reverse below) or a provided subroutine (see eval_APROD
!       and eval_ASCOL below).
!
!   %B is a REAL array of length %m, which must be set by the user to the
!    value of b, the linear term of the residuals, in the least-squares
!    objective function. The i-th component of B, i = 1, ...., o,
!    should contain the value of b_i.
!
!   %COHORT is a INTEGER array of length %n, whose j-th component may be set to
!    the number, between 1 and %m, of the cohort to which variable x_j belongs,
!    or to 0 if the variable belong to no cohort. If COHORT is unallocated,
!    all variables will be assumed to belong to a single cohort.
!
!   %W is a REAL array of length %o, that may be set by the user
!    to the values of the components of the weights w.
!    If %W is unallocated, the weights will all be taken to be 1.0.
!
!   %X_s is a REAL array of length %n, that may be set by the user
!    to the values of the components of the shifts x_s. 
!    If %X_s is unallocated, the shifts will all be taken to be 0.0.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %Y is a REAL array of length m (1 if m is not present), which, 
!    on successful exit, will contain the Lagrange multiplers y.
!
!   %Z is a REAL array of length %n, which need not be set on input. On
!    successful exit, it will contain estimates of the values of the dual
!    variables, i.e., Lagrange multipliers corresponding to the simple bound
!    constraints x >= 0.
!
!   %R is a REAL array of length %o, which need not be set on input. On
!    successful exit, it will contain the residual vector r(x) = A x - b. The
!    i-th component of R, i = 1, ...., o, will contain the value of r_i(x).
!
!   %X_status is an INTEGER array of length %n, that will be set on exit to
!    indicate the likely ultimate status of the simple bound constraints.
!    Possible values are
!    X_status( i ) < 0, the i-th bound constraint is likely in the active set,
!                       on its lower bound,
!                  > 0, the i-th bound constraint is likely in the active set
!                       on its upper bound, and
!                  = 0, the i-th bound constraint is likely not in the active
!                       set
!    It need not be set on entry.
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
!       from the user. The vector v will be provided in reverse%v and the
!       required product must be returned in reverse%p. SLLS_solve must 
!       then be re-entered with reverse%eval_status set to 0, and any remaining
!       arguments unchanged. Should the user be unable to form the product,
!       this should be flagged by setting reverse%eval_status to a nonzero value
!
!     3 The product A^T * v of the transpose of the matrix A with a given
!       vector v is required from the user. The vector v will be provided in
!       reverse%v and the required product must be returned in 
!       reverse%p. SLLS_solve must then be re-entered with 
!       reverse%eval_status set to 0, and any remaining arguments unchanged. 
!       Should the user be unable to form the product, this should be flagged 
!       by setting reverse%eval_status to a nonzero value
!
!     4 The j-th column of Ao is required from the user, where 
!       reverse%index holds the value of j. The resulting NONZEROS 
!       and their correspinding row indices of the j-th column of Ao must
!       be placed in reverse%p( 1 : reverse%lp ) and
!       reverse%ip( 1 : reverse%lp ) with reverse%lp
!       set accordingly. SLLS_solve should then be re-entered with all other 
!       arguments unchanged. Once again reverse%eval_status should be set to 
!       zero unless the column cannot be formed, in which case a nonzero 
!       value should be returned.
!
!     5 The product A * v of the matrix A with a given sparse vector v is
!       required from the user. Only components
!         reverse%iv( reverse%lvl : reverse%lvu )
!       of the vector v stored in reverse%v are nonzero. The required
!       product should be returned in reverse%p. SLLS_solve must then be 
!       re-entered with all other arguments unchanged. Typically v will be 
!       very sparse (i.e., reverse%lvu-reverse%lvl will be small).
!       reverse%eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     6 Specified components of the product A^T * v of the transpose of the
!       matrix A with a given vector v stored in reverse%v are required
!       from the user. Only components indexed by
!         reverse%iv( reverse%lvl : reverse%lvu )
!       of the product should be computed, and these should be recorded in
!         reverse%p( reverse%iv( reverse%lvl : reverse%lvu ) )
!       and SLLS_solve then re-entered with all other arguments unchanged.
!       reverse%eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     7 The product P^-1 * v involving the preconditioner P with a specified
!       vector v is required from the user. Here P should be a symmtric,
!       postive-definite approximation of A^T A. The vector v will be provided
!       in reverse%v and the required product must be returned in 
!       reverse%p. SLLS_solve must then be re-entered with 
!       reverse%eval_status set to 0, and any remaining arguments unchanged. 
!       Should the user be unable to form the product, this should be flagged 
!       by setting reverse%eval_status to a nonzero value. This return can 
!       only happen when control%preciditioner is not 0 or 1.
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
!                          'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS', 'COORDINATE' }
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
!     time%total = the total CPU time spent in the package.
!     time%clock_total = the total clock time spent in the package.
!
!  userdata is a scalar variable of type USERDATA_type which may be used
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
!  reverse is an OPTIONAL structure of type REVERSE_type which is used
!   to pass intermediate data to and from SLLS_solve. This will only be 
!   necessary if reverse-communication is to be used to form matrix-vector 
!   products of the form Ao * v, find columns of Ao or compute preconditioning 
!   steps of the form P^{-1} * v. If  reverse is present (and eval_APROD,
!   eval_ASCOL, eval_AFPROD or eval_PREC is absent), reverse communication will
!   be used and the user must monitor the value of inform%status(see above) to 
!   await instructions about required  matrix-vector products.
!
!  eval_APROD is an OPTIONAL subroutine which if present must have the
!   arguments given below (see the interface blocks). The sum p + Ao * v
!   (if transpose is .FALSE.) or p + Ao^T v (if transpose is .TRUE.)
!   involving the given matrix A and vectors p and v stored in
!   P and V must be returned in P. The status variable should be set
!   to 0 unless the product is impossible in which case status should
!   be set to a nonzero value. If eval_APROD is not present, SLLS_solve
!   will either return to the user each time an evaluation is required
!   (see reverse above) or form the product directly from user-provided prob%Ao.
!
!  eval_ASCOL is an OPTIONAL subroutine which if present must have the
!   arguments given below (see the interface blocks). The index-th column
!   of Ao should be returned in COL as a spare vector. Specifically,
!   the NONZEROS in the index-th column of Ao must be placed in their
!   appropriate comnponents of COL, while a list of indices of the
!   nonzeros placed in ICOL( 1 : lcol ). The status variable should 
!   be set to 0 unless the column is unavailable in which case status should 
!   be set to a nonzero value. If eval_ASCOL is not present, SLLS_solve will 
!   either return to the user each time an evaluation is required 
!   (see reverse above) or form the product directly from user-provided prob%Ao.
!
!  eval_AFPROD is an OPTIONAL subroutine which if present must have the
!   arguments given below (see the interface blocks). The product Ao * v
!   (if transpose is .FALSE.) or Ao^T v (if transpose is .TRUE.) involving
!   the given matrix Ao and the vector v stored in v must be returned
!   in p. If transpose is .FALSE., only the components of v with
!   indices FREE(:n_free) should be used, the remaining components should be
!   treated as zero. If transpose is .TRUE., all of v should be used, but
!   only the components p(IFREE(:nfree) need be computed, the remainder
!   will be ignored. The status variable should be set to 0 unless the product
!   is impossible in which case status should be set to a nonzero value.
!   If eval_AFPROD is not present, SLLS_solve will either return to the user
!   each time an evaluation is required (see reverse above) or form the
!   product directly from user-provided prob%Ao.
!
!  eval_PREC is an OPTIONAL subroutine which if present must have the arguments
!   given below (see the interface blocks). The product P^{-1} * v of the given
!   preconditioner P and vector v stored in v must be returned in p.
!   The intention is that P is an approximation to A^T A. The status variable
!   should be set to 0 unless the product is impossible in which case status
!   should be set to a nonzero value. If eval_PREC is not present, SLLS_solve
!   will return to the user each time a preconditioning operation is required
!   (see reverse above) when control%preconditioner is not 0 or 1.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  dummy arguments

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     TYPE ( SLLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse
     OPTIONAL :: eval_APROD, eval_ASCOL, eval_AFPROD, eval_PREC

!  interface blocks

     INTERFACE
       SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       LOGICAL, INTENT( IN ) :: transpose
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
       END SUBROUTINE eval_APROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_ASCOL( status, userdata, index, COL, ICOL, lcol )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       INTEGER ( KIND = ip_ ), INTENT( IN ) :: index
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: COL
       INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: ICOL
       INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: lcol
       END SUBROUTINE eval_ASCOL
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_AFPROD( status, userdata, transpose, V, P, FREE, n_free )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
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
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       END SUBROUTINE eval_PREC
     END INTERFACE

!  local variables

     INTEGER ( KIND = ip_ ) :: i, j, k, l, ne, nap, minloc_g( 1 )
     REAL ( KIND = rp_ ) :: time_now, clock_now
     REAL ( KIND = rp_ ) :: val, x_j, g_j, lambda
     CHARACTER ( LEN = 6 ) :: string_iter
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  enter or re-enter the package and jump to appropriate re-entry point

     IF ( inform%status == 1 ) data%branch = 100
     IF ( inform%status == 10 ) data%branch = 150

     SELECT CASE ( data%branch )
     CASE ( 100 ) ; GO TO 100
     CASE ( 150 ) ; GO TO 150  ! re-entry with new data
     CASE ( 190 ) ; GO TO 190  ! re-entry with p = p + Av
     CASE ( 210 ) ; GO TO 210  ! re-entry with p = p + Av
     CASE ( 220 ) ; GO TO 220  ! re-entry with p = p + A^T v
     CASE ( 300 ) ; GO TO 300  ! re-entry with p = A v (dense or sparse)
     CASE ( 400 ) ; GO TO 400  ! re-entry with p = p + A v
     CASE ( 420 ) ; GO TO 420  ! re-entry with p = A v (dense or sparse)
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

     CALL CPU_TIME( data%time_start )  ; CALL CLOCK_time( data%clock_start )

!  set initial timing breakdowns

     inform%time%total = 0.0 ; CALL CLOCK_time( data%clock_start )

!  check that optional arguments are consistent -

!  operations regarding simplices

     data%multiple_simplices = ALLOCATED( prob%COHORT ) .AND. prob%m >= 1
     IF ( data%multiple_simplices ) THEN
       array_name = 'slls: data%S_ptr'
       CALL SPACE_resize_array( prob%m + 1, data%S_ptr, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       data%S_ptr( 1 : prob%m ) = 0
       DO j = 1, prob%n
         i = prob%COHORT( j )
         IF ( i > prob%n .OR. i < 0 ) THEN
           inform%status = GALAHAD_error_restrictions ; GO TO 910
         ELSE IF ( i > 0 ) THEN
           data%S_ptr( i ) = data%S_ptr( i ) + 1
         END IF
       END DO
       IF ( MINVAL( data%S_ptr( 1 : prob%m ) ) == 0 ) THEN
         inform%status = GALAHAD_error_restrictions ; GO TO 910
       END IF
     END IF

!  operations with A

     data%use_aprod = PRESENT( eval_APROD ) .AND. PRESENT( eval_ASCOL ) .AND.  &
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
     IF ( data%out > 0 .AND. control%print_level >= 1 ) THEN
       IF ( data%multiple_simplices ) THEN
         WRITE( control%out, "( A, ' constraints are the intersection of ',    &
        &  I0, ' non-overlapping simplices' )" ) prefix, prob%m
       ELSE
         WRITE( control%out, "( A, ' constraint is a single simplex' )" ) prefix
       END IF
     END IF

!  see if W = I

       IF ( ALLOCATED( prob%W ) ) THEN
         data%w_eq_identity = SIZE( prob%W ) < prob%o
       ELSE
         data%w_eq_identity = .TRUE.
       END IF
       IF ( .NOT. data%w_eq_identity ) THEN
         IF ( COUNT( prob%W( : prob%o ) <= zero ) > 0 ) THEN
           IF ( control%error > 0 ) WRITE( control%error,                      &
             "( A, ' error: input entries of W must be strictly positive' )" ) &
             prefix
           inform%status = GALAHAD_error_restrictions
           GO TO 910
         ELSE IF ( COUNT( prob%W( : prob%o ) == one ) == prob%o ) THEN
           data%w_eq_identity = .TRUE.
         END IF
       END IF

!  see if X_s = 0

       IF ( ALLOCATED( prob%X_s ) ) THEN
          data%shifts = SIZE( prob%X_s ) >= prob%n
       ELSE
         data%shifts = .FALSE.
       END IF

!  if required, write out problem

     IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
       WRITE( control%out, "( ' o, n = ', I0, 1X, I0 )" ) prob%o, prob%n
       WRITE( control%out, "( ' B = ', /, ( 5ES12.4 ) )" ) prob%B( : prob%o )
       IF ( data%w_eq_identity ) THEN
         WRITE( control%out, "( ' W = identity' )" )
       ELSE
         WRITE( control%out, "( ' W = ', /, ( 5ES12.4 ) )" ) prob%W( : prob%o )
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

!  if required, use the initial variable status implied by X_status

     IF ( control%cold_start == 0 ) THEN
       DO i = 1, prob%n
         IF ( prob%X_status( i ) < 0 ) prob%X( i ) = zero
       END DO
     END IF

!  allocate workspace arrays

     array_name = 'slls: data%X_new'
     CALL SPACE_resize_array( prob%n, data%X_new, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  if there are multiple simplices, make a continguous list of variables 
!  (the cohort) for each simplex. That is assign
!
!    cohort     0     1     2    ...    m   
!    S_ind:  | ... | ... | ... | ... | ... |
!                   ^     ^     ^     ^     ^
!    S_ptr:         |     |     |     |     |

     IF ( data%multiple_simplices ) THEN

!write(6,*) ' cohort ', cohort
!write(6,*) ' x ', prob%X
!  allocate further workspace arrays

       array_name = 'slls: data%S_ind'
       CALL SPACE_resize_array( prob%n, data%S_ind, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  next compute how many variables each simplex uses (temporarily in S_ptr)

       data%S_ptr( 1 : prob%m ) = 0
       l = 0
       DO i = 1, prob%n
         j = prob%COHORT( i )
         IF ( j > 0 ) THEN
           data%S_ptr( j ) = data%S_ptr( j ) + 1
         ELSE
           l = l + 1
         END IF
       END DO
!      WRITE( 6, "( ' size cohorts', 5( ' ', I0 ) )" ) l, data%S_ptr

!  record the maximum number

       data%n_c = MAXVAL( data%S_ptr( 1 : prob%m ) )

!  now assign the starting address for each cohort in S_ptr

       l = l + 1
       DO i = 1, prob%m
         j = data%S_ptr( i )
         data%S_ptr( i ) = l
         l = l + j
       END DO
 !     WRITE( 6, "( ' starts cohorts', 5( ' ', I0 ) )" ) data%S_ptr

!  next, put the variables for each cohort in adjacent positions in S_val
!  and adjust S_ptr accordingly so that the last variable in cohort j is
!  in position S_ptr(j), with any remaining variables at the start of S_val

       l = 0
       DO i = 1, prob%n
         j = prob%COHORT( i )
         IF ( j > 0 ) THEN
           data%S_ind( data%S_ptr( j ) ) = i
           data%S_ptr( j ) = data%S_ptr( j ) + 1
         ELSE
           l = l + 1
           data%S_ind( l ) = i
         END IF 
       END DO

!  finally, recover the starting address S_ptr for each cohort (and
!  set S_ptr( m + 1 ) to be one beyond n

       data%S_ptr( prob%m + 1 ) = prob%n + 1
       DO i = prob%m, 2, - 1
         data%S_ptr( i ) = data%S_ptr( i - 1 )
       END DO
       data%S_ptr( 1 ) = l + 1
!      WRITE( 6, "( ' start cohorts', 5( ' ', I0 ) )" ) data%S_ptr
!      DO i = 1, prob%m
!       WRITE( 6, "( ' cohort ', I0, ':', 5( ' ', I0 ) )" ) &
!         i, ( data%S_ind( j ), j = data%S_ptr( i ), data%S_ptr( i + 1 ) - 1 )
!      END DO
!      IF ( data%S_ptr( 1 ) > 1 ) WRITE( 6, "( ' cohort 0:', 5( ' ', I0 ) )" ) &
!         ( data%S_ind( j ), j = 1, data%S_ptr( 1 ) - 1 )

!  allocate space for the sub-projections into each simplex

       array_name = 'slls: data%X_c'
       CALL SPACE_resize_array( data%n_c, data%X_c,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%X_c_proj'
       CALL SPACE_resize_array( data%n_c, data%X_c_proj,                       &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  check that input estimate of the solution is in the intersection of 
!  simplices, and if not project it so that it is

       CALL SLLS_project_onto_simplices( prob%n, prob%m, data%n_c, data%S_ptr, &
                                         data%S_ind, prob%X, data%X_new,       &
                                         data%X_c, data%X_c_proj, i )
     ELSE
       CALL SLLS_project_onto_simplex( prob%n, prob%X, data%X_new, i )
     END IF

!  check that the projection succeeded

    IF ( i < 0 ) THEN
       inform%status = GALAHAD_error_sort
       GO TO 910
     ELSE IF ( i > 0 ) THEN
       prob%X( : prob%n ) = data%X_new( : prob%n )
       IF ( data%printi ) WRITE( control%out,                                  &
       "( ' ', /, A, '   **  Warning: input point projected onto simplex' )" ) &
         prefix
     END IF

!  allocate further workspace arrays

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
       array_name = 'slls: reverse%iv'
       CALL SPACE_resize_array( prob%n, reverse%iv, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: reverse%ip'
       CALL SPACE_resize_array( prob%o, reverse%ip, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: reverse%v'
       CALL SPACE_resize_array( MAX( prob%o, prob%n ), reverse%v,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: reverse%p'
       CALL SPACE_resize_array( MAX( prob%o, prob%n ), reverse%p,        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     IF ( data%multiple_simplices ) THEN
       array_name = 'slls: prob%Y'
       CALL SPACE_resize_array( prob%m, prob%Y, inform%status,                 &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%VT'
       CALL SPACE_resize_array( prob%n, prob%m, data%VT, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%EVT'
       CALL SPACE_resize_array( prob%m, prob%m, data%EVT, inform%status,       &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%Y'
       CALL SPACE_resize_array( prob%m, 1_ip_, data%Y, inform%status,          &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!      data%search_data%m = m
       array_name = 'slls: data%search_data%I_len'
       CALL SPACE_resize_array( prob%m, data%search_data%I_len, inform%status, &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%search_data%rhom_0'
       CALL SPACE_resize_array( 0, prob%m, data%search_data%rhom_0,            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%search_data%rhom_1'
       CALL SPACE_resize_array( 0, prob%m, data%search_data%rhom_1,            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%search_data%rhom_2'
       CALL SPACE_resize_array( 0, prob%m, data%search_data%rhom_2,            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%search_data%xim'
       CALL SPACE_resize_array( prob%m, data%search_data%xim,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
       array_name = 'slls: prob%Y'
     ELSE
       CALL SPACE_resize_array( 1, prob%Y, inform%status,                      &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910
     END IF

     IF ( control%exact_arc_search .AND. PRESENT( eval_ASCOL ) .AND.           &
          .NOT. ( data%explicit_a .OR. data%reverse ) ) THEN
       array_name = 'slls: data%search_data%IP'
       CALL SPACE_resize_array( prob%o, data%search_data%IP, inform%status,    &
              inform%alloc_status, array_name = array_name,                    &
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

       SELECT CASE( SMT_get( prob%Ao%type ) )
       CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
         ne = prob%o * prob%n
       CASE ( 'SPARSE_BY_ROWS' )
         ne = prob%Ao%ptr( prob%o + 1 ) - 1
       CASE ( 'SPARSE_BY_COLUMNS' )
         ne = prob%Ao%ptr( prob%n + 1 ) - 1
       CASE ( 'COORDINATE' )
         ne = prob%Ao%ne
       END SELECT

       array_name = 'slls: data%AT_sbls%val'
       CALL SPACE_resize_array( ne, data%AT_sbls%val, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

       array_name = 'slls: data%AT_sbls%col'
       CALL SPACE_resize_array( ne, data%AT_sbls%col, inform%status,           &
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
     END IF

!  solution vector (used as workspace elsewhere)

     array_name = 'slls: data%SBLS_sol'
     CALL SPACE_resize_array( prob%n + prob%o, data%SBLS_sol, inform%status,   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  ----------------------
!  re-entry with new data
!  ----------------------

 150 CONTINUE

!  set the regularization weight

     IF ( prob%regularization_weight > zero ) THEN
       data%weight = prob%regularization_weight
     ELSE
       data%weight = zero
     END IF

!  initialize y and z

     prob%Y = zero ; prob%Z = zero

!  build a copy of A stored by columns

     IF ( data%explicit_a ) THEN
       CALL CONVERT_to_sparse_column_format( prob%Ao, data%Ao,                 &
                                             control%CONVERT_control,          &
                                             inform%CONVERT_inform )

!  weight by W if required

       IF ( .NOT. data%w_eq_identity ) THEN
         DO j = 1, prob%n
           DO k = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
             data%Ao%val( k )                                                  &
               = data%Ao%val( k ) * SQRT( prob%W( data%Ao%row( k ) ) )
           END DO
         END DO
       END IF
     END IF

!  set A e if needed later, where (here) e is a vector of ones

!    IF ( control%exact_arc_search .AND. .NOT. data%multiple_simplices ) THEN
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
!        reverse%p( : prob%o ) = zero
         reverse%v( : prob%n ) = one
         reverse%transpose = .FALSE.
         data%branch = 190 ; inform%status = 2 ; RETURN
       END IF
     END IF

!  re-entry point after the A e product

 190 CONTINUE
!    IF ( control%exact_arc_search .AND. .NOT. data%multiple_simplices .AND.   &
     IF ( control%exact_arc_search .AND. data%reverse_prod )                   &
       data%AE( : prob%o ) = reverse%p( : prob%o )

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
       IF ( data%weight > zero ) data%DIAG( : prob%n ) =                      &
          data%DIAG( : prob%n ) + data%weight
!write(6,"( ' diag ', 4ES12.4 )" )  data%DIAG( : prob%n )
       IF ( data%set_printm ) WRITE( data%out,                                 &
         "( /, A, ' diagonal preconditioner, min, max =', 2ES11.4 )" ) prefix, &
           MINVAL( data%DIAG( : prob%n ) ), MAXVAL( data%DIAG( : prob%n ) )
     END IF

!  continue setting the block matrix if it is to be used to solve the subproblem

     IF ( data%direct_subproblem_solve ) THEN

!  the 2,2 block

!  regularized case

       data%stabilisation_weight                                               &
         = MAX( data%weight, control%stabilisation_weight, zero )
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

!  ------------------------
!  start the main iteration
!  ------------------------

     data%change_status = prob%n

     IF ( data%set_printi ) WRITE( data%out,                                   &
       "( /, A, 9X, 'S=steepest descent, F=factorization used' )" ) prefix

 200 CONTINUE ! mock iteration loop
       CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
       inform%time%total = time_now - data%time_start
       inform%time%clock_total = clock_now - data%clock_start

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
!        reverse%p( : prob%o ) = - prob%B
!        reverse%p( : prob%o ) = zero
         reverse%v( : prob%n ) = prob%X
!write(6,"(' v', /, ( 5ES12.4 ) )" ) reverse%v( : prob%n )
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
         prob%R( : prob%o ) = reverse%p( : prob%o ) - prob%B( : prob%o )
!        prob%C( : prob%o ) = reverse%p( : prob%o )
!        reverse%p( : prob%n ) = zero
         reverse%v( : prob%o ) = prob%R( : prob%o )
         reverse%transpose = .TRUE.
         data%branch = 220 ; inform%status = 3 ; RETURN
       END IF

!  re-entry point after the Jacobian-transpose vector product

 220   CONTINUE
       IF ( data%reverse_prod ) prob%G( : prob%n ) = reverse%p( : prob%n )
!write(6,"( ' G =', / ( 5ES12.4 ) )" ) prob%G

!  compute the objective function

       inform%ls_obj                                                           &
         = half * DOT_PRODUCT( prob%R( : prob%o ), prob%R( : prob%o ) )

!  adjust the value and gradient to account for any regularization term

       IF ( data%weight > zero ) THEN
         IF ( data%shifts ) THEN
           data%S( : prob%n ) = prob%X( : prob%n ) - prob%X_s( : prob%n )
           inform%obj = inform%ls_obj + half * data%weight *                   &
             DOT_PRODUCT( data%S( : prob%n ), data%S( : prob%n ) )
           prob%G( : prob%n )                                                  &
             = prob%G( : prob%n ) + data%weight * data%S( : prob%n )
         ELSE
           inform%obj = inform%ls_obj + half * data%weight *                   &
             DOT_PRODUCT( prob%X( : prob%n ), prob%X( : prob%n ) )
           prob%G( : prob%n )                                                  &
             = prob%G( : prob%n ) + data%weight * prob%X( : prob%n )
         END IF
       ELSE
         inform%obj = inform%ls_obj
       END IF

!  record the dual variables

       prob%Z( : prob%n ) = prob%G( : prob%n )

!  compute the norm of the projected gradient

       val = MIN( one, one / TWO_NORM( prob%G( : prob%n ) ) )
       data%S = prob%X - val * prob%G( : prob%n )
       IF ( data%multiple_simplices ) THEN
!write(6,"( ' ! x            ', 5ES12.4 )" ) prob%X
!write(6,"( ' ! x-alpha g    ', 5ES12.4 )" ) data%S
         CALL SLLS_project_onto_simplices( prob%n, prob%m, data%n_c,           &
                                           data%S_ptr,                         &
                                           data%S_ind, data%S, data%X_new,     &
                                           data%X_c, data%X_c_proj, i )
!write(6,"( ' ! p(x-alpha g) ', 5ES12.4 )" ) data%X_new
       ELSE
         CALL SLLS_project_onto_simplex( prob%n, data%S, data%X_new, i )
       END IF
       inform%norm_pg = MAXVAL( ABS( data%X_new - prob%X ) )
!write(6,"( ' ! ||pg|| ', ES12.4 )" ) inform%norm_pg

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
           "( /, A, ' steepest descent search direction', / )" ) prefix

!  assign the search direction

         IF ( data%preconditioner /= 0 ) THEN
           data%D( : prob%n ) =                                                &
             - prob%G( : prob%n ) / SQRT( data%DIAG( : prob%n ) )
         ELSE
           data%D( : prob%n ) = - prob%G( : prob%n )
         END IF
         data%string_cg_iter = '     S'

!  if there are multiple cohorts, compute the Cauchy direction 
!  d_Cj^C = - argmin g_Cj^T s_Cj : e_cj^T s_cj = 0, s_Cj >= - x_Cj

         IF ( data%multiple_simplices ) THEN
           data%D( : prob%n ) = - prob%X( : prob%n )

!  consider each cohort in turn

           DO i = 1, prob%m

!  find the smallest component of g, g_j

             g_j = infinity
             DO k = data%S_ptr( i ), data%S_ptr( i + 1 ) - 1
               l = data%S_ind( k )
               IF ( prob%G( l ) < g_j ) THEN
                 g_j = prob%G( l )
                 j = l
               END IF
             END DO

!  now assign s^C_i = - x_i (i /= j) and s^C_j = 1 - x_j in d

             data%D( j ) = one - prob%X( j )
           END DO

!  if any variable x_j is unconstrained, set s^C_j = 1 if g_j <= 0 and  
!  s^C_j = -1 otherwise

           IF ( data%S_ptr( 1 ) > 1 ) THEN
             DO k = 1, data%S_ptr( 1 ) - 1
               j = data%S_ind( k )
               IF ( prob%G( j ) <= zero ) THEN
                 data%D( j ) = one
               ELSE
                 data%D( j ) = - one
               END IF
             END DO
           END IF

!  debug output

           IF ( data%printd ) THEN
             DO i = 1, prob%m
               WRITE( data%out, "( ' cohort ', I0, ' e''s = ', ES12.4 )" )     &
                 i, SUM( data%D( data%S_ind( data%S_ptr( i ) :                 &
                                             data%S_ptr( i + 1 ) - 1 ) ) )
             END DO
             DO j = 1, prob%n
               WRITE( data%out, "( 'x,d', I3, 2ES12.4 )" )                     &
                j, prob%X( j ), data%D( j ) 
             END DO
           END IF

!  else compute the Cauchy direction d^C = - argmin g^T s : e^T s = 0, s >= - x

         ELSE
           data%D( : prob%n ) = - prob%X( : prob%n )

!  find the smallest component of g, g_j

           minloc_g = MINLOC( prob%G( : prob%n ) )
           j = minloc_g( 1 )
!write(6,"('c', /, ( 5ES12.4) )" ) prob%R( : prob%o )
!write(6,"('g', /, ( 5ES12.4) )" ) prob%G( : prob%n )
!write(6,*) ' minloc g ', minloc_g( 1 )

!  now assign s^C_i = - x_i (i /= j) and s^C_j = 1 - x_j in d

           data%D( j ) = one - prob%X( j )
         END IF
         GO TO 310
       END IF

!  - - - - - - - - - - - - - - - augmented system - - - - - - - - - - - - - - -

!  compute the search direction d by minimizing the objective over
!  the free subspace by solving the augmented system

!    (    I        Ao_F   ) (  p  ) = (   b - A x  )  (a)
!    ( Ao_F^T  - weight I ) ( q_F )   ( weight x_F )

       IF ( data%direct_subproblem_solve ) THEN

!  set up the block matrices. Copy the free columns of A into the rows
!  of AT

!        IF ( data%n_free <= 12 ) write(6,"( ' free ', 12I5 )" ) data%FREE( : data%n_free )
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

           IF ( data%shifts ) THEN
             data%SBLS_sol( prob%o + i )                                       &
               = data%stabilisation_weight * ( prob%X( j ) - prob%X_s( j ) )
           ELSE
             data%SBLS_sol( prob%o + i )                                       &
               = data%stabilisation_weight * prob%X( j )
           END IF
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
                          data%SBLS_data, control%SBLS_control,                &
                          inform%SBLS_inform, data%SBLS_sol )

!  record the components q_F in data%S

         data%S( : data%n_free )                                               &
           = data%SBLS_sol( prob%o + 1 : prob%o + data%n_free )

!  if there are multiple cohorts, then solve

!    (    I        Ao_F   ) (  U^T  ) = (   0   )        (b)
!    ( Ao_F^T  - weight I ) ( V^T_F )   ( E^T_F )

!  and then form

!   (  r  ) = (  p  ) - (  U^T  ) y, 
!   ( d_F )   ( q_F )   ( V_F^T )

!  where E_F V_F^T y = E_F q_F                           (c)

         IF ( data%multiple_simplices ) THEN

!  for each cohort, set up its contribution to the right-hand side of (b)

           DO i = 1, prob%m
             data%SBLS_sol( : prob%o ) = zero
             DO j = 1, data%n_free
               IF ( prob%COHORT( data%FREE( j ) ) == i ) THEN
                 data%SBLS_sol( prob%o + j ) = one
               ELSE
                 data%SBLS_sol( prob%o + j ) = zero
               END IF
             END DO

!  solve system (b) for this right-hand side

             CALL SBLS_solve( prob%o, data%n_free, data%AT_sbls, data%C_sbls,  &
                              data%SBLS_data, control%SBLS_control,            &
                              inform%SBLS_inform, data%SBLS_sol )
!  record V_F^T

             data%VT( : data%n_free, i )                                       &
               = data%SBLS_sol( prob%o + 1 : prob%o + data%n_free )
           END DO

!  form the matrix - E_F V_F^T and right-hand side - E_F q_F

           data%EVT( : prob%m, : prob%m ) = zero ; data%Y( : prob%m, 1 ) = zero
           DO j = 1, data%n_free
             i = prob%COHORT( data%FREE( j ) )
             IF ( i <= 0 ) CYCLE
             data%EVT( i, 1 : prob%m )                                         &
               = data%EVT( i, 1 : prob%m ) - data%VT( j, 1 : prob%m )
             data%Y( i, 1 ) = data%Y( i, 1 ) - data%S( j )
           END DO

!  factorize this matrix

           CALL POTRF( 'L', prob%m, data%EVT, prob%m, inform%lapack_error )
           IF ( inform%lapack_error /= 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 910
           END IF

!  solve system (c) to get y

           CALL POTRS( 'L', prob%m, 1_ip_, data%EVT, prob%m, data%Y, prob%m,   &
                       inform%lapack_error )
           IF ( inform%lapack_error /= 0 ) THEN
             inform%status = GALAHAD_error_lapack ; GO TO 910
           END IF

!  form d_F = q_F - V_F^T y

           data%D( : prob%n ) = zero
           DO i = 1, data%n_free
             j = data%FREE( i )
             data%D( j ) = data%S( i )                                         &
                - DOT_PRODUCT( data%VT( i, : prob%m ), data%Y( : prob%m, 1 ) )
           END DO

!  debug output

           IF ( data%printd ) THEN
             DO i = 1, prob%m
               WRITE( data%out, "( ' cohort ', I0, ' e''s = ', ES12.4 )" )     &
                 i, SUM( data%D( data%S_ind( data%S_ptr( i ) :                 &
                                             data%S_ptr( i + 1 ) - 1 ) ) )
             END DO
             DO j = 1, prob%n
               WRITE( data%out, "( 'x,d', I3, 2ES12.4 )" )                     &
                j, prob%X( j ), data%D( j ) 
             END DO
           END IF

!  record y

           prob%Y( : prob%m ) = data%Y( : prob%m, 1 )

!  alternatively, if there is only a single cohort, then solve

!    (    I        Ao_F   ) (  u  ) = (  0  )            (d)
!    ( Ao_F^T  - weight I ) ( v_F )   ( e_F )

!  and form

!   (  r  ) = (  p  ) - lambda (  u  ), 
!   ( d_F )   ( q_F )          ( v_F )

!  where lambda = q_F^T e_F / v_F^T e_F                  (e)

         ELSE  ! single cohort

!  set up the right-hand side vector for system (d)

           data%SBLS_sol( : prob%o ) = zero
           data%SBLS_sol( prob%o + 1 : prob%o + data%n_free ) = one

!  solve system (d)

           CALL SBLS_solve( prob%o, data%n_free, data%AT_sbls, data%C_sbls,    &
                            data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform, data%SBLS_sol )

!  compute q_F^T e_F / v_F^T e_f

           lambda = SUM( data%S( : data%n_free ) ) /                           &
                      SUM( data%SBLS_sol( prob%o + 1 : prob%o + data%n_free ) )

!  record y

           prob%Y( 1 ) = lambda

!  extract the search direction d_f from (e)

           data%D( : prob%n ) = zero
           DO i = 1, data%n_free
             j = data%FREE( i )
             data%D( j ) = data%S( i ) - lambda * data%SBLS_sol( prob%o + i )
           END DO

         END IF

!write(6,*) ' ----------- nfree ', data%n_free
         IF ( data%printm ) WRITE( data%out, "( ' dtg, dtd = ', 2ES12.4 )" )   &
           DOT_PRODUCT( data%D( : prob%n ), prob%G( : prob%n ) ),              &
           DOT_PRODUCT( data%D( : prob%n ), data%D( : prob%n ) )
         data%string_cg_iter = '     F'
!write(6,"( ' ||D|| =', ES12.4 )" ) MAXVAL( ABS( data%D( : prob%n ) ) )
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
         reverse%lvl = 1
         reverse%lvu = data%n_free
         reverse%iv( : data%n_free ) = data%FREE( : data%n_free )
       END IF

!  minimization loop

!write(6,"( ' nfree = ', I0, ' FREE = ', /, ( 10I8 ) )" ) &
! data%n_free, data%FREE( : data%n_free )
!write(6,"( ' x = ', /, ( 5ES12.4 ) )" ) data%X_new( : prob%n )

 300   CONTINUE ! mock CGLS loop

!  find an improved point, X_new, in the multiple-simplex case by
!  conjugate-gradient least-squares ...

         IF ( data%multiple_simplices ) THEN

!  ... using the available Jacobian ...

           IF (  data%explicit_a ) THEN
             IF ( data%preconditioner == 0 ) THEN
               CALL SLLSM_cgls( prob%o, prob%n, prob%m, data%n_free,           &
                                prob%COHORT, data%weight,                      &
                                data%out, data%printm, data%printd, prefix,    &
!                               data%out, .TRUE., .FALSE., prefix,             &
                                data%phi_new, data%X_new, data%R, prob%Y,      &
                                data%FREE, control%stop_cg_relative,           &
                                control%stop_cg_absolute, data%cg_iter,        &
                                control%cg_maxit, data%subproblem_data,        &
                                userdata, data%cgls_status,                    &
                                inform%alloc_status, inform%bad_alloc,         &
                                prob%X_s, Ao_ptr = data%Ao%ptr,                &
                                Ao_row = data%Ao%row, Ao_val = data%Ao%val )
             ELSE IF ( data%preconditioner == 1 ) THEN
               CALL SLLSM_cgls( prob%o, prob%n, prob%m, data%n_free,           &
                                prob%COHORT, data%weight,                      &
                                data%out, data%printm, data%printd, prefix,    &
!                               data%out, .TRUE., .FALSE., prefix,             &
                                data%phi_new, data%X_new, data%R, prob%Y,      &
                                data%FREE, control%stop_cg_relative,           &
                                control%stop_cg_absolute, data%cg_iter,        &
                                control%cg_maxit, data%subproblem_data,        &
                                userdata, data%cgls_status,                    &
                                inform%alloc_status, inform%bad_alloc,         &
                                prob%X_s, Ao_ptr = data%Ao%ptr,                &
                                Ao_row = data%Ao%row, Ao_val = data%Ao%val,    &
                                preconditioned = .TRUE., DPREC = data%DIAG )
             ELSE
               CALL SLLSM_cgls( prob%o, prob%n, prob%m, data%n_free,           &
                                prob%COHORT, data%weight,                      &
                                data%out, data%printm, data%printd, prefix,    &
                                data%phi_new, data%X_new, data%R, prob%Y,      &
                                data%FREE, control%stop_cg_relative,           &
                                control%stop_cg_absolute, data%cg_iter,        &
                                control%cg_maxit, data%subproblem_data,        &
                                userdata, data%cgls_status,                    &
                                inform%alloc_status, inform%bad_alloc,         &
                                prob%X_s, Ao_ptr = data%Ao%ptr,                &
                                Ao_row = data%Ao%row, Ao_val = data%Ao%val,    &
                                reverse = reverse, preconditioned = .TRUE.,    &
                                eval_PREC = eval_PREC )
             END IF

!  ... or products via the user's subroutine or reverse communication ...

           ELSE
             IF ( data%preconditioner == 0 ) THEN
               CALL SLLSM_cgls( prob%o, prob%n, prob%m, data%n_free,           &
                                prob%COHORT, data%weight,                      &
                                data%out, data%printm, data%printd, prefix,    &
                                data%phi_new, data%X_new, data%R, prob%Y,      &
                                data%FREE, control%stop_cg_relative,           &
                                control%stop_cg_absolute, data%cg_iter,        &
                                control%cg_maxit, data%subproblem_data,        &
                                userdata, data%cgls_status,                    &
                                inform%alloc_status, inform%bad_alloc,         &
                                prob%X_s, eval_AFPROD = eval_AFPROD,           &
                                reverse = reverse )
             ELSE IF ( data%preconditioner == 1 ) THEN
               CALL SLLSM_cgls( prob%o, prob%n, prob%m, data%n_free,           &
                                prob%COHORT, data%weight,                      &
                                data%out, data%printm, data%printd, prefix,    &
                                data%phi_new, data%X_new, data%R, prob%Y,      &
                                data%FREE, control%stop_cg_relative,           &
                                control%stop_cg_absolute, data%cg_iter,        &
                                control%cg_maxit, data%subproblem_data,        &
                                userdata, data%cgls_status,                    &
                                inform%alloc_status, inform%bad_alloc,         &
                                prob%X_s, eval_AFPROD = eval_AFPROD,           &
                                reverse = reverse, preconditioned = .TRUE.,    &
                                DPREC = data%DIAG )
             ELSE
               CALL SLLSM_cgls( prob%o, prob%n, prob%m, data%n_free,           &
                                prob%COHORT, data%weight,                      &
                                data%out, data%printm, data%printd, prefix,    &
                                data%phi_new, data%X_new, data%R, prob%Y,      &
                                data%FREE, control%stop_cg_relative,           &
                                control%stop_cg_absolute, data%cg_iter,        &
                                control%cg_maxit, data%subproblem_data,        &
                                userdata, data%cgls_status,                    &
                                inform%alloc_status, inform%bad_alloc,         &
                                prob%X_s, eval_AFPROD = eval_AFPROD,           &
                                reverse = reverse, preconditioned = .TRUE.,    &
                                eval_PREC = eval_PREC )
             END IF
           END IF

!  find an improved point, X_new, in the multiple-simplex case by
!  conjugate-gradient least-squares ...

         ELSE 

!  ... using the available Jacobian ...

           IF (  data%explicit_a ) THEN
             IF ( data%preconditioner == 0 ) THEN
               CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,       &
                               data%out, data%printm, data%printd, prefix,     &
!                              data%out, .TRUE., .FALSE., prefix,              &
                               data%phi_new, data%X_new, data%R, prob%Y( 1 ),  &
                               data%FREE, control%stop_cg_relative,            &
                               control%stop_cg_absolute, data%cg_iter,         &
                               control%cg_maxit, data%subproblem_data,         &
                               userdata, data%cgls_status,                     &
                               inform%alloc_status, inform%bad_alloc,          &
                               prob%X_s, Ao_ptr = data%Ao%ptr,                 &
                               Ao_row = data%Ao%row, Ao_val = data%Ao%val )
             ELSE IF ( data%preconditioner == 1 ) THEN
               CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,       &
                               data%out, data%printm, data%printd, prefix,     &
!                              data%out, .TRUE., .FALSE., prefix,              &
                               data%phi_new, data%X_new, data%R, prob%Y( 1 ),  &
                               data%FREE, control%stop_cg_relative,            &
                               control%stop_cg_absolute, data%cg_iter,         &
                               control%cg_maxit, data%subproblem_data,         &
                               userdata, data%cgls_status,                     &
                               inform%alloc_status, inform%bad_alloc,          &
                               prob%X_s, Ao_ptr = data%Ao%ptr,                 &
                               Ao_row = data%Ao%row, Ao_val = data%Ao%val,     &
                               preconditioned = .TRUE., DPREC = data%DIAG )
             ELSE
               CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,       &
                               data%out, data%printm, data%printd, prefix,     &
                               data%phi_new, data%X_new, data%R, prob%Y( 1 ),  &
                               data%FREE, control%stop_cg_relative,            &
                               control%stop_cg_absolute, data%cg_iter,         &
                               control%cg_maxit, data%subproblem_data,         &
                               userdata, data%cgls_status,                     &
                               inform%alloc_status, inform%bad_alloc,          &
                               prob%X_s, Ao_ptr = data%Ao%ptr,                 &
                               Ao_row = data%Ao%row, Ao_val = data%Ao%val,     &
                               reverse = reverse, preconditioned = .TRUE.,     &
                               eval_PREC = eval_PREC )
             END IF

!  ... or products via the user's subroutine or reverse communication ...

           ELSE
             IF ( data%preconditioner == 0 ) THEN
               CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,       &
                               data%out, data%printm, data%printd, prefix,     &
                               data%phi_new, data%X_new, data%R, prob%Y( 1 ),  &
                               data%FREE, control%stop_cg_relative,            &
                               control%stop_cg_absolute, data%cg_iter,         &
                               control%cg_maxit, data%subproblem_data,         &
                               userdata, data%cgls_status,                     &
                               inform%alloc_status, inform%bad_alloc,          &
                               prob%X_s, eval_AFPROD = eval_AFPROD,            &
                               reverse = reverse )
             ELSE IF ( data%preconditioner == 1 ) THEN
               CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,       &
                               data%out, data%printm, data%printd, prefix,     &
                               data%phi_new, data%X_new, data%R, prob%Y( 1 ),  &
                               data%FREE, control%stop_cg_relative,            &
                               control%stop_cg_absolute, data%cg_iter,         &
                               control%cg_maxit, data%subproblem_data,         &
                               userdata, data%cgls_status,                     &
                               inform%alloc_status, inform%bad_alloc,          &
                               prob%X_s, eval_AFPROD = eval_AFPROD,            &
                               reverse = reverse, preconditioned = .TRUE.,     &
                               DPREC = data%DIAG )
             ELSE
               CALL SLLS_cgls( prob%o, prob%n, data%n_free, data%weight,       &
                               data%out, data%printm, data%printd, prefix,     &
                               data%phi_new, data%X_new, data%R, prob%Y( 1 ),  &
                               data%FREE, control%stop_cg_relative,            &
                               control%stop_cg_absolute, data%cg_iter,         &
                               control%cg_maxit, data%subproblem_data,         &
                               userdata, data%cgls_status,                     &
                               inform%alloc_status, inform%bad_alloc,          &
                               prob%X_s, eval_AFPROD = eval_AFPROD,            &
                               reverse = reverse, preconditioned = .TRUE.,     &
                               eval_PREC = eval_PREC )
             END IF
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
           data%branch = 300 ; inform%status = 5 ; RETURN

!  form the sparse matrix-vector product A^T * v

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

       IF ( data%multiple_simplices ) THEN


!  form A u (store in AE)

       END IF

!  re-entry point after the A u product

 400   CONTINUE
!      IF ( data%multiple_simplices .AND. data%reverse_prod )                  &
!        data%AE( : prob%n ) = reverse%p( : prob%n )

!  SBLS_sol is used as a surrogate for AE

       data%SBLS_sol( : prob%o ) = data%AE( : prob%o ) 

 420   CONTINUE ! mock arc search loop

!  find an improved point in the multiple-simplex case

         IF ( data%multiple_simplices ) THEN

!  find an improved point, X_new, by exact arc-search using the available
!  Jacobian ...

           IF ( data%explicit_a ) THEN
             CALL SLLSM_exact_arc_search( prob%n, prob%o, prob%m, data%n_c,    &
                                          prob%COHORT, data%S_ptr, data%S_ind, &
                                          data%weight, data%out,               &
                                          data%printm, data%printd,            &
 !                                        .TRUE., .FALSE.,                     &
                                          prefix, data%arc_search_status,      &
                                          data%X_new, data%R, data%D, data%AD, &
                                          data%sbls_sol, data%segment,         &
                                          data%n_free, data%FREE,              &
                                          data%search_data, userdata,          &
                                          data%f_new, data%phi_new,            &
                                          data%alpha_new,                      &
                                          prob%X_s, Ao_ptr = data%Ao%ptr,      &
                                          Ao_row = data%Ao%row,                &
                                          Ao_val = data%Ao%val )

!  ... or products via the user's subroutine or reverse communication ...

           ELSE
             CALL SLLSM_exact_arc_search( prob%n, prob%o, prob%m, data%n_c,    &
                                          prob%COHORT, data%S_ptr, data%S_ind, &
                                          data%weight, data%out,               &
                                          data%printm, data%printd,            &
                                          prefix, data%arc_search_status,      &
                                          data%X_new, data%R, data%D, data%AD, &
                                          data%sbls_sol, data%segment,         &
                                          data%n_free, data%FREE,              &
                                          data%search_data, userdata,          &
                                          data%f_new, data%phi_new,            &
                                          data%alpha_new,                      &
                                          prob%X_s, eval_APROD = eval_APROD,   &
                                          eval_ASCOL = eval_ASCOL,             &
                                          reverse = reverse )
           END IF

!  alternatively, find an improved point in the single-simplex case

         ELSE

!  find an improved point, X_new, by exact arc-search using the available
!  Jacobian ...

           IF ( data%explicit_a ) THEN
             CALL SLLS_exact_arc_search( prob%n, prob%o, data%weight,          &
                                         data%out, data%printm, data%printd,   &
!                                        data%out, .TRUE., .FALSE.,            &
                                         prefix, data%arc_search_status,       &
                                         data%X_new, data%R, data%D, data%AD,  &
                                         data%sbls_sol, data%segment,          &
                                         data%n_free, data%FREE,               &
                                         data%search_data, userdata,           &
                                         data%f_new, data%phi_new,             &
                                         data%alpha_new,                       &
                                         prob%X_s, Ao_ptr = data%Ao%ptr,       &
                                         Ao_row = data%Ao%row,                 &
                                         Ao_val = data%Ao%val )

!  ... or products via the user's subroutine or reverse communication ...

           ELSE
             CALL SLLS_exact_arc_search( prob%n, prob%o, data%weight,          &
                                         data%out, data%printm, data%printd,   &
                                         prefix, data%arc_search_status,       &
                                         data%X_new, data%R, data%D, data%AD,  &
                                         data%sbls_sol, data%segment,          &
                                         data%n_free, data%FREE,               &
                                         data%search_data, userdata,           &
                                         data%f_new, data%phi_new,             &
                                         data%alpha_new,                       &
                                         prob%X_s, eval_APROD = eval_APROD,    &
                                         eval_ASCOL = eval_ASCOL,              &
                                         reverse = reverse )
           END IF
         END IF

!  check the output status ("if" rather than "case" as tests use non-constants)

!  error exit without the new point

         IF ( data%arc_search_status <= - 2 ) THEN
           IF ( data%printe ) WRITE( control%error, 2010 )                     &
               prefix, data%arc_search_status, 'SLLS_exact_arc_search'
           inform%status = data%arc_search_status
           GO TO 910

!  successful exit with the new point

         ELSE IF ( data%arc_search_status <= 0 ) THEN
           data%norm_step =                                                    &
             MAXVAL( ABS( data%X_new( : prob%n ) - prob%X( : prob%n ) ) )

!  check function value if required

           IF ( data%printd ) THEN
             IF ( data%explicit_a ) THEN
               data%AE( : prob%o ) = - prob%B( : prob%o )
               DO j = 1, prob%n
                 x_j = data%X_new( j )
                 DO k = data%Ao%ptr( j ), data%Ao%ptr( j + 1 ) - 1
                   i = data%Ao%row( k )
                   data%AE( i ) = data%AE( i ) + data%Ao%val( k ) * x_j
                 END DO
               END DO
             ELSE IF ( data%use_aprod ) THEN
               data%AE( : prob%o ) = - prob%B( : prob%o )
               CALL eval_APROD( data%eval_status, userdata, .FALSE.,           &
                                data%X_new, data%AE )
             ELSE
               GO TO 500
             END IF
             val = DOT_PRODUCT( data%AE( : prob%o ), data%AE( : prob%o ) )
             IF ( data%weight > zero ) THEN
               IF ( data%shifts ) THEN
                 val = val + data%weight *                                     &
                   DOT_PRODUCT( data%X_new - prob%X_s, data%X_new - prob%X_s )
               ELSE
                 val = val + data%weight * DOT_PRODUCT( data%X_new, data%X_new )
               END IF
             END IF
             val = half * val
             WRITE( data%out, "( A, ' recurred, actual phi = ', 2ES22.14 )" )  &
               prefix, data%f_new, val
           END IF
           GO TO 500

!  compute the status-th column of A

         ELSE IF ( data%arc_search_status <= prob%n ) THEN
           reverse%index = data%arc_search_status
           data%branch = 420 ; inform%status = 4 ; RETURN

!  compute the product A d

         ELSE
           reverse%v( : prob%n ) = data%D( : prob%n )
           data%branch = 420 ; inform%status = 2 ; RETURN
         END IF
         GO TO 420 ! end of arc length loop

 440   CONTINUE
       data%alpha_0 = alpha_search

 450   CONTINUE ! mock arc search loop

!  find an improved point in the multiple-simplex case

         IF ( data%multiple_simplices ) THEN

!  find an improved point, X_new, by inexact arc_search, using the available
!  Jacobian ...

           IF (  data%explicit_a ) THEN
             CALL SLLSM_inexact_arc_search( prob%n, prob%o, prob%m,            &
                                            data%n_c, data%S_ptr, data%S_ind,  &
                                            data%weight, data%out,             &
                                            data%printm, data%printd,          &
!                                           .TRUE., .FALSE.,                   &
                                            prefix, data%arc_search_status,    &
                                            data%X_new, prob%R, data%D,        &
                                            data%sbls_sol, data%R,             &
                                            control%alpha_initial,             &
                                            control%alpha_reduction,           &
                                            control%arcsearch_acceptance_tol,  &
                                            control%arcsearch_max_steps,       &
!                                           control%alpha_max,                 &
!                                           control%advance,                   &
                                            data%n_free, data%FREE,            &
                                            data%search_data, userdata,        &
                                            data%X_c, data%X_c_proj,           &
                                            data%f_new, data%phi_new,          &
                                            data%alpha_new,                    &
                                            prob%X_s, Ao_ptr = data%Ao%ptr,    &
                                            Ao_row = data%Ao%row,              &
                                            Ao_val = data%Ao%val )

!  ... or products via the user's subroutine or reverse communication

           ELSE
             CALL SLLSM_inexact_arc_search( prob%n, prob%o, prob%m,            &
                                            data%n_c, data%S_ptr, data%S_ind,  &
                                            data%weight, data%out,             &
                                            data%printm, data%printd,          &
                                            prefix, data%arc_search_status,    &
                                            data%X_new, prob%R, data%D,        &
                                            data%sbls_sol, data%R,             &
                                            control%alpha_initial,             &
                                            control%alpha_reduction,           &
                                            control%arcsearch_acceptance_tol,  &
                                            control%arcsearch_max_steps,       &
!                                           control%alpha_max,                 &
!                                           control%advance,                   &
                                            data%n_free, data%FREE,            &
                                            data%search_data, userdata,        &
                                            data%X_c, data%X_c_proj,           &
                                            data%f_new, data%phi_new,          &
                                            data%alpha_new,                    &
                                            prob%X_s, eval_APROD = eval_APROD, &
                                            reverse = reverse )
           END IF

!  alternatively, find an improved point in the single-simplex case

         ELSE

!  find an improved point, X_new, by inexact arc_search, using the available
!  Jacobian ...

           IF (  data%explicit_a ) THEN
             CALL SLLS_inexact_arc_search( prob%n, prob%o, data%weight,        &
                                           data%out, data%printm, data%printd, &
!                                          data%out, .TRUE., .FALSE.,          &
                                           prefix, data%arc_search_status,     &
                                           data%X_new, prob%R, data%D,         &
                                           data%sbls_sol, data%R,              &
                                           control%alpha_initial,              &
                                           control%alpha_reduction,            &
                                           control%arcsearch_acceptance_tol,   &
                                           control%arcsearch_max_steps,        &
!                                          control%alpha_max,                  &
!                                          control%advance,                    &
                                           data%n_free, data%FREE,             &
                                           data%search_data, userdata,         &
                                           data%f_new, data%phi_new,           &
                                           data%alpha_new,                     &
                                           prob%X_s, Ao_ptr = data%Ao%ptr,     &
                                           Ao_row = data%Ao%row,               &
                                           Ao_val = data%Ao%val )

!  ... or products via the user's subroutine or reverse communication

           ELSE
             CALL SLLS_inexact_arc_search( prob%n, prob%o, data%weight,        &
                                           data%out, data%printm, data%printd, &
                                           prefix, data%arc_search_status,     &
                                           data%X_new, prob%R, data%D,         &
                                           data%sbls_sol, data%R,              &
                                           control%alpha_initial,              &
                                           control%alpha_reduction,            &
                                           control%arcsearch_acceptance_tol,   &
                                           control%arcsearch_max_steps,        &
!                                          control%alpha_max,                  &
!                                          control%advance,                    &
                                           data%n_free, data%FREE,             &
                                           data%search_data, userdata,         &
                                           data%f_new, data%phi_new,           &
                                           data%alpha_new,                     &
                                           prob%X_s, eval_APROD = eval_APROD,  &
                                           reverse = reverse )
           END IF
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
       inform%ls_obj = data%f_new
       inform%obj = data%phi_new
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

     IF ( ALLOCATED( prob%X_status ) ) THEN
       prob%X_status = - 1
       prob%X_status( data%FREE( : data%n_free ) ) = 0
!      WRITE( 6, "( ' X_status = ', /, ( 20I3 ) )" ) prob%X_status
       DO i = 1, prob%n
         IF ( prob%X( i ) > epsmch ) THEN
           prob%X_status( i ) = 0
         ELSE
           prob%X_status( i ) = - 1
         END IF
       END DO
     END IF
     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = time_now - data%time_start
     inform%time%clock_total = clock_now - data%clock_start
     IF ( data%printd ) WRITE( control%out, 2000 ) prefix, ' leaving '
     RETURN

!  error returns

 910 CONTINUE
     CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = time_now - data%time_start
     inform%time%clock_total = clock_now - data%clock_start

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
     TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse

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
       CALL REVERSE_terminate( reverse, inform%status, inform%alloc_status,    &
          bad_alloc = inform%bad_alloc, out = control%error,                   &
          deallocate_error_fatal = control%deallocate_error_fatal )
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

     array_name = 'slls: data%S_ptr'
     CALL SPACE_dealloc_array( data%S_ptr,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%S_ind'
     CALL SPACE_dealloc_array( data%S_ind,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%VT'
     CALL SPACE_dealloc_array( data%VT,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%EVT'
     CALL SPACE_dealloc_array( data%EVT,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%Y'
     CALL SPACE_dealloc_array( data%Y,                                         &
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

     array_name = 'slls: data%X_c'
     CALL SPACE_dealloc_array( data%X_c,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%X_c_proj'
     CALL SPACE_dealloc_array( data%X_c_proj,                                  &
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

     array_name = 'slls: data%search_data%P'
     CALL SPACE_dealloc_array( data%search_data%P,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%IP'
     CALL SPACE_dealloc_array( data%search_data%IP,                            &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%I_len'
     CALL SPACE_dealloc_array( data%search_data%I_len,                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%rhom_0'
     CALL SPACE_dealloc_array( data%search_data%rhom_0,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%rhom_1'
     CALL SPACE_dealloc_array( data%search_data%rhom_1,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%rhom_2'
     CALL SPACE_dealloc_array( data%search_data%rhom_2,                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%search_data%xim'
     CALL SPACE_dealloc_array( data%search_data%xim,                           &
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

     array_name = 'slls: data%prob%G'
     CALL SPACE_dealloc_array( data%prob%G,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'slls: data%prob%COHORT'
     CALL SPACE_dealloc_array( data%prob%COHORT,                               &
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

     CALL REVERSE_terminate( data%reverse, inform%status, inform%alloc_status, &
         bad_alloc = inform%bad_alloc, out = control%error,                    &
         deallocate_error_fatal = control%deallocate_error_fatal )

     RETURN

!  End of subroutine SLLS_full_terminate

     END SUBROUTINE SLLS_full_terminate

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

     array_name = 'slls: data%Y'
     CALL SPACE_dealloc_array( data%Y,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     array_name = 'slls: data%ETPEM'
     CALL SPACE_dealloc_array( data%ETPEM,                                     &
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

!       write(6,"( ' tau, h ', I10, 2ES12.4) ") j, sum / REAL( j, KIND = rp_ ),h
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

!  S L L S _ P R O J E C T _ O N T O _ S I M P L I C E S    S U B R O U T I N E

      SUBROUTINE SLLS_project_onto_simplices( n, m, n_c, S_ptr, S_ind, X,      &
                                              X_proj, X_c, X_c_proj, status )

!  Find the projection of a given x onto the intersection of multiple
!  non-overlapping unit simplices {x | e_Ci^T x_Ci = 1, x_Ci >= 0}, i = 1,...,m

!  The algorithm is essentially that from
!   E. van den Berg and M. P. Friedlander.
!   Probing the Pareto frontier for basis pursuit solutions.
!   SIAM Journal on Scientific Computing, 31(2):890—-912, 2008.

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  n      (INTEGER) the number of variables
!  m      (INTEGER) the number of simplices
!  n_c    (INTEGER) the maximum number of variables in a simplex
!  S_ptr  (INTEGER array of at least m+1) S_ptr(i) is the position in S_ind 
!         of the first index of a variable in simplex i (and one beyond the end)
!  S_ind  (INTEGER array of at least n) the indices of variables in each 
!         simplex, those for simplex i occur directly before those for simplex 
!         i+1, i=1,..,m-1. The indices of any variable that is not in any
!         simplex occur in positions 1 to S+ptr(1)-1
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
!
!  Workspace arguments
!
!  X_c      (REAL array of length at least n_c) workspace
!  X_c_proj (REAL array of length at least n_c) workspace

!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n, m, n_c
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m + 1 ) :: S_ptr
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: S_ind
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X_proj
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n_c ) :: X_c, X_c_proj

!  local variables

      INTEGER ( KIND = ip_ ) :: i, next_ptr, ni

!  leave any non-simplex variable alone

      IF ( S_ptr( 1 ) > 1 )                                                    &
        X_proj( S_ind( 1 : S_ptr( 1 ) - 1 ) ) = X( S_ind( 1 : S_ptr( 1 ) - 1 ) )

!  consider each cohort in turn 

      DO i = 1, m
        next_ptr = S_ptr( i + 1 )

!  the ith cohort has ni components

        ni = next_ptr - S_ptr( i )

!  copy these components of X into X_c

        X_c( 1 : ni ) = X( S_ind( S_ptr( i ) : next_ptr - 1 ) )

!  find the projection of these into the unit simplex

        CALL SLLS_project_onto_simplex( ni, X_c, X_c_proj, status )

!  map the projections back into the overall projection

        X_proj( S_ind( S_ptr( i ) : next_ptr - 1 ) ) = X_c_proj( 1 : ni )
      END DO

      RETURN

!  End of subroutine SLLS_project_onto_simplices

      END SUBROUTINE SLLS_project_onto_simplices

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
!       WRITE( 6, "( ' fix variable ', I0 )" ) i

!  step to the boundary

        V( i ) = zero
        DO l = 1, n_free
          j = FREE( l )
          V( j ) = V( j ) + t_min * S( j )
!         WRITE(6,"( ' x ', I8, ES12.4 )" ) j, V( j )
        END DO

!  update the search direction

        ete = ete - one
        gamma = S( i ) / ete
!       WRITE( 6, "( ' gamma = ', ES12.4 )" ) gamma
        DO i_free = 1, n_free
          j = FREE( i_free )
          IF ( S( j ) >= zero .AND. S( j ) + gamma < zero ) &
            write( 6, "( ' variable ', I6, ' now a candidate, t = ', ES12.4 )")&
               j, - V( j ) / ( S( j ) + gamma )
          S( j ) = S( j ) + gamma
!         WRITE( 6, "( ' s ', I8, ES12.4 )" ) j, S( j )
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
                                        f_opt, phi_opt, t_opt, X_s,            &
                                        Ao_val, Ao_row, Ao_ptr, reverse,       &
                                        eval_APROD, eval_ASCOL )

!  Let Delta = { x | e^T x = 1, x >= 0 } be the unit simplex. Follow the
!  projection path P( x + t d ) from a given x in Delta and direction d as
!  t increases from zero, and P(v) projects v onto Delta, and stop at the
!  first (local) minimizer of
!    1/2 || A P( x + t d ) - b ||^2 + 1/2 weight || P( x + t d ) - x_s ||^2

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
!            [1,n] the user should provide data for the status-th column of A
!                and re-enter the subroutine with all non-optional arguments
!                unchanged. Row indices and values for the nonzero components
!                of the column should be provided in reverse%ip(:nz) and
!                reverse%p(:nz), where nz is stored in reverse%lp;
!                these arrays should be allocated before use, and lengths of
!                at most o suffice. This value of status will only occur if
!                Ao_val, Ao_row and Ao_ptr are absent
!            >n the user should form the vector p = A v and re-enter the 
!                subroutine with all non-optional arguments unchanged. 
!                v will be provided in reverse%v(:n), and the product 
!                p must be returned in reverse%p(:n)
!  X        (REAL array of length at least n) the initial point x
!  R        (REAL array of length at least o) the residual A x - b
!  D        (REAL array of length at least n) the direction d
!  AD       (REAL array of length at least o) used as workspace
!  AE       (REAL array of length at least o) the vector A e where e is the
!           vector of ones, but subsequently used as workspace
!  segment  (INTEGER) the number of segments searched
!  n_free   (INTEGER) the number of free variables (i.e., variables not at zero)
!  FREE     (INTEGER array of length at least n) FREE(:n_free) are the indices
!            of the free variables
!  data     (structure of type slls_search_data_type) private data that is
!            preserved between calls to the subroutine
!  userdata  (structure of type USERDATA_type ) data that may be passed
!             between calls to the evaluation subroutine eval_ascol
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  f_opt    (REAL) the optimal objective value
!  phi_opt  (REAL) the optimal regularized objective value
!  t_opt    (REAL) the optimal step length
!
!  ALLOCATABLE ARGUMENTS

!  x_s      (REAL array of length n) the values of the (nonzeros) shifts
!            if allocated
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
 !           nonzeros of A, stored by consecutive columns. N.B. If present,
!            Ao_row and Ao_ptr must also be present
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!            of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!            Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!            with Ao_ptr(n+1) pointin to the storage location 1 beyond
!            the last entry in A
!  reverse (structure of type reverse_type) data that is provided by the
!           user when prompted by status > 0
!  eval_aprod (subroutine) that provides products with A(see slls_solve)
!  eval_ascol (subroutine) that provides a sparse column of A(see slls_solve)
!
!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n, o, out
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status, segment, n_free
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      LOGICAL, INTENT( IN ):: summary, debug
      REAL ( KIND = rp_ ), INTENT( OUT ) :: f_opt, phi_opt, t_opt
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: FREE
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, D
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R, AD, AE
      TYPE ( SLLS_search_data_type ), INTENT( INOUT ) :: data
      TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( IN ), DIMENSION( : ) :: X_s
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse
      OPTIONAL :: eval_APROD, eval_ASCOL

!  interface blocks

       INTERFACE
         SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
         USE GALAHAD_USERDATA_precision
         INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
         TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
         LOGICAL, INTENT( IN ) :: transpose
         REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
         REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
         END SUBROUTINE eval_APROD
       END INTERFACE

      INTERFACE
        SUBROUTINE eval_ASCOL( status, userdata, index, COL, ICOL, lcol )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: index
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: COL
        INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: ICOL
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: lcol
        END SUBROUTINE eval_ASCOL
      END INTERFACE

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, l, i_fixed, now_fixed, lp
!     REAL ( KIND = rp_ ) :: s_j
      REAL ( KIND = rp_ ) :: a, d_j, f_1, f_2, phi_1, phi_2, t
      REAL ( KIND = rp_ ) :: t_max = ten ** 20
!     REAL ( KIND = rp_ ), DIMENSION( n ) :: V, PROJ
!     REAL ( KIND = rp_ ), DIMENSION( o ) :: AE_tmp, AD_tmp

!  if a re-entry occurs, branch to the appropriate place in the code

      IF ( status > n ) GO TO 10
      IF ( status > 0 ) GO TO 200

!  see if shifts x_s have been provided

      IF ( ALLOCATED( X_s ) ) THEN
        data%shifts = SIZE( X_s ) >= n
      ELSE
        data%shifts = .FALSE.
      END IF

!  check to see if A has been provided

      IF ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) .AND.                     &
           PRESENT( Ao_val ) ) THEN
        data%present_a = .TRUE.
        data%reverse_a = .FALSE. ; data%reverse_as = .FALSE.
      ELSE
        data%present_a = .FALSE.

!  check to see if access to products with A by reverse communication has
!  been provided

        IF ( PRESENT( reverse ) ) THEN
          data%reverse_a = .TRUE. ; data%reverse_as = .TRUE.

!  check to see if access to products with A by subroutine call has been
!  provided

        ELSE IF ( PRESENT( eval_APROD ) .AND. PRESENT( eval_ASCOL ) ) THEN
          data%reverse_a = .FALSE. ; data%reverse_as = .FALSE.

!  if none of thses option is available, exit

        ELSE
          status = - 2 ; RETURN
        END IF
      END IF

!  ensure that the path goes somewhere

      IF ( MAXVAL( ABS( D ) ) == zero ) THEN
        t_opt = zero ; f_opt = data%f_0 ; phi_opt = data%phi_0 ; status = - 1
        RETURN
      END IF
      data%phi_1_stop = 100.0_rp_ * epsmch * REAL( n, KIND = rp_ )

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
!  s = d - ( e^T d / e^T e ) e and product As = Ad - ( e^T d / e^T e ) Ae

      data%t_total = zero
      data%ete = REAL( n, KIND = rp_ )
      data%gamma = SUM( D( : n ) ) / data%ete

!  project d into s (and store in D)

      IF ( data%gamma /= zero ) D = D - data%gamma

!  compute As (and store in AD)

       IF ( data%present_a ) THEN
         AD( : o ) = zero
         DO j = 1, n
           d_j = D( j )
           DO l = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
             i = Ao_row( l )
             AD( i ) = AD( i ) + Ao_val( l ) * d_j
           END DO
         END DO
       ELSE IF ( data%reverse_a ) THEN
         reverse%v( : n ) = D( : n )
         reverse%transpose = .FALSE.
         status = n + 1 ; RETURN
       ELSE
         AD( : o ) = zero
         CALL eval_APROD( status, userdata, .FALSE., D, AD )
         IF ( status /= GALAHAD_ok ) THEN
           status = GALAHAD_error_evaluation ; RETURN
         END IF
       END IF

!  re-entry point after forming the A d product

  10   CONTINUE
       IF ( data%reverse_a ) AD( : o ) = reverse%p( : o )

!  initialize f_0 = 1/2 || A x - b ||^2 and 
!  phi_0 = f_0 + 1/2 weight || x - x_s ||^2

      data%f_0 = half * DOT_PRODUCT( R, R )
      IF ( weight > zero ) THEN
        IF ( data%shifts ) THEN
          data%rho_0 = DOT_PRODUCT( X - X_s( : n ), X - X_s( : n ) )
        ELSE
          data%rho_0 = DOT_PRODUCT( X, X )
        END IF
        data%phi_0 = data%f_0 + half * weight * data%rho_0
      ELSE
        data%phi_0 = data%f_0
      END IF

!  if there is a regularization term, initialize rho_1 = (x - x_s)^T s,
!  rho_2 = ||s||^2 and xi = x_s^T e

      IF ( weight > zero ) THEN
        IF ( data%shifts ) THEN
          data%rho_1 = DOT_PRODUCT( X - X_s( : n ), D )
          data%xi = SUM( X_s( : n ) )
        ELSE
          data%rho_1 = DOT_PRODUCT( X, D )
        END IF
        data%rho_2 = DOT_PRODUCT( D, D )
      END IF

!  FREE(:n_free) are indices of variables that are free

      n_free = n
      FREE = (/ ( j, j = 1, n ) /)

!  main loop (mock do loop to allow reverse communication)

      segment = 1
      IF ( summary ) WRITE( out,  "( A, ' segment   phi_0       phi_1     ',   &
     &             '  phi_2       t_break       t_opt' )" ) prefix
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

!  compute the slope f_1/phi_1 and curvature f_2/phi_2 along the current segment

        f_1 = DOT_PRODUCT( R, AD ) ; f_2 = DOT_PRODUCT( AD, AD )
        IF ( weight > zero ) THEN
          phi_1 = f_1 + weight * data%rho_1 ; phi_2 = f_2 + weight * data%rho_2
        ELSE
          phi_1 = f_1 ; phi_2 = f_2
        END IF

!  stop if the slope is positive

        IF ( phi_1 > - data%phi_1_stop ) THEN
          t_opt = data%t_total ; f_opt = data%f_0 ; phi_opt = data%phi_0
          IF ( summary ) WRITE( out,  "( A, ' phi_opt =', ES12.4, ' at t =',   &
         &    ES11.4, ' at start of segment ', I0 )" )                         &
                prefix, phi_opt, t_opt, segment
          GO TO 900
        END IF

!  compute the step to the minimizer along the segment

        t_opt = - phi_1 / phi_2

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
          data%phi_0, phi_1, phi_2, data%t_total + data%t_break,               &
          data%t_total + t_opt
!       IF ( data%t_break == t_max .AND. debug )                               &
!         WRITE( out,  "( ' s = ', /, ( 5ES12.4 ) )" ) S

!  stop if the minimizer on the segment occurs before the end of the segment

        IF ( t_opt > zero .AND. t_opt <= data%t_break ) THEN
          f_opt = data%f_0 + f_1 * t_opt + half * f_2 * t_opt ** 2
          phi_opt = data%phi_0 + phi_1 * t_opt + half * phi_2 * t_opt ** 2
          DO l = 1, n_free
            j = FREE( l )
            X( j ) = X( j ) + t_opt * D( j )
!           IF ( debug ) WRITE( out, "( ' x ', I8, ES12.4 )" ) j, X( j )
          END DO
          t_opt = data%t_total + t_opt
          IF ( summary ) WRITE( out,  "( A, ' phi_opt =', ES12.4, ' at t =',   &
         &   ES11.4, ' in segment ', I0 )" ) prefix, phi_opt, t_opt, segment
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
        IF ( data%shifts ) data%x_s_fixed = X_s( now_fixed )
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

!  update f_0 and phi_0

        data%f_0 = data%f_0 + f_1 * data%t_break                               &
                     + half * f_2 * data%t_break ** 2
        data%phi_0 = data%phi_0 + phi_1 * data%t_break                         &
                     + half * phi_2 * data%t_break ** 2

!  if the fixed column of A is only availble by reverse communication, get it

        IF ( data%reverse_as ) THEN
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
        ELSE IF ( data%reverse_as ) THEN
          DO l = 1, reverse%lp
            i = reverse%ip( l ) ; a = reverse%p( l )
            AE( i ) = AE( i ) - a
            AD( i ) = AD( i ) - data%s_fixed * a
          END DO
        ELSE
          CALL eval_ASCOL( status, userdata, now_fixed, data%P, data%IP, lp )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; RETURN
          END IF

          DO l = 1, lp
            i = data%IP( l ) ; a = data%P( l )
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

          IF ( data%shifts ) THEN
            data%rho_1 = data%rho_1 + data%t_break * data%rho_2                &
              + data%gamma * ( one - data%xi + data%x_s_fixed *                &
                               REAL( n_free + 1, KIND = rp_ ) )
            data%xi = data%xi - data%x_s_fixed
          ELSE
            data%rho_1 = data%rho_1 + data%t_break * data%rho_2 + data%gamma
          END IF
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

! -*-*-*- S L L S M _ E X A C T _ A R C _ S E A R C H  S U B R O U T I N E -*-*-

      SUBROUTINE SLLSM_exact_arc_search( n, o, m, n_c, COHORT, S_ptr, S_ind,   &
                                         weight, out, summary, debug, prefix,  &
                                         status, X, R, D, AD, AE, segment,     &
                                         n_free, FREE, data, userdata,         &
                                         f_opt, phi_opt, t_opt, X_s,           &
                                         Ao_val, Ao_row, Ao_ptr, reverse,      &
                                         eval_APROD, eval_ASCOL )

!  Let Delta_i = { x | e_Ci^T x_Ci = 1, x_Ci >= 0 } be unit simplices over a
!  set of non-overlapping index sets (cohorts) Ci in {i,...,n} for i=1,...,m.
!  Follow the projection path P( x + t d ) from a given x in the intersection 
!  Delta of the Delta_i and direction d as t increases from zero, and P(v) 
!  projects v onto Delta, and stop at the first (local) minimizer of
!    1/2 || A P( x + t d ) - b ||^2 + 1/2 weight || P( x + t d ) - x_s ||^2

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  n        (INTEGER) the number of variables
!  o        (INTEGER) the number of observations
!  m        (INTEGER) the number of cohorts
!  n_c      (INTEGER) the maximum number of variables in any cohort
!  COHORT   (INTEGER array of length n) states the cohort that each variable
!            belongs to, variable j lies in cohort COHORT(j), j = 1,...,n
!            (or is unconstrained if COHORT(j) = 0) 
!  S_ptr    (INTEGER array of length m+1) pointer to the start of each cohort,
!            S_ptr(i) points to the start of cohort i, i = 1,...,m, and
!            S_ptr(m+1) = n+1
!  S_ind    (INTEGER array of length n) variables in each cohort, variables
!            with indices S_ind(S_ptr(i):S_ptr(i+1)-1) occur in cohort i
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
!            [1,n] the user should provide data for the status-th column of A
!                and re-enter the subroutine with all non-optional arguments
!                unchanged. Row indices and values for the nonzero components
!                of the column should be provided in reverse%ip(:nz) and
!                reverse%p(:nz), where nz is stored in reverse%lp;
!                these arrays should be allocated before use, and lengths of
!                at most o suffice. This value of status will only occur if
!                Ao_val, Ao_row and Ao_ptr are absent
!  X        (REAL array of length at least n) the initial point x
!  R        (REAL array of length at least o) the residual A x - b
!  D        (REAL array of length at least n) the direction d
!  AD       (REAL array of length at least o) a vector used as workspace
!  AE       (REAL array of length at least o) the vector A e where e is the
!           vector of ones, but subsequently used as workspace
!  segment  (INTEGER) the number of segments searched
!  n_free   (INTEGER) the number of free variables (i.e., variables not at zero)
!  FREE     (INTEGER array of length at least n) FREE(:n_free) are the indices
!            of the free variables
!  data     (structure of type slls_search_data_type) private data that is
!            preserved between calls to the subroutine
!  userdata  (structure of type USERDATA_type ) data that may be passed
!             between calls to the evaluation subroutine eval_ascol
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  f_opt    (REAL) the optimal objective value
!  phi_opt  (REAL) the optimal regularized objective value
!  t_opt    (REAL) the optimal step length
!
!  ALLOCATABLE ARGUMENTS

!  X_s      (REAL array of length n) the values of the (nonzeros) shifts
!            if allocated
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!            nonzeros of A, stored by consecutive columns. N.B. If present,
!            Ao_row and Ao_ptr must also be present
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!            of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!            Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!            with Ao_ptr(n+1) pointin to the storage location 1 beyond
!            the last entry in A
!  reverse (structure of type reverse_type) data that is provided by the
!           user when prompted by status > 0
!  eval_aprod (subroutine) that provides products with A(see slls_solve)
!  eval_ascol (subroutine) that provides a sparse column of A(see slls_solve)
!
!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n, o, m, n_c, out
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status, segment, n_free
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      LOGICAL, INTENT( IN ):: summary, debug
      REAL ( KIND = rp_ ), INTENT( OUT ) :: f_opt, phi_opt, t_opt
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: COHORT, S_ind
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m + 1 ) :: S_ptr
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: FREE
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, D
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R, AD, AE
      TYPE ( SLLS_search_data_type ), INTENT( INOUT ) :: data
      TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( IN ), DIMENSION( : ) :: X_s
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse
      OPTIONAL :: eval_APROD, eval_ASCOL

!  interface blocks

      INTERFACE
        SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        LOGICAL, INTENT( IN ) :: transpose
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
        END SUBROUTINE eval_APROD
      END INTERFACE

      INTERFACE
        SUBROUTINE eval_ASCOL( status, userdata, index, COL, ICOL, lcol )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: index
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: COL
        INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: ICOL
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: lcol
        END SUBROUTINE eval_ASCOL
      END INTERFACE

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, l, l_j, lp, i_fixed, now_fixed
!     REAL ( KIND = rp_ ) :: s_j
      REAL ( KIND = rp_ ) :: a, x_j, d_j, f_1, f_2, phi_1, phi_2, t
      REAL ( KIND = rp_ ) :: t_max = ten ** 20
!     REAL ( KIND = rp_ ), DIMENSION( n ) :: V
!     REAL ( KIND = rp_ ), DIMENSION( n ) :: V, PROJ
!     REAL ( KIND = rp_ ), DIMENSION( o ) :: AE_tmp, AD_tmp
!     REAL ( KIND = rp_ ), DIMENSION( m ) :: C

!  if a re-entry occurs, branch to the appropriate place in the code

      IF ( status > n ) GO TO 10
      IF ( status > 0 ) GO TO 200

!  see if shifts x_s have been provided

      IF ( ALLOCATED( X_s ) ) THEN
        data%shifts = SIZE( X_s ) >= n
      ELSE
        data%shifts = .FALSE.
      END IF

!  check to see if A has been provided

      IF ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) .AND.                     &
           PRESENT( Ao_val ) ) THEN
        data%present_a = .TRUE.
        data%reverse_a = .FALSE. ; data%reverse_as = .FALSE.
      ELSE
        data%present_a = .FALSE.

!  check to see if access to products with A by reverse communication has
!  been provided

        IF ( PRESENT( reverse ) ) THEN
          data%reverse_a = .TRUE. ; data%reverse_as = .TRUE.

!  check to see if access to products with A by subroutine call has been
!  provided

        ELSE IF ( PRESENT( eval_APROD ) .AND. PRESENT( eval_ASCOL ) ) THEN
          data%reverse_a = .FALSE. ; data%reverse_as = .FALSE.

!  if none of thses option is available, exit

        ELSE
          status = - 2 ; RETURN
        END IF
      END IF

!  ensure that the path goes somewhere

      IF ( MAXVAL( ABS( D ) ) == zero ) THEN
        t_opt = zero ; f_opt = data%f_0 ; phi_opt = data%phi_0 ; status = - 1
        RETURN
      END IF
      data%phi_1_stop = 100.0_rp_ * epsmch * REAL( n, KIND = rp_ )

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

!  project d onto the manifolds v_Cj^T s_Cj = 0 to compute the search 
!  direction s = d - u and product As = Ad - Au, where u = sum w_j e_Cj
!  and w_j = e_Cj^T d_Cj / |Cj|

      data%t_total = zero

!  project d into s (and store in D)

      DO i = 1, m
        l = S_ptr( i + 1 ) - 1
        l_j = l + 1 - S_ptr( i )
        data%I_len( i ) = l_j
        IF ( l_j > 0 ) THEN
          D( S_ind( S_ptr( i ) : l ) ) = D( S_ind( S_ptr( i ) : l ) ) -        &
            SUM( D( S_ind( S_ptr( i ) : l ) ) ) / REAL( l_j, KIND = rp_ )
        END IF  
      END DO
 
!  compute As (and store in AD)

      IF ( data%present_a ) THEN
        AD( : o ) = zero
        DO j = 1, n
          d_j = D( j )
          DO l = Ao_ptr( j ), Ao_ptr( j + 1 ) - 1
            i = Ao_row( l )
            AD( i ) = AD( i ) + Ao_val( l ) * d_j
          END DO
        END DO
      ELSE IF ( data%reverse_a ) THEN
        reverse%v( : n ) = D( : n )
        reverse%transpose = .FALSE.
        status = n + 1 ; RETURN
      ELSE
        AD( : o ) = zero
        CALL eval_APROD( status, userdata, .FALSE., D, AD )
        IF ( status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_evaluation ; RETURN
        END IF
      END IF

!  re-entry point after forming the A d product

  10  CONTINUE
      IF ( data%reverse_a ) AD( : o ) = reverse%p( : o )

!  initialize f_0 = 1/2 || A x - b ||^2 + 1/2 weight || x - x_s ||^2, and
!  if there is a regularization term, initialize rhom_1 = (x - x_s)^T s,
!  rhom_2 = ||s||^2 and xi = x_s^T e

      data%f_0 = half * DOT_PRODUCT( R, R )
      IF ( weight > zero ) THEN
        data%rhom_0( 0 : m ) = zero ; data%rhom_1( 0 : m ) = zero
        data%rhom_2( 0 : m ) = zero
        data%rho_0 = zero ; data%rho_2 = zero ; data%rho_2 = zero
        IF ( data%shifts ) data%xim( 1 : m ) = zero 
        DO j = 1, n
          IF ( data%shifts ) THEN
            x_j = X( j ) - X_s( j )
          ELSE
            x_j = X( j )
          END IF
          d_j = D( j )
          i = COHORT( j )
          IF ( i > 0 ) THEN
            IF ( data%shifts ) data%xim( i ) = data%xim( i ) + X_s( j )
          ELSE
            i = 0
          END IF
          data%rhom_0( i ) = data%rhom_0( i ) + x_j * x_j
          data%rhom_1( i ) = data%rhom_1( i ) + x_j * d_j
          data%rhom_2( i ) = data%rhom_2( i ) + d_j * d_j
        END DO
        data%phi_0 = data%f_0 + half * weight * SUM( data%rhom_0( 0 : m ) )
      ELSE
        data%phi_0 = data%f_0
      END IF

!  FREE(:n_free) are indices of variables that are free

      n_free = n
      FREE = (/ ( j, j = 1, n ) /)

!  main loop (mock do loop to allow reverse communication)

      segment = 1
      IF ( summary ) WRITE( out,  "( A, ' segment   phi_0       phi_1     ',   &
     &             '  phi_2       t_break       t_opt    fixed' )" ) prefix

  100 CONTINUE
if(summary) write(6,"( ' free(2) ', I0 )" ) free(2)

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

!  compute the slope phi_1 and curvature phi_2 along the current segment

        f_1 = DOT_PRODUCT( R, AD ) ; f_2 = DOT_PRODUCT( AD, AD )
        IF ( weight > zero ) THEN
          phi_1 = f_1 + weight * SUM( data%rhom_1( 0 : m ) )
          phi_2 = f_2 + weight * SUM( data%rhom_2( 0 : m ) )
        ELSE
          phi_1 = f_1
          phi_2 = f_2
        END IF

!  stop if the slope is positive

        IF ( phi_1 > - data%phi_1_stop ) THEN
          t_opt = data%t_total ; f_opt = data%f_0 ; phi_opt = data%phi_0
          IF ( summary ) WRITE( out,  "( A, ' phi_opt =', ES12.4, ' at t =',   &
         &    ES11.4, ' at start of segment ', I0 )" )                         &
                prefix, phi_opt, t_opt, segment
          GO TO 900
        END IF

!  compute the step to the minimizer along the segment

        t_opt = - phi_1 / phi_2

!  compute the step along s to the boundary of Delta^n, as well as the
!  index i_fixed of the position in FREE of the variable that meets its bound
!  and ic, the cohort containing that variable

        IF ( debug )                                                           &
          WRITE( out,  "( A, 14X, '  t           x           s' )" ) prefix
        data%t_break = t_max
        DO i = 1, n_free
          j = FREE( i )
          IF ( COHORT( j ) <= 0 ) CYCLE
          IF ( D( j ) < zero ) THEN
            t = - X( j ) / D( j )
            IF ( debug )                                                       &
              WRITE( out,  "( A, I8, 3ES12.4 )" ) prefix, j, t, X( j ), D( j )
            IF ( t < data%t_break ) THEN
              i_fixed = i ; data%t_break = t ; data%ic = COHORT( j )
            END IF
          END IF
        END DO
        IF ( summary ) WRITE( out,  "( A, I8, 5ES12.4, 1X, I0 )" ) prefix,     &
          segment, data%phi_0, phi_1, phi_2, data%t_total + data%t_break,      &
          data%t_total + t_opt, FREE( i_fixed )
!       IF ( data%t_break == t_max .AND. debug )                               &
!         WRITE( out,  "( ' s = ', /, ( 5ES12.4 ) )" ) S

!  stop if the minimizer on the segment occurs before the end of the segment

        IF ( t_opt > zero .AND. t_opt <= data%t_break ) THEN
          f_opt = data%f_0 + f_1 * t_opt + half * f_2 * t_opt ** 2
          phi_opt = data%phi_0 + phi_1 * t_opt + half * phi_2 * t_opt ** 2
          DO l = 1, n_free
            j = FREE( l )
            X( j ) = X( j ) + t_opt * D( j )
!           IF ( debug ) WRITE( out, "( ' x ', I8, ES12.4 )" ) j, X( j )
          END DO
          t_opt = data%t_total + t_opt
          IF ( summary ) WRITE( out,  "( A, ' phi_opt =', ES12.4, ' at t =',   &
         &   ES11.4, ' in segment ', I0 )" ) prefix, phi_opt, t_opt, segment
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

        data%s_fixed = D( now_fixed )
        data%gamma = data%s_fixed / REAL( data%I_len( data%ic ) - 1, KIND = rp_)
        IF ( data%shifts ) data%x_s_fixed = X_s( now_fixed )
!       IF ( debug ) WRITE( out, "( ' data%gamma = ', ES12.4 )" ) data%gamma
        DO i = 1, n_free
          j = FREE( i )
!         IF ( D( j ) >= zero .AND. D( j ) + data%gamma < zero .AND. debug )   &
!           WRITE( out,                                                        &
!              "( ' variable ', I6, ' now a candidate, t = ', ES12.4 )")       &
!              j, - X( j ) / ( D( j ) + data%gamma )
          IF ( COHORT( j ) == data%ic ) D( j ) = D( j ) + data%gamma
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

!  update f_0 and phi_0

        data%f_0                                                               &
          = data%f_0 + f_1 * data%t_break + half * f_2 * data%t_break ** 2
        data%phi_0                                                             &
          = data%phi_0 + phi_1 * data%t_break + half * phi_2 * data%t_break ** 2

!  if the fixed column of A is only availble by reverse communication, get it

        IF ( data%reverse_as ) THEN
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
        ELSE IF ( data%reverse_as ) THEN
          DO l = 1, reverse%lp
            i = reverse%ip( l ) ; a = reverse%p( l )
            AE( i ) = AE( i ) - a
            AD( i ) = AD( i ) - data%s_fixed * a
          END DO
        ELSE
          CALL eval_ASCOL( status, userdata, now_fixed, data%P, data%IP, lp )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; RETURN
          END IF

          DO l = 1, lp
            i = data%IP( l ) ; a = data%P( l )
            AE( i ) = AE( i ) - a
            AD( i ) = AD( i ) - data%s_fixed * a
          END DO
        END IF

        R = R + data%t_break * AD
        AD = AD + data%gamma * AE

!  update rhom_0, rhom_1 and rhom_2 if required. Loop over cohorts

        IF ( weight > zero ) THEN
!         DO i = 1, m
          DO i = 0, m

!  the fixed variable belongs to cohort i

            IF ( i == data%ic ) THEN
              data%rhom_0( i ) = data%rhom_0( i ) + data%t_break *             &
                ( two * data%rhom_1( i ) + data%t_break * data%rhom_2( i ) )
              IF ( data%shifts ) THEN
                data%rhom_1( i ) = data%rhom_1( i ) + data%t_break *           &
                  data%rhom_2( i ) + data%gamma * ( one - data%xim( i )        &
                    + data%x_s_fixed * REAL( data%I_len( i ), KIND = rp_ ) )
                data%xim( i ) = data%xim( i ) - data%x_s_fixed
!               data%I_len( i ) = data%I_len( i ) - 1
              ELSE
                data%rhom_1( i ) = data%rhom_1( i ) + data%t_break *           &
                  data%rhom_2( i ) + data%gamma
              END IF
              data%rhom_2( i ) = data%rhom_2( i )                              &
                - data%s_fixed * ( data%s_fixed + data%gamma )

!  the fixed variable doesn't belongs to cohort i

            ELSE
              data%rhom_0( i ) = data%rhom_0( i ) + data%t_break *             &
                ( two * data%rhom_1( i ) + data%t_break * data%rhom_2( i ) )
              data%rhom_1( i ) = data%rhom_1( i )                              &
                + data%t_break * data%rhom_2( i )
            END IF
          END DO
        END IF
        data%I_len( data%ic ) = data%I_len( data%ic ) - 1

        segment = segment + 1
        IF ( segment < n ) GO TO 100

!  end of main (mock) loop

  900 CONTINUE

!  record free variables

      n_free = 0
      DO j = 1, n
        IF ( COHORT( j ) <= 0 .OR. X( j ) >= x_zero ) THEN
          n_free = n_free + 1
          FREE( n_free ) = j
        END IF
      END DO

!     IF ( n_free <= 12 ) write(6,"( ' ** free ', 12I5 )" ) FREE( : n_free )
      status = 0

      RETURN

!  End of subroutine SLLSM_exact_arc_search

      END SUBROUTINE SLLSM_exact_arc_search

! -*-*- S L L S _ I N E X A C T _ A R C _ S E A R C H  S U B R O U T I N E -*-*-

      SUBROUTINE SLLS_inexact_arc_search( n, o, weight, out, summary, debug,   &
                                          prefix, status, X, R, D, S, R_t,     &
                                          t_0, beta, eta, max_steps,           &
!                                         t_max, advance,                      &
                                          n_free, FREE, data, userdata,        &
                                          f_opt, phi_opt, t_opt,               &
                                          X_s, Ao_val, Ao_row,                 &
                                          Ao_ptr, reverse , eval_APROD )

!  Let Delta = { x | e^T x = 1, x >= 0 } be the unit simplex. Follow the
!  projection path x(t) = P( x + t d ) from a given x and direction d for
!  a sequence of decreasing/increasing values of t, from an initial value
!  t_0 > 0, to find an approximate local minimizer of the regularized
!  least-squares objective
!
!    f(x) = 1/2 || A_o x - b ||^2 + 1/2 weight || x - x_s ||^2 for x = P(x(t))
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
!  userdata  (structure of type USERDATA_type ) data that may be passed
!             between calls to the evaluation subroutine eval_prod
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  f_opt    (REAL) the optimal objective value
!  phi_opt  (REAL) the optimal regularized objective value
!  t_opt    (REAL) the optimal step length
!
!  ALLOCATABLE ARGUMENTS

!  X_s      (REAL array of length n) the values of the (nonzeros) shifts
!            if allocated
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!            nonzeros of A, stored by consecutive columns. N.B. If present,
!            Ao_row and Ao_ptr must also be present
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!            of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!            Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!            with Ao_ptr(n+1) pointin to the storage location 1 beyond
!            the last entry in A
!  reverse (structure of type reverse_type) data that is provided by the
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
      REAL ( KIND = rp_ ), INTENT( OUT ) :: f_opt, phi_opt, t_opt
      REAL ( KIND = rp_ ), INTENT( IN ) :: t_0, beta, eta
!     REAL ( KIND = rp_ ), INTENT( IN ) :: t_max
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: FREE
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: D
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( o ) :: R
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, S
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R_t
      TYPE ( SLLS_search_data_type ), INTENT( INOUT ) :: data
      TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( IN ), DIMENSION( : ) :: X_s
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse
      OPTIONAL :: eval_APROD

!  interface blocks

      INTERFACE
        SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        LOGICAL, INTENT( IN ) :: transpose
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
        END SUBROUTINE eval_APROD
      END INTERFACE

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, l
      REAL ( KIND = rp_ ) :: f_t, phi_t, gts_t, s_j

!  if a re-entry occurs, branch to the appropriate place in the code

      IF ( status > 0 ) GO TO 200

!  see if shifts x_s have been provided

      IF ( ALLOCATED( X_s ) ) THEN
        data%shifts = SIZE( X_s ) >= n
      ELSE
        data%shifts = .FALSE.
      END IF

!  check to see if Ao has been provided

      IF ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) .AND.                     &
           PRESENT( Ao_val ) ) THEN
        data%present_a = .TRUE. ; data%reverse_a = .FALSE.
      ELSE
        data%present_a = .FALSE.

!  check to see if access to products with A by reverse communication has
!  been provided

        IF ( PRESENT( reverse ) ) THEN
          data%reverse_a = .TRUE.

!  check to see if access to products with A by subroutine call has been
!  provided

        ELSE IF ( PRESENT( eval_APROD ) ) THEN
          data%reverse_a = .FALSE.

!  if none of thses option is available, exit

        ELSE
          status = - 2 ; RETURN
        END IF
      END IF

!  compute r^T r and f = 1/2 || r ||^2 + 1/2 weight || x ||^2

      data%rtr = DOT_PRODUCT( R, R )
      data%f_0 = half * data%rtr
      IF ( weight > zero ) THEN
        IF ( data%shifts ) THEN
          data%xtx = DOT_PRODUCT( X - X_s( : n ), X - X_s( : n ) )
        ELSE
          data%xtx = DOT_PRODUCT( X, X )
        END IF
        data%phi_0 = data%f_0 + half * weight * data%xtx
      ELSE
        data%phi_0 = data%f_0
      END IF

!  ensure that the path goes somewhere

      IF ( MAXVAL( ABS( D ) ) == zero ) THEN
        t_opt = zero ; f_opt = data%f_0 ; phi_opt = data%phi_0 ; status = - 1
        RETURN
      END IF

      IF ( summary ) THEN
        WRITE( out, 2000 ) prefix, prefix
        WRITE( out, 2010 ) prefix, 0, zero, zero, data%phi_0
      END IF

!  main loop (mock do loop to allow reverse communication)

      data%step = 1 ; data%t = t_0 ; data%backwards = .TRUE.
  100 CONTINUE

!  store the projection P(x + t d) of x + t d onto Delta^n in s

        CALL SLLS_project_onto_simplex( n, X + data%t * D, S, status )
        IF ( weight > zero ) THEN
          IF ( data%shifts ) THEN
            data%xtx = DOT_PRODUCT( S - X_s( : n ), S - X_s( : n ) )
          ELSE
            data%xtx = DOT_PRODUCT( S, S )
          END IF
        END IF

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

        ELSE IF ( data%reverse_a ) THEN
!         reverse%p( : o ) = R
          reverse%v( : n ) = S
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
        IF ( data%reverse_a ) R_t = R + reverse%p( : o )
        IF ( debug ) WRITE( out, "( ' R_t =', / ( 5ES12.4 ) )" ) R

!  compute f_t = 1/2 || r_t ||^2 + 1/2 weight || x_t ||^2

        f_t = half * DOT_PRODUCT( R_t, R_t )
        IF ( weight > zero ) THEN
          phi_t = f_t + half * weight * data%xtx
        ELSE
          phi_t = f_t
        END IF

!  compute g^T s_t = r^T A s_t = r^T r_t - r^T r + weight * x^T s_t

        gts_t = DOT_PRODUCT( R_t, R ) - data%rtr
        IF ( weight > zero ) THEN
          IF ( data%shifts ) THEN
            gts_t = gts_t + weight * DOT_PRODUCT( X - X_s( : n ), S )
          ELSE
            gts_t = gts_t + weight * DOT_PRODUCT( X, S )
          END IF
        END IF
        IF ( debug ) WRITE( out, 2000 ) prefix, prefix
        IF ( summary ) WRITE( out, 2010 )                                      &
          prefix, data%step, data%t, TWO_NORM( S ), phi_t

!  test for sufficient decrease

        IF ( data%backwards ) THEN ! this is irrelevant at present

!  exit if the decrease is sufficient with the optimal x, f and status arrays

          IF ( phi_t <= data%phi_0 + eta * gts_t ) THEN
            X = X + S ; f_opt = f_t ; phi_opt = phi_t ; t_opt = data%t
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

 2000 FORMAT( /, A, '   inexact arc search ', /,                               &
                 A, '       k      t           s          phi' )
 2010 FORMAT( A, I8, 3ES12.4 )

!  End of subroutine SLLS_inexact_arc_search

      END SUBROUTINE SLLS_inexact_arc_search

! -*-*- S L L S M _ I N E X A C T _ A R C _ S E A R C H  S U B R O U T I N E -*-

      SUBROUTINE SLLSM_inexact_arc_search( n, o, m, n_c, S_ptr, S_ind,         &
                                           weight, out, summary, debug,        &
                                           prefix, status, X, R, D, S, R_t,    &
                                           t_0, beta, eta, max_steps,          &
!                                          t_max, advance,                     &
                                           n_free, FREE, data, userdata,       &
                                           X_c, X_c_proj,                      &
                                           f_opt, phi_opt, t_opt,              &
                                           X_s, Ao_val, Ao_row,                &
                                           Ao_ptr, reverse, eval_APROD )

!  Let Delta_i = { x | e_Ci^T x_Ci = 1, x_Ci >= 0 } be unit simplices 
!  over a set of nonoverlapping index sets C_i in {i,...,n} for i=1,...,m.
!  Follow the projection path P( x + t d ) from a given x in the intersection 
!  Delta of the Delta_i and direction d for a sequence of decreasing/increasing
!  values of t, from an initial value t_0 > 0, to find an approximate local 
!  minimizer of the regularized least-squares objective
!
!    f(x) = 1/2 || A_o x - b ||^2 + 1/2 weight || x - x_s ||^2 for x = P(x(t))
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
!  m         (INTEGER) the number of cohorts
!  n_c       (INTEGER) the maximum number of variables in any cohort
!  S_ptr     (INTEGER array of length m+1) pointer to the start of each cohort,
!             S_ptr(i) points to the start of cohort i, i = 1,...,m, and
!             S_ptr(m+1) = n+1
!  S_ind     (INTEGER array of length n) variables in each cohort, variables
!             with indices S_ind(S_ptr(i):S_ptr(i+1)-1) occur in cohort i
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
!  userdata  (structure of type USERDATA_type ) data that may be passed
!             between calls to the evaluation subroutine eval_prod
!
!  OUTPUT arguments that need not be set on entry to the subroutine
!
!  f_opt    (REAL) the optimal objective value
!  phi_opt  (REAL) the optimal regularized objective value
!  t_opt    (REAL) the optimal step length
!
!  ALLOCATABLE ARGUMENTS

!  x_s      (REAL array of length n) the values of the (nonzeros) shifts
!            if allocated
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!            nonzeros of A, stored by consecutive columns. N.B. If present,
!            Ao_row and Ao_ptr must also be present
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!            of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!            Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!            with Ao_ptr(n+1) pointin to the storage location 1 beyond
!            the last entry in A
!  reverse (structure of type reverse_type) data that is provided by the
!           user when prompted by status > 0
!  eval_aprod (subroutine) that provides products with A (see slls_solve)
!
!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: n, o, m, n_c, out, max_steps
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status, n_free
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      LOGICAL, INTENT( IN ):: summary, debug
!     LOGICAL, INTENT( IN ):: advance
      REAL ( KIND = rp_ ), INTENT( OUT ) :: f_opt, phi_opt, t_opt
      REAL ( KIND = rp_ ), INTENT( IN ) :: t_0, beta, eta
!     REAL ( KIND = rp_ ), INTENT( IN ) :: t_max
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: S_ind
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( m + 1 ) :: S_ptr
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n ) :: FREE
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: D
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( o ) :: R
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X, S
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R_t
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n_c ) :: X_c, X_c_proj
      TYPE ( SLLS_search_data_type ), INTENT( INOUT ) :: data
      TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( IN ), DIMENSION( : ) :: X_s
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse
      OPTIONAL :: eval_APROD

!  interface blocks

      INTERFACE
        SUBROUTINE eval_APROD( status, userdata, transpose, V, P )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        LOGICAL, INTENT( IN ) :: transpose
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
        END SUBROUTINE eval_APROD
      END INTERFACE

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, l
      REAL ( KIND = rp_ ) :: f_t, phi_t, gts_t, s_j

!  if a re-entry occurs, branch to the appropriate place in the code

      IF ( status > 0 ) GO TO 200

!  see if shifts x_s have been provided

      IF ( ALLOCATED( X_s ) ) THEN
        data%shifts = SIZE( X_s ) >= n
      ELSE
        data%shifts = .FALSE.
      END IF

!  check to see if A has been provided

      IF ( PRESENT( Ao_ptr ) .AND. PRESENT( Ao_row ) .AND.                     &
           PRESENT( Ao_val ) ) THEN
        data%present_a = .TRUE. ; data%reverse_a = .FALSE.
      ELSE
        data%present_a = .FALSE.

!  check to see if access to products with A by reverse communication has
!  been provided

        IF ( PRESENT( reverse ) ) THEN
          data%reverse_a = .TRUE.

!  check to see if access to products with A by subroutine call has been
!  provided

        ELSE IF ( PRESENT( eval_APROD ) ) THEN
          data%reverse_a = .FALSE.

!  if none of thses option is available, exit

        ELSE
          status = - 2 ; RETURN
        END IF
      END IF

!  compute r^T r and f = 1/2 || r ||^2 + 1/2 weight || x ||^2

      data%rtr = DOT_PRODUCT( R, R )
      data%f_0 = half * data%rtr
      IF ( weight > zero ) THEN
        IF ( data%shifts ) THEN
          data%xtx = DOT_PRODUCT( X - X_s( : n ), X - X_s( : n ) )
        ELSE
          data%xtx = DOT_PRODUCT( X, X )
        END IF
        data%phi_0 = data%f_0 + half * weight * data%xtx
      ELSE
        data%phi_0 = data%f_0
      END IF

!  ensure that the path goes somewhere

      IF ( MAXVAL( ABS( D ) ) == zero ) THEN
        t_opt = zero ; f_opt = data%f_0 ; phi_opt = data%phi_0 ; status = - 1
        RETURN
      END IF

      IF ( summary ) THEN
        WRITE( out, 2000 ) prefix, prefix
        WRITE( out, 2010 ) prefix, 0, zero, zero, data%f_0
      END IF

!  main loop (mock do loop to allow reverse communication)

      data%step = 1 ; data%t = t_0 ; data%backwards = .TRUE.
  100 CONTINUE

!  store the projection P(x + t d) of x + t d onto Delta^n in s

        CALL SLLS_project_onto_simplices( n, m, n_c, S_ptr, S_ind,             &
                                          X + data%t * D, S, X_c, X_c_proj,    &
                                          status )
        IF ( weight > zero ) THEN
          IF ( data%shifts ) THEN
            data%xtx = DOT_PRODUCT( S - X_s( : n ), S - X_s( : n ) )
          ELSE
            data%xtx = DOT_PRODUCT( S, S )
          END IF
        END IF

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

        ELSE IF ( data%reverse_a ) THEN
!         reverse%p( : o ) = R
          reverse%v( : n ) = S
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
        IF ( data%reverse_a ) R_t = R + reverse%p( : o )
        IF ( debug ) WRITE( out, "( ' R_t =', / ( 5ES12.4 ) )" ) R

!  compute f_t = 1/2 || r_t ||^2 + 1/2 weight || x_t ||^2

        f_t = half * DOT_PRODUCT( R_t, R_t )
        IF ( weight > zero ) THEN
          phi_t = f_t + half * weight * data%xtx
        ELSE
          phi_t = f_t
        END IF

!  compute g^T s_t = r^T A s_t = r^T r_t - r^T r + weight * x^T s_t

        gts_t = DOT_PRODUCT( R_t, R ) - data%rtr
        IF ( weight > zero ) THEN
          IF ( data%shifts ) THEN
            gts_t = gts_t + weight * DOT_PRODUCT( X - X_s( : n ), S )
          ELSE
            gts_t = gts_t + weight * DOT_PRODUCT( X, S )
          END IF
        END IF
        IF ( debug ) WRITE( out, 2000 ) prefix, prefix
        IF ( summary ) WRITE( out, 2010 )                                      &
          prefix, data%step, data%t, TWO_NORM( S ), phi_t

!  test for sufficient decrease

        IF ( data%backwards ) THEN ! this is irrelevant at present

!  exit if the decrease is sufficient with the optimal x, f and status arrays

          IF ( phi_t <= data%phi_0 + eta * gts_t ) THEN
            X = X + S ; f_opt = f_t ; phi_opt = phi_t ; t_opt = data%t
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

 2000 FORMAT( /, A, '   inexact arc search ', /,                               &
                 A, '       k      t           s          phi' )
 2010 FORMAT( A, I8, 3ES12.4 )

!  End of subroutine SLLSM_inexact_arc_search

      END SUBROUTINE SLLSM_inexact_arc_search

! -*-*-*-*-*-*-*-*-  S L L S _ C G L S   S U B R O U T I N E  -*-*-*-*-*-*-*-*-

      SUBROUTINE SLLS_cgls( o, n, n_free, weight, out, summary, debug, prefix, &
                            f, X, R, y, FREE,                                  &
                            stop_cg_relative, stop_cg_absolute,                &
                            iter, maxit, data, userdata, status, alloc_status, &
                            bad_alloc, X_s, Ao_ptr, Ao_row, Ao_val,            &
                            eval_AFPROD, eval_PREC, DPREC, reverse,            &
                            preconditioned, B )

!  Find the minimizer of the constrained (regularized) least-squares
!  objective function

!    f(x) =  1/2 || A_o x - b ||_2^2 + 1/2 weight * ||x - x_s||_2^2

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
!  userdata (structure of type USERDATA_type) that may be used to pass
!          data to and from the optional eval_* subroutines
!  alloc_status  (INTEGER) status of the most recent array (de)allocation
!  bad_alloc (CHARACTER string of length 80) that provides information
!          following an unsuccesful call
!  y       (REAL) the Lagrange multiplier for the constraint
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
!              where the i-th component of v is stored in reverse%v(i)
!              for i = FREE( : n_free ), should be returned in reverse%p.
!              The argument reverse%eval_status should be set to
!              0 if the calculation succeeds, and to a nonzero value otherwise.
!            3 [Dense in, sparse out] The components of the product
!              p = A^T * v, where v is stored in reverse%v, should be
!              returned in reverse%p. Only components p_i with indices
!              i = FREE( : n_free ) need be assigned, the remainder will be
!              ignored. The argument reverse%eval_status should be set to 0
!              if the calculation succeeds, and to a nonzero value otherwise.
!            4 the product p = P^-1 v between the inverse of the preconditionr
!              P and the vector v, where v is stored in reverse%v, should be
!              returned in reverse%p. Only the components of v with indices
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
!  ALLOCATABLE ARGUMENTS

!  x_s      (REAL array of length n) the values of the (nonzeros) shifts
!            if allocated
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!            nonzeros of A, stored by consecutive columns
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!            of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!            Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!            with Ao_ptr(n+1) pointin to the storage location 1 beyond
!            the last entry in A
!  eval_AFPROD subroutine that performs products with A, see the argument
!            list for SLLS_solve
!  eval_PREC subroutine that performs the preconditioning operation p = P v
!            see the argument list for SLLS_solve
!  DPREC   (REAL array of length n) the values of a diagonal preconditioner
!            that aims to approximate A^T A
!  preconditioned (LOGICAL) prsent and set true is there a preconditioner
!  reverse (structure of type reverse_type) used to communicate
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
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n_free ) :: FREE
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: Y
      TYPE ( SLLS_subproblem_data_type ), INTENT( INOUT ) :: data
      TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( IN ), DIMENSION( : ) :: X_s
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      OPTIONAL :: eval_AFPROD, eval_PREC
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: DPREC
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( o ) :: B
      LOGICAL, OPTIONAL, INTENT( IN ) :: preconditioned
      TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse

!  interface blocks

      INTERFACE
        SUBROUTINE eval_AFPROD( status, userdata, transpose, V, P,             &
                                FREE, n_free )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
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
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
        END SUBROUTINE eval_PREC
      END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, H, X, R, alpha_max, feas_tol, max_segments, out, print_level, prefix

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

!  see if shifts x_s have been provided

      IF ( ALLOCATED( X_s ) ) THEN
        data%shifts = SIZE( X_s ) >= n
      ELSE
        data%shifts = .FALSE.
      END IF

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

      data%G = zero

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
        reverse%v( : o ) = R( : o )
        reverse%transpose = .TRUE.
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
          data%G( FREE( : n_free ) ) = reverse%p( FREE( : n_free ) )
        ELSE
          data%G( : n ) = reverse%p( : n )
        END IF
      END IF

!  include the gradient of the regularization term if present

      IF ( data%regularization ) THEN
        IF ( n_free < n ) THEN
          IF ( data%shifts ) THEN
            data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )            &
              + weight * ( X( FREE( : n_free ) ) - X_s( FREE( : n_free ) ) )
          ELSE
            data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )            &
              + weight * X( FREE( : n_free ) )
          END IF
        ELSE
          IF ( data%shifts ) THEN
            data%G( : n ) = data%G( : n ) + weight * ( X - X_s( : n ) )
          ELSE
            data%G( : n ) = data%G( : n ) + weight * X
          END IF
        END IF
        IF ( data%shifts ) THEN
          f = f + half * weight * TWO_NORM( X - X_s( : n ) ) ** 2
        ELSE
          f = f + half * weight * TWO_NORM( X ) ** 2
        END IF
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
          reverse%v( : n ) = one
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
          data%PE( : n ) = reverse%p( : n )
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
          reverse%v( : n ) = data%G( : n )
          data%branch = 80 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   80 CONTINUE
      IF ( data%preconditioned ) THEN
        IF ( data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PG( : n ) = reverse%p( : n )
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

!  save the Lagrange multipler estimate

      y = val

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
          reverse%v( : n ) = data%PG( : n )
          data%branch = 90 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   90 CONTINUE

!  form the re-projection

      IF ( data%preconditioned ) THEN
        IF ( data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PG( : n ) = reverse%p( : n )
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
      IF ( summary ) WRITE( out, "( A, '    cgls stopping tolerance =',        &
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
          reverse%v( : n ) = data%P( : n )
          reverse%transpose = .FALSE.
          data%branch = 110 ; status = 2
          RETURN
        END IF

!  return from reverse communication

  110   CONTINUE
        IF ( data%reverse_afprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%Q( : o ) = reverse%p( : o )
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
          reverse%p( : n ) = zero
          reverse%v( : o ) = R
          reverse%transpose = .TRUE.
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
            data%G( FREE( : n_free ) ) = reverse%p( FREE( : n_free ) )
          ELSE
            data%G( : n ) = reverse%p( : n )
          END IF
        END IF
!write(6,"( ' g ', /, ( 5ES12.4 ) )" ) data%G( : n )

!  include the gradient of the regularization term if present

        IF ( data%regularization ) THEN
          IF ( n_free < n ) THEN
            IF ( data%shifts ) THEN
              data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )          &
                + weight * ( X( FREE( : n_free ) ) - X_s( FREE( : n_free ) ) )
            ELSE
              data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )          &
                + weight * X( FREE( : n_free ) )
            END IF
          ELSE
            IF ( data%shifts ) THEN
              data%G( : n ) = data%G( : n ) + weight * ( X - X_s( : n ) )
            ELSE
              data%G( : n ) = data%G( : n ) + weight * X
            END IF
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
            reverse%v( : n ) = data%G( : n )
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
            data%PG( : n ) = reverse%p( : n )
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

!  save the Lagrange multipler estimate

        y = val

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
            reverse%v( : n ) = data%PG( : n )
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
            data%PG( : n ) = reverse%p( : n )
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

! -*-*-*-*-*-*-*-*-  S L L S M _ C G L S   S U B R O U T I N E  -*-*-*-*-*-*-*-

      SUBROUTINE SLLSM_cgls( o, n, m, n_free, COHORT, weight,                  &
                             out, summary, debug, prefix, f, X, R, Y, FREE,    &
                             stop_cg_relative, stop_cg_absolute, iter, maxit,  &
                             data, userdata, status, alloc_status,             &
                             bad_alloc, X_s, Ao_ptr, Ao_row, Ao_val,           &
                             eval_AFPROD, eval_PREC, DPREC, reverse,           &
                             preconditioned, B )

!  Find the minimizer of the constrained (regularized) least-squares
!  objective function

!    f(x) =  1/2 || A_o x - b ||_2^2 + 1/2 weight * ||x - x_s||_2^2

!  for which certain components of x are fixed at zero, and the remainder
!  sum to zero

!  ------------------------------- dummy arguments -----------------------
!
!  INPUT arguments that are not altered by the subroutine
!
!  o       (INTEGER) the number of rows of A = number of observations
!  n       (INTEGER) the number of columns of A = number of variables
!  m       (INTEGER) the number of cohorts
!  n_free  (INTEGER) the number of components of x that are not fixed at zero
!  COHORT  (INTEGER array of length n) states the cohort that each variable
!           belongs to, variable j lies in cohort COHORT(j), j = 1,...,n
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
!  userdata (structure of type USERDATA_type) that may be used to pass
!          data to and from the optional eval_* subroutines
!  alloc_status  (INTEGER) status of the most recent array (de)allocation
!  bad_alloc (CHARACTER string of length 80) that provides information
!          following an unsuccesful call
!  Y       (REAL array of length at least m) the minimizer. The values
!           of the Lagrange multipliers
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
!              where the i-th component of v is stored in reverse%v(i)
!              for i = FREE( : n_free ), should be returned in reverse%p.
!              The argument reverse%eval_status should be set to
!              0 if the calculation succeeds, and to a nonzero value otherwise.
!            3 [Dense in, sparse out] The components of the product
!              p = A^T * v, where v is stored in reverse%v, should be
!              returned in reverse%p. Only components p_i with indices
!              i = FREE( : n_free ) need be assigned, the remainder will be
!              ignored. The argument reverse%eval_status should be set to 0
!              if the calculation succeeds, and to a nonzero value otherwise.
!            4 the product p = P^-1 v between the inverse of the preconditionr
!              P and the vector v, where v is stored in reverse%v, 
!              should be returned in reverse%p. Only the components of v 
!              with indices i = FREE( : n_free ) are nonzero, and only the 
!              components of p with indices i = FREE( : n_free ) are needed. 
!              The argument reverse%eval_status should  be set to 0 if the 
!              calculation succeeds, and to a nonzero value otherwise.
!          < 0 an error exit
!
!  WORKSPACE
!
!  data (structure of type SLLS_subproblem_data_type)
!
!  ALLOCATABLE ARGUMENTS
!
!  X_s      (REAL array of length n) the values of the (nonzeros) shifts
!            if allocated
!
!  OPTIONAL ARGUMENTS
!
!  Ao_val   (REAL array of length Ao_ptr( n + 1 ) - 1) the values of the
!            nonzeros of A, stored by consecutive columns
!  Ao_row   (INTEGER array of length Ao_ptr( n + 1 ) - 1) the row indices
!            of the nonzeros of A, stored by consecutive columns
!  Ao_ptr   (INTEGER array of length n + 1) the starting positions in
!            Ao_val and Ao_row of the i-th column of A, for i = 1,...,n,
!            with Ao_ptr(n+1) pointin to the storage location 1 beyond
!            the last entry in A
!  eval_AFPROD subroutine that performs products with A, see the argument
!            list for SLLS_solve
!  eval_PREC subroutine that performs the preconditioning operation p = P v
!            see the argument list for SLLS_solve
!  DPREC   (REAL array of length n) the values of a diagonal preconditioner
!            that aims to approximate A^T A
!  preconditioned (LOGICAL) present and set true is there a preconditioner
!  reverse (structure of type reverse_type) used to communicate
!           reverse communication data to and from the subroutine

!  ------------------ end of dummy arguments --------------------------

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ):: o, n, m, n_free, maxit, out
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: iter, status
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: alloc_status
      LOGICAL, INTENT( IN ) :: summary, debug
      REAL ( KIND = rp_ ), INTENT( IN ) :: weight
      REAL ( KIND = rp_ ), INTENT( INOUT ) :: f
      REAL ( KIND = rp_ ), INTENT( IN ) :: stop_cg_relative, stop_cg_absolute
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n_free ) :: FREE
      INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: COHORT
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( o ) :: R
      TYPE ( SLLS_subproblem_data_type ), INTENT( INOUT ) :: data
      TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
      REAL ( KIND = rp_ ), ALLOCATABLE, INTENT( IN ), DIMENSION( : ) :: X_s

      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ),                          &
                                        DIMENSION( n + 1 ) :: Ao_ptr
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_row
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Ao_val
      OPTIONAL :: eval_AFPROD, eval_PREC
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: DPREC
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( o ) :: B
      LOGICAL, OPTIONAL, INTENT( IN ) :: preconditioned
      TYPE ( REVERSE_type ), OPTIONAL, INTENT( INOUT ) :: reverse

!  interface blocks

      INTERFACE
        SUBROUTINE eval_AFPROD( status, userdata, transpose, v, p,  &
                                FREE, n_free )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        LOGICAL, INTENT( IN ) :: transpose
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_free
        INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: FREE
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: v
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: p
        END SUBROUTINE eval_AFPROD
      END INTERFACE

      INTERFACE
        SUBROUTINE eval_PREC( status, userdata, v, p )
        USE GALAHAD_USERDATA_precision
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
        TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: v
        REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: p
        END SUBROUTINE eval_PREC
      END INTERFACE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      n, H, X, R, alpha_max, feas_tol, max_segments, out, print_level, prefix

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

!  see if shifts x_s have been provided

      IF ( ALLOCATED( X_s ) ) THEN
        data%shifts = SIZE( X_s ) >= n
      ELSE
        data%shifts = .FALSE.
      END IF

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

      array_name = 'slls_cgls: data%ETPEM'
      CALL SPACE_resize_array( m, data%ETPEM, status, alloc_status,            &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

      array_name = 'slls_cgls: data%Y'
      CALL SPACE_resize_array( m, data%Y, status, alloc_status,               &
             array_name = array_name, bad_alloc = bad_alloc, out = out )
      IF ( status /= GALAHAD_ok ) GO TO 900

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

      data%G = zero

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
        reverse%v( : o ) = R( : o )
        reverse%transpose = .TRUE.
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
          data%G( FREE( : n_free ) ) = reverse%p( FREE( : n_free ) )
        ELSE
          data%G( : n ) = reverse%p( : n )
        END IF
      END IF

!  include the gradient of the regularization term if present

      IF ( data%regularization ) THEN
        IF ( n_free < n ) THEN
          IF ( data%shifts ) THEN
            data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )            &
              + weight * ( X( FREE( : n_free ) ) - X_s( FREE( : n_free ) ) )
          ELSE
            data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )            &
              + weight * X( FREE( : n_free ) )
          END IF
        ELSE
          IF ( data%shifts ) THEN
            data%G( : n ) = data%G( : n ) + weight * ( X - X_s( : n ) )
          ELSE
            data%G( : n ) = data%G( : n ) + weight * X
          END IF
        END IF
        IF ( data%shifts ) THEN
          f = f + half * weight * TWO_NORM( X - X_s( : n ) ) ** 2
        ELSE
          f = f + half * weight * TWO_NORM( X ) ** 2
        END IF
      END IF

!  set preconditioned vector of ones P_j^{-1} e_Cj

!  a) evaluation via preconditioner-inverse-vector product call

      IF ( data%preconditioned ) THEN
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
          reverse%v( : n ) = one
          data%branch = 70 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   70 CONTINUE
      data%ETPEM( : m ) = zero
      IF ( data%preconditioned ) THEN
        IF ( data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PE( : n ) = reverse%p( : n )
        END IF

!  compute the products e_Cj^T e_Cj and e_Cj^T P_j^{-1} e_Cj for free variables

        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i <= 0 ) CYCLE
            data%ETPEM( i ) = data%ETPEM( i ) + data%PE( j )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i <= 0 ) CYCLE
            data%ETPEM( i ) = data%ETPEM( i ) + data%PE( j )
          END DO
        END IF
      ELSE
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i <= 0 ) CYCLE
            data%ETPEM( i ) = data%ETPEM( i ) + one
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i <= 0 ) CYCLE
            data%ETPEM( i ) = data%ETPEM( i ) + one
          END DO
        END IF
      END IF

!  set the initial preconditioned gradient

!  a) evaluation via preconditioner-inverse-vector product call

      IF ( data%preconditioned ) THEN
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
          reverse%v( : n ) = data%G( : n )
          data%branch = 80 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   80 CONTINUE
      IF ( data%preconditioned ) THEN
        IF ( data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PG( : n ) = reverse%p( : n )
        END IF

!  find its projection onto E s = 0 as
!  pg = P^{-1} g - P^{-1} E^T y, where y = ( E P^{-1} E^T )^{-1} E P^{-1} g

        data%Y( : m ) = zero
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
          END DO
        END IF 
        data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 )                                                       &
              data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 )                                                       &
              data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
          END DO         
        END IF

!  or set the initial unpreconditioned search direction (as above with P = I)

      ELSE
        data%Y( : m ) = zero
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%G( j )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%G( j )
          END DO
        END IF 
        data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 ) THEN
              data%PG( j ) = data%G( j ) - data%Y( i )
            ELSE
              data%PG( j ) = data%G( j )
            END IF
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 ) THEN
              data%PG( j ) = data%G( j ) - data%Y( i )
            ELSE
              data%PG( j ) = data%G( j )
            END IF
          END DO         
        END IF
      END IF

!  save the Lagrange multipler estimates

      Y( : m ) = data%Y( : m )

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
          reverse%v( : n ) = data%PG( : n )
          data%branch = 90 ; status = 4
          RETURN
        END IF
      END IF

!  return from reverse communication

   90 CONTINUE

!  form the re-projection on E s = 0

      IF ( data%preconditioned ) THEN
        IF (  data%reverse_prec ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%PG( : n ) = reverse%p( : n )
        END IF

        data%Y( : m ) = zero
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
          END DO
        END IF 
        data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 )                                                       &
              data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 )                                                       &
              data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
          END DO         
        END IF
      ELSE
        data%Y( : m ) = zero
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
          END DO
        END IF 
        data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
        IF ( n_free < n ) THEN
          DO l = 1, n_free
            j = FREE( l ) ; i = COHORT( j )
            IF ( i > 0 ) data%PG( j ) = data%PG( j ) - data%Y( i )
          END DO
        ELSE
          DO j = 1, n
            i = COHORT( j )
            IF ( i > 0 ) data%PG( j ) = data%PG( j ) - data%Y( i )
          END DO         
        END IF
      END IF

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
      IF ( summary ) WRITE( out, "( A, '    cgls stopping tolerance =',        &
     &   ES11.4, / )" ) prefix, data%stop_cg

!  print details of the intial point

      IF ( summary ) THEN
        WRITE( out, 2000 )
        WRITE( out, 2010 ) iter, f, norm_g
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
          reverse%v( : n ) = data%P( : n )
          reverse%transpose = .FALSE.
          data%branch = 110 ; status = 2
          RETURN
        END IF

!  return from reverse communication

  110   CONTINUE
        IF ( data%reverse_afprod ) THEN
          IF ( reverse%eval_status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF
          data%Q( : o ) = reverse%p( : o )
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

!  update the value of the objective function

        f = f - half * alpha * alpha * curv
        norm_r = SQRT( two * f )

!  compute the gradient g = A^T r at x and its norm

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
          CALL eval_AFPROD( status, userdata, transpose = .TRUE., V = R,       &
                            P = data%G, FREE = FREE, n_free = n_free )
          IF ( status /= GALAHAD_ok ) THEN
            status = GALAHAD_error_evaluation ; GO TO 900
          END IF

!  c) evaluation via reverse communication

        ELSE
          reverse%p( : n ) = zero
          reverse%v( : o ) = R
          reverse%transpose = .TRUE.
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
            data%G( FREE( : n_free ) ) = reverse%p( FREE( : n_free ) )
          ELSE
            data%G( : n ) = reverse%p( : n )
          END IF
        END IF

!  include the gradient of the regularization term if present

        IF ( data%regularization ) THEN
          IF ( n_free < n ) THEN
            IF ( data%shifts ) THEN
              data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )          &
                + weight * ( X( FREE( : n_free ) ) - X_s( FREE( : n_free ) ) )
            ELSE
              data%G( FREE( : n_free ) ) = data%G( FREE( : n_free ) )          &
                + weight * X( FREE( : n_free ) )
            END IF
          ELSE
            IF ( data%shifts ) THEN
              data%G( : n ) = data%G( : n ) + weight * ( X - X_s( : n ) )
            ELSE
              data%G( : n ) = data%G( : n ) + weight * X
            END IF
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
            reverse%v( : n ) = data%G( : n )
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
            data%PG( : n ) = reverse%p( : n )
          END IF

!  find the projected preconditioned gradient by projection onto E s = 0 as
!  pg = P^{-1} g - P^{-1} E^T y, where y = ( E P^{-1} E^T )^{-1} E P^{-1} g

          data%Y( : m ) = zero
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
            END DO
          END IF 
          data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 )                                                     &
                data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 )                                                     &
                data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
            END DO         
          END IF

!  or find the projected unpreconditioned gradient by projection onto E s = 0
!  (as above with P = I)

        ELSE
          data%Y( : m ) = zero
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%G( j )
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%G( j )
            END DO
          END IF 
          data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 ) THEN
                data%PG( j ) = data%G( j ) - data%Y( i )
              ELSE
                data%PG( j ) = data%G( j )
              END IF
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 ) THEN
                data%PG( j ) = data%G( j ) - data%Y( i )
              ELSE
                data%PG( j ) = data%G( j )
              END IF
            END DO         
          END IF
        END IF

!  save the Lagrange multipler estimates

        Y( : m ) = data%Y( : m )

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
            reverse%v( : n ) = data%PG( : n )
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
            data%PG( : n ) = reverse%p( : n )
          END IF

          data%Y( : m ) = zero
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
            END DO
          END IF 
          data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 )                                                     &
                data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 )                                                     &
                data%PG( j ) = data%PG( j ) - data%Y( i ) * data%PE( j )
            END DO         
          END IF
        ELSE
          data%Y( : m ) = zero
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 ) data%Y( i ) = data%Y( i ) + data%PG( j )
            END DO
          END IF 
          data%Y( : m ) = data%Y( : m ) / data%ETPEM( : m )
          IF ( n_free < n ) THEN
            DO l = 1, n_free
              j = FREE( l ) ; i = COHORT( j )
              IF ( i > 0 ) data%PG( j ) = data%PG( j ) - data%Y( i )
            END DO
          ELSE
            DO j = 1, n
              i = COHORT( j )
              IF ( i > 0 ) data%PG( j ) = data%PG( j ) - data%Y( i )
            END DO         
          END IF
        END IF

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

!  End of subroutine SLLSM_cgls

      END SUBROUTINE SLLSM_cgls

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!- G A L A H A D -  S L L S _ i m p o r t _ w i t h o u t _ a  S U B R O U TINE

     SUBROUTINE SLLS_import_without_a( control, data, status, n, o, m, COHORT )

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
!  m is a scalar variable of type default integer, that holds the
!   number of cohorts
!
!  COHORT is an optional rank-one array of type default integer and length n
!   that must be set so that its j-th component is a number, between 1 and m, 
!   of the cohort to which variable x_j belongs, or to 0 if the variable 
!   belong to no cohort. If m or COHORT is absent, all variables will be 
!   assumed to belong to a single cohort
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, m
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: COHORT

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

!  if there are multiple cohorts, record them

     IF ( PRESENT( COHORT ) ) THEN
       data%prob%m = m
       array_name = 'slls: data%prob%COHORT'
       CALL SPACE_resize_array( n, data%prob%COHORT,                           &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900
       IF ( data%f_indexing ) THEN
         data%prob%COHORT( : n ) = MAX( COHORT( : n ), 0 )
       ELSE
         data%prob%COHORT( : n ) = MAX( COHORT( : n ) + 1, 0 )
       END IF
     ELSE
       data%prob%m = 1
     END IF

!  record that the Jacobian is not explicitly available

     data%explicit_a = .FALSE.

!  allocate vector space if required

     array_name = 'slls: data%prob%B'
     CALL SPACE_resize_array( o, data%prob%B,                                  &
            data%slls_inform%status, data%slls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
     IF ( data%slls_inform%status /= 0 ) GO TO 900

     array_name = 'slls: data%prob%X'
     CALL SPACE_resize_array( n, data%prob%X,                                  &
            data%slls_inform%status, data%slls_inform%alloc_status,            &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
     IF ( data%slls_inform%status /= 0 ) GO TO 900

     array_name = 'slls: data%prob%Y'
     CALL SPACE_resize_array( data%prob%m, data%prob%Y,                        &
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

     array_name = 'slls: data%prob%R'
     CALL SPACE_resize_array( o, data%prob%R,                                  &
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

     array_name = 'slls: data%prob%X_status'
     CALL SPACE_resize_array( n, data%prob%X_status,                           &
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

     SUBROUTINE SLLS_import( control, data, status, n, o, m, Ao_type, Ao_ne,   &
                             Ao_row, Ao_col, Ao_ptr, COHORT )

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
!  m is a scalar variable of type default integer, that holds the number of
!   cohorts
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
!  COHORT is an optional rank-one array of type default integer and length n
!   that must be set so that its j-th component is a number, between 1 and m, 
!   of the cohort to which variable x_j belongs, or to 0 if the variable 
!   belong to no cohort. If m or COHORT is absent, all variables will be 
!   assumed to belong to a single cohort
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, o, m, Ao_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: Ao_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: Ao_ptr
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: COHORT

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug

!  assign space for vector data

     CALL SLLS_import_without_a( control, data, status, n, o, m,               &
                                 COHORT = COHORT )
     IF ( status /= GALAHAD_ready_to_solve ) GO TO 900

     error = data%slls_control%error
     space_critical = data%slls_control%space_critical
     deallocate_error_fatal = data%slls_control%space_critical

!  record that the Jacobian is explicitly available

     data%explicit_a = .TRUE.

!  set Ao appropriately in the qpt storage type

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
     data%slls_inform%status = 1
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
                                    regularization_weight, X, Y, Z, R, G,      &
                                    X_stat, W, X_s, eval_PREC )

!  solve the simplex-constrained linear least-squares problem whose structure
!  was previously imported. See SLLS_solve for a description of the required
!  arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type SLLS_full_data_type used for internal data
!
!  userdata is a scalar variable of type USERDATA_type which may be
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
!   the constraint Jacobian Ao in the storage scheme specified in slls_import.
!
!  B is a rank-one array of dimension o and type default
!   real, that holds the vector of observations, b.
!   The i-th component of B, i = 1, ... , o, contains (b)_i.
!
!  regularization_weight is an optional scalar of type default real that
!   holds the value of the non-negative regularization weight, sigma.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  Y is a rank-one array of dimension m and type default
!   real, that holds the vector of the Lagrange multipliers, y, on exit.
!   The i-th component of Y, j = 1, ... , m, contains (y)_i.
!
!  Z is a rank-one array of dimension n and type default
!   real, that holds the vector of the dual variables, z, on exit.
!   The j-th component of Z, j = 1, ... , n, contains (z)_j.
!
!  R is a rank-one array of dimension o and type default
!   real, that holds the vector of residuals, r = A x - b on exit.
!   The i-th component of R, i = 1, ... , o, contains (r)_i.
!
!  G is a rank-one array of dimension n and type default
!   real, that holds the gradient, g = A^T r on exit.
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
!  W is an OPTIONAL rank-one array of dimension o and type default real,
!   that holds the vector of diagonal weights, w. The j-th component  of W, 
!   j = 1, ... , n, contains (w)_j. If W is absent, weights of one will be used.
!
!  X_s is an OPTIONAL rank-one array of dimension n and type default real,
!   that holds the vector of shifts, x_s. The j-th component  of X_s, j = 1, 
!   ... , n, contains (x_s)_j. If W is absent, shifts of zero will be used.
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
     TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), INTENT( IN ) :: regularization_weight
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: Ao_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Y, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R, G
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( : ) :: X_stat
     REAL ( KIND = rp_ ), OPTIONAL, DIMENSION( : ), INTENT( IN ) :: W, X_s
     OPTIONAL :: eval_PREC

!  interface blocks

     INTERFACE
       SUBROUTINE eval_PREC( status, userdata, V, P )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
       END SUBROUTINE eval_PREC
     END INTERFACE

!  local variables

     INTEGER ( KIND = ip_ ) :: n, o, error
     CHARACTER ( LEN = 80 ) :: array_name
     LOGICAL :: deallocate_error_fatal, space_critical

     status = data%slls_inform%status

!  check that space for the constraint Jacobian has been provided

     IF ( .NOT. data%explicit_a ) GO TO 900

!  recover the dimensions

     n = data%prob%n ; o = data%prob%o

!  save the regularization weight

     data%prob%regularization_weight = regularization_weight

!  save the observations

     data%prob%B( : o ) = B( : o )

!  save the initial primal and dual variables

     data%prob%X( : n ) = X( : n )
!    data%prob%Z( : n ) = Z( : n )
     IF ( data%slls_control%cold_start == 0 )                                  &
       data%prob%X_status( : n ) = X_stat( : n )

!  save the Jacobian entries

     IF ( data%prob%Ao%ne > 0 )                                                &
       data%prob%Ao%val( : data%prob%Ao%ne ) = Ao_val( : data%prob%Ao%ne )

!  save the weights if they are present

     IF ( PRESENT( W ) ) THEN
       array_name = 'slls: data%prob%W'
       CALL SPACE_resize_array( o, data%prob%W,                                &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900
       data%prob%W( : o ) = W( : o )
     END IF

!  save the shifts if they are present

     IF ( PRESENT( X_s ) ) THEN
       array_name = 'slls: data%prob%X_s'
       CALL SPACE_resize_array( n, data%prob%X_s,                              &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
            bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900
       data%prob%X_s( : n ) = X_s( : n )
     END IF

!  call the solver

     data%slls_inform%status = status
     CALL SLLS_solve( data%prob, data%slls_data, data%slls_control,            &
                      data%slls_inform, data%userdata, eval_PREC = eval_PREC )
     status = data%slls_inform%status

!  recover the optimal primal and dual variables, and Lagrange multipliers

     X( : n ) = data%prob%X( : n )
     Y( : data%prob%m ) = data%prob%Y( : data%prob%m )
     Z( : n ) = data%prob%Z( : n )

!  recover the residual value and gradient

     R( : o ) = data%prob%R( : o )
     G( : n ) = data%prob%G( : n )

!  recover the status of x

     X_stat( : n ) = data%prob%X_status( : n )

     RETURN

!  error returns

 900 CONTINUE
     status = GALAHAD_error_h_not_permitted
     RETURN

!  End of subroutine SLLS_solve_given_a

     END SUBROUTINE SLLS_solve_given_a

!- G A L A H A D -  S L L S _ s o l v e _ r e v e r s e _ a _ p r o d SUBROUTINE

     SUBROUTINE SLLS_solve_reverse_a_prod( data, status, eval_status, B,       &
                                           regularization_weight,              &
                                           X, Y, Z, R, G, X_stat,              &
                                           V, P, IV, lvl, lvu, index,          &
                                           IP, lp, W, X_s )

!  solve the simplex-constrained linear least-squares problem whose structure
!  was previously imported, and for which the action of Ao and its traspose
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
!     2 The product Ao * v of the matrix Ao with a given vector v is required
!       from the user. The vector v will be provided in V and the
!       required product must be returned in P. SLLS_solve must then
!       be re-entered with eval_status set to 0, and any remaining
!       arguments unchanged. Should the user be unable to form the product,
!       this should be flagged by setting eval_status to a nonzero value
!
!     3 The product Ao^T * v of the transpose of the matrix Ao with a given
!       vector v is required from the user. The vector v will be provided in
!       V and the required product must be returned in P.
!       SLLS_solve must then be re-entered with eval_status set to 0,
!       and any remaining arguments unchanged. Should the user be unable to
!       form the product, this should be flagged by setting eval_status
!       to a nonzero value
!
!     4 The j-th column of Ao is required from the user, where index holds 
!       the value of j. The resulting NONZEROS and their correspinding 
!       row indices of the j-th column of Ao must be placed in 
!       P( 1 : lp ) and IP( 1 : lp ),
!       respectively, with lp set accordingly. SLLS_solve should
!       then be re-entered with all other arguments unchanged. Once again 
!       eval_status should be set to zero unless the column cannot 
!       be formed, in which case a nonzero value should be returned.
!
!     5 The product Ao * v of the matrix A with a given sparse vector v is
!       required from the user. Only components IV( lvl : lvu )
!       of the vector v stored in V are nonzero. The required product
!       should be returned in p. SLLS_solve must then be re-entered
!       with all other arguments unchanged. Typically v will be very sparse
!       (i.e., lvu-lvl will be small).
!       eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     6 Specified components of the product Ao^T * v of the transpose of the
!       matrix Ao with a given vector v stored in v are required from
!       the user. Only components indexed by IV( lvl : lvu ) of the product
!       should be computed, and these should be recorded in P( iv( lvl : lvu ) )
!       and SLLS_solve then re-entered with all other arguments unchanged.
!       eval_status should be set to zero unless the product cannot
!       be formed, in which case a nonzero value should be returned.
!
!     7 The product P^-1 * v involving the preconditioner P with a specified
!       vector v is required from the user. Here P should be a symmtric,
!       postive-definite approximation of Ao^T Ao. The vector v will be provided
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
!  regularization_weight is an optional scalar of type default real that
!   holds the value of the non-negative regularization weight, sigma.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  Y is a rank-one array of dimension m and type default
!   real, that holds the vector of the Lagrange multipliers, y, on exit.
!   The i-th component of Y, j = 1, ... , m, contains (y)_i.
!
!  Z is a rank-one array of dimension n and type default
!   real, that holds the vector of the dual variables, z, on exit.
!   The j-th component of Z, j = 1, ... , n, contains (z)_j.
!
!  R is a rank-one array of dimension o and type default
!   real, that holds the vector of residuals, r = A x - b on exit.
!   The i-th component of R, i = 1, ... , o, contains (r)_i.
!
!  G is a rank-one array of dimension n and type default
!   real, that holds the gradient, g = A^T r on exit.
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
!  W is an OPTIONAL rank-one array of dimension o and type default real,
!   that holds the vector of diagonal weights, w. The j-th component  of W, 
!   j = 1, ... , n, contains (w)_j. If W is absent, weights of one will be used.
!
!  X_s is an OPTIONAL rank-one array of dimension n and type default real,
!   that holds the vector of shifts, x_s. The j-th component  of X_s, j = 1, 
!   ... , n, contains (x_s)_j. If W is absent, shifts of zero will be used.
!
!  The remaining components V, ... , lp need not be set
!  on initial entry, but must be set as instructed by status as above.

     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: eval_status
     TYPE ( SLLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( IN ) :: regularization_weight
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
!    REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X, Y, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Y, Z
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: R, G
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( : ) :: X_stat
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: lp
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: lvl, lvu, index
     INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: IP
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : ) :: IV
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: P
     REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( : ) :: V
     REAL ( KIND = rp_ ), OPTIONAL, DIMENSION( : ), INTENT( IN ) :: W, X_s

!  local variables

     INTEGER ( KIND = ip_ ) :: n, o, error
     CHARACTER ( LEN = 80 ) :: array_name
     LOGICAL :: deallocate_error_fatal, space_critical

!  recover the dimensions

     n = data%prob%n ; o = data%prob%o

!  save the regularization weight

     data%prob%regularization_weight = regularization_weight

     SELECT CASE ( status )

!  initial entry

     CASE( 1 )

!  save the observations

       data%prob%B( : o ) = B( : o )

!  save the initial primal and dual variables

       data%prob%X( : n ) = X( : n )
!      data%prob%Z( : n ) = Z( : n )
       IF ( data%slls_control%cold_start == 0 )                                &
         data%prob%X_status( : n ) = X_stat( : n )

!  save the weights if they are present

       IF ( PRESENT( W ) ) THEN
         array_name = 'slls: data%prob%W'
         CALL SPACE_resize_array( o, data%prob%W,                              &
                data%slls_inform%status, data%slls_inform%alloc_status,        &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
         IF ( data%slls_inform%status /= 0 ) GO TO 900
         data%prob%W( : o ) = W( : o )
       END IF

!  save the shifts if they are present

       IF ( PRESENT( X_s ) ) THEN
         array_name = 'slls: data%prob%X_s'
         CALL SPACE_resize_array( n, data%prob%X_s,                            &
                data%slls_inform%status, data%slls_inform%alloc_status,        &
                array_name = array_name,                                       &
                deallocate_error_fatal = deallocate_error_fatal,               &
                exact_size = space_critical,                                   &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
         IF ( data%slls_inform%status /= 0 ) GO TO 900
         data%prob%X_s( : n ) = X_s( : n )
       END IF

!  allocate space for reverse-communication data

       error = data%slls_control%error
       space_critical = data%slls_control%space_critical
       deallocate_error_fatal = data%slls_control%space_critical

       array_name = 'slls: data%reverse%IP'
       CALL SPACE_resize_array( n, data%reverse%IP,                            &
              data%slls_inform%status, data%slls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%slls_inform%bad_alloc, out = error )
       IF ( data%slls_inform%status /= 0 ) GO TO 900

       array_name = 'slls: data%reverse%IV'
       CALL SPACE_resize_array( n, data%reverse%IV,                            &
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

       data%slls_inform%status = 1

!  save Jacobian-vector product information on re-entries

     CASE( 2, 5 )
       data%reverse%eval_status = eval_status
       data%reverse%p( : o ) = P( : o )
     CASE( 3, 7 )
       data%reverse%eval_status = eval_status
       data%reverse%p( : n ) = P( : n )
     CASE( 4 )
       data%reverse%eval_status = eval_status
       data%reverse%p( ip( 1 : lp ) ) = P( ip( 1 : lp ) )
       data%reverse%ip( 1 : lp ) = IP( 1 : lp )
       data%reverse%lp = lp
     CASE( 6 )
       data%reverse%eval_status = eval_status
       data%reverse%p( data%reverse%iv( lvl : lvu ) )                          &
         = P( data%reverse%iv( lvl : lvu ) )
     CASE DEFAULT
       data%slls_inform%status = GALAHAD_error_input_status
       GO TO 900
     END SELECT

!  call the solver

     CALL SLLS_solve( data%prob, data%slls_data, data%slls_control,            &
                      data%slls_inform, data%userdata, reverse = data%reverse )
     status = data%slls_inform%status

!  recover the optimal primal and dual variables, and Lagrange multipliers

     X( : n ) = data%prob%X( : n )
     Y( : data%prob%m ) = data%prob%Y( : data%prob%m )
     Z( : n ) = data%prob%Z( : n )

!  recover the residual value and gradient

     R( : o ) = data%prob%R( : o )
     G( : n ) = data%prob%G( : n )

!  recover the status of x

     X_stat( : n ) = data%prob%X_status( : n )

!  record Jacobian-vector product information for reverse communication

     SELECT CASE ( status )
     CASE( 2, 7 )
       V( : n ) = data%reverse%v( : n )
     CASE( 3 )
       V( : o ) = data%reverse%v( : o )
     CASE( 4 )
       index = data%reverse%index
     CASE( 5 )
       lvl = data%reverse%lvl ; lvu = data%reverse%lvu
       IV( lvl : lvu ) = data%reverse%iv( lvl : lvu )
       V( iv( lvl : lvu ) ) = data%reverse%v( iv( lvl : lvu ) )
     CASE( 6 )
       lvl = data%reverse%lvl ; lvu = data%reverse%lvu
       IV( lvl : lvu ) = data%reverse%iv( lvl : lvu )
       V( : o ) = data%reverse%v( : o )
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
