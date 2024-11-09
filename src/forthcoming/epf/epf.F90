! THIS VERSION: GALAHAD 5.1 - 2024-08-08 AT 09:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ E P F   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 5.1. May 9th 2024

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_EPF_precision

!     --------------------------------------------------------------------
!    |                                                                    |
!    | EPF, an exponential penalty function algorithm for                 |
!    |       nonlinearly-constrained optimization                         |
!    |                                                                    |
!    |   Aim: find a (local) minimizer of the objective f(x)              |
!    |        subject to x^l <= x <= x^u and c^l <= c(x) <= c^u           |
!    |                                                                    |
!    |   by minimizing the exponential penalty/multiplier function        |
!    |                                                                    |
!    |    phi(x,mu) = f(x) + mu sum_j=1^n [ z_j^l(x,mu) + z_j^u(x,mu) ] + |
!    |                     + mu sum_i=1^m [ w_i^l(x,mu) + w_i^u(x,mu) ]   |
!    |                                                                    |
!    |   where z_j^l(x,mu) = v_j^l e^((x_j^l-x_j)/mu)                     |
!    |         z_j^u(x,mu) = v_j^u e^((x_j-x_j^u)/mu)                     |
!    |         y_i^l(x,mu) = w_i^l e^((c_i^l-c_i(x))/mu)                  |
!    |   and   y_i^u(x,mu) = w_i^u e^((c_i(x)-c_i^u)/mu)                  |
!    |                                                                    |
!    |   for a suitable sequence of (mu,v^l,v^u,w^l,w^u) > 0              |
!    |                                                                    |
!     --------------------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM, INFINITY_NORM
     USE GALAHAD_STRING, ONLY: STRING_integer_6
     USE GALAHAD_SMT_precision
     USE GALAHAD_BSC_precision
     USE GALAHAD_MOP_precision, ONLY: MOP_Ax
     USE GALAHAD_TRU_precision
     USE GALAHAD_SSLS_precision
!    USE GALAHAD_PSLS_precision

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: EPF_initialize, EPF_read_specfile, EPF_solve,                   &
               EPF_terminate, NLPT_problem_type, GALAHAD_userdata_type,        &
               SMT_type, SMT_put

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: point1 = 0.1_rp_

     INTEGER ( KIND = ip_ ), PARAMETER :: iter_advanced_max = 5
     REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

!--------------------------------------
!   G l o b a l   P a r a m e t e r s
!--------------------------------------

     LOGICAL, PUBLIC, PARAMETER :: EPF_available = .TRUE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: EPF_control_type

!   error and warning diagnostics occur on stream error

       INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

       INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER ( KIND = ip_ ) :: print_level = 0

!   any printing will start on this iteration

       INTEGER ( KIND = ip_ ) :: start_print = - 1

!   any printing will stop on this iteration

       INTEGER ( KIND = ip_ ) :: stop_print = - 1

!   the number of iterations between printing

       INTEGER ( KIND = ip_ ) :: print_gap = 1

!   the maximum number of iterations permitted

       INTEGER ( KIND = ip_ ) :: max_it = 100

!   the maximum number of function evaluations permitted

       INTEGER ( KIND = ip_ ) :: max_eval = 10000

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER ( KIND = ip_ ) :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   update the Lagrange multipliers/dual variables from iteration
!    update_multipliers_itmin (<0 means never) and once the primal
!    infeasibility is below update_multipliers_tol

       INTEGER ( KIND = ip_ ) :: update_multipliers_itmin = 0
       REAL ( KIND = rp_ ) :: update_multipliers_tol = infinity

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = rp_ ) :: infinity = ten ** 19

!   the required absolute and relative accuracies for the primal infeasibility

        REAL ( KIND = rp_ ) :: stop_abs_p = epsmch
        REAL ( KIND = rp_ ) :: stop_rel_p = epsmch

!   the required absolute and relative accuracies for the dual infeasibility

        REAL ( KIND = rp_ ) :: stop_abs_d = epsmch
        REAL ( KIND = rp_ ) :: stop_rel_d = epsmch

!   the required absolute and relative accuracies for the complementarity

        REAL ( KIND = rp_ ) :: stop_abs_c = epsmch
        REAL ( KIND = rp_ ) :: stop_rel_c = epsmch

!   the smallest the step can be before termination

       REAL ( KIND = rp_ ) :: stop_s = epsmch

!   the initial value of the penalty parameter (non-positive sets automatically)

!      REAL ( KIND = rp_ ) :: initial_mu = point1
       REAL ( KIND = rp_ ) :: initial_mu = - one

!   the amount by which the penalty parameter is decreased

       REAL ( KIND = rp_ ) :: mu_reduce = half

!   the smallest value the objective function may take before the problem
!    is marked as unbounded

       REAL ( KIND = rp_ ) :: obj_unbounded = - epsmch ** ( - 2 )

!   perform an advanced start at the end of every iteration when the KKT
!   residuals are smaller than %advanced_start (-ve means never)

       REAL ( KIND = rp_ ) :: advanced_start = ten ** ( - 2 )

!  stop the advanced start search once the residuals sufficientl small

       REAL ( KIND = rp_ ) :: advanced_stop = ten ** ( - 8 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: clock_time_limit = - one

!   is the Hessian matrix of second derivatives available or is access only
!    via matrix-vector products?

       LOGICAL :: hessian_available = .TRUE.

!   use a direct (factorization) or (preconditioned) iterative method to
!    find the search direction

       LOGICAL :: subproblem_direct = .TRUE.

!   if %space_critical true, every effort will be made to use as little
!    space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for BSC

       TYPE ( BSC_control_type ) :: BSC_control

!  control parameters for TRU

       TYPE ( TRU_control_type ) :: TRU_control

!  control parameters for SSLS

       TYPE ( SSLS_control_type ) :: SSLS_control

     END TYPE EPF_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: EPF_time_type

!  the total CPU time spent in the package

       REAL :: total = 0.0

!  the CPU time spent preprocessing the problem

       REAL :: preprocess = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

       REAL :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

       REAL :: factorize = 0.0

!  the CPU time spent computing the search direction

       REAL :: solve = 0.0

!  the total clock time spent in the package

       REAL ( KIND = rp_ ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

       REAL ( KIND = rp_ ) :: clock_preprocess = 0.0

!  the clock time spent analysing the required matrices prior to factorization

       REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

       REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

       REAL ( KIND = rp_ ) :: clock_solve = 0.0

     END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: EPF_inform_type

!  return status. See EPF_solve for details

       INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER ( KIND = ip_ ) :: iter = 0

!  the total number of evaluations of the objection function

       INTEGER ( KIND = ip_ ) :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER ( KIND = ip_ ) :: g_eval = 0

!  the total number of evaluations of the Hessian of the objection function

       INTEGER ( KIND = ip_ ) :: h_eval = 0

!  the number of free variables

       INTEGER ( KIND = ip_ ) :: n_free = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by EPF_solve

       REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the norm of the primal infeasibility at the best estimate of the solution
!    determined by EPF_solve

       REAL ( KIND = rp_ ) :: primal_infeasibility = HUGE( one )

!  the norm of the dual infeasibility at the best estimate of the solution
!    determined by EPF_solve

       REAL ( KIND = rp_ ) :: dual_infeasibility = HUGE( one )

!  the norm of the complementary slackness at the best estimate of the solution
!    determined by EPF_solve

       REAL ( KIND = rp_ ) :: complementary_slackness = HUGE( one )

!  the current value of the penalty parameter

       REAL ( KIND = rp_ ) :: mu = zero

!  timings (see above)

       TYPE ( EPF_time_type ) :: time

!  inform parameters for BSC

       TYPE ( BSC_inform_type ) :: BSC_inform

!  inform parameters for TRU

       TYPE ( TRU_inform_type ) :: TRU_inform

!  inform parameters for SSLS

       TYPE ( SSLS_inform_type ) :: SSLS_inform

     END TYPE EPF_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: EPF_data_type
       INTEGER ( KIND = ip_ ) :: branch = 1
       INTEGER ( KIND = ip_ ) :: eval_status, out, start_print, stop_print
       INTEGER ( KIND = ip_ ) :: print_level, print_gap, jumpto, iter_advanced
       INTEGER ( KIND = ip_ ) :: h_ne, j_ne, jtdzj_ne, np1, npm
       INTEGER ( KIND = ip_ ) :: n_c_l, n_c_u, n_x_l, n_x_u, n_c
       INTEGER ( KIND = ip_ ) :: n_j_l, n_j_u, n_i_l, n_i_u, n_b

       REAL :: time_start, time_record, time_now
       REAL ( KIND = rp_ ) :: clock_start, clock_record, clock_now
       REAL ( KIND = rp_ ) :: mu, max_mu, stop_p, stop_d, stop_c, f_old
       REAL ( KIND = rp_ ) :: old_primal_infeasibility, rnorm, rnorm_old

       LOGICAL :: printi, printt, printm, printw, printd
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printm, set_printw, set_printd
       LOGICAL :: reverse_fc, reverse_gj, reverse_hl
       LOGICAL :: reverse_hprod, reverse_prec
       LOGICAL :: constrained, map_h_to_jtj, no_bounds, header, update_vw
       LOGICAL :: eval_fc, eval_gj, eval_hl, initialize_munu, update_munu

       CHARACTER ( LEN = 1 ) :: negcur, bndry, perturb, hard

!  workspace arrays

       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: V_status
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_map
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: Dz_map
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: B_rows
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Dy
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Dz
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V_l
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: V_u
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W_l
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: W_u
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y_l
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y_u
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z_l
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z_u
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: MU_l
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: MU_u
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: NU_l
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: NU_u

       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: R
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y_l_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y_u_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z_l_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z_u_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: J_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: H_old

!  local copy of control parameters

       TYPE ( EPF_control_type ) :: control

!  penalty-function problem data

       TYPE ( NLPT_problem_type ) :: epf

!  J(transpose) if required

       TYPE ( SMT_type ) :: JT

!  data for BSC

       TYPE ( BSC_data_type ) :: BSC_data

!  data for TRU

       TYPE ( TRU_data_type ) :: TRU_data

!  SSLS_data

       TYPE ( SMT_type ) :: B_ssls
       TYPE ( SMT_type ) :: C_ssls
       TYPE ( SSLS_data_type ) :: SSLS_data

     END TYPE EPF_data_type

   CONTAINS

!-*-*-  G A L A H A D -  E P F _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE EPF_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for EPF controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( EPF_data_type ), INTENT( INOUT ) :: data
     TYPE ( EPF_control_type ), INTENT( OUT ) :: control
     TYPE ( EPF_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initalize BSC components

     CALL BSC_initialize( data%BSC_data, control%BSC_control,                  &
                          inform%BSC_inform )
     control%BSC_control%prefix = '" - BSC:"                     '

!  initalize TRU components

     CALL TRU_initialize( data%TRU_data, control%TRU_control,                  &
                          inform%TRU_inform )
     control%TRU_control%prefix = '" - TRU:"                     '

!  initial private data. Set branch for initial entry

     data%branch = 1

     RETURN

!  End of subroutine EPF_initialize

     END SUBROUTINE EPF_initialize

!-*-*-*-*-   E P F _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE EPF_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by EPF_initialize could (roughly)
!  have been set as:

! BEGIN EPF SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  maximum-number-of-iterations                    100
!  maximum-number-of-evaluations                   10000
!  update-multipliers-from-iteration               0
!  infinity-value                                  1.0D+19
!  absolute-primal-accuracy                        1.0D-5
!  relative-primal-accuracy                        1.0D-5
!  absolute-dual-accuracy                          1.0D-5
!  relative-dual-accuracy                          1.0D-5
!  absolute-complementary-slackness-accuracy       1.0D-5
!  relative-complementary-slackness-accuracy       1.0D-5
!  minimum-step-allowed                            2.0D-16
!  initial-penalty-parameter                       1.0D-1
!  penalty-parameter-reduction-factor              0.5
!  update-multipliers-feasibility-tolerance        1.0D+20
!  minimum-objective-before-unbounded              -1.0D+32
!  advanced-start                                   1.0D-2
!  advanced-stop                                    1.0D-8
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  hessian-available                               yes
!  sub-problem-direct                              no
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END EPF SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( EPF_control_type ), INTENT( INOUT ) :: control
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: start_print = print_level + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_print = start_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_gap = stop_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_it = print_gap + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_eval = max_it + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: update_multipliers_itmin             &
                                            = max_eval + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_unit                           &
                                            = update_multipliers_itmin + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: infinity = alive_unit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_p = infinity + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_p = stop_abs_p + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_d = stop_rel_p + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_d = stop_abs_d + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_c = stop_rel_d + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_c = stop_abs_c + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_s = stop_rel_c + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: initial_mu = stop_s + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: mu_reduce = initial_mu + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: update_multipliers_tol               &
                                            = mu_reduce + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: obj_unbounded                        &
                                            = update_multipliers_tol + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: advanced_start                       &
                                            = obj_unbounded + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: advanced_stop = advanced_start + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit = advanced_stop + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: hessian_available                    &
                                            = clock_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: subproblem_direct                    &
                                            = hessian_available + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = subproblem_direct + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal               &
                                            = space_critical + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_file                           &
                                            = deallocate_error_fatal + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = alive_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'EPF '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( max_it )%keyword = 'maximum-number-of-iterations'
     spec( max_eval )%keyword = 'maximum-number-of-evaluations'
     spec( update_multipliers_itmin )%keyword                                  &
       = 'update-multipliers-from-iteration'
     spec( alive_unit )%keyword = 'alive-device'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
     spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
     spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
     spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
     spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
     spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
     spec( stop_s )%keyword = 'minimum-steplength-allowed'
     spec( initial_mu )%keyword = 'initial-penalty-parameter'
     spec( mu_reduce )%keyword = 'penalty-parameter-reduction-factor'
     spec( update_multipliers_tol )%keyword                                    &
       = 'update-multipliers-feasibility-tolerance'
     spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
     spec( advanced_start )%keyword = 'advanced-start'
     spec( advanced_stop )%keyword = 'advanced-stop'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

     spec( hessian_available )%keyword = 'hessian-available'
     spec( subproblem_direct )%keyword = 'sub-problem-direct'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( alive_file )%keyword = 'alive-filename'
     spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

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
     CALL SPECFILE_assign_value( spec( max_it ),                               &
                                 control%max_it,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_eval ),                             &
                                 control%max_eval,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( update_multipliers_itmin ),             &
                                 control%update_multipliers_itmin,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_p ),                           &
                                 control%stop_abs_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_p ),                           &
                                 control%stop_rel_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_d ),                           &
                                 control%stop_abs_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_d ),                           &
                                 control%stop_rel_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_c ),                           &
                                 control%stop_abs_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_c ),                           &
                                 control%stop_rel_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_s ),                               &
                                 control%stop_s,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_mu ),                           &
                                 control%initial_mu,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_reduce ),                            &
                                 control%mu_reduce,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( update_multipliers_tol ),               &
                                 control%update_multipliers_tol,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( obj_unbounded ),                        &
                                 control%obj_unbounded,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( advanced_start ),                       &
                                 control%advanced_start,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( advanced_stop ),                        &
                                 control%advanced_stop,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( hessian_available ),                    &
                                 control%hessian_available,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( subproblem_direct ),                    &
                                 control%subproblem_direct,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( alive_file ),                           &
                                 control%alive_file,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  read the controls for any auxiliary modules used

     IF ( PRESENT( alt_specname ) ) THEN
       CALL BSC_read_specfile( control%BSC_control, device,                    &
              alt_specname = TRIM( alt_specname ) // '-BSC' )
       CALL TRU_read_specfile( control%TRU_control, device,                    &
              alt_specname = TRIM( alt_specname ) // '-TRU' )
       CALL SSLS_read_specfile( control%SSLS_control, device,                  &
             alt_specname = TRIM( alt_specname ) // '-SSLS' )
     ELSE
       CALL BSC_read_specfile( control%BSC_control, device )
       CALL TRU_read_specfile( control%TRU_control, device )
       CALL SSLS_read_specfile( control%SSLS_control, device )
     END IF

     RETURN

!  End of subroutine EPF_read_specfile

     END SUBROUTINE EPF_read_specfile

!-*-*-*-  G A L A H A D -  E P F _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE EPF_solve( nlp, control, inform, data, userdata,               &
                           eval_FC, eval_GJ, eval_HL, eval_HLPROD, eval_PREC )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  EPF_solve, an exponential penalty function algorithm to find a local
!    minimizer of a given objective where the variables are required to
!    satisfy (nonlinear) constraints

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_EPF.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_EPF_precision
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  H is scalar variable of type SMT_TYPE that holds the Hessian matrix H. The
!   following components are used here:
!
!   H%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of H%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of H%type must
!    contain the string COORDINATE, for the sparse row-wise storage scheme, the
!    first fourteen components of H%type must contain the string SPARSE_BY_ROWS,
!    and for the diagonal storage scheme, the first eight components of H%type
!    must contain the string DIAGONAL.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into H%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Hessian we wish to
!    store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%H%type, 'COORDINATE' )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   H%ne is a scalar variable of type default integer, that holds the number of
!    entries in the  lower triangular part of H in the sparse co-ordinate
!    storage scheme. It need not be set for any of the other three schemes.
!
!   H%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the lower triangular part of the Hessian
!    matrix H in any of the available storage schemes.
!
!   H%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of the lower triangular part of H in the sparse
!    co-ordinate storage scheme. It need not be allocated for any of the other
!    three schemes.
!
!   H%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the lower triangular part of H in either
!    the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!    be allocated when the dense or diagonal storage schemes are used.
!
!   H%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of  each row of the lower
!    triangular part of H, as well as the total number of entries plus one,
!    in the sparse row-wise storage scheme. It need not be allocated when the
!    other schemes are used.
!
!  G is a rank-one allocatable array of dimension n and type default real,
!   that holds the gradient g of the objective function. The j-th component of
!   G, j = 1,  ... ,  n, contains g_j.
!
!  f is a scalar variable of type default real, that holds the value of
!   the objective function.
!
!  X is a rank-one allocatable array of dimension n and type default real, that
!   holds the values x of the optimization variables. The j-th component of
!   X, j = 1, ... , n, contains x_j.
!
!  X_l is a rank-one allocatable array of dimension n and type default real,
!   that holds the values x_l of the lower bounds on the optimization
!   variables x. The j-th component of X_l, j = 1, ... , n, contains (x_l)j.
!
!  X_u is a rank-one allocatable array of dimension n and type default real,
!   that holds the values x_u of the upper bounds on the optimization
!   variables x. The j-th component of X_u, j = 1, ... , n, contains (x_u)j.
!
!  pname is a scalar variable of type default character and length 10, which
!   contains the ``name'' of the problem for printing. The default ``empty''
!   string is provided.
!
!  VNAMES is a rank-one allocatable array of dimension n and type default
!   character and length 10, whose j-th entry contains the ``name'' of the j-th
!   variable for printing. This is only used  if ``debug''printing
!   control%print_level > 4) is requested, and will be ignored if the array is
!   not allocated.
!
! control is a scalar variable of type EPF_control_type. See EPF_control_type
!  in the module specification statements above for details
!
! inform is a scalar variable of type EPF_inform_type. See EPF_inform_type
!  in the module specification statements above for details.
!
!  On initial entry, inform%status should be set to 1. On exit, the following
!   inform%status will have been set as follows:
!
!     0. The run was succesful
!
!    -1. An allocation error occurred. A message indicating the offending
!        array is written on unit control%error, and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -2. A deallocation error occurred.  A message indicating the offending
!        array is written on unit control%error and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -3. The restriction nlp%n > 0 or requirement that prob%H_type contains
!        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
!          or 'DIAGONAL' has been violated.
!    -4. One or more of the simple bound restrictions (x_l)_i <= (x_u)_i
!        is violated.
!    -7. The objective function appears to be unbounded from below
!    -9. The analysis phase of the factorization failed; the return status
!        from the factorization package is given in the component
!        inform%factor_status
!   -10. The factorization failed; the return status from the factorization
!        package is given in the component inform%factor_status.
!   -11. The solution of a set of linear equations using factors from the
!        factorization package failed; the return status from the factorization
!        package is given in the component inform%factor_status.
!   -16. The problem is so ill-conditioned that further progress is impossible.
!   -18. Too many iterations have been performed. This may happen if
!        control%max_it is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -19. The CPU time limit has been reached. This may happen if
!        control%cpu_time_limit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -40. The user has forced termination of solver by removing the file named
!        control%alive_file from unit unit control%alive_unit.
!   -84. Too many evaluations have been performed. This may happen if
!        control%max_it is too small, but may also be symptomatic of
!        a badly scaled problem.
!
!     2. The user should compute the objective function value f(x) and the
!        constraint function values c(x) at the point x indicated in nlp%X
!        and then re-enter the subroutine. The required values should be set in
!        nlp%f and nlp%C respectively, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) and/or c(x)  - for instance, if
!        any of the functions is undefined at x - the user need not set nlp%f
!        or nlp%C, but should then set data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and the Jacobian of
!        the constraints nabla_x c(x) and then re-enter the subroutine. The
!        value of the i-th component of the gradient should be set in nlp%G(i),
!        for i = 1, ..., n, while the nonzeros of the Jacobian should be set
!        in nlp%J%val in the same order as in the storage scheme already
!        established in nlp%J,, and data%eval_status should be set to 0. If the
!        user is unable to evaluate any of the components of the gradient or
!        Jacobian - for instance if a component of the gradient or Jacobian
!        is undefined at x - the user need not set nlp%G or nlp%J%val, but
!        should then set data%eval_status to a non-zero value.
!     4. The user should compute the Hessian of the Lagrangian function
!        nabla_xx f(x) - sum_i=1^m y_i c_i(x) at the point x indicated in nlp%X
!        and y in nlp%Y and then re-enter the subroutine. The nonzeros of the
!        Hessian should be set in nlp%H%val in the same order as in the storage
!        scheme already established in nlp%H, and data%eval_status should be
!        set to 0. If the user is unable to evaluate a component of the Hessian
!        - for instance, if a component of the Hessian is undefined at x - the
!        user need not set nlp%H%val, but should then set data%eval_status to
!        a non-zero value.
!     5. The user should compute each of the gradient, Jacobian and Hessian
!        as described in 3 and 4 above, and then re-enter the subroutine
!        with data%eval_status set to 0. If the user is unable to evaluate
!        any of this data, nlp%G, nlp%J%val and nlp%H%val need not be set but
!        then data%eval_status should be set to a non-zero value.
!     6. The user should compute the product
!        ( nabla_xx f(x) - sum_i=1^m y_i c_i(x) ) v of the Hessian of the
!        Lagrangian function nabla_xx f(x) - sum_i=1^m y_i c_i(x) at the point
!        x indicated in nlp%X with the vector v, and add the result to the
!        vector u and then re-enter the subroutine. The vectors u and v are
!        given in data%U and data%V respectively, the resulting vector u +
!        nabla_xx f(x)v should be set in data%U and  data%eval_status should
!        be set to 0. If the user is unable to evaluate the product - for
!        instance, if a component of the Hessian is undefined at x - the user
!        need not alter data%U, but should then set data%eval_status to a
!        non-zero value.
!    [7. The user should compute the product u = P(x)v of their preconditioner
!        P(x) at the point x indicated in nlp%X with the vector v and then
!        re-enter the subroutine. The vectors v is given in data%V, the
!        resulting vector u = P(x)v should be set in data%U and
!        data%eval_status should be set to 0. If the user is unable to evaluate
!        the product - for instance, if a component of the preconditioner is
!        undefined at x - the user need not set data%U, but should then set
!        data%eval_status to a non-zero value. *** IGNORE - NOT IMPLEMENTED ***]
!
!  data is a scalar variable of type EPF_data_type used for internal data.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be
!   used to pass user data to and from the eval_* subroutines (see below)
!   Available components which may be allocated as required are:
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
!  eval_FC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If C is present, the values of the constraint functions
!   c(x) evaluated at x=X must be returned in C, and the status variable set
!   to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_FC is not present, EPF_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_GJ is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). If G is present, the components of
!   the gradient nabla_x f(x) of the objective function evaluated at x=X
!   must be returned in G. If GJ is present, the nonzeros of the Jacobian
!   nabla_x c(x) evaluated at x=X must be returned in J_val in the same
!   order as presented in nlp%J, and the status variable set to 0.
!   If the evaluation is impossible at x=X, status should be set to a
!   nonzero value. If eval_GJ is not present, EPF_solve will return to the
!   user with inform%status = 3 or 5 each time an evaluation is required.
!
!  eval_HL is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) - sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated
!   at x=X and y=Y must be returned in H_val in the same order as presented in
!   nlp%H, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_HL is not present,
!   EPF_solve will return to the user with inform%status = 4 or 5 each time
!   an evaluation is required.
!
!  eval_HLPROD is an optional subroutine which if present must have
!   the arguments given below (see the interface blocks). The sum
!   u + nabla_xx ( f(x) - sum_i=1^m y_i c_i(x) ) v of the product of the Hessian
!   nabla_xx f(x) + sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated
!   at x=X and y=Y with the vector v=V and the vector u=U must be returned in U,
!   and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_HPROD is not present,
!   EPF_solve will return to the user with inform%status = 6 each time an
!   evaluation is required.
!
!  eval_PREC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = P(x) v of the
!   user's preconditioner P(x) evaluated at x=X with the vector v=V, the result
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_PREC
!   is not present, EPF_solve will return to the user with inform%status = 7
!   each time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( EPF_control_type ), INTENT( IN ) :: control
     TYPE ( EPF_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( EPF_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_FC, eval_GJ, eval_HL, eval_HLPROD, eval_PREC

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_FC( status, X, userdata, F, C )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( kind = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( kind = rp_ ), OPTIONAL, INTENT( OUT ) :: F
       REAL ( kind = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FC

       SUBROUTINE eval_GJ( status, X, userdata, G, J_val )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
       REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_GJ

       SUBROUTINE eval_HL( status, X, Y, userdata, H_val, no_f )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) ::H_val
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL

       SUBROUTINE eval_HLPROD( status, X, Y, userdata, U, V, no_f, got_h )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HLPROD

       SUBROUTINE eval_PREC( status, X, userdata, U, V )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V, X
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_PREC
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, ii, ic, ir, j, k, l
     REAL ( KIND = rp_ ) :: penalty_term
     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) data%branch = 10

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 210 ) ! function and derivative evaluations
       GO TO 210
     CASE ( 420 ) ! Hessian evaluation
       GO TO 420
     CASE ( 430 ) ! objective and constraint evaluation
       GO TO 430
     CASE ( 440 ) ! gradient and Jacobian evaluation
       GO TO 440
     END SELECT

!  =================
!  0. Initialization
!  =================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 .OR. nlp%m < 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  is the problem constrained?

     data%constrained = nlp%m > 0

!  check that the simple bounds are consistent

     DO i = 1, nlp%n
       IF ( nlp%X_l( i ) > nlp%X_u( i ) ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 990
       END IF
     END DO

     IF ( data%constrained ) THEN
       DO i = 1, nlp%m
         IF ( nlp%C_l( i ) > nlp%C_u( i ) ) THEN
           inform%status = GALAHAD_error_bad_bounds
           GO TO 990
         END IF
       END DO
     END IF

!  set the initial penalty parameter

     data%mu = MAX( control%initial_mu, epsmch )

!  has the problem bounds?

     data%n_c_l = COUNT( nlp%C_l >= - control%infinity )
     data%n_c_u = COUNT( nlp%C_u <= control%infinity )
     data%n_x_l = COUNT( nlp%X_l >= - control%infinity )
     data%n_x_u = COUNT( nlp%X_u <= control%infinity )
     data%n_c = data%n_c_l + data%n_c_u + data%n_x_l + data%n_x_u
     data%no_bounds = data%n_c == 0
     data%np1 = nlp%n + 1 ; data%npm = nlp%n + data%n_c

!  allocate workspace for the problem:

     data%epf%n = nlp%n
     data%epf%pname = 'EPF       '

!  dual variable estimates for the simple-bound constraints

     array_name = 'EPF: data%Z_l'
     CALL SPACE_resize_array( nlp%n, data%Z_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%Z_u'
     CALL SPACE_resize_array( nlp%n, data%Z_u, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%NU_l'
     CALL SPACE_resize_array( nlp%n, data%NU_l, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%NU_u'
     CALL SPACE_resize_array( nlp%n, data%NU_u, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%V_l'
     CALL SPACE_resize_array( nlp%n, data%V_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%V_u'
     CALL SPACE_resize_array( nlp%n, data%V_u, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: nlp%Z'
     CALL SPACE_resize_array( nlp%n, nlp%Z, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%Dz'
     CALL SPACE_resize_array( nlp%n, data%Dz, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

!  Lagrange multiplier estimates for the general constraints

     array_name = 'EPF: data%Y_l'
     CALL SPACE_resize_array( nlp%m, data%Y_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%Y_u'
     CALL SPACE_resize_array( nlp%m, data%Y_u, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%MU_l'
     CALL SPACE_resize_array( nlp%m, data%MU_l, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%MU_u'
     CALL SPACE_resize_array( nlp%m, data%MU_u, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%W_l'
     CALL SPACE_resize_array( nlp%m, data%W_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%W_u'
     CALL SPACE_resize_array( nlp%m, data%W_u, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: nlp%Y'
     CALL SPACE_resize_array( nlp%m, nlp%Y, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%Dy'
     CALL SPACE_resize_array( nlp%m, data%Dy, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%epf%X'
     CALL SPACE_resize_array( nlp%n, data%epf%X, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: data%epf%G'
     CALL SPACE_resize_array( nlp%n, data%epf%G, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'EPF: nlp%gL'
     CALL SPACE_resize_array( nlp%n, nlp%gL, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 900

!  V_status( j ), j = 1, ..., n, will contain the status of the
!  j-th variable as the current iteration progresses. Possible values
!  are 0 if the variable lies away from its bounds, 1 and 2 if it lies
!  on its lower or upper bounds (respectively) - these may be problem
!  bounds or trust-region bounds, and 3 or 4 if the variable is fixed

     array_name = 'EPF: data%V_status'
     CALL SPACE_resize_array( nlp%n, data%V_status, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

!  choose starting values for the dual variables and Lagrange multipliers
!  (to do properly!)

     data%Z_l( : nlp%n ) = one ; data%Z_u( : nlp%n ) = one
     data%Y_l( : nlp%m ) = one ; data%Y_u( : nlp%m ) = one

!  ensure that the data is consistent

     data%control = control

!  decide how much reverse communication is required

!    IF ( data%constrained ) THEN
       data%reverse_fc = .NOT. PRESENT( eval_FC )
       data%reverse_gj = .NOT. PRESENT( eval_GJ )
       IF ( data%control%hessian_available ) THEN
         data%reverse_hl = .NOT. PRESENT( eval_HL )
       ELSE
         data%control%subproblem_direct = .FALSE.
         data%reverse_hl = .FALSE.
       END IF
       data%reverse_hprod = .NOT. PRESENT( eval_HLPROD )
!    END IF
     data%reverse_prec = .NOT. PRESENT( eval_PREC )
     data%map_h_to_jtj = data%control%hessian_available

!  control the output printing

     IF ( data%control%start_print < 0 ) THEN
       data%start_print = - 1
     ELSE
       data%start_print = data%control%start_print
     END IF

     IF ( data%control%stop_print < 0 ) THEN
       data%stop_print = data%control%max_it + 1
     ELSE
       data%stop_print = data%control%stop_print
     END IF

     IF ( control%print_gap < 2 ) THEN
       data%print_gap = 1
     ELSE
       data%print_gap = control%print_gap
     END IF

     data%out = data%control%out
     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printw with a few vectors

     data%set_printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
     END IF

!  create a file which the user may subsequently remove to cause
!  immediate termination of a run

     IF ( control%alive_unit > 0 ) THEN
      INQUIRE( FILE = control%alive_file, EXIST = alive )
      IF ( .NOT. alive ) THEN
         OPEN( control%alive_unit, FILE = control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'NEW' )
         REWIND control%alive_unit
         WRITE( control%alive_unit, "( ' GALAHAD rampages onwards ' )" )
         CLOSE( control%alive_unit )
       END IF
     END IF

!  record the number of nonzeos in the upper triangle of the Hessian

     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       data%h_ne = nlp%H%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%h_ne = nlp%H%PTR( nlp%n + 1 ) - 1
     CASE ( 'DENSE' )
       data%h_ne = nlp%H%n * ( nlp%n + 1 ) / 2
     CASE DEFAULT
       data%h_ne = nlp%n
     END SELECT

!  record the number of nonzeros in the constraint Jacobian

     SELECT CASE ( SMT_get( nlp%J%type ) )
     CASE ( 'COORDINATE' )
       data%J_ne = nlp%J%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%J_ne = nlp%J%ptr( nlp%m + 1 ) - 1
     CASE ( 'DENSE' )
       data%J_ne = nlp%m * nlp%n
     END SELECT

     nlp%H%m = nlp%n ; nlp%H%n = nlp%n ; nlp%H%ne = data%h_ne

!  set up data structure to record the assembled Hessian of the penalty
!  function, if required. Firstly, compute the number of nonzeros in J(x)

     IF ( data%control%subproblem_direct ) THEN
       nlp%J%n = nlp%n ; nlp%J%m = nlp%m
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         nlp%J%ne = nlp%J%m * nlp%J%n
       CASE ( 'SPARSE_BY_ROWS' )
         nlp%J%ne = nlp%J%ptr( nlp%m + 1 ) - 1
       END SELECT

!  provide J(transpose) = JT in coordinate form

       data%JT%n = nlp%m ; data%JT%m = nlp%n ; data%JT%ne = nlp%J%ne
       CALL SMT_put( data%JT%type, 'COORDINATE', inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; GO TO 980 ; END IF

       array_name = 'nls: data%JT%row'
       CALL SPACE_resize_array( data%JT%ne, data%JT%row, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%JT%col'
       CALL SPACE_resize_array( data%JT%ne, data%JT%col, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'nls: data%JT%val'
       CALL SPACE_resize_array( data%JT%ne, data%JT%val, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 980

!  assign the row and column indices of JT, leaving the values of JT ordered
!  as in J

       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, nlp%m
           DO j = 1, nlp%n
             l = l + 1
             data%JT%row( l ) = j ; data%JT%col( l ) = i
           END DO
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, nlp%m
           DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
             data%JT%row( l ) = nlp%J%col( l ) ; data%JT%col( l ) = i
           END DO
         END DO
       CASE ( 'COORDINATE' )
         DO l = 1, nlp%J%ne
           data%JT%row( l ) = nlp%J%col( l )
           data%JT%col( l ) = nlp%J%row( l )
         END DO
       END SELECT

!  record the sparsity pattern of J^T D_y J in data%epf%H

       data%control%BSC_control%new_a = 3
       data%control%BSC_control%extra_space_s = 0
       data%control%BSC_control%s_also_by_column = data%map_h_to_jtj
       CALL BSC_form( nlp%n, nlp%m, data%JT, data%epf%H, data%BSC_data,        &
                      data%control%BSC_control, inform%BSC_inform )
       data%control%BSC_control%new_a = 1
       data%jtdzj_ne = data%epf%H%ne

!   if required, find a mapping for the entries of H(x,y) and D_z into the
!   existing structure in data%epf%H for J^T D_y J, expanding the structure
!   as necessary

       IF ( data%map_h_to_jtj ) THEN
         array_name = 'epf: data%H_map'
         CALL SPACE_resize_array( data%h_ne, data%H_map, inform%status,        &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'epf: data%Dz_map'
         CALL SPACE_resize_array( nlp%n, data%Dz_map, inform%status,           &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = data%control%deallocate_error_fatal,  &
                exact_size = data%control%space_critical,                      &
                bad_alloc = inform%bad_alloc, out = data%control%error )
         IF ( inform%status /= 0 ) GO TO 980

         DO j = 1, nlp%n
           IF ( nlp%X_l( j ) >= - control%infinity .OR.                        &
                nlp%X_u( j ) <= control%infinity ) THEN
             data%Dz_map( j ) = 1
           ELSE
             data%Dz_map( j ) = 0
           END IF
         END DO
         CALL EPF_map_set( data%epf%H, nlp%H, data%Dz_map, data%H_map,         &
                           inform%status, inform%alloc_status )
         IF ( inform%status /= 0 ) GO TO 980
       END IF
     ELSE
write(6,*) ' No indirect subproblem option as yet'
stop
     END IF

!  ensure that the initial point is feasible

     nlp%X( : nlp%n ) = EPF_projection( nlp%n, nlp%X, nlp%X_l, nlp%X_u )

!  find the initial active set for x

     DO i = 1, nlp%n
       IF ( nlp%X_u( i ) * ( one - SIGN( epsmch, nlp%X_u( i ) ) ) <=           &
            nlp%X_l( i ) * ( one + SIGN( epsmch, nlp%X_l( i ) ) ) ) THEN
         data%V_status( i ) = 3
         nlp%X( i ) = half * ( nlp%X_l( i ) + nlp%X_u( i ) )
       ELSE IF ( nlp%X( i ) <=                                                 &
                 nlp%X_l( i ) * ( one + SIGN( epsmch, nlp%X_l( i ) ) ) ) THEN
         data%V_status( i ) = 1
       ELSE IF ( nlp%X( i ) >=                                                 &
                 nlp%X_u( i ) * ( one - SIGN( epsmch, nlp%X_u( i ) ) ) ) THEN
         data%V_status( i ) = 2
       ELSE
         data%V_status( i ) = 0
       END IF
     END DO

!  set initial estimates of the dual variables ...

     data%V_l = zero ; data%V_u = zero
     DO j = 1, nlp%n
       IF ( nlp%X_l( j ) >= - control%infinity ) data%V_l( j ) = one
       IF ( nlp%X_u( j ) <= control%infinity )  data%V_u( j ) = one
     END DO

!  ... and Lagrange multipliers

     data%W_l = zero ; data%W_u = zero
     DO i = 1, nlp%m
       IF ( nlp%C_l( i ) >= - control%infinity ) data%W_l( i ) = one
       IF ( nlp%C_u( i ) <= control%infinity ) data%W_u( i ) = one
     END DO

!  set space for matrices and vectors required for advanced start if necessary

     IF ( data%control%advanced_start > zero ) THEN
!      data%np1 = nlp%n + 1 ; data%npm = nlp%n + nlp%m

       array_name = 'colt: data%B_rows'
       CALL SPACE_resize_array( nlp%m, data%B_rows,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%R'
       CALL SPACE_resize_array( data%npm, data%R,                              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%X_old'
       CALL SPACE_resize_array( nlp%n, data%X_old,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%Y_l_old'
       CALL SPACE_resize_array( nlp%m, data%Y_l_old,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%Y_u_old'
       CALL SPACE_resize_array( nlp%m, data%Y_u_old,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%Z_l_old'
       CALL SPACE_resize_array( nlp%n, data%Z_l_old,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%Z_u_old'
       CALL SPACE_resize_array( nlp%n, data%Z_u_old,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%C_old'
       CALL SPACE_resize_array( nlp%m, data%C_old,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%G_old'
       CALL SPACE_resize_array( nlp%n, data%G_old,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%J_old'
       CALL SPACE_resize_array( data%J_ne, data%J_old,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%H_old'
       CALL SPACE_resize_array( data%H_ne, data%H_old,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

!  advanced starts will require access to the matrices
!
!   B = ( - J(x) ) (with rows from blocks 1 & 2 and 3 & 4 intertwined) and
!       (   J(x) )
!       ( -  I   )
!       (    I   )
!
!   C = diag( - mu Y_l^-1  - mu Y_u^-1  - mu Z_l^-1  - mu Z_u^-1 )

!  assign row numbers for B. For i = 1,...,m,
!    B_rows( i ) =  l row i of c occurs in row l of B (single sided bound) or
!                   in rows l and l+1 of B (double bounds)

       ii = 1
       DO i = 1, nlp%m
         data%B_rows( i ) = ii
         IF ( nlp%C_l( i ) >= - control%infinity ) ii = ii + 1
         IF ( nlp%C_u( i ) <= control%infinity ) ii = ii + 1
       END DO

!  discover how much space is needed for the B block

       data%n_j_l = 0 ; data%n_j_u = 0 ; data%n_i_l = 0 ; data%n_i_u = 0

       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, nlp%m
           IF ( nlp%C_l( i ) >= - control%infinity )                           &
             data%n_j_l = data%n_j_l + nlp%n
           IF ( nlp%C_u( i ) <= control%infinity )                             &
             data%n_j_u = data%n_j_u + nlp%n
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, nlp%m
           IF ( nlp%C_l( i ) >= - control%infinity )                           &
             data%n_j_l = data%n_j_l + nlp%J%ptr( i + 1 ) - nlp%J%ptr( i )
           IF ( nlp%C_u( i ) <= control%infinity )                             &
             data%n_j_u = data%n_j_u + nlp%J%ptr( i + 1 ) - nlp%J%ptr( i )
         END DO
       CASE ( 'COORDINATE' )
         DO l = 1, nlp%J%ne
           i = nlp%J%row( l )
           IF ( nlp%C_l( i ) >= - control%infinity )                           &
             data%n_j_l = data%n_j_l + 1
           IF ( nlp%C_u( i ) <= control%infinity )                             &
             data%n_j_u = data%n_j_u + 1
          END DO
       END SELECT

       data%n_b = data%n_j_l + data%n_j_u + data%n_x_l + data%n_x_u

!  set space for the B block

       data%B_ssls%n = data%n_c ; data%C_ssls%m = nlp%n
       data%B_ssls%ne = data%n_b
       CALL SMT_put( data%B_ssls%type, 'COORDINATE', inform%alloc_status )

       array_name = 'colt: data%B%row'
       CALL SPACE_resize_array( data%B_ssls%ne, data%B_ssls%row,               &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%B%col'
       CALL SPACE_resize_array( data%B_ssls%ne, data%B_ssls%col,               &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

       array_name = 'colt: data%B%val'
       CALL SPACE_resize_array( data%B_ssls%ne, data%B_ssls%val,               &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

!  set up the structure of the B block. First insert the J(x) blocks

       k = 1
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         DO i = 1, nlp%m
           ir = data%B_rows( i )
           DO j = 1, nlp%n
             ii = ir
             IF ( nlp%C_l( i ) >= - control%infinity ) THEN
               data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
               k = k + 1 ; ii = ii + 1
             END IF
             IF ( nlp%C_u( i ) <= control%infinity ) THEN
               data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
               k = k + 1
             END IF
           END DO
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, nlp%m
           ir = data%B_rows( i )
           DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
             j = nlp%J%col( l ) ; ii = ir
             IF ( nlp%C_l( i ) >= - control%infinity ) THEN
               data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
               k = k + 1 ; ii = ii + 1
             END IF
             IF ( nlp%C_u( i ) <= control%infinity ) THEN
               data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
               k = k + 1
             END IF
           END DO
         END DO
       CASE ( 'COORDINATE' )
         DO l = 1, nlp%J%ne
           i = nlp%J%row( l ) ; ii = data%B_rows( i ) ; j = nlp%J%col( l )
           IF ( nlp%C_l( i ) >= - control%infinity ) THEN
             data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
             k = k + 1 ; ii = ii + 1
           END IF
           IF ( nlp%C_u( i ) <= control%infinity ) THEN
             data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
             k = k + 1
           END IF
         END DO
       END SELECT

!  now insert the I blocks, including their (constant) values

       DO j = 1, nlp%n
         ir = ii
         IF ( nlp%X_l( j ) >= - control%infinity ) THEN
           data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
           data%B_ssls%val( k ) = - one
           k = k + 1 ; ii = ii + 1
         END IF
         IF ( nlp%X_u( j ) <= control%infinity ) THEN
           data%B_ssls%row( k ) = ii ; data%B_ssls%col( k ) = j
           data%B_ssls%val( k ) = one
           k = k + 1
         END IF
       END DO
write(6,"( ' B ' )" )
do l = 1, data%B_ssls%ne
write(6,"( ' row, col ', 2i7 )" ) data%B_ssls%row( l ), data%B_ssls%col( l )
end do

!  set space for the C block

       data%C_ssls%n = data%n_c ; data%C_ssls%m = data%n_c
       data%C_ssls%ne = data%n_c
       CALL SMT_put( data%C_ssls%type, 'DIAGONAL', inform%alloc_status )

       array_name = 'colt: data%C%val'
       CALL SPACE_resize_array( data%C_ssls%ne, data%C_ssls%val,               &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%control%error )
       IF ( inform%status /= 0 ) GO TO 900

!  set up and analyse the data structures for the matrix required for the
!  advanced start

       CALL SSLS_analyse( nlp%n, data%n_c, nlp%H, data%B_ssls, data%C_ssls,    &
                          data%SSLS_data, data%control%SSLS_control,           &
                          inform%SSLS_inform )
     END IF
!stop

!  ensure that the function values and derivatives are evaluated initially

     data%eval_fc = .TRUE. ; data%eval_gj = .TRUE. ; data%eval_hl = .TRUE.

!  initialize control parameters

    data%control%tru_control%hessian_available = .TRUE.
!   data%initialize_munu = .FALSE.
    data%initialize_munu = .TRUE.
    IF ( data%control%update_multipliers_itmin < 0 )                           &
      data%control%update_multipliers_itmin = data%control%max_it + 1

    data%epf%X( : nlp%n ) = nlp%X( : nlp%n )
    inform%iter = 0

!  compute stopping tolerances

!   data%stop_p = MAX( control%stop_abs_p,                                     &
!                      control%stop_rel_p * inform%primal_infeasibility )
!   data%stop_d = MAX( control%stop_abs_d,                                     &
!                      control%stop_rel_d * inform%dual_infeasibility )
!   data%stop_c = MAX( control%stop_abs_c,                                     &
!                      control%stop_rel_c * inform%complementary_slackness )

    data%stop_p = control%stop_abs_p
    data%stop_d = control%stop_abs_d
    data%stop_c = control%stop_abs_c

!   IF ( data%printi .AND. data%print_iteration_header )                       &
!      WRITE( data%out, 2010) prefix
!   IF ( data%print_iteration_header ) data%print_1st_header = .FALSE.

!  ---------------------------------------------------------
!  1. outer iteration (set and adjust penalty function) loop
!  ---------------------------------------------------------

 100 CONTINUE

!  control%print_level = 1

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = GALAHAD_error_alive ; GO TO 800
         END IF
       END IF

!  check to see if the iteration limit has been exceeded

       inform%iter = inform%iter + 1
       IF ( inform%iter > data%control%max_it ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 800
       END IF

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 )   &
           THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
!         data%control%GLTR_control%print_level = data%print_level_gltr
         data%control%TRU_control%print_level = control%TRU_control%print_level
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
!         data%control%GLTR_control%print_level = 0
         data%control%TRU_control%print_level = 0
       END IF
       data%print_iteration_header                                             &
         = data%print_level > 1 .OR. data%print_1st_header .OR.                &
           data%control%TRU_control%print_level > 0

!  -----------------------------------------------------------------
!  2. start the inner-iteration (minimize the penalty function) loop
!  -----------------------------------------------------------------

       inform%tru_inform%status = 1
       inform%tru_inform%iter = 0
       data%control%tru_control%maxit = MIN( control%tru_control%maxit,        &
         data%control%max_eval - inform%f_eval )
       IF (data%printi .AND.  data%control%TRU_control%print_level > 0 )       &
         WRITE( data%out, "( '' )" )
 200   CONTINUE

!  solve the problem using a trust-region method

         CALL TRU_solve( data%epf, data%control%tru_control,                   &
                         inform%tru_inform, data%tru_data, userdata )

!  reverse communication request for more information

         SELECT CASE ( inform%tru_inform%status )

!  evaluate the objective and constraint functions
!  -----------------------------------------------

         CASE ( 2 )
           IF ( data%eval_fc ) THEN
             IF ( data%reverse_fc ) THEN
               nlp%X( : nlp%n ) = data%epf%X( : nlp%n )
               data%branch = 210 ; inform%status = 2 ; RETURN
             ELSE
               CALL eval_FC( data%eval_status, data%epf%X( : nlp%n ),          &
                             userdata, nlp%f, nlp%C( : nlp%m ) )
             END IF
           END IF

!  evaluate the gradient and Jacobian values
!  .........................................

         CASE ( 3 )
           IF ( data%eval_gj ) THEN
             IF ( data%reverse_gj ) THEN
               nlp%X( : nlp%n ) = data%epf%X( : nlp%n )
               data%branch = 210 ; inform%status = 3 ; RETURN
             ELSE
               CALL eval_GJ( data%eval_status, data%epf%X( : nlp%n ),          &
                             userdata, nlp%G( : nlp%n ),                       &
                             nlp%J%val( 1 : nlp%J%ne ) )
             END IF
           END IF

!  obtain the values of Hessian of the Lagrangian
!  ..............................................

         CASE ( 4 )
           IF ( data%eval_hl .AND. data%control%subproblem_direct ) THEN
             IF ( data%reverse_hl ) THEN
               nlp%X( : nlp%n ) = data%epf%X( : nlp%n )
               data%branch = 210 ; inform%status = 4 ; RETURN
             ELSE
               CALL eval_HL( data%eval_status, data%epf%X( : nlp%n ),          &
                             - nlp%Y( : nlp%m ), userdata,                     &
                             nlp%H%val( : nlp%H%ne ) )
             END IF
           END IF

!  error exit from inner-iteration loop
!  ....................................

         CASE ( : - 1 )
           nlp%X( : nlp%n ) = data%epf%X( : nlp%n )
           GO TO 300

!  terminal exit from inner-iteration loop
!  .......................................

         CASE DEFAULT
           nlp%X( : nlp%n ) = data%epf%X( : nlp%n )

!  note that the function and first derivatives have been recorded at
!  the terminating point

           data%eval_fc = .FALSE. ; data%eval_gj = .FALSE.
           GO TO 300
         END SELECT

!  return from reverse communication with the function or derivative information

 210     CONTINUE
         SELECT CASE ( inform%tru_inform%status )

!  obtain the value of the penalty function
!  ........................................

         CASE ( 2 )

!  record the objective value

           IF ( data%eval_fc ) THEN
             inform%obj = nlp%f
             inform%f_eval = inform%f_eval + 1

!  compute the (infinity-) norm of the infeasibility

             inform%primal_infeasibility                                       &
               = MAX( EPF_infeasibility( nlp%n, data%qpf%X, nlp%X_l, nlp%X_u,  &
                                         control%infinity ),                   &
                      EPF_infeasibility( nlp%m, nlp%C, nlp%C_l, nlp%C_u,       &
                                         control%infinity ) )
           ELSE
             data%eval_fc = .TRUE.
           END IF
!          WRITE( 6, * ) ' f, ||c|| = ', inform%obj, inform%primal_infeasibility

!  initialize the penalty parameters at the start of the first major iteration

           IF ( data%initialize_munu ) THEN
             data%initialize_munu = .FALSE.

!  initialize the penalty parameters to equilibrate the constraints

             IF ( control%initial_mu <= zero ) THEN

!  for the simple bound constraints

               DO j = 1, nlp%n
                 IF ( nlp%X_l( j ) >= - control%infinity ) THEN
                   data%NU_l( j )                                              &
                     = MAX( one, ( nlp%X_l( j ) - data%epf%X( j ) ) )
                 ELSE
                   data%NU_l( j ) = zero
                 END IF
                 IF ( nlp%X_u( j ) <= control%infinity ) THEN
                   data%NU_u( j )                                              &
                     = MAX( one, ( nlp%X_u( j ) - data%epf%X( j ) ) )
                 ELSE
                   data%NU_u( j ) = zero
                 END IF
               END DO

!  for the general constraints

               DO i = 1, nlp%m
                 IF ( nlp%C_l( i ) >= - control%infinity ) THEN
                   data%MU_l( i ) = MAX( one, ABS( nlp%C_l( i ) - nlp%C( i ) ) )
                 ELSE
                   data%MU_l( i ) = zero
                 END IF
                 IF ( nlp%C_u( i ) <= control%infinity ) THEN
                   data%MU_u( i ) = MAX( one, ABS( nlp%C_u( i ) - nlp%C( i ) ) )
                 ELSE
                   data%MU_u( i ) = zero
                 END IF
               END DO

!  initialize the penalty parameters to the default value

             ELSE

!  for the simple bound constraints

               DO j = 1, nlp%n
                 IF ( nlp%X_l( j ) >= - control%infinity ) THEN
                   data%NU_l( j ) = control%initial_mu
                 ELSE
                   data%NU_l( j ) = zero
                 END IF
                 IF ( nlp%X_u( j ) <= control%infinity ) THEN
                   data%NU_u( j ) = control%initial_mu
                 ELSE
                   data%NU_u( j ) = zero
                 END IF
               END DO

!  for the general constraints

               DO i = 1, nlp%m
                 IF ( nlp%C_l( i ) >= - control%infinity ) THEN
                   data%MU_l( i ) = control%initial_mu
                 ELSE
                   data%MU_l( i ) = zero
                 END IF
                 IF ( nlp%C_u( i ) <= control%infinity ) THEN
                   data%MU_u( i ) = control%initial_mu
                 ELSE
                   data%MU_u( i ) = zero
                 END IF
               END DO
             END IF

!  record the largest value

             data%max_mu = MAX( MAXVAL( data%NU_l( : nlp%n ) ),                &
                                MAXVAL( data%NU_u( : nlp%n ) ) )
             IF ( nlp%m > 0 ) data%max_mu = MAX( data%max_mu,                  &
                                MAXVAL( data%MU_l( : nlp%m ) ),                &
                                MAXVAL( data%MU_u( : nlp%m ) ) )
           END IF

!  compute the individual and combined dual-variable

!    z_j^l(x,nu) = v_j^l e^((x_j^l-x_j)/nu^l_j)
!    z_j^u(x,uu) = v_j^u e^((x_j-x_j^u)/nu^u_j)
!    z_j(x,nu) = z_j^l(x,nu^l_j) - z_j^u(x,nu^u_j)
!    Dz_jj(x,nu) = z_j^l(x,nu^l_j)/nu^l_j + z_j^u(x,nu^u_j)/nu^u_j

!  & Lagrange-multiplier estimates,

!    y_i^l(x,mu) = w_i^l e^((c_i^l-c_i(x))/mu^l_i)
!    y_i^u(x,mu) = w_i^u e^((c_i(x)-c_i^u)/mu^u_i)
!    y_i(x,mu) = y_i^l(x,mu^l_i) - y_i^u(x,mu^u_i)
!    Dy_ii(x,mu) = y_i^l(x,mu^l_i)/mu^l_i + y_i^u(x,mu^u_i)/mu^u_i

!  and the total penalty term

!    sum_j=1^n [z_j^l(x,nu^l_j) + z_j^u(x,nu^u_j)] +
!    sum_i=1^m [y_i^l(x,mu^l_i) + y_i^u(x,mu^u_i)]

!  for the dual variables:

           penalty_term = zero
           DO j = 1, nlp%n
             IF ( nlp%X_l( j ) >= - control%infinity ) THEN
               data%Z_l( j ) = data%V_l( j ) *                                 &
                 EXP( ( nlp%X_l( j ) - data%epf%X( j ) ) / data%NU_l( j ) )
               nlp%Z( j ) = - data%Z_l( j )
               data%Dz( j ) = data%Z_l( j ) / data%NU_l( j )
               penalty_term = penalty_term + data%NU_l( j ) * data%Z_l( j )
             ELSE
               nlp%Z( j ) = zero
               data%Dz( j ) = zero
             END IF
             IF ( nlp%X_u( j ) <= control%infinity ) THEN
               data%Z_u( j ) = data%V_u( j ) *                                 &
                 EXP( ( data%epf%X( j ) - nlp%X_u( j ) ) / data%NU_u( j ) )
               nlp%Z( j ) = nlp%Z( j ) + data%Z_u( j )
               data%Dz( j ) = data%Dz( j ) + data%Z_u( j ) / data%NU_u( j )
               penalty_term = penalty_term + data%NU_u( j ) * data%Z_u( j )
             END IF
           END DO

!  for the Lagrange multipliers:

           DO i = 1, nlp%m
             IF ( nlp%C_l( i ) >= - control%infinity ) THEN
               data%Y_l( i ) = data%W_l( i ) *                                 &
                 EXP( ( nlp%C_l( i ) - nlp%C( i ) ) / data%MU_l( i ) )
               nlp%Y( i ) = - data%Y_l( i )
               data%Dy( i ) = data%Y_l( i ) / data%MU_l( i )
               penalty_term = penalty_term + data%MU_l( i ) * data%Y_l( i )
             ELSE
               nlp%Y( i ) = zero
               data%Dy( i ) = zero
             END IF
             IF ( nlp%C_u( i ) <= control%infinity ) THEN
               data%Y_u( i ) = data%W_u( i ) *                                 &
                 EXP( ( nlp%C( i ) - nlp%C_u( i ) ) / data%MU_u( i ) )
               nlp%Y( i ) = nlp%Y( i ) + data%Y_u( i )
               data%Dy( i ) = data%Dy( i ) + data%Y_u( i ) / data%MU_u( i )
               penalty_term = penalty_term + data%MU_u( i ) * data%Y_u( i )
             END IF
           END DO

!write(6,*) ' new x ',  nlp%X( : nlp%n )
!write(6,*) ' new y ',  nlp%Y( : nlp%m )
!write(6,*) ' new z ',  nlp%Z( : nlp%n )

!  compute the value of the penalty function

!    phi(x,mu) = f(x) + sum_j=1^n [ nu^y_i z_j^l(x,nu) + nu^u_i z_j^u(x,nu) ] +
!                     + sum_i=1^m [ mu^l_i w_i^l(x,mu) + mu^u_i w_i^u(x,mu) ]

           data%epf%f = nlp%f + penalty_term
           IF ( data%printd )                                                  &
             WRITE( data%out, "( ' penalty value =', ES22.14 )" ) data%epf%f

!  test to see if the penalty function appears to be unbounded from below

           IF ( data%epf%f < control%obj_unbounded ) THEN
             inform%status = GALAHAD_error_unbounded ; GO TO 990
           END IF

!  obtain the value of the gradient of the penalty function
!  ........................................................

         CASE ( 3 )
           IF ( data%eval_gj ) THEN
             inform%g_eval = inform%g_eval + 1
           ELSE
             data%eval_gj = .TRUE.
           END IF

!  compute the gradient of the penalty function:

!    nabla phi(x,mu,nu) = g(x) - z(x,nu) - J^T(x) y(x,mu)

           data%epf%G( : nlp%n ) = nlp%G( : nlp%n ) + nlp%Z( : nlp%n )
           CALL MOP_Ax( one, nlp%J, nlp%Y( : nlp%m ), one,                     &
                        data%epf%G( : nlp%n ), transpose = .TRUE.,             &
                        m_matrix = nlp%m, n_matrix = nlp%n )
           IF ( data%printd ) WRITE( data%out,                                 &
             "( ' penalty gradient =', /, ( 4ES20.12 ) )" ) data%epf%G( : nlp%n)

!  if required, print details of the current point

!          IF ( data%printd ) THEN
!            WRITE ( data%out, 2210 ) prefix
!            DO i = 1, nlp%n
!              WRITE( data%out, 2230 ) prefix, i,                              &
!                nlp%X_l( i ), data%epf%X( i ), nlp%X_u( i ), nlp%G( i )
!            END DO
!          END IF

!  obtain the value of the Hessian of the penalty function
!  .......................................................

         CASE ( 4 )

!  if required, form the Hessian of the penalty function
!
!    Hess phi = H(x,y(x,mu,nu)) + J^T(x) Dy(x,mu) J(x) + Dx(x,nu)
!
!  where H(x,y) is the Hessian of the Lagrangian

           IF ( data%control%subproblem_direct ) THEN
             IF ( data%eval_hl ) THEN
               inform%h_eval = inform%h_eval + 1
             ELSE
               data%eval_hl = .TRUE.
             END IF

!  start by recording the transpose of the Jacobian, J^T(x)

             data%JT%val( : data%JT%ne ) = nlp%J%val( : data%JT%ne )

!  insert the values of J(x)^T D(x,mu) J(x) into Hess_phi

             CALL BSC_form( nlp%n, nlp%m, data%JT, data%epf%H, data%BSC_data,  &
                            data%control%BSC_control, inform%BSC_inform,       &
                            D = data%Dy( : nlp%m ) )

             data%epf%H%val( data%jtdzj_ne + 1 : data%epf%H%ne ) = zero

!  append the values of H(x,y(x,mu)) if they are required

             DO l = 1, nlp%H%ne
               j = data%H_map( l )
               data%epf%H%val( j ) = data%epf%H%val( j ) + nlp%H%val( l )
             END DO

!  and those from Dz(x,mu)

             DO j = 1, nlp%n
               IF ( nlp%X_l( j ) >= - control%infinity .OR.                    &
                    nlp%X_u( j ) <= control%infinity ) THEN
                 i = data%Dz_map( j )
                 data%epf%H%val( i ) = data%epf%H%val( i ) + data%Dz( j )
               END IF
             END DO

!  debug printing for H

             IF ( data%printd ) THEN
!            IF ( data%printi ) THEN
               WRITE( data%out, "( A, ' penalty Hessian =' )" ) prefix
               WRITE( data%out, "( SS, ( A, : , 2( 2I7, ES20.12, : ) ) )" )    &
                 ( prefix, ( data%epf%H%row( l + j ), data%epf%H%col( l + j ), &
                    data%epf%H%val( l + j ), j = 0,                            &
                      MIN( 1, data%epf%H%ne - l ) ), l = 1, data%epf%H%ne, 2 )
             END IF
           END IF
         END SELECT

!  record successful evaluation

         data%tru_data%eval_status = 0
         GO TO 200

!  ---------------------------
!  end of inner-iteration loop
!  ---------------------------

 300   CONTINUE

!  compute the dual infeasibility

       inform%dual_infeasibility = TWO_NORM( data%epf%G( : nlp%n ) )

!      IF ( inform%iter > 2 ) WRITE(6, "( ' feas / old_feas =', ES12.4 ) ")    &
!        inform%primal_infeasibility /  data%old_primal_infeasibility
       data%old_primal_infeasibility = inform%primal_infeasibility

!  update the Lagrange multipliers and dual variables

       data%update_vw =                                                        &
         inform%primal_infeasibility <= data%control%update_multipliers_tol    &
         .AND. inform%iter > data%control%update_multipliers_itmin

!      data%update_vw = .TRUE.
!      data%update_vw = .FALSE.
       IF ( data%update_vw ) THEN

!  for the dual variables:

         DO j = 1, nlp%n
           IF ( nlp%X_l( j ) >= - control%infinity )                           &
             data%V_l( j ) = data%Z_l( j )
           IF ( nlp%X_u( j ) <= control%infinity )                             &
             data%V_u( j ) = data%Z_u( j )
         END DO

!  for the Lagrange multipliers:

         DO i = 1, nlp%m
           IF ( nlp%C_l( i ) >= - control%infinity )                           &
             data%W_l( i ) = data%Y_l( i )
           IF ( nlp%C_u( i ) <= control%infinity )                             &
             data%W_u( i ) = data%Y_u( i )
         END DO

!  if required, print the new Lagrange multiplier and dual variable estimates

         IF ( data%printd ) THEN
           WRITE( data%out, 2250 ) 'X  ', nlp%X
           WRITE( data%out, 2250 ) 'V_l', data%V_l
           WRITE( data%out, 2250 ) 'V_u', data%V_u
           IF ( nlp%m > 0 ) THEN
             WRITE( data%out, 2250 ) 'W_l', data%W_l
             WRITE( data%out, 2250 ) 'W_u', data%W_u
           END IF
         END IF
       END IF

!  compute the complemntary slackness

       inform%complementary_slackness                                          &
         = MAX( EPF_complementarity( nlp%n, nlp%X, nlp%X_l, nlp%X_u,           &
                                     data%Z_l, data%Z_u, control%infinity ),   &
                EPF_complementarity( nlp%m, nlp%C, nlp%C_l, nlp%C_u,           &
                                     data%Y_l, data%Y_u, control%infinity ) )

!  if required, print details of the latest major iteration

       IF ( data%printi ) THEN
         IF ( inform%iter == 1 )                                               &
           WRITE( data%out, "( /, A, '  Problem: ', A, ' (n = ', I0, ', m = ', &
          &  I0, '): EPF stopping tolerance =', ES11.4 )" )                    &
             prefix, TRIM( nlp%pname ), nlp%n, nlp%m, data%stop_d
         IF ( data%print_iteration_header ) THEN
           WRITE( data%out, 2010 ) prefix
           data%print_1st_header = .FALSE.
         END IF

         CALL CLOCK_time( data%clock_now )
         data%clock_now = data%clock_now - data%clock_start
         IF ( data%printi ) WRITE( data%out,                                   &
           "( A, I6, ES16.8, 4ES9.1, I6, F9.2 )" )                             &
             prefix, inform%iter, inform%obj, inform%primal_infeasibility,     &
             inform%dual_infeasibility, inform%complementary_slackness,        &
             data%max_mu, inform%tru_inform%iter, inform%tru_inform%status,    &
             data%clock_now

         IF ( data%printm ) WRITE( data%out,                                   &
           "( A, ' objective value      = ', ES22.14, /,                       &
          &   A, ' current gradient     = ', ES12.4, /,                        &
          &   A, ' primal infeasibility = ', ES12.4, /,                        &
          &   A, ' dual infeasibility   = ', ES12.4, /,                        &
          &   A, ' complementarity      = ', ES12.4 )" )                       &
            prefix, inform%obj, prefix, inform%dual_infeasibility,             &
            prefix, inform%primal_infeasibility,                               &
            prefix, inform%dual_infeasibility,                                 &
            prefix, inform%complementary_slackness
       END IF

       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            inform%dual_infeasibility <= data%stop_d .AND.                     &
            inform%complementary_slackness <= data%stop_c ) THEN
         GO TO 800
       END IF

!  update the penalty parameters

       data%update_munu = .TRUE.
       IF ( data%update_munu ) THEN

!  for the simple-bound constraints:

         DO j = 1, nlp%n
           IF ( nlp%X_l( j ) >= - control%infinity )                           &
             data%NU_l( j ) = control%mu_reduce * data%NU_l( j )
           IF ( nlp%X_u( j ) <= control%infinity )                             &
             data%NU_u( j ) = control%mu_reduce * data%NU_u( j )
         END DO

!  for the general constraints:

         DO i = 1, nlp%m
           IF ( nlp%C_l( i ) >= - control%infinity )                           &
             data%MU_l( i ) = control%mu_reduce * data%MU_l( i )
           IF ( nlp%C_u( i ) <= control%infinity )                             &
             data%MU_u( i ) = control%mu_reduce * data%MU_u( i )
         END DO

!  record the largest value

         data%max_mu = MAX( MAXVAL( data%NU_l( : nlp%n ) ),                    &
                            MAXVAL( data%NU_u( : nlp%n ) ) )
         IF ( nlp%m > 0 ) data%max_mu = MAX( data%max_mu,                      &
                            MAXVAL( data%MU_l( : nlp%m ) ),                    &
                            MAXVAL( data%MU_u( : nlp%m ) ) )
       END IF

       IF ( data%rnorm <= zero ) GO TO 500

!  To find an "advanced" starting point for the next major iteration, let

!   r(x,y_l,y_u,z_l,z_u) = ( g(x) - J^T (x) y_l + J^T (x) y_u - z_l + z_u )
!                          (      c_l-c(x) - mu ( log y_l - log w_l )     )
!                          (      c(x)-c_u - mu ( log y_u - log w_u )     )
!                          (        x_l-x  - mu ( log z_l - log v_l )     )
!                          (        x-x_u  - mu ( log z_u - log v_u )     )

!  Then apply Newton's method to find a solution to

!   ( H_L(x,y_u-y_l)  - J^T (x)   J^T (x)   - I        I     ) (  dx  )
!   (    - J(x)    - mu Y_l^-1                               ) ( dy_l )
!   (      J(x)              - mu Y_u^-1                     ) ( dy_u ) = - r,
!   (    -  I                          - mu Z_l^-1           ) ( dz_l )
!   (       I                                    - mu Z_u^-1 ) ( dz_l )

!  update (x,y_l,y_u,z_l,z_u) <- (x,y_l,y_u,z_l,z_u) + (dx,dy_l,dy_u,dz_l,dz_u)
!  and recur until ||r_+|| > ||r||

!  Note that, in the notation of GALAHAD's SSLS package,

!   A = H_L(x,y_u-y_l)
!   B = ( - J(x) )
!       (   J(x) )
!       ( -  I   )
!       (    I   )

!  and

!   C = diag( mu Y_l^-1  mu Y_u^-1  mu Z_l^-1  mu Z_u^-1 )

!  compute the gradient of the Lagrangian

       nlp%gL( : nlp%n ) = nlp%G( : nlp%n ) + nlp%Z( : nlp%n )
       CALL mop_AX( one, nlp%J, nlp%Y, one, nlp%gL, transpose = .TRUE.,        &
                    m_matrix = nlp%m, n_matrix = nlp%n )

!  compute minus the new residuals, - r(x,y,z)

       data%R( : nlp%n ) = - nlp%Gl( : nlp%n )
       k = nlp%n + 1
       DO i = 1, nlp%m
         IF ( nlp%C_l( i ) >= - control%infinity ) THEN
           data%R( k ) = nlp%C( i ) - nlp%C_l( i ) +                           &
             data%MU_l( i ) * ( LOG( data%Y_l( i ) ) - LOG( data%W_l( i ) ) )
!write(6,*) k,  data%R( k ), 'c_l'

write(6,"( 'c-c_l, mu(logy-logw)', 2ES12.4 )") nlp%C( i ) - nlp%C_l( i ), &
             data%MU_l( i ) * ( LOG( data%Y_l( i ) ) - LOG( data%W_l( i ) ) )

           k = k + 1
         END IF
         IF ( nlp%C_u( i ) <= control%infinity ) THEN
           data%R( k ) = nlp%C_u( i ) - nlp%C( i ) +                           &
             data%MU_u( i ) * ( LOG( data%Y_u( i ) ) - LOG( data%W_u( i ) ) )
!write(6,*) k,  data%R( k ), 'c_u'
           k = k + 1
         END IF
       END DO

       DO j = 1, nlp%n
         IF ( nlp%X_l( j ) >= - control%infinity ) THEN
           data%R( k ) = nlp%X( j ) - nlp%X_l( j ) +                           &
            data%NU_l( j ) * ( LOG( data%Z_l( j ) ) - LOG( data%V_l( j ) ) )
!write(6,*) k,  data%R( k ), 'x_l'
!write(6,*) nlp%X( j ), nlp%X_l( j ), &
!            data%NU_l( j ), LOG( data%Z_l( j ) ), LOG( data%V_l( j ) ), &
!            data%Z_l( j ), data%V_l( j )


write(6,"( 'x-x_l, mu(logz-logv)', 2ES12.4 )") nlp%X( j ) - nlp%X_l( j ), &
            data%NU_l( j ) * ( LOG( data%Z_l( j ) ) - LOG( data%V_l( j ) ) )
write(6,"( 'z_l, ve^(-x/mu), z_l/mu', 3ES12.4 )") data%Z_l( j ), &
            data%V_l( j ) * EXP( - (nlp%X( j ) - nlp%X_l( j ) )/data%NU_l(j)), &
            data%Z_l( j ) / data%NU_l( j )
           k = k + 1
         END IF
         IF ( nlp%X_u( j ) <= control%infinity ) THEN
           data%R( k ) = nlp%X_u( j )  - nlp%X( j ) +                          &
            data%NU_u( j ) * ( LOG( data%Z_u( j ) ) - LOG( data%V_u( j ) ) )
!write(6,*) k,  data%R( k ), 'x_u'
           k = k + 1
         END IF
       END DO
       data%rnorm = TWO_NORM( data%R( : data%npm ) )
write(6,"( ' ||r|| = ', ES11.4 )" ) data%rnorm, TWO_NORM( data%R( : nlp%n ) )

       IF ( data%rnorm > data%control%advanced_start ) GO TO 500

!  loop to search for an advanced starting point

       data%iter_advanced = 0
  410  CONTINUE
!write(6,*) ' iter = ', data%iter_advanced
!  obtain the Hessian of the Lagrangian H(x,y)

       IF ( data%reverse_hl ) THEN
         data%branch = 420 ; inform%status = 9 ; RETURN
       ELSE
         CALL eval_HL( data%eval_status, nlp%X, - nlp%Y, userdata, nlp%H%val )
         IF ( data%printd ) WRITE(6,"( ' nls%H', / ( 3( 2I6, ES12.4 )))" )     &
          ( nlp%H%row( i ), nlp%H%col( i ), nlp%H%val( i ), i = 1, nlp%H%ne )
       END IF

!  insert the latest values into B and C

  420  CONTINUE

!  first insert the values from J(x) into the C block

       k = 1
       SELECT CASE ( SMT_get( nlp%J%type ) )
       CASE ( 'DENSE' )
         l = 1
         DO i = 1, nlp%m
           DO j = 1, nlp%n
             IF ( nlp%C_l( i ) >= - control%infinity ) THEN
               data%B_ssls%val( k ) = - nlp%J%col( l )
               k = k + 1
             END IF
             IF ( nlp%C_u( i ) <= control%infinity ) THEN
               data%B_ssls%val( k ) = nlp%J%col( l )
               k = k + 1
             END IF
             l = l + 1
           END DO
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, nlp%m
           ir = data%B_rows( i )
           DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
             j = nlp%J%col( l ) ; ii = ir
             IF ( nlp%C_l( i ) >= - control%infinity ) THEN
               data%B_ssls%val( k ) = - nlp%J%col( l )
               k = k + 1
             END IF
             IF ( nlp%C_u( i ) <= control%infinity ) THEN
               data%B_ssls%val( k ) = nlp%J%col( l )
               k = k + 1
             END IF
           END DO
         END DO
       CASE ( 'COORDINATE' )
         DO l = 1, nlp%J%ne
           i = nlp%J%row( l )
           IF ( nlp%C_l( i ) >= - control%infinity ) THEN
             data%B_ssls%val( k ) = - nlp%J%col( l )
             k = k + 1
           END IF
           IF ( nlp%C_u( i ) <= control%infinity ) THEN
             data%B_ssls%val( k ) = nlp%J%col( l )
             k = k + 1
           END IF
         END DO
       END SELECT

!  now insert the values into the C block

       k = 1
       DO j = 1, nlp%m
         IF ( nlp%C_l( j ) >= - control%infinity ) THEN
           data%C_ssls%val( k ) = data%MU_l( j ) / data%Y_l( j )
           k = k + 1
         END IF
         IF ( nlp%C_u( j ) <= control%infinity ) THEN
           data%C_ssls%val( k ) = data%MU_u( j ) / data%Y_u( j )
           k = k + 1
         END IF
       END DO
       DO j = 1, nlp%n
         IF ( nlp%X_l( j ) >= - control%infinity ) THEN
           data%C_ssls%val( k ) = data%NU_l( j ) / data%Z_l( j )
           k = k + 1
         END IF
         IF ( nlp%X_u( j ) <= control%infinity ) THEN
           data%C_ssls%val( k ) = data%NU_u( j ) / data%Z_u( j )
           k = k + 1
         END IF
       END DO

!  form and factorize the matrix

!    ( H(x,y)  B^T(x) )
!    (  B(x)  - C(x) )

       CALL SSLS_factorize( nlp%n, data%n_c, nlp%H, data%B_ssls, data%C_ssls,  &
                            data%SSLS_data, data%control%SSLS_control,         &
                            inform%SSLS_inform )
!write(6,*) ' ssls status ', inform%SSLS_inform%status

!write(6,"( 'K = ')" )
!do i = 1, data%SSLS_data%K%ne
! write(6,"( 2I7, ES12.4 )" ) data%SSLS_data%K%row(i),data%SSLS_data%K%col(i), &
!                             data%SSLS_data%K%val(i)
!end do

!  find r = ( dx, dy )

write(6,*) ' r ', data%R
stop
write(6,*) ' ||r|| ', TWO_NORM( data%R )
       CALL SSLS_solve( nlp%n, data%n_c, data%R, data%SSLS_data,               &
                        data%control%SSLS_control,inform%SSLS_inform )
write(6,*) ' ||r|| ', TWO_NORM( data%R )
!write(6,*) ' ||v|| ', TWO_NORM( data%R( : nlp%n ) )
!write(6,*) ' ||g|| ', TWO_NORM( nlp%G( : nlp%n ) )

!  make a copy of x, y, z, c, g, J and H in case the advanced starting point
!  is poor

       data%f_old = nlp%f
       data%X_old( : nlp%n ) = nlp%X( : nlp%n )
       data%Y_l_old( : nlp%m ) = data%Y_l( : nlp%m )
       data%Y_u_old( : nlp%m ) = data%Y_u( : nlp%m )
       data%Z_l_old( : nlp%n ) = data%Z_l( : nlp%n )
       data%Z_u_old( : nlp%n ) = data%Z_u( : nlp%n )
       data%C_old( : nlp%m ) = nlp%C( : nlp%m )
       data%G_old( : nlp%n ) = nlp%G( : nlp%n )
       data%J_old( : data%J_ne ) = nlp%J%val( : data%J_ne )
       data%H_old( : data%H_ne ) = nlp%H%val( : data%H_ne )

!  compute the trial x and y

       nlp%X( : nlp%n ) = nlp%X( : nlp%n ) + data%R( : nlp%n )

       k = nlp%n + 1
       DO i = 1, nlp%m
         IF ( nlp%C_l( i ) >= - control%infinity ) THEN
           data%Y_l( i ) = data%Y_l( i ) + data%R( k )
           nlp%Y( i ) = - data%Y_l( i )
           k = k + 1
         ELSE
           nlp%Y( i ) = zero
         END IF
         IF ( nlp%C_u( i ) <= control%infinity ) THEN
           data%Y_u( i ) = data%Y_u( i ) + data%R( k )
           nlp%Y( i ) = nlp%Y( i ) + data%Y_u( i )
           k = k + 1
         END IF
       END DO

       DO j = 1, nlp%n
         IF ( nlp%X_l( j ) >= - control%infinity ) THEN
           data%Z_l( j ) = data%Z_l( j ) + data%R( k )
           nlp%Z( j ) = - data%Z_l( j )
           k = k + 1
         ELSE
           nlp%Z( j ) = zero
         END IF
         IF ( nlp%X_u( j ) <= control%infinity ) THEN
           data%Z_u( j ) = data%Z_u( j ) + data%R( k )
           nlp%Z( j ) = nlp%Z( j ) + data%Z_u( j )
           k = k + 1
         END IF
       END DO

!write(6,*) ' new x ',  nlp%X( : nlp%n )
!write(6,*) ' new y ',  nlp%Y( : nlp%m )
!write(6,*) ' new z ',  nlp%Z( : nlp%n )

!  compute the new objective and constraint values

       IF ( data%reverse_fc ) THEN
         data%branch = 430 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
       END IF

!  compute the new gradient and Jacobian values

  430  CONTINUE
       IF ( data%reverse_gj ) THEN
         data%branch = 440 ; inform%status = 4 ; RETURN
       ELSE
         CALL eval_GJ( data%eval_status, nlp%X, userdata,                     &
                       nlp%G, nlp%J%val )
       END IF

!  compute the gradient of the Lagrangian

  440  CONTINUE
       nlp%gL( : nlp%n ) = nlp%G( : nlp%n ) + nlp%Z( : nlp%n )
       CALL mop_AX( one, nlp%J, nlp%Y, one, nlp%gL, transpose = .TRUE.,        &
                    m_matrix = nlp%m, n_matrix = nlp%n )

!  compute minus the new residuals, - r(x,y,z)

       data%R( : nlp%n ) = - nlp%Gl( : nlp%n )
       k = nlp%n + 1
       DO i = 1, nlp%m
         IF ( nlp%C_l( i ) >= - control%infinity ) THEN
           data%R( k ) = nlp%C( i ) - nlp%C_l( i ) +                           &
             data%MU_l( i ) * ( LOG( data%Y_l( i ) ) - LOG( data%W_l( i ) ) )
           k = k + 1
         END IF
         IF ( nlp%C_u( i ) <= control%infinity ) THEN
           data%R( k ) = nlp%C_u( i ) - nlp%C( i ) +                           &
             data%MU_u( i ) * ( LOG( data%Y_u( i ) ) - LOG( data%W_u( i ) ) )
           k = k + 1
         END IF
       END DO

       DO j = 1, nlp%n
         IF ( nlp%X_l( j ) >= - control%infinity ) THEN
           data%R( k ) = nlp%X( j ) - nlp%X_l( j ) +                           &
            data%NU_l( j ) * ( LOG( data%Z_l( j ) ) - LOG( data%V_l( j ) ) )
           k = k + 1
         END IF
         IF ( nlp%X_u( j ) <= control%infinity ) THEN
           data%R( k ) = nlp%X_u( j )  - nlp%X( j ) +                          &
            data%NU_u( j ) * ( LOG( data%Z_u( j ) ) - LOG( data%V_u( j ) ) )
           k = k + 1
         END IF
       END DO

       data%rnorm_old = data%rnorm
       data%rnorm = TWO_NORM( data%R( : data%npm ) )
!write(6,*) ' ||r|| = ', data%rnorm

!  check to see if the residuals are sufficiently small

       IF ( data%rnorm < data%control%advanced_stop ) GO TO 500

!  check to see if the residuals have increased

       IF ( data%rnorm_old < data%rnorm ) THEN

!  make a copy of f, x, y, c, g & J in case the advanced starting point is poor

         nlp%f = data%f_old
         nlp%X( : nlp%n ) = data%X_old( : nlp%n )
         data%Y_l( : nlp%m ) = data%Y_l_old( : nlp%m )
         data%Y_u( : nlp%m ) = data%Y_u_old( : nlp%m )
         data%Z_l( : nlp%n ) = data%Z_l_old( : nlp%n )
         data%Z_u( : nlp%n ) = data%Z_u_old( : nlp%n )
         nlp%C( : nlp%m ) = data%C_old( : nlp%m )
         nlp%G( : nlp%n ) = data%G_old( : nlp%n )
         nlp%J%val( : data%J_ne ) = data%J_old( : data%J_ne )
         nlp%H%val( : data%H_ne ) = data%H_old( : data%H_ne )
         GO TO 500

!  check to see if the advanced start iteration limit has been reasched

       ELSE
          IF ( data%iter_advanced > iter_advanced_max ) GO TO 500
       END IF
       data%iter_advanced = data%iter_advanced + 1
       GO TO 410

!  end of advanced starting point search

  500  CONTINUE

!  ---------------------------
!  end of outer-iteration loop
!  ---------------------------

       inform%tru_inform%status = 1
       GO TO 100

 800 CONTINUE

!  compute the norm of the projected gradient

    inform%dual_infeasibility = TWO_NORM( data%epf%G( : nlp%n ) )

!  debug printing for X and G

     IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
       WRITE ( data%out, 2000 ) prefix, TRIM( nlp%pname ), nlp%n
       WRITE ( data%out, 2200 ) prefix, inform%f_eval, prefix, inform%g_eval,  &
         prefix, inform%h_eval, prefix, inform%iter, prefix,                   &
         prefix, inform%obj, prefix, inform%dual_infeasibility
       WRITE ( data%out, 2210 ) prefix
!      l = nlp%n
       l = 2
       DO j = 1, 2
          IF ( j == 1 ) THEN
             ir = 1 ; ic = MIN( l, nlp%n )
          ELSE
             IF ( ic < nlp%n - l ) WRITE( data%out, 2240 ) prefix
             ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
          END IF
          IF ( ALLOCATED( nlp%vnames ) ) THEN
            DO i = ir, ic
               WRITE( data%out, 2220 ) prefix, nlp%vnames( i ),                &
                 nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), data%epf%G( i )
            END DO
          ELSE
            DO i = ir, ic
               WRITE( data%out, 2230 ) prefix, i,                              &
                 nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), data%epf%G( i )
            END DO
          END IF
       END DO
     END IF

!  record the clock time

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     data%time_now = data%time_now - data%time_start
     data%clock_now = data%clock_now - data%clock_start
     IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,       &
    &    F12.2,  ' seconds' )" ) prefix, data%clock_now
     IF ( ( data%control%cpu_time_limit >= zero .AND.                          &
            data%time_now > data%control%cpu_time_limit ) .OR.                 &
          ( data%control%clock_time_limit >= zero .AND.                        &
            data%clock_now > data%control%clock_time_limit ) ) THEN
       inform%status = GALAHAD_error_cpu_limit ; GO TO 900
     END IF

!  =========================
!  End of the main iteration
!  =========================

 900 CONTINUE

!  print details of solution

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi ) THEN
!      WRITE ( data%out, 2000 ) nlp%pname, nlp%n
       WRITE ( data%out, 2200 ) prefix, inform%f_eval, prefix, inform%g_eval,  &
                                prefix, inform%h_eval, prefix, inform%iter,    &
                                prefix, inform%obj, prefix,                    &
                                inform%dual_infeasibility
!      WRITE ( data%out, 2210 )
!      IF ( data%print_level > 3 ) THEN
!         l = nlp%n
!      ELSE
!         l = 2
!      END IF
!      DO j = 1, 2
!         IF ( j == 1 ) THEN
!            ir = 1 ; ic = MIN( l, nlp%n )
!         ELSE
!            IF ( ic < nlp%n - l ) WRITE( data%out, 2240 )
!            ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
!         END IF
!         DO i = ir, ic
!            WRITE ( data%out, 2220 ) nlp%vnames( i ), nlp%X_l( i ),
!              nlp%X( i ), nlp%X_u( i ), data%epf%G( i )
!         END DO
!      END DO
!!$    IF ( .NOT. data%monotone ) WRITE( data%out,                             &
!!$        "( A, '  Non-monotone method used (history = ', I0, ')' )" ) prefix,&
!!$      data%non_monotone_history
    IF ( data%no_bounds .AND. data%control%subproblem_direct ) THEN
!!$      IF ( inform%TRS_inform%dense_factorization ) THEN
!!$        WRITE( data%out,                                                    &
!!$        "( A, '  Direct solution (eigen solver SYSV',                       &
!!$       &      ') of the trust-region sub-problem' )" ) prefix
!!$      ELSE
!!$        WRITE( data%out,                                                    &
!!$        "( A, '  Direct solution (solver ', A,                              &
!!$       &      ') of the trust-region sub-problem' )" )                      &
!!$           prefix, TRIM( data%control%TRS_control%definite_linear_solver )
!!$      END IF
!!$      WRITE( data%out, "( A, '  Number of factorization = ', I0,            &
!!$     &     ', factorization time = ', F0.2, ' seconds'  )" ) prefix,        &
!!$        inform%TRS_inform%factorizations,                                   &
!!$        inform%TRS_inform%time%clock_factorize
!!$      IF ( TRIM( data%control%TRS_control%definite_linear_solver ) ==       &
!!$           'pbtr' ) THEN
!!$        WRITE( data%out, "( A, '  Max entries in factors = ', I0,           &
!!$       & ', semi-bandwidth = ', I0  )" ) prefix, inform%max_entries_factors,&
!!$           inform%TRS_inform%SLS_inform%semi_bandwidth
!!$      ELSE
!!$        WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )       &
!!$          prefix, inform%max_entries_factors
!!$      END IF
!!$    ELSE
!!$      IF ( data%nprec > 0 )                                                 &
!!$        WRITE( data%out, "( A, '  Final Hessian semi-bandwidth (original,', &
!!$       &     ' re-ordered) = ', I0, ', ', I0 )" ) prefix,                   &
!!$          inform%PSLS_inform%semi_bandwidth,                                &
!!$          inform%PSLS_inform%reordered_semi_bandwidth
!!$      IF ( data%no_bounds ) THEN
!!$        SELECT CASE ( data%nprec )
!!$        CASE ( - 3 )
!!$          WRITE( data%out, "( A, '  User-defined norm used' )" )            &
!!$            prefix
!!$        CASE ( - 2 )
!!$          WRITE( data%out, "( A, 2X, I0, '-step Limited Memory ',           &
!!$         &  'norm used' )" ) prefix, data%lbfgs_mem
!!$        CASE ( - 1 )
!!$          WRITE( data%out, "( A, '  Two-norm used' )" ) prefix
!!$        CASE ( 1 )
!!$          WRITE( data%out, "( A, '  Diagonal norm used' )" ) prefix
!!$        CASE ( 2 )
!!$          WRITE( data%out, "( A, '  Band norm (semi-bandwidth ',            &
!!$         &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
!!$        CASE ( 3 )
!!$          WRITE( data%out, "( A, '  Re-ordered band norm (semi-bandwidth ', &
!!$         &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
!!$        CASE ( 4 )
!!$          WRITE( data%out, "( A, '  SE (solver ', A, ') full norm used' )" )&
!!$            prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
!!$        CASE ( 5 )
!!$          WRITE( data%out, "( A, '  GMPS (solver ', A, ') full norm used')")&
!!$            prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
!!$        CASE ( 6 )
!!$          WRITE( data%out, "(A,'  Lin-More''(', I0, ') incomplete Cholesky',&
!!$         &  ' factorization used ' )" ) prefix, data%control%icfs_vectors
!!$        END SELECT
!!$      ELSE
!!$        SELECT CASE ( data%nprec )
!!$        CASE ( - 3 )
!!$          WRITE( data%out, "( A, '  User-defined preconditioner used' )" )  &
!!$            prefix
!!$        CASE ( - 2 )
!!$          WRITE( data%out, "( A, 2X, I0, '-step Limited Memory ',           &
!!$         &  'preconditioner used' )" ) prefix, data%lbfgs_mem
!!$        CASE ( - 1 )
!!$          WRITE( data%out, "( A, '  No preconditioner used' )" ) prefix
!!$        CASE ( 1 )
!!$          WRITE( data%out, "( A, '  Diagonal preconditioner used' )" ) prefix
!!$        CASE ( 2 )
!!$          WRITE( data%out, "( A, '  Band preconditioner (semi-bandwidth ',  &
!!$         &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
!!$        CASE ( 3 )
!!$          WRITE( data%out, "( A, '  Re-ordered band preconditioner',        &
!!$         &   ' (semi-bandwidth ', I0, ') used' )" )                         &
!!$            prefix, inform%PSLS_inform%semi_bandwidth_used
!!$        CASE ( 4 )
!!$          WRITE( data%out, "( A, '  SE (solver ', A,                        &
!!$         &   ') full preconditioner used' )" )                              &
!!$            prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
!!$        CASE ( 5 )
!!$          WRITE( data%out, "( A, '  GMPS (solver ', A,                      &
!!$         &   ') full preconditioner used' )" )                              &
!!$            prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
!!$        CASE ( 6 )
!!$          WRITE( data%out, "(A,'  Lin-More''(', I0, ') incomplete Cholesky',&
!!$         &  ' factorization used ' )" ) prefix, data%control%icfs_vectors
!!$        END SELECT
!!$      END IF
!!$      IF ( data%control%renormalize_radius ) WRITE( data%out,               &
!!$         "( A, '  Radius renormalized' )" ) prefix
       END IF
       WRITE ( data%out, "( A, ' Total time = ', 0P, F0.2, ' seconds', / )" )  &
         prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'EPF_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' Problem: ', A, ' n = ', I8 )
 2010 FORMAT( A, '  iter  f               pr-feas  du-feas  cmp-slk  ',        &
              ' max mu inner stop cpu time' )
 2200 FORMAT( /, A, ' # function evaluations  = ', I0,                         &
              /, A, ' # gradient evaluations  = ', I0,                         &
              /, A, ' # Hessian evaluations   = ', I0,                         &
              /, A, ' # major iterations      = ', I0,                         &
             //, A, ' Terminal objective value =', ES22.14,                    &
              /, A, ' Terminal gradient norm   =', ES12.4 )
 2210 FORMAT( /, A, ' name             X_l        X         X_u         G ' )
 2220 FORMAT(  A, 1X, A10, 4ES12.4 )
 2230 FORMAT(  A, 1X, I10, 4ES12.4 )
 2240 FORMAT( A, ' .          ........... ...........' )
 2250 FORMAT( ' ', A3, ' =      ', 5ES12.4, / ( 6ES12.4 ) )

 !  End of subroutine EPF_solve

     END SUBROUTINE EPF_solve

!-*-*-  G A L A H A D -  E P F _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE EPF_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( EPF_data_type ), INTENT( INOUT ) :: data
     TYPE ( EPF_control_type ), INTENT( IN ) :: control
     TYPE ( EPF_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'EPF: data%V_status'
     CALL SPACE_dealloc_array( data%V_status,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%H_map'
     CALL SPACE_dealloc_array( data%H_map,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Dz_map'
     CALL SPACE_dealloc_array( data%Dz_map,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Dy'
     CALL SPACE_dealloc_array( data%Dy,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Dz'
     CALL SPACE_dealloc_array( data%Dz,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%V_l'
     CALL SPACE_dealloc_array( data%V_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%V_u'
     CALL SPACE_dealloc_array( data%V_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%W_l'
     CALL SPACE_dealloc_array( data%W_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%W_u'
     CALL SPACE_dealloc_array( data%W_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Y_l'
     CALL SPACE_dealloc_array( data%Y_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Y_u'
     CALL SPACE_dealloc_array( data%Y_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Z_l'
     CALL SPACE_dealloc_array( data%Z_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Z_u'
     CALL SPACE_dealloc_array( data%Z_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%MU_l'
     CALL SPACE_dealloc_array( data%MU_l,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%MU_u'
     CALL SPACE_dealloc_array( data%MU_u,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%NU_l'
     CALL SPACE_dealloc_array( data%NU_l,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%NU_u'
     CALL SPACE_dealloc_array( data%NU_u,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%JT%row'
     CALL SPACE_dealloc_array( data%JT%row,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%JT%col'
     CALL SPACE_dealloc_array( data%JT%col,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%JT%val'
     CALL SPACE_dealloc_array( data%JT%val,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%epf%G'
     CALL SPACE_dealloc_array( data%epf%G,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%epf%H%row'
     CALL SPACE_dealloc_array( data%epf%H%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%epf%H%col'
     CALL SPACE_dealloc_array( data%epf%H%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%epf%H%val'
     CALL SPACE_dealloc_array( data%epf%H%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%X_old'
     CALL SPACE_dealloc_array( data%X_old,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Y_l_old'
     CALL SPACE_dealloc_array( data%Y_l_old,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Y_u_old'
     CALL SPACE_dealloc_array( data%Y_u_old,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Z_l_old'
     CALL SPACE_dealloc_array( data%Z_l_old,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%Z_u_old'
     CALL SPACE_dealloc_array( data%Z_u_old,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%G_old'
     CALL SPACE_dealloc_array( data%G_old,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%C_old'
     CALL SPACE_dealloc_array( data%C_old,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%J_old'
     CALL SPACE_dealloc_array( data%J_old,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'EPF: data%H_old'
     CALL SPACE_dealloc_array( data%H_old,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Deallocate all arrays allocated within BSC

     CALL BSC_terminate( data%BSC_data, data%control%BSC_control,              &
                          inform%BSC_inform )
     inform%status = inform%BSC_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%BSC_inform%alloc_status
       inform%bad_alloc = inform%BSC_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Close and delete 'alive' file

     IF ( control%alive_unit > 0 ) THEN
       INQUIRE( FILE = control%alive_file, EXIST = alive )
       IF ( alive .AND. control%alive_unit > 0 ) THEN
         OPEN( control%alive_unit, FILE = control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'UNKNOWN' )
         REWIND control%alive_unit
         CLOSE( control%alive_unit, STATUS = 'DELETE' )
       END IF
     END IF

     RETURN

!  End of subroutine EPF_terminate

     END SUBROUTINE EPF_terminate

!-*-*-*-  G A L A H A D -  E P F _ p r o j e c t i o n   F U N C T I O N -*-*-

     FUNCTION EPF_projection( n, X, X_l, X_u )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the projection of x into the set x_l <= x <= x_u

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u
     REAL ( KIND = rp_ ), DIMENSION( n ) :: EPF_projection

!  compute the projection

     EPF_projection = MAX( X_l, MIN( X, X_u ) )
     RETURN

 !  End of function EPF_projection

     END FUNCTION EPF_projection

!-*-  G A L A H A D -  E P F _ i n f e a s i b i l i t y   F U N C T I O N -*-

     FUNCTION EPF_infeasibility( n, X, X_l, X_u, infinity )
     REAL ( KIND = rp_ ) :: EPF_infeasibility

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the infinity norm of the infeasiblity of x wrt the set
!  x_l <= x <= x_u, where any bound with absolute value > infinity is absent

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
     REAL ( KIND = rp_ ), INTENT( IN ) :: infinity
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: j

!  compute the infeasibility

     EPF_infeasibility = zero
     DO j = 1, n
       IF ( X_l( j ) >= - infinity )                                           &
         EPF_infeasibility = MAX( EPF_infeasibility, X_l( j ) - X( j ) )
       IF ( X_u( j ) <= infinity )                                             &
         EPF_infeasibility = MAX( EPF_infeasibility, X( j ) - X_u( j ) )
     END DO
     RETURN

 !  End of function EPF_infeasibility

     END FUNCTION EPF_infeasibility

!-  G A L A H A D -  E P F _ c o m p l e m e n t a r i t y   F U N C T I O N -

     FUNCTION EPF_complementarity( n, X, X_l, X_u, Z_l, Z_u, infinity )
     REAL ( KIND = rp_ ) :: EPF_complementarity

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the complementarity of (x,z) wrt the set x_l <= x <= x_u,
!  where any bound with absolute value > infinity is absent

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
     REAL ( KIND = rp_ ), INTENT( IN ) :: infinity
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: Z_l, Z_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: j

!  compute the infeasibility

     EPF_complementarity = zero
     DO j = 1, n
       IF ( X_l( j ) >= - infinity )                                           &
         EPF_complementarity = MAX( EPF_complementarity,                       &
                                    ( X( j ) - X_l( j ) ) * Z_l( j ) )
       IF ( X_u( j ) <= infinity )                                             &
         EPF_complementarity = MAX( EPF_complementarity,                       &
                                    - ( X( j ) - X_u( j ) ) * Z_u( j ) )
     END DO
     RETURN

 !  End of function EPF_complementarity

     END FUNCTION EPF_complementarity

!-*-*-  E P F _ r e d u c e d  _ g r a d i e n t _ n o r m  F U C T I O N  -*-

     FUNCTION EPF_reduced_gradient_norm( n, X, G, X_l, X_u )
     REAL ( KIND = rp_ ) :: EPF_reduced_gradient_norm

!  Compute the norm of the reduced gradient in the feasible box

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) ::  n
     REAL ( KIND = rp_ ), INTENT( IN  ), DIMENSION( n ) :: X, G, X_l, X_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i
     REAL ( KIND = rp_ ) :: gi, reduced_gradient_norm

     reduced_gradient_norm = zero
     DO i = 1, n
       gi = G( i )
       IF ( gi == zero ) CYCLE

!  Compute the projection of the gradient within the box

       IF ( gi < zero ) THEN
         gi = - MIN( ABS( X_u( i ) - X( i ) ), - gi )
       ELSE
         gi = MIN( ABS( X_l( i ) - X( i ) ), gi )
       END IF
       reduced_gradient_norm = MAX( reduced_gradient_norm, ABS( gi ) )
     END DO
     EPF_reduced_gradient_norm = reduced_gradient_norm

     RETURN

!  End of EPF_reduced_gradient_norm

     END FUNCTION EPF_reduced_gradient_norm

!-*-*-*-*-*-*-*-*-*-*-  E P F _ a c t i v e  F U C T I O N  -*-*-*-*-*-*-*-*-

     FUNCTION EPF_active( n, X, X_l, X_u )
     INTEGER ( KIND = ip_ ) :: EPF_active

!  Count the number of active bounds

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), INTENT( IN ) ::  n
     REAL ( KIND = rp_ ), INTENT( IN  ), DIMENSION( n ) :: X, X_l, X_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, n_active

     n_active = 0
     DO i = 1, n
       IF ( ABS( X( i ) - X_l( i ) ) <= epsmch ) THEN
         n_active = n_active + 1
       ELSE IF ( ABS( X( i ) - X_u( i ) ) <= epsmch ) THEN
         n_active = n_active + 1
       END IF
     END DO
     EPF_active = n_active

     RETURN

!  End of EPF_active

     END FUNCTION EPF_active

!-*-*-*-  G A L A H A D -  E P F _ m a p _ s e t   S U B R O U T I N E  -*-*-*-

     SUBROUTINE EPF_map_set( A, B, MAP_D, MAP_B, status, alloc_status )

!  find a mapping of the entries of the matrix B and the diagonal matrix D
!  into A. A should be stored by columns (either as a sparse or dense matrix)
!  while B can be in any supported GALAHAD format. The entries of MAP_D(i)
!  should be set to 0 (entry i is a zero) or 1 (it isn't a zero); on exit
!  the 1s will be replaced by the position in A, and A will have been expanded
!  to include entries in B and D that were missing; in that case A will no
!  longer be arranged by columns, but will still be in coordianate form

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SMT_type ), INTENT( INOUT ) :: A
     TYPE ( SMT_type ), INTENT( IN ) :: B
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( B%n ) :: MAP_D
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( B%ne ) :: MAP_B
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, j, l, ll, new_entries, a_ne, buffer
     LOGICAL :: diagonal
     INTEGER ( KIND = ip_ ), DIMENSION( B%m ) :: IW
     INTEGER ( KIND = ip_ ), DIMENSION( B%n + 1 ) :: PTR
     INTEGER ( KIND = ip_ ), DIMENSION( B%ne ) :: ROW, ORDER

!  count the numbers of entries in each column of B

     SELECT CASE ( SMT_get( B%type ) )
     CASE ( 'DENSE' )
       DO j = 1, B%n
         PTR( j + 1 ) = B%n + 1 - j
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       PTR( 2 : B%n + 1 ) = 0
       DO i = 1, B%n
         DO l = B%ptr( i ), B%ptr( i + 1 ) - 1
           j = B%col( l ) + 1
           PTR( j ) = PTR( j ) + 1
         END DO
       END DO
     CASE ( 'COORDINATE' )
       PTR( 2 : B%n + 1 ) = 0
       DO l = 1, B%ne
         j = B%col( l ) + 1
         PTR( j ) = PTR( j ) + 1
       END DO
     CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
       PTR( 2 : B%n + 1 ) = 1
     END SELECT

!  set the starting addresses for each column of B

     PTR( 1 ) = 1
     DO i = 2, B%n + 1
       PTR( i ) = PTR( i ) + PTR( i - 1 )
     END DO

!  compute the column ordering of B

     SELECT CASE ( SMT_get( B%type ) )
     CASE ( 'DENSE' )
       l = 0
       DO i = 1, B%n
         DO j = 1, i
           l = l + 1
           ll = PTR( j )
           ROW( ll ) = i
           ORDER( ll ) = l
           PTR( j ) = ll + 1
         END DO
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       DO i = 1, B%n
         DO l = B%ptr( i ), B%ptr( i + 1 ) - 1
           j = B%col( l )
           ll = PTR( j )
           ROW( ll ) = i
           ORDER( ll ) = l
           PTR( j ) = ll + 1
         END DO
       END DO
     CASE ( 'COORDINATE' )
       DO l = 1, B%ne
         j = B%col( l )
         ll = PTR( j )
         ROW( ll ) = B%row( l )
         ORDER( ll ) = l
         PTR( j ) = ll + 1
       END DO
     CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
       DO l = 1, B%n
         j = l
         ll = PTR( j )
         ROW( ll ) = j
         ORDER( ll ) = l
         PTR( j ) = ll + 1
       END DO
     END SELECT

!  reset the starting addresses for each column of B

     DO i = B%n, 1, - 1
       PTR( i + 1 ) = PTR( i )
     END DO
     PTR( 1 ) = 1

!  for each column in turn, find the position in A of each entry of B if it
!  exists; count any that aren't in A in preparation for a second pass.

     IW( : B%m ) = 0 ; new_entries = 0
     IF ( SMT_get( A%type ) == 'SPARSE_BY_COLUMNS' .OR.                        &
          SMT_get( A%type ) == 'COORDINATE' ) THEN
       DO j = 1, A%n

!  flag all entries in the j-th column of A by setting the component of IW > 0

         DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
           IW( A%row( l ) ) = l
         END DO

!  record whether the matrix D has an entry in this column

         diagonal = MAP_D( j ) == 1

!  march through the entries in the j-th column of B and D trying to match
!  them to A; record the number that do not occur in A

         DO l = PTR( j ), PTR( j + 1 ) - 1
           i =  ROW( l )
           IF ( IW( i ) > 0 ) THEN
             MAP_B( ORDER( l ) ) = IW( i )
             IF ( diagonal .AND. i == j ) THEN
               diagonal = .FALSE.
               MAP_D( j ) = IW( i )
             END IF
           ELSE
             IF ( diagonal .AND. i == j ) diagonal = .FALSE.
             new_entries = new_entries + 1
           END IF
         END DO
         IF ( diagonal ) new_entries = new_entries + 1

!  reset IW to 0 before processing the next column

         DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
           IW( A%row( l ) ) = 0
         END DO
       END DO
     ELSE ! dense A
       ll = 0
       DO j = 1, A%n
         MAP_D( j ) = ll + 1
         DO i = 1, A%m
           ll = ll + 1
           IW( i ) = ll
         END DO
         DO l = PTR( j ), PTR( j + 1 ) - 1
           MAP_B( ORDER( l ) ) = IW( ROW( l ) )
         END DO
         IW( : A%m ) = 0
       END DO
     END IF

!  if B has entries that do not occur in A, perform a second pass

     IF ( new_entries > 0 ) THEN

!  extend the list of row and column indices and values in A

       a_ne = A%ptr( A%n + 1 ) - 1 ; A%ne = a_ne + new_entries
       OPEN( newunit = buffer, STATUS = 'SCRATCH' )
       CALL SPACE_extend_array( A%row, a_ne, a_ne, A%ne, A%ne, buffer,         &
                                status, alloc_status )
       IF ( status /= GALAHAD_ok ) RETURN
       CALL SPACE_extend_array( A%col, a_ne, a_ne, A%ne, A%ne, buffer,         &
                                status, alloc_status )
       IF ( status /= GALAHAD_ok ) RETURN
       CALL SPACE_extend_array( A%val, a_ne, a_ne, A%ne, A%ne, buffer,         &
                                status, alloc_status )
       IF ( status /= GALAHAD_ok ) RETURN
       CLOSE( buffer )

!  insert the entries from B that weren't in A

       new_entries = 0
       DO j = 1, A%n

!  again flag all entries in the j-th column by setting the component of IW > 0
!  and record whether the matrix D has an entry in this column


         DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
           IW( A%row( l ) ) = l
         END DO
         diagonal = MAP_D( j ) == 1
!write(6,*) j, diagonal, ' MAP_D', MAP_D( j )

!  march through the entries in the j-th column of B and D processing those
!  that do not occur in A

         DO l = PTR( j ), PTR( j + 1 ) - 1
           i =  ROW( l )
           IF ( IW( i ) > 0 ) THEN
             IF ( diagonal .AND. i == j ) diagonal = .FALSE.
           ELSE
             new_entries = new_entries + 1
             a_ne = a_ne + 1
!write(6,*) ' a a_ne ', a_ne
             A%row( a_ne ) = ROW( l )
             A%col( a_ne ) = j
!write(6,*) ' row, col ', A%row( a_ne ), A%col( a_ne )
             MAP_B( ORDER( l ) ) = a_ne
!write(6,*) '  MAP_B(', ORDER( l ), ') = ', a_ne
             IF ( diagonal .AND. i == j ) THEN
               diagonal = .FALSE.
!write(6,*) '  MAP_D(', j, ') = ', a_ne
               MAP_D( j ) = a_ne
             END IF
           END IF
         END DO
         IF ( diagonal ) THEN
           new_entries = new_entries + 1
           a_ne = a_ne + 1
           A%row( a_ne ) = j
           A%col( a_ne ) = j
!write(6,*) ' row, col ', A%row( a_ne ), A%col( a_ne )
!write(6,*) '  MAP_D(', j, ') = ', a_ne
           MAP_D( j ) = a_ne
         END IF

!  reset IW to 0 before processing the next column

         DO l = A%ptr( j ), A%ptr( j + 1 ) - 1
           IW( A%row( l ) ) = 0
         END DO
       END DO
     END IF

     status = GALAHAD_ok
     RETURN

!  end of subroutine EPF_map_set

     END SUBROUTINE EPF_map_set

!  End of module GALAHAD_EPF

   END MODULE GALAHAD_EPF_precision
