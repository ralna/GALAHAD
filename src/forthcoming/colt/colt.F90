! THIS VERSION: GALAHAD 5.1 - 2024-09-11 AT 15:20 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D _ C O L T   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Jessica Farmer, Jaroslav Fowkes and Nick Gould, for GALAHAD productions

!  History -
!   initial version, GALAHAD Version 4.2, October 13th 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_COLT_precision

!     ----------------------------------------------------------
!    |                                                          |
!    | COLT: Constrained Optimization via Least-squares Targets |
!    |                                                          |
!    | Aim: to find a (local) minimizer of the nonlinear        |
!    | programming problem                                      |
!    |                                                          |
!    |  minimize               f (x)                            |
!    |  subject to          a_i^T x   = b_i      i in E_l       |
!    |             b_i^l <= a_i^T x  <= b_i^u    i in I_l       |
!    |                       c_i (x)  =  0       i in E_g       |
!    |             c_i^l <=  c_i (x) <= c_i^u    i in I_g       |
!    |  and          x^l <=       x  <= x^u                     |
!    |                                                          |
!     ----------------------------------------------------------

     USE GALAHAD_KINDS_precision
!$   USE omp_lib
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_STRING
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_SMT_precision
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_OPT_precision
     USE GALAHAD_NLS_precision
     USE GALAHAD_SSLS_precision
     USE GALAHAD_MOP_precision
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_ROOTS_precision, ONLY: ROOTS_quadratic

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: COLT_initialize, COLT_read_specfile, COLT_solve, COLT_track,    &
               COLT_terminate, NLPT_problem_type, GALAHAD_userdata_type,       &
               SMT_type, SMT_put

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one ) / two
     REAL ( KIND = rp_ ), PARAMETER :: root_tol = epsmch ** 0.75
!    LOGICAL, PARAMETER :: print_debug = .TRUE.
     LOGICAL, PARAMETER :: print_debug = .FALSE.
     INTEGER ( KIND = ip_ ), PARAMETER :: iter_advanced_max = 5
     INTEGER ( KIND = ip_ ), PARAMETER :: track_out = 29

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: COLT_control_type

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

!   removal of the file alive_file from unit alive_unit causes execution
!    to cease

       INTEGER ( KIND = ip_ ) :: alive_unit = 60
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d                       '

!   maximum number of iterations

       INTEGER ( KIND = ip_ ) :: maxit = 100

!   scale the constraints
!    = 0 unscaled
!    = 1 scale by the infinity norms of the Jacobian rows at the initial point
!    = 2 scale as in 1 but rescale relative to the largest

       INTEGER ( KIND = ip_ ) :: scale_constraints = 0

!   should the initial target be used or ignored?
!    = 0 no
!    = 1 yes, aim for given initial f(x_0)
!    = 2 yes, aim for initial target value (below)

       INTEGER :: use_initial_target = 0

!   any bound larger than infinity in modulus will be regarded as infinite

       REAL ( KIND = rp_ ) :: infinity = ten ** 19

!   overall convergence tolerances. The iteration will terminate when the norm
!    of violation of the constraints (the "primal infeasibility") is smaller
!    than stop_p, the norm of the gradient of the Lagrangian function (the
!    "dual infeasibility") is smaller than stop_d, and the norm of the
!    complementary slackness is smaller than stop_c

!   the required absolute and relative accuracies for the primal infeasibility

       REAL ( KIND = rp_ ) :: stop_abs_p = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: stop_rel_p = epsmch

!   the required absolute and relative accuracies for the dual infeasibility

       REAL ( KIND = rp_ ) :: stop_abs_d = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: stop_rel_d = epsmch

!   the required absolute and relative accuracies for the complementarity

       REAL ( KIND = rp_ ) :: stop_abs_c = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: stop_rel_c = epsmch

!   the required absolute and relative accuracies for the infeasibility
!    The iteration will stop at a minimizer of the infeasibility if the
!    gradient of the infeasibility (J^T c) is smaller in norm than
!    control%stop_abs_i times the norm of c

       REAL ( KIND = rp_ ) :: stop_abs_i = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: stop_rel_i = epsmch

!   the maximum infeasibility tolerated will be the larger of
!    max_abs_i and max_rel_i times the initial infeasibility

       REAL ( KIND = rp_ ) :: max_abs_i = ten
       REAL ( KIND = rp_ ) :: max_rel_i = ten

!   the minimum and maximum constraint scaling factors allowed with
!    scale_constraints > 0

       REAL ( KIND = rp_ ) :: min_constraint_scaling = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: max_constraint_scaling = ten ** 5

!   the fraction of the predicted target improvement allowed

       REAL ( KIND = rp_ ) :: target_fraction = 0.999_rp_

!   the algorithm will terminate if the bracket [target_lower,target_upper]
!   is smaller than small_bracket_tol

       REAL ( KIND = rp_ ) :: small_bracket_tol = ten ** ( - 7 )

!  if an inital target is sought, sets its value and weight

       REAL ( KIND = rp_ ) :: initial_target = - ten ** 4
       REAL ( KIND = rp_ ) :: initial_target_weight = ten ** ( - 8 )

!   perform an advanced start at the end of every iteration when the KKT
!   residuals are smaller than %advanced_start (-ve means never)

       REAL ( KIND = rp_ ) :: advanced_start = ten ** ( - 2 )

!  stop the advanced start search once the residuals sufficientl small

       REAL ( KIND = rp_ ) :: advanced_stop = ten ** ( - 8 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: clock_time_limit = - one

!   full_solution specifies whether the full solution or only highlights
!    will be printed

       LOGICAL :: full_solution = .TRUE.

!   if space_critical is true, every effort will be made to use as little
!    space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!   if deallocate_error_fatal is true, any array/pointer deallocation error
!    will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for NLS

       TYPE ( NLS_control_type ) :: NLS_initial_control
       TYPE ( NLS_control_type ) :: NLS_control

!  control parameters for SSLS

       TYPE ( SSLS_control_type ) :: SSLS_control

     END TYPE COLT_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: COLT_time_type

!  the total CPU time spent in the package

       REAL ( KIND = rp_ ) :: total = 0.0

!  the CPU time spent preprocessing the problem

       REAL ( KIND = rp_ ) :: preprocess = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

       REAL ( KIND = rp_ ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

       REAL ( KIND = rp_ ):: factorize = 0.0

!  the CPU time spent computing the search direction

       REAL ( KIND = rp_ ) :: solve = 0.0

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
     END TYPE COLT_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: COLT_inform_type

!  return status. See COLT_solve for details

       INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the name of the user-supplied evaluation routine for which an error ocurred

       CHARACTER ( LEN = 12 ) :: bad_eval = REPEAT( ' ', 12 )

!  the total number of iterations performed

       INTEGER ( KIND = ip_ ) :: iter = 0

!  the value of the objective function at the best estimate of the solution
!   determined by COLT_solve

       REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the value of the primal infeasibility

       REAL ( KIND = rp_ ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

       REAL ( KIND = rp_ ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

       REAL ( KIND = rp_ ) :: complementary_slackness = HUGE( one )

!  the value of the target

       REAL ( KIND = rp_ ) :: target = HUGE( one )

!  the number of objective and constraint function evaluations

       INTEGER ( KIND = ip_ ) :: fc_eval = 0

!  the number of gradient and Jacobian evaluations

       INTEGER ( KIND = ip_ ) :: gj_eval = 0

!  the number of Hessian evaluations

       INTEGER ( KIND = ip_ ) :: h_eval = 0

!  the number of threads used

       INTEGER ( KIND = ip_ ) :: threads = 1

!  timings (see above)

       TYPE ( COLT_time_type ) :: time

!  inform parameters for NLS

       TYPE ( NLS_inform_type ) :: NLS_inform

!  inform parameters for SSLS

       TYPE ( SSLS_inform_type ) :: SSLS_inform

     END TYPE COLT_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: COLT_nls_dims_type
       INTEGER ( KIND = ip_ ) :: n, m, j_ne, h_ne, p_ne
       INTEGER ( KIND = ip_ ) :: n_f, m_f, j_f_ne, h_f_ne, p_f_ne
     END TYPE COLT_nls_dims_type

     TYPE, PUBLIC :: COLT_data_type
       INTEGER ( KIND = ip_ ) :: branch, eval_status, out, error
       INTEGER ( KIND = ip_ ) :: print_level, start_print, stop_print
       INTEGER ( KIND = ip_ ) :: h_ne, j_ne, n_slacks, nt, i_point, np1, npm
       INTEGER ( KIND = ip_ ) :: iter_advanced
       REAL :: time_start, time_now
       REAL ( KIND = rp_ ) :: clock_start, clock_now
       REAL ( KIND = rp_ ) :: stop_p, stop_d, stop_c, stop_i, s_norm
       REAL ( KIND = rp_ ) :: discrepancy, rnorm, rnorm_old
       REAL ( KIND = rp_ ) :: target_lower, target_upper, f_old
       REAL ( KIND = rp_ ) :: tl, tm, tu, cl, cm, cu, phil, phim, phiu
       LOGICAL :: set_printt, set_printi, set_printw, set_printd
       LOGICAL :: set_printm, printe, printi, printt, printm, printw, printd
       LOGICAL :: print_iteration_header, print_1st_header, accepted
       LOGICAL :: reverse_fc, reverse_gj, reverse_hl
       LOGICAL :: reverse_hj, reverse_hocprods
       LOGICAL :: target_bracketed, from_left, converged, got_fc, got_gj
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: SLACKS
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C_scale, W, V, V2
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: t, f, c, phi
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_old, Y_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C_old, G_old, J_old

!  copy of controls

       TYPE ( COLT_control_type ) :: control

!   NLS data

       TYPE ( COLT_nls_dims_type ) :: nls_dims
       TYPE ( NLPT_problem_type ) :: nls
       TYPE ( NLS_data_type ) :: NLS_data
       TYPE ( GALAHAD_userdata_type ) :: nls_userdata

!  SSLS_data

       TYPE ( SMT_type ) :: C_ssls
       TYPE ( SSLS_data_type ) :: SSLS_data

     END TYPE COLT_data_type

   CONTAINS

!-*  G A L A H A D -  C O L T _ I N I T I A L I Z E  S U B R O U T I N E  *-

     SUBROUTINE COLT_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for COLT controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( COLT_data_type ), INTENT( OUT ) :: data
     TYPE ( COLT_control_type ), INTENT( OUT ) :: control
     TYPE ( COLT_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!  initalize NLS components

     CALL NLS_initialize( data%NLS_data, control%NLS_initial_control,          &
                          inform%NLS_inform )
     CALL NLS_initialize( data%NLS_data, control%NLS_control,                  &
                          inform%NLS_inform )
     RETURN

!  End of subroutine COLT_initialize

     END SUBROUTINE COLT_initialize

!-*-*-   C O L T _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE COLT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by COLT_initialize could (roughly)
!  have been set as:

! BEGIN COLT SPECIFICATIONS (DEFAULT)
! error-printout-device                             6
! printout-device                                   6
! alive-device                                      60
! print-level                                       1
! start-print                                       -1
! stop-print                                        -1
! iterations-between-printing                       1
! maximum-number-of-iterations                      50
! scale-constraints                                 0
! use-initial-target                                0
! infinity-value                                    1.0D+19
! absolute-primal-accuracy                          6.0D-6
! relative-primal-accuracy                          2.0D-16
! absolute-dual-accuracy                            6.0D-6
! relative-dual-accuracy                            2.0D-16
! absolute-complementary-slackness-accuracy         6.0D-6
! relative-complementary-slackness-accuracy         2.0D-16
! absolute-infeasiblity-tolerated                   6.0D-6
! relative-infeasiblity-tolerated                   2.0D-16
! maximum-absolute-infeasibility                    10.0
! maximum-relative-infeasibility                    10.0
! minimum-constraint-scaling-factor                 1.0D-5
! maximum-constraint-scaling-factor                 1.0D+5
! target-fraction                                   0.999
! small-bracket-tolerance                           1.0D-7
! initial-target                                    -1.0D+4
! initial-target-weight                             1.0D-8
! advanced-start                                    1.0D-2
! advanced-stop                                     1.0D-8
! maximum-cpu-time-limit                            -1.0
! maximum-clock-time-limit                          -1.0
! print-full-solution                               no
! space-critical                                    no
! deallocate-error-fatal                            no
! alive-filename                                    ALIVE.d
! output-line-prefix                                ""
! END COLT SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( COLT_control_type ), INTENT( INOUT ) :: control
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
     CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_unit = out + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_level = alive_unit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: start_print = print_level + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_print  = start_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_gap = stop_print + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: maxit = print_gap + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: scale_constraints = maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: use_initial_target                   &
                                             = scale_constraints + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: infinity = use_initial_target + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_p = infinity + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_p = stop_abs_p + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_d = stop_rel_p + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_d = stop_abs_d + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_c = stop_rel_d + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_c = stop_abs_c + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_abs_i = stop_rel_c + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_rel_i = stop_abs_i + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_abs_i = stop_rel_i + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_rel_i = max_abs_i + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: min_constraint_scaling = max_abs_i + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_constraint_scaling               &
                                            = min_constraint_scaling + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: target_fraction                      &
                                            = max_constraint_scaling + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: small_bracket_tol                    &
                                            = target_fraction + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: initial_target                       &
                                            = small_bracket_tol + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: initial_target_weight                &
                                            = initial_target + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: advanced_start                       &
                                            = initial_target_weight + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: advanced_stop = advanced_start + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit = advanced_stop + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: full_solution                        &
                                            = clock_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical                       &
                                            = full_solution + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal               &
                                            = space_critical + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_file                           &
                                            = deallocate_error_fatal + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = alive_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'COLT          '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  define the keywords

     spec%keyword = ''

!  integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( alive_unit )%keyword = 'alive-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print  )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( scale_constraints )%keyword = 'scale-constraints'
     spec( use_initial_target )%keyword = 'use-initial-target'

!  real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
     spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
     spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
     spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
     spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
     spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
     spec( stop_abs_i )%keyword = 'absolute-infeasiblity-tolerated'
     spec( stop_rel_i )%keyword = 'relative-infeasiblity-tolerated'
     spec( min_constraint_scaling )%keyword                                    &
       = 'minimum-constraint-scaling-factor'
     spec( max_constraint_scaling )%keyword                                    &
       = 'maximum-constraint-scaling-factor'
     spec( target_fraction )%keyword = 'target-fraction'
     spec( small_bracket_tol )%keyword = 'small-bracket-tolerance'
     spec( initial_target )%keyword = 'initial-target'
     spec( initial_target_weight )%keyword = 'initial-target-weight'
     spec( advanced_start )%keyword = 'advanced-start'
     spec( advanced_stop )%keyword = 'advanced-stop'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  logical key-words

     spec( full_solution )%keyword = 'print-full-solution'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  character key-words

     spec( alive_file )%keyword = 'alive-filename'
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
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print  ),                          &
                                 control%stop_print ,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_gap ),                            &
                                 control%print_gap,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_constraints ),                    &
                                 control%scale_constraints,                    &
                                 control%error )

     CALL SPECFILE_assign_value( spec( use_initial_target ),                   &
                                 control%use_initial_target,                   &
                                 control%error )
!  set real values

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
     CALL SPECFILE_assign_value( spec( stop_abs_i ),                           &
                                 control%stop_abs_i,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_i ),                           &
                                 control%stop_rel_i,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_constraint_scaling ),               &
                                 control%min_constraint_scaling,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_constraint_scaling ),               &
                                 control%max_constraint_scaling,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( target_fraction ),                      &
                                 control%target_fraction,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( small_bracket_tol ),                    &
                                 control%small_bracket_tol,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_target ),                       &
                                 control%initial_target,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_target_weight ),                &
                                 control%initial_target_weight,                &
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

!  set logical values

     CALL SPECFILE_assign_value( spec( full_solution ),                        &
                                 control%full_solution,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  set character values

     CALL SPECFILE_assign_string( spec( alive_file ), control%alive_file,      &
                                  control%error )
     CALL SPECFILE_assign_value( spec( prefix ), control%prefix,               &
                                 control%error )

!  read the controls for the sub-problem solvers and preconditioner

     IF ( PRESENT( alt_specname ) ) THEN

!  set NLS control values

       CALL NLS_read_specfile( control%NLS_control, device,                    &
             alt_specname = TRIM( alt_specname ) // '-NLS' )
       CALL NLS_read_specfile( control%NLS_initial_control, device,            &
             alt_specname = TRIM( alt_specname ) // '-NLS-INITIAL' )
       CALL SSLS_read_specfile( control%SSLS_control, device,                  &
             alt_specname = TRIM( alt_specname ) // '-SSLS' )

     ELSE

!  set NLS control values

       CALL NLS_read_specfile( control%NLS_control, device )
       CALL NLS_read_specfile( control%NLS_initial_control, device,            &
             alt_specname = 'NLS-INITIAL' )
       CALL SSLS_read_specfile( control%SSLS_control, device )
     END IF

     RETURN

!  End of subroutine COLT_read_specfile

     END SUBROUTINE COLT_read_specfile

!-*-*-*-  G A L A H A D -  C O L T _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE COLT_solve( nlp, control, inform, data, userdata,              &
                            eval_FC, eval_J, eval_SGJ, eval_HL, eval_HLC,      &
                            eval_HJ, eval_HOCPRODS, eval_HCPRODS )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  COLT_solve, a method for finding a local minimizer of a function subject
!  to general constraints and simple bounds on the sizes of the variables

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_COLT.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_COLT_precision
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  m is a scalar variable of type default integer, that holds the number of
!   constraints
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
!    entries in the lower triangular part of H in the sparse co-ordinate
!    storage scheme. It need not be set for any of the other three schemes.
!
!   H%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the  lower triangular part of the Hessian
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
!    integer, that holds the starting position of each row of the lower
!    triangular part of H, as well as the total number of entries plus one,
!    in the sparse row-wise storage scheme. It need not be allocated when the
!    other schemes are used.
!
!  Go is scalar variable of type SVT_TYPE that holds the gradient of the
!   objective function g. The following components are used here:
!
!   Go%sparse is a scalar variable of type default logical, that should be set
!    to .TRUE. if only the structural nonzeros of the vector g are provided.
!    If it is .FALSE., all values are provided (in the natural order
!
!   Go%ne is a scalar variable of type default integer, that holds the
!    number of nonzeros in the vector g. If Go%sparse is .FALSE., GO%ne
!    should be n
!
!   Go%val is a rank-one allocatable array of type default real, that
!    holds the values of the nonzeros in the vector g. If Go%sparse is .FALSE.,
!    Go%val(i) = g_i for i = 1,...,n
!
!   Go%ind is a rank-one allocatable array of type default integer, that
!    holds the indices of the nonzeros in the vector g. If Go%sparse is .FALSE.,
!    GO%ind need not be provided
!
!  J is scalar variable of type SMT_TYPE that holds the Jacobian matrix J. The
!   following components are used here:
!
!   J%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of J%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of J%type must
!    contain the string COORDINATE, for the sparse row-wise storage scheme, and
!    the first fourteen components of J%type must contain the string
!    SPARSE_BY_ROWS.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into J%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Hessian we wish to
!    store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%J%type, 'COORDINATE' )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   J%ne is a scalar variable of type default integer, that holds the number of
!    entries in J in the sparse co-ordinate storage scheme. It need not be set
!    for any of the other two schemes.
!
!   J%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the Jacobian matrix J in any of the available
!    storage schemes.
!
!   J%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of J in the sparse co-ordinate storage scheme. It need not
!    be allocated for any of the other two schemes.
!
!   J%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of J in either the sparse co-ordinate,
!    or the sparse row-wise storage scheme. It need not be allocated when the
!    dense storage scheme is used.
!
!   J%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of each row of J, as well as
!    the total number of entries plus one, in the sparse row-wise storage
!    scheme. It need not be allocated when the other schemes are used.
!
!  G is a rank-one allocatable array of dimension n and type default real,
!   that holds the gradient g of the objective function. The j-th component of
!   G, j = 1,  ... ,  n, contains g_j.
!
!  f is a scalar variable of type default real, that holds the value of
!   the objective function.
!
!  C is a rank-one allocatable array of dimension n and type default real,
!   that holds the constraint value c. The i-th component of C, i = 1, ... , m,
!   contains g_j.
!
!  X is a rank-one allocatable array of dimension n and type default real, that
!   holds the values x of the optimization variables. The j-th component of
!   X, j = 1, ... , n, contains x_j.
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
! control is a scalar variable of type TRU_control_type. See TRU_initialize
!  for details
!
! inform is a scalar variable of type TRU_inform_type. On initial entry,
!  inform%status should be set to 1. On exit, the following components will
!  have been set:
!
!  status is a scalar variable of type default integer, that gives
!   the exit status from the package. Possible values are:
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
!    -3. The restriction nlp%n > 0 or requirement that nlp%H_type contains
!        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
!          or 'DIAGONAL' has been violated.
!    -5. The problem appears (locally) to have no feasible point.
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
!        control%maxit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -19. Too much time has passed. This may happen if control%cpu_time_limit or
!        control%clock_time_limit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -40. The user has forced termination of the solver by removing the file
!        named control%alive_file from unit control%alive_unit.
!   -78. A problem evaluation error occurred.
!
!  -111. There are no general constraints.
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
!        nabla_xx f(x) + sum_i=1^m y_i c_i(x) at the point x indicated in nlp%X
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
!     7. The user should compute the matrix whose columns are the products
!        H_i(x)v between the HessianH_i(x) of the ith residual function at
!        the point x indicated in nlp%X a given vector v held in data%V.
!        The nonzeros for column i must be stored in nlp%P%val(l), for
!        l = nlp%P%ptr(i), ...,  nlp%P%ptr(i+1) for each i = 1,...,m,
!        in the same order as the row indices were assigned on input in
!        nlp%P%row(l). If the user is unable to evaluate the products -
!        for instance, if a component of H_i(x) is undefined at x - the
!        user need not assign nlp%P%val, but should then set
!        data%eval_status to a non-zero value.
!     8. ...
!     9. ...
!
!  alloc_status is a scalar variable of type default integer, that gives
!   the status of the last attempted array allocation or deallocation.
!   This will be 0 if status = 0.
!
!  bad_alloc is a scalar variable of type default character
!   and length 80, that  gives the name of the last internal array
!   for which there were allocation or deallocation errors.
!   This will be the null string if status = 0.
!
!  iter is a scalar variable of type default integer, that holds the
!   number of iterations performed.
!
!  cg_iter is a scalar variable of type default integer, that gives the
!   total number of conjugate-gradient iterations required.
!
!  fc_eval is a scalar variable of type default integer, that gives the
!   total number of objective and constraint function evaluations performed.
!
!  gj_eval is a scalar variable of type default integer, that gives the
!   total number of objective gradient and constraint Jacobian evaluations
!   performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of Lagrangian Hessian evaluations performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_g is a scalar variable of type default real, that holds the
!   value of the norm of the objective function gradient at the best estimate
!   of the solution found.
!
!  time is a scalar variable of type TRU_time_type whose components are used to
!   hold elapsed CPU and clock times for the various parts of the calculation.
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total CPU time spent in the package.
!
!    analyse is a scalar variable of type default real, that gives
!      the CPU time spent analysing required matrices prior to factorization.
!
!    factorize is a scalar variable of type default real, that gives
!      the CPU time spent factorizing the required matrices.
!
!    solve is a scalar variable of type default real, that gives
!     the CPU time spent using the factors to solve relevant linear equations.
!
!    clock_total is a scalar variable of type default real, that gives
!     the total clock time spent in the package.
!
!    clock_analyse is a scalar variable of type default real, that gives
!      the clock time spent analysing required matrices prior to factorization.
!
!    clock_factorize is a scalar variable of type default real, that gives
!      the clock time spent factorizing the required matrices.
!
!    clock_solve is a scalar variable of type default real, that gives
!     the clock time spent using the factors to solve relevant linear equations.
!
!  data is a scalar variable of type TRU_data_type used for internal data.
!
!  userdata is a scalar variable of type GALAHAD_userdata_type which may be
!   used to pass user data to and from the eval_* subroutines (see below)
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
!  eval_FC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If C is present, the values of the constraint functions
!   c(x) evaluated at x=X must be returned in C, and the status variable set
!   to 0. If the evaluation is impossible at X, status should be set to a
!   nonzero value. If eval_FC is not present, COLT_solve will return to the
!   user with inform%status = 2 each time an evaluation is required.
!
!  eval_J is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Jacobian
!   nabla_x c(x) of the residual function evaluated at x=X must be returned in
!   J_val in the same order as presented in nlp%J,, and the status variable set
!   to 0. If the evaluation is impossible at X, status should be set to a
!   nonzero value. If eval_J is not present, COLT_solve will return to the
!   user with inform%status = 3 each time an evaluation is required.
!
!  eval_SGJ is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). If G is present, the components of
!   the nonzeros of the gradient nabla_x f(x) of the objective function
!   evaluated at x=X in the same order as presented in nlp%GO,
!   must be returned in G. If J is present, the nonzeros of the Jacobian
!   nabla_x c(x) evaluated at x=X must be returned in J_val in the same
!   order as presented in nlp%J, and the status variable set to 0.
!   If the evaluation is impossible at x=X, status should be set to a
!   nonzero value. If eval_SGJ is not present, COLT_solve will return to the
!   user with inform%status = 4 each time an evaluation is required.
!
!  eval_HL is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) - sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated
!   at x=X and y=Y must be returned in H_val in the same order as
!   presented in nlp%H, and the status variable set to 0. If the evaluation is
!   impossible at X, status should be set to a nonzero value. If eval_HL is
!   not present, COLT_solve will return to the user with inform%status = 9
!   each time an evaluation is required.
!
!  eval_HLC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the weighted Hessian
!   H(x,y) = sum_i y_i nabla_xx c_i(x) of the residual function evaluated at
!   x=X and y=Y must be returned in H_val in the same order as presented in
!   nlp%H, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_HLC is not present,
!   COLT_solve will return to the user with inform%status = 5 each time an
!   evaluation is required.
!
!  eval_HJ is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   y_0 nabla_xx f(x) - sum_i=1^m y_i c_i(x) of the John function evaluated
!   at x=X, y_0 = y0 and y=Y must be returned in H_val in the same order as
!   presented in nlp%H, and the status variable set to 0. If the evaluation is
!   impossible at X, status should be set to a nonzero value. If eval_HJ is
!   not present, COLT_solve will return to the user with inform%status = 6
!   each time an evaluation is required.
!
!  eval_HCPRODS is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The nonzeros of
!   the matrix whose ith column is the product nabla_xx c_i(x) v between
!   the Hessian of the ith residual function evaluated at x=X and the
!   vector v=V must be returned in PC_val in the same order as presented in
!   nlp%P, and the status variable set to 0. If the evaluation is impossible
!   at X, status should be set to a nonzero value. If eval_HCPRODS is not
!   present, NLS_solve will return to the user with inform%status = 7
!   each time an evaluation is required. The Hessians have already been
!   evaluated or used at x=X if got_h is .TRUE.
!
!  eval_HOCPRODS is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The nonzeros of
!   the matrix whose ith column is the product nabla_xx c_i(x) v between
!   the Hessian of the ith residual function evaluated at x=X and the
!   vector v=V must be returned in PC_val in the same order as presented in
!   nlp%P, and the status variable set to 0. The nonzeros of the vector
!   the product nabla_xx f(x) v between the Hessian of the objective
!   function evaluated at x=X and the vector v=V must be returned in PO_val.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_HOCPRODS is not present, NLS_solve will return to the user
!   with inform%status = 8 each time an evaluation is required. The Hessians
!   have already been evaluated or used at x=X if got_h is .TRUE.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( COLT_control_type ), INTENT( INOUT ) :: control
     TYPE ( COLT_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( COLT_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_FC, eval_J, eval_SGJ, eval_HL, eval_HLC, eval_HJ,        &
                 eval_HOCPRODS, eval_HCPRODS

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_FC( status, X, userdata, f, C )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), OPTIONAL, INTENT( OUT ) :: f
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FC
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_J( status, X, userdata, J )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: J
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_J
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_SGJ( status, X, userdata, G, J )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: J
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_SGJ
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HL( status, X, Y, userdata, Hval, no_f )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HLC( status, X, Y, userdata, Hval )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_HLC
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HJ( status, X, y0, Y, userdata, Hval )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), INTENT( IN ) :: y0
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_HJ
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HCPRODS( status, X, V, userdata, PCval, got_h )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: PCval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HCPRODS
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HOCPRODS( status, X, V, userdata, POval, PCval, got_h )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: POval
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: PCval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HOCPRODS
     END INTERFACE

!  ==========================================================================
!
!  Problem -
!
!    minimize   f(x) subject to  c_l <= c(x) <= c_u and  x_l <= x <= x_u
!
!  where c_l and c_u are vectors of lower and upper bounds on the constraint
!  vector-valued function c( x ), and x_l and x_u are vectors of lower and
!  upper bounds on the primal variables x.
!
!  ==========================================================================

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, ir, ic, j, j_ne, l, nroots
     REAL ( KIND = rp_ ) :: y0, multiplier_norm, t_phi, t_c, frac, num, den
     REAL ( KIND = rp_ ) :: dl, dm, a0, a1, a2, root1, root2, alpha
     LOGICAL :: names
     CHARACTER ( LEN = 80 ) :: array_name

!  functions

!$   INTEGER :: OMP_GET_MAX_THREADS

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 900
     END IF
     IF ( inform%status == 1 ) data%branch = 10

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! initial objective and constraint evaluation
       GO TO 20
     CASE ( 100 ) ! various evaluations
       GO TO 100
     CASE ( 200 ) ! various evaluations
       GO TO 200
     CASE ( 310 ) ! gradient and Jacobian evaluation
       GO TO 310
!    CASE ( 340 ) ! Hessian evaluation
!      GO TO 340
     CASE ( 400 ) ! various evaluations
       GO TO 400
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
!$   inform%threads = INT( OMP_GET_MAX_THREADS( ), KIND = ip_ )
     inform%status = GALAHAD_ok
     inform%alloc_status = 0 ; inform%bad_alloc = ''
     inform%iter = 0
     inform%fc_eval = 0 ; inform%gj_eval = 0 ; inform%h_eval = 0

     inform%obj = HUGE( one )

!  copy control parameters so that the package may alter values if necessary

     data%control = control

!  decide how much reverse communication is required

     data%reverse_fc = .NOT. PRESENT( eval_FC )
     data%reverse_gj = .NOT. PRESENT( eval_SGJ )
     data%reverse_hl = .NOT. PRESENT( eval_HL )
     data%reverse_hj = .NOT. PRESENT( eval_HJ )
     data%reverse_hocprods = .NOT. PRESENT( eval_HOCPRODS )

!  control the output printing

     data%out = data%control%out ; data%error = data%control%error

!  error output

     data%printe = data%error > 0 .AND. data%control%print_level >= 1

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printm, but with checking of residuals, etc

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debugging printing with significant arrays printed

     data%set_printd = data%out > 0 .AND. data%control%print_level >= 5

!  print level shorthands

     IF ( data%control%start_print < 0 ) THEN
       data%start_print = - 1
     ELSE
       data%start_print = data%control%start_print
     END IF

     IF ( data%control%stop_print < 0 ) THEN
       data%stop_print = data%control%maxit + 1
     ELSE
       data%stop_print = data%control%stop_print
     END IF

!  set print agenda for the first iteration

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE.
       data%printd = .FALSE. ; data%print_level = 0
     END IF

     data%print_iteration_header = data%print_level > 0
     data%print_1st_header = .TRUE.

     IF ( data%printd ) WRITE( data%out, "( ' (A1:1-2)' )" )

! -----------------------------------------------------------------------------
!                            SET UP THE PROBLEM
! -----------------------------------------------------------------------------

     nlp%P%ne = nlp%P%ptr( nlp%m + 1 ) - 1

!  project x to ensure feasibility

     nlp%X = MAX( nlp%X_l, MIN( nlp%X, nlp%X_u ) )

!  set up static data for the least-squares target objective

     CALL COLT_setup_problem( nlp, data%nls, data%nls_dims, data%control,      &
                              inform, data%n_slacks, data%SLACKS,              &
                              h_available = .TRUE., p_available = .TRUE. )

!  evaluate the objective and general constraint function values

     IF ( data%reverse_fc ) THEN
       data%branch = 20 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
       IF ( data%eval_status /= 0 ) THEN
         inform%bad_eval = 'eval_FC'
         inform%status = GALAHAD_error_evaluation ; GO TO 900
       END IF
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     inform%obj = nlp%f
     inform%fc_eval = inform%fc_eval + 1

     inform%primal_infeasibility = TWO_NORM( nlp%C )

!  print problem name and header, if requested

     IF ( data%printi ) WRITE( data%out,                                       &
         "( A, ' +', 76( '-' ), '+', /,                                        &
      &     A, 14X, 'Constrained Optimization via Least-squares Targets', /,   &
      &     A, ' +', 76( '-' ), '+' )" ) prefix, prefix, prefix

!  determine the number of nonzeros in the Hessian

     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       data%H_ne = nlp%H%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%H_ne = nlp%H%ptr( nlp%m + 1 ) - 1
     CASE ( 'DENSE' )
       data%H_ne = ( nlp%n * ( nlp%n+ 1 ) ) / 2
     END SELECT

!  determine the number of nonzeros in the constraint Jacobian

     SELECT CASE ( SMT_get( nlp%J%type ) )
     CASE ( 'COORDINATE' )
       data%J_ne = nlp%J%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%J_ne = nlp%J%ptr( nlp%m + 1 ) - 1
     CASE ( 'DENSE' )
       data%J_ne = nlp%m * nlp%n
     END SELECT

!  set up space for general vectors used

     array_name = 'colt: nlp%gL'
     CALL SPACE_resize_array( nlp%n, nlp%gL,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: nlp%X_status'
     CALL SPACE_resize_array( nlp%n, nlp%X_status,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: nlp%C_status'
     CALL SPACE_resize_array( nlp%m, nlp%C_status,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

!  set initial values

     data%accepted = .TRUE.
     data%s_norm = zero
     data%from_left = .FALSE.

!  set scale factors if required

     IF ( data%control%scale_constraints > 0 ) THEN
       array_name = 'colt: data%C_scale'
       CALL SPACE_resize_array( nlp%m, data%C_scale,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910
       data%C_scale( : nlp%m ) = one
     END IF

!  set space for matrices and vectors required for advanced start if necessary

     IF ( data%control%advanced_start > zero ) THEN
       data%np1 = nlp%n + 1 ; data%npm = nlp%n + nlp%m

       data%C_ssls%n = nlp%m ; data%C_ssls%m = nlp%m ; data%C_ssls%ne = 1
       CALL SMT_put( data%C_ssls%type, 'SCALED_IDENTITY', inform%alloc_status )

       array_name = 'colt: data%C%val'
       CALL SPACE_resize_array( data%C_ssls%ne, data%C_ssls%val,               &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: data%V'
       CALL SPACE_resize_array( data%npm, data%V,                              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: data%V2'
       CALL SPACE_resize_array( data%npm, data%V2,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: data%X_old'
       CALL SPACE_resize_array( nlp%n, data%X_old,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: data%Y_old'
       CALL SPACE_resize_array( nlp%m, data%Y_old,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: data%C_old'
       CALL SPACE_resize_array( nlp%m, data%C_old,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: data%G_old'
       CALL SPACE_resize_array( nlp%Go%ne, data%G_old,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: data%J_old'
       CALL SPACE_resize_array( data%J_ne, data%J_old,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set up and analyse the data structures for the matrix required for the
!  advanced start

       CALL SSLS_analyse( nlp%n, nlp%m, nlp%H, nlp%J, data%C_ssls,             &
                          data%SSLS_data, data%control%SSLS_control,           &
                          inform%SSLS_inform )
     END IF

!  compute norms of the primal and dual feasibility and the complemntary
!  slackness

     multiplier_norm = one
     inform%primal_infeasibility =                                             &
      OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                       &
                                nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ) )
     inform%dual_infeasibility = zero
!  NB. Compute this properly later
     inform%complementary_slackness =                                          &
       OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),                   &
          nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),            &
          nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                         &
          nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) ) / multiplier_norm

!  compute the stopping tolerances

     data%stop_p = MAX( data%control%stop_abs_p,                               &
        data%control%stop_rel_p * inform%primal_infeasibility )
     data%stop_d = MAX( data%control%stop_abs_d,                               &
       data%control%stop_rel_d * inform%dual_infeasibility )
     data%stop_c = MAX( data%control%stop_abs_c,                               &
       data%control%stop_rel_c * inform%complementary_slackness )
     data%stop_i = MAX( data%control%stop_abs_i, data%control%stop_rel_i )
     IF ( data%printi ) WRITE( data%out,                                       &
         "(  /, A, '  Primal    convergence tolerance =', ES11.4,              &
        &    /, A, '  Dual      convergence tolerance =', ES11.4,              &
        &    /, A, '  Slackness convergence tolerance =', ES11.4 )" )          &
             prefix, data%stop_p, prefix, data%stop_d, prefix, data%stop_c


     IF ( control%use_initial_target > 0 ) GO TO 190

!  ----------------------------------------------------------------------------
!                 INITIAL TARGET PHASE: AIM FOR FEASIBLE POINT
!  ----------------------------------------------------------------------------

!  solve the feasibility problem: min 1/2||c(x)||^2 using the nls solver.
!  All function evlautions are for the vector of residuals c(x)

     inform%NLS_inform%status = 1
     data%nls%X( : data%nls%n ) = nlp%X
!write(6,*) ' x ', nlp%X
     data%control%NLS_initial_control%jacobian_available = 2
     data%control%NLS_initial_control%subproblem_control%jacobian_available = 2
     data%control%NLS_initial_control%hessian_available = 2
     data%nls%m = data%nls%m - 1

!  record dimensions for feasibility problem

     data%nls%n = data%nls_dims%n_f ; data%nls%m = data%nls_dims%m_f
     data%nls%J%ne = data%nls_dims%j_f_ne ; data%nls%H%ne = data%nls_dims%h_f_ne
     data%nls%P%ne = data%nls_dims%p_f_ne

!    DO        ! loop to solve the feasibility problem
 100 CONTINUE  ! mock loop to solve the feasibility problem

!  call the least-squares solver

       CALL NLS_solve( data%nls, data%control%NLS_initial_control,             &
                       inform%NLS_inform, data%NLS_data, data%NLS_userdata )

!  respond to requests for further details

       SELECT CASE ( inform%NLS_inform%status )

!  obtain the residuals (and the objective function)

       CASE ( 2 )
         IF ( data%reverse_fc ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 100 ; inform%status = 2 ; RETURN
         ELSE
           CALL eval_FC( data%eval_status, data%nls%X, userdata,               &
                         nlp%f, nlp%C )
           data%nls%C( : nlp%m ) = nlp%C( : nlp%m )
           IF ( print_debug ) WRITE(6,"( ' nls%C = ', /, ( 5ES12.4 ) )" )      &
             data%nls%C( : data%nls%m )
         END IF

!  obtain the Jacobian

       CASE ( 3 )
         IF ( data%reverse_gj ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 100 ; inform%status = 3 ; RETURN
         ELSE
           CALL eval_J( data%eval_status, data%nls%X, userdata,                &
                        nlp%J%val )
           SELECT CASE ( SMT_get( data%nls%J%type ) )
           CASE ( 'DENSE' )
             j_ne = 0 ; l = 0
             DO i = 1, nlp%m  ! from each row of J(x) in turn
               data%nls%J%val( j_ne + 1 : j_ne + nlp%n )                       &
                 = nlp%J%val( l + 1 : l + nlp%n )
               j_ne = j_ne + data%nls%n ; l = l + nlp%n
             END DO
           CASE ( 'SPARSE_BY_ROWS' )
             DO i = 1, nlp%m  ! from each row of J(x) in turn
               j_ne = data%nls%J%ptr( i ) - 1
               DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
                 j_ne = j_ne + 1
                 data%nls%J%val( j_ne ) = nlp%J%val( l )
               END DO
             END DO
           CASE ( 'COORDINATE' )
             data%nls%J%val( : nlp%J%ne ) = nlp%J%val( : nlp%J%ne ) ! from J
             IF ( print_debug ) WRITE(6,"( ' nls%J', / ( 3( 2I6, ES12.4 )))")  &
              ( data%nls%J%row( i ), data%nls%J%col( i ),                      &
                data%nls%J%val( i ), i = 1, nlp%J%ne )
           END SELECT
         END IF

!  obtain the Hessian

       CASE ( 4 )
         IF ( data%reverse_hj ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 100 ; inform%status = 5 ; RETURN
         ELSE
           CALL eval_HLC( data%eval_status, data%nls%X,                        &
                          data%NLS_data%Y( 1 : nlp%m ),                        &
                          userdata, data%nls%H%val )
           IF ( print_debug ) WRITE(6,"( ' nls%H', / ( 3( 2I6, ES12.4 )))" )   &
            ( data%nls%H%row( i ), data%nls%H%col( i ),                        &
              data%nls%H%val( i ), i = 1, data%nls%H%ne )
         END IF

!  form a Jacobian-vector product

       CASE ( 5 )
         write(6,*) ' no nls_status = 5 as yet, stopping'
         stop
!        CALL JACPROD( data%eval_status, data%nls%X,                           &
!                      data%NLS_userdata, data%NLS_data%transpose,             &
!                      data%NLS_data%U, data%NLS_data%V )

!  form a Hessian-vector product

       CASE ( 6 )
         write(6,*) ' no nls_status = 6 as yet, stopping'
         stop
!        CALL HESSPROD( data%eval_status, data%nls%X, data%NLS_data%Y,         &
!                       data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  form residual Hessian-vector products

       CASE ( 7 )
         IF ( data%reverse_hocprods ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 100 ; inform%status = 7 ; RETURN
         ELSE
           CALL eval_HCPRODS( data%eval_status, data%nls%X,                    &
                              data%nls_data%V, userdata,                       &
                              data%nls%P%val( 1 : nlp%P%ne ) )
           IF ( print_debug ) THEN
             WRITE(6,"( ' nls%P' )" )
             DO j = 1, nlp%m
               WRITE(6,"( 'column ',  I0, /, / ( 4( I6, ES12.4 ) ) )" ) &
                 j, ( data%nls%P%row( i ), data%nls%P%val( i ), i = &
                      data%nls%P%ptr( j ), data%nls%P%ptr( j + 1 ) - 1 )
             END DO
           END IF
         END IF

!  apply the preconditioner

       CASE ( 8 )
         write(6,*) ' no nls_status = 8 as yet, stopping'
         stop
!        CALL SCALE( data%eval_status, data%nls%X,                             &
!                    data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  terminal exit from feasibility problem solver

       CASE DEFAULT
         nlp%X = data%nls%X( : data%nls%n )
         GO TO 180
       END SELECT

!  perform another iteration

       GO TO 100 ! end of mock loop to solve the feasibility problem

!    END DO ! end of loop to solve the feasibility problem
 180 CONTINUE

     inform%obj = nlp%f
     inform%primal_infeasibility = inform%nls_inform%norm_c

!  check for feasibility

     IF ( inform%primal_infeasibility > data%stop_p ) THEN
       WRITE( data%out, "( ' no feasible point found, infeasibility = ',       &
      &  ES12.4 )" )  inform%primal_infeasibility
       inform%status = GALAHAD_error_primal_infeasible
       GO TO 910
     ELSE
       WRITE( data%out, "( ' feasible point found, infeasibility and',         &
      & ' objective =', 2ES12.4 )" ) inform%primal_infeasibility, inform%obj
     END IF

!  set up initial target

     y0 = one
     data%target_upper = inform%obj
     data%target_lower = - infinity
     inform%target = data%target_upper - one

!!!! remove

write(6,*) ' fixed initial value ... remove!!'
     inform%target = -10.0_rp_


     data%target_bracketed = .FALSE.
     data%converged = .FALSE.
!    WRITE( data%out, 2070 ) data%target_lower, inform%target, data%target_upper
     data%nt = 0

!  restore dimensions for target problem

     data%nls%n = data%nls_dims%n ; data%nls%m = data%nls_dims%m
     data%nls%J%ne = data%nls_dims%j_ne ; data%nls%H%ne = data%nls_dims%h_ne
     data%nls%P%ne = data%nls_dims%p_ne

     GO TO 290

!  ----------------------------------------------------------------------------
!                 INITIAL TARGET PHASE: AIM FOR SPECIFIED TARGET
!  ----------------------------------------------------------------------------

 190 CONTINUE

!  set up weight and target values

     array_name = 'colt: data%W'
     CALL SPACE_resize_array( data%nls%m, data%W,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910
     data%W( : nlp%m ) = one
     data%W( data%nls%m ) = control%initial_target_weight
     inform%target = control%initial_target

!  solve the target problem: min 1/2||c(x)||^2 + omega/2 (f(x)-t)^2 for
!  a given target t and weight omega

     inform%NLS_inform%status = 1
     data%nls%X( : data%nls%n ) = nlp%X
!    WRITE(6,*) ' x ', nlp%X
     data%control%NLS_control%jacobian_available = 2
     data%control%NLS_control%subproblem_control%jacobian_available = 2
     data%control%NLS_control%hessian_available = 2
     data%got_fc = .TRUE.

!    DO        ! loop to solve problem
 200   CONTINUE  ! mock loop to solve problem

!  call the least-squares solver to find x(t)

       CALL NLS_solve( data%nls, data%control%NLS_control,                     &
                       inform%NLS_inform, data%NLS_data, data%NLS_userdata,    &
                       W = data%W )

!  respond to requests for further details

       SELECT CASE ( inform%NLS_inform%status )

!  obtain the residuals

       CASE ( 2 )
         IF ( data%reverse_fc ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 200 ; inform%status = 2 ; RETURN
         ELSE
           IF ( data%got_fc ) THEN ! skip eval_fc if f and c are already known
             data%got_fc = .FALSE.
           ELSE
             CALL eval_FC( data%eval_status, data%nls%X, userdata,             &
                           nlp%f, nlp%C )
           END IF
           data%discrepancy = nlp%f - inform%target
           data%nls%C( : nlp%m ) = nlp%C( : nlp%m )
           data%nls%C( data%nls%m ) = data%discrepancy
           IF ( print_debug ) WRITE(6,"( ' nls%C = ', /, ( 5ES12.4 ) )" )      &
             data%nls%C( : data%nls%m )
         END IF

!  obtain the Jacobian

       CASE ( 3 )
         IF ( data%reverse_gj ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 200 ; inform%status = 4 ; RETURN
         ELSE
           CALL eval_SGJ( data%eval_status, data%nls%X, userdata,              &
                          nlp%Go%val, nlp%J%val )
           SELECT CASE ( SMT_get( data%nls%J%type ) )
           CASE ( 'DENSE' )
             j_ne = 0 ; l = 0
             DO i = 1, nlp%m  ! from each row of J(x) in turn
               data%nls%J%val( j_ne + 1 : j_ne + nlp%n )                       &
                 = nlp%J%val( l + 1 : l + nlp%n )
               j_ne = j_ne + data%nls%n ; l = l + nlp%n
             END DO
             IF ( nlp%Go%sparse ) THEN ! from g(x)
               DO i = 1, nlp%Go%ne
                 data%nls%J%val( j_ne + nlp%Go%ind( i ) ) = nlp%Go%val( i )
               END DO
             ELSE
               data%nls%J%val( j_ne + 1 : j_ne + nlp%n )                       &
                 = nlp%Go%val( 1 : nlp%n )
             END IF
           CASE ( 'SPARSE_BY_ROWS' )
             DO i = 1, nlp%m  ! from each row of J(x) in turn
               j_ne = data%nls%J%ptr( i ) - 1
               DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
                 j_ne = j_ne + 1
                 data%nls%J%val( j_ne ) = nlp%J%val( l )
               END DO
             END DO
             j_ne = data%nls%J%ptr( data%nls%m ) - 1
             DO j = 1, nlp%Go%ne
               j_ne = j_ne + 1
               data%nls%J%val( j_ne ) = nlp%Go%val( j )
             END DO
           CASE ( 'COORDINATE' )
             data%nls%J%val( : nlp%J%ne ) = nlp%J%val( : nlp%J%ne ) ! from J
             j_ne = nlp%J%ne + data%n_slacks
             DO j = 1, nlp%Go%ne
               j_ne = j_ne + 1
               data%nls%J%val( j_ne ) = nlp%Go%val( j )
             END DO
             IF ( print_debug ) WRITE(6,"( ' nls%J', / ( 3( 2I6, ES12.4 )))")  &
              ( data%nls%J%row( i ), data%nls%J%col( i ),                      &
                data%nls%J%val( i ), i = 1, data%nls%J%ne )
           END SELECT
         END IF

!  obtain the Hessian

       CASE ( 4 )
         IF ( data%reverse_hj ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 200 ; inform%status = 6 ; RETURN
         ELSE
           CALL eval_HJ( data%eval_status, data%nls%X,                         &
                         data%NLS_data%Y( data%nls%m ),                        &
                         data%NLS_data%Y( 1 : nlp%m ),                         &
                         userdata, data%nls%H%val )
           IF ( print_debug ) WRITE(6,"( ' nls%H', / ( 3( 2I6, ES12.4 )))" )   &
            ( data%nls%H%row( i ), data%nls%H%col( i ),                        &
              data%nls%H%val( i ), i = 1, data%nls%H%ne )
         END IF

!  form a Jacobian-vector product

       CASE ( 5 )
         write(6,*) ' no nls_status = 5 as yet, stopping'
         stop
!        CALL JACPROD( data%eval_status, data%nls%X,                           &
!                      data%NLS_userdata, data%NLS_data%transpose,             &
!                      data%NLS_data%U, data%NLS_data%V )

!  form a Hessian-vector product

       CASE ( 6 )
         write(6,*) ' no nls_status = 6 as yet, stopping'
         stop
!        CALL HESSPROD( data%eval_status, data%nls%X, data%NLS_data%Y,         &
!                       data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  form residual Hessian-vector products

       CASE ( 7 )
         IF ( data%reverse_hocprods ) THEN
           nlp%X = data%nls%X( : data%nls%n )
           data%branch = 200 ; inform%status = 8 ; RETURN
         ELSE
           CALL eval_HOCPRODS( data%eval_status, data%nls%X,                   &
                               data%nls_data%V, userdata,                      &
                               data%nls%P%val( nlp%P%ne + 1 :                  &
                                               data%nls%P%ne ),                &
                               data%nls%P%val( 1 : nlp%P%ne ) )
           IF ( print_debug ) THEN
             WRITE(6,"( ' nls%P' )" )
             DO j = 1, data%nls%n
               WRITE(6,"( 'column ',  I0, /, / ( 4( I6, ES12.4 ) ) )" ) &
                 j, ( data%nls%P%row( i ), data%nls%P%val( i ), i = &
                      data%nls%P%ptr( j ), data%nls%P%ptr( j + 1 ) - 1 )
             END DO
           END IF
         END IF

!  apply the preconditioner

       CASE ( 8 )
         write(6,*) ' no nls_status = 8 as yet, stopping'
         stop
!        CALL SCALE( data%eval_status, data%nls%X,                             &
!                    data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  terminal exit from the minimization loop

       CASE DEFAULT
         nlp%X = data%nls%X( : data%nls%n )
         GO TO 280
       END SELECT
       GO TO 200
!    END DO ! end of loop to solve the target problem

 280 CONTINUE

     inform%obj = nlp%f
     inform%primal_infeasibility = TWO_NORM( nlp%C )
     WRITE( data%out, "( /, ' f, ||c||, evals = ', 2ES13.5, 1X, I0 )" )        &
       nlp%f, inform%primal_infeasibility, inform%NLS_inform%iter

!  with luck :) we arrive at a target well below the minimizer

     inform%target = inform%obj
     data%target_upper = infinity
     data%target_lower = inform%target

stop


!  ----------------------------------------------------------------------------
!                  START OF MAIN LOOP OVER EVOLVING TARGETS
!  ----------------------------------------------------------------------------

!  starting from an initial target (target_upper) at a feasible point, the
!  aim is to move left (i.e., decrease the target) until an infeasible target
!  is found (target_lower). This then brackets the optimal target. Thereafter
!  a sequence of targets is chosen to shrink the bracket

 290 CONTINUE
     IF ( data%printt ) WRITE( data%out, 2070 ) data%target_lower,             &
       inform%target, data%target_upper, nlp%f, inform%primal_infeasibility, 0
     IF ( data%printm ) WRITE( data%out, 2080 ) nlp%X

!  solve the target problem: min 1/2||c(x),f(x)-t||^2 for a given target t

     IF ( data%printd ) WRITE( data%out, "( A, ' (A1:3)' )" ) prefix
!    DO
 300   CONTINUE
!      IF ( inform%status == GALAHAD_ok ) GO TO 900

       IF ( data%printd ) THEN
         WRITE( data%out, "( A, ' X ', /, ( 5ES12.4 ) )" )                     &
           prefix, nlp%X( : nlp%n )
         WRITE( data%out, "( A, ' C ', /, ( 5ES12.4 ) )" )                     &
           prefix, nlp%C( : nlp%m )
       END IF

!  obtain the gradient of the objective function and the Jacobian
!  of the constraints. The data is stored in a sparse format

       IF ( data%accepted ) THEN
         inform%gj_eval = inform%gj_eval + 1
         IF ( data%reverse_gj ) THEN
           data%branch = 310 ; inform%status = 3 ; RETURN
         ELSE
           CALL eval_SGJ( data%eval_status, nlp%X, userdata, nlp%Go%val,       &
                          nlp%J%val )
           IF ( data%eval_status /= 0 ) THEN
             inform%bad_eval = 'eval_SGJ'
             inform%status = GALAHAD_error_evaluation ; GO TO 900
           END IF
         END IF
       END IF

!  return from reverse communication to obtain the gradient and Jacobian

  310  CONTINUE
!    write( 6, "( ' sparse gradient ( ind, val )', /, ( 3( I6, ES12.4 ) ) )" ) &
!      ( nlp%Go%ind( i ), nlp%Go%val( i ), i = 1, nlp%Go%ne )

!  compute the gradient of the Lagrangian

       nlp%gL( : nlp%n ) = nlp%Z( : nlp%n )
       DO i = 1, nlp%Go%ne
         j = nlp%Go%ind( i )
         nlp%gL( j ) = nlp%gL( j ) + nlp%Go%val( i )
       END DO
       CALL mop_AX( one, nlp%J, nlp%Y, one, nlp%gL, transpose = .TRUE.,        &
                    m_matrix = nlp%m, n_matrix = nlp%n )
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &


!      WRITE(6,*) ' gl, y ', maxval( nlp%gL ), maxval( nlp%Y )
!      WRITE( data%out, "(A, /, ( 4ES20.12 ) )" ) ' gl_after ',  nlp%gl

!write(6,*) ' y ', nlp%Y( : nlp%m )
       inform%obj = nlp%f
       inform%dual_infeasibility =                                             &
         OPT_dual_infeasibility( nlp%n, nlp%gL( : nlp%n ) ) / multiplier_norm
!write(6,*) ' gl ', nlp%gL( : nlp%n )

!  ---------------------
!  check for termination
!  ---------------------

!  exit if an approximate KKT point has been found

       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            inform%dual_infeasibility <= data%stop_d .AND.                     &
            inform%complementary_slackness <= data%stop_c ) THEN
         IF ( data%printt ) WRITE( data%out,                                   &
                "( /, A, ' Termination criteria satisfied ' )" ) prefix
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  print details of the current iteration

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR.                                 &
              data%print_1st_header) WRITE( data%out, 2000 ) prefix
         data%print_1st_header = .FALSE.
         data%print_iteration_header = .FALSE.
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out, 2010 )                                             &
             prefix, inform%iter, inform%obj,                                  &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness, data%s_norm,                      &
             inform%target, inform%NLS_inform%iter, data%clock_now
         ELSE
           WRITE( data%out, 2020 ) prefix, inform%iter,                        &
             inform%obj, inform%primal_infeasibility,                          &
             inform%dual_infeasibility, inform%complementary_slackness,        &
             inform%target, inform%NLS_inform%iter, data%clock_now
         END IF

!        IF ( mod( inform%iter, 200 ) == 0 ) WRITE( data%out, 2100 )
!        WRITE( data%out, 2110 ) inform%iter, data%p_mode, nlp%f,              &
!          data%primal_viol, data%comp_viol, data%sigma
       END IF

!  ----------------------------------
!  check for unsuccessful termination
!  ----------------------------------

!  exit if the iteration limit has been exceeded

       inform%iter = inform%iter + 1

       IF ( inform%iter > data%control%maxit ) THEN
         IF ( data%printi )                                                    &
           WRITE( data%out, "( /, A, ' Iteration limit exceeded ' )" ) prefix
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  exit if the elapsed-time limit has been exceeded

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
             REAL( data%time_now - data%time_start, rp_ )                      &
               > data%control%cpu_time_limit ) .OR.                            &
             ( data%control%clock_time_limit >= zero .AND.                     &
               data%clock_now - data%clock_start                               &
              > data%control%clock_time_limit ) ) THEN

         IF ( data%printi )                                                    &
           WRITE( data%out, "( /, A, ' Time limit exceeded ' )" ) prefix
         inform%status = GALAHAD_error_time_limit ; GO TO 900
       END IF

!  compute the Hessian

!       inform%h_eval = inform%h_eval + 1
!       IF ( data%accepted ) THEN
!!        data%WORK_m( : nlp%m ) = nlp%Y( : nlp%m )  ! temporary copy
!         IF ( data%control%scale_constraints > 0 )                            &
!           nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) / data%C_scale( : nlp%m )

!         IF ( data%reverse_hj ) THEN
!            data%branch = 340 ; inform%status = 4 ; RETURN
!         ELSE
!            CALL eval_HJ( data%eval_status, nlp%X, y0, nlp%Y, userdata,       &
!                          nlp%H%val )
!            IF ( data%eval_status /= 0 ) THEN
!              inform%bad_eval = 'eval_HJ'
!              inform%status = GALAHAD_error_evaluation ; GO TO 900
!            END IF
!         END IF
!       END IF

!  return from reverse communication to obtain the Hessian of the Lagrangian

! 340  CONTINUE

!  solve the target problem: find x(t) = (local) argmin 1/2||c(x),f)x)-t||^2
!  using the the nls solver.  All function evlautions are for the vector of
!  residuals (c(x),f(x)-t)

       inform%NLS_inform%status = 1
       data%nls%X( : data%nls%n ) = nlp%X
!      WRITE(6,*) ' x ', nlp%X
       data%control%NLS_control%jacobian_available = 2
       data%control%NLS_control%subproblem_control%jacobian_available = 2
       data%control%NLS_control%hessian_available = 2
       data%got_fc = .TRUE. ; data%got_gj = .TRUE.

write(6,*) ' t ', inform%target
write(6,*) ' X ', nlp%X( : nlp%n )

!      DO        ! loop to solve problem
   400 CONTINUE  ! mock loop to solve problem

!  call the least-squares solver to find x(t)

         CALL NLS_solve( data%nls, data%control%NLS_control,                   &
                         inform%NLS_inform, data%NLS_data, data%NLS_userdata )

!  respond to requests for further details

         SELECT CASE ( inform%NLS_inform%status )

!  obtain the residuals

         CASE ( 2 )
           IF ( data%reverse_fc ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 400 ; inform%status = 2 ; RETURN
           ELSE
             IF ( data%got_fc ) THEN ! skip eval_fc if f and c are already known
               data%got_fc = .FALSE.
             ELSE
               CALL eval_FC( data%eval_status, data%nls%X, userdata,           &
                             nlp%f, nlp%C )
             END IF
             data%discrepancy = nlp%f - inform%target
             data%nls%C( : nlp%m ) = nlp%C( : nlp%m )
             data%nls%C( data%nls%m ) = data%discrepancy
             IF ( print_debug ) WRITE(6,"( ' nls%C = ', /, ( 5ES12.4 ) )" )    &
               data%nls%C( : data%nls%m )
           END IF

!  obtain the Jacobian

         CASE ( 3 )
           IF ( data%reverse_gj ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 400 ; inform%status = 4 ; RETURN
           ELSE
             IF ( data%got_gj ) THEN ! skip eval_SGJ if g & j are already known
               data%got_gj = .FALSE.
             ELSE
               CALL eval_SGJ( data%eval_status, data%nls%X, userdata,          &
                              nlp%Go%val, nlp%J%val )
             END IF
             SELECT CASE ( SMT_get( data%nls%J%type ) )
             CASE ( 'DENSE' )
               j_ne = 0 ; l = 0
               DO i = 1, nlp%m  ! from each row of J(x) in turn
                 data%nls%J%val( j_ne + 1 : j_ne + nlp%n )                     &
                   = nlp%J%val( l + 1 : l + nlp%n )
                 j_ne = j_ne + data%nls%n ; l = l + nlp%n
               END DO
               IF ( nlp%Go%sparse ) THEN ! from g(x)
                 DO i = 1, nlp%Go%ne
                   data%nls%J%val( j_ne + nlp%Go%ind( i ) ) = nlp%Go%val( i )
                 END DO
               ELSE
                 data%nls%J%val( j_ne + 1 : j_ne + nlp%n )                     &
                   = nlp%Go%val( 1 : nlp%n )
               END IF
             CASE ( 'SPARSE_BY_ROWS' )
               DO i = 1, nlp%m  ! from each row of J(x) in turn
                 j_ne = data%nls%J%ptr( i ) - 1
                 DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
                   j_ne = j_ne + 1
                   data%nls%J%val( j_ne ) = nlp%J%val( l )
                 END DO
               END DO
               j_ne = data%nls%J%ptr( data%nls%m ) - 1
               DO j = 1, nlp%Go%ne
                 j_ne = j_ne + 1
                 data%nls%J%val( j_ne ) = nlp%Go%val( j )
               END DO
             CASE ( 'COORDINATE' )
               data%nls%J%val( : nlp%J%ne ) = nlp%J%val( : nlp%J%ne ) ! from J
               j_ne = nlp%J%ne + data%n_slacks
               DO j = 1, nlp%Go%ne
                 j_ne = j_ne + 1
                 data%nls%J%val( j_ne ) = nlp%Go%val( j )
               END DO
               IF ( print_debug ) WRITE(6,"( ' nls%J', / ( 3( 2I6, ES12.4 )))")&
                ( data%nls%J%row( i ), data%nls%J%col( i ),                    &
                  data%nls%J%val( i ), i = 1, data%nls%J%ne )
             END SELECT
           END IF

!  obtain the Hessian

         CASE ( 4 )
           IF ( data%reverse_hj ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 400 ; inform%status = 6 ; RETURN
           ELSE
             CALL eval_HJ( data%eval_status, data%nls%X,                       &
                           data%NLS_data%Y( data%nls%m ),                      &
                           data%NLS_data%Y( 1 : nlp%m ),                       &
                           userdata, data%nls%H%val )
             IF ( print_debug ) WRITE(6,"( ' nls%H', / ( 3( 2I6, ES12.4 )))" ) &
              ( data%nls%H%row( i ), data%nls%H%col( i ),                      &
                data%nls%H%val( i ), i = 1, data%nls%H%ne )
           END IF

!  form a Jacobian-vector product

         CASE ( 5 )
           write(6,*) ' no nls_status = 5 as yet, stopping'
           stop
!          CALL JACPROD( data%eval_status, data%nls%X,                         &
!                        data%NLS_userdata, data%NLS_data%transpose,           &
!                        data%NLS_data%U, data%NLS_data%V )

!  form a Hessian-vector product

         CASE ( 6 )
           write(6,*) ' no nls_status = 6 as yet, stopping'
           stop
!          CALL HESSPROD( data%eval_status, data%nls%X, data%NLS_data%Y,       &
!                         data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  form residual Hessian-vector products

         CASE ( 7 )
           IF ( data%reverse_hocprods ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 400 ; inform%status = 8 ; RETURN
           ELSE
             CALL eval_HOCPRODS( data%eval_status, data%nls%X,                 &
                                 data%nls_data%V, userdata,                    &
                                 data%nls%P%val( nlp%P%ne + 1 :                &
                                                 data%nls%P%ne ),              &
                                 data%nls%P%val( 1 : nlp%P%ne ) )
             IF ( print_debug ) THEN
               WRITE(6,"( ' nls%P' )" )
               DO j = 1, data%nls%n
                 WRITE(6,"( 'column ',  I0, /, / ( 4( I6, ES12.4 ) ) )" ) &
                   j, ( data%nls%P%row( i ), data%nls%P%val( i ), i = &
                        data%nls%P%ptr( j ), data%nls%P%ptr( j + 1 ) - 1 )
               END DO
             END IF
           END IF

!  apply the preconditioner

         CASE ( 8 )
           write(6,*) ' no nls_status = 8 as yet, stopping'
           stop
!          CALL SCALE( data%eval_status, data%nls%X,                           &
!                      data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  terminal exit from the minimization loop

         CASE DEFAULT
           nlp%X = data%nls%X( : data%nls%n )
           GO TO 390
         END SELECT
         GO TO 400
!      END DO ! end of loop to solve the target problem

   390  CONTINUE
        inform%primal_infeasibility = TWO_NORM( nlp%C )
!write(6,*) ' t - f_t', inform%target - nlp%f
!write(6,*) ' c/(f-t) ', nlp%C( : nlp%m ) / ( nlp%f - inform%target )

!write(6,*) ' t, Phi ', inform%target, SQRT( two * inform%NLS_inform%obj )

       data%discrepancy = nlp%f - inform%target
       data%V( : nlp%n ) = - nlp%Gl( : nlp%n )
       data%V( data%np1 : data%npm ) = data%discrepancy - nlp%C( : nlp%m )
       data%rnorm = TWO_NORM( data%V( : data%npm ) )
!write(6,*) ' old discrepancy, t ', data%discrepancy, inform%target
!write(6,*) ' old ||r|| = ', data%rnorm


!  -----------------
!  adjust the target
!  -----------------

!  the current target point is feasible

       IF ( inform%primal_infeasibility <= data%stop_p .AND.                   &
            inform%target <= nlp%f ) THEN
!      IF ( inform%primal_infeasibility <= ten ** ( - 5 ) ) THEN
!write(6,*) ' infeas, bracket t - f_t', inform%primal_infeasibility, data%target_bracketed,  inform%target - nlp%f

!  the target is already bracketed, improve the upper bound

         IF ( data%target_bracketed ) THEN
           data%target_upper = inform%target

!  if the target interval is small enough, stop

           IF ( data%target_upper - data%target_lower <                        &
                  data%control%small_bracket_tol ) THEN
             WRITE( 6, "( ' bracket, tol ', 2ES12.4 )" ) data%target_upper -   &
               data%target_lower, data%control%small_bracket_tol
             data%converged = .TRUE.

!  otherwise try the mid point of the interval as the next target

           ELSE
             frac = 0.9_rp_
             inform%target = data%target_lower + frac *                        &
                              ( data%target_upper - data%target_lower )
           END IF

!  the target is not yet bracketed, so continue to move left

         ELSE
           inform%target = inform%target - two ** ( inform%iter - 1 )
         END IF

!  the target point is infeasible

       ELSE

!  set the multiplier estimates

         data%discrepancy = nlp%f - inform%target
         nlp%Y( : nlp%m ) = nlp%C( : nlp%m ) / data%discrepancy
!write(6,*) ' x ',  nlp%X( : nlp%n )
!write(6,*) ' y ',  nlp%Y( : nlp%m )
!write(6,*) ' discrepancy ', data%discrepancy

!  the new target gives a lower bound, and thus the target is now bounded

         data%target_lower = inform%target
         data%target_bracketed = .TRUE.
         data%from_left = .TRUE.

!  if the target interval is small enough, stop

         IF ( data%target_upper - data%target_lower <                          &
                data%control%small_bracket_tol ) THEN
           WRITE( 6, "( ' bracket, tol ', 2ES12.4 )" ) data%target_upper -     &
             data%target_lower, data%control%small_bracket_tol
           data%converged = .TRUE.

!  move the target right, towwards the minimizer

         ELSE
           data%nt = MIN( data%nt + 1, 2 )
!          data%nt = MIN( data%nt + 1, 3 )

!  single point to the left of the minimizer

           IF ( data%nt == 1 ) THEN
             data%tu = inform%target
             data%phiu = SQRT( two * inform%NLS_inform%obj )
             data%cu = inform%primal_infeasibility
!            inform%target = half * ( data%target_upper + data%target_lower )
             inform%target = inform%target + 0.001_rp_

!  two points to the left of the minimizer, linear interpolate to get next
!  target: find where linear fit through (tm,vm) and (tu,vu) intersects the
!  t axis

!  fit l(t) = ao + a1 t
!   vm = ao + a1 tm
!   vu = ao + a1 tu
!  =>
!   a1 = (vu - vm ) / ( tu - t1 )
!   a0 = vm - a1 tm
!
!   l(t*) = 0 =>
!   t* = - a0 / a1 = tm - vm / a1
!      = ( tm (vu - vm ) - ( tu - t1 ) vm ) / (vu - vm )
!      = ( tm vu - tu vm ) / (vu - vm )

           ELSE IF ( data%nt == 2 ) THEN
             data%tm = data%tu ; data%phim = data%phiu ; data%cm = data%cu
             data%tu = inform%target
             data%phiu = SQRT( two * inform%NLS_inform%obj )
             data%cu = inform%primal_infeasibility
             t_phi = ( data%tm * data%phiu - data%tu * data%phim ) /          &
                     ( data%phiu - data%phim )
             t_c = ( data%tm * data%cu - data%tu * data%cm ) /                &
                   ( data%cu - data%cm )

!            inform%target = inform%target + control%target_fraction *         &
!              ( MIN( data%target_upper, t_phi ) - inform%target )
             IF ( t_phi <= t_c ) THEN
               WRITE(6,"( ' t_* in [', ES22.14, ',', ES22.14, ']' )" )         &
                 t_phi, t_c
               inform%target = t_phi
               data%target_upper = t_c
             ELSE
               WRITE(6,"( ' t_* in [', ES22.14, ',', ES22.14, ']' )" )         &
                 t_c, t_phi
               inform%target = t_c
               data%target_upper = t_phi
             END IF

!  three (or more) points to the left of the minimizer, quadratic interpolate
!  to get the next target: find where quadatic fit through (tl,vl), (tm,vm)
!  and (tu,vu) intersects the t axis

!  fit q(t) = ao + a1 t + a2 t^2 given
!   vl = ao + a1 tl + a2 tl^2
!   vm = ao + a1 tm + a2 tm^2
!   vu = ao + a1 tu + a2 tu^2
!  =>
!   dl = ( vm - vl ) / ( tm - tl ) = a1 + a2 ( tm + t1 )
!   dm = ( vu - vm ) / ( tu - tm ) = a1 + a2 ( tu + tm )
!   a2 = ( dm - dl ) / ( tu - t1 )
!   a1 = dl - a2 ( tm + t1 )
!   a0 = vl - tl ( a1 + a2 tl )

!  solve q(t*) = 0, t* = largest root

           ELSE
             data%tl = data%tm ; data%phil = data%phim ; data%cl = data%cm
             data%tm = data%tu ; data%phim = data%phiu ; data%cm = data%cu
             data%tu = inform%target
             data%phiu = SQRT( two * inform%NLS_inform%obj )
             data%cu = inform%primal_infeasibility

             dl = ( data%phim - data%phil ) / ( data%tm - data%tl )
             dm = ( data%phiu - data%phim ) / ( data%tu - data%tm )
             a2 = ( dm - dl ) / ( data%tu - data%tl )
             a1 = dl - a2 * ( data%tm + data%tl )
             a0 = data%phil - data%tl * ( a1 + a2 * data%tl)

!write(6,*) ' a0, a1, a2 ', a0, a1, a2
!write(6,"(A, 2ES24.16)") 'l', data%tl, data%phil
!write(6,"(A, 2ES24.16)") 'm', data%tm, data%phim
!write(6,"(A, 2ES24.16)") 'u', data%tu, data%phiu

             CALL ROOTS_quadratic( a0, a1, a2, root_tol,                       &
                                   nroots, root1, root2, debug = .FALSE. )
             t_phi = root2
             write(6,*) ' q t_phi, nroots ', t_phi, nroots

             inform%target = inform%target + control%target_fraction *         &
               ( MIN( data%target_upper, t_phi ) - inform%target )
           END IF

!  two points to the left of the minimizer, quadratic interpolate to get next
!  target: find where quadratic fit through (tm,vm) and (tu,vu) with vurvature
!  -sigma at tu intersects the t axis

!  fit q(t) = a + b t - sigma ( t -tu )^2 given
!   vm = a + b tm - sigma ( tm - tu )^2
!   vu = a + b tu
!  =>
!   vu - vm = b ( tu - tm ) + sigma ( tm - tu )( tu + tm )^2
!   b = (vu - vm ) / ( tu - tm ) - sigma( tm + tu )
!   a = vu - b tu

!  q(t) = ao + a1 t + a2 t^2
!       = ( a - sigma tu^2 ) + ( b + 2 sigma tu ) t - sigma t^2

!  a1 = b + 2 sigma tu = (vu - vm ) / ( tu - tm ) + sigma( tu - tm )
!  a0 = vu - tu [ (vu - vm ) / ( tu - tm ) - sigma( tm + tu ) ]
!     = [ vu ( tu - tm ) - tu (vu - vm ) ] / ( tu - tm ) + sigma( tm + tu ) tu
!     = ( tu vm - vu tm ) / ( tu - tm ) + sigma( tm + tu ) tu

!  solve q(t*) = 0, t* = largest root

         END IF
       END IF

       IF ( data%printt ) WRITE( data%out, 2070 ) data%target_lower,           &
                            inform%target, data%target_upper, nlp%f,           &
                            inform%primal_infeasibility, inform%NLS_inform%iter
       IF ( data%printm ) WRITE( data%out, 2080 ) nlp%X

       IF ( data%converged ) THEN
         write(6,*) ' forcing stop'
         stop
       END IF

!  set print agenda for the next iteration

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print ) THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%print_level = data%control%print_level
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE.
         data%printd = .FALSE. ; data%print_level = 0
       END IF

       data%print_iteration_header = data%print_level > 1 .OR.                 &
         data%control%NLS_control%print_level > 0

       IF ( data%printd ) THEN
         WRITE( data%out,"( '     X_l           X         X_u' )" )
         DO i = 1, nlp%n
         WRITE( data%out,"( 3ES12.4 )" ) nlp%X_l(i), nlp%X(i), nlp%X_u(i)
         END DO
       END IF

!  obtain a better starting point:

!  find
!
!    ( dx, dy ) = ( dx_1, dy_1 ) + alpha ( dx_2, dy_2 )
!
!  where
!
!    alpha = dx_1^T g(x) / ( 1 - dx_2^T g(x) ),
!
!  ( H(x,y)    J^T(x)    ) ( dx_1 ) = - (   g(x) + J^T(x) y   ) = r(x,y,t) and
!  (  J(x)  - (f(x)-t) I ) ( dy_1 )     ( c(x) - (f(x) - t) y )
!
!  ( H(x,y)    J^T(x)    ) ( dx_2 ) = ( 0 )
!  (  J(x)  - (f(x)-t) I ) ( dy_2 )   ( y )

!! ** TO DO

!  compute the discrepamcy f(x)- t and residual r(x,y,t)

!write(6,*) ' -------start------------'
       data%discrepancy = nlp%f - inform%target
!write(6,*) ' discrepancy, t ', data%discrepancy, inform%target
!write(6,*) ' x ',  nlp%X( : nlp%n )
!write(6,*) ' y ',  nlp%Y( : nlp%m )
       data%V( : nlp%n ) = - nlp%Gl( : nlp%n )
       data%V( data%np1 : data%npm )                                           &
         = data%discrepancy * nlp%Y( : nlp%m ) - nlp%C( : nlp%m )
       data%rnorm = TWO_NORM( data%V( : data%npm ) )
!write(6,*) ' ||r||, r_start = ', data%rnorm, data%control%advanced_start

!  check whether to attempt an advanced starting point

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
         IF ( print_debug ) WRITE(6,"( ' nls%H', / ( 3( 2I6, ES12.4 )))" )   &
          ( nlp%H%row( i ), nlp%H%col( i ), nlp%H%val( i ), i = 1, nlp%H%ne )
       END IF

!  form and factorize the matrix

!    ( H(x,y)    J^T(x)    )
!    (  J(x)  - (f(x)-t) I )

  420  CONTINUE
       data%C_ssls%val( 1 ) = data%discrepancy
       CALL SSLS_factorize( nlp%n, nlp%m, nlp%H, nlp%J, data%C_ssls,           &
                            data%SSLS_data, data%control%SSLS_control,         &
                            inform%SSLS_inform )
!write(6,*) ' ssls status ', inform%SSLS_inform%status

!write(6,"( 'K = ')" )
do i = 1, data%SSLS_data%K%ne
! write(6,"( 2I7, ES12.4 )" ) data%SSLS_data%K%row(i),data%SSLS_data%K%col(i), &
!                             data%SSLS_data%K%val(i)
end do

!  find v = ( dx1, dy1 )

       CALL SSLS_solve( nlp%n, nlp%m, data%V, data%SSLS_data,                  &
                        data%control%SSLS_control,inform%SSLS_inform )
!write(6,*) ' ||v|| ', TWO_NORM( data%V( : nlp%n ) )
!write(6,*) ' ||g|| ', TWO_NORM( nlp%Go%val( : nlp%Go%ne ) )

!  find v2 = ( dx2, dy2 )

       data%V2( : nlp%n ) = zero
       data%V2( data%np1 : data%npm ) = nlp%Y
       CALL SSLS_solve( nlp%n, nlp%m, data%V2, data%SSLS_data,                 &
                        data%control%SSLS_control, inform%SSLS_inform )
!write(6,*) ' ||v2|| ', TWO_NORM( data%V2( : nlp%n ) )

!write(6,*) ' v ', data%V( : nlp%n )
!write(6,*) ' v2 ',data%V2( : nlp%n )
!write(6,*) ' g ', nlp%Go%val( : nlp%Go%ne )


!  compute alpha

       IF ( nlp%Go%sparse ) THEN ! from g(x)
         num = zero ; den = one
         DO i = 1, nlp%Go%ne
           j = nlp%Go%ind( i )
           num = num + nlp%Go%val( i ) * data%V( j )
           den = den - nlp%Go%val( i ) * data%V2( j )
         END DO
       ELSE
         num = DOT_PRODUCT( data%V( : nlp%n ), nlp%Go%val( : nlp%n ) )
         den = one - DOT_PRODUCT( data%V2( : nlp%n ), nlp%Go%val( : nlp%n ) )
       END IF
       alpha = num / den

!write(6,*) ' alpha ', alpha
!  recover v = ( dx, dy )

       data%V( : data%npm )                                                    &
         = data%V( : data%npm ) + alpha * data%V2( : data%npm )

!  make a copy of x, y, c, g and J in case the advanced starting point is poor

       data%f_old = nlp%f
       data%X_old( : nlp%n ) = nlp%X( : nlp%n )
       data%Y_old( : nlp%m ) = nlp%Y( : nlp%m )
       data%C_old( : nlp%m ) = nlp%C( : nlp%m )
       data%G_old( : nlp%Go%ne ) = nlp%Go%val( : nlp%Go%ne )
       data%J_old( : data%J_ne ) = nlp%J%val( : data%J_ne )

!  compute the trial x and y

       nlp%X( : nlp%n ) = nlp%X( : nlp%n ) + data%V( : nlp%n )
       nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) + data%V( data%np1 : data%npm )
!write(6,*) ' new x ',  nlp%X( : nlp%n )
!write(6,*) ' new y ',  nlp%Y( : nlp%m )

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
         CALL eval_SGJ( data%eval_status, nlp%X, userdata,                     &
                        nlp%Go%val, nlp%J%val )
       END IF

!  compute the gradient of the Lagrangian

  440  CONTINUE
       nlp%gL( : nlp%n ) = nlp%Z( : nlp%n )
       DO i = 1, nlp%Go%ne
         j = nlp%Go%ind( i )
         nlp%gL( j ) = nlp%gL( j ) + nlp%Go%val( i )
       END DO
       CALL mop_AX( one, nlp%J, nlp%Y, one, nlp%gL, transpose = .TRUE.,        &
                    m_matrix = nlp%m, n_matrix = nlp%n )

!  compute the new discrepamcy f(x)- t and residual r(x,y,t)

       data%discrepancy = nlp%f - inform%target
!write(6,*) ' discrepancy, t ', data%discrepancy, inform%target
       data%V( : nlp%n ) = - nlp%Gl( : nlp%n )
       data%V( data%np1 : data%npm )                                           &
         = data%discrepancy * nlp%Y( : nlp%m ) - nlp%C( : nlp%m )
       data%rnorm_old = data%rnorm
       data%rnorm = TWO_NORM( data%V( : data%npm ) )
!write(6,*) ' ||r|| = ', data%rnorm

!  check to see if the residuals are sufficiently small

       IF ( data%rnorm < data%control%advanced_stop ) GO TO 500

!  check to see if the residuals have increased

       IF ( data%rnorm_old < data%rnorm ) THEN

!  make a copy of f, x, y, c, g & J in case the advanced starting point is poor

         nlp%f = data%f_old
         nlp%X( : nlp%n ) = data%X_old( : nlp%n )
         nlp%Y( : nlp%m ) = data%Y_old( : nlp%m )
         nlp%C( : nlp%m ) = data%C_old( : nlp%m )
         nlp%Go%val( : nlp%Go%ne ) = data%G_old( : nlp%Go%ne )
         nlp%J%val( : data%J_ne ) = data%J_old( : data%J_ne )
         GO TO 500

!  check to see if the advanced start iteration limit has been reasched

       ELSE
          IF ( data%iter_advanced > iter_advanced_max ) GO TO 500
       END IF
       data%iter_advanced = data%iter_advanced + 1
       GO TO 410

!  end of advanced starting point search

  500 CONTINUE
!write(6,*) ' --------end-------------'

!  ----------------------------------------------------------------------------
!                      END OF MAIN LOOP TARGET LOOP
!  ----------------------------------------------------------------------------

       GO TO 300
!    END DO

!  summarize the final results

 900 CONTINUE
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = REAL( data%time_now - data%time_start, rp_ )
     inform%time%clock_total = data%clock_now - data%clock_start

!  restore scaled-cnstraint data

     IF ( data%control%scale_constraints > 0 ) THEN
       nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) / data%C_scale( : nlp%m )
       nlp%C( : nlp%m ) = nlp%C( : nlp%m ) * data%C_scale( : nlp%m )
       WHERE ( nlp%C_l( : nlp%m ) > - data%control%infinity )                  &
         nlp%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) * data%C_scale( : nlp%m )
       WHERE ( nlp%C_u( : nlp%m ) < data%control%infinity )                    &
         nlp%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) * data%C_scale( : nlp%m )
     END IF

!  print details of the final iteration

     IF ( data%printi ) THEN
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         data%control%NLS_control%print_level > 0
       IF ( inform%status == GALAHAD_ok .OR.                                   &
            inform%status == GALAHAD_error_primal_infeasible ) THEN
         IF ( data%print_iteration_header .OR.                                 &
              data%print_1st_header) WRITE( data%out, 2000 ) prefix
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out, 2010 )                                             &
             prefix, inform%iter, inform%obj,                                  &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness, data%s_norm,                      &
             inform%target, inform%NLS_inform%iter, inform%time%clock_total
         ELSE
           WRITE( data%out, 2020 ) prefix, inform%iter, inform%obj,            &
             inform%primal_infeasibility, inform%dual_infeasibility,           &
             inform%complementary_slackness,                                   &
             inform%target, inform%NLS_inform%iter, inform%time%clock_total
         END IF

         IF ( inform%status == GALAHAD_ok ) THEN
           WRITE(  data%out,                                                   &
             "( /, A, ' Approximate locally-optimal solution found' )" ) prefix
         ELSE
           WRITE(  data%out,                                                   &
             "( /, A, ' Approximate infeasible critical point found' )" ) prefix
         END IF
       END IF

!      WRITE( data%out, "( A, ' +', 76( '-' ), '+' )" ) prefix
!      WRITE( data%out, "( A, ' Status: ', I0 )" ) prefix, inform%status
     END IF

!  print details of the solution

     IF ( data%control%print_level > 0 .AND. data%out > 0 ) THEN
       l = 2
       IF ( data%control%full_solution ) l = nlp%n
       IF ( data%control%print_level >= 10 ) l = nlp%n

       names = ALLOCATED( nlp%VNAMES )
       IF ( names ) THEN
         WRITE( data%out, "( /, A, ' Solution: ', /, A, '                   ', &
        &         '             <------ Bounds ------> ', /, A,                &
        &         '      # name          value   ',                            &
        &         '    Lower       Upper       Dual' )" ) prefix, prefix, prefix
       ELSE
         WRITE( data%out, "( /, A, ' Solution: ', /, A, '        ',            &
        &         '           <------ Bounds ------> ', /, A,                  &
        &         '      #    value   ',                                       &
        &         '    Lower       Upper       Dual' )" ) prefix, prefix, prefix
       END IF
       DO j = 1, 2
         IF ( j == 1 ) THEN
           ir = 1 ; ic = MIN( l, nlp%n )
         ELSE
           IF ( names ) THEN
             IF ( ic < nlp%n - l ) WRITE( data%out, 2040 ) prefix
           ELSE
             IF ( ic < nlp%n - l ) WRITE( data%out, 2060 ) prefix
           END IF
           ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
         END IF
         IF ( names ) THEN
           DO i = ir, ic
             WRITE( data%out, 2030 ) prefix, i, nlp%VNAMES( i ), nlp%X( i ),   &
               nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
           END DO
         ELSE
           DO i = ir, ic
             WRITE( data%out, 2050 ) prefix, i, nlp%X( i ),                    &
               nlp%X_l( i ), nlp%X_u( i ), nlp%Z( i )
           END DO
         END IF
       END DO

       IF ( nlp%m > 0 ) THEN
         l = 2
         IF ( data%control%full_solution ) l = nlp%m
         IF ( data%control%print_level >= 10 ) l = nlp%m

         names = ALLOCATED( nlp%CNAMES )
         IF ( names ) THEN
           WRITE( data%out, "( /, A, ' Constraints:', /, A, '              ',  &
          &       '                    <------ Bounds ------> ', /, A,         &
          &       '      # name           value       ',                       &
          &       'Lower       Upper    Multiplier' )" ) prefix, prefix, prefix
         ELSE
           WRITE( data%out, "( /, A, ' Constraints:', /, A, '              ',  &
          &       '         <------ Bounds ------> ', /, A,                    &
          &       '      #     value       ',                                  &
          &       'Lower       Upper    Multiplier' )" ) prefix, prefix, prefix
         END IF
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1 ; ic = MIN( l, nlp%m )
           ELSE
             IF ( names ) THEN
               IF ( ic < nlp%m - l ) WRITE( data%out, 2040 ) prefix
             ELSE
               IF ( ic < nlp%m - l ) WRITE( data%out, 2060 ) prefix
             END IF
             ir = MAX( ic + 1, nlp%m - ic + 1 ) ; ic = nlp%m
           END IF
           IF ( names ) THEN
             DO i = ir, ic
               WRITE( data%out, 2030 ) prefix, i, nlp%CNAMES( i ), nlp%C( i ), &
                 nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
             END DO
           ELSE
             DO i = ir, ic
               WRITE( data%out, 2050 ) prefix, i, nlp%C( i ),                  &
                 nlp%C_l( i ), nlp%C_u( i ), nlp%Y( i )
             END DO
           END IF
         END DO
       END IF

       IF ( nlp%m > 0 ) THEN
         multiplier_norm = MAXVAL( ABS( nlp%Y( : nlp%m ) ) )
       ELSE
         multiplier_norm = zero
       END IF

       WRITE( data%out, "( /, A, ' Problem: ', A10, 17X,                       &
      &          ' Solver: COLT', /, A,                                        &
      &  ' n              =     ', bn, I11,                                    &
      &  '     m               =', bn, I11, /,                                 &
      & A, ' Objective      = ', ES15.8,                                       &
      &    '     Complementarity =', ES11.4, /,                                &
      & A, ' Violation      =     ', ES11.4,                                   &
      &    '     Dual infeas.    =', ES11.4, /,                                &
      & A, ' Max multiplier =     ', ES11.4,                                   &
      &    '     Max dual var.   =', ES11.4, /,                                &
      & A, ' Iterations     =     ', bn, I11,                                  &
      &    '     Time            =', F11.2 , /,                                &
      & A, ' Function evals =     ', bn, I11 )" )                              &
        prefix, nlp%pname, prefix, nlp%n, nlp%m, prefix, inform%obj,           &
        inform%complementary_slackness, prefix, inform%primal_infeasibility,   &
        inform%dual_infeasibility, prefix, multiplier_norm,                    &
        MAXVAL( ABS( nlp%Z( : nlp%n ) ) ), prefix, inform%iter,                &
        inform%time%clock_total, prefix, inform%fc_eval
     END IF

     IF ( data%control%error > 0 .AND. data%control%print_level > 0 ) THEN
       SELECT CASE ( inform%status )
       CASE( - 111 )
         WRITE( data%control%error, "( /, A, ' Error return from ', A, ' -',   &
        & /, A, '  There are no general constraints' )" )                      &
          prefix, 'Colt_solve', prefix
       CASE DEFAULT
         CALL SYMBOLS_status( inform%status, data%control%error, prefix,       &
                              'Colt_solve' )
       END SELECT
     END IF
     RETURN

!  -------------
!  Error returns
!  -------------

!  allocation errors

 910 CONTINUE
     inform%status = GALAHAD_error_allocate
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = REAL( data%time_now - data%time_start, rp_ )
     inform%time%clock_total = data%clock_now - data%clock_start
     RETURN

!  non-executable statements

 2000 FORMAT( /, A, ' iter     obj fun   pr_feas du_feas cmp_slk ',            &
                    '  step      target     its   time' )
 2010 FORMAT( A, I5, ES14.6, 3ES8.1, ES8.1, ES14.6, I4, F8.1 )
 2020 FORMAT( A, I5, ES14.6, 3ES8.1, '     -  ', ES14.6, I4, F8.1 )
 2030 FORMAT( A, I7, 1X, A10, 4ES12.4 )
 2040 FORMAT( A, 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2050 FORMAT( A, I7, 4ES12.4 )
 2060 FORMAT( A, 6X, '.', 4( 2X, 10( '.' ) ) )
 2070 FORMAT( '      lower         new target         upper     ',             &
              '       f          ||c||   eval', /, 4ES16.8, ES9.2, I6 )
 2080 FORMAT( ' X = ', 5ES12.4, /, ( 5X,  5ES12.4 ) )


!  end of subroutine COLT_solve

     END SUBROUTINE COLT_solve

!-  G A L A H A D -  C O L T _ s e t u p _ p r o b l e m  S U B R O U T I N E  -

     SUBROUTINE COLT_setup_problem( nlp, nls, nls_dims, control, inform,       &
                                    n_slacks, SLACKS, h_available, p_available )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  transfer the data for the constrained optimization problem (nlp)
!     min f(x): c(x) = c, c_l <= c <= c_u, x_l <= x <= x_u
!  to that for the least-squares problem (nls)
!     min 1/2 || (c(x)-c,f(x)-t) ||^2 :  c_l <= c <= c_u, x_l <= x <= x_u
!  for given t

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  **NB  currently, all constraints are equations, and variables are free **
!  nlsi is nls without f-t component, only sizes components recorded

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( IN ) :: nlp
     TYPE ( NLPT_problem_type ), INTENT( OUT ) :: nls
     TYPE ( COLT_nls_dims_type ) :: nls_dims
     TYPE ( COLT_control_type ), INTENT( IN ) :: control
     TYPE ( COLT_inform_type ), INTENT( INOUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT) :: n_slacks
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ),                      &
                             INTENT( OUT ) :: SLACKS
     LOGICAL, OPTIONAL, INTENT( IN ) :: h_available, p_available

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, j, j_ne, l, len_array, lchp,lohp, nnzohp
     LOGICAL :: set_h, set_p
     CHARACTER ( LEN = 80 ) :: array_name

!  count the number of slack variables

     n_slacks = 0
     DO i = 1, nlp%m
       IF ( ( nlp%C_l( i ) > - control%infinity .OR.                           &
              nlp%C_u( i ) < control%infinity ) .AND.                          &
              nlp%C_l( i ) < nlp%C_u( i ) ) THEN
          n_slacks = n_slacks + 1
       END IF
     END DO

!  allocate space to hold the indices of constraints that have slack variables

     array_name = 'colt: SLACKS'
     CALL SPACE_resize_array( n_slacks, SLACKS,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  ... and record the indices

     n_slacks = 0
     DO i = 1, nlp%m
       IF ( ( nlp%C_l( i ) > - control%infinity .OR.                           &
              nlp%C_u( i ) < control%infinity ) .AND.                          &
              nlp%C_l( i ) < nlp%C_u( i ) ) THEN
          n_slacks = n_slacks + 1
          SLACKS( n_slacks ) = i
       END IF
     END DO

!  set the numbers of variables and resdiduals for nls

     nls%n = nlp%n + n_slacks ; nls%m = nlp%m + 1
     nls_dims%n = nls%n ; nls_dims%m = nls%m
     nls_dims%n_f = nls%n ; nls_dims%m_f = nlp%m

!  allocate space for the variables, residuals and gradients for nls

     array_name = 'colt: nls%X'
     CALL SPACE_resize_array( nls%n, nls%X,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: nls%C'
     CALL SPACE_resize_array( nls%m, nls%C,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: nls%G'
     CALL SPACE_resize_array( nls%n, nls%G,                                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  allocate space for the bounds and assign values

     array_name = 'colt: nls%X_l'
     CALL SPACE_resize_array( nls%n, nls%X_l,                                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: nls%X_u'
     CALL SPACE_resize_array( nls%n, nls%X_u,                                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     nls%X_l( : nlp%n ) = nlp%X_l( : nlp%n )
     nls%X_u( : nlp%n ) = nlp%X_u( : nlp%n )
     j = nlp%n
     DO i = 1, nlp%m
       IF ( ( nlp%C_l( i ) > - control%infinity .OR.                           &
              nlp%C_u( i ) < control%infinity ) .AND.                          &
              nlp%C_l( i ) < nlp%C_u( i ) ) THEN
          j = j + 1
          nls%X_l( j ) = nlp%C_l( i )
          nls%X_u( j ) = nlp%C_u( i )
       END IF
     END DO

!  set up space for the residual Jacobian (  J(x)  I_s ), where I_s is for the
!                                         ( g^T(x)  0  )  slack variables

     len_array = lEN_TRIM( SMT_get( nlp%J%type ) )
     array_name( 1 : len_array ) = SMT_get( nlp%J%type )
     CALL SMT_put( nls%J%type, array_name( 1 : len_array ),                    &
                   inform%alloc_status )
     IF ( inform%alloc_status /= 0 ) THEN
       inform%status = GALAHAD_error_allocate ; RETURN ; END IF

!  calculate the size of the resiual Jacobian

     nls%J%m = nls%m ; nls%J%n = nls%n
     SELECT CASE ( SMT_get( nls%J%type ) )
     CASE ( 'DENSE' )
       nls_dims%j_f_ne = nls_dims%m_f *  nls_dims%n_f
       nls_dims%j_ne = nls%J%m * nls%J%n
     CASE ( 'SPARSE_BY_ROWS' )
       nls_dims%j_f_ne = nlp%J%ptr( nlp%m + 1 ) - 1 + n_slacks
       nls_dims%j_ne =  nls_dims%j_f_ne + nlp%Go%ne
     CASE ( 'COORDINATE' )
       nls_dims%j_f_ne = nlp%J%ne + n_slacks
       nls_dims%j_ne =  nls_dims%j_f_ne + nlp%Go%ne
     END SELECT
     nls%J%ne = nls_dims%j_ne

!  make room for the values

     array_name = 'colt: nls%J%val'
     CALL SPACE_resize_array( nls%J%ne, nls%J%val,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  set up space for, and transfer, the sparsity pattern along with fixed values

     SELECT CASE ( SMT_get( nls%J%type ) )
     CASE ( 'DENSE' )

!  set fixed values

       j_ne = nlp%n ; j = 0
       DO i = 1, nlp%m
         nls%J%val( j_ne + 1 : j_ne + n_slacks ) = zero ! from I_s
         IF ( ( nlp%C_l( i ) > - control%infinity .OR.                         &
                nlp%C_u( i ) < control%infinity ) .AND.                        &
                 nlp%C_l( i ) < nlp%C_u( i ) ) THEN
           j = j + 1
           nls%J%val( j_ne + j ) = - one
         END IF
         j_ne = j_ne + nls%n
       END DO
       IF ( nlp%Go%sparse ) THEN ! from g^T and the following zero block
         nls%J%val( j_ne - nlp%n + 1 : j_ne + n_slacks ) = zero
       ELSE
         nls%J%val( j_ne + 1 : j_ne + n_slacks ) = zero
       END IF

     CASE ( 'SPARSE_BY_ROWS' )

!  assign space

       array_name = 'colt: nls%J%ptr'
       CALL SPACE_resize_array( nls%m + 1, nls%J%ptr,                          &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set pattern and fixed values

       j_ne = 0 ; j = nlp%n
       DO i = 1, nlp%m  ! from each row of J(x) in turn
         nls%J%ptr( i ) = j_ne + 1
         DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
           j_ne = j_ne + 1
           nls%J%col( j_ne ) = nlp%J%col( l )
         END DO
         IF ( ( nlp%C_l( i ) > - control%infinity .OR.                         &
                nlp%C_u( i ) < control%infinity ) .AND.                        &
                nlp%C_l( i ) < nlp%C_u( i ) ) THEN  ! from a row of I_s
           j = j + 1 ; j_ne = j_ne + 1
           nls%J%col( j_ne ) = j
           nls%J%val( j_ne ) = - one
        END IF
       END DO
       nls%J%ptr( nlp%m + 1 ) = j_ne + 1
       IF ( nlp%Go%sparse ) THEN ! from g(x)
         DO i = 1, nlp%Go%ne
           j = nlp%Go%ind( i ) ; j_ne = j_ne + 1
           nls%J%col( j_ne ) = j
         END DO
       ELSE
         DO j = 1, nlp%n
           j_ne = j_ne + 1
           nls%J%col( j_ne ) = j
         END DO
       END IF
       nls%J%ptr( nls%m + 1 ) = j_ne + 1

     CASE ( 'COORDINATE' )

!  assign space

       array_name = 'colt: nls%J%row'
       CALL SPACE_resize_array( nls%J%ne, nls%J%row,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'colt: nls%J%col'
       CALL SPACE_resize_array( nls%J%ne, nls%J%col,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

!  set pattern and fixed values

       nls%J%row( : nlp%J%ne ) = nlp%J%row( : nlp%J%ne ) ! from J(x)
       nls%J%col( : nlp%J%ne ) = nlp%J%col( : nlp%J%ne )
       j = nlp%n ; j_ne = nlp%J%ne
       DO i = 1, nlp%m
         IF ( ( nlp%C_l( i ) > - control%infinity .OR.                         &
                nlp%C_u( i ) < control%infinity ) .AND.                        &
                nlp%C_l( i ) < nlp%C_u( i ) ) THEN  ! from I_s
           j = j + 1 ; j_ne = j_ne + 1
           nls%J%row( j_ne ) = i
           nls%J%col( j_ne ) = j
           nls%J%val( j_ne ) = - one
         END IF
       END DO
       IF ( nlp%Go%sparse ) THEN ! from g(x)
         DO i = 1, nlp%Go%ne
           j = nlp%Go%ind( i ) ; j_ne = j_ne + 1
           nls%J%row( j_ne ) = nls%m
           nls%J%col( j_ne ) = j
         END DO
       ELSE
         DO j = 1, nlp%n
           j_ne = j_ne + 1
           nls%J%row( j_ne ) = nls%m
           nls%J%col( j_ne ) = j
         END DO
       END IF
     END SELECT

     IF ( SMT_get( nls%J%type ) == 'DENSE' ) THEN
     ELSE
       IF ( SMT_get( nls%J%type ) == 'SPARSE_BY_ROWS' ) THEN
       ELSE ! 'COORDINATE'
       END IF

!  transfer the spasity pattern

       SELECT CASE ( SMT_get( nls%J%type ) )
       CASE ( 'SPARSE_BY_ROWS' )
       CASE ( 'COORDINATE' )
       END SELECT
     END IF

!  if required set up space for the Hessian, y_0 H(x) + sum_i=1^m y_i H_i(x),
!  of the John function

     IF ( PRESENT( h_available ) ) THEN
       set_h = h_available
     ELSE
       set_h = .FALSE.
     END IF

     IF ( set_h ) THEN
       len_array = lEN_TRIM( SMT_get( nlp%H%type ) )
       array_name( 1 : len_array ) = SMT_get( nlp%H%type )
       CALL SMT_put( nls%H%type, array_name( 1 : len_array ),                  &
                     inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; RETURN ; END IF

       nls%H%n = nls%n
       IF ( array_name( 1 : len_array ) == 'DENSE' ) THEN
         nls%H%ne = nls%H%n * ( nls%H%n + 1 ) / 2
       ELSE
         IF ( array_name( 1 : len_array ) == 'SPARSE_BY_ROWS' ) THEN
           nls%H%ne = nlp%H%ptr( nlp%n + 1 ) - 1
           array_name = 'colt: nls%H%ptr'
           CALL SPACE_resize_array( nls%m + 1, nls%H%ptr,                      &
                  inform%status, inform%alloc_status, array_name = array_name, &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910
         ELSE ! 'COORDINATE'
           nls%H%ne = nlp%H%ne
           array_name = 'colt: nls%H%row'
           CALL SPACE_resize_array( nls%H%ne, nls%H%row,                       &
                  inform%status, inform%alloc_status, array_name = array_name, &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 910
         END IF
         array_name = 'colt: nls%H%col'
         CALL SPACE_resize_array( nls%H%ne, nls%H%col,                         &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 910

!  transfer the spasity pattern

         SELECT CASE ( SMT_get( nls%H%type ) )
         CASE ( 'SPARSE_BY_ROWS' )
           nls%H%ptr( : nlp%n + 1 ) = nlp%H%ptr( : nlp%n + 1 ) ! from H(x,y)
           nls%H%ptr( nlp%n + 2 : nls%n + 1 ) = nlp%H%ptr( nlp%n + 1 )
           nls%H%col( : nlp%H%ne ) = nlp%H%col( : nlp%H%ne )
         CASE ( 'COORDINATE' )
           nls%H%row( : nlp%H%ne ) = nlp%H%row( : nlp%H%ne ) ! from H(x,y)
           nls%H%col( : nlp%H%ne ) = nlp%H%col( : nlp%H%ne )
         END SELECT
       END IF
       nls_dims%h_ne = nls%H%ne ; nls_dims%h_f_ne = nls%H%ne

!  make room for the values

       array_name = 'colt: nls%H%val'
       CALL SPACE_resize_array( nls%H%ne, nls%H%val,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910
     ELSE
       nls_dims%h_ne = - 1 ; nls_dims%h_f_ne = - 1  ! unset values
     END IF

!  if required set up space for the matrix of residual Hessian-vector products,
!  P = (H_i(x)v,H(x)r)

     IF ( PRESENT( p_available ) ) THEN
       set_p = h_available
     ELSE
       set_p = .FALSE.
     END IF

     IF ( set_p ) THEN
       CALL CUTEST_cdimchp( inform%status, lchp )
       CALL CUTEST_cdimohp( inform%status, lohp )

       nls_dims%p_f_ne = lchp ; nls_dims%p_ne = lchp + lohp
       nls%P%ne = nls_dims%p_ne

       array_name = 'colt: nls%P%ptr'
       CALL SPACE_resize_array( nls%m + 1, nls%P%ptr,                          &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910
       array_name = 'colt: nls%P%row'
       CALL SPACE_resize_array( nls%P%ne, nls%P%row,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       len_array = lEN_TRIM( SMT_get( nlp%P%type ) )
       array_name( 1 : len_array ) = SMT_get( nlp%P%type )
       CALL SMT_put( nls%P%type, array_name( 1 : len_array ),                  &
                     inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_allocate ; RETURN ; END IF


       CALL CUTEST_cchprodsp( inform%status, nlp%m, lchp,                      &
                               nls%P%row, nls%P%PTR )
       CALL CUTEST_cohprodsp( inform%status, nnzohp, lohp,                     &
                               nls%P%row( lchp + 1 : nls%P%ne ) )
       nls%P%PTR( nls%m + 1 ) = nls%P%ne + 1

!  make room for the values of P

       array_name = 'colt: nls%P%val'
       CALL SPACE_resize_array( nls%P%ne, nls%P%val,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910
     ELSE
       nls_dims%p_f_ne = - 1 ; nls_dims%p_ne = - 1   ! unset values
     END IF


!  successful setup

     inform%status = GALAHAD_ok
     RETURN

!  allocation errors

 910 CONTINUE
     inform%status = GALAHAD_error_allocate
     RETURN

!  end of subroutine COLT_setup_problem

     END SUBROUTINE COLT_setup_problem

!-*-*-*-*-  G A L A H A D -  C O L T _ t r a c k  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE COLT_track( nlp, control, inform, data, userdata,              &
                            n_points, t_lower, t_upper,                        &
                            eval_FC, eval_J, eval_SGJ, eval_HLC, eval_HJ,      &
                            eval_HOCPRODS, eval_HCPRODS )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  COLT_track, a method to track the value of "the" minimizer of
!      1/2 (f(x)-t)^2 + 1/2 ||c(x)||^2
!  for a sequence of equi-distributed values of t in [t_lower,t_upper]

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  All arguments as for COLT_solve, with additionally
!
!  n_points is a scalar variable of type default integer, that holds the
!   number of equi-distributed points t in [t_lower,t_upper] including
!   the end points
!
!  t_lower is a scalar variable of type default real, that holds the
!    value of the lower interval bound, t_lower
!
!  t_upper is a scalar variable of type default real, that holds the
!    value of the upper interval bound, t_upper
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( COLT_control_type ), INTENT( INOUT ) :: control
     TYPE ( COLT_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( COLT_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = rp_ ), INTENT( IN ) :: t_lower, t_upper
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_points

     OPTIONAL :: eval_FC, eval_J, eval_SGJ, eval_HLC, eval_HJ,                 &
                 eval_HOCPRODS, eval_HCPRODS

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_FC( status, X, userdata, f, C )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), OPTIONAL, INTENT( OUT ) :: f
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FC
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_J( status, X, userdata, J )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: J
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_J
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_SGJ( status, X, userdata, G, J )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: J
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_SGJ
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HLC( status, X, Y, userdata, Hval )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_HLC
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HJ( status, X, y0, Y, userdata, Hval )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), INTENT( IN ) :: y0
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_HJ
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HCPRODS( status, X, V, userdata, PCval, got_h )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: PCval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HCPRODS
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HOCPRODS( status, X, V, userdata, POval, PCval, got_h )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: POval
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: PCval
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HOCPRODS
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER ( KIND = ip_ ) :: i, j, j_ne, l
     REAL ( KIND = rp_ ) :: y0, multiplier_norm
     CHARACTER ( LEN = 80 ) :: array_name

!  functions

!$   INTEGER :: OMP_GET_MAX_THREADS

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 900
     END IF
     IF ( inform%status == 1 ) data%branch = 10

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! initial objective and constraint evaluation
       GO TO 20
     CASE ( 210 ) ! gradient and Jacobian evaluation
       GO TO 210
     CASE ( 300 ) ! various evaluations
       GO TO 300
     END SELECT

!  =================
!  0. Initialization
!  =================

  10 CONTINUE

     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
!$   inform%threads = INT( OMP_GET_MAX_THREADS( ), KIND = ip_ )
     inform%status = GALAHAD_ok
     inform%alloc_status = 0 ; inform%bad_alloc = ''
     inform%iter = 0
     inform%fc_eval = 0 ; inform%gj_eval = 0 ; inform%h_eval = 0

     inform%obj = HUGE( one )

!  copy control parameters so that the package may alter values if necessary

     data%control = control

!  decide how much reverse communication is required

     data%reverse_fc = .NOT. PRESENT( eval_FC )
     data%reverse_gj = .NOT. PRESENT( eval_SGJ )
     data%reverse_hj = .NOT. PRESENT( eval_HJ )
     data%reverse_hocprods = .NOT. PRESENT( eval_HOCPRODS )

!  control the output printing

     data%out = data%control%out ; data%error = data%control%error

!  error output

     data%printe = data%error > 0 .AND. data%control%print_level >= 1

!  basic single line of output per iteration

     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printm, but with checking of residuals, etc

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printm but also with an indication of where in the code we are

     data%set_printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debugging printing with significant arrays printed

     data%set_printd = data%out > 0 .AND. data%control%print_level >= 5

!  print level shorthands

     IF ( data%control%start_print < 0 ) THEN
       data%start_print = - 1
     ELSE
       data%start_print = data%control%start_print
     END IF

     IF ( data%control%stop_print < 0 ) THEN
       data%stop_print = data%control%maxit + 1
     ELSE
       data%stop_print = data%control%stop_print
     END IF

!  set print agenda for the first iteration

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE.
       data%printd = .FALSE. ; data%print_level = 0
     END IF

     data%print_iteration_header = data%print_level > 0
     data%print_1st_header = .TRUE.

     IF ( data%printd ) WRITE( data%out, "( ' (A1:1-2)' )" )

! -----------------------------------------------------------------------------
!                            SET UP THE PROBLEM
! -----------------------------------------------------------------------------

     nlp%P%ne = nlp%P%ptr( nlp%m + 1 ) - 1

!  project x to ensure feasibility

     nlp%X = MAX( nlp%X_l, MIN( nlp%X, nlp%X_u ) )

!  set up static data for the least-squares target objective

     CALL COLT_setup_problem( nlp, data%nls, data%nls_dims, data%control,      &
                              inform, data%n_slacks, data%SLACKS,              &
                              h_available = .TRUE., p_available = .TRUE. )

!  evaluate the objective and general constraint function values

     IF ( data%reverse_fc ) THEN
       data%branch = 20 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_FC( data%eval_status, nlp%X, userdata, nlp%f, nlp%C )
       IF ( data%eval_status /= 0 ) THEN
         inform%bad_eval = 'eval_FC'
         inform%status = GALAHAD_error_evaluation ; GO TO 900
       END IF
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     inform%obj = nlp%f
     inform%fc_eval = inform%fc_eval + 1

     inform%primal_infeasibility = TWO_NORM( nlp%C )

!  print problem name and header, if requested

     IF ( data%printi ) WRITE( data%out,                                       &
         "( A, ' +', 76( '-' ), '+', /,                                        &
      &     A, 14X, 'Constrained Optimization via Least-squares Targets', /,   &
      &     A, ' +', 76( '-' ), '+' )" ) prefix, prefix, prefix

!  determine the number of nonzeros in the Hessian

     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       data%H_ne = nlp%H%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%H_ne = nlp%H%ptr( nlp%m + 1 ) - 1
     CASE ( 'DENSE' )
       data%H_ne = ( nlp%n * ( nlp%n+ 1 ) ) / 2
     END SELECT

!  determine the number of nonzeros in the constraint Jacobian

     SELECT CASE ( SMT_get( nlp%J%type ) )
     CASE ( 'COORDINATE' )
       data%J_ne = nlp%J%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%J_ne = nlp%J%ptr( nlp%m + 1 ) - 1
     CASE ( 'DENSE' )
       data%J_ne = nlp%m * nlp%n
     END SELECT

!  set up space for records of t, f, c and phi

     array_name = 'colt: data%t'
     CALL SPACE_resize_array( n_points, data%t,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: data%f'
     CALL SPACE_resize_array( n_points, data%f,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: data%c'
     CALL SPACE_resize_array( n_points, data%c,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: data%phi'
     CALL SPACE_resize_array( n_points, data%phi,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

!  set up space for general vectors used

     array_name = 'colt: nlp%gL'
     CALL SPACE_resize_array( nlp%n, nlp%gL,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: nlp%X_status'
     CALL SPACE_resize_array( nlp%n, nlp%X_status,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'colt: nlp%C_status'
     CALL SPACE_resize_array( nlp%m, nlp%C_status,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( inform%status /= 0 ) GO TO 910

!  set initial values

     data%accepted = .TRUE.
     data%s_norm = zero

!  set scale factors if required

     IF ( data%control%scale_constraints > 0 ) THEN
       array_name = 'colt: data%C_scale'
       CALL SPACE_resize_array( nlp%m, data%C_scale,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = data%control%deallocate_error_fatal,    &
              exact_size = data%control%space_critical,                        &
              bad_alloc = inform%bad_alloc, out = data%error )
       IF ( inform%status /= 0 ) GO TO 910
       data%C_scale( : nlp%m ) = one
     END IF

!  compute norms of the primal and dual feasibility and the complemntary
!  slackness

     multiplier_norm = one
     inform%primal_infeasibility =                                             &
      OPT_primal_infeasibility( nlp%m, nlp%C( : nlp%m ),                       &
                                nlp%C_l( : nlp%m ), nlp%C_u( : nlp%m ) )
     inform%dual_infeasibility = zero
!  NB. Compute this properly later
     inform%complementary_slackness =                                          &
       OPT_complementary_slackness( nlp%n, nlp%X( : nlp%n ),                   &
          nlp%X_l( : nlp%n ), nlp%X_u( : nlp%n ), nlp%Z( : nlp%n ),            &
          nlp%m, nlp%C( : nlp%m ), nlp%C_l( : nlp%m ),                         &
          nlp%C_u( : nlp%m ), nlp%Y( : nlp%m ) ) / multiplier_norm

!  compute the stopping tolerances

     data%stop_p = MAX( data%control%stop_abs_p,                               &
        data%control%stop_rel_p * inform%primal_infeasibility )
     data%stop_d = MAX( data%control%stop_abs_d,                               &
       data%control%stop_rel_d * inform%dual_infeasibility )
     data%stop_c = MAX( data%control%stop_abs_c,                               &
       data%control%stop_rel_c * inform%complementary_slackness )
     data%stop_i = MAX( data%control%stop_abs_i, data%control%stop_rel_i )
     IF ( data%printi ) WRITE( data%out,                                       &
         "(  /, A, '  Primal    convergence tolerance =', ES11.4,              &
        &    /, A, '  Dual      convergence tolerance =', ES11.4,              &
        &    /, A, '  Slackness convergence tolerance =', ES11.4 )" )          &
             prefix, data%stop_p, prefix, data%stop_d, prefix, data%stop_c

!  set up initial target

     y0 = one
     data%target_upper = inform%obj
     data%target_lower = - infinity
     inform%target = t_lower
     data%i_point = 1

!  restore dimensions for target problem

     data%nls%n = data%nls_dims%n ; data%nls%m = data%nls_dims%m
     data%nls%J%ne = data%nls_dims%j_ne ; data%nls%H%ne = data%nls_dims%h_ne
     data%nls%P%ne = data%nls_dims%p_ne

!  ----------------------------------------------------------------------------
!                  START OF MAIN LOOP OVER EVOLVING TARGETS
!  ----------------------------------------------------------------------------

!  solve the target problem: min 1/2||c(x),f)x)-t||^2 for a given target t

     IF ( data%printd ) WRITE( data%out, "( A, ' (A1:3)' )" ) prefix

!    DO
  200  CONTINUE
       inform%target = t_lower + ( REAL( data%i_point - 1, KIND = rp_ ) /      &
           REAL( n_points - 1, KIND = rp_ ) ) * (  t_upper -  t_lower )
!      inform%target = t_lower + ( REAL( n_points - data%i_point, KIND = rp_ ) &
!          / REAL( n_points - 1, KIND = rp_ ) ) * (  t_upper -  t_lower )

!      nlp%X( 1 ) = inform%target
       IF ( data%printd ) THEN
         WRITE( data%out, "( A, ' X ', /, ( 5ES12.4 ) )" )                     &
           prefix, nlp%X( : nlp%n )
         WRITE( data%out, "( A, ' C ', /, ( 5ES12.4 ) )" )                     &
           prefix, nlp%C( : nlp%m )
       END IF

!  obtain the gradient of the objective function and the Jacobian
!  of the constraints. The data is stored in a sparse format

       IF ( data%accepted ) THEN
         inform%gj_eval = inform%gj_eval + 1
         IF ( data%reverse_gj ) THEN
           data%branch = 210 ; inform%status = 3 ; RETURN
         ELSE
           CALL eval_SGJ( data%eval_status, nlp%X, userdata, nlp%Go%val,       &
                          nlp%J%val )
           IF ( data%eval_status /= 0 ) THEN
             inform%bad_eval = 'eval_SGJ'
             inform%status = GALAHAD_error_evaluation ; GO TO 900
           END IF
         END IF
       END IF

!  return from reverse communication to obtain the gradient and Jacobian

  210  CONTINUE
!    write( 6, "( ' sparse gradient ( ind, val )', /, ( 3( I6, ES12.4 ) ) )" ) &
!      ( nlp%Go%ind( i ), nlp%Go%val( i ), i = 1, nlp%Go%ne )

!  compute the gradient of the Lagrangian

       nlp%gL( : nlp%n ) = nlp%Z( : nlp%n )
       DO i = 1, nlp%Go%ne
         j = nlp%Go%ind( i )
         nlp%gL( j ) = nlp%gL( j ) + nlp%Go%val( i )
       END DO
       CALL mop_AX( one, nlp%J, nlp%Y, one, nlp%gL, transpose = .TRUE.,      &
                    m_matrix = nlp%m, n_matrix = nlp%n )
!                   out = data%out, error = data%error,                        &
!                   print_level = data%print_level,                            &

!      WRITE(6,*) ' gl, y ', maxval( nlp%gL ), maxval( nlp%Y )
!      WRITE( data%out, "(A, /, ( 4ES20.12 ) )" ) ' gl_after ',  nlp%gl

       inform%obj = nlp%f

!  exit if the elapsed-time limit has been exceeded

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
             REAL( data%time_now - data%time_start, rp_ )                      &
               > data%control%cpu_time_limit ) .OR.                            &
             ( data%control%clock_time_limit >= zero .AND.                     &
               data%clock_now - data%clock_start                               &
              > data%control%clock_time_limit ) ) THEN

         IF ( data%printi )                                                    &
           WRITE( data%out, "( /, A, ' Time limit exceeded ' )" ) prefix
         inform%status = GALAHAD_error_time_limit ; GO TO 900
       END IF

!  solve the target problem: find x(t) = (local) argmin 1/2||c(x),f)x)-t||^2
!  using the the nls solver.  All function evlautions are for the vector of
!  residuals (c(x),f(x)-t)

       inform%NLS_inform%status = 1
       data%nls%X( : data%nls%n ) = nlp%X
       data%control%NLS_control%jacobian_available = 2
       data%control%NLS_control%subproblem_control%jacobian_available = 2
       data%control%NLS_control%hessian_available = 2

!      DO        ! loop to solve problem
   300 CONTINUE  ! mock loop to solve problem

!  call the least-squares solver to find x(t)

         CALL NLS_solve( data%nls, data%control%NLS_control,                   &
                         inform%NLS_inform, data%NLS_data, data%NLS_userdata )

!  respond to requests for further details

         SELECT CASE ( inform%NLS_inform%status )

!  obtain the residuals

         CASE ( 2 )
           IF ( data%reverse_fc ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 300 ; inform%status = 2 ; RETURN
           ELSE
             CALL eval_FC( data%eval_status, data%nls%X, userdata,             &
                           nlp%f, nlp%C )
             data%discrepancy = nlp%f - inform%target
             data%nls%C( : nlp%m ) = nlp%C( : nlp%m )
             data%nls%C( data%nls%m ) = data%discrepancy
             IF ( print_debug ) WRITE(6,"( ' nls%C = ', /, ( 5ES12.4 ) )" )    &
               data%nls%C( : data%nls%m )
           END IF

!  obtain the Jacobian

         CASE ( 3 )
           IF ( data%reverse_gj ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 300 ; inform%status = 4 ; RETURN
           ELSE
             CALL eval_SGJ( data%eval_status, data%nls%X, userdata,            &
                            nlp%Go%val, nlp%J%val )
             SELECT CASE ( SMT_get( data%nls%J%type ) )
             CASE ( 'DENSE' )
               j_ne = 0 ; l = 0
               DO i = 1, nlp%m  ! from each row of J(x) in turn
                 data%nls%J%val( j_ne + 1 : j_ne + nlp%n )                     &
                   = nlp%J%val( l + 1 : l + nlp%n )
                 j_ne = j_ne + data%nls%n ; l = l + nlp%n
               END DO
               IF ( nlp%Go%sparse ) THEN ! from g(x)
                 DO i = 1, nlp%Go%ne
                   data%nls%J%val( j_ne + nlp%Go%ind( i ) ) = nlp%Go%val( i )
                 END DO
               ELSE
                 data%nls%J%val( j_ne + 1 : j_ne + nlp%n )                     &
                   = nlp%Go%val( 1 : nlp%n )
               END IF
             CASE ( 'SPARSE_BY_ROWS' )
               DO i = 1, nlp%m  ! from each row of J(x) in turn
                 j_ne = data%nls%J%ptr( i ) - 1
                 DO l = nlp%J%ptr( i ), nlp%J%ptr( i + 1 ) - 1
                   j_ne = j_ne + 1
                   data%nls%J%val( j_ne ) = nlp%J%val( l )
                 END DO
               END DO
               j_ne = data%nls%J%ptr( data%nls%m ) - 1
               DO j = 1, nlp%Go%ne
                 j_ne = j_ne + 1
                 data%nls%J%val( j_ne ) = nlp%Go%val( j )
               END DO
             CASE ( 'COORDINATE' )
               data%nls%J%val( : nlp%J%ne ) = nlp%J%val( : nlp%J%ne ) ! from J
               j_ne = nlp%J%ne + data%n_slacks
               DO j = 1, nlp%Go%ne
                 j_ne = j_ne + 1
                 data%nls%J%val( j_ne ) = nlp%Go%val( j )
               END DO
               IF ( print_debug ) WRITE(6,"( ' nls%J', / ( 3( 2I6, ES12.4 )))")&
                ( data%nls%J%row( i ), data%nls%J%col( i ),                    &
                  data%nls%J%val( i ), i = 1, data%nls%J%ne )
             END SELECT
           END IF

!  obtain the Hessian

         CASE ( 4 )
           IF ( data%reverse_hj ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 300 ; inform%status = 6 ; RETURN
           ELSE
             CALL eval_HJ( data%eval_status, data%nls%X,                       &
                           data%NLS_data%Y( data%nls%m ),                      &
                           data%NLS_data%Y( 1 : nlp%m ),                       &
                           userdata, data%nls%H%val )
             IF ( print_debug ) WRITE(6,"( ' nls%H', / ( 3( 2I6, ES12.4 )))" ) &
              ( data%nls%H%row( i ), data%nls%H%col( i ),                      &
                data%nls%H%val( i ), i = 1, data%nls%H%ne )
           END IF

!  form a Jacobian-vector product

         CASE ( 5 )
           write(6,*) ' no nls_status = 5 as yet, stopping'
           stop
!          CALL JACPROD( data%eval_status, data%nls%X,                         &
!                        data%NLS_userdata, data%NLS_data%transpose,           &
!                        data%NLS_data%U, data%NLS_data%V )

!  form a Hessian-vector product

         CASE ( 6 )
           write(6,*) ' no nls_status = 6 as yet, stopping'
           stop
!          CALL HESSPROD( data%eval_status, data%nls%X, data%NLS_data%Y,       &
!                         data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  form residual Hessian-vector products

         CASE ( 7 )
           IF ( data%reverse_hocprods ) THEN
             nlp%X = data%nls%X( : data%nls%n )
             data%branch = 300 ; inform%status = 8 ; RETURN
           ELSE
             CALL eval_HOCPRODS( data%eval_status, data%nls%X,                 &
                                 data%nls_data%V, userdata,                    &
                                 data%nls%P%val( nlp%P%ne + 1 :                &
                                                 data%nls%P%ne ),              &
                                 data%nls%P%val( 1 : nlp%P%ne ) )
             IF ( print_debug ) THEN
               WRITE(6,"( ' nls%P' )" )
               DO j = 1, data%nls%n
                 WRITE(6,"( 'column ',  I0, /, / ( 4( I6, ES12.4 ) ) )" ) &
                   j, ( data%nls%P%row( i ), data%nls%P%val( i ), i = &
                        data%nls%P%ptr( j ), data%nls%P%ptr( j + 1 ) - 1 )
               END DO
             END IF
           END IF

!  apply the preconditioner

         CASE ( 8 )
           write(6,*) ' no nls_status = 8 as yet, stopping'
           stop
!          CALL SCALE( data%eval_status, data%nls%X,                           &
!                      data%NLS_userdata, data%NLS_data%U, data%NLS_data%V )

!  terminal exit from the minimization loop

         CASE DEFAULT
           nlp%X = data%nls%X( : data%nls%n )
           GO TO 390
         END SELECT
         GO TO 300
!      END DO ! end of loop to solve the target problem

   390  CONTINUE

!  record the current t, f, c and phi

        inform%primal_infeasibility = TWO_NORM( nlp%C )
!       WRITE( data%out, "( ' i, target, f, ||c|| = ', I6, 3ES16.8 )" )        &
!        data%i_point, inform%target, nlp%f, inform%primal_infeasibility
        WRITE( data%out,                                                       &
           "( ' i, it, target, phi, ||g|| = ', 2I5, F7.2, 2ES16.8 )" )         &
         data%i_point, inform%nls_inform%iter, inform%target,                  &
         inform%nls_inform%obj, inform%nls_inform%norm_g
!write(6,"( ' X = ', ( 4ES14.6 ) )" ) nlp%X

        data%t( data%i_point ) = inform%target
        data%f( data%i_point ) = nlp%f
        data%c( data%i_point ) = inform%primal_infeasibility
        data%phi( data%i_point ) = inform%NLS_inform%obj

       IF ( data%printd ) THEN
         WRITE( data%out,"( '     X_l           X         X_u' )" )
         DO i = 1, nlp%n
         WRITE( data%out,"( 3ES12.4 )" ) nlp%X_l(i), nlp%X(i), nlp%X_u(i)
         END DO
       END IF

!  ----------------------------------------------------------------------------
!                      END OF MAIN LOOP TARGET LOOP
!  ----------------------------------------------------------------------------

       data%i_point = data%i_point + 1
       IF ( data%i_point > n_points ) GO TO 900
       GO TO 200
!    END DO

!  summarize the final results

 900 CONTINUE

     OPEN( UNIT = track_out, FILE = 'track.m' )
     WRITE( track_out, "( 't = [' )" )
     WRITE( track_out, "( ( 1P4E16.8, '...' ) )" )  data%t( : n_points - 1 )
     WRITE( track_out, "( 1PE16.8, '];' )" )  data%t( n_points )

     WRITE( track_out, "( 'f = [' )" )
     WRITE( track_out, "( ( 1P4E16.8, '...' ) )" )  data%f( : n_points - 1 )
     WRITE( track_out, "( 1PE16.8, '];' )" )  data%f( n_points )

     WRITE( track_out, "( 'c = [' )" )
     WRITE( track_out, "( ( 1P4E16.8, '...' ) )" )  data%c( : n_points - 1 )
     WRITE( track_out, "( 1PE16.8, '];' )" )  data%c( n_points )

     WRITE( track_out, "( 'phi = [' )" )
     WRITE( track_out, "( ( 1P4E16.8, '...' ) )" )  data%phi( : n_points - 1 )
     WRITE( track_out, "( 1PE16.8, '];' )" )  data%phi( n_points )
     CLOSE( UNIT = track_out )

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = REAL( data%time_now - data%time_start, rp_ )
     inform%time%clock_total = data%clock_now - data%clock_start

     DEALLOCATE( data%t, data%f, data%c, data%phi )

!  restore scaled-cnstraint data

     IF ( data%control%scale_constraints > 0 ) THEN
       nlp%Y( : nlp%m ) = nlp%Y( : nlp%m ) / data%C_scale( : nlp%m )
       nlp%C( : nlp%m ) = nlp%C( : nlp%m ) * data%C_scale( : nlp%m )
       WHERE ( nlp%C_l( : nlp%m ) > - data%control%infinity )                  &
         nlp%C_l( : nlp%m ) = nlp%C_l( : nlp%m ) * data%C_scale( : nlp%m )
       WHERE ( nlp%C_u( : nlp%m ) < data%control%infinity )                    &
         nlp%C_u( : nlp%m ) = nlp%C_u( : nlp%m ) * data%C_scale( : nlp%m )
     END IF

     RETURN

!  -------------
!  Error returns
!  -------------

!  allocation errors

 910 CONTINUE
     inform%status = GALAHAD_error_allocate
     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%total = REAL( data%time_now - data%time_start, rp_ )
     inform%time%clock_total = data%clock_now - data%clock_start
     RETURN

!  end of subroutine COLT_track

     END SUBROUTINE COLT_track

!-*-*-  G A L A H A D -  C O L T _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE COLT_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( COLT_data_type ), INTENT( INOUT ) :: data
     TYPE ( COLT_control_type ), INTENT( IN ) :: control
     TYPE ( COLT_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated within NLS

     CALL NLS_terminate( data%NLS_data, data%control%NLS_control,              &
                         inform%NLS_inform )
     inform%status = inform%NLS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%NLS_inform%alloc_status
       inform%bad_alloc = inform%NLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  deallocate all remaining allocated arrays

     array_name = 'colt: data%C_scale'
     CALL SPACE_dealloc_array( data%C_scale,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%W'
     CALL SPACE_dealloc_array( data%W,                                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%V2'
     CALL SPACE_dealloc_array( data%V2,                                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%X_old'
     CALL SPACE_dealloc_array( data%X_old,                                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%Y_old'
     CALL SPACE_dealloc_array( data%Y_old,                                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%C_old'
     CALL SPACE_dealloc_array( data%C_old,                                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%G_old'
     CALL SPACE_dealloc_array( data%G_old,                                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'colt: data%J_old'
     CALL SPACE_dealloc_array( data%J_old,                                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            bad_alloc = inform%bad_alloc, out = data%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine COLT_terminate

     END SUBROUTINE COLT_terminate

!  End of module GALAHAD_COLT

   END MODULE GALAHAD_COLT_precision
