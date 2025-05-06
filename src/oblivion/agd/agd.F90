! THIS VERSION: GALAHAD 5.2 - 2025-05-02 AT 09:20 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ A D G   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Fowkes/Gould/Montoison/Orban, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 5.2. May 2nd 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_AGD_precision

!     --------------------------------------------------------
!    |                                                        |
!    | AGD, a first-order (accelerated steepest-descent)      |
!    |  algorithm for unconstrained optimization              |
!    |                                                        |
!    |   Aim: find a (local) minimizer of the objective f(x)  |
!    |                                                        |
!     --------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
     USE GALAHAD_STRING, ONLY: STRING_integer_6
!    USE SPDSOL
!    USE HSL_MI13
!    USE IEEE_ARITHMETIC

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: AGD_initialize, AGD_read_specfile, AGD_solve,                   &
               AGD_terminate, NLPT_problem_type, GALAHAD_userdata_type
!,        &
!              AGD_import, AGD_solve_direct, AGD_solve_reverse,                &
!              AGD_full_initialize, AGD_full_terminate, AGD_reset_control,     &
!              AGD_information

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE AGD_initialize
       MODULE PROCEDURE AGD_initialize, AGD_full_initialize
     END INTERFACE AGD_initialize

     INTERFACE AGD_terminate
       MODULE PROCEDURE AGD_terminate, AGD_full_terminate
     END INTERFACE AGD_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER ( KIND = ip_ ), PARAMETER  :: nskip_prec_max = 0
     INTEGER ( KIND = ip_ ), PARAMETER  :: history_max = 100
     LOGICAL, PARAMETER  :: debug_model_4 = .TRUE.
!    LOGICAL, PARAMETER  :: test_s = .TRUE.
     LOGICAL, PARAMETER  :: test_s = .FALSE.
     REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: three = 3.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
     REAL ( KIND = rp_ ), PARAMETER :: tenth = 0.1_rp_
     REAL ( KIND = rp_ ), PARAMETER :: sixteenth = 0.0625_rp_
     REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: hundred = 100.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: twelve = 12.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: sixteen = 16.0_rp_
     REAL ( KIND = rp_ ), PARAMETER :: point9 = 0.9_rp_
     REAL ( KIND = rp_ ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = rp_ ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = rp_ ), PARAMETER :: teneps = ten * epsmch
     REAL ( KIND = rp_ ), PARAMETER :: rho_quad = epsmch * ten ** 4

     REAL ( KIND = rp_ ), PARAMETER :: gamma_1 = sixteenth
     REAL ( KIND = rp_ ), PARAMETER :: gamma_2 = half
     REAL ( KIND = rp_ ), PARAMETER :: gamma_3 = two
     REAL ( KIND = rp_ ), PARAMETER :: gamma_4 = sixteen
     REAL ( KIND = rp_ ), PARAMETER :: mu_1 = one - ten ** ( - 8 )
     REAL ( KIND = rp_ ), PARAMETER :: mu_2 = point1
     REAL ( KIND = rp_ ), PARAMETER :: theta = half

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: AGD_control_type

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

!   the maximum number of iterations performed

       INTEGER ( KIND = ip_ ) :: maxit = 100

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER ( KIND = ip_ ) :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   overall convergence tolerances. The iteration will terminate when the
!     norm of the gradient of the objective function is smaller than
!       MAX( %stop_g_absolute, %stop_g_relative * norm of the initial gradient )
!     or if the step is less than %stop_s

#ifdef REAL_32
       REAL ( KIND = rp_ ) :: stop_g_absolute = ten ** ( - 3 )
       REAL ( KIND = rp_ ) :: stop_g_relative = ten ** ( - 4 )
#else
       REAL ( KIND = rp_ ) :: stop_g_absolute = ten ** ( - 5 )
       REAL ( KIND = rp_ ) :: stop_g_relative = ten ** ( - 8 )
#endif
       REAL ( KIND = rp_ ) :: stop_s = epsmch

!  initial estimate of the gradient Lipschitz constant

       REAL ( KIND = rp_ ) :: l_g_initial = ten ** ( - 3 )

!  initial estimate of the Hessian Lipschitz constant > 0

       REAL ( KIND = rp_ ) :: l_h_initial = ten ** ( - 16 )

!  estimated gradient Lipschitz constant increase factor > 1

       REAL ( KIND = rp_ ) :: l_g_increase = two

!  estimated gradient Lipschitz constant decrease factor in (0,1]

       REAL ( KIND = rp_ ) :: l_g_decrease = point9

!   the smallest value the onjective function may take before the problem
!    is marked as unbounded

       REAL ( KIND = rp_ ) :: obj_unbounded = - epsmch ** ( - 2 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = rp_ ) :: clock_time_limit = - one

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE AGD_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: AGD_time_type

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

     TYPE, PUBLIC :: AGD_inform_type

!  return status. See AGD_solve for details

       INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER ( KIND = ip_ ) :: iter = 0

!  the total number of CG iterations performed

       INTEGER ( KIND = ip_ ) :: cg_iter = 0

!  the total number of evaluations of the objection function

       INTEGER ( KIND = ip_ ) :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER ( KIND = ip_ ) :: g_eval = 0

!  the value of the objective function at the best estimate of the solution
!   determined by AGD_solve

       REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the norm of the gradient of the objective function at the best estimate
!   of the solution determined by AGD_solve

       REAL ( KIND = rp_ ) :: norm_g = HUGE( one )

!  estimate of the gradient Lipschitz constant

       REAL ( KIND = rp_ ) :: l_g = ten ** ( - 3 )

!  estimate of the Hessian Lipschitz constant > 0

       REAL ( KIND = rp_ ) :: l_h = ten ** ( - 16 )

!  the current value of the trust-region radius

       REAL ( KIND = rp_ ) :: radius = zero

!  timings (see above)

       TYPE ( AGD_time_type ) :: time

     END TYPE AGD_inform_type

!  - - - - - - - - - -
!   data derived types
!  - - - - - - - - - -

     TYPE, PUBLIC :: AGD_data_type
       INTEGER ( KIND = ip_ ) :: branch = 1
       INTEGER ( KIND = ip_ ) :: eval_status, out, start_print, stop_print
       INTEGER ( KIND = ip_ ) :: print_level, print_gap, k
       REAL :: time_start, time_record, time_now
       REAL ( KIND = rp_ ) :: clock_start, clock_record, clock_now
       REAL ( KIND = rp_ ) :: f_ref, f_trial, f_best, theta, s
       REAL ( KIND = rp_ ) :: f_0, f_xk, f_xkm1, f_yk, norm_xkmxkm1, norm_ykmxk
       REAL ( KIND = rp_ ) :: old_radius, radius_trial, etat, ometat
       REAL ( KIND = rp_ ) :: stop_g, s_new_norm, hmax_error, zeta, rk, rkp1
       LOGICAL :: printi, printt, printd, printm
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printd, set_printm
       LOGICAL :: f_is_nan, reverse_f, reverse_g
       LOGICAL :: sparse_hessian, successful
       CHARACTER ( LEN = 1 ) :: epoch = ' '
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: X_old
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DX
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G_x
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G_y
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G_x_old

!  copy of controls

       TYPE ( AGD_control_type ) :: control

     END TYPE AGD_data_type

     TYPE, PUBLIC :: AGD_full_data_type
       LOGICAL :: f_indexing = .TRUE.
       TYPE ( AGD_data_type ) :: agd_data
       TYPE ( AGD_control_type ) :: agd_control
       TYPE ( AGD_inform_type ) :: agd_inform
       TYPE ( NLPT_problem_type ) :: nlp
       TYPE ( GALAHAD_userdata_type ) :: userdata
     END TYPE AGD_full_data_type

   CONTAINS

!-*-*-  G A L A H A D -  A D G _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE AGD_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TRU controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AGD_data_type ), INTENT( INOUT ) :: data
     TYPE ( AGD_control_type ), INTENT( OUT ) :: control
     TYPE ( AGD_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initial private data. Set branch for initial entry

     data%branch = 1

     RETURN

!  End of subroutine AGD_initialize

     END SUBROUTINE AGD_initialize

!- G A L A H A D -  A D G _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE AGD_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TRU controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AGD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( AGD_control_type ), INTENT( OUT ) :: control
     TYPE ( AGD_inform_type ), INTENT( OUT ) :: inform

     CALL AGD_initialize( data%agd_data, data%agd_control, data%agd_inform )
     control = data%agd_control
     inform = data%agd_inform

     RETURN

!  End of subroutine AGD_full_initialize

     END SUBROUTINE AGD_full_initialize

!-*-*-*-*-   A D G _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE AGD_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by AGD_initialize could (roughly)
!  have been set as:

! BEGIN TRU SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  maximum-number-of-iterations                    100
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  absolute-gradient-accuracy-required             1.0D-5
!  relative-gradient-reduction-required            1.0D-5
!  minimum-step-allowed                            2.0D-16
!  initial-gradient-lipschitz-constant             1.0E-3
!  initial-hessian-lipschitz-constant              1.0E-16
!  gradient-lipschitz-constant-increase            2.0E+0
!  gradient-lipschitz-constant-decrease            0.9E+0
!  minimum-objective-before-unbounded              -1.0D+32
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
!  output-line-prefix                              ""
! END TRU SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AGD_control_type ), INTENT( INOUT ) :: control
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
     INTEGER ( KIND = ip_ ), PARAMETER :: maxit = print_gap + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_unit = maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_g_absolute = alive_unit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_g_relative = stop_g_absolute + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: stop_s = stop_g_relative + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: l_g_initial = stop_s + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: l_h_initial = l_g_initial + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: l_g_increase = l_h_initial + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: l_g_decrease = l_g_increase + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: obj_unbounded = l_g_decrease + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: cpu_time_limit = obj_unbounded + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = clock_time_limit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal               &
                                            = space_critical + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: alive_file                           &
                                            = deallocate_error_fatal + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = alive_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'AGD '
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
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( alive_unit )%keyword = 'alive-device'

!  Real key-words

     spec( stop_g_absolute )%keyword = 'absolute-gradient-accuracy-required'
     spec( stop_g_relative )%keyword = 'relative-gradient-reduction-required'
     spec( stop_s )%keyword = 'minimum-step-allowed'
     spec( l_g_initial )%keyword = 'initial-gradient-lipschitz-constant'
     spec( l_h_initial )%keyword = 'initial-hessian-lipschitz-constant'
     spec( l_g_increase )%keyword = 'gradient-lipschitz-constant-increase'
     spec( l_g_decrease )%keyword = 'gradient-lipschitz-constant-decrease'
     spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

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
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( stop_g_absolute ),                      &
                                 control%stop_g_absolute,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_g_relative ),                      &
                                 control%stop_g_relative,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_s ),                               &
                                 control%stop_s,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( l_g_initial ),                          &
                                 control%l_g_initial,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( l_h_initial ),                          &
                                 control%l_h_initial,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( l_g_increase ),                         &
                                 control%l_g_increase,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( l_g_decrease ),                         &
                                 control%l_g_decrease,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( obj_unbounded ),                        &
                                 control%obj_unbounded,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

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

     RETURN

     END SUBROUTINE AGD_read_specfile

!-*-*-*-  G A L A H A D -  A G D _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE AGD_solve( nlp, control, inform, data, userdata,               &
                           eval_F, eval_G )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  AGD_solve, an accelerated-gradient-descent method for finding a local
!    unconstrained minimizer of a given function

!  This is based on Algorithm 4.1 by Naoki Marumo & Akiko Takeda,
!   "Parameter-free accelerated gradient descent for nonconvex minimization",
!   SIAM J. Optimization 34(2) pp 2093-2120 (2024)

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_AGD.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_AGD_precision
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
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
! control is a scalar variable of type AGD_control_type. See AGD_initialize
!  for details
!
! inform is a scalar variable of type AGD_inform_type. On initial entry,
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
!    -3. The restriction nlp%n > 0 or requirement that prob%H_type contains
!        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
!          or 'DIAGONAL' has been violated.
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
!   -19. The CPU time limit has been reached. This may happen if
!        control%cpu_time_limit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -40. The user has forced termination of solver by removing the file named
!        control%alive_file from unit unit control%alive_unit.
!
!     2. The user should compute the objective function value f(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The required
!        value should be set in nlp%f, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%f, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value of the i-th component of the gradient should be
!        set in nlp%G(i), for i = 1, ..., n and data%eval_status should be set
!        to 0. If the user is unable to evaluate a component of nabla_x f(x)
!        - for instance if a component of the gradient is undefined at x - the
!        user need not set nlp%G, but should then set data%eval_status to a
!        non-zero value.
!     4. The user should compute the Hessian of the objective function
!        nabla_xx f(x) at the point x indicated in nlp%X and then re-enter the
!        subroutine. The value l-th component of the Hessian stored according to
!        the scheme input in the remainder of nlp%H should be set in
!        nlp%H%val(l), for l = 1, ..., nlp%H%ne and data%eval_status should be
!        set to 0. If the user is unable to evaluate a component of
!        nabla_xx f(x) - for instance, if a component of the Hessian is
!        undefined at x - the user need not set nlp%H%val, but should then set
!        data%eval_status to a non-zero value.
!     5. The user should compute the product nabla_xx f(x)v of the Hessian
!        of the objective function nabla_xx f(x) at the point x indicated in
!        nlp%X with the vector v and add the result to the vector u and then
!        re-enter the subroutine. The vectors u and v are given in data%U and
!        data%V respectively, the resulting vector u + nabla_xx f(x)v should be
!        set in data%U and  data%eval_status should be set to 0. If the user is
!        unable to evaluate the product - for instance, if a component of the
!        Hessian is undefined at x - the user need not alter data%U, but
!        should then set data%eval_status to a non-zero value.
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( AGD_control_type ), INTENT( IN ) :: control
     TYPE ( AGD_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( AGD_data_type ), INTENT( INOUT ) :: data
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_F, eval_G

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_F( status, X, userdata, f )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), INTENT( OUT ) :: f
       REAL ( KIND = rp_ ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_F
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_USERDATA_precision
       INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     REAL ( KIND = rp_ ) :: zeta_new
     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter
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
     CASE ( 20 )  ! initial objective evaluation
       GO TO 20
     CASE ( 30 )  ! initial gradient evaluation
       GO TO 30
     CASE ( 110 ) ! objective evaluation at x_k
       GO TO 110
     CASE ( 120 ) ! objective evaluation at y_k
       GO TO 120
     CASE ( 130 ) ! gradient evaluation at y_k
       GO TO 130
     CASE ( 140 ) ! gradient evaluation at x_k
       GO TO 140
     END SELECT

!  ============================================================================
!  Initialization
!  ============================================================================

 10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  decide how much reverse communication is required

     data%reverse_f = .NOT. PRESENT( eval_F )
     data%reverse_g = .NOT. PRESENT( eval_G )

!  ensure that the data is consistent

     data%control = control

!  control the output printing

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

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter - 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printd = .FALSE.
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

!  allocate workspace

     array_name = 'agd: data%DX'
     CALL SPACE_resize_array( nlp%n, data%DX, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'agd: data%X_old'
     CALL SPACE_resize_array( nlp%n, data%X_old, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'agd: data%Y'
     CALL SPACE_resize_array( nlp%n, data%Y, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'agd: data%Z'
     CALL SPACE_resize_array( nlp%n, data%Z, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'agd: data%G_x'
     CALL SPACE_resize_array( nlp%n, data%G_x, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'agd: data%G_y'
     CALL SPACE_resize_array( nlp%n, data%G_y, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'agd: data%G_x_old'
     CALL SPACE_resize_array( nlp%n, data%G_x_old, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( data%reverse_g ) THEN
       array_name = 'agd: data%X'
       CALL SPACE_resize_array( nlp%n, data%X, inform%status,                  &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  initialize (x_0, y_0) <- (x_init, x_init)

     data%Y( : nlp%n ) =  nlp%X( : nlp%n ) 

!  evaluate the objective function at the initial point

     IF ( data%reverse_f ) THEN
       data%branch = 20 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, data%f_0 )
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     inform%f_eval = inform%f_eval + 1
     IF ( data%reverse_f ) data%f_0 = nlp%f
     data%f_xk = data%f_0

!  evaluate the gradient of the objective function at the initial point

     IF ( data%reverse_g ) THEN
!      data%X( : nlp%n ) = nlp%X(  : nlp%n )
!      nlp%X(  : nlp%n ) = data%Y( : nlp%n )
       data%branch = 30 ; inform%status = 3 ; RETURN
     ELSE
       CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    data%G_x( : nlp%n ) )
     END IF

!  return from reverse communication to obtain the gradient

  30 CONTINUE
     inform%g_eval = inform%g_eval + 1
!    IF ( data%reverse_g ) nlp%X(  : nlp%n ) = data%X( : nlp%n )
     IF ( data%reverse_g ) data%G_x( : nlp%n ) = nlp%G( : nlp%n )
!    data%G_y( : nlp%n ) = nlp%G( : nlp%n )
     data%G_y( : nlp%n ) = data%G_x( : nlp%n )

!  initialize counters and scalars

     inform%iter = 0 ! K <- 0
     data%k = 0 ! k <- 0
     inform%l_g = control%l_g_initial ! l_g <- l_g_init
     inform%l_h = control%l_h_initial ! l_h <- l_h_init
     data%s = zero
     data%f_xkm1 = data%f_0
     inform%norm_g = TWO_NORM( data%G_x( : nlp%n ) )

!  compute the stopping tolerance

     data%stop_g = MAX( control%stop_g_absolute,                               &
                        control%stop_g_relative * inform%norm_g )

     IF ( data%printi ) WRITE( data%out, "( /, A, '  Problem: ', A,            &
    &   ' (n = ', I0, '): TRU stopping tolerance =', ES11.4, / )" )            &
       prefix, TRIM( nlp%pname ), nlp%n, data%stop_g

!  ============================================================================
!  Main loop
!  ============================================================================

 100 CONTINUE

!  record current time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter - 1 - data%start_print, data%print_gap ) == 0 )   &
           THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printd = data%set_printd
         data%print_level = data%control%print_level
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1

!  print one-line summary

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2020 ) prefix
           WRITE( data%out, 2030 ) prefix
         END IF
         data%print_1st_header = .FALSE.
         char_iter = ADJUSTR( STRING_integer_6( inform%iter ) )
         IF ( inform%iter > 0 ) THEN
           WRITE( data%out, 2000 ) prefix, char_iter, data%epoch,             &
             data%f_xk, inform%norm_g, inform%l_g, inform%l_h, data%clock_now
         ELSE
           WRITE( data%out, 2000 ) prefix, char_iter, data%epoch,             &
             data%f_xk, inform%norm_g, inform%l_g, inform%l_h, data%clock_now
         END IF
       END IF

!  stop if the gradient is small enough

       IF ( inform%norm_g <= data%stop_g ) THEN
         inform%status = GALAHAD_ok ; GO TO 800
       END IF

!  check to see if the iteration limit has been exceeded

       IF ( inform%iter > data%control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 800
       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = GALAHAD_error_alive
           RETURN
         END IF
       END IF

       inform%iter = inform%iter + 1  ! K <- K + 1
       data%k = data%k + 1 ! k <- k + 1
       data%rk = REAL( data%k, rp_ ) 
       data%rkp1 = REAL( data%k + 1, rp_ )        

!  compute the acceleration parameter, theta

       data%theta = data%rk / data%rkp1

!  update x_k = y_k-1 - ( 1 / l_g ) g( y_k-1 )

       data%X_old( : nlp%n ) = nlp%X( : nlp%n )
       nlp%X( : nlp%n ) = data%Y( : nlp%n ) - data%G_y( : nlp%n ) / inform%l_g
       data%DX( : nlp%n ) = nlp%X( : nlp%n ) - data%X_old( : nlp%n )

       IF ( data%printd ) THEN
         WRITE( data%out, "( ' X_old =    ', 5ES12.4, /, ( 12X, 5ES12.4 ) )" ) &
           data%X_old( : nlp%n )
         WRITE( data%out, "( ' X =        ', 5ES12.4, /, ( 12X, 5ES12.4 ) )" ) &
           nlp%X( : nlp%n )
         WRITE( data%out, "( ' Y =        ', 5ES12.4, /, ( 12X, 5ES12.4 ) )" ) &
           data%Y( : nlp%n )
         WRITE( data%out, "( ' G_y =      ', 5ES12.4, /, ( 12X, 5ES12.4 ) )" ) &
           data%G_y( : nlp%n )
       END IF

!  compute the norm ||x_k - x_k-1|| and update s_k = sum_i=0^k ||x_i - x_i-1||^2

       data%norm_xkmxkm1 = TWO_NORM( data%DX( : nlp%n ) )
       data%s = data%s + data%norm_xkmxkm1 ** 2

!  evaluate the objective function at x_k

       IF ( data%reverse_f ) THEN
         data%branch = 110 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, data%f_xk )
       END IF

!  return from reverse communication to obtain the objective value

 110   CONTINUE
       IF ( data%reverse_f ) data%f_xk = nlp%f
       inform%f_eval = inform%f_eval + 1

!  check for an unsuccessful epoch, and restart if so

       IF ( data%f_xk > data%f_0                                               &
             - half * inform%l_g * data%s / data%rkp1 ) THEN
         inform%l_g = control%l_g_increase * inform%l_g
         data%f_0 = data%f_xkm1
         data%f_xk = data%f_xkm1
         nlp%X( : nlp%n ) =  data%X_old( : nlp%n )
         data%Y( : nlp%n ) =  data%X_old( : nlp%n ) 
         data%G_y( : nlp%n ) = data%G_x( : nlp%n ) 
         data%Z( : nlp%n ) = data%Y( : nlp%n )
         data%s = zero
         data%zeta = one
         data%k = 0
         data%epoch = 'u'
         GO TO 100
       END IF

!  update y_k = x_k + ( k/(k+1)) (x_k - x_k-1 )

       data%Y( : nlp%n ) = nlp%X( : nlp%n ) + data%theta * data%DX( : nlp%n )

!  compute the norms ||x_k - x_k-1|| and ||y_k - x_k||, and update
!   S_k = sum_i=0^k ||x_i - x_i-1||^2

       data%norm_ykmxk = TWO_NORM( data%Y( : nlp%n ) - nlp%X( : nlp%n ) )

!  evaluate the objective function at y_k

       IF ( data%reverse_f ) THEN
         data%X( : nlp%n ) = nlp%X(  : nlp%n )
         nlp%X(  : nlp%n ) = data%Y( : nlp%n )
         data%branch = 120 ; inform%status = 2 ; RETURN
       ELSE
         CALL eval_F( data%eval_status, data%Y( : nlp%n ), userdata, data%f_yk )
       END IF

!  return from reverse communication to obtain the objective value

 120   CONTINUE
       IF ( data%reverse_f ) data%f_yk = nlp%f
       inform%f_eval = inform%f_eval + 1

!  evaluate the gradient of the objective function at y_k

       data%G_x( : nlp%n ) = nlp%G( : nlp%n )
       IF ( data%reverse_g ) THEN
         data%branch = 130 ; inform%status = 3 ; RETURN
       ELSE
         CALL eval_G( data%eval_status, data%Y( : nlp%n ), userdata,           &
                      data%G_y( : nlp%n ) )
       END IF

!  return from reverse communication to obtain the gradient

 130   CONTINUE
       inform%g_eval = inform%g_eval + 1
       IF ( data%reverse_g ) THEN
         data%G_y( : nlp%n ) = nlp%G( : nlp%n )
         nlp%X(  : nlp%n ) = data%X( : nlp%n )
       END IF

!  evaluate the gradient of the objective function at x_k

       data%G_x_old( : nlp%n ) = data%G_x( : nlp%n )
       IF ( data%reverse_g ) THEN
         data%branch = 140 ; inform%status = 3 ; RETURN
       ELSE
         CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,            &
                      data%G_x( : nlp%n ) )
       END IF

!  return from reverse communication to obtain the gradient

 140   CONTINUE
       inform%g_eval = inform%g_eval + 1
       IF ( data%reverse_g ) data%G_x( : nlp%n ) = nlp%G( : nlp%n )
       inform%norm_g = TWO_NORM( data%G_x( : nlp%n ) )

!  improve the estimate of the Hessian Lipschitz constant (M+H (4.9))

       inform%l_h = MAX( inform%l_h,                                           &
          twelve * ( data%f_yk - data%f_xk - half *                            &
          DOT_PRODUCT( data%G_y( : nlp%n ) + data%G_x( : nlp%n ),              &
                       data%Y( : nlp%n ) - nlp%X( : nlp%n ) ) )                &
            / data%norm_ykmxk ** 3,                                            &
          TWO_NORM( data%G_y( : nlp%n )                                        &
                    + data%theta * data%G_x_old( : nlp%n )                     &
                    - ( one +  data%theta ) * data%G_x( : nlp%n ) )            &
            / ( data%theta * data%norm_xkmxkm1 ** 2 ) )

!  check for a very successful epoch, and restart if so

       IF ( data%rkp1 ** 5 * inform%l_h ** 2 * data%s > inform%l_g ** 2 ) THEN
         inform%l_g = control%l_g_decrease * inform%l_g
         data%f_0 = data%f_xk
         data%f_xkm1 = data%f_0
         data%Y( : nlp%n ) =  nlp%X( : nlp%n ) 
         data%G_y( : nlp%n ) = data%G_x( : nlp%n ) 
         data%Z( : nlp%n ) = data%Y( : nlp%n )
         data%k = 0
         data%s = zero
         data%zeta = one
         data%epoch = 'v'
         GO TO 100
       END IF
       data%epoch = ' '

!  update the averaged solution

       zeta_new = one + data%theta * data%zeta
       data%Z( : nlp%n ) = ( data%Y( : nlp%n )                                 &
        + data%theta * data%zeta * data%Z( : nlp%n ) ) / zeta_new
       data%zeta = zeta_new

!  ============================================================================
!  end of main loop
!  ============================================================================

     GO TO 100

 800 CONTINUE
     inform%obj = data%f_xk
     nlp%G( : nlp%n ) = data%G_x( : nlp%n )
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     inform%status = GALAHAD_ok
     RETURN

!  -------------
!  Error returns
!  -------------

!  allocation error

 980 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) WRITE( control%out,                                    &
       "( A, ' ** Message from -AGD_solve-', /,  A,                            &
      &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )        &
       prefix, prefix, inform%bad_alloc, inform%alloc_status
     RETURN

 990 CONTINUE
     inform%obj = data%f_xk
     nlp%G( : nlp%n ) = data%G_x( : nlp%n )
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, control%out, prefix, 'UGO_solve' )
       WRITE( control%out, "( ' ' )" )
     END IF
     RETURN

!  non-executable statements

2000 FORMAT( A, A6, 1X, A1, ES22.14, ES11.4, 2ES8.1, F8.2 )
2020 FORMAT( A, '        (s=succesful,u=unsuccesful,v=very successful)' )
2030 FORMAT( A, '    It             f              grad    ',                  &
                '  L_g     L_h      time' )

!  end of subroutine AGD_solve

     END SUBROUTINE AGD_solve

!-*-*-  G A L A H A D -  A G D _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE AGD_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AGD_data_type ), INTENT( INOUT ) :: data
     TYPE ( AGD_control_type ), INTENT( IN ) :: control
     TYPE ( AGD_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'agd: data%DX'
     CALL SPACE_dealloc_array( data%DX,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%X'
     CALL SPACE_dealloc_array( data%X,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%X_old'
     CALL SPACE_dealloc_array( data%X_old,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%Y'
     CALL SPACE_dealloc_array( data%Y,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%Z'
     CALL SPACE_dealloc_array( data%Z,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%G_x'
     CALL SPACE_dealloc_array( data%G_x,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%G_y'
     CALL SPACE_dealloc_array( data%G_y,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%G_x_old'
     CALL SPACE_dealloc_array( data%G_x_old,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

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

!  End of subroutine AGD_terminate

     END SUBROUTINE AGD_terminate

!-  G A L A H A D -  A G D _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE AGD_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( AGD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( AGD_control_type ), INTENT( IN ) :: control
     TYPE ( AGD_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL AGD_terminate( data%agd_data, data%agd_control, data%agd_inform )
     inform = data%agd_inform

!  deallocate any internal problem arrays

     array_name = 'agd: data%nlp%X'
     CALL SPACE_dealloc_array( data%nlp%X,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'agd: data%nlp%G'
     CALL SPACE_dealloc_array( data%nlp%G,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine AGD_full_terminate

     END SUBROUTINE AGD_full_terminate

   END MODULE GALAHAD_AGD_precision
