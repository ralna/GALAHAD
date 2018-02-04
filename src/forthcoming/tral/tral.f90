! THIS VERSION: GALAHAD 2.6 - 26/06/2013 AT 08:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ T R A L   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. June 25th 2012

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_TRAL_double

!     ------------------------------------------------------------------
!    |                                                                   |
!    | TRAL, a trust-region augmented Lagrangian algorithm for           |
!    |       nonlinearly-constrained optimization                        |
!    |                                                                   |
!    |   Aim: find a (local) minimizer of the objective f(x)             |
!    |        subject to x_l <= x <= x_u and c_l <= c(x) <= c_u          |
!    |                                                                   |
!    |        by minimizing                                              |
!    |                                                                   |
!    |        phi(x,c) = f(x) + y^T(c(x)-c) + 0.5 rho ||c(x)-c)||_2^2    |
!    |        subject to x_l <= x <= x_u and c_l <= c(x) <= c_u          |
!    |                                                                   |
!    |  for a suitable sequence of y and rho                             |
!    |                                                                   |
!    |  Note: grad phi = ( g(x) +J^T(x) y(x,rho) )                       |
!    |                   (       - y(x,rho)      )                       |
!    |   Hess phi = ( H(x,y(x,rho)) + rho J^T(x) J(x)  - rho J^T(x) )    |
!    |              (    - rho J(x)                      rho I      )    |
!    |   and                                                             |
!    |   Hess phi (p) = ( H(x,y(x,rho) p + J^T(x) r )                    |
!    |            (q)   (          - r              )                    |
!    |   where y(x,rho) = y + rho (c(x)-c), H(x,y) = Hess Lagrangian     |
!    |   and r = rho [ J(x) p - q )                                      |
!    |                                                                   |
!     ------------------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SLS_double, ONLY: SLS_coord_to_extended_csr
     USE GALAHAD_PSLS_double
     USE GALAHAD_GLTR_double
     USE GALAHAD_TRS_double
     USE LANCELOT_CAUCHY_double
     USE LANCELOT_CG_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM, INFINITY_NORM
     USE GALAHAD_STRING_double, ONLY: STRING_integer_6
     USE GALAHAD_MOP_double, ONLY: mop_Ax
!    USE SPDSOL
!    USE HSL_MI13

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: TRAL_initialize, TRAL_read_specfile, TRAL_solve,                &
               TRAL_terminate, NLPT_problem_type, NLPT_userdata_type,          &
               SMT_type, SMT_put

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER, PARAMETER  :: nskip_prec_max = 0
     INTEGER, PARAMETER  :: history_max = 100
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: sixteenth = 0.0625_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: sixteen = 16.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = ten ** ( - 5 )
     REAL ( KIND = wp ), PARAMETER :: tenm8 = ten ** ( - 9 )
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: rho_quad = epsmch * ten ** 4

     REAL ( KIND = wp ), PARAMETER :: gamma_1 = sixteenth
     REAL ( KIND = wp ), PARAMETER :: gamma_2 = half
     REAL ( KIND = wp ), PARAMETER :: gamma_3 = two
     REAL ( KIND = wp ), PARAMETER :: gamma_4 = sixteen
     REAL ( KIND = wp ), PARAMETER :: mu_1 = one - ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: mu_2 = point1
     REAL ( KIND = wp ), PARAMETER :: theta = half
     REAL ( KIND = wp ), PARAMETER :: stptol = point1

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: TRAL_control_type

!   error and warning diagnostics occur on stream error 
   
       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!   any printing will start on this iteration

       INTEGER :: start_print = - 1

!   any printing will stop on this iteration

       INTEGER :: stop_print = - 1

!   the number of iterations between printing

       INTEGER :: print_gap = 1

!   the maximum number of iterations performed

       INTEGER :: maxit = 100

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   more_toraldo >= 1 gives the number of More'-Toraldo projected searches 
!     to be used to improve upon the Cauchy point, anything else is for the 
!     standard add-one-at-a-time CG search

       INTEGER :: more_toraldo = 0

!   non-monotone <= 0 monotone strategy used, anything else non-monotone
!     strategy with this history length used

       INTEGER :: non_monotone = 1

!   specify the model used. Possible values are
!
!      0  dynamic (*not yet implemented*)
!      1  first-order (no Hessian)
!      2  second-order (exact Hessian)
!      3  barely second-order (identity Hessian)
!      4  secant second-order (limited-memory BFGS) (*not yet implemented*)
!      5  secant second-order (limited-memory SR1) (*not yet implemented*)

       INTEGER :: model = 2

!   specify the preconditioner used for iterative methods.
!    Possible values for P are
!
!     -3  users own preconditioner
!     -2  P = limited-memory BFGS matrix (with %lbfgs_vectors history)
!     -1  identity
!      0  automatic (*not yet implemented*)
!      1  diagonal, P = diag( max( Hessian, %min_diagonal ) )
!      2  banded, P = band( Hessian ) with semi-bandwidth %semi_bandwidth
!      3  full factorization, P = Hessian, Schnabel-Eskow modification
!      4  full factorization, P = Hessian, GMPS modification (*not yet *)
!      5  incomplete factorization of Hessian, Lin-More'
!      6  incomplete factorization of Hessian, Munskgaard (*not yet *)
!      7  expanding band of Hessian (*not yet implemented*)

       INTEGER :: preconditioner = 1

!   specify the semi-bandwidth of the band matrix P if required

       INTEGER :: semi_bandwidth = 5

!   number of vectors used by the L-BFGS matrix P if required

       INTEGER :: lbfgs_vectors = 10

!   number of vectors used by the Lin-More' incomplete factorization 
!    matrix P if required

       INTEGER :: icfs_vectors = 10

!   overall convergence tolerances. The iteration will terminate when the
!     norm of the gradient of the objective function is smaller than 
!       MAX( %stop_pg_absolute, %stop_pg_relative * norm of the initial gradient
!     or if the step is less than %stop_s

       REAL ( KIND = wp ) :: stop_pg_absolute = tenm5
       REAL ( KIND = wp ) :: stop_pg_relative = tenm8
       REAL ( KIND = wp ) :: stop_s = epsmch

!   try to pick a good initial trust-region radius using %advanced_start
!    iterates of a variant on the strategy of Sartenaer SISC 18(6)1990:1788-1803

       INTEGER :: advanced_start = 0

!   any bound larger than infinity in modulus will be regarded as infinite 

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   initial value for the trust-region radius

       REAL ( KIND = wp ) :: initial_radius = hundred

!   maximum permitted trust-region radius

       REAL ( KIND = wp ) :: maximum_radius = ten ** 8

!  The initial value of the penalty parameter

     control%initial_rho = ten

!   required relative reduction in the resuiduals from CG

       REAL ( KIND = wp ) :: stop_rel_cg = 0.01_wp

!   a potential iterate will only be accepted if the actual decrease
!    f - f(x_new) is larger than %eta_successful times that predicted
!    by a quadratic model of the decrease. The trust-region radius will be
!    increased if this relative decrease is greater than %eta_very_successful
!    but smaller than %eta_too_successful

       REAL ( KIND = wp ) :: eta_successful = ten ** ( - 8 )
       REAL ( KIND = wp ) :: eta_very_successful = point9
       REAL ( KIND = wp ) :: eta_too_successful = two

!   on very successful iterations, the trust-region radius will be increased by
!    the factor %radius_increase, while if the iteration is unsucceful, the 
!    radius will be decreased by a factor %radius_reduce but no more than
!    %radius_reduce_max

       REAL ( KIND = wp ) :: radius_increase = two
       REAL ( KIND = wp ) :: radius_reduce = half
       REAL ( KIND = wp ) :: radius_reduce_max = sixteenth

!   the smallest value the onjective function may take before the problem
!    is marked as unbounded

       REAL ( KIND = wp ) :: obj_unbounded = - epsmch ** ( - 2 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

!   is the Hessian matrix of second derivatives available or is access only
!    via matrix-vector products?

       LOGICAL :: hessian_available = .TRUE.

!   use a direct (factorization) or (preconditioned) iterative method to 
!    find the search direction

       LOGICAL :: subproblem_direct = .FALSE.

!   is a retrospective strategy to be used to update the trust-region radius?

       LOGICAL :: retrospective_trust_region = .FALSE.

!   should the radius be renormalized to account for a change in preconditioner?

       LOGICAL :: renormalize_radius = .FALSE.

!   should an ellipsoidal trust-region be used rather than an infinity norm one?

       LOGICAL :: two_norm_tr = .FALSE.

!   is the exact Cauchy point required rather than an approximation?

       LOGICAL :: exact_gcp = .TRUE.

!  should the minimizer of the quadratic model within the intersection of the 
!   trust-region and feasible box be found (to a prescribed accuracy) rather
!   than a (much) cheaper approximation?

       LOGICAL :: accurate_bqp = .FALSE.

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

!  control parameters for TRS

       TYPE ( TRS_control_type ) :: TRS_control

!  control parameters for GLTR

       TYPE ( GLTR_control_type ) :: GLTR_control

!  control parameters for PSLS

       TYPE ( PSLS_control_type ) :: PSLS_control

     END TYPE TRAL_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: TRAL_time_type

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

       REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

       REAL ( KIND = wp ) :: clock_preprocess = 0.0

!  the clock time spent analysing the required matrices prior to factorization

       REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

       REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

       REAL ( KIND = wp ) :: clock_solve = 0.0

     END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

     TYPE, PUBLIC :: TRAL_inform_type

!  return status. See TRAL_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER :: iter = 0

!  the total number of CG iterations performed

       INTEGER :: cg_iter = 0

!  the maximum number of CG iterations allowed per iteration

       INTEGER :: cg_maxit

!  the total number of evaluations of the objection function

       INTEGER :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER :: g_eval = 0

!  the total number of evaluations of the Hessian of the objection function

       INTEGER :: h_eval = 0

!  the number of free variables

       INTEGER :: n_free = - 1

!  the maximum number of factorizations in a sub-problem solve

       INTEGER :: factorization_max = 0

!  the return status from the factorization

       INTEGER :: factorization_status = 0

!   the maximum number of entries in the factors

        INTEGER ( KIND = long ) :: max_entries_factors = 0

!  the total integer workspace required for the factorization

       INTEGER :: factorization_integer = - 1

!  the total real workspace required for the factorization

       INTEGER :: factorization_real = - 1

!  the average number of factorizations per sub-problem solve

       REAL ( KIND = wp ) :: factorization_average = zero

!  the value of the objective function at the best estimate of the solution 
!   determined by TRAL_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the norm of the projected gradient of the objective function at the best 
!   estimate of the solution determined by TRAL_solve

       REAL ( KIND = wp ) :: norm_pg = HUGE( one )

!  the current value of the trust-region radius

       REAL ( KIND = wp ) :: radius = zero

!  the current value of the penalty parameter

       REAL ( KIND = wp ) :: rho = zero

!  timings (see above)

       TYPE ( TRAL_time_type ) :: time

!  inform parameters for TRS

       TYPE ( TRS_inform_type ) :: TRS_inform

!  inform parameters for GLTR

       TYPE ( GLTR_info_type ) :: GLTR_inform

!  inform parameters for PSLS

       TYPE ( PSLS_inform_type ) :: PSLS_inform

     END TYPE TRAL_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: TRAL_data_type
       INTEGER :: branch = 1
       INTEGER :: eval_status, out, start_print, stop_print, advanced_start_iter
       INTEGER :: print_level, print_level_gltr, print_level_trs, ref( 1 )
       INTEGER :: len_history, ibound, ipoint, icp, lbfgs_mem, max_hist
       INTEGER :: nprec, nskip_lbfgs, nskip_prec, non_monotone_history, jumpto
       INTEGER :: h_ne, nnz_p_l, nnz_p_u, nnz_hp, n_prods, more_toraldo_its
       INTEGER :: print_gap, n_fix, nfree_cp, itercg, n_phi

       REAL :: time_start, time_record, time_now
       REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
       REAL ( KIND = wp ) :: phi, phi_trial, phi_ref, phi_best
       REAL ( KIND = wp ) :: f_trial, m_best, ratio
       REAL ( KIND = wp ) :: radius, old_radius, radius_trial, etat, ometat
       REAL ( KIND = wp ) :: dxtdg, dgtdg, df, stg, hstbs, s_norm, radius_max
       REAL ( KIND = wp ) :: stop_pg, s_new_norm, rho_g, g_model, rg_norm
       REAL ( KIND = wp ) :: diagonal_min, diagonal_max, step, dxsqr, cg_stop
       REAL ( KIND = wp ) :: model, model_new, model_cp, model_start, step_max

       LOGICAL :: printi, printt, printm, printw, printd
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printm, set_printw, set_printd
       LOGICAL :: monotone, new_h, got_h, poor_model, reuse_cp
       LOGICAL :: reverse_f, reverse_g, reverse_h, reverse_hprod, reverse_prec
       LOGICAL :: dense_p, diagonal_preconditioner, refactorize
       LOGICAL :: first_cp, no_bounds, no_bounds_cg, constrained

       CHARACTER ( LEN = 1 ) :: negcur, bndry, perturb, hard

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: V_status
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEX_nz_p
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEX_nz_hp
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEX_used_hp
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: POSITION_diagonal
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: FIX
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: MAP
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V_phi
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VC_phi
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_phi
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_best
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V_cauchy
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HP
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WK
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WK2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHO
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ALPHA
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DMO_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PHI_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_diagonal
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX_bqp
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: DX
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: DG
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BANDH
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BND
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BND_radius

       TYPE ( SMT_type ) :: H_by_cols
       TYPE ( CAUCHY_save_type ) :: CAUCHY_save
       TYPE ( CG_save_type ) :: CG_save

!  copy of controls

       TYPE ( TRAL_control_type ) :: control

!  data and history for TRS

       TYPE ( TRS_data_type ) :: TRS_data
       TYPE ( TRS_history_type ), DIMENSION( history_max ) :: history

!  data for GLTR

       TYPE ( GLTR_data_type ) :: GLTR_data

!  data for PSLS

       TYPE ( PSLS_data_type ) :: PSLS_data

     END TYPE TRAL_data_type

   CONTAINS

!-*-*-  G A L A H A D -  T R A L _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE TRAL_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for TRAL controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TRAL_data_type ), INTENT( INOUT ) :: data
     TYPE ( TRAL_control_type ), INTENT( OUT ) :: control
     TYPE ( TRAL_inform_type ), INTENT( OUT ) :: inform        

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initalize TRS components

     CALL TRS_initialize( data%TRS_data, control%TRS_control,                  &
                          inform%TRS_inform )
     control%TRS_control%prefix = '" - TRS:"                     '

!  initalize GLTR components

     CALL GLTR_initialize( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )
     control%GLTR_control%prefix = '" - GLTR:"                    '

!  initalize PSLS components

     CALL PSLS_initialize( data%PSLS_data, control%PSLS_control,               &
                           inform%PSLS_inform )
     control%PSLS_control%prefix = '" - PSLS:"                    '

!  initial private data. Set branch for initial entry

     data%branch = 1

     RETURN

!  End of subroutine TRAL_initialize

     END SUBROUTINE TRAL_initialize

!-*-*-*-*-   T R A L _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE TRAL_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by TRAL_initialize could (roughly) 
!  have been set as:

! BEGIN TRAL SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  maximum-number-of-iterations                    100
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  advanced-start                                  5
!  more-toraldo-search-length                      0
!  history-length-for-non-monotone-descent         0
!  model-used                                      2
!  preconditioner-used                             1
!  semi-bandwidth-for-band-norm                    5
!  number-of-lbfgs-vectors                         5
!  number-of-lin-more-vectors                      5
!  infinity-value                                  1.0D+19
!  absolute-gradient-accuracy-required             1.0D-5
!  relative-gradient-reduction-required            1.0D-5
!  minimum-step-allowed                            2.0D-16
!  initial-trust-region-radius                     1.0D+0
!  maximum-trust-region-radius                     1.0D+19
!  initial-penalty-parameter                       1.0D+1
!  inner-iteration-relative-accuracy-required      0.01
!  successful-iteration-tolerance                  0.01
!  very-successful-iteration-tolerance             0.9
!  too-successful-iteration-tolerance              2.0
!  trust-region-increase-factor                    2.0
!  trust-region-decrease-factor                    0.5
!  trust-region-maximum-decrease-factor            0.0625
!  minimum-objective-before-unbounded              -1.0D+32
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  hessian-available                               yes
!  sub-problem-direct                              no
!  two-norm-trust-region-used                      no
!  exact-GCP-used                                  yes
!  retrospective-trust-region                      no
!  subproblem-solved-accurately                    no
!  renormalize-radius                              no
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END TRAL SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TRAL_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: start_print = print_level + 1
     INTEGER, PARAMETER :: stop_print = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print + 1
     INTEGER, PARAMETER :: maxit = print_gap + 1
     INTEGER, PARAMETER :: alive_unit = maxit + 1
     INTEGER, PARAMETER :: more_toraldo = alive_unit + 1
     INTEGER, PARAMETER :: non_monotone = more_toraldo + 1
     INTEGER, PARAMETER :: model = non_monotone + 1
     INTEGER, PARAMETER :: preconditioner = model + 1
     INTEGER, PARAMETER :: semi_bandwidth = preconditioner + 1
     INTEGER, PARAMETER :: lbfgs_vectors = semi_bandwidth + 1
     INTEGER, PARAMETER :: icfs_vectors = lbfgs_vectors + 1
     INTEGER, PARAMETER :: advanced_start = icfs_vectors + 1
     INTEGER, PARAMETER :: infinity = advanced_start + 1
     INTEGER, PARAMETER :: stop_pg_absolute = infinity + 1
     INTEGER, PARAMETER :: stop_pg_relative = stop_pg_absolute + 1
     INTEGER, PARAMETER :: stop_s = stop_pg_relative + 1
     INTEGER, PARAMETER :: initial_radius = stop_s + 1
     INTEGER, PARAMETER :: maximum_radius = initial_radius + 1
     INTEGER, PARAMETER :: initial_rho = maximum_radius + 1
     INTEGER, PARAMETER :: stop_rel_cg = initial_rho + 1
     INTEGER, PARAMETER :: eta_successful = stop_rel_cg + 1
     INTEGER, PARAMETER :: eta_very_successful = eta_successful + 1
     INTEGER, PARAMETER :: eta_too_successful = eta_very_successful + 1
     INTEGER, PARAMETER :: radius_increase = eta_too_successful + 1
     INTEGER, PARAMETER :: radius_reduce = radius_increase + 1
     INTEGER, PARAMETER :: radius_reduce_max = radius_reduce + 1
     INTEGER, PARAMETER :: obj_unbounded = radius_reduce_max + 1
     INTEGER, PARAMETER :: cpu_time_limit = obj_unbounded + 1
     INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER, PARAMETER :: hessian_available = clock_time_limit + 1
     INTEGER, PARAMETER :: subproblem_direct = hessian_available + 1
     INTEGER, PARAMETER :: retrospective_trust_region = subproblem_direct + 1
     INTEGER, PARAMETER :: renormalize_radius = retrospective_trust_region + 1
     INTEGER, PARAMETER :: two_norm_tr = renormalize_radius + 1
     INTEGER, PARAMETER :: exact_gcp = two_norm_tr + 1
     INTEGER, PARAMETER :: accurate_bqp = exact_gcp + 1
     INTEGER, PARAMETER :: space_critical = accurate_bqp + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'TRAL '
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
     spec( more_toraldo )%keyword = 'more-toraldo-search-length'
     spec( non_monotone )%keyword = 'history-length-for-non-monotone-descent'
     spec( model )%keyword = 'model-used'
     spec( preconditioner )%keyword = 'norm-used'
     spec( semi_bandwidth )%keyword = 'semi-bandwidth-for-band-norm'
     spec( lbfgs_vectors )%keyword = 'number-of-lbfgs-vectors'
     spec( icfs_vectors )%keyword = 'number-of-lin-more-vectors'
     spec( advanced_start )%keyword = 'advanced-start'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( stop_pg_absolute )%keyword = 'absolute-gradient-accuracy-required'
     spec( stop_pg_relative )%keyword = 'relative-gradient-reduction-required'
     spec( stop_s )%keyword = 'minimum-step-allowed'
     spec( initial_radius )%keyword = 'initial-trust-region-radius'
     spec( maximum_radius )%keyword = 'maximum-trust-region-radius'
     spec( initial_rho )%keyword = 'initial-penalty parameter'
     spec( stop_rel_cg )%keyword = 'inner-iteration-relative-accuracy-required'
     spec( eta_successful )%keyword = 'successful-iteration-tolerance'
     spec( eta_very_successful )%keyword = 'very-successful-iteration-tolerance'
     spec( eta_too_successful )%keyword = 'too-successful-iteration-tolerance'
     spec( radius_increase )%keyword = 'trust-region-increase-factor'
     spec( radius_reduce )%keyword = 'trust-region-decrease-factor'
     spec( radius_reduce_max )%keyword = 'trust-region-maximum-decrease-factor'
     spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

     spec( hessian_available )%keyword = 'hessian-available'
     spec( subproblem_direct )%keyword = 'sub-problem-direct'
     spec( retrospective_trust_region )%keyword = 'retrospective-trust-region'
     spec( renormalize_radius )%keyword = 'renormalize-radius'
     spec( two_norm_tr )%keyword = 'two-norm-trust-region-used'
     spec( exact_gcp )%keyword = 'exact-GCP-used'
     spec( accurate_bqp )%keyword = 'subproblem-solved-accurately'
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
     CALL SPECFILE_assign_value( spec( more_toraldo ),                         &
                                 control%more_toraldo,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( non_monotone ),                         &
                                 control%non_monotone,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( model ),                                &
                                 control%model,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( preconditioner ),                       &
                                 control%preconditioner,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( semi_bandwidth ),                       &
                                 control%semi_bandwidth,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lbfgs_vectors ),                        &
                                 control%lbfgs_vectors,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( icfs_vectors ),                         &
                                 control%icfs_vectors,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( advanced_start ),                       &
                                 control%advanced_start,                       &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_pg_absolute ),                     &
                                 control%stop_pg_absolute,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_pg_relative ),                     &
                                 control%stop_pg_relative,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_s ),                               &
                                 control%stop_s,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_radius ),                       &
                                 control%initial_radius,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maximum_radius ),                       &
                                 control%maximum_radius,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_rho ),                          &
                                 control%initial_rho,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_cg ),                          &
                                 control%stop_rel_cg,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_successful ),                       &
                                 control%eta_successful,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_very_successful ),                  &
                                 control%eta_very_successful,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( eta_too_successful ),                   &
                                 control%eta_too_successful,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius_increase ),                      &
                                 control%radius_increase,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius_reduce ),                        &
                                 control%radius_reduce,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius_reduce_max ),                    &
                                 control%radius_reduce_max,                    &
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

     CALL SPECFILE_assign_value( spec( hessian_available ),                    &
                                 control%hessian_available,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( subproblem_direct ),                    &
                                 control%subproblem_direct,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( retrospective_trust_region ),           &
                                 control%retrospective_trust_region,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( renormalize_radius ),                   &
                                 control%renormalize_radius,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( two_norm_tr ),                          &
                                 control%two_norm_tr,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( exact_gcp ),                            &
                                 control%exact_gcp,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( accurate_bqp ),                         &
                                 control%accurate_bqp,                         &
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

!  read the controls for the sub-problem solvers and preconditioner

     IF ( PRESENT( alt_specname ) ) THEN
       CALL TRS_read_specfile( control%TRS_control, device,                    &
                                alt_specname = TRIM( alt_specname ) // '-TRS' )
       CALL GLTR_read_specfile( control%GLTR_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-GLTR' )
       CALL PSLS_read_specfile( control%PSLS_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-PSLS' )
     ELSE
       CALL TRS_read_specfile( control%TRS_control, device )
       CALL GLTR_read_specfile( control%GLTR_control, device )
       CALL PSLS_read_specfile( control%PSLS_control, device )
     END IF

     RETURN

!  End of subroutine TRAL_read_specfile

     END SUBROUTINE TRAL_read_specfile

!-*-*-*-  G A L A H A D -  T R A L _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE TRAL_solve( nlp, control, inform, data, userdata,              &
                           eval_F, eval_G, eval_H, eval_HPROD,                 &
                           eval_FC, eval_GJ, eval_HL, eval_HLPROD, eval_PREC )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  TRAL_solve, a trust-region method for finding a local minimizer of a given 
!    function where the variables are constrained to lie in a "box"

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_TRAL. 
!
!  ** NB. default real/complex means double precision real/complex in 
!  ** GALAHAD_TRAL_double
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
!    the values of the entries of the  lower triangular part of the Hessian 
!    matrix H in any of the available storage schemes.
!
!   H%row is a rank-one allocatable array of type default integer, that holds 
!    the row indices of the  lower triangular part of H in the sparse 
!    co-ordinate storage scheme. It need not be allocated for any of the other 
!    three schemes.
!
!   H%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the  lower triangular part of H in either
!    the sparse co-ordinate, or the sparse row-wise storage scheme. It need not 
!    be allocated when the dense or diagonal storage schemes are used.
!
!   H%ptr is a rank-one allocatable array of dimension n+1 and type default 
!    integer, that holds the starting position of  each row of the  lower 
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
! control is a scalar variable of type TRAL_control_type. See TRAL_initialize
!  for details
!
! inform is a scalar variable of type TRAL_inform_type. On initial entry, 
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
!        control%maxit is too small, but may also be symptomatic of 
!        a badly scaled problem.
!   -19. The CPU time limit has been reached. This may happen if
!        control%cpu_time_limit is too small, but may also be symptomatic of 
!        a badly scaled problem.
!   -40. The user has forced termination of solver by removing the file named 
!        control%alive_file from unit unit control%alive_unit.
! 
!   [for unconstrained problems]
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
!     6. The user should compute the product u = P(x)v of their preconditioner 
!        P(x) at the point x indicated in nlp%X with the vector v and then
!        re-enter the subroutine. The vectors v is given in data%V, the
!        resulting vector u = P(x)v should be set in data%U and 
!        data%eval_status should be set to 0. If the user is unable to evaluate
!        the product - for instance, if a component of the preconditioner is 
!        undefined at x - the user need not set data%U, but should then set 
!        data%eval_status to a non-zero value.
!
!   [for constrained problems]
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
!  factorization_status is a scalar variable of type default integer, that 
!   gives the return status from the matrix factorization.
!
!  factorization_integer is a scalar variable of type default integer,
!   that gives the amount of integer storage used for the matrix factorization.
!
!  factorization_real is a scalar variable of type default integer, 
!   that gives the amount of real storage used for the matrix factorization.
!
!  f_eval is a scalar variable of type default integer, that gives the
!   total number of objective function evaluations performed.
!
!  g_eval is a scalar variable of type default integer, that gives the
!   total number of objective function gradient evaluations performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of objective function Hessian evaluations performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_pg is a scalar variable of type default real, that holds the value of 
!   the norm of the projected gradient of the objective function at the best 
!   estimate of the solution found.
!
!  time is a scalar variable of type TRAL_time_type whose components are used to
!   hold elapsed CPU and clock times for the various parts of the calculation.
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total CPU time spent in the package.
!
!    preprocess is a scalar variable of type default real, that gives the
!      CPU time spent reordering the problem to standard form prior to solution.
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
!    clock_preprocess is a scalar variable of type default real, that gives
!      the clock time spent reordering the problem to standard form prior 
!      to solution.
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
!  data is a scalar variable of type TRAL_data_type used for internal data.
!
!  userdata is a scalar variable of type NLPT_userdata_type which may be used 
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
! [for unconstrained problems]
!
!  eval_F is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_F is not present, TRAL_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient 
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in 
!   G, and the status variable set to 0. If the evaluation is impossible at X, 
!   status should be set to a nonzero value. If eval_G is not present, 
!   TRAL_solve will return to the user with inform%status = 3 each time an 
!   evaluation is required.
!
!  eval_H is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) of the objective function evaluated at x=X must be returned in
!   H in the same order as presented in nlp%H, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero 
!   value. If eval_H is not present, TRAL_solve will return to the user with 
!   inform%status = 4 each time an evaluation is required.
!
!  eval_HPROD is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The sum u + nabla_xx f(x) v of the 
!   product of the Hessian nabla_xx f(x) of the objective function evaluated 
!   at x=X with the vector v=V and the vector u=U must be returned in U, and the
!   status variable set to 0. If the evaluation is impossible at X, status 
!   should be set to a nonzero value. If eval_HPROD is not present, TRAL_solve 
!   will return to the user with inform%status = 5 each time an evaluation is 
!   required.
!
!  eval_PREC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = P(x) v of the 
!   user's preconditioner P(x) evaluated at x=X with the vector v=V, the result 
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_PREC 
!   is not present, TRAL_solve will return to the user with inform%status = 6 
!   each time an evaluation is required.
!
! [for constrained problems]
!
!  eval_FC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If C is present, the values of the constraint functions
!   c(x) evaluated at x=X must be returned in C, and the status variable set
!   to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_FC is not present, FASTR_solve will
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
!   nonzero value. If eval_GJ is not present, FASTR_solve will return to the 
!   user with inform%status = 3 or 5 each time an evaluation is required.
!
!  eval_HL is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) - sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated 
!   at x=X and y=Y must be returned in H_val in the same order as presented in 
!   nlp%H, and the status variable set to 0. If the evaluation is impossible 
!   at X, status should be set to a nonzero value. If eval_HL is not present, 
!   FASTR_solve will return to the user with inform%status = 4 or 5 each time 
!   an evaluation is required.
!
!  eval_HLPROD is an optional subroutine which if present must have
!   the arguments given below (see the interface blocks). The sum 
!   u + nabla_xx ( f(x) - sum_i=1^m y_i c_i(x) ) v of the product of the Hessian
!   nabla_xx f(x) + sum_i=1^m y_i c_i(x) of the Lagrangian function evaluated 
!   at x=X and y=Y with the vector v=V and the vector u=U must be returned in U,
!   and the status variable set to 0. If the evaluation is impossible at X, 
!   status should be set to a nonzero value. If eval_HPROD is not present, 
!   FASTR_solve will return to the user with inform%status = 6 each time an 
!   evaluation is required.
!
!  eval_PREC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = P(x) v of the 
!   user's preconditioner P(x) evaluated at x=X with the vector v=V, the result 
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_PREC 
!   is not present, TRU_solve will return to the user with inform%status = 7
!   each time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( TRAL_control_type ), INTENT( IN ) :: control
     TYPE ( TRAL_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( TRAL_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_F, eval_G, eval_H, eval_HPROD, eval_PREC,                &
                 eval_FC, eval_GJ, eval_HL, eval_HLPROD

!----------------------------------
!   I n t e r f a c e   B l o c k s 
!----------------------------------

     INTERFACE
       SUBROUTINE eval_F( status, X, userdata, f )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), INTENT( OUT ) :: f
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_F

       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G

       SUBROUTINE eval_H( status, X, userdata, Hval )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_H

       SUBROUTINE eval_HPROD( status, X, userdata, U, V, got_h )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HPROD

       SUBROUTINE eval_FC( status, X, userdata, F, C )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( kind = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( kind = wp ), OPTIONAL, INTENT( OUT ) :: F
       REAL ( kind = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FC

       SUBROUTINE eval_GJ( status, X, userdata, G, J_val )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G
       REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: J_val
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_GJ

       SUBROUTINE eval_HL( status, X, Y, userdata, H_val, no_f ) 
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) ::H_val
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       END SUBROUTINE eval_HL

       SUBROUTINE eval_HLPROD( status, X, Y, userdata, U, V, no_f, got_h )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HLPROD

       SUBROUTINE eval_PREC( status, X, userdata, U, V )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V, X
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_PREC
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, ic, ir, i_fixed, k, l, n_active, facts_this_solve
     INTEGER :: duplicates, out_of_range, upper, missing_diagonals
     REAL ( KIND = wp ) :: val, beta, ared, prered, rounding, delta_norm
     REAL ( KIND = wp ) :: tau, tau_1, tau_2, tau_min, tau_max, distan
     REAL ( KIND = wp ) :: c_diff, c_norm
     LOGICAL :: alive
     CHARACTER ( LEN = 6 ) :: char_iter, char_facts, char_sit, char_sit2
     CHARACTER ( LEN = 6 ) :: char_active
     CHARACTER ( LEN = 80 ) :: array_name
!    REAL ( KIND = wp ), DIMENSION( nlp%n ) :: V

!  prefix for all output 

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) data%branch = 1

     SELECT CASE ( data%branch )
     CASE ( 1 )  ! initialization
       GO TO 10
     CASE ( 2 )  ! initial objective evaluation
       GO TO 20
     CASE ( 3 )  ! initial gradient evaluation
       GO TO 30
     CASE ( 4 )  ! Hessian evaluation
       GO TO 110
     CASE ( 5 )  ! Hessian-vector product or preconditioner
       GO TO 210
     CASE ( 6 )  ! Hessian-vector product
       GO TO 480
     CASE ( 7 )  ! objective evaluation
       GO TO 520
     CASE ( 8 )  ! Hessian-vector product
       GO TO 540
     CASE ( 9 )  ! gradient evaluation
       GO TO 550
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
       DO i = 1, nlp%n
         IF ( nlp%C_l( i ) > nlp%C_u( i ) ) THEN
           inform%status = GALAHAD_error_bad_bounds
           GO TO 990
         END IF
       END DO
     END IF

!  has the problem bounds?

    IF ( data%constrained ) THEN
      data%no_bounds = COUNT( nlp%X_l >= - control%infinity ) +                &
                       COUNT( nlp%X_u <= control%infinity ) +                  &
                       COUNT( nlp%C_l >= - control%infinity ) +                &
                       COUNT( nlp%C_u <= control%infinity ) == 0
    ELSE
      data%no_bounds = COUNT( nlp%X_l >= - control%infinity ) +                &
                       COUNT( nlp%X_u <= control%infinity ) == 0
    END IF

!  record the problem dimensions

     data%n_phi = nlp%n + nlp%m
     nlp%H%n = nlp%n

!  allocate sufficient space for the problem

     array_name = 'TRAL: data%V_phi'
     CALL SPACE_resize_array( data%n_phi, data%V_phi, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%G_phi'
     CALL SPACE_resize_array( data%n_phi, data%G_phi, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%VC_phi'
     CALL SPACE_resize_array( data%n_phi, data%VC_phi, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%Y'
     CALL SPACE_resize_array( nlp%m, data%Y, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%V_current'
     CALL SPACE_resize_array( data%n_phi, data%V_current, inform%status,       &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%G_current'
     CALL SPACE_resize_array( daat%n_phi, data%G_current, inform%status,       &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%S'
     CALL SPACE_resize_array( data%n_phi, data%S, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%V'
     CALL SPACE_resize_array( nlp%n, data%V, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%WK'
     CALL SPACE_resize_array( data%n_phi, data%WK, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%WK2'
     CALL SPACE_resize_array( nlp%n, data%WK2, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%V_cauchy'
     CALL SPACE_resize_array( nlp%n, data%V_cauchy, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%V_trial'
     CALL SPACE_resize_array( data%n_phi, data%V_trial, inform%status,         &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%C_trial'
     CALL SPACE_resize_array( nlp%m, data%C_trial, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( control%advanced_start > 0 ) THEN
       array_name = 'TRAL: data%X_best'
       CALL SPACE_resize_array( nlp%n, data%X_best, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

     array_name = 'TRAL: data%P'
     CALL SPACE_resize_array( data%n_phi, data%P, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%HP'
     CALL SPACE_resize_array( data%n_phi, data%HP, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%BND'
     CALL SPACE_resize_array( data%n_phi, 2, data%BND, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( data%control%more_toraldo > 0 ) THEN
       array_name = 'TRAL: data%BND_radius'
       CALL SPACE_resize_array( data%n_phi, 2, data%BND_radius, inform%status, &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

     IF ( control%accurate_bqp ) THEN
       array_name = 'TRAL: data%DX_bqp'
       CALL SPACE_resize_array( data%n_phi, data%DX_bqp, inform%status,        &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  V_status( j ), j = 1, ..., n_phi, will contain the status of the
!  j-th variable as the current iteration progresses. Possible values
!  are 0 if the variable lies away from its bounds, 1 and 2 if it lies
!  on its lower or upper bounds (respectively) - these may be problem
!  bounds or trust-region bounds, and 3 or 4 if the variable is fixed

     array_name = 'TRAL: data%V_status'
     CALL SPACE_resize_array( data%n_phi, data%V_status, inform%status,        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

!  INDEX_nz_p( j ), j = nnz_p_l, ..., nnz_p_u will give the indices of the 
!  nonzeros in the vector p required for the matrix-vector product H * p

     array_name = 'TRAL: data%INDEX_nz_p'
     CALL SPACE_resize_array( data%n_phi, data%INDEX_nz_p, inform%status,      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

!  INDEX_nz_hp( j ), j = 1, ..., nnz_hp will give the indices of the nonzeros
!  in the vector obtained as a result of the matrix-vector product H * p

     array_name = 'TRAL: data%INDEX_nz_hp'
     CALL SPACE_resize_array( data%n_phi, data%INDEX_nz_hp, inform%status,     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

!  INDEX_used_hp( j ) = 1, ..., n will have the value k if the j-th 
!  component of H * p was nonzero in the the k-th product and < k otherwise

     array_name = 'TRAL: data%INDEX_used_hp'
     CALL SPACE_resize_array( nlp%n, data%INDEX_used_hp, inform%status,        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

!  FIX( j ) = 1, ..., n_fix will give the indices of variables fixed as the CG
!  or More-Toraldo search proceeds

     array_name = 'TRAL: data%FIX'
     IF ( control%more_toraldo > 0 ) THEN
       CALL SPACE_resize_array( data%n_phi, data%FIX, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
     ELSE
       CALL SPACE_resize_array( 1, data%FIX, inform%status,                    &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
     END IF
     IF ( inform%status /= 0 ) GO TO 980

!  ensure that the data is consistent

     data%control = control
     data%control%TRS_control%initial_multiplier = zero
     data%control%PSLS_control%preconditioner = control%preconditioner
     data%control%PSLS_control%semi_bandwidth = control%semi_bandwidth
     data%control%PSLS_control%icfs_vectors = control%icfs_vectors

     data%non_monotone_history = data%control%non_monotone
     IF ( data%non_monotone_history <= 0 ) data%non_monotone_history = 1
     data%monotone = data%non_monotone_history == 1
     inform%radius = data%control%initial_radius
     inform%rho = data%control%initial_rho
     data%etat = half * ( data%control%eta_very_successful +                   &
                  data%control%eta_successful )
     data%ometat = one - data%etat
     data%advanced_start_iter = 0
     data%lbfgs_mem = MAX( 1, data%control%lbfgs_vectors )
     data%negcur = ' '
     data%reuse_cp = .FALSE.
     inform%max_entries_factors = 0

!  decide how much reverse communication is required

     IF ( data%constrained ) THEN
       data%reverse_f = .NOT. PRESENT( eval_FC )
       data%reverse_g = .NOT. PRESENT( eval_GJ )
       IF ( data%control%model == 2 ) THEN
         IF ( data%control%hessian_available ) THEN
           data%reverse_h = .NOT. PRESENT( eval_HL )
         ELSE
           data%control%subproblem_direct = .FALSE.
           IF ( data%control%preconditioner >= 0 )                             &
             data%control%preconditioner = - 1
           data%reverse_h = .FALSE.
         END IF
         data%reverse_hprod = .NOT. PRESENT( eval_HLPROD )
       ELSE
         data%control%preconditioner = - 1
         data%control%hessian_available = .FALSE.
         data%reverse_h = .FALSE.
         data%reverse_hprod = .FALSE.
         IF ( data%control%model /= 1 .AND. data%control%model /= 3 )          &
           data%control%model = 3
         IF ( data%control%model == 1 .OR. data%control%model == 3 )           &
           data%control%GLTR_control%steihaug_toint = .TRUE.
       END IF
     ELSE
       data%reverse_f = .NOT. PRESENT( eval_F )
       data%reverse_g = .NOT. PRESENT( eval_G )
       IF ( data%control%model == 2 ) THEN
         IF ( data%control%hessian_available ) THEN
           data%reverse_h = .NOT. PRESENT( eval_H )
         ELSE
           data%control%subproblem_direct = .FALSE.
           IF ( data%control%preconditioner >= 0 )                             &
             data%control%preconditioner = - 1
           data%reverse_h = .FALSE.
         END IF
         data%reverse_hprod = .NOT. PRESENT( eval_HPROD )
       ELSE
         data%control%preconditioner = - 1
         data%control%hessian_available = .FALSE.
         data%reverse_h = .FALSE.
         data%reverse_hprod = .FALSE.
         IF ( data%control%model /= 1 .AND. data%control%model /= 3 )          &
           data%control%model = 3
         IF ( data%control%model == 1 .OR. data%control%model == 3 )           &
           data%control%GLTR_control%steihaug_toint = .TRUE.
       END IF
     END IF
     data%reverse_prec = .NOT. PRESENT( eval_PREC )

     data%nprec = data%control%preconditioner
     data%control%GLTR_control%unitm = data%nprec == - 1
     data%control%PSLS_control%preconditioner = data%nprec
     data%control%PSLS_control%semi_bandwidth = data%control%semi_bandwidth
     data%control%PSLS_control%icfs_vectors = data%control%icfs_vectors
     data%control%PSLS_control%new_structure = .TRUE.

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
     data%print_level_gltr = data%control%GLTR_control%print_level
     data%print_level_trs = data%control%TRS_control%print_level
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

!  allocate further arrays

     IF ( .NOT. data%monotone ) THEN
       array_name = 'TRAL: data%PHI_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%PHI_hist,  &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'TRAL: data%DMO_hist'
       CALL SPACE_resize_array( data%non_monotone_history + 1, data%DMO_hist,  &
              inform%status,                                                   &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!    IF ( .NOT. data%control%subproblem_direct .OR.                            &
!         data%control%retrospective_trust_region ) THEN
       array_name = 'TRAL: data%U'
       CALL SPACE_resize_array( nlp%n, data%U, inform%status,                  &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
!    END IF
    
       array_name = 'TRAL: data%DX'
       CALL SPACE_resize_array( nlp%n, data%lbfgs_mem, data%DX, inform%status, &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

     IF ( data%nprec == - 2 ) THEN

!  a limited-memory BFGS matrix is to be used

       array_name = 'TRAL: data%DX'
       CALL SPACE_resize_array( nlp%n, data%lbfgs_mem, data%DX, inform%status, &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'TRAL: data%DG'
       CALL SPACE_resize_array( nlp%n, data%lbfgs_mem, data%DG, inform%status, &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'TRAL: data%RHO'
       CALL SPACE_resize_array( data%lbfgs_mem, data%RHO, inform%status,       &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'TRAL: data%ALPHA'
       CALL SPACE_resize_array( data%lbfgs_mem, data%ALPHA, inform%status,     &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

!  parameters needed for the limited-memory BFGS preconditioner

       data%ibound = - 1 ; data%ipoint = 0 ; data%nskip_lbfgs  = 0 
     END IF
     data%nskip_prec = nskip_prec_max

!  record the number of nonzeos in the upper triangle of the Hessian

     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       data%h_ne = nlp%H%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%h_ne = nlp%H%PTR( nlp%H%n + 1 ) - 1
     CASE ( 'DENSE' ) 
       data%h_ne = nlp%H%n * ( nlp%H%n + 1 ) / 2
     CASE DEFAULT
       data%h_ne = nlp%H%n
     END SELECT

!  set up storage to hold the complete columns (both triangles) of the Hessian 
!  in increasing order

     data%H_by_cols%ne = data%h_ne
     IF ( SMT_get( nlp%H%type ) /= 'DIAGONAL' ) THEN
       array_name = 'TRAL: data%MAP'
       CALL SPACE_resize_array( data%h_ne, 2, data%MAP,                        &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

     array_name = 'TRAL: data%H_by_cols%PTR'
     CALL SPACE_resize_array( nlp%H%n + 1, data%H_by_cols%PTR,                 &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       CALL SLS_coord_to_extended_csr( nlp%H%n, nlp%H%ne, nlp%H%ROW,           &
                                       nlp%H%COL, data%MAP,                    &
                                       data%H_by_cols%PTR, duplicates,         &
                                       out_of_range, upper, missing_diagonals )
     CASE ( 'SPARSE_BY_ROWS' )
       CALL SMT_put( data%H_by_cols%type, 'SPARSE_BY_ROWS',                    &
                     inform%alloc_status )

       array_name = 'TRAL: data%H_by_cols%ROW'
       CALL SPACE_resize_array( data%H_by_cols%ne, data%H_by_cols%ROW,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       DO i = 1, nlp%H%n
         data%H_by_cols%ROW( nlp%H%PTR( i ) : nlp%H%PTR( i + 1 ) - 1 ) = i
       END DO
       CALL SLS_coord_to_extended_csr( nlp%H%n, data%H_by_cols%ne,             &
                                       data%H_by_cols%ROW,                     &
                                       nlp%H%COL, data%MAP,                    &
                                       data%H_by_cols%PTR, duplicates,         &
                                       out_of_range, upper, missing_diagonals )
     CASE ( 'DENSE' ) 
       CALL SMT_put( data%H_by_cols%type, 'DENSE', inform%alloc_status )   

       array_name = 'TRAL: data%H_by_cols%ROW'
       CALL SPACE_resize_array( data%H_by_cols%ne, data%H_by_cols%ROW,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'TRAL: data%H_by_cols%COL'
       CALL SPACE_resize_array( data%H_by_cols%ne, data%H_by_cols%COL,         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       l = 0
       DO i = 1, nlp%H%n
         DO j = 1, i
           l = l + 1
           data%H_by_cols%ROW( l ) = i ; data%H_by_cols%COL( l ) = j
         END DO
       END DO
       CALL SLS_coord_to_extended_csr( nlp%H%n, data%H_by_cols%ne,             &
                                       data%H_by_cols%ROW,                     &
                                       data%H_by_cols%COL, data%MAP,           &
                                       data%H_by_cols%PTR, duplicates,         &
                                       out_of_range, upper, missing_diagonals )
     CASE DEFAULT
       DO i = 1, nlp%H%n + 1
         data%H_by_cols%PTR( i ) = i
       END DO
     END SELECT

!  extend the column data

     data%H_by_cols%n = nlp%H%n
     data%H_by_cols%ne = data%H_by_cols%PTR( nlp%H%n + 1 ) - 1

     array_name = 'TRAL: data%H_by_cols%ROW'
     CALL SPACE_resize_array( data%H_by_cols%ne, data%H_by_cols%ROW,           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'TRAL: data%H_by_cols%VAL'
     CALL SPACE_resize_array( data%H_by_cols%ne, data%H_by_cols%VAL,           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     SELECT CASE ( SMT_get( nlp%H%type ) )
     CASE ( 'COORDINATE' )
       DO l = 1, nlp%H%ne
         k = data%MAP( l, 1 )
         IF ( k > 0 ) data%H_by_cols%ROW( k ) = nlp%H%COL( l )
         k = data%MAP( l, 2 )
         IF ( k > 0 ) data%H_by_cols%ROW( k ) = nlp%H%ROW( l )
       END DO
     CASE ( 'SPARSE_BY_ROWS' )
       DO i = 1, nlp%H%n
         DO l = nlp%H%PTR( i ), nlp%H%PTR( i + 1 ) - 1
           k = data%MAP( l, 1 )
           IF ( k > 0 ) data%H_by_cols%ROW( k ) = nlp%H%COL( l )
           k = data%MAP( l, 2 )
           IF ( k > 0 ) data%H_by_cols%ROW( k ) = i
         END DO
       END DO
     CASE ( 'DENSE' )
       l = 0
       DO i = 1, nlp%H%n
         DO j = 1, i
           l = l + 1
           k = data%MAP( l, 1 )
           IF ( k > 0 ) data%H_by_cols%ROW( k ) = j
           k = data%MAP( l, 2 )
           IF ( k > 0 ) data%H_by_cols%ROW( k ) = i
         END DO
       END DO
     CASE DEFAULT
       DO i = 1, nlp%H%n
         data%H_by_cols%ROW( i ) = i
       END DO
     END SELECT

!  if the preconditioner is diagonal, record the positions of the diagonal 
!  entris in POSITION_diagonal; if there is no diagonal, flag it as 0

     data%diagonal_preconditioner = data%control%preconditioner == 1
     IF ( data%diagonal_preconditioner ) THEN
       array_name = 'TRAL: data%POSITION_diagonal'
       CALL SPACE_resize_array( nlp%n, data%POSITION_diagonal, inform%status,  &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'TRAL: data%H_diagonal'
       CALL SPACE_resize_array( nlp%n, data%H_diagonal, inform%status,         &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       DO j = 1, nlp%n
         data%POSITION_diagonal( j ) = 0
         DO k = data%H_by_cols%PTR( j ), data%H_by_cols%PTR( j + 1 ) - 1
           IF ( j == data%H_by_cols%ROW( k ) ) THEN
             data%POSITION_diagonal( j ) = k
             EXIT
           END IF
         END DO
       END DO
     END IF

!  ensure that the initial point is feasible

     nlp%X( : nlp%n ) = TRAL_projection( nlp%n, nlp%X, nlp%X_l, nlp%X_u )

!  find the initial active set for x

     DO i = 1, nlp%n
      IF ( nlp%X( i ) <=                                                       &
            nlp%X_l( i ) * ( one + SIGN( epsmch, nlp%X_l( i ) ) ) )            &
        data%V_status( i ) = 1
      IF ( nlp%X( i ) >=                                                       &
             nlp%X_u( i ) * ( one - SIGN( epsmch, nlp%X_u( i ) ) ) )           &
        data%V_status( i ) = 2
      IF ( nlp%X_u( i ) * ( one - SIGN( epsmch, nlp%X_u( i ) ) ) <=            &
           nlp%X_l( i ) * ( one + SIGN( epsmch, nlp%X_l( i ) ) ) )             &
        data%V_status( i ) = 3
      IF ( data%V_status( i ) == 3 )                                           &
         nlp%X( i ) = half * ( nlp%X_l( i ) + nlp%X_u( i ) )
     END DO

!  evaluate the objective function at the initial point

     IF ( data%reverse_f ) THEN
       data%branch = 2 ; inform%status = 2 ; RETURN
     ELSE
       IF ( data%constrained ) THEN
         CALL eval_FC( data%eval_status, nlp%X( : nlp%n ), userdata,           &
                       inform%obj, nlp%C( : nlp%m ) )
       ELSE
         CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, inform%obj )
       END IF
     END IF

!  return from reverse communication to obtain the objective value

  20 CONTINUE
     IF ( data%reverse_f ) inform%obj = nlp%f
     inform%f_eval = inform%f_eval + 1

!  set up the variables for the merit function phi(x,c)

     data%V_phi(  : nlp%n ) = nlp%X( : nlp%n )

!  ensure that the constraint variables c are feasible

     IF ( data%constrained ) THEN
       data%V_phi( nlp%n + 1 : data%n_phi ) =                                  &
         TRAL_projection( nlp%m, nlp%C, nlp%C_l, nlp%C_u )

!  find the initial active set for c

       DO i = 1, nlp%m
        IF ( data%V( nlp%n + i ) <=                                            &
              nlp%C_l( i ) * ( one + SIGN( epsmch, nlp%C_l( i ) ) ) )          &
          data%V_status( nlp%n + i ) = 1
        IF ( data%V( nlp%n + i ) >=                                            &
               nlp%C_u( i ) * ( one - SIGN( epsmch, nlp%C_u( i ) ) ) )         &
          data%V_status( nlp%n + i ) = 2
        IF ( nlp%C_u( i ) * ( one - SIGN( epsmch, nlp%C_u( i ) ) ) <=          &
             nlp%C_l( i ) * ( one + SIGN( epsmch, nlp%C_l( i ) ) ) )           &
          data%V_status( nlp%n + i ) = 3
        IF ( data%V_status(nlp%n +  i ) == 3 )                                 &
           nlp%C( i ) = half * ( nlp%C_l( i ) + nlp%C_u( i ) )
          data%V_phi( nlp%n + i ) = nlp%C( i )
       END DO

!  when the problem is constrained, compute the merit function
!    phi(x,c) = f(x) + y^T(c(x)-c) + 0.5 rho ||c(x)-c)||_2^2

       data%phi = nlp%f ; c_norm = zero
       DO i = 1, nlp%m
         c_diff = nlp%C( i ) - data%V_phi( nlp%n + i )
         data%phi = data%phi + nlp%Y( i ) * c_diff
         c_norm = c_norm + c_diff * c_diff

!  record y + rho * ( c(x) - c )

         data%Y( i ) =  nlp%Y( i ) + inform%row * c_diff
       END DO
       data%phi = data%phi + half * inform%rho * c_norm

!  in the unconstrained case, record phi(x) = f(x)

     ELSE
       data%phi = nlp%f
     END IF

!  test to see if the objective appears to be unbounded from below

     IF ( data%phi < control%obj_unbounded ) THEN 
       inform%status = GALAHAD_error_unbounded ; GO TO 990
     END IF 

     data%phi_ref = data%phi
     IF ( .NOT. data%monotone ) THEN
        data%PHI_hist = data%phi_ref ; data%DMO_hist = zero ; data%max_hist = 1
     END IF

!  evaluate the gradient of the objective function

     IF ( data%reverse_g ) THEN
       data%branch = 3 ; inform%status = 3 ; RETURN
     ELSE
       IF ( data%constrained ) THEN
         CALL eval_GJ( data%eval_status, nlp%X( : nlp%n ),                     &
                       userdata, nlp%G( : nlp%n ), nlp%J%val( : nlp%J_ne ) )
       ELSE
         CALL eval_G( data%eval_status, nlp%X( : nlp%n ),                      &
                      userdata, nlp%G( : nlp%n ) )
       END IF
     END IF

!  return from reverse communication to obtain the gradient

  30 CONTINUE
     inform%g_eval = inform%g_eval + 1

!  in the unconstrained case, record the gradient of phi is g(x)

     data%G_phi( : nlp%n ) = nlp%G( : nlp%n )

!  when the problem is constrained, compute the gradient of phi,
!    grad phi = ( g(x) +J^T(x) y(x,rho) ), 
!               (       - y(x,rho)      )
!  where y(x,rho) = y + rho (c(x)-c)

     IF ( data%constrained ) THEN
       CALL mop_Ax( one, nlp%J, data%Y( : nlp%m ), one, data%G_phi( : nlp%n ), &
                    transpose = .TRUE., m_matrix = nlp%m, n_matrix = nlp%n )
       data%G_phi( nlp%n + 1 : data%n_phi ) = - data%Y( : nlp%m )
     END IF

!  if required, print details of the current point

!    IF ( data%printd ) THEN
!      WRITE ( data%out, 2210 ) prefix
!      DO i = 1, nlp%n
!        WRITE( data%out, 2230 ) prefix, i,                                    &
!          nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%G( i )
!      END DO
!    END IF

!  compute the norm of the projected gradient

     data%WK( : nlp%n ) =                                                      &
       TRAL_projection( nlp%n, data%V_phi( : nlp%n ) - nlp%G_phi( : nlp%n ),   &
                        nlp%X_l, nlp%X_u )
     data%WK( nlp%n + 1 : data%n_all ) =                                       &
       TRAL_projection( nlp%m, data%V_phi( nlp%n + 1 : data%n_all )            &
                        - nlp%G_phi( nlp%n + 1 : data%n_all ), nlp%Cl, nlp%C_u )
     inform%norm_pg = TWO_NORM( data%V_phi( : data%n_all ) -                   &
                                data%WK( : data%n_all )

!  compute the stopping tolerance

     data%stop_pg = MAX( control%stop_pg_absolute,                             &
                         control%stop_pg_relative * inform%norm_pg )

!    data%new_h = data%control%hessian_available
     data%new_h = .TRUE.

     IF ( data%printi ) WRITE( data%out, "( A, '  Problem: ', A,               &
    &   ' (n = ', I0, '): TRAL stopping tolerance =', ES11.4, / )" )           &
       prefix, TRIM( nlp%pname ), nlp%n, data%stop_pg

!  =======================
!  Start of main iteration
!  =======================

 100 CONTINUE
       n_active = TRAL_active( nlp%n, nlp%X, nlp%X_l, nlp%X_u )

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 )   &
           THEN
         data%printi = data%set_printi ; data%printt = data%set_printt 
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%control%GLTR_control%print_level = data%print_level_gltr
         data%control%TRS_control%print_level = data%print_level_trs
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%GLTR_control%print_level = 0
         data%control%TRS_control%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         ( data%control%GLTR_control%print_level > 0 .AND. .NOT.               &
           data%control%subproblem_direct ) .OR.                               &
         ( data%control%TRS_control%print_level > 0 .AND.                      &
           data%control%subproblem_direct )

!  print one-line summary

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           IF ( data%no_bounds ) THEN
             IF ( data%control%subproblem_direct ) THEN
               WRITE( data%out, 2010 ) prefix
             ELSE
               WRITE( data%out, 2020 ) prefix
             END IF
           ELSE
             IF ( data%control%subproblem_direct ) THEN
               WRITE( data%out, 2030 ) prefix
             ELSE
               WRITE( data%out, 2040 ) prefix
             END IF
           END IF
         END IF
         data%print_1st_header = .FALSE.
         char_iter = STRING_integer_6( inform%iter )
         IF ( data%no_bounds ) THEN
           IF ( inform%iter > 0 ) THEN
             IF ( data%control%subproblem_direct ) THEN
               char_facts = STRING_integer_6( inform%TRS_inform%factorizations )
               WRITE( data%out, 2050 ) prefix, char_iter, data%hard,           &
                  data%negcur, data%bndry, inform%obj, inform%norm_pg,         &
                  data%ratio, inform%radius, inform%TRS_inform%multiplier,     &
                  char_facts, data%clock_now
             ELSE
               char_sit = STRING_integer_6( inform%GLTR_inform%iter )
               char_sit2 = STRING_integer_6( inform%GLTR_inform%iter_pass2 )
               WRITE( data%out, 2060 ) prefix, char_iter, data%negcur,         &
                  data%bndry, data%perturb, inform%obj, inform%norm_pg,        &
                  data%ratio, inform%radius, char_sit, char_sit2,              &
                  data%clock_now
             END IF
           ELSE
             WRITE( data%out, 2070 ) prefix,                                   &
                  char_iter, inform%obj, inform%norm_pg, inform%radius
           END IF
         ELSE
           char_active = STRING_integer_6( n_active )
           IF ( inform%iter > 0 ) THEN
             IF ( data%control%subproblem_direct ) THEN
               char_facts = STRING_integer_6( inform%TRS_inform%factorizations )
               WRITE( data%out, 2080 ) prefix, char_iter, data%hard,           &
                  data%negcur, data%bndry, inform%obj, inform%norm_pg,         &
                  data%ratio, inform%radius, char_active, char_facts,          &
                  data%clock_now
             ELSE
               char_sit = STRING_integer_6( data%itercg )
               WRITE( data%out, 2080 ) prefix, char_iter, data%negcur,         &
                  data%bndry, data%perturb, inform%obj, inform%norm_pg,        &
                  data%ratio, inform%radius, char_active, char_sit,            &
                  data%clock_now
             END IF
           ELSE
             WRITE( data%out, 2090 ) prefix,                                   &
                  char_iter, inform%obj, inform%norm_pg, inform%radius,        &
                  char_active
           END IF
         END IF
       END IF

!  =======================
!  1. Test for convergence
!  =======================

!  stop if the gradient is small enough

       IF ( inform%norm_pg <= data%stop_pg ) THEN
         inform%status = GALAHAD_ok ; GO TO 900
       END IF

!  stop if the gradient is swamped by the Hessian

       IF ( data%control%hessian_available .AND. inform%iter > 0 ) THEN
         IF ( inform%norm_pg <= MIN( one,                                      &
                MAXVAL( ABS( nlp%H%val( : nlp%H%ne ) ) ) * epsmch ) ) THEN
           inform%status = GALAHAD_error_ill_conditioned ; GO TO 900
         END IF
       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = - 40
           RETURN
         END IF
       END IF

!  check to see if the iteration limit has been exceeded

       inform%iter = inform%iter + 1
       IF ( inform%iter > data%control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  debug printing for X and G

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2000 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2200 ) prefix, inform%f_eval, prefix, inform%g_eval,&
           prefix, inform%h_eval, prefix, inform%iter, prefix, inform%cg_iter, &
           prefix, inform%obj, prefix, inform%norm_pg
         WRITE ( data%out, 2210 ) prefix
!        l = nlp%n
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
                 WRITE( data%out, 2220 ) prefix, nlp%vnames( i ),              &
                   nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%G( i )
              END DO
            ELSE
              DO i = ir, ic
                 WRITE( data%out, 2230 ) prefix, i,                            &
                   nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%G( i )
              END DO
            END IF
         END DO
       END IF

!  recompute the Hessian if it has changed

       data%perturb = ' '
       IF ( data%new_h ) data%nskip_prec = data%nskip_prec + 1
       IF ( data%new_h .AND. data%control%hessian_available ) THEN
         data%got_h = .FALSE.

!  form the Hessian or a preconditioner based on the Hessian

         IF ( data%nskip_prec > nskip_prec_max ) THEN
           IF ( data%reverse_h ) THEN
             data%branch = 4 ; inform%status = 4 ; RETURN
           ELSE
             IF ( data%constrained ) THEN
               CALL eval_HL( data%eval_status, nlp%X( : nlp%n ),               &
                             nlp%Y( : nlp%m ), userdata,                       &
                             nlp%H%val( : nlp%H%ne ) )
             ELSE
               CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                &
                            userdata, nlp%H%val( : nlp%H%ne ) )
             END IF
           END IF
         END IF
       END IF

!  return from reverse communication to obtain the Hessian

 110   CONTINUE
       IF ( data%new_h .AND. data%control%hessian_available ) THEN
         IF ( data%nskip_prec > nskip_prec_max ) THEN
           inform%h_eval = inform%h_eval + 1  ; data%got_h = .TRUE.

!  Hess phi = ( H(x,y(x,rho)) + rho J^T(x) J(x)  - rho J^T(x) )
!             (    - rho J(x)                      rho I      )
!  where y(x,rho) = y + rho (c(x)-c) and H(x,y) = Hess Lagrangian

!  debug printing for H 

           IF ( data%printd ) THEN
             WRITE( data%out, "( A, ' Hessian ' )" ) prefix
             DO l = 1, nlp%H%ne
               WRITE( data%out, "( A, 2I7, ES24.16 )" ) prefix,                &
                 nlp%H%row( l ), nlp%H%col( l ), nlp%H%val( l )
             END DO
           END IF
         END IF

!  extend the Hessian data

         IF ( SMT_get( nlp%H%type ) /= 'DIAGONAL' ) THEN
           DO j = 1, 2
             DO l = 1, data%h_ne
               k = data%MAP( l, j )
               IF ( k > 0 ) THEN
                 data%H_by_cols%VAL( k ) = nlp%H%VAL( l )
               ELSE IF ( k < 0 ) THEN
                 data%H_by_cols%VAL( - k ) =                                   &
                   data%H_by_cols%VAL( - k ) + nlp%H%VAL( l )
               END IF
             END DO
           END DO

!  if needed, recod the diagonal

           IF ( data%diagonal_preconditioner ) THEN
             DO j = 1, nlp%n
               k = data%POSITION_diagonal( j )
               IF ( k /= 0 ) THEN
                 data%H_diagonal( j ) = data%H_by_cols%VAL( k )
               ELSE
                 data%H_diagonal( j ) = zero
               END IF
             END DO
           END IF
         ELSE
           data%H_by_cols%VAL( : nlp%n ) = nlp%H%VAL( : nlp%n )
           IF ( data%diagonal_preconditioner )                                 &
             data%H_diagonal( : nlp%n ) = data%H_by_cols%VAL( : nlp%n )
         END IF
       END IF

!  if the Hessian has changed, recompute the preconditioner

       IF ( data%no_bounds .AND. .NOT. data%control%subproblem_direct ) THEN
         IF ( data%new_h ) THEN
           IF ( data%nskip_prec > nskip_prec_max ) THEN
             IF ( data%nprec > 0 .AND. data%control%hessian_available ) THEN

!  form and factorize the preconditioner

               IF ( data%printt ) WRITE( data%out,                             &
                     "( A, ' Computing preconditioner' )" ) prefix
               CALL PSLS_form_and_factorize( nlp%H, data%PSLS_data,            &
                 data%control%PSLS_control, inform%PSLS_inform )

!  check for error returns

               IF ( inform%PSLS_inform%status /= 0 ) THEN
                 inform%status = inform%PSLS_inform%status  ; GO TO 900
               END IF
               IF ( inform%PSLS_inform%perturbed ) data%perturb = 'p'
               data%control%PSLS_control%new_structure = .FALSE.

!  if a limited memory BFGS preconditioner is to be used, 
!  initialize the parameters

             ELSE IF ( data%nprec == - 2 ) THEN
               data%ibound = MIN( data%ibound + 1, data%lbfgs_mem )
               IF ( inform%iter > data%nskip_lbfgs + 1 ) THEN
                 data%dxtdg = DOT_PRODUCT( data%DG( : , data%icp ),            &
                                           data%DX( : , data%icp ) )
                 data%dgtdg = DOT_PRODUCT( data%DG( : , data%icp ),            &
                                           data%DG( : , data%icp ) )
               END IF
             END IF
             data%nskip_prec = 0
           END IF
         END IF
       END IF

!  ========================================================
!  2. Update the trust-region radius and other book-keeping
!  ========================================================

       IF ( inform%iter > 1 ) THEN
         data%old_radius = inform%radius

!  if the iteration has increased the objective, decrease the radius so that 
!  had the objective been quadratic the next iteration would be very successful

         IF ( data%ratio < zero ) THEN
           inform%radius =                                                     &
             MIN( data%control%radius_reduce * data%s_norm, inform%radius *    &
                  MAX( data%control%radius_reduce_max,                         &
                       data%ometat * data%stg / ( data%df +                    &
                         data%ometat * data%stg + data%etat * data%model ) ) )

!  if the iteration was very unsuccesful, decrease the radius to chop off the
!  current step 

         ELSE IF ( data%ratio < data%control%eta_successful ) THEN
           inform%radius = data%control%radius_reduce * data%s_norm

!  compute the new norm of the step 

         ELSE
           IF ( data%control%preconditioner /= - 1 ) THEN
             IF ( data%control%renormalize_radius ) THEN
               IF ( data%control%subproblem_direct ) THEN
                 data%s_new_norm = data%s_norm
               ELSE
                 data%s_new_norm = PSLS_norm( nlp%H, data%S, data%PSLS_data,   &
                     data%control%PSLS_control, inform%PSLS_inform )
                 IF ( inform%PSLS_inform%status == GALAHAD_norm_unknown ) THEN
                   data%s_new_norm = data%s_norm
                 ELSE IF ( inform%PSLS_inform%status /= 0 ) THEN
                   GO TO 980
                 END IF
               END IF
               IF ( data%printt )                                              &
                 WRITE( data%out, "( A, ' ratio new, old norms = ', ES12.4 )" )&
                   prefix, data%s_new_norm / data%s_norm
             ELSE
               data%s_new_norm = data%s_norm
             END IF

!  if the norm has changed, adjust the radius accordingly

             inform%radius = inform%radius * ( data%s_new_norm / data%s_norm )
             data%s_norm = data%s_new_norm
           END IF

!  a traditional radius update strategy will be used
!  -------------------------------------------------

           IF ( .NOT. data%control%retrospective_trust_region ) THEN

!  if the iteration was very (but not too) successful, increase the radius

             IF ( data%ratio >= data%control%eta_very_successful .AND.         &
                  data%ratio <= data%control%eta_too_successful ) THEN
               IF ( ABS( data%ratio - one ) <= rho_quad .AND.                  &
                    data%rho_g <= rho_quad ) THEN
                 inform%radius = data%control%maximum_radius
               ELSE    
                 IF ( data%control%radius_increase * data%s_norm >             &
                      inform%radius ) inform%radius =                          &
                        MIN( data%control%maximum_radius,                      &
                             data%control%radius_increase * data%s_norm )
               END IF
             END IF

!  a retrospective radius update strategy will be used
!  ---------------------------------------------------

           ELSE

!  compute the new Hessian-step product

             data%U( : nlp%n ) = zero
             SELECT CASE( data%control%model )

!  linear model

             CASE ( 1 )

!  quadratic model with true Hessian

             CASE ( 2 )

!  if the Hessian has been calculated, form the product directly

               IF ( data%control%hessian_available ) THEN
                 DO l = 1, nlp%H%ne
                   i = nlp%H%row( l ) ; j = nlp%H%col( l )
                   val = nlp%H%val( l )
                   data%U( i ) = data%U( i ) + val * data%S( j )
                   IF ( i /= j ) data%U( j ) = data%U( j ) + val * data%S( i )
                 END DO
                 data%V( : nlp%n ) = data%U( : nlp%n )

!  if the Hessian is unavailable, obtain a matrix-free product 
!   Hess phi (p) = ( H(x,y(x,rho) p + J^T(x) r )
!            (q)   (          - r              )
!  where y(x,rho) = y + rho (c(x)-c), H(x,y) = Hess Lagrangian 
!  and r = rho [ J(x) p - q )

             ELSE
               data%V( : nlp%n ) = data%S( : nlp%n )
               IF ( data%reverse_hprod ) THEN
                 data%branch = 5 ; inform%status = 5 ; RETURN
               ELSE
                 IF ( data%constrained ) THEN
                   CALL eval_HLPROD( data%eval_status, nlp%X( : nlp%n ),       &
                                     nlp%Y( : nlp%m ), userdata,               &
                                     data%U( : nlp%n ), data%S( : nlp%n ) )
                 ELSE
                   CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),        &
                                    userdata, data%U( : nlp%n ),               &
                                    data%S( : nlp%n ) )
                 END IF
               END IF
             END IF

!  quadratic model with identity Hessian

             CASE ( 3 )
!              data%U( : nlp%n ) = data%U( : nlp%n ) + data%S( : nlp%n )
             END SELECT
           END IF
         END IF
       END IF

!  return from reverse communication to obtain the Hessian-vector product

   210 CONTINUE
       IF ( inform%iter > 1 ) THEN
         IF ( data%ratio >= data%control%eta_successful ) THEN
           IF ( data%control%retrospective_trust_region ) THEN

!  compute the new model change

             data%stg = DOT_PRODUCT( data%S( : nlp%n ), nlp%G( : nlp%n ) )
             data%model = - data%stg +                                         &
               half * DOT_PRODUCT( data%S( : nlp%n ), data%U( : nlp%n ) )

             rounding =                                                        &
               MAX( one, ABS( inform%obj ) ) * REAL( nlp%n, KIND = wp ) * epsmch
             ared = data%df + rounding
             prered = - data%model - rounding
             IF ( ABS( ared ) < teneps .AND. ABS( inform%obj ) > teneps )      &
               ared = prered
             data%ratio = - ared / prered

!  if the iteration has increased the objective, decrease the radius so that 
!  had the objective been quadratic the next iteration would be very successful

             IF ( data%ratio < zero ) THEN
               data%poor_model = .FALSE.
               inform%radius =                                                 &
                 MIN( data%control%radius_reduce * data%s_norm, inform%radius  &
                      * MAX( data%control%radius_reduce_max,                   &
                           - data%ometat * data%stg / ( - data%df -            &
                           data%ometat * data%stg + data%etat * data%model ) ) )

!  if the iteration was very unsuccesful, decrease the radius to chop off the
!  current step 

             ELSE IF ( data%ratio < data%control%eta_successful ) THEN
               data%poor_model = .FALSE.
               inform%radius = data%control%radius_reduce * data%s_norm
             ELSE IF ( data%ratio >= data%control%eta_very_successful .AND.    &
                       data%ratio <= data%control%eta_too_successful ) THEN
               IF ( ABS( data%ratio - one ) <= rho_quad .AND.                  &
                    data%rho_g <= rho_quad ) THEN
                 inform%radius = data%control%maximum_radius
               ELSE    
                 IF ( data%control%radius_increase * data%s_norm >             &
                      inform%radius ) inform%radius =                          &
                        MIN( data%control%maximum_radius,                      &
                             data%control%radius_increase * data%s_norm )
               END IF
             END IF
           END IF
         END IF

!  update the preconditioner

         IF ( data%new_h ) THEN
           IF ( data%nprec == - 2 ) THEN
             data%dxtdg = DOT_PRODUCT( data%V_phi - data%V_current,            &
                                       data%G_phi - data%G_current )
             IF ( data%dxtdg > zero ) THEN
               data%icp = data%ipoint + 1
               data%DX( : , data%icp ) = data%V_phi - data%V_current
               data%DG( : , data%icp ) = data%G_phi - data%G_current
               data%ipoint = data%icp
               IF ( data%ipoint == data%lbfgs_mem ) data%ipoint = 0
             ELSE
               data%nskip_lbfgs = data%nskip_lbfgs + 1
               IF ( data%printt ) WRITE( data%out,                             &
                   "( /, A, ' Preconditioner update skipped ' )" ) prefix
             END IF
           END IF
         END IF
       END IF

!  ======================
!  problem without bounds
!  ======================

       IF ( data%no_bounds ) THEN
         data%control%GLTR_control%stop_relative                               &
           = MIN( data%control%GLTR_control%stop_relative,                     &
                  inform%norm_pg ** 0.1 )

         data%model = zero ; data%S( : nlp%n ) = zero
         data%G_current( : nlp%n ) = nlp%G( : nlp%n )
         IF ( data%new_h ) THEN
            inform%GLTR_inform%status = 1
         ELSE
           IF ( .NOT. data%control%GLTR_control%steihaug_toint )               &
            inform%GLTR_inform%status = 4
         END IF

!  Start of the generalized Lanczos iteration
!  ..........................................

  460    CONTINUE

!  perform a generalized Lanczos iteration

           CALL GLTR_solve( nlp%n, inform%radius, data%model,                  &
                          data%S( : nlp%n ),                                   &
                          data%G_current( : nlp%n ), data%V( : nlp%n ),        &
                          data%GLTR_data, data%control%GLTR_control,           &
                          inform%GLTR_inform )

           SELECT CASE( inform%GLTR_inform%status )

!  form the preconditioned gradient

           CASE ( 2, 6 )

!  use the factors obtained from PSLS

             IF ( data%nprec > 0 ) THEN
               CALL PSLS_solve( data%V, data%PSLS_data,                        &
                                data%control%PSLS_control, inform%PSLS_inform )

!  compute the precoditioned gradient BFGS * g using Nocedal's LBFGS formula

             ELSE IF ( data%nprec == - 2 ) THEN
               IF ( inform%iter > data%nskip_lbfgs + 1 ) THEN
                 data%icp = data%ipoint
                 IF ( data%ipoint == 0 ) data%icp = data%lbfgs_mem
                 data%RHO( data%icp ) = one / data%dxtdg
                 data%icp = data%ipoint + 1
                 DO i = 1, data%ibound
                   data%icp = data%icp - 1
                   IF ( data%icp == 0 ) data%icp = data%lbfgs_mem
                   data%ALPHA( data%icp ) = data%RHO( data%icp ) *             &
                                  DOT_PRODUCT( data%DX( : , data%icp ), data%V )
                   data%V = data%V -                                           &
                              data%ALPHA( data%icp ) * data%DG( : , data%icp )
                 END DO
                 data%V( : nlp%n ) = data%V( : nlp%n ) *                       &
                                      ( data%dxtdg / data%dgtdg )
                 DO i = 1, data%ibound
                   beta = data%ALPHA( data%icp ) - data%RHO( data%icp ) *      &
                            DOT_PRODUCT( data%DG( : , data%icp ), data%V )
                   data%V( : nlp%n ) = data%V( : nlp%n ) +                     &
                                         beta * data%DX( : , data%icp )
                   data%icp = data%icp + 1
                   IF ( data%icp == data%lbfgs_mem + 1 ) data%icp = 1
                 END DO
               END IF
             ELSE IF ( data%nprec == - 3 ) THEN
               IF ( data%reverse_prec ) THEN
                 data%branch = 5 ; inform%status = 6 ; RETURN
               ELSE
                 IF ( data%constrained ) THEN
                   CALL eval_HLPROD( data%eval_status, nlp%X( : nlp%n ),       &
                                     nlp%Y( : nlp%m ), userdata,               &
                                     data%U( : nlp%n ), data%V( : nlp%n ) )
                 ELSE
                   CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),        &
                                    userdata, data%U( : nlp%n ),               &
                                    data%V( : nlp%n ) )
                 END IF
                 data%V( : nlp%n ) = data%U( : nlp%n )
               END IF
             END IF

!  form the Hessian-vector product

           CASE ( 3, 7 )
             data%U( : nlp%n ) = zero
             SELECT CASE( data%control%model )

!  linear model

             CASE ( 1 )

!  quadratic model with true Hessian

             CASE ( 2 )

!  if the Hessian has been calculated, form the product directly

               IF ( data%control%hessian_available ) THEN
                 DO l = 1, nlp%H%ne
                   i = nlp%H%row( l ) ; j = nlp%H%col( l ) ; val = nlp%H%val( l )
                   data%U( i ) = data%U( i ) + val * data%V( j )
                   IF ( i /= j ) data%U( j ) = data%U( j ) + val * data%V( i )
                 END DO
                 data%V( : nlp%n ) = data%U( : nlp%n )

!  if the Hessian is unavailable, obtain a matrix-free product 

               ELSE
                 IF ( data%reverse_hprod ) THEN
                   data%branch = 6 ; inform%status = 5 ; RETURN
                 ELSE
                   IF ( data%constrained ) THEN
                     CALL eval_HLPROD( data%eval_status, nlp%X( : nlp%n ),     &
                                       nlp%Y( : nlp%m ), userdata,             &
                                       data%U( : nlp%n ), data%V( : nlp%n ) )
                   ELSE
                     CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),      &
                                      userdata, data%U( : nlp%n ),             &
                                      data%V( : nlp%n ) )
                   END IF
                 END IF
               END IF

!  quadratic model with identity Hessian

             CASE ( 3 )
!              data%U( : nlp%n ) = data%U( : nlp%n ) + data%V( : nlp%n )
             END SELECT

!  restore the gradient

           CASE ( 5 )
             data%G_current( : nlp%n ) = nlp%G( : nlp%n )

!  successful return

           CASE ( GALAHAD_ok, GALAHAD_warning_on_boundary,                     &
                  GALAHAD_error_max_iterations )
             GO TO 490

!  error returns

           CASE DEFAULT
             IF ( data%printt ) WRITE( data%out, "( /,                         &
             &  A, ' Error return from GLTR, status = ', I0 )" ) prefix,       &
               inform%GLTR_inform%status
             inform%status = inform%GLTR_inform%status
             GO TO 900
           END SELECT

!  return from reverse communication to obtain the Hessian-vector product
!  or preconditioned vector

  480      CONTINUE
           IF ( .NOT. data%control%hessian_available ) THEN
             IF ( inform%GLTR_inform%status == 3 .OR.                          &
                  inform%GLTR_inform%status == 7 ) THEN
               inform%h_eval = inform%h_eval + 1 ; data%got_h = .TRUE.
               data%V( : nlp%n ) = data%U( : nlp%n )
             END IF
           END IF
           IF ( ( inform%GLTR_inform%status == 2 .OR.                          &
                  inform%GLTR_inform%status == 6 ) .AND.                       &
                  data%nprec == - 3 .AND. data%reverse_prec ) THEN
             data%V( : nlp%n ) = data%U( : nlp%n )
           END IF
         GO TO 460

!  End of the generalized Lanczos iteration
!  ........................................

  490    CONTINUE

!  Record whether there is negative curvature or if the boundary is encountered

         IF ( inform%GLTR_inform%negative_curvature ) THEN
           data%negcur = 'n'
         ELSE
           data%negcur = ' '
         END IF

         data%s_norm = inform%GLTR_inform%mnormx
         IF ( ABS( inform%radius - inform%GLTR_inform%mnormx ) <= 1.0D-8 ) THEN
           data%bndry = 'b'
         ELSE
           data%bndry = ' '
         END IF

!  Record the total number of Lanczos iterations

         inform%cg_iter = inform%cg_iter +                                     &
           inform%GLTR_inform%iter + inform%GLTR_inform%iter_pass2
         IF ( data%printt ) WRITE( data%out,                                   &
            "( /, A, ' CG iterations required = ', I8 )" )                     &
              prefix, inform%GLTR_inform%iter

         data%V_trial( : data%n_phi )                                          &
           = data%V_phi( : data%n_phi ) + data%S( : data%n_phi )

!  If necessary, temporarily store the old gradient

         IF ( data%nprec == - 2 )                                              &
           data%G_current( : data%n_phi ) = data%G_phi( : data%n_phi )

!  ===================
!  problems with bound
!  ===================

       ELSE

!  =====================================
!  3. Compute a Generalized Cauchy Point
!  =====================================

!  3a. Direct solution
!  -------------------

!        IF ( data%control%subproblem_direct ) THEN
!          GO TO 500
  !      END IF

!  3b. Iterative solution
!  ----------------------

!  initialize INDEX_used_hp

         data%n_prods = 0
         data%INDEX_used_hp( : nlp%n ) = 0

!  estimate the norm of the preconditioning matrix by computing its smallest
!  and largest (in magnitude) diagonals

         data%diagonal_min = HUGE( one ) ; data%diagonal_max = zero
         IF ( data%diagonal_preconditioner ) THEN
           DO i = 1, nlp%n
             IF ( data%V_status( i ) == 0 ) THEN
              data%diagonal_min = MIN( data%diagonal_min, data%H_diagonal( i ) )
              data%diagonal_max = MAX( data%diagonal_max, data%H_diagonal( i ) )
             END IF
           END DO
         END IF

!  if all the diagonals are small, the norm will be estimated as one

         IF ( data%diagonal_max <= epsmch ) THEN
           data%diagonal_min = one ; data%diagonal_max = one
         END IF

!  initialize values for the generalized Cauchy point calculation

         data%step_max = zero ; data%model_start = zero
         DO i = 1, nlp%n

!  set the bounds on the variables for the model problem. If a two-norm
!  trust region is to be used, the bounds are just the box constraints

           IF ( data%control%two_norm_tr ) THEN
             data%BND( i, 1 ) = nlp%X_l( i )
             data%BND( i, 2 ) = nlp%X_u( i )
           ELSE

!  if an infinity-norm trust region is to be used, the bounds are the
!  intersection of the trust region with the box constraints

             IF ( data%diagonal_preconditioner ) THEN
               distan = inform%radius / SQRT( data%H_diagonal( i ) )
             ELSE
               distan = inform%radius
             END IF
             data%BND( i, 1 ) = MAX( nlp%X_l( i ), nlp%X( i ) - distan )
             data%BND( i, 2 ) = MIN( nlp%X_u( i ), nlp%X( i ) + distan )
             IF ( data%control%more_toraldo > 0 ) THEN
               data%BND_radius( i, 1 ) = nlp%X( i ) - distan
               data%BND_radius( i, 2 ) = nlp%X( i ) + distan
             END IF
           END IF

!  compute the Cauchy direction, data%P, as a scaled steepest-descent
!  direction. Normalize the diagonal scalings if necessary

           data%V_current( i ) = data%V_phi( i )
           data%G_current( i ) = data%G_phi( i )
           data%P( i ) = zero
           IF ( data%reuse_cp ) CYCLE
           IF ( data%diagonal_preconditioner ) THEN
             data%P( i ) = - data%G_phi( i ) / data%H_diagonal( i )
             data%H_diagonal( i ) = data%H_diagonal( i ) / data%diagonal_max
           ELSE
             data%P( i ) = - data%G_phi( i )
           END IF

!  if an approximation to the Cauchy point is to be used, calculate a
!  suitable initial estimate of the line minimum, step_max

           IF ( data%control%exact_gcp ) THEN
             data%step_max = HUGE( one )
           ELSE  
             IF ( data%P( i ) /= zero ) THEN
               IF ( data%P( i ) > zero ) THEN
                 data%step_max = MAX( ( nlp%X_u( i ) - nlp%X( i ) )            &
                                        / data%P( i ), data%step_max )
               ELSE
                 data%step_max = MAX( ( nlp%X_l( i ) - nlp%X( i ) )            &
                                        / data%P( i ), data%step_max )
               END IF
             END IF
           END IF

!  release any artificially fixed variables from their bounds

           IF ( data%V_status( i ) == 4 ) data%V_status( i ) = 0
         END DO
         IF ( data%control%accurate_bqp ) data%DX_bqp = zero

!  the value of factorize controls whether a new factorization of the Hessian 
!  of the model is obtained (.TRUE.) or whether a Schur-complement update to an 
!  existing factorization is required (.FALSE.) when forming the preconditioner

         data%refactorize = .TRUE.

!  if a previously calculated generalized Cauchy point still lies within the
!  trust-region bounds, it will be reused

         IF ( data%reuse_cp ) THEN

!  retrieve the Cauchy point

           data%V_trial( : data%n_phi ) = data%V_cauchy( : data%n_phi )

!  retrieve the set of free variables

           inform%n_free = data%nfree_cp
!          data%INDEX_nz_p( : inform%n_free ) = IFREEC( : inform%n_free )
           data%V_status( data%INDEX_nz_p( : inform%n_free ) ) = 0

!  skip the remainder of the step

           data%reuse_cp = .FALSE.
           IF ( data%printt ) WRITE( data%out,                                 &
             "( /, A, ' Reusing previous generalized Cauchy point ' )" ) prefix
           GO TO 390
         END IF

!  find the norm of the reduced gradient

         data%rg_norm                                                          &
           = TRAL_reduced_gradient_norm( nlp%n, nlp%X, nlp%G, nlp%X_l, nlp%X_u )

!  evaluate the generalized Cauchy point, data%V_trial

         data%jumpto = 1
         data%first_cp = .TRUE.
         data%more_toraldo_its = 0
         data%radius = inform%radius

  310    CONTINUE
         CALL CPU_time( data%time_record )
         CALL CLOCK_time( data%clock_record )

!  the exact generalized Cauchy point is required

         IF ( data%control%exact_gcp ) THEN
           CALL CAUCHY_get_exact_gcp(                                          &
               data%n_phi, data%V_current, data%V_trial, data%G_current,       &
               data%BND, data%V_status, data%model_start, data%step_max,       &
               epsmch, data%control%two_norm_tr, data%dxsqr, data%radius,      &
               data%model, data%P, data%HP, data%INDEX_nz_p, inform%n_free,    &
               data%nnz_p_l, data%nnz_p_u, data%nnz_hp, data%INDEX_nz_hp,      &
               data%out, data%jumpto, data%print_level, one,                   &
               data%WK, data%CAUCHY_save )

!  an approximation to the Cauchy point suffices

         ELSE
           CALL CAUCHY_get_approx_gcp(                                         &
               data%n_phi, data%V_current, data%V_trial, data%G_current,       &
               data%BND,                                                       &
               data%V_status, data%model_start, epsmch, data%step_max, point1, &
               data%control%two_norm_tr, data%radius, data%model, data%P,      &
               data%HP, data%INDEX_nz_p, inform%n_free, data%nnz_p_l,          &
               data%nnz_p_u, data%out, data%jumpto, data%print_level, one,     &
             data%WK, data%WK2, data%CAUCHY_save )
         END IF
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!        data%tca = data%tca + data%time_now - data%time_record

!  a further matrix-vector product is required

         IF ( data%jumpto > 0 ) THEN
           CALL CPU_time( data%time_record )
           CALL CLOCK_time( data%clock_record )
           data%n_prods = data%n_prods + 1

!  calculate the product of the Hessian with the vector P

           data%dense_p = data%jumpto == 2 .OR.                                &
                        ( data%jumpto == 4 .AND. data%control%exact_gcp ) 
           CALL TRAL_hessian_times_vector( nlp%n, data%INDEX_nz_p, data%nnz_p_l,&
                                          data%nnz_p_u, data%INDEX_nz_hp,      &
                                          data%nnz_hp, data%INDEX_used_hp,     &
                                          data%n_prods, data%P, data%HP,       &
                                          data%H_by_cols, data%dense_p )
           IF ( data%printd .AND. data%jumpto == 3 ) WRITE( data%out, "( A,    &
          & ' Nonzeros of Hessian * P are in positions', /, ( '    ', 10I7 ) )"&
               prefix, data%INDEX_nz_hp( : data%nnz_hp )
           CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!          data%tmv = data%tmv + data%time_now - data%time_record

!  if required, print a list of the nonzeros of P

           IF ( data%jumpto == 3 .AND. data%printd )                           &
             WRITE( data%out, 2260 ) prefix, data%n_prods,                     &
               data%INDEX_used_hp( : data%nnz_hp )

!  continue the Cauchy point calculation

           GO TO 310
         END IF

!  check to see if there are any remaining free variables

         IF ( data%nnz_p_l > data%nnz_p_u ) THEN
           IF ( data%printt ) WRITE( data%out,                                 &
             "( /, A, '    No free variables - search direction complete ' )" )&
               prefix
           GO TO 500
         ELSE 
           IF ( data%printt ) WRITE( data%out,                                 &
             "( /, A, ' There are now ', I0, ' free variables ' )" )           &
               prefix, data%nnz_p_u - data%nnz_p_l + 1
         END IF
    
         IF ( data%control%more_toraldo > 0 .AND.                              &
              data%more_toraldo_its > data%control%more_toraldo ) GO TO 500

!  store the Cauchy point and its gradient for future use

         data%V_cauchy( : data%n_phi ) = data%V_trial( : data%n_phi )
         data%model_cp = data%model

!  store the set of free variables at Cauchy point for future use

         data%nfree_cp = inform%n_free
!        IFREEC( : data%nfreec ) = data%INDEX_nz_p( : data%nfreec )

!  see if an accurate approximation to the minimum of the quadratic model is to
!  be sought

         IF ( data%control%accurate_bqp ) THEN

!  fix the variables which the Cauchy point predicts are active at the solution

           IF ( data%first_cp ) THEN
             data%first_cp = .FALSE.
             WHERE( data%V_status == 1 .OR. data%V_status == 2 )
               data%V_status = 4
               data%BND( : data%n_phi, 1 ) = data%V_trial( : data%n_phi )
               data%BND( : data%n_phi, 2 ) = data%V_trial( : data%n_phi )
             END WHERE
           ELSE

!  update the step taken and the set of variables which are considered free

             inform%n_free = 0
             DO i = 1, nlp%n
               data%P( i ) = data%P( i ) + data%DX_bqp( i )
               IF ( data%P( i ) /= zero .OR. data%V_status( i ) == 0 ) THEN
                 inform%n_free = inform%n_free + 1
                 data%INDEX_nz_p( inform%n_free ) = i
               END IF
             END DO
             data%nnz_p_u = inform%n_free
           END IF
         END IF

         IF ( data%control%more_toraldo > 0 ) data%P( : data%n_phi )           &
             = data%V_trial( : data%n_phi ) - data%V_current( : data%n_phi )

!  if required, print the active set at the generalized Cauchy point

  390    CONTINUE
         IF ( data%printw ) THEN
           WRITE( data%out, "( / )" )
           DO i = 1, nlp%n
             IF ( data%V_status( i ) == 2 .AND.  data%V_trial( i ) >=          &
               nlp%X_u( i ) - ABS( nlp%X_u( i ) ) * epsmch ) WRITE( data%out,  &
               "( A, ' The variable number ', I0, ' is at its upper bound' )" )&
                 prefix, i
             IF ( data%V_status( i ) == 1 .AND. data%V_trial( i ) <=           &
               nlp%X_l( i ) + ABS( nlp%X_l( i ) ) * epsmch ) WRITE( data%out,  &
               "( A, ' The variable number ', I0, ' is at its lower bound' )" )&
                 prefix, i
             IF ( data%V_status( i ) == 4 ) WRITE( data%out,                   &
               "( A, ' The variable number ', I0, ' is temporarily fixed ' )" )&
               prefix, i
           END DO
         END IF

!  ===================================================================
!  4. Compute a Newton-like step s beyond the Generalized Cauchy Point
!  ===================================================================

!  if the model at v_k is q(v_k+s) = f_k + s^T g_k + 0.5 s^T H s,
!  the model q(vc_k + t ) = [ f_k + dv^T g_k + 0.5 dv^T H dv ] +
!                           t^T [ g_k + H dv ] + 0.5 t^T H t

         data%jumpto = 1

!  if an iterative method is to be used, set up convergence tolerances

!      data%cg_stop = MAX( data%resmin, MIN( control%stop_rel_cg, data%rg_norm)&
!        * data%rg_norm * data%rg_norm ) * data%diagonal_min / data%diagonal_max
         data%cg_stop = MIN( control%stop_rel_cg, data%rg_norm )               &
           * data%rg_norm * data%rg_norm * data%diagonal_min / data%diagonal_max
!        IF ( data%control%two_norm_tr .AND. .NOT. data%direct ) THEN
         IF ( data%control%two_norm_tr ) THEN
           data%dxsqr = DOT_PRODUCT( data%P( : nlp%n ), data%P( : nlp%n ) )
           IF ( data%printw ) WRITE( data%out,                                 &
               "( /, A, ' Two-norm of step to Cauchy point = ', ES12.4 )" )    &
                 prefix, SQRT( data%dxsqr )
         END IF
         data%step = inform%radius

!  set a limit on the number of CG iterations that are to be allowed

         data%itercg = 0
         inform%cg_maxit = nlp%n
         data%no_bounds_cg                                                     &
           = data%control%more_toraldo > 0 .AND. data%control%two_norm_tr

!  calculate an approximate minimizer of the model within the specified bounds

  400    CONTINUE
!write(6,*)  ' jumpto ', data%jumpto, data%nnz_p_l, data%nnz_p_u
         IF ( data%jumpto /= 4 ) THEN

!  the product of the Hessian with the vector P is required

           IF ( data%jumpto /= 2 ) THEN
             data%n_prods = data%n_prods + 1
             data%nnz_p_l = 1
             CALL CPU_time( data%time_record )
             CALL CLOCK_time( data%clock_record )

!  set the required components of Q to zero

             IF ( data%jumpto == 1 ) THEN
               data%HP = zero
             ELSE
               data%HP( data%INDEX_nz_p( : data%nnz_p_u ) ) = zero
             END IF

!  compute the matrix-vector product with the dense vector P

             data%dense_p = .TRUE.
             CALL TRAL_hessian_times_vector( nlp%n,                            &
                                            data%INDEX_nz_p, data%nnz_p_l,     &
                                            data%nnz_p_u, data%INDEX_nz_hp,    &
                                            data%nnz_hp, data%INDEX_used_hp,   &
                                            data%n_prods, data%P, data%HP,     &
                                            data%H_by_cols, data%dense_p )
             CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!            data%tmv = data%tmv + data%time_now - data%time_record
!write(6,*)  ' p ', data%P
!write(6,*)  ' hp ', data%HP

!  if required, print a list of the nonzeros of P

             IF ( data%printd )                                                &
               WRITE( data%out, 2260 ) prefix, data%n_prods,                   &
                 data%INDEX_used_hp( : data%nnz_hp )
               IF ( data%jumpto == 1 ) THEN

!  if required, print the step taken

               IF ( data%out > 0 .AND. data%print_level >= 20 )                &
                 WRITE( data%out, 2250 ) prefix, data%P( : nlp%n )

!  compute the value of the model at the generalized Cauchy point and then
!  reset P to zero

               data%model_new = data%model
               data%model = data%model_start
               DO j = 1, data%nnz_p_u
                 i = data%INDEX_nz_p( j )
                 data%model = data%model                                       &
                   + ( data%G_current( i ) + half * data%HP( i ) ) * data%P( i )
                 data%P( i ) = zero
               END DO

!  if required, compare the recurred and calculated model values

               IF ( data%printw ) WRITE( data%out,                             &
                 "( A, ' *** Calculated quadratic at CP ', ES22.14, /,         &
              &     A, ' *** Recurred   quadratic at CP ', ES22.14 )" )        &
                 prefix, data%model, prefix, data%model_new
             END IF
           ELSE

!  evaluate the 'preconditioned' gradient. If the user has supplied a
!  preconditioner, return to the calling program

!            IF ( data%control%preconditioner == - 3 ) THEN
!              inform%status = - 9 ; GO TO 800
!            ELSE

!  if required, use a preconditioner

             CALL CPU_time( data%time_record )
             CALL CLOCK_time( data%clock_record )
!            CALL PRECN_use_preconditioner(                                    &
!              data%ifactr, data%munks, data%use_band, data%seprec,            &
!              data%icfs, nlp%n, ng, nel, data%ntotel, data%nnza,              &
!              data%maxsel, data%nadd, data%nvargp, data%nfreef,               &
!              data%nfixed, control%io_buffer, data%refact, data%nnz_p_u,      &
!              data%INDEX_nz_p, ISTADH, ICNA, ISTADA, INTVAR, IELVAR,          &
!              data%nvrels, IELING, ISTADG, ISTAEV, IFREE,  A, FUVALS,         &
!              data%lnguvl, FUVALS, data%lnhuvl, GVALS( : , 2 ),               &
!              GVALS( : , 3 ), data%P , data%HP, GSCALE_used, ESCALE,          &
!              GXEQX_used , INTREP, RANGE , data%icfact,                       &
!              inform%ciccg, inform%nsemib, inform%ratio,                      &
!              data%print_level, data%error, data%out, data%infor,             &
!              alloc_status, bad_alloc,                                        &
!              ITYPEE, DIAG, OFFDIA, IW, IKEEP, IW1, IVUSE,                    &
!              H_col_ptr, L_col_ptr, W, W1, RHS, RHS2, P2,                     &
!              G, ISTAGV, ISVGRP,                                              &
!              lirnh, ljcnh, lh, lirnh_min, ljcnh_min, lh_min,                 &
!              LINK_col, POS_in_H, llink, lpos,                                &
!              IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                         &
!              matrix, SILS_data, control%SILS_cntl,                           &
!              inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,        &
!              data%PRECN, SCU_matrix, SCU_data, inform%SCU_info,              &
!              data%ASMBL, data%skipg, KNDOFG )

!  form and factorize the preconditioner

  410        CONTINUE
!write(6,*) ' refactor ', data%refactorize
             IF ( data%refactorize ) THEN
               IF ( data%printt ) WRITE( data%out,                             &
                      "( A, ' Computing preconditioner' )" ) prefix
               CALL PSLS_form_and_factorize( nlp%H, data%PSLS_data,            &
                   data%control%PSLS_control, inform%PSLS_inform,              &
                   SUB = data%INDEX_nz_p( : data%nnz_p_u ) )

!  check for error returns

               IF ( inform%PSLS_inform%status /= 0 ) THEN
                 inform%status = inform%PSLS_inform%status  ; GO TO 900
               END IF
               IF ( inform%PSLS_inform%perturbed ) data%perturb = 'p'
               data%control%PSLS_control%new_structure = .FALSE.

!  update the preconditioner

             ELSE
               CALL PSLS_update_factors( data%FIX( : data%n_fix ),             &
                                         data%PSLS_data,                       &
                                         data%control%PSLS_control,            &
                                         inform%PSLS_inform )

!  check for error returns

               IF ( inform%PSLS_inform%status > 0 ) THEN
                 CALL PSLS_index_submatrix( data%nnz_p_u, data%INDEX_nz_p,     &
                                            data%PSLS_data )
                 data%refactorize = .TRUE.
                 GO TO 410
               ELSE IF ( inform%PSLS_inform%status /= 0 ) THEN
                 inform%status = inform%PSLS_inform%status  ; GO TO 900
               END IF
             END IF

!  apply the preconditioner

             data%HP = data%WK
!write(6,*) ' p ', data%HP
             CALL PSLS_solve( data%HP, data%PSLS_data,                         &
                              data%control%PSLS_control, inform%PSLS_inform )
!write(6,*) ' hp ', data%HP

             CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!            data%tls = data%tls + data%time_now - data%time_record
           END IF
         END IF
!      END IF

!  the minimization will take place over all variables which are not on the
!  trust-region boundary with negative gradients pushing over the boundary

!      IF ( data%direct ) THEN

!  - - - - - - - - - - - - direct method - - - - - - - - - - - - - - - -

!  minimize the quadratic using a direct method. The method used is a
!  multifrontal symmetric indefinite factorization scheme. Evaluate the
!  gradient of the quadratic at data%V_trial

!        inform%n_free = 0
!        data%g_model = zero
!        DO i = 1, n
!          IF ( data%V_status( i ) == 0 ) THEN
!            inform%n_free = inform%n_free + 1
!            data%INDEX_nz_p( inform%n_free ) = i
!            gi = data%G_current( i ) + data%HP( i )
!            data%P( inform%n_free ) = gi
!            data%g_model = MAX( data%g_model, ABS( gi ) )
!          ELSE
!            gi = zero
!          END IF
!          data%P( i ) = zero ; QGRAD( i ) = gi
!        END DO
!        data%nnz_p_u = inform%n_free

!  check if the gradient of the model at the generalized Cauchy point
!  is already small enough. Compute the ( scaled ) step moved from the
!  previous to the current iterate

!        IF ( data%control%two_norm_tr ) THEN
!          data%step =                                                         &
!            TWO_NORM( data%V_trial( : nlp%n ) - nlp%X( : nlp%n ) )
!        ELSE
!          data%step =                                                         &
!            INFINITY_NORM( data%V_trial( : nlp%n ) - nlp%X( : nlp%n ) )
!        END IF

!  if the step taken is small relative to the trust-region radius,
!  ensure that an accurate approximation to the minimizer of the
!  model is found

!        IF ( data%step <= stptol * inform%radius ) THEN
!          IF ( MAX( data%resmin, data%step * data%cg_stop /                   &
!            ( inform%radius * stptol ) ) >= data%g_model ) GO TO 500
!        ELSE
!          IF ( data%g_model * data%g_model < data%cg_stop ) GO TO 500
!        END IF

!  factorize the matrix and obtain the solution to the linear system, a
!  direction of negative curvature or a descent direction for the quadratic
!  model

!        CALL CPU_time( data%time_record ); CALL CLOCK_time( data%clock_record )
!        IF ( data%control%more_toraldo > 0 ) THEN
!          CALL FRNTL_get_search_direction(                                    &
!              nlp%n, ng, nel, data%ntotel, data%nnza, data%maxsel,            &
!              data%nvargp, control%io_buffer, INTVAR, IELVAR,                 &
!              data%nvrels, INTREP, IELING, ISTADG, ISTAEV, A, ICNA,           &
!              ISTADA, FUVALS, data%lnguvl, FUVALS, data%lnhuvl, ISTADH,       &
!              GXEQX_used, GVALS( : , 2 ), GVALS( : , 3 ), data%INDEX_nz_p,    &
!              data%nnz_p_u, QGRAD, data%P, data%V_trial,                      &
!              data%BND_radius, data%model, GSCALE_used, ESCALE,               &
!              data%V_current, data%control%two_norm_tr, data%no_bounds,       &
!              data%dxsqr, data%radius, data%cg_stop, data%number,             &
!              data%next, data%modchl, RANGE, inform%nsemib, inform%ratio,     &
!              data%print_level, data%error, data%out, data%infor,             &
!              alloc_status, bad_alloc, ITYPEE, DIAG, OFFDIA, IVUSE,           &
!              RHS, RHS2, P2, ISTAGV, ISVGRP,                                  &
!              lirnh, ljcnh, lh, LINK_col, POS_in_H, llink, lpos,              &
!              IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                         &
!              matrix, SILS_data, control%SILS_cntl,                           &
!              inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,        &
!              SCU_matrix, SCU_data, inform%SCU_info, data%ASMBL,              &
!              data%skipg, KNDOFG )
!        ELSE
!          CALL FRNTL_get_search_direction(                                    &
!              nlp%n, ng, nel, data%ntotel, data%nnza, data%maxsel,            &
!              data%nvargp, control%io_buffer, INTVAR, IELVAR,                 &
!              data%nvrels, INTREP, IELING, ISTADG, ISTAEV, A, ICNA,           &
!              ISTADA, FUVALS, data%lnguvl, FUVALS, data%lnhuvl, ISTADH,       &
!              GXEQX_used, GVALS( : , 2 ), GVALS( : , 3 ), data%INDEX_nz_p,    &
!              data%nnz_p_u, QGRAD, data%P, data%V_trial,                      &
!              BND, data%model, GSCALE_used, ESCALE, data%V_current,           &
!              data%control%two_norm_tr, data%no_bounds, data%dxsqr,           &
!              data%radius, data%cg_stop, data%number, data%next,              &
!              data%modchl, RANGE, inform%nsemib, inform%ratio,                &
!              data%print_level, data%error, data%out, data%infor,             &
!              alloc_status, bad_alloc, ITYPEE, DIAG, OFFDIA, IVUSE,           &
!              RHS, RHS2, P2, ISTAGV, ISVGRP,                                  &
!              lirnh, ljcnh, lh, LINK_col, POS_in_H, llink, lpos,              &
!              IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                         &
!              matrix, SILS_data, control%SILS_cntl,                           &
!              inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,        &
!              SCU_matrix, SCU_data, inform%SCU_info, data%ASMBL,              &
!              data%skipg, KNDOFG )
!        END IF
!        CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!        data%tls = data%tls + data%time_now - data%time_record
!        inform%n_free = data%nnz_p_u

!  check for error returns

!        IF ( data%infor == 10 ) THEN
!          inform%status = 4 ; GO TO 820
!        END IF
!        IF ( data%infor == 11 ) THEN 
!          inform%status = 5 ; GO TO 820 
!        END IF
!        IF ( data%infor == 12 ) GO TO 990
!        IF ( data%infor >= 6 ) THEN
!          inform%status = data%infor ; GO TO 820 
!        END IF

!  save details of the system solved

!        data%fill = MAX( data%fill, inform%ratio )
!        data%ISYS( data%infor ) = data%ISYS( data%infor ) + 1
!        data%lisend = data%LSENDS( data%infor )

!  compute the ( scaled ) step from the previous to the current iterate
!  in the appropriate norm

!        IF ( data%control%two_norm_tr ) THEN
!          data%step =                                                         &
!            TWO_NORM( data%V_trial( : nlp%n ) - nlp%X( : nlp%n ) )
!        ELSE
!          data%step =                                                         &
!            INFINITY_NORM( data%V_trial( : nlp%n ) - nlp%X( : nlp%n ) )
!        END IF

!  for debugging, compute the directional derivative and curvature
!  along the direction P

!        IF ( data%printm ) THEN
!          data%INDEX_used_hp( : data%nnz_hp ) = 0
!          data%nnz_p_l = 0
!          DO i = 1, data%nnz_p_u
!            IF ( data%INDEX_nz_p( i ) > 0 ) THEN
!              data%nnz_p_l = data%nnz_p_l + 1
!              data%INDEX_nz_p( data%nnz_p_l ) = data%INDEX_nz_p( i )
!            END IF
!          END DO
!          data%nnz_p_u = data%nnz_p_l ; inform%n_free = data%nnz_p_u
!          data%nnz_p_l = 1 ; data%n_prods = 1

!  evaluate the product of the Hessian with the dense vector P

!        CALL CPU_time( data%time_record ); CALL CLOCK_time( data%clock_record )
!          data%HP = zero
!          data%dense_p = .TRUE.
!          CALL TRAL_hessian_times_vector( nlp%n, data%INDEX_nz_p, data%nnz_p_l,&
!                                         data%nnz_p_u, data%INDEX_nz_hp,      &
!                                         data%nnz_hp, data%INDEX_used_hp,     &
!                                         data%n_prods, data%P, data%HP,       &
!                                         data%H_by_cols, data%dense_p )
!          CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!          data%tmv = data%tmv + data%time_now - data%time_record

!  compute the curvature

!!         data%curv =                                                         &
!!          DOT_PRODUCT( data%HP( data%INDEX_nz_p( : data%nnz_p_u ) ),         &
!!                       data%P( data%INDEX_nz_p( : data%nnz_p_u)))
!          data%curv = zero
!          DO i = 1, data%nnz_p_u
!            data%curv = data%curv +                                           &
!              data%HP( data%INDEX_nz_p( i ) ) * data%P( data%INDEX_nz_p( i ) )
!          END DO

!  compare the calculated and recurred curvature

!          WRITE( data%out, "( ' curv  = ', ES12.4 )" ) data%curv
!          WRITE( data%out, "( ' FRNTL - infor = ', I1 )" ) data%infor
!          IF ( data%infor == 1 .OR. data%infor == 3 .OR.                      &
!               data%infor == 5 ) THEN
!            DO j = 1, data%nnz_p_u
!              i = data%INDEX_nz_p( j )
!              WRITE( data%out, "( ' P, H * P( ', I6, ' ), RHS( ', I6,         &
!             &' ) = ', 3ES15.7 )" ) i, i, data%P( i ), data%HP( i ), QGRAD( i )
!            END DO
!          END IF
!        END IF
!      ELSE

!  - - - - - - - - - - - - iterative method - - - - - - - - - - - - - -

!  minimize the quadratic using an iterative method. The method used is a 
!  safeguarded preconditioned conjugate gradient scheme

         CALL CPU_time( data%time_record ); CALL CLOCK_time( data%clock_record )
         IF ( data%control%more_toraldo > 0 ) THEN
           inform%cg_maxit = COUNT( data%V_status == 0 )
           IF ( data%more_toraldo_its > 0 )                                    &
             inform%cg_maxit = MAX( 10, inform%cg_maxit / 2 )
!          IF ( data%more_toraldo_its > - 1 ) inform%cg_maxit = 10
           CALL CG_solve(                                                      &
               data%n_phi, data%V_current, data%V_trial, data%G_current,       &
               data%BND_radius, data%n_phi, data%V_status, data%cg_stop,       &
               data%model, data%WK,                                            &
               inform%status, data%P, data%HP, data%INDEX_nz_p, inform%n_free, &
               data%nnz_p_u, data%control%two_norm_tr, data%radius,            &
               data%no_bounds_cg, data%g_model, data%dxsqr, data%out,          &
               data%jumpto, data%print_level, one, data%itercg,                &
               inform%cg_maxit, i_fixed, data%WK2, data%CG_save )
         ELSE
           inform%cg_maxit = 3 * COUNT( data%V_status == 0 )
           CALL CG_solve(                                                      &
               data%n_phi, data%V_current, data%V_trial, data%G_current,       &
               data%BND,                                                       &
               data%n_phi, data%V_status, data%cg_stop, data%model, data%WK, &
               inform%status, data%P, data%HP, data%INDEX_nz_p, inform%n_free, &
               data%nnz_p_u, data%control%two_norm_tr, data%radius,            &
               data%no_bounds_cg, data%g_model, data%dxsqr, data%out,          &
               data%jumpto, data%print_level, one, data%itercg,                &
               inform%cg_maxit, i_fixed, data%WK2, data%CG_save )
         END IF
         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!        data%tls = data%tls + data%time_now - data%time_record

         IF ( data%jumpto == 0 .OR. data%jumpto == 4 .OR.                      &
              data%jumpto == 5 ) THEN
           IF ( data%control%two_norm_tr ) THEN
             data%step =                                                       &
               TWO_NORM( data%V_trial( : nlp%n ) - nlp%X( : nlp%n ) )
           ELSE
             data%step =                                                       &
               INFINITY_NORM( data%V_trial( : nlp%n ) - nlp%X( : nlp%n ) )
           END IF
         END IF

!  the norm of the gradient of the quadratic model is smaller than cg_stop. 
!  Perform additional tests to see if the current iterate is acceptable

         data%nnz_p_u = inform%n_free

!  if the (scaled) step taken is small relative to the trust-region radius,
!  ensure that an accurate approximation to the minimizer of the model is found

         IF ( data%jumpto == 4 ) THEN
           IF ( data%step <= stptol * inform%radius .AND.                      &
!              .NOT. control%quadratic_problem .AND.                           &
               .NOT. data%control%accurate_bqp )  THEN
!            IF ( MAX( data%resmin, data%step * data%cg_stop /                 &
!              ( inform%radius * stptol ) ) >=  data%g_model ) THEN
             IF ( data%step * data%cg_stop / ( inform%radius * stptol )        &
                  >=  data%g_model ) THEN
               IF ( data%printw ) WRITE( data%out,                             &
                 "( A, ' Norm of trial step ', ES12.4 )" ) prefix, data%step
               data%jumpto = 0
             ELSE
               IF ( data%printw ) WRITE( data%out,                             &
                 "( /, A, ' CG tolerance of ', ES12.4, ' has not been',        &
              &       ' achieved. ', /, A, ' Actual step length = ', ES12.4,   &
              &       ' Radius = ', ES12.4 )" )                                &
                prefix, data%step * data%cg_stop / ( inform%radius * stptol ), &
                prefix, data%step, inform%radius
             END IF
           ELSE
             data%jumpto = 0
           END IF

!  a bound has been encountered in CG. If the bound is a trust-region bound,
!  stop the minimization

         ELSE IF ( data%jumpto == 5 ) THEN
           data%refactorize = .FALSE.
           data%n_fix = 1 ; data%FIX( data%n_fix ) = ABS( i_fixed )
           IF ( data%control%two_norm_tr ) THEN
             data%jumpto = 2
           ELSE
             IF ( data%control%accurate_bqp ) THEN
               data%jumpto = 2
             ELSE
               data%jumpto = 0
               data%radius = inform%radius

!  the bound encountered is an upper bound

               IF ( i_fixed > 0 ) THEN
                 IF ( data%diagonal_preconditioner ) THEN
                   IF ( nlp%X_u( i_fixed ) < nlp%X( i_fixed ) +                &
                        data%radius / SQRT( data%H_diagonal( i_fixed ) ) )     &
                     data%jumpto = 2
                 ELSE
                   IF ( nlp%X_u( i_fixed ) < nlp%X( i_fixed ) + data%radius  ) &
                     data%jumpto = 2
                 END IF
               ELSE

!  the bound encountered is a lower bound

                 IF ( data%diagonal_preconditioner ) THEN
                   IF ( nlp%X_l( - i_fixed ) > nlp%X( - i_fixed ) -            &
                        data%radius / SQRT( data%H_diagonal( - i_fixed ) ) )   &
                     data%jumpto = 2
                 ELSE
                   IF ( nlp%X_l( - i_fixed ) >                                 &
                        nlp%X( - i_fixed ) - data%radius ) data%jumpto = 2
                 END IF
               END IF
             END IF
           END IF
           IF ( data%printw .AND. data%jumpto == 2 ) WRITE( data%out,          &
            "( /, A, ' Restarting the conjugate gradient iteration ' )" ) prefix
         END IF

!  if the bound encountered was a problem bound, continue minimizing the model

         IF ( data%jumpto > 0 ) GO TO 400
!          data%cgend = data%CGENDS( inform%status - 9 )
!        END IF

!  if required, compute the value of the model from first principles

         IF ( data%printd ) THEN
           data%n_prods = data%n_prods + 1
           inform%n_free = nlp%n
           data%nnz_p_l = 1 ; data%nnz_p_u = inform%n_free

!  compute the step taken, P

           DO i = 1, data%n_phi
             data%INDEX_nz_p( i ) = i
             data%P( i ) = data%V_trial( i ) - data%V_phi( i )
           END DO

!  evaluate the product of the Hessian with the dense vector P

           CALL CPU_time( data%time_record )
           CALL CLOCK_time( data%clock_record )
           data%HP = zero
           data%dense_p = .TRUE.
           CALL TRAL_hessian_times_vector( nlp%n, data%INDEX_nz_p,             &
                                          data%nnz_p_l,                        &
                                          data%nnz_p_u, data%INDEX_nz_hp,      &
                                          data%nnz_hp, data%INDEX_used_hp,     &
                                          data%n_prods, data%P, data%HP,       &
                                          data%H_by_cols, data%dense_p )
           CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!          data%tmv = data%tmv + data%time_now - data%time_record

!  if required, print the step taken

           IF ( data%out > 0 .AND. data%print_level >= 20 )                    &
             WRITE( data%out, 2250 ) prefix, data%P

!  compute the model value, model_new, and reset P to zero

           data%model_new = zero
           DO j = 1, data%nnz_p_u
             i = data%INDEX_nz_p( j )
             data%model_new = data%model_new +                                 &
               ( nlp%G( i ) + half * data%HP( i ) ) * data%P( i )
             data%P( i ) = zero
           END DO
           WRITE( data%out,                                                    &
            "( A, ' *** Calculated quadratic at end CG ', ES22.14, /,          &
          &    A, ' *** Recurred   quadratic at end CG ', ES22.14 )" )         &
             prefix, data%model_new, prefix, data%model
         END IF

!  --------------------------------------
!  If required: More'-Toraldo projections
!  --------------------------------------

         IF ( data%control%more_toraldo > 0 ) THEN
           j = 0
           DO i = 1, nlp%n
             IF ( data%V_trial( i ) < nlp%X_l( i ) .OR.                        &
                  data%V_trial( i ) > nlp%X_u( i ) ) THEN
!              WRITE(6,"(3ES12.4)" ) nlp%X_l(i), data%V_trial( i ), nlp%X_u( i )
               IF ( data%printt ) WRITE( data%out,                             &
             "( /, A, '    Problem bound would be violated so .... ' )" ) prefix
               j = 1
               EXIT 
             END IF
           END DO
         
!  compute P, the step taken to the Cauchy point

           IF ( j == 1 ) THEN
             inform%n_free = nlp%n
             data%nnz_p_u = inform%n_free
           
             data%P( : data%n_phi )                                            &
               = data%V_cauchy( : data%n_phi ) - data%V_phi( : data%n_phi )
!            DO i = 1, nlp%n ; data%INDEX_nz_p( i ) = i; END DO

!  evaluate the product of the Hessian with the dense vector P

             CALL CPU_time( data%time_record )
             CALL CLOCK_time( data%clock_record )
             data%HP = zero
             data%dense_p = .TRUE.
             CALL TRAL_hessian_times_vector( nlp%n, data%INDEX_nz_p,           &
                                            data%nnz_p_l,                      &
                                            data%nnz_p_u, data%INDEX_nz_hp,    &
                                            data%nnz_hp, data%INDEX_used_hp,   &
                                            data%n_prods, data%P, data%HP,     &
                                            data%H_by_cols, data%dense_p )
             CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!            data%tmv = data%tmv + data%time_now - data%time_record

!  recover the Cauchy point and its function and gradient values

             data%model_start = data%model_cp
             data%V_current( : data%n_phi ) = data%V_cauchy( : data%n_phi )
             data%G_current( : data%n_phi )                                    &
               = data%G_phi( : data%n_phi ) + data%HP( : data%n_phi )
               
!  set the Cauchy direction

             data%P( : data%n_phi )                                            &
               = data%V_trial( : data%n_phi ) - data%V_cauchy( : data%n_phi )
             data%step_max = MIN( data%step_max, one )
             
             IF ( data%control%two_norm_tr )                                   &
               data%radius = TWO_norm( data%P( : data%n_phi ) )

!  ensure that a new Schur complement is calculated. Restore the complete
!  list of variables that were free when the factorization was calculated

             data%jumpto = 1
             data%more_toraldo_its = data%more_toraldo_its + 1
             GO TO 310
           END IF
         END IF

!  if required, an accurate approximation to the minimum of the quadratic 
!  model is to be sought

         IF ( data%control%accurate_bqp ) THEN

!  compute the gradient value

           inform%n_free = nlp%n
           data%nnz_p_u = inform%n_free

!  compute the step taken

           DO i = 1, data%n_phi
             data%INDEX_nz_p( i ) = i
             data%DX_bqp( i ) = data%V_trial( i ) - nlp%X( i )
           END DO

!  evaluate the product of the Hessian with the dense step vector

           CALL CPU_time( data%time_record )
           CALL CLOCK_time( data%clock_record )
!          data%HP = zero
           data%dense_p = .TRUE.
           CALL TRAL_hessian_times_vector( nlp%n, data%INDEX_nz_p, data%nnz_p_l,&
                                          data%nnz_p_u, data%INDEX_nz_hp,      &
                                          data%nnz_hp, data%INDEX_used_hp,     &
                                          data%n_prods, data%DX_bqp, data%HP,  &
                                          data%H_by_cols, data%dense_p )
           CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
!          data%tmv = data%tmv + data%time_now - data%time_record

!  compute the model gradient at data%V_trial

           data%G_current( : data%n_phi )                                      &
             = data%G_phi( : data%n_phi ) + data%HP( : data%n_phi )
           delta_norm = MAXVAL( ABS( data%HP( : data%n_phi ) ) )

!  find the norm of the reduced gradient

           data%g_model = TRAL_reduced_gradient_norm( nlp%n, data%V_trial,     &
                                           data%G_current, nlp%X_l, nlp%X_u )
!  check for convergence of the inner iteration

           IF ( data%printt )                                                  &
             WRITE( data%out, "( /, A, '    ** Model gradient is ', ES12.4,    &
            &  ' Required accuracy is ', ES12.4 )" )                           &
               prefix, data%g_model, SQRT( data%cg_stop)
           IF ( data%g_model * data%g_model > data%cg_stop .AND.               &
                delta_norm > epsmch ) THEN

!  the approximation to the minimizer of the quadratic model is not yet
!  good enough. Perform another iteration

!  store the function value at the starting point for the Cauchy search

             data%model_start = data%model

!  set the staring point for the Cauchy step and the Cauchy direction

             data%V_current( : data%n_phi ) = data%V_trial( : data%n_phi )
             data%P( : data%n_phi ) = - data%G_current( : data%n_phi )

!  if possible, use the existing preconditioner

             IF ( data%refactorize ) THEN
               data%refactorize = .TRUE.
             ELSE

!  ensure that a new Schur complement is calculated. Restore the complete
!  list of variables that were free when the factorization was calculated

                data%refactorize = .FALSE.
!               IFREE( : data%nfreef ) = ABS( IFREE( : data%nfreef ) )
              END IF
              data%jumpto = 1
              GO TO 310
            END IF
         END IF
         inform%cg_iter = inform%cg_iter + data%itercg
       END IF






































!  ========================================
!  5. check for acceptance of the new point
!  ========================================

 500   CONTINUE
       data%S( : data%n_phi )                                                  &
         = data%V_trial( : data%n_phi ) - data%V_phi( : data%n_phi ) 
       data%s_norm = TWO_norm( data%S( : data%n_phi ) )

!  see if the correction will make any difference

       IF ( MAXVAL( ABS( data%S( : data%n_phi ) ) /                            &
              MAX( one, nlp%X( : data%n_phi ) ) ) <= data%control%stop_s ) THEN
         inform%status = GALAHAD_error_tiny_step ; GO TO 900
       END IF

!  compute the slope and curvature along the step

       data%stg = DOT_PRODUCT( data%S( : data%n_phi ), nlp%G( : data%n_phi ) )
       data%hstbs = data%model - data%stg

!  prepare for advanced starting-point calculation if requested

       IF ( inform%iter == 1 .AND. data%control%advanced_start > 0 ) THEN
         IF ( data%hstbs > zero ) THEN
           data%radius_max = - half * data%stg * data%s_norm / data%hstbs
         ELSE
           data%radius_max = data%control%maximum_radius
         END IF
         inform%radius = data%s_norm
         data%X_best( : data%n_phi )  = nlp%X( : data%n_phi ) 
         data%phi_best = data%phi
         data%m_best = data%model
       END IF

!  record the current point

       data%V_current( : data%n_phi ) = data%V_phi( : data%n_phi )

!  form the trial point

 510   CONTINUE
       daat%V_phi( : data%n_phi )                                              &
         = data%V_current( : data%n_phi ) + data%S( : data%n_phi )
       
!  evaluate the objective function at the trial point

       IF ( data%reverse_f ) THEN
         nlp%X( : nlp%n ) = data%V_phi( : nlp%n )
         data%branch = 7 ; inform%status = 2 ; RETURN
       ELSE
         IF ( data%constrained ) THEN
           CALL eval_FC( data%eval_status,  data%V_phi( : nlp%n ), userdata,   &
                         data%f_trial, data%C_trial( : nlp%m ) )
         ELSE
           CALL eval_F( data%eval_status,  data%V_phi( : nlp%n ), userdata,    &
                        data%f_trial )
         END IF
       END IF

!  return from reverse communication to obtain the objective value

 520   CONTINUE
       IF ( data%reverse_f ) THEN
         data%f_trial = nlp%f
         IF ( data%constrained ) data%C_trial( : nlp%m ) = nlp%C( : nlp%m )
       END IF
       inform%f_eval = inform%f_eval + 1

!  when the problem is constrained, compute the merit function
!    phi(x,c) = f(x) + y^T(c(x)-c) + 0.5 rho ||c(x)-c)||_2^2

       IF ( data%constrained ) THEN
         data%phi_trial = data%f_trial ; c_norm = zero
         DO i = 1, nlp%m
           c_diff = data%C_trial( i ) - data%V_phi( nlp%n + i )
           data%phi_trial = data%phi_trial + nlp%Y( i ) * c_diff
           c_norm = c_norm + c_diff * c_diff
         END DO
         data%phi_trial = data%phi_trial + half * inform%rho * c_norm

!  in the unconstrained case, record phi(x) = f(x)

       ELSE
         data%phi_trial = data%f_trial
       END IF
!  test to see if the objective appears to be unbounded from below

       IF ( data%phi_trial < control%obj_unbounded ) THEN 
         inform%status = GALAHAD_error_unbounded ; GO TO 990
       END IF 

!  Advanced starting point
!  .......................

!  if an advanced starting point/radius is desired, proceed as per
!  Sartenaer (SISC 18(6) 1990:1788-1803)

       IF ( inform%iter == 1 .AND. data%control%advanced_start > 0 ) THEN
         data%advanced_start_iter = data%advanced_start_iter + 1

!  If the predicted radius is larger than its upper bound, exit

         IF ( inform%radius >= data%radius_max ) GO TO 530

!  perform another iteration

         IF ( data%advanced_start_iter <= data%control%advanced_start ) THEN

!  compute the change in objective and the slope

           data%df = inform%obj - data%phi_trial

!  record any improvement in the merit value

           IF ( data%phi_trial < data%phi_best ) THEN
             data%X_best( : data%n_phi ) = data%V_phi( : data%n_phi )
             data%phi_best = data%phi_trial
             data%m_best = data%model
           END IF

!  compute the ratio of actual to predicted reduction over the current iteration

           ared = data%df + MAX( one, ABS( inform%obj ) ) * teneps
           prered = - data%model + MAX( one, ABS( inform%obj ) ) * teneps
           IF ( ABS( ared ) < teneps .AND. ABS( inform%obj ) > teneps )        &
             ared = prered
           data%ratio = ared / prered
           IF ( data%printm ) WRITE( data%out, "( /, A, ' actual, predicted',  &
          & ' reductions = ', 2ES12.4 )" ) prefix, ared, prered

!  compute radius adjustment factors

           tau_1 = - theta * data%stg / ( data%hstbs - ( one - theta ) *       &
                    ( data%phi_trial - inform%obj - data%stg ) )
           tau_2 = theta * data%stg / ( data%hstbs - ( one + theta ) *         &
                    ( data%phi_trial - inform%obj - data%stg ) )
           tau_min = MIN( tau_1, tau_2 )
           tau_max = MAX( tau_1, tau_2 )

!  very good agreement - increase step using Sartenaer's formula

           IF ( ABS( data%ratio - one ) <= mu_2 ) THEN
             IF ( tau_max < one ) THEN
               tau = gamma_3
             ELSE IF ( tau_max > gamma_4 ) THEN
               tau = gamma_4
             ELSE IF ( tau_1 >= one .AND. tau_1 <= gamma_4 .AND.               &
                       tau_2 < one ) THEN
               tau = tau_1
             ELSE IF ( tau_2 >= one .AND. tau_2 <= gamma_4 .AND.               &
                       tau_1 < one ) THEN
               tau = tau_2
             ELSE
               tau = tau_max
             END IF

!  poor agreement  - decrease step using Sartenaer's formula

           ELSE IF ( ABS( data%ratio - one ) > mu_1 ) THEN
             IF ( tau_min > one ) THEN
               tau = gamma_2
             ELSE IF ( tau_max < gamma_1 ) THEN
               tau = gamma_1
             ELSE IF ( tau_min < gamma_1 .AND. tau_max >= one ) THEN
               tau = gamma_1
             ELSE IF ( tau_1 >= gamma_1 .AND. tau_1 < one .AND.                &
                     ( tau_2 < gamma_1 .OR. tau_2 >= one ) ) THEN
               tau = tau_1
             ELSE IF ( tau_2 >= gamma_1 .AND. tau_2 < one .AND.                &
                     ( tau_1 < gamma_1 .OR. tau_1 >= one ) ) THEN
               tau = tau_2
             ELSE
               tau = tau_max
             END IF

!  acceptable agreement - refine step

           ELSE
             IF ( tau_max < gamma_2 ) THEN
               tau = gamma_2
             ELSE IF ( tau_max > gamma_3 ) THEN
               tau = gamma_3
             ELSE
               tau = tau_max
             END IF
           END IF

!  restrict any increasze so that the radius does not exceed its maximim value

           tau = MIN( tau, data%radius_max / data%radius )

!  update the radius and step length

           data%old_radius = inform%radius
           inform%radius = inform%radius * tau
           data%s_norm = data%s_norm * tau

!  update the slope, curvature and model value

           data%stg = tau * data%stg
           data%hstbs = tau * tau * data%hstbs
           data%model = data%stg + data%hstbs

           IF ( data%printi ) THEN
             IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
               IF ( data%no_bounds ) THEN
                 IF ( data%control%subproblem_direct ) THEN
                   WRITE( data%out, 2010 ) prefix
                 ELSE
                   WRITE( data%out, 2020 ) prefix
                 END IF
               ELSE
                 IF ( data%control%subproblem_direct ) THEN
                   WRITE( data%out, 2030 ) prefix
                 ELSE
                   WRITE( data%out, 2040 ) prefix
                 END IF
               END IF
             END IF
             data%print_1st_header = .FALSE.
             char_iter = STRING_integer_6( inform%iter )
             IF ( data%no_bounds ) THEN
               IF ( inform%iter > 0 ) THEN
                 IF ( data%control%subproblem_direct ) THEN
                   char_facts                                                  &
                     = STRING_integer_6( inform%TRS_inform%factorizations )
                   WRITE( data%out, 2050 ) prefix, char_iter, data%hard,       &
                      data%negcur, data%bndry, inform%obj, inform%norm_pg,     &
                      data%ratio, inform%radius, inform%TRS_inform%multiplier, &
                      char_facts, data%clock_now
                 ELSE
                   char_sit = STRING_integer_6( inform%GLTR_inform%iter )
                   char_sit2 = STRING_integer_6( inform%GLTR_inform%iter_pass2 )
                   WRITE( data%out, 2060 ) prefix, char_iter, data%negcur,     &
                      data%bndry, data%perturb, inform%obj, inform%norm_pg,    &
                      data%ratio, inform%radius, char_sit, char_sit2,          &
                      data%clock_now
                 END IF
               ELSE
                 WRITE( data%out, 2070 ) prefix,                               &
                      char_iter, inform%obj, inform%norm_pg, inform%radius
               END IF
             ELSE
               char_active = STRING_integer_6( n_active )
               IF ( inform%iter > 0 ) THEN
                 IF ( data%control%subproblem_direct ) THEN
                   char_facts                                                  &
                     = STRING_integer_6( inform%TRS_inform%factorizations )
                   WRITE( data%out, 2080 ) prefix, char_iter, data%hard,       &
                      data%negcur, data%bndry, inform%obj, inform%norm_pg,     &
                      data%ratio, inform%radius, char_active, char_facts,      &
                      data%clock_now
                 ELSE
                   char_sit = STRING_integer_6( data%itercg )
                   WRITE( data%out, 2080 ) prefix, char_iter, data%negcur,     &
                      data%bndry, data%perturb, inform%obj, inform%norm_pg,    &
                      data%ratio, inform%radius, char_active, char_sit,        &
                      data%clock_now
                 END IF
               ELSE
                 WRITE( data%out, 2090 ) prefix,                               &
                      char_iter, inform%obj, inform%norm_pg, inform%radius,    &
                      char_active
               END IF
             END IF
           END IF

           IF ( data%control%subproblem_direct ) THEN
             inform%TRS_inform%factorizations = 0
           ELSE
             inform%GLTR_inform%iter = 0
             inform%GLTR_inform%iter_pass2 = 0
           END IF

!  form the next trial step

           data%S( : nlp%n ) = tau * data%S( : nlp%n )
           GO TO 510
         END IF

!  record the best value found

 530     CONTINUE
         inform%iter = inform%iter + data%advanced_start_iter - 1
         IF ( data%phi_best < data%phi ) THEN
           data%S( : data%n_phi )                                              &
             = data%X_best( : data%n_phi ) - data%V_phi( : data%n_phi )
           data%V_phi( : data%n_phi ) = data%X_best( : data%n_phi )
           data%phi_trial = data%phi_best
           data%model = data%m_best
         END IF
       END IF

!  compute the change in objective and the slope

       data%df = inform%obj - data%phi_trial

!  compute the ratio of actual to predicted reduction over the current iteration

       rounding =                                                              &
         MAX( one, ABS( inform%obj ) ) * REAL( nlp%n, KIND = wp ) * epsmch

       ared = data%df + rounding
       prered = - data%model + rounding
       IF ( ABS( ared ) < teneps .AND. ABS( inform%obj ) > teneps )            &
         ared = prered
       data%ratio = ared / prered
       IF ( data%printm ) WRITE( data%out, "( /, A, ' acutual, predicted',     &
      &   ' reductions = ', 2ES12.4 )" ) prefix, ared, prered

!  compute the ratio of actual to predicted reduction over the recent history

       IF ( .NOT. data%monotone ) THEN

!  compute the largest f in the history

         data%ref = MAXLOC( data%PHI_hist( data%non_monotone_history + 2       &
                            - data%max_hist : data%non_monotone_history + 1 ) )
         data%phi_ref = data%PHI_hist( data%ref( 1 ) )

!  use the larger of these two ratios to assess progress

         data%ratio = MAX( data%ratio, ( data%phi_trial - data%phi_ref ) /     &
           ( SUM( data%DMO_hist( data%ref( 1 ) + 1 :                           &
                            data%non_monotone_history + 1 ) ) + data%model ) )
       END IF

!  if the function and model values agree very closely, examime the
!  corresponding gradients

       IF ( ABS( data%ratio - one ) <= rho_quad ) THEN

!  compute the gradient of the model at the new point; store in U

         data%U( : nlp%n ) = nlp%G( : nlp%n )
         IF ( data%control%hessian_available ) THEN
           DO l = 1, nlp%H%ne
             i = nlp%H%row( l ) ; j = nlp%H%col( l ) ; val = nlp%H%val( l )
             data%U( i ) = data%U( i ) + val * data%S( j )
             IF ( i /= j )                                                     &
               data%U( j ) = data%U( j ) + val * data%S( i )
           END DO
 
!  if necessary, return to the user to obtain the model Hessian product with s

         ELSE
           IF ( data%reverse_hprod ) THEN
             data%V( : nlp%n ) = data%S( : nlp%n )
             data%branch = 8 ; inform%status = 5 ; RETURN
           ELSE
             IF ( data%constrained ) THEN
               CALL eval_HLPROD( data%eval_status, nlp%X( : nlp%n ),           &
                                 nlp%Y( : nlp%m ), userdata,                   &
                                 data%U( : nlp%n ), data%S( : nlp%n ) )
             ELSE
               CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),            &
                                userdata, data%U( : nlp%n ),                   &
                                data%S( : nlp%n ) )
             END IF
           END IF
         END IF
       END IF

!  the new point is acceptable

 540   CONTINUE
       IF ( data%ratio >= data%control%eta_successful ) THEN
         data%poor_model = .FALSE.
         inform%obj = data%f_trial
         nlp%X( : nlp%n ) = data%V_phi( : nlp%n )

!  evaluate the gradient of the objective function and, if necessary, the 
!  Jacobian of the constraints
 
         IF ( data%reverse_g ) THEN
            data%branch = 9 ; inform%status = 3 ; RETURN
         ELSE
           IF ( data%constrained ) THEN
             CALL eval_GJ( data%eval_status, nlp%X( : nlp%n ),                 &
                           userdata, nlp%G( : nlp%n ), nlp%J%val( : nlp%J_ne ) )
           ELSE
             CALL eval_G( data%eval_status, nlp%X( : nlp%n ),                  &
                          userdata, nlp%G( : nlp%n ) )
         END IF
       ELSE
         data%poor_model = .TRUE.
         nlp%X( : nlp%n ) = data%V_current( : nlp%n )
       END IF

!  return from reverse communication to obtain the gradient and Jacobian

 550   CONTINUE

       IF ( data%ratio >= data%control%eta_successful ) THEN
         inform%g_eval = inform%g_eval + 1

!  in the unconstrained case, record the gradient of phi is g(x)

         data%G_phi( : nlp%n ) = nlp%G( : nlp%n )

!  when the problem is constrained, compute the gradient of phi,
!    grad phi = ( g(x) +J^T(x) y(x,rho) ), 
!               (       - y(x,rho)      )
!  where y(x,rho) = y + rho (c(x)-c)

         IF ( data%constrained ) THEN
           CALL mop_Ax( one, nlp%J, data%Y( : nlp%m ), one,                    &
                        data%G_phi( : nlp%n ), transpose = .TRUE.,             &
                        m_matrix = nlp%m, n_matrix = nlp%n )
           data%G_phi( nlp%n + 1 : data%n_phi ) = - data%Y( : nlp%m )
         END IF

!  compute the norm of the projected gradient

         data%WK( : nlp%n ) =                                                  &
           TRAL_projection( nlp%n, data%V_phi( : nlp%n ) -                     &
                            nlp%G_phi( : nlp%n ), nlp%X_l, nlp%X_u )
         data%WK( nlp%n + 1 : data%n_all ) =                                   &
           TRAL_projection( nlp%m, data%V_phi( nlp%n + 1 : data%n_all ) -      &
                            nlp%G_phi( nlp%n + 1 : data%n_all ),               &
                            nlp%Cl, nlp%C_u )
         inform%norm_pg = TWO_NORM( data%V_phi( : data%n_all ) -               &
                                    data%WK( : data%n_all )

         data%new_h = .TRUE.

!  update the history

         IF ( data%monotone ) THEN
           data%phi_ref = data%phi
         ELSE

!  shift history of function and model values

           DO i = 1, data%non_monotone_history
             data%PHI_hist( i ) = data%PHI_hist( i + 1 )
             data%DMO_hist( i ) = data%DMO_hist( i + 1 )
           END DO

!  replace the oldest 

           data%PHI_hist( data%non_monotone_history + 1 ) = data%phi
           data%DMO_hist( data%non_monotone_history + 1 ) = data%model

!  find how much past history is allowed

           data%max_hist = MIN( data%max_hist + 1, data%non_monotone_history )
         END IF

!  the new point is not acceptable

       ELSE
         data%new_h = .FALSE.
       END IF

!  compute rho_g, the relative difference in model and true gradients

       IF ( ABS( data%ratio - one ) <= rho_quad ) THEN
         data%rho_g =                                                          &
           MAXVAL( ABS( data%U - nlp%G ) ) / MAXVAL( ABS( nlp%G  ) )
       ELSE
         data%rho_g = - one
       END IF

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start
       IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,     &
      &    F12.2,  ' seconds' )" ) prefix, data%clock_now
       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
              data%time_now > data%control%cpu_time_limit ) .OR.               &
            ( data%control%clock_time_limit >= zero .AND.                      &
              data%clock_now > data%control%clock_time_limit ) ) THEN
         inform%status = GALAHAD_error_cpu_limit ; GO TO 900
       END IF

     GO TO 100

!  =========================
!  End of the main iteration
!  =========================

 900 CONTINUE

!  compute the norm of the projected gradient

     data%WK = TRAL_projection( nlp%n, nlp%X - nlp%G, nlp%X_l, nlp%X_u )
     inform%norm_pg = TWO_NORM( nlp%X - data%WK )

!  print details of solution

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi ) THEN
!      WRITE ( data%out, 2000 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2200 ) inform%f_eval, inform%g_eval, inform%h_eval,   &
!         inform%iter, inform%cg_iter, inform%obj, inform%norm_pg
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
!              nlp%X( i ), nlp%X_u( i ), nlp%G( i )
!         END DO
!      END DO
       WRITE( data%out, "( /, A, '  Problem: ', A,                             &
      &   ' (n = ', I0, '): TRAL stopping tolerance =', ES11.4 )" )             &
         prefix, TRIM( nlp%pname ), nlp%n, data%stop_pg
       IF ( .NOT. data%monotone ) WRITE( data%out,                             &
           "( A, '  Non-monotone method used (history = ', I0, ')' )" ) prefix,&
         data%non_monotone_history
       IF ( data%no_bounds .AND. data%control%subproblem_direct ) THEN
         IF ( inform%TRS_inform%dense_factorization ) THEN
           WRITE( data%out,                                                    &
           "( A, '  Direct solution (eigen solver SYSV',                       &
          &      ') of the trust-region sub-problem' )" ) prefix
         ELSE
           WRITE( data%out,                                                    &
           "( A, '  Direct solution (solver ', A,                              &
          &      ') of the trust-region sub-problem' )" )                      &
              prefix, TRIM( data%control%TRS_control%definite_linear_solver )
         END IF
         WRITE( data%out, "( A, '  Number of factorization = ', I0,            &
        &     ', factorization time = ', F0.2, ' seconds'  )" ) prefix,        &
           inform%TRS_inform%factorizations,                                   &
           inform%TRS_inform%time%clock_factorize
         IF ( TRIM( data%control%TRS_control%definite_linear_solver ) ==       &
              'pbtr' ) THEN
           WRITE( data%out, "( A, '  Max entries in factors = ', I0,           &
          & ', semi-bandwidth = ', I0  )" ) prefix, inform%max_entries_factors,&
              inform%TRS_inform%SLS_inform%semi_bandwidth
         ELSE
           WRITE( data%out, "( A, '  Max entries in factors = ', I0 )" )       &
             prefix, inform%max_entries_factors
         END IF
       ELSE
         IF ( data%nprec > 0 )                                                 &
           WRITE( data%out, "( A, '  Final Hessian semi-bandwidth (original,', &
          &     ' re-ordered) = ', I0, ', ', I0 )" ) prefix,                   &
             inform%PSLS_inform%semi_bandwidth,                                &
             inform%PSLS_inform%reordered_semi_bandwidth
         IF ( data%no_bounds ) THEN
           SELECT CASE ( data%nprec )
           CASE ( - 3 ) 
             WRITE( data%out, "( A, '  User-defined norm used' )" )            &
               prefix
           CASE ( - 2 ) 
             WRITE( data%out, "( A, 2X, I0, '-step Limited Memory ',           &
            &  'norm used' )" ) prefix, data%lbfgs_mem
           CASE ( - 1 ) 
             WRITE( data%out, "( A, '  Two-norm used' )" ) prefix
           CASE ( 1 ) 
             WRITE( data%out, "( A, '  Diagonal norm used' )" ) prefix
           CASE ( 2 ) 
             WRITE( data%out, "( A, '  Band norm (semi-bandwidth ',            &
            &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
           CASE ( 3 ) 
             WRITE( data%out, "( A, '  Re-ordered band norm (semi-bandwidth ', &
            &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
           CASE ( 4 ) 
             WRITE( data%out, "( A, '  SE (solver ', A, ') full norm used' )" )&
               prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
           CASE ( 5 ) 
             WRITE( data%out, "( A, '  GMPS (solver ', A, ') full norm used')")&
               prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
           CASE ( 6 ) 
             WRITE( data%out, "(A,'  Lin-More''(', I0, ') incomplete Cholesky',&
            &  ' factorization used ' )" ) prefix, data%control%icfs_vectors
           END SELECT
         ELSE
           SELECT CASE ( data%nprec )
           CASE ( - 3 ) 
             WRITE( data%out, "( A, '  User-defined preconditioner used' )" )  &
               prefix
           CASE ( - 2 ) 
             WRITE( data%out, "( A, 2X, I0, '-step Limited Memory ',           &
            &  'preconditioner used' )" ) prefix, data%lbfgs_mem
           CASE ( - 1 ) 
             WRITE( data%out, "( A, '  No preconditioner used' )" ) prefix
           CASE ( 1 ) 
             WRITE( data%out, "( A, '  Diagonal preconditioner used' )" ) prefix
           CASE ( 2 ) 
             WRITE( data%out, "( A, '  Band preconditioner (semi-bandwidth ',  &
            &   I0, ') used' )" ) prefix, inform%PSLS_inform%semi_bandwidth_used
           CASE ( 3 ) 
             WRITE( data%out, "( A, '  Re-ordered band preconditioner',        &
            &   ' (semi-bandwidth ', I0, ') used' )" )                         &
               prefix, inform%PSLS_inform%semi_bandwidth_used
           CASE ( 4 ) 
             WRITE( data%out, "( A, '  SE (solver ', A,                        &
            &   ') full preconditioner used' )" )                              &
               prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
           CASE ( 5 ) 
             WRITE( data%out, "( A, '  GMPS (solver ', A,                      &
            &   ') full preconditioner used' )" )                              &
               prefix, TRIM( data%control%PSLS_control%definite_linear_solver )
           CASE ( 6 ) 
             WRITE( data%out, "(A,'  Lin-More''(', I0, ') incomplete Cholesky',&
            &  ' factorization used ' )" ) prefix, data%control%icfs_vectors
           END SELECT
         END IF
         IF ( data%control%renormalize_radius ) WRITE( data%out,               &
            "( A, '  Radius renormalized' )" ) prefix
       END IF
       WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" ) &
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
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'TRAL_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' Problem: ', A, ' n = ', I8 )
 2010 FORMAT( A, '    It         f        pgrad     ',                         &
             ' ratio   radius  multplr # fact        time ' )
 2020 FORMAT( A, '    It         f        pgrad     ',                         &
             ' ratio   radius  pass 1 pass 2        time ' )
 2030 FORMAT( A, '    It         f        pgrad     ',                         &
             ' ratio   radius # active # fact        time ' )
 2040 FORMAT( A, '    It         f        pgrad     ',                         &
             ' ratio   radius # active cg its        time ' )
 2050 FORMAT( A, A6, 3A1, 2ES12.4, 3ES9.1, A7, F12.2 )
 2060 FORMAT( A, A6, 3A1, 2ES12.4, 2ES9.1, 2A7, F12.2 )
 2070 FORMAT( A, A6, 1X, 2ES12.4, 9X, ES9.1 )
 2080 FORMAT( A, A6, 3A1, 2ES12.4, 2ES9.1, 2X, A7, A7, F12.2 )
 2090 FORMAT( A, A6, 2X, 2ES12.4, 9X, ES9.1, 2X, A7 )
 2200 FORMAT( /, A, ' # function evaluations  = ', I10,                        &
              /, A, ' # gradient evaluations  = ', I10,                        &
              /, A, ' # Hessian evaluations   = ', I10,                        &
              /, A, ' # major  iterations     = ', I10,                        &
              /, A, ' # minor (cg) iterations = ', I10,                        &
             //, A, ' Current objective value = ', ES22.14,                    &
              /, A, ' Current gradient norm   = ', ES12.4 )
 2210 FORMAT( /, A, ' name             X_l        X         X_u         G ' )
 2220 FORMAT(  A, 1X, A10, 4ES12.4 )
 2230 FORMAT(  A, 1X, I10, 4ES12.4 )
 2240 FORMAT( A, ' .          ........... ...........' )
 2250 FORMAT( /, A, ' Change in X = ', / ( '    ', 6ES12.4 ) )
 2260 FORMAT( /, A, ' The matrix-vector product involved indices marked ', I0, &
              ' in the following list ', /, ( '    ', 12I6 ) )   

 !  End of subroutine TRAL_solve

     END SUBROUTINE TRAL_solve

!- G A L A H A D -  T R A L _ u p d a t e _ h i s t o r y  S U B R O U T I N E -

     SUBROUTINE TRAL_update_history( history, max_hist, F_hist, F_ref, f )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: history
     INTEGER, INTENT( INOUT ) :: max_hist
     REAL ( KIND = wp ), INTENT( OUT ) :: F_ref
     REAL ( KIND = wp ), INTENT( IN ) :: f
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( history + 1 ) :: F_hist

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i

!  Shift history of function values

     DO i = 1, history
       F_hist( i ) = F_hist( i + 1 )
     END DO

!  Replace the oldest 

     F_hist( history + 1 ) = f

!  Find how much past history is allowed

     max_hist = MIN( max_hist + 1, history )

!  Compute the largest f in the history

     f_ref = MAXVAL( F_hist( history + 2 - max_hist : history + 1 ) )

     RETURN

 !  End of subroutine TRAL_solve

     END SUBROUTINE TRAL_update_history

!-*-*-  G A L A H A D -  T R A L _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE TRAL_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( TRAL_data_type ), INTENT( INOUT ) :: data
     TYPE ( TRAL_control_type ), INTENT( IN ) :: control
     TYPE ( TRAL_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'TRAL: data%X_best'
     CALL SPACE_dealloc_array( data%X_best,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%V_cauchy'
     CALL SPACE_dealloc_array( data%V_cauchy,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%V_current'
     CALL SPACE_dealloc_array( data%V_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%V_trial'
     CALL SPACE_dealloc_array( data%V_trial,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%G_current'
     CALL SPACE_dealloc_array( data%G_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%WK'
     CALL SPACE_dealloc_array( data%WK,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%WK2'
     CALL SPACE_dealloc_array( data%WK2,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%RHO'
     CALL SPACE_dealloc_array( data%RHO,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%ALPHA'
     CALL SPACE_dealloc_array( data%ALPHA,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%DMO_hist'
     CALL SPACE_dealloc_array( data%DMO_hist,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%F_hist'
     CALL SPACE_dealloc_array( data%F_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%DX_bqp'
     CALL SPACE_dealloc_array( data%DX_bqp,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%DX'
     CALL SPACE_dealloc_array( data%DX,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%DG'
     CALL SPACE_dealloc_array( data%DG,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%BANDH'
     CALL SPACE_dealloc_array( data%BANDH,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%BND'
     CALL SPACE_dealloc_array( data%BND,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%BND_radius'
     CALL SPACE_dealloc_array( data%BND_radius,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%INDEX_used_hp'
     CALL SPACE_dealloc_array( data%INDEX_used_hp,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%V_status'
     CALL SPACE_dealloc_array( data%V_status,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%INDEX_nz_p'
     CALL SPACE_dealloc_array( data%INDEX_nz_p,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%INDEX_nz_hp'
     CALL SPACE_dealloc_array( data%INDEX_nz_hp,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%POSITION_diagonal'
     CALL SPACE_dealloc_array( data%POSITION_diagonal,                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%H_diagonal'
     CALL SPACE_dealloc_array( data%H_diagonal,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%P'
     CALL SPACE_dealloc_array( data%P,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%HP'
     CALL SPACE_dealloc_array( data%HP,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%MAP'
     CALL SPACE_dealloc_array( data%MAP,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%H_by_cols%ROW'
     CALL SPACE_dealloc_array( data%H_by_cols%ROW,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%H_by_cols%COL'
     CALL SPACE_dealloc_array( data%H_by_cols%COL,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%H_by_cols%VAL'
     CALL SPACE_dealloc_array( data%H_by_cols%VAL,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'TRAL: data%H_by_cols%PTR'
     CALL SPACE_dealloc_array( data%H_by_cols%PTR,                             &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Deallocate all arrays allocated within PSLS

     CALL PSLS_terminate( data%PSLS_data, data%control%PSLS_control,           &
                          inform%PSLS_inform )
     inform%status = inform%PSLS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%PSLS_inform%alloc_status
       inform%bad_alloc = inform%PSLS_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arrays allocated within GLTR

     CALL GLTR_terminate( data%GLTR_data, data%control%GLTR_control,           &
                          inform%GLTR_inform )
     inform%status = inform%GLTR_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%GLTR_inform%alloc_status
       inform%bad_alloc = inform%GLTR_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arraysn allocated within TRS

     CALL TRS_terminate( data%TRS_data, data%control%TRS_control,              &
                          inform%TRS_inform )
     inform%status = inform%TRS_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%TRS_inform%alloc_status
       inform%bad_alloc = inform%TRS_inform%bad_alloc
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

!  End of subroutine TRAL_terminate

     END SUBROUTINE TRAL_terminate

!-*- T R A L _ h e s s i a n _ t i m e s _ v e c t o r  S U B R O U T I N E  -*-

     SUBROUTINE TRAL_hessian_times_vector( n, INDEX_nz_p, nnz_p_l, nnz_p_u,    &
                                           INDEX_nz_hp, nnz_hp, INDEX_used_hp, &
                                           n_prods, P, HP, H_by_cols, dense_p )

!  Evaluate HP, the product of the Hessian, stored in extended 
!  (both lower and upper triangules) column format, with the vector P

!  The nonzero components of P have indices INDEX_nz_p( i ), 
!    i = nnz_p_l, ..., nvar_u.
!  The nonzero components of the product HP have indices INDEX_nz_hp( i ),
!  i = 1, ..., nnz_hp
!  The components of INDEX_used_hp must be less than n_prods on entry; on exit 
!  they will be no larger than n_prods, and the i-th component equal to 
!  n_prods if and only if HP(i) is nonzero

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     INTEGER, INTENT( IN ) :: nnz_p_l, nnz_p_u , n_prods
     INTEGER, INTENT( INOUT ) :: nnz_hp
     LOGICAL, INTENT( IN ) :: dense_p
     INTEGER, INTENT( IN ), DIMENSION( n ) :: INDEX_nz_p
     INTEGER, INTENT( INOUT ), DIMENSION( n ) :: INDEX_nz_hp
     INTEGER, INTENT( INOUT ), DIMENSION( n ) :: INDEX_used_hp
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: P
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: HP
     TYPE ( SMT_type ), INTENT( IN ) :: H_by_cols

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k, l
     REAL ( KIND = wp ) :: pj

!  p is not sparse

     IF ( dense_p ) THEN
       HP( : n ) = zero
       INDEX_used_hp = n_prods
       DO j = 1, n
         pj = P( j )
         DO k = H_by_cols%PTR( j ), H_by_cols%PTR( j + 1 ) - 1
           i = H_by_cols%ROW( k )
           HP( i ) = HP( i ) + H_by_cols%VAL( k ) * pj
         END DO
         INDEX_nz_hp( j ) = j
       END DO
       nnz_hp = n

!  p is sparse

     ELSE
       nnz_hp = 0
       DO l = nnz_p_l, nnz_p_u
         j = INDEX_nz_p( l )
         pj = P( j )
         DO k = H_by_cols%PTR( j ), H_by_cols%PTR( j + 1 ) - 1
           i = H_by_cols%ROW( k )
           IF ( INDEX_used_hp( i ) < n_prods ) THEN
             HP( i ) = H_by_cols%VAL( k ) * pj
             INDEX_used_hp( i ) = n_prods
             nnz_hp = nnz_hp + 1
             INDEX_nz_hp( nnz_hp ) = i
           ELSE
             HP( i ) = HP( i ) + H_by_cols%VAL( k ) * pj
           END IF
         END DO
       END DO
     END IF

     RETURN

!  End of subroutine TRAL_hessian_times_vector

     END SUBROUTINE TRAL_hessian_times_vector

!-*-*-*-  G A L A H A D -  T R A L _ p r o j e c t i o n   F U N C T I O N -*-*-

     FUNCTION TRAL_projection( n, X, X_l, X_u )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the projection of x into the set x_l <= x <= x_u

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u
     REAL ( KIND = wp ), DIMENSION( n ) :: TRAL_projection

!  compute the projection

     TRAL_projection = MAX( X_l, MIN( X, X_u ) )
     RETURN

 !  End of function TRAL_projection

     END FUNCTION TRAL_projection

!-*-*-  T R A L _ r e d u c e d  _ g r a d i e n t _ n o r m  F U C T I O N  -*-

     FUNCTION TRAL_reduced_gradient_norm( n, X, G, X_l, X_u )
     REAL ( KIND = wp ) :: TRAL_reduced_gradient_norm

!  Compute the norm of the reduced gradient in the feasible box

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) ::  n
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: X, G, X_l, X_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: gi, reduced_gradient_norm

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
     TRAL_reduced_gradient_norm = reduced_gradient_norm

     RETURN

!  End of TRAL_reduced_gradient_norm

     END FUNCTION TRAL_reduced_gradient_norm

!-*-*-*-*-*-*-*-*-*-*-  T R A L _ a c t i v e  F U C T I O N  -*-*-*-*-*-*-*-*-

     FUNCTION TRAL_active( n, X, X_l, X_u )
     INTEGER :: TRAL_active

!  Count the number of active bounds

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) ::  n
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: X, X_l, X_u

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, n_active

     n_active = 0
     DO i = 1, n
       IF ( ABS( X( i ) - X_l( i ) ) <= epsmch ) THEN
         n_active = n_active + 1
       ELSE IF ( ABS( X( i ) - X_u( i ) ) <= epsmch ) THEN
         n_active = n_active + 1
       END IF
     END DO
     TRAL_active = n_active

     RETURN

!  End of TRAL_active

     END FUNCTION TRAL_active

!  End of module GALAHAD_TRAL

   END MODULE GALAHAD_TRAL_double


