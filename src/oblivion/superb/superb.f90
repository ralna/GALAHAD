! THIS VERSION: GALAHAD 2.5 - 08/02/2013 AT 10:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ S U P E R B   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started October 21st, 2002
!   originally released GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SUPERB_double

!     ----------------------------------------------------------
!    |                                                          |
!    | SUPERB, the Sequential Unconstrained minimization of an  |
!    |  l_p Penalty function treating Equality and inequality   |
!    |  Restrictions by Barrier terms                           |
!    |                                                          |
!    | Aim: to find a (local) minimizer of the nonlinear        |
!    | programming problem                                      |
!    |                                                          |
!    |  minimize   f(x)                                         |
!    |  subject to a_i^T x = b_i               i in A           !
!    |             b_i^l <= a_i^T x <= b_i^u   i in L           |
!    |             c_i(x) =  0                 i in E           |
!    |             c_i^l <= c_i(x) <= c_i^u    i in I           |
!    |    and      x^l <= x  <= x^u                             |
!    |                                                          |
!     ----------------------------------------------------------

     USE CUTEst_interface_double
     USE GALAHAD_NORMS_double
     USE GALAHAD_SYMBOLS
!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SPACE_double
     USE GALAHAD_SILS_double
!    USE GALAHAD_QPT_double
     USE GALAHAD_WCP_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_STRING_double, ONLY: STRING_pleural
     USE GALAHAD_SORT_double
     USE GALAHAD_GLTR_double
     USE GALAHAD_PTRANS_double

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: SUPERB_initialize, SUPERB_read_specfile, SUPERB_solve,          &
               SUPERB_terminate

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point99 = 0.99_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

     REAL ( KIND = wp ), PARAMETER :: h_min = one
     REAL ( KIND = wp ), PARAMETER :: k_diag = one
     REAL ( KIND = wp ), PARAMETER :: h_diag = one
     REAL ( KIND = wp ), PARAMETER :: res_large = one
!    REAL ( KIND = wp ), PARAMETER :: mult_min = ten ** ( - 12 )
     REAL ( KIND = wp ), PARAMETER :: mult_min = ten ** ( - 15 )
!    REAL ( KIND = wp ), PARAMETER :: zeta_tol = point01
     REAL ( KIND = wp ), PARAMETER :: zeta_tol = 0.001_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
!    REAL ( KIND = wp ), PARAMETER :: reduce_factor = 0.95_wp
     REAL ( KIND = wp ), PARAMETER :: reduce_factor = half
     REAL ( KIND = wp ), PARAMETER :: max_elastic = ten
!    REAL ( KIND = wp ), PARAMETER :: max_elastic = hundred
!    REAL ( KIND = wp ), PARAMETER :: max_elastic = ten ** 5
!    LOGICAL, PARAMETER :: print_debug = .TRUE.
     LOGICAL, PARAMETER :: print_debug = .FALSE.
     LOGICAL, PARAMETER :: degeneracy_check = .TRUE.
!    LOGICAL, PARAMETER :: degeneracy_check = .FALSE.

!  ===================================
!  The SUPERB_data_type derived type
!  ===================================

     TYPE, PUBLIC :: SUPERB_data_type
       TYPE ( QPT_problem_type ) :: prob
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_stat 
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: LINEAR

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: XSTATE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: XFREE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: S_type
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: SSTATE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ELASTICS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_l_P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_u_P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_l_P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_u_P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U_P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U_u_P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_feas
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: LAMBDA
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_C
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B_CM
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD_b
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD_m
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DV
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BEST
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SCALE_S
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN
       CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: X_name
       CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: C_name

       TYPE ( GLTR_data_type ) :: gltr_data
       TYPE ( WCP_data_type ) :: WCP_data
       TYPE ( SMT_type ) :: K
       TYPE ( SILS_factors ) :: FACTORS
       TYPE ( SILS_control ) :: CNTL
       TYPE ( PTRANS_trans_type ) :: ptrans_transform
       TYPE ( PTRANS_data_type ) :: ptrans_data
     END TYPE SUPERB_data_type

!  ======================================
!  The SUPERB_control_type derived type
!  ======================================

     TYPE, PUBLIC :: SUPERB_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit
       INTEGER :: start_print, stop_print, print_gap, precon, model
       INTEGER :: semibandwidth, io_buffer
       INTEGER :: non_monotone, first_derivatives, second_derivatives
       INTEGER :: cg_maxit, itref_max, indmin, valmin
       INTEGER :: elastic_type_equations, elastic_type_inequalities
       INTEGER :: lanczos_itmax, n_pr_feas_increase_max
       INTEGER :: scale_x, scale_s, scale_c, scale_f
!      INTEGER :: icfact, max_sc, more_toraldo
       REAL ( KIND = wp ) :: stop_p, stop_c, stop_d, acccg, initial_radius
       REAL ( KIND = wp ) :: rho_successful, rho_very_successful, maximum_radius
!      REAL ( KIND = wp ) :: gamma_smallest, gamma_decrease
       REAL ( KIND = wp ) :: radius_decrease_factor
       REAL ( KIND = wp ) :: radius_small_increase_factor
       REAL ( KIND = wp ) :: radius_increase_factor
       REAL ( KIND = wp ) :: penalty_increase_factor
       REAL ( KIND = wp ) :: barrier_decrease_factor
       REAL ( KIND = wp ) :: initial_mu, initial_nu, minimum_mu, maximum_nu
       REAL ( KIND = wp ) :: stop_c_factor, stop_d_factor, infinity
       REAL ( KIND = wp ) :: prfeas, dufeas, inner_fraction_opt, pivot_tol
       REAL ( KIND = wp ) :: inner_stop_relative, inner_stop_absolute
       REAL ( KIND = wp ) :: mu_meaningful_model, mu_meaningful_group
       REAL ( KIND = wp ) :: max_stop_d, max_stop_c, max_pr_feas_growth
!      REAL ( KIND = wp ) :: eta_extremely_successful
!      REAL ( KIND = wp ) :: firstg, firstc
!      LOGICAL :: quadratic_problem, two_norm_tr, exact_gcp
!      LOGICAL :: accurate_bqp, structured_tr, print_max
       LOGICAL :: magical_steps, use_primal_dual, exact_linesearch
       LOGICAL :: superlinear_decrease, get_feasible_first, eliminate_elastics
       LOGICAL :: magical_path, bound_elastics, fulsol, print_matrix
       LOGICAL :: explicit_linear_constraints
       LOGICAL :: space_critical, deallocate_error_fatal
       CHARACTER ( LEN = 30 ) :: alive_file
       TYPE ( GLTR_control_type ) :: gltr_control        
       TYPE ( WCP_control_type ) :: WCP_control
     END TYPE SUPERB_control_type

!  =====================================
!  The SUPERB_time_type derived type
!  =====================================

     TYPE, PUBLIC :: SUPERB_time_type
       REAL :: total, preprocess, analyse, factorize, solve
       REAL :: phase1_total, phase1_analyse, phase1_factorize, phase1_solve
     END TYPE

!  =====================================
!  The SUPERB_inform_type derived type
!  =====================================

     TYPE, PUBLIC :: SUPERB_inform_type
       INTEGER :: status, alloc_status, iter, cg_iter, itcgmx
       INTEGER :: f_eval, g_eval, nvar, ngeval, iskip, ifixed, nfacts, nmods
       INTEGER :: factorization_status
       INTEGER :: factorization_integer, factorization_real
       REAL ( KIND = wp ) :: merit, obj, pjgnrm, pr_feas, du_feas, comp_slack
       REAL ( KIND = wp ) :: ratio, mu, radius, ciccg
       LOGICAL :: newsol
       CHARACTER ( LEN = 10 ) :: pname
       CHARACTER ( LEN = 80 ) :: bad_alloc
       TYPE ( GLTR_info_type ) :: gltr_inform
       TYPE ( PTRANS_inform_type ) :: ptrans_inform
       TYPE ( SUPERB_time_type ) :: time
       TYPE ( WCP_inform_type ) :: WCP_info
     END TYPE SUPERB_inform_type

!  =====================================
!  The SUPERB_phi_data_type derived type
!  =====================================

     TYPE, PRIVATE :: SUPERB_phi_data_type
       REAL ( KIND = wp ) :: mu, nu, scale_s, smax, dc1, dc2
     END TYPE SUPERB_phi_data_type

!--------------------------------
!   I n t e r f a c e  B l o c k
!--------------------------------

!     INTERFACE TWO_NORM
!       FUNCTION SNRM2( n, X, incx )
!       REAL :: SNRM2
!       INTEGER, INTENT( IN ) :: n, incx
!       REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!       END FUNCTION SNRM2
!
!       FUNCTION DNRM2( n, X, incx )
!       DOUBLE PRECISION :: DNRM2
!       INTEGER, INTENT( IN ) :: n, incx
!       DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!       END FUNCTION DNRM2
!     END INTERFACE 

   CONTAINS

!-*-*-*-*  G A L A H A D -  SUPERB_initialize  S U B R O U T I N E -*-*-*-*

     SUBROUTINE SUPERB_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SUPERB controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SUPERB_data_type ), INTENT( INOUT ) :: data
     TYPE ( SUPERB_control_type ), INTENT( OUT ) :: control
     TYPE ( SUPERB_inform_type ), INTENT( OUT ) :: inform

!    INTEGER, PARAMETER :: lmin = 1
     INTEGER, PARAMETER :: lmin = 10000

     inform%status = GALAHAD_ok
 
!  Initalize SILS components

      CALL SILS_initialize( FACTORS = data%FACTORS, control = data%CNTL )
      data%CNTL%ordering = 3
!57V2 data%CNTL%ordering = 2
!57V3 data%CNTL%ordering = 5
!57V2 data%CNTL%scaling = 0
!57V2 data%CNTL%static_tolerance = zero
!57V2 data%CNTL%static_level = zero

!  Intialize GLTR data

     CALL GLTR_initialize( data%gltr_data, control%gltr_control,               &
                           inform%GLTR_inform )

!  Error and ordinary output unit numbers

     control%error = 6
     control%out = 6

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

     control%alive_unit = 60
     control%alive_file = 'ALIVE.d'

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

     control%print_level = 0

!  Maximum number of iterations

     control%maxit = 1000

!   Any printing will start on this iteration

     control%start_print = - 1

!   Any printing will stop on this iteration

     control%stop_print = - 1

!   Printing will only occur every print_gap iterations

     control%print_gap = 1

!  Fulsol specifies whether the full solution or only highlights
!  will be printed

     control%fulsol = .TRUE.

!   Precon specifies the preconditioner to be used for the CG. 
!     Possible values are
!
!      0  automatic 
!      1  no preconditioner, i.e, the identity within full factorization
!      2  full factorization
!      3  band within full factorization
!      4  diagonal using the barrier terms within full factorization

     control%precon = 0

!   Model specifies the Hessian approximation used.
!     Possible values are
!
!      0  automatic 
!      1  quadratic (Newton, H = Hessian Lagrangian)
!      2  linear (H = 0)
!      3  linear+ (H = I)

     control%model = 1

!  The number of vectors allowed in Lin and More's incomplete factorization

!    control%icfact = 5

!  The semi-bandwidth of the band factorization

     control%semibandwidth = 5

!   The maximum dimension of the Schur complement 

!    control%max_sc = 100

!  Unit number of i/o buffer for writing temporary files (if needed)

     control%io_buffer = 75

!  more_toraldo >= 1 gives the number of More'-Toraldo projected searches 
!                to be used to improve upon the Cauchy point, anything
!                else is for the standard add-one-at-a-time CG search

!    control%more_toraldo = 0

!  non-monotone <= 0 monotone strategy used, anything else non-monotone
!                strategy with this history length used.

!    control%non_monotone = 1
     control%non_monotone = 0

!  first_derivatives = 0 if exact first derivatives are given, = 1 if forward
!             finite difference approximations are to be calculated, and 
!             = 2 if central finite difference approximations are to be used

     control%first_derivatives = 0

!  second_derivatives specifies the approximation to the second derivatives
!                used. 0=exact, 1=BFGS, 2=DFP, 3=PSB, 4=SR1

     control%second_derivatives = 0

!   cg_maxit. The maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the dimension of the system + 1

     control%cg_maxit = 200

!  itref_max. The maximum number of iterative refinements allowed

      control%itref_max = 1

!  control the use of elastics for equality constraints via
!    elastic_type_equations. Possible values are:
!    0  symmetric formulation
!    1  elastics will be added
!    2  negative elastics will be added
!    3  automatic choice of elastics
!   -1  elastics will be added + subsequent removal allowed
!   -2  negative elastics will be added + subsequent removal allowed
!   -3  automatic choice of elastics + subsequent removal allowed

      control%elastic_type_equations = - 3

!  control the use of elastics for inequality constraints via 
!    elastic_type_inequalities. Possible values are:
!    1  elastics will be added
!   -1  elastics will be added + subsequent removal allowed

      control%elastic_type_inequalities = - 1

!   indmin. An initial guess as to the integer workspace required by SILS

      control%indmin = 1000

!   valmin. An initial guess as to the real workspace required by SILS

      control%valmin = 1000

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of violation of the constraints (the "primal infeasibility") is smaller than 
!  control%stop_p and the norm of the gradient of the Lagrangian function (the
!  "dual infeasibility") is smaller than control%stop_d

     control%stop_p = tenm5
     control%stop_c = tenm5
     control%stop_d = tenm5
     
!   prfeas & dufeas. The initial primal (dual) variables will not be closer 
!    than prfeas (dufeas) from their bounds 

      control%prfeas = one
      control%dufeas = one

!  Require a relative reduction in the resuiduals from CG of at least acccg

     control%acccg = 0.01_wp

!  The initial trust-region radius - a non-positive value allows the
!  package to choose its own

!    control%initial_radius = - one
     control%initial_radius = hundred

!  The largest possible trust-region radius

     control%maximum_radius = ten ** 20

!  Parameters that define when to decrease/increase the trust-region 
!  (specialists only!)

     control%rho_successful = point01
!    control%rho_successful = point1
!    control%rho_very_successful = 0.75_wp
     control%rho_very_successful = point9
!    control%rho_very_successful = point99
     
     control%radius_decrease_factor = half
     control%radius_small_increase_factor = 1.1_wp
     control%radius_increase_factor = two
     
!    control%eta_extremely_successful = 0.95_wp
!    control%mu_meaningful_model = 0.01_wp
!    control%mu_meaningful_group = 0.1_wp

!  The initial values of the penalty and barrier parameters - negative 
!  values will cause the parameters to be determined automatically

     control%initial_mu = - one
!    control%initial_mu = ten ** 4
!    control%initial_mu = one
     control%initial_nu = one

!  The smallest value of mu allowed before the problem is declared 
!  locally infeasible

     control%minimum_mu = EPSILON( one )

!  The largest value of nu allowed before the problem is declared 
!  locally infeasible

     control%maximum_nu = one / EPSILON( one )

!  The inner iteration will be stopped when the dual feasibility
!  and complementary slackness have fallen below 
!    stop_d_factor * mu and stop_c_factor * mu 
!  respectively

     control%stop_d_factor = one
     control%stop_c_factor = one

!  The factors by which the barrier and penalty parameters are changed

     control%barrier_decrease_factor = point1
     control%penalty_increase_factor = ten

!  The required accuracy of the norm of the projected gradient at the end
!  of the first major iteration

!    control%firstg = point1

!  The required accuracy of the norm of the constraints at the end
!  of the first major iteration

!    control%firstc = point1

!  The largest permitted dual feasibility and complementarity at the
!  end of each major iteration

     control%max_stop_d = one
     control%max_stop_c = one

!  Any bound larger than infinity in absolute value is infinite

     control%infinity = infinity

!   inner_stop_relative and inner_stop_absolute. The search direction is
!    considered as an acceptable approximation to the minimizer of the
!    model if the gradient of the model in the preconditioning(inverse) 
!    norm is less than 
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )

!    control%inner_stop_relative = zero
     control%inner_stop_relative = point01
     control%inner_stop_absolute = SQRT( epsmch )

!   inner_fraction_opt. a search direction which gives at least 
!    inner_fraction_opt times the optimal model decrease will be found

     control%inner_fraction_opt = point1

!   pivot_tol. The threshold pivot used by the matrix factorization.
!    See the documentation for SILS for details

     control%pivot_tol = epsmch ** 0.75

!  Is the function quadratic ? 

!    control%quadratic_problem = .FALSE.

!  two_norm_tr is true if a 2-norm trust-region is to be used, and false 
!                for the infinity norm

!    control%two_norm_tr = .FALSE.

!  exact_gcp is true if the exact Cauchy point is required, and false if an
!                approximation suffices

!    control%exact_gcp = .TRUE.

!  magical_steps is true if magical steps are to be used to improve upon
!                already accepted points, and false otherwise

     control%magical_steps = .TRUE.
!    control%magical_steps = .FALSE.

!  accurate_bqp is true if the the minimizer of the quadratic model within
!                the intersection of the trust-region and feasible box
!                is to be sought (to a prescribed accuracy), and false 
!                if an approximation suffices

!    control%accurate_bqp = .FALSE.

!  structured_tr is true if a structured trust region will be used, and false
!                if a standard trust-region suffices

!    control%structured_tr = .FALSE.

!  For printing, if we are maximizing rather than minimizing, print_max
!  should be .TRUE.

!    control%print_max = .FALSE.

!  use_primal_dual is false if (at least initially) primal multiplier estimates
!   should be used, and true if primal-dual ones are to be prefered

     control%use_primal_dual = .TRUE.

!  use an exact linesearch? Or an Armijo approximation (much cheaper!)

     control%exact_linesearch = .FALSE.

!  handle linear constraints explicitly or treat them as general constraints

!    control%explicit_linear_constraints = .TRUE.
     control%explicit_linear_constraints = .FALSE.

!  Asymptotically decrease the barrier parameter superlinearly?

     control%superlinear_decrease = .TRUE.

!  Control the maximu number of Lanczos iterations that GLTR will
!  take on the trust-region boundary

     control%lanczos_itmax = 5

!  Get feasible at all costs

!    control%get_feasible_first = .TRUE.
     control%get_feasible_first = .FALSE.

!  Eliminate the elastic variables from the linear systems

     control%eliminate_elastics = .TRUE.

!  Use magical steps in the search computation

!    control%magical_path = .FALSE.
     control%magical_path = .TRUE.

!  Impose an upper bound on the elastics

     control%bound_elastics = .TRUE.
!    control%bound_elastics = .FALSE.

!  Scale the initial variabes to make them of O(1)

     control%scale_x = 0
!    control%scale_x = 1

!  Scale the initial constraints to make them of O(1)

     control%scale_c = 0
!    control%scale_c = 1

!  Scale the initial elastic variabes to make them of O(1)

     control%scale_s = 0
!    control%scale_s = 1

!  Scale the initial objective value to make it of O(1)

     control%scale_f = 0
!    control%scale_f = 1

!  Quit the inner iteration if the primal infeasibility has increased by
!  a factor pr_feas_minimal_increase on n_pr_feas_increase_max successive
!  iterations

     control%max_pr_feas_growth = 1.01_wp
     control%n_pr_feas_increase_max = 5

!  If space_critical is true, every effort will be made to use as little
!  space as possible. This may result in longer computation times

     control%space_critical = .FALSE.

!   If deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

     control%deallocate_error_fatal  = .FALSE.

!!  Remove on release !!
     control%print_matrix = .FALSE.

     RETURN

!  End of subroutine SUPERB_initialize

     END SUBROUTINE SUPERB_initialize

!-*-*-   S U P E R B _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE SUPERB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by SUPERB_initialize could (roughly) 
!  have been set as:

! BEGIN SUPERB SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  alive-device                                   60
!  print-level                                    0
!  maximum-number-of-iterations                   1000
!  start-print                                    -1 
!  stop-print                                     -1
!  iterations-between-printing                    1
!  print-full-solution                            YES
!  print-matrix                                   NO
!  preconditioner-used                            0
!  model-used                                     1
!  elastic-type-for-equality-constraints          -1
!  elastic-type-for-inequality-constraints        -1
!  semi-bandwidth-for-band-preconditioner         5
!  unit-number-for-temporary-io                   75
!  history-length-for-non-monotone-descent        0
!  max-lanczos-iterations                         5
!  scale-initial-variables                        1
!  scale-initial-constraints                      1
!  scale-initial-elastic-variables                1
!  scale-initial-objective                        1
!  maximum-iterations-feasibilty-growth-allowed   5
!  first-derivative-approximations                EXACT
!  second-derivative-approximations               EXACT
!  primal-accuracy-required                       1.0D-5
!  dual-accuracy-required                         1.0D-5
!  complementarity-accuracy-required              1.0D-5
!  inner-iteration-relative-accuracy-required     0.01
!  initial-trust-region-radius                    -1.0
!  maximum-radius                                 1.0D+20
!  rho-successful                                 0.01
!  rho-very-successful                            0.9
!  radius-decrease-factor                         0.5
!  radius-small-increase-factor                   1.1
!  radius-increase_factor                         2.0
!  initial-penalty-parameter                      1.0D+4
!  initial-penalty-parameter                      1.0
!  minimum-barrier-parameter                      1.0D-15
!  maximum-penalty-parameter                      1.0D+15
!  stop-dual-factor                               1.0
!  stop-complementarity-factor                    1.0
!  barrier-decrease-factor                        0.1
!  penalty-increase-factor                        10.0
!  no-dual-updates-until-penalty-parameter-below  0.1
!  minimum-initial-primal-feasibility             1.0
!  minimum-initial-dual-feasibility               1.0
!  maximum-stop-dual-feasibility                  1.0
!  maximum-stop-complementarity                   1.0
!  infinity-value                                 1.0D+19
!  maximal-primal-feasibility-growth-allowed      1.01
!  inner-iteration-stop-relative                  0.01
!  inner-iteration-stop-abslute                   1.0D-8
!  use-primal-dual                                YES
!  magical-steps-allowed                          YES
!  use-magical-steps-in-linesearch                YES
!  exact-linesearch                               NO
!  superlinear-decrease                           YES
!  search-along-the-magical-path                  YES
!  bound-elastics-from-above                      YES
!  eliminate-elastics-in-linear-algebra           YES
!  get-feasible-first                             YES
!  explicit-linear-constraints                    YES
!  space-critical                                 NO
!  deallocate-error-fatal                         NO
!  alive-filename                                 ALIVE.d
! END SUPERB SPECIFICATIONS

!  Dummy arguments

     TYPE ( SUPERB_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: lspec = 64
     CHARACTER( LEN = 6 ), PARAMETER :: specname = 'SUPERB'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

     spec(  1 )%keyword = 'error-printout-device'
     spec(  2 )%keyword = 'printout-device'
     spec(  3 )%keyword = 'alive-device'
     spec(  4 )%keyword = 'print-level' 
     spec(  5 )%keyword = 'maximum-number-of-iterations'
     spec(  6 )%keyword = 'start-print'
     spec(  7 )%keyword = 'stop-print'
     spec(  8 )%keyword = 'iterations-between-printing'
     spec(  9 )%keyword = 'preconditioner-used'
     spec( 10 )%keyword = 'elastic-type-for-equality-constraints'
     spec( 51 )%keyword = 'elastic-type-for-inequality-constraints'
     spec( 11 )%keyword = 'semi-bandwidth-for-band-preconditioner'
     spec( 12 )%keyword = 'max-lanczos-iterations'
     spec( 13 )%keyword = 'unit-number-for-temporary-io'
     spec( 14 )%keyword = 'history-length-for-non-monotone-descent'
     spec( 15 )%keyword = 'first-derivative-approximations'
     spec( 16 )%keyword = 'second-derivative-approximations'
     spec( 48 )%keyword = 'maximum-iterations-feasibilty-growth-allowed'
     spec( 50 )%keyword = 'model-used'
     spec( 52 )%keyword = 'scale-initial-variables'
     spec( 53 )%keyword = 'scale-initial-elastic-variables'
     spec( 54 )%keyword = 'scale-initial-constraints'
     spec( 55 )%keyword = 'scale-initial-objective'
     spec( 60 )%keyword = 'print-matrix'

!  Real key-words

     spec( 17 )%keyword = 'primal-accuracy-required'
     spec( 18 )%keyword = 'dual-accuracy-required'
     spec( 19 )%keyword = 'complementarity-accuracy-required'
     spec( 20 )%keyword = 'inner-iteration-relative-accuracy-required'
     spec( 21 )%keyword = 'initial-trust-region-radius'
     spec( 22 )%keyword = 'maximum-radius'
     spec( 23 )%keyword = 'rho-successful'
     spec( 24 )%keyword = 'rho-very-successful'
     spec( 25 )%keyword = 'radius-decrease-factor'
     spec( 26 )%keyword = 'radius-small-increase-factor'
     spec( 27 )%keyword = 'radius-increase-factor'
     spec( 28 )%keyword = 'maximum-penalty-parameter'
     spec( 56 )%keyword = 'mimimum-barrier-parameter'
     spec( 29 )%keyword = 'stop-dual-factor'
     spec( 30 )%keyword = 'stop-complementarity-factor'
     spec( 31 )%keyword = 'initial-barrier-parameter'
     spec( 32 )%keyword = 'initial-penalty-parameter'
     spec( 33 )%keyword = 'barrier-decrease-factor'
     spec( 34 )%keyword = 'penalty-increase-factor'
     spec( 35 )%keyword = 'infinity-value'
     spec( 36 )%keyword = 'minimum-initial-primal-feasibility'
     spec( 37 )%keyword = 'minimum-initial-dual-feasibility'
     spec( 38 )%keyword = 'maximum-stop-dual-feasibility'
     spec( 39 )%keyword = 'maximum-stop-complementarity'
     spec( 49 )%keyword = 'maximal-primal-feasibility-growth-allowed'
     spec( 58 )%keyword = 'inner-iteration-stop-relative'
     spec( 59 )%keyword = 'inner-iteration-stop-abslute'

!  Logical key-words

!    spec( 38 )%keyword = 'subproblem-solved-accuractely'
!    spec( 39 )%keyword = 'exact-GCP-used'
     spec( 40 )%keyword = 'superlinear-decrease'
     spec( 41 )%keyword = 'magical-steps-allowed'
     spec( 42 )%keyword = 'exact-linesearch'
     spec( 43 )%keyword = 'eliminate-elastics-in-linear-algebra'
     spec( 44 )%keyword = 'get-feasible-first'
     spec( 45 )%keyword = 'use-primal-dual'
     spec( 46 )%keyword = 'search-along-the-magical-path'
     spec( 47 )%keyword = 'bound-elastics-from-above'
     spec( 57 )%keyword = 'print-full-solution'
     spec( 61 )%keyword = 'explicit-linear-constraints'
     spec( 62 )%keyword = 'space-critical'
     spec( 63 )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( lspec )%keyword = 'alive-filename'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_integer( spec( 1 ), control%error,                   &
                                   control%error )
     CALL SPECFILE_assign_integer( spec( 2 ), control%out,                     &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 3 ), control%out,                     &
                                   control%alive_unit )                         
     CALL SPECFILE_assign_integer( spec( 4 ), control%print_level,             &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 5 ), control%maxit,                   &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 6 ), control%start_print,             &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 7 ), control%stop_print,              &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 8 ), control%print_gap,               &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 9 ), control%precon,                   &
                                  control%error )                           
     CALL SPECFILE_assign_integer( spec( 10 ), control%elastic_type_equations, &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 51 ),                                 &
                                   control%elastic_type_inequalities,          &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 11 ), control%semibandwidth,          &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 12 ), control%lanczos_itmax,          &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 13 ), control%io_buffer,              &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 14 ), control%non_monotone,           &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 15 ), control%first_derivatives,       &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 16 ), control%second_derivatives,      &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 48 ), control%n_pr_feas_increase_max,  &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 50 ), control%model,                   &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 52 ), control%scale_x,                 &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 53 ), control%scale_s,                 &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 54 ), control%scale_c,                 &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 55 ), control%scale_f,                 &
                                   control%error )                           

!  Set real values

     CALL SPECFILE_assign_real( spec( 17 ), control%stop_p,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 18 ), control%stop_d,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 19 ), control%stop_c,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 20 ), control%acccg,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 21 ), control%initial_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 22 ), control%maximum_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 23 ), control%rho_successful,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 24 ), control%rho_very_successful,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 25 ),                                    &
                 control%radius_decrease_factor, control%error )
     CALL SPECFILE_assign_real( spec( 26 ),                                    &
                 control%radius_small_increase_factor, control%error )
     CALL SPECFILE_assign_real( spec( 27 ),                                    &
                 control%radius_increase_factor, control%error )
     CALL SPECFILE_assign_real( spec( 29 ), control%stop_d_factor,             &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 30 ), control%stop_c_factor,             &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 31 ), control%initial_mu,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 32 ), control%initial_nu,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 56 ), control%minimum_mu,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 28 ), control%maximum_nu,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 33 ), control%barrier_decrease_factor,   &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 34 ), control%penalty_increase_factor,   &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 35 ), control%infinity,                  &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 36 ), control%prfeas,                    &
                                control%error )     
     CALL SPECFILE_assign_real( spec( 37 ), control%dufeas,                    &
                                control%error )
     CALL SPECFILE_assign_real( spec( 38 ), control%max_stop_d,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 39 ), control%max_stop_c,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 49 ), control%max_pr_feas_growth,        &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 58 ), control%inner_stop_relative,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 59 ), control%inner_stop_absolute,       &
                                control%error )                           

!  Set logical values

     CALL SPECFILE_assign_logical( spec( 40 ), control%superlinear_decrease,   &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 41 ), control%magical_steps,          &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 42 ), control%exact_linesearch,       &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 43 ), control%eliminate_elastics,     &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 44 ), control%get_feasible_first,     &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 45 ), control%use_primal_dual,        &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 46 ), control%magical_path,           &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 47 ), control%bound_elastics,         &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 57 ), control%fulsol,                 &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 60 ), control%print_matrix,           &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 61 ),                                 &
                                   control%explicit_linear_constraints,        &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 62 ), control%space_critical,         &
                                   control%error )
     CALL SPECFILE_assign_logical( spec( 63 ),                                 &
                                   control%deallocate_error_fatal,             &
                                   control%error )

!  Set character values

     CALL SPECFILE_assign_string( spec( lspec ), control%alive_file,           &
                                  control%error )                           

!  Read the controls for GLTR

     IF ( PRESENT( alt_specname ) ) THEN
       CALL GLTR_read_specfile( control%GLTR_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-GLTR' )
     ELSE
       CALL GLTR_read_specfile( control%GLTR_control, device )
     END IF

     RETURN

     END SUBROUTINE SUPERB_read_specfile

!-*-*-*-  G A L A H A D -  S U P E R B _ s o l v e  S U B R O U T I N E  -*-*-*-

!    SUBROUTINE SUPERB_solve( input, one_norm, control, inform, data )
     SUBROUTINE SUPERB_solve( input, io_buffer, control, inform, data )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  SUPERB_solve, a method for finding a local minimizer of a function subject 
!  to general constraints and simple bounds on the sizes of the variables.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: input, io_buffer
!    LOGICAL, INTENT( IN ) :: one_norm
     TYPE ( SUPERB_control_type ), INTENT( INOUT ) :: control
     TYPE ( SUPERB_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( SUPERB_data_type ), INTENT( INOUT ) :: data

!  Local variables

     INTEGER :: m, n, nfree, nmhist, H_ne, J_ne, J_len, H_len, l_suc, out, error
     INTEGER :: i, j, l, start_print, stop_print, print_level, cutest_status
     INTEGER :: m_l, m_le, m_li, m_ne, m_ni, ir, ic, liw, inform_sort
     INTEGER :: ii, nnzhs, nnzk, nnzks, precon, nsemib, lk, A_ne, zeig
     INTEGER :: merit_error, itref_max, cg_iter, nbacts, nfrpel, Hfree_ne
     INTEGER :: search_error, i_mu, i_nu, i_mu_end, i_nu_end
     INTEGER :: nelastic, nelastic_old
     INTEGER :: degen, len_s_u, n_pr_feas_increase, stop_inner_status
!    REAL ( KIND = wp ) :: first_radius, norm_gb, J_norm, a_norm
     REAL ( KIND = wp ) :: ared, pred, ratio, slope, curv, old_radius, old_nu
     REAL ( KIND = wp ) :: merit, merit_trial, f, f_trial, step, model
     REAL ( KIND = wp ) :: theta_c, theta_d, res_norm, vTHv, theta_p
     REAL ( KIND = wp ) :: teneps, ar_h, pr_h, nu, mu, val, h_norm, zeta
     REAL ( KIND = wp ) :: merit_min, merit_ref, merit_current, sigma_r, sigma_c
     REAL ( KIND = wp ) :: prfeas, dufeas, b_term, g_term
     REAL ( KIND = wp ) :: step_max, barrier, penalty, h_perturb, mult_max
     REAL ( KIND = wp ) :: initial_radius, jTd, delta, kkt_best, mult_max_eq
     REAL ( KIND = wp ) :: alpha, alpha_x, alpha_s, alpha_c, pr_feas_old
     REAL ( KIND = wp ) :: alpha_y, alpha_z, alpha_u, infeas, comp_slack_primal
     REAL ( KIND = wp ) :: b_cl, b_cu, grad_l, grad_x, old_mu
     REAL ( KIND = wp ) :: mu_try, nu_try, mu_best, nu_best, ratio_final
!    REAL ( KIND = wp ) :: a_max, h_max
!    LOGICAL :: set_first_radius, s_lower
     LOGICAL :: grlagf, new_inner, analyse, auto
     LOGICAL :: new_derivatives, got_ratio, use_primal_dual, blank
     LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
     LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe 
     LOGICAL :: set_printd2, printd2, set_printd4, printd4
     LOGICAL :: refact, big_res, gltr_iter_eq_0, g_recent, y_recent
     LOGICAL :: scale_xcf, invalid
     REAL :: dum, time, time_new, time_total
     CHARACTER ( LEN = 1 ) :: mo, new_mo, restrict
     CHARACTER ( LEN = 80 ) :: array_name

!  Derived types

     TYPE ( SILS_ainfo ) :: AINFO
     TYPE ( SILS_finfo ) :: FINFO

!  Parameters

     INTEGER, PARAMETER :: hist = 5
     REAL, PARAMETER :: max_ratio = 2.0

     REAL ( KIND = wp ), PARAMETER :: step_tiny = ten * epsmch
     REAL ( KIND = wp ), PARAMETER :: nu_1 = point01
     REAL ( KIND = wp ), PARAMETER :: eta = ten ** ( - 4 )
!    REAL ( KIND = wp ), PARAMETER :: eta = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: beta = 1.01_wp
     REAL ( KIND = wp ), PARAMETER :: initial_target_min = one

!  Initialize

     CALL CPU_TIME( time_total ) ; inform%time%total = time_total

     inform%iter = 0 ; inform%nfacts = 0 ; inform%nmods = 0 
     inform%f_eval = 0 ; inform%g_eval = 0

     inform%pname = " unknown "
     f = HUGE( one ) ; inform%obj = f
     inform%pr_feas = HUGE( one ) ; inform%du_feas = HUGE( one )
     inform%status = GALAHAD_ok

     IF ( control%initial_radius > zero ) THEN
       initial_radius = control%initial_radius
     ELSE
       initial_radius = one
     END IF

     teneps = ten * epsmch

!  nmhist is the length of the history if a non-monotone strategy is to be used
  
     nmhist = control%non_monotone

     step = zero
     use_primal_dual = control%use_primal_dual
     nsemib = control%semibandwidth
     data%CNTL%u = control%pivot_tol ; data%CNTL%pivoting = 1
     data%CNTL%lp = - 1 ; data%CNTL%mp = - 1 ; data%CNTL%wp = - 1

!  ===========================
!  Control the output printing
!  ===========================

     IF ( control%start_print < 0 ) THEN
       start_print = - 1
     ELSE
       start_print = control%start_print
     END IF

     IF ( control%stop_print < 0 ) THEN
       stop_print = control%maxit + 1
     ELSE
       stop_print = control%stop_print
     END IF

     out = control%out ; error = control%error 
     set_printe = error > 0 .AND. control%print_level >= 1

!  Basic single line of output per iteration

     set_printi = out > 0 .AND. control%print_level >= 1 

!  As per printi, but with additional timings for various operations

     set_printt = out > 0 .AND. control%print_level >= 2 

!  As per printm, but with checking of residuals, etc

     set_printm = out > 0 .AND. control%print_level >= 3 

!  As per printm but also with an indication of where in the code we are

     set_printw = out > 0 .AND. control%print_level >= 4

!  Full debugging printing with significant arrays printed

     set_printd = out > 0 .AND. control%print_level >= 5

!  Full debugging (level 2) printing with significant arrays printed

     set_printd2 = out > 0 .AND. control%print_level >= 6

!  Full debugging (level 4) printing with significant arrays printed

     set_printd4 = out > 0 .AND. control%print_level >= 8

!  Start setting control parameters

     IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
       printe = set_printe ; printi = set_printi ; printt = set_printt
       printm = set_printm ; printw = set_printw ; printd = set_printd
       printd2 = set_printd2 ; printd4 = set_printd4
       print_level = control%print_level
     ELSE
       printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
       printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
       printd2 = .FALSE. ; printd4 = .FALSE.
       print_level = 0
     END IF

     precon = control%precon 
     auto = precon <= 0
     IF ( .NOT. auto ) THEN
       IF ( precon >= 5 ) THEN
         IF ( printi ) WRITE( out,                                             &
           "( ' precon = ', I6, ' out of range [0,4]. Reset to 4') ") precon
         precon = 4
       END IF
     END IF

!  Discover how many variables and constraints are involved in the problem

     CALL CUTEST_cdimen( cutest_status, input, n, m )
     IF ( cutest_status /= 0 ) GO TO 930

!  Allocate sufficient space to hold the problem

     array_name = 'superb: data%X'
     CALL SPACE_resize_array( n, data%X, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%X_l'
     CALL SPACE_resize_array( n, data%X_l, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%X_u'
     CALL SPACE_resize_array( n, data%X_u, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%LAMBDA'
     CALL SPACE_resize_array( m, data%LAMBDA, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Z'
     CALL SPACE_resize_array( n, data%Z, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%C_l'
     CALL SPACE_resize_array( m, data%C_l, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%C_u'
     CALL SPACE_resize_array( m,data%C_u , inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!    IF (  set_printd ) THEN
     array_name = 'superb: data%C_feas'
     CALL SPACE_resize_array( m, data%C_feas, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910
!    END IF

     array_name = 'superb: data%XSTATE'
     CALL SPACE_resize_array( n, data%XSTATE, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%EQUATN'
     CALL SPACE_resize_array( m, data%EQUATN, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%LINEAR'
     CALL SPACE_resize_array( m, data%LINEAR, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  Set up the correct data structures for subsequent computations.

     CALL CUTEST_csetup( cutest_status, input, control%error, io_buffer,       &
                         n, m, data%X, data%X_l, data%X_u,                     &
                         data%LAMBDA, data%C_l, data%C_u,                      &
                         data%EQUATN, data%LINEAR, 1, 1, 0 )
     IF ( cutest_status /= 0 ) GO TO 930
     IF ( printd ) WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  data%X( : n )

!  Determine how many linear and equality constraints there are

     m_le = 0 ; m_li = 0 ; m_ne  = 0 ; m_ni = 0

     DO i = 1, m
!      write(6,"( I8, 2L2 )" ) i, data%LINEAR( i ), data%EQUATN( i )
       IF ( data%LINEAR( i ) ) THEN
         IF ( data%EQUATN( i ) ) THEN
           m_le = m_le + 1
         ELSE
           m_li = m_li + 1
         END IF
       ELSE
         IF ( data%EQUATN( i ) ) THEN
           m_ne = m_ne + 1
         ELSE
           m_ni = m_ni + 1
         END IF
       END IF
     END DO

!  Starting addresses for constraints:

!    -----------------------------------------------------------------------
!    | linear equal | linear inequal | nonlinear equal | nonlinear inequal |
!    -----------------------------------------------------------------------
!     ^              ^              ^ ^                 ^                   ^
!     |              |            m_l |                 |                   | 
!    m_le           m_li             m_ne              m_ni                 m 

     m_l = m_le + m_li
     m_ni = m - m_ni + 1
     m_ne = m_l + 1
     m_li = m_le + 1
     m_le = 1
     liw = MAX( m, n ) + 1

!    write(6,"( 6I8 )" ) m_le, m_li, m_l, m_ne, m_ni, m 

!    write(out,"('cl',/,(5ES12.4))") data%C_l
!    write(out,"('cu',/,(5ES12.4))") data%C_u

     array_name = 'superb: data%EQUATN'
     CALL SPACE_dealloc_array( data%EQUATN,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Determine which variables are fixed
     
     nfree = 0
     DO i = 1, n
       IF ( data%X_l( i ) == data%X_u( i ) ) THEN
         data%XSTATE( i ) = 0
       ELSE
         nfree = nfree + 1
         data%XSTATE( i ) = nfree
       END IF
     END DO

     array_name = 'superb: data%XFREE'
     CALL SPACE_resize_array( nfree, data%XFREE, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     nfree = 0
     DO i = 1, n
       IF ( data%X_l( i ) /= data%X_u( i ) ) THEN
         nfree = nfree + 1
         data%XFREE( nfree ) = i
       END IF
     END DO

!  Determine how many nonzeros are required to store the matrix of 
!  gradients of the objective function and constraints, when the matrix 
!  is stored in sparse format.

     CALL CUTEST_cdimsj( cutest_status, J_ne )
     IF ( cutest_status /= 0 ) GO TO 930

!  Determine how many nonzeros are required to store the Hessian matrix of the
!  Lagrangian, when the matrix is stored as a sparse matrix in "co-ordinate" 
!  format
 
     CALL CUTEST_cdimsh( cutest_status, H_ne )
     IF ( cutest_status /= 0 ) GO TO 930

!  Allocate further space to hold the problem

     array_name = 'superb: data%C_name'
     CALL SPACE_resize_array( m, data%C_name, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%X_name'
     CALL SPACE_resize_array( n, data%X_name, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%X_trial'
     CALL SPACE_resize_array( n, data%X_trial, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%C'
     CALL SPACE_resize_array( m, data%C, inform%status,                        &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%C_trial'
     CALL SPACE_resize_array( m, data%C_trial, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%G'
     CALL SPACE_resize_array( n, data%prob%G, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%X'
     CALL SPACE_resize_array( n, data%prob%X, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%X_l'
     CALL SPACE_resize_array( n, data%prob%X_l, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%X_u'
     CALL SPACE_resize_array( n, data%prob%X_u, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%Y'
     CALL SPACE_resize_array( m, data%prob%Y, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%Z'
     CALL SPACE_resize_array( n, data%prob%Z, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%C'
     CALL SPACE_resize_array( m, data%prob%C, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%C_l'
     CALL SPACE_resize_array( m, data%prob%C_l, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%C_u'
     CALL SPACE_resize_array( m, data%prob%C_u, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%A%row'
     CALL SPACE_resize_array( J_ne, data%prob%A%row, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%A%col'
     CALL SPACE_resize_array( J_ne, data%prob%A%col, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%A%val'
     CALL SPACE_resize_array( J_ne, data%prob%A%val, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%H%row'
     CALL SPACE_resize_array( H_ne, data%prob%H%row, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%H%col'
     CALL SPACE_resize_array( H_ne, data%prob%H%col, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%H%val'
     CALL SPACE_resize_array( H_ne, data%prob%H%val, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%IW'
     CALL SPACE_resize_array( liw, data%IW, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%S_type'
     CALL SPACE_resize_array( m, data%S_type, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%SSTATE'
     CALL SPACE_resize_array( m, data%SSTATE, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Y_l'
     CALL SPACE_resize_array( m, data%Y_l, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Y_u'
     CALL SPACE_resize_array( m, data%Y_u, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Z_l'
     CALL SPACE_resize_array( n, data%Z_l, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Z_u'
     CALL SPACE_resize_array( n, data%Z_u, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%prob%A%ptr'
     CALL SPACE_resize_array( m + 1, data%prob%A%ptr, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Y_l_P'
     CALL SPACE_resize_array( m, data%Y_l_P, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Y_u_P'
     CALL SPACE_resize_array( m, data%Y_u_P, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Z_l_P'
     CALL SPACE_resize_array( n, data%Z_l_P, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%Z_u_P'
     CALL SPACE_resize_array( n, data%Z_u_P, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!  Ensure that the initial point is feasible with respect to its simple bounds
!  and that the corresponding multipliers are feasible

     prfeas = control%prfeas ; dufeas = control%dufeas
     DO i = 1, n
       data%Z_l( i ) = zero
       data%Z_u( i ) = zero
       IF ( data%X_u( i ) - data%X_l( i ) > two * prfeas ) THEN
         data%X( i ) = MIN( MAX( data%X( i ), data%X_l( i ) + prfeas ),        &
                                 data%X_u( i ) - prfeas ) 
       ELSE IF ( data%X_l( i ) == data%X_u( i ) ) THEN
         data%X( i ) = data%X_l( i ) ; data%X_trial( i ) = data%X( i )
       ELSE IF ( data%X_l( i ) < data%X_u( i ) ) THEN
         data%X( i ) = half * ( data%X_u( i ) + data%X_l( i ) )
       ELSE
         inform%status = - 5 ; GO TO 990
       END IF
!      WRITE( out, "( 4ES12.4 )" )                                             &
!        data%X_l( i ), data%X( i ), data%X_u( i ), prfeas
       IF ( data%X_l( i ) > - control%infinity ) data%Z_l( i ) = dufeas
       IF ( data%X_u( i ) <   control%infinity ) data%Z_u( i ) = dufeas
     END DO

!  Ensure that the constraint bounds are consistent

     DO i = 1, m
       IF ( data%C_u( i ) <  data%C_l( i ) ) THEN
         inform%status = GALAHAD_error_bad_bounds ; GO TO 990
       END IF
     END DO

!  Obtain the names of the problem, its variables and general constraints

     CALL CUTEST_cnames( cutest_status, n, m,                                  &
                         inform%pname, data%X_name, data%C_name )
     IF ( cutest_status /= 0 ) GO TO 930

     WRITE( out, "( /, ' Solver:         SUPERB', /, ' Problem: ', 7X, A10 )" )&
      inform%pname

!  If required, find a strictly interior feasible point for any linear
!  constraints

     IF ( control%explicit_linear_constraints                                  &
          .AND. COUNT( data%LINEAR ) > 0 ) THEN

! Intialize data for the feasible-point LP problem

       data%prob%new_problem_structure = .TRUE.
       data%prob%n = n
       data%prob%X_l = data%X_l
       data%prob%X_u = data%X_u
       data%prob%Hessian_kind = 0
       data%prob%gradient_kind = 0
       data%prob%f = zero
       data%prob%X = data%X
       IF ( ALLOCATED( data%prob%A%type ) ) DEALLOCATE( data%prob%A%type )
       CALL SMT_put( data%prob%A%type, 'COORDINATE', inform%alloc_status )

!  Make a list of the linear constraints

       data%prob%m = 0
       DO i = 1, m
         IF ( data%LINEAR( i ) ) THEN
           data%prob%m = data%prob%m + 1
           data%S_type( i ) = data%prob%m

!  Record the constraint bounds for linear constraints

           data%prob%C_l( data%prob%m ) = data%C_l( i )
           data%prob%C_u( data%prob%m ) = data%C_u( i )
         ELSE
           data%S_type( i ) = 0
         END IF
       END DO

!  Evaluate the gradients of the constraints

       grlagf = .FALSE. ; J_len = J_ne
       CALL CUTEST_csgr( cutest_status, n, m, data%X, data%LAMBDA, grlagf,     &
                         J_ne, J_len, data%prob%A%val, data%prob%A%col,        &
                         data%prob%A%row )
       IF ( cutest_status /= 0 ) GO TO 930
       inform%g_eval = inform%g_eval + 1

!  Untangle the part corresponding to the linear constraints

       data%prob%A%ne = 0
       DO l = 1, J_ne
         IF ( data%prob%A%row( l ) /= 0 ) THEN
           i = data%S_type( data%prob%A%row( l ) )
           IF ( i > 0 ) THEN
             data%prob%A%ne = data%prob%A%ne + 1
             data%prob%A%row( data%prob%A%ne ) = i
             data%prob%A%col( data%prob%A%ne ) = data%prob%A%col( l )
             data%prob%A%val( data%prob%A%ne ) = data%prob%A%val( l )
           END IF
         END IF
       END DO

!  Now find the strict-interior feasible point

!      IF ( printi ) WRITE( out,                                               &
!        "( /, 2X, 20('-'), ' PHASE-1 for linear constraints ', 20('-') )" )
!      CALL WCP_initialize( data%WCP_data, control%WCP_control )
!      control%WCP_control%print_level = 1
!      control%WCP_control%mu_target = one
!      CALL WCP_solve( data%prob, data%WCP_data, control%WCP_control,          &
!                       inform%WCP_info )
!      CALL WCP_terminate( data%WCP_data, control%WCP_control,                 &
!                          inform%WCP_info )
!      WRITE( out, "( '    i    c_l    c     c_u' )" )
!      DO i = 1, m
!        WRITE( out, "( I6, 3ES12.4 )" )                                       &
!          i, data%C_l( i ), data%C( i ), data%C_u( i )
!      END DO
!      IF ( printi ) WRITE( out,                                               &
!      "( /, 2X, 17('-'), ' end of PHASE-1 for linear constraints ', 16('-'))")
!      STOP
     END IF

     IF ( use_primal_dual ) THEN
       WRITE( out, "( ' Primal-dual updates used ' )" )
     ELSE
       WRITE( out, "( ' Primal updates used ' )" )
     END IF

!  Evaluate the objective and general constraint function values
   
     CALL CUTEST_cfn( cutest_status, n, m, data%X, f, data%C )
     IF ( cutest_status /= 0 ) GO TO 930
     inform%obj = f ; inform%f_eval = inform%f_eval + 1
     g_recent = .FALSE.

!  If desired scale the variables and/or constraints     

     scale_xcf = control%scale_x > 0 .OR. control%scale_c > 0 .OR.             &
                 control%scale_f > 0
     IF ( scale_xcf ) THEN

!  If needed at this stage, compute the objective/constraint Jacobian

       IF ( control%scale_c > 2 .OR. control%scale_f > 2 ) THEN
         grlagf = .FALSE. ; J_len = J_ne
         CALL CUTEST_csgr( cutest_status,  n, m, data%X, data%LAMBDA, grlagf,  &
                           J_ne, J_len, data%prob%A%val, data%prob%A%col,      &
                           data%prob%A%row )
         IF ( cutest_status /= 0 ) GO TO 930
         inform%g_eval = inform%g_eval + 1
       END IF

!  Set up the data structures for the scalings, and assign default values

       CALL PTRANS_initialize( inform%ptrans_inform )
       CALL PTRANS_default( n, m, data%ptrans_transform, inform%ptrans_inform )
       IF ( inform%ptrans_inform%alloc_status /= 0 ) THEN
         inform%alloc_status = inform%ptrans_inform%alloc_status
         inform%bad_alloc = 'ptrans_transform' ; GO TO 910 ; END IF

!  Scale the variables

       IF ( control%scale_x > 1 ) THEN
         DO i = 1, n
            IF ( data%X_l( i ) < data%X_u( i ) ) THEN
              IF ( data%X_u( i ) < control%infinity ) THEN
                IF ( data%X_l( i ) > - control%infinity ) THEN
                  data%ptrans_transform%X_shift( i ) =                         &
                    half * ( data%X_u( i ) + data%X_l( i ) )
                  data%ptrans_transform%X_scale( i ) =                         &
                   half * ( data%X_u( i ) - data%X_l( i ) )
                ELSE
                  data%ptrans_transform%X_shift( i ) = data%X_u( i )
                  data%ptrans_transform%X_scale( i )                           &
                   = data%X_u( i ) - data%X( i )
                END IF
              ELSE IF ( data%X_l( i ) > - control%infinity ) THEN
                data%ptrans_transform%X_shift( i ) = data%X_l( i )
                data%ptrans_transform%X_scale( i ) = data%X( i ) - data%X_l( i )
              END IF
            END IF
         END DO

! Try this:
!        data%ptrans_transform%X_scale( 1 : n ) =                              &
!          100.0_wp * data%ptrans_transform%X_scale( 1 : n )

         IF ( printd ) THEN
           WRITE( out, "( '  shift_x ', /, ( 3ES22.14 ) )" )                   &
             data%ptrans_transform%X_shift( 1 : n )
           WRITE( out, "( '  scale_x ', /, ( 3ES22.14 ) )" )                   &
             data%ptrans_transform%X_scale( 1 : n )
         ELSE IF ( printi ) THEN
           WRITE( out, "( '  max shift_x ', /, ES22.14 )" )                    &
             MAXVAL( ABS( data%ptrans_transform%X_shift( 1 : n ) ) )
           WRITE( out, "( '  max scale_x ', /, ES22.14 )" )                    &
             MAXVAL( ABS( data%ptrans_transform%X_scale( 1 : n ) ) )
         END IF
       END IF


!  Compute the scaled infinity norms of the gradients of each function - store 
!  these in c_trial and f_trial

       IF ( control%scale_c > 2 .OR. control%scale_f > 2 ) THEN
         data%C_trial = one ; f_trial = one
         IF ( control%scale_x > 0 ) THEN
           DO l = 1, J_ne
             i = data%prob%A%row( l ) ; j = data%prob%A%col( l )
             IF ( i > 0 ) THEN
               data%C_trial( i ) = MAX( data%C_trial( i ),                     &
                 ABS( data%ptrans_transform%X_scale( j ) * data%prob%A%val( l)))
             ELSE
               f_trial = MAX( f_trial,                                         &
                 ABS( data%ptrans_transform%X_scale( j ) * data%prob%A%val( l)))
             END IF
           END DO
         ELSE
           DO l = 1, J_ne
             i = data%prob%A%row( l ) ; j = data%prob%A%col( l )
             IF ( i > 0 ) THEN
               data%C_trial( i ) = MAX( data%C_trial( i ),                     &
                                        ABS( data%prob%A%val( l ) ) )
             ELSE
               f_trial = MAX( f_trial, ABS( data%prob%A%val( l ) ) )
             END IF
           END DO
         END IF
       END IF

!  Scale the constraints

       IF ( control%scale_c > 1 ) THEN

!  Scale and shift so that shifts try to make c of O(1)

         IF ( control%scale_c == 2 ) THEN
           DO i = 1, m
             IF ( data%C_l( i ) < data%C_u( i ) ) THEN
               IF ( data%C_u( i ) < control%infinity ) THEN
                 IF ( data%C_l( i ) > - control%infinity ) THEN
                   data%ptrans_transform%C_shift( i ) =                        &
                     half * ( data%C_u( i ) + data%C_l( i ) )
                   data%ptrans_transform%C_scale( i ) =                        &
                     MAX( one, half * ( data%C_u( i ) - data%C_l( i ) ) )
                 ELSE
                   data%ptrans_transform%C_shift( i ) = data%C_u( i )
                   data%ptrans_transform%C_scale( i ) =                        &
                     MAX( one, ABS( data%C_u( i ) - data%C( i ) ) )
                 END IF
               ELSE IF ( data%C_l( i ) > - control%infinity ) THEN
                 data%ptrans_transform%C_shift( i ) = data%C_l( i )
                 data%ptrans_transform%C_scale( i ) =                          &
                   MAX( one, ABS( data%C( i ) - data%C_l( i ) ) )
               END IF
             END IF
           END DO

!  Scale and shift so that shifts try to make O(1) changes to x make O(1)
!  changes to c

         ELSE
           DO i = 1, m
             IF ( data%C_l( i ) < data%C_u( i ) ) THEN
               IF ( data%C_u( i ) < control%infinity ) THEN
                 IF ( data%C_l( i ) > - control%infinity ) THEN
                   data%ptrans_transform%C_shift( i ) =                        &
                     half * ( data%C_u( i ) + data%C_l( i ) )
                 ELSE
                   data%ptrans_transform%C_shift( i ) = data%C_u( i )
                 END IF
               ELSE IF ( data%C_l( i ) > - control%infinity ) THEN
                 data%ptrans_transform%C_shift( i ) = data%C_l( i )
               END IF
             ELSE
               data%ptrans_transform%C_shift( i ) = data%C_l( i )
             END IF
!            data%ptrans_transform%C_scale( i ) = one / data%C_trial( i )
             data%ptrans_transform%C_scale( i ) = data%C_trial( i )
           END DO
         END IF
         IF ( printd ) THEN
           WRITE( out, "( '  shift_c ', /, ( 3ES22.14 ) )" )                   &
             data%ptrans_transform%C_shift( 1 : m )
           WRITE( out, "( '  scale_c ', /, ( 3ES22.14 ) )" )                   &
             data%ptrans_transform%C_scale( 1 : m )
         ELSE IF ( printi ) THEN
           WRITE( out, "( '  max shift_c ', /, ES22.14 )" )                    &
             MAXVAL( ABS( data%ptrans_transform%C_shift( 1 : m ) ) )
           WRITE( out, "( '  max scale_c ', /, ES22.14 )" )                    &
             MAXVAL( ABS( data%ptrans_transform%C_scale( 1 : m ) ) )
         END IF
       END IF

!  Scale the objective

       IF ( control%scale_f > 1 ) THEN

!  Scale and shift so that shifts try to make f of O(1)

         IF ( control%scale_f == 2 ) THEN
           data%ptrans_transform%f_shift = f
           data%ptrans_transform%f_scale = one

!  Scale and shift so that shifts try to make O(1) changes to x make O(1)
!  changes to f

         ELSE
           data%ptrans_transform%f_shift = f
!          data%ptrans_transform%f_scale = one / f_trial
           data%ptrans_transform%f_scale = f_trial
         END IF
         IF ( printi ) THEN
           WRITE( out, "( '  shift_f ', /, ES22.14 )" )                        &
             data%ptrans_transform%f_shift
           WRITE( out, "( '  scale_f ', /, ES22.14 )" )                        &
             data%ptrans_transform%f_scale
         END IF
       END IF

!  Apply the scalings

       CALL PTRANS_trans( n, m, data%ptrans_transform, control%infinity, f = f,&
                          X = data%X, X_l = data%X_l, X_u = data%X_u,          &
                          C = data%C, C_l = data%C_l, C_u = data%C_u,          &
                          V_m = data%LAMBDA )

!  Scale the gradient if appropriate, and untangle A: 
!  separate the gradient terms from the constraint Jacobian

       IF ( control%scale_c > 2 .OR. control%scale_f > 2 ) THEN
         g_recent = .TRUE.
         data%prob%A%ne = 0 ; data%prob%G( : n ) = zero
         DO l = 1, J_ne
           i = data%prob%A%row( l ) ; j = data%prob%A%col( l )
           IF ( i > 0 ) THEN
             data%prob%A%val( l ) = (  data%ptrans_transform%X_scale( j ) /    &
               data%ptrans_transform%C_scale( i ) ) * data%prob%A%val( l )
             data%prob%A%ne = data%prob%A%ne + 1
             data%prob%A%row( data%prob%A%ne ) = data%prob%A%row( l )
             data%prob%A%col( data%prob%A%ne ) = data%prob%A%col( l )
             data%prob%A%val( data%prob%A%ne ) = data%prob%A%val( l )
!            write(6,"(2I8,ES12.4)") data%prob%A%row( data%prob%A%ne ),        &
!                                    data%prob%A%col( data%prob%A%ne ),        &
!                                    data%prob%A%val( data%prob%A%ne ) 
           ELSE
             data%prob%A%val( l ) = (  data%ptrans_transform%X_scale( j ) /    &
               data%ptrans_transform%f_scale ) * data%prob%A%val( l )
             data%prob%G( data%prob%A%col( l ) ) = data%prob%A%val( l )
           END IF
         END DO

!  Now reorder A so that it is stored by rows

         CALL SORT_reorder_by_rows( m, n, data%prob%A%ne,                      &
           data%prob%A%row, data%prob%A%col, J_ne, data%prob%A%val,            &
           data%prob%A%ptr, m + 1, data%IW, liw, control%error, control%out,   &
           inform_sort )

         IF ( inform_sort /= 0 ) THEN
           WRITE( control%error, "( ' on exit from SORT_reorder_by_rows,',     &
          &   ' inform = ', I8, '. Terminating ' )" ) inform_sort
           GO TO 990
         END IF
       END IF

     END IF

!  Assign the initial barrier and penalty parameters

     mu = control%initial_mu ; nu = control%initial_nu
     y_recent = .FALSE.

!  S_type is used to record which constraints have added elastic or
!  slack variables and how they have been added.

!  For equality constraints:
!   S_type =   0   -s <= c <= s with merit term nu * s
!          =   1   no elastic, c >= 0 with merit term nu * c
!          = - 1   no elastic, - c >= 0 with merit term - nu * c 
!          =   2   c + s >= 0 and s >= 0  with merit term nu * ( c + 2 s )
!          = - 2   s - c >= 0 and s >= 0  with merit term nu * ( 2 s - c )
!          =   4   linear constraint, no slack

!  For inequality constraints:
!   S_type =   1   no elastic (i.e., sufficiently feasible)
!          =   2   c - c_l + s >= 0, s + c_u - c >= 0 and s >= 0  
!                  with merit term nu * s
!          =   3   c - c_l + s >= 0, c_u - c >= 0 and s >= 0  
!                  with merit term nu * s (not implemented as yet)
!          =  -3   c - c_l >= 0, s + c_u - c >= 0 and s >= 0  
!                  with merit term nu * s (not implemented as yet)
!          =   4   linear constraint, slack

!  Ensure that the initial values of the elastic variables and
!  their multipliers are feasible

     DO i = 1, m
       data%Y_l( i ) = zero
       data%Y_u( i ) = zero
       IF ( data%C_l( i ) == data%C_u( i ) ) THEN
         IF ( data%LINEAR( i ) .AND. control%explicit_linear_constraints ) THEN
           data%S_type( i ) = 4
         ELSE IF ( control%elastic_type_equations == 0 ) THEN
           data%Y_l( i ) = dufeas + nu
           data%Y_u( i ) = dufeas + nu
           data%S_type( i ) = 0
         ELSE IF ( ABS( control%elastic_type_equations ) == 1 .OR.             &
                 ( ABS( control%elastic_type_equations ) == 3 .AND.            &
                   data%C( i ) >= data%C_l( i ) ) ) THEN
           data%Y_l( i ) = dufeas + nu
           IF ( ABS( control%elastic_type_equations ) == 3 .AND.               &
                data%C( i ) >= data%C_l( i ) + prfeas ) THEN
             data%S_type( i ) = 1
           ELSE
             data%S_type( i ) = 2
           END IF
         ELSE
           data%Y_u( i ) = dufeas + nu
           IF ( ABS( control%elastic_type_equations ) == 3 .AND.               &
                data%C( i ) <= data%C_u( i ) - prfeas ) THEN
             data%S_type( i ) = - 1
           ELSE
             data%S_type( i ) = - 2
           END IF
         END IF
       ELSE IF ( data%C_l( i ) < data%C_u( i ) ) THEN
         IF ( data%LINEAR( i ) .AND. control%explicit_linear_constraints ) THEN
           data%S_type( i ) = 4
         ELSE
           IF ( data%C_u( i ) <   control%infinity ) data%Y_u( i ) = dufeas + nu
           IF ( data%C_l( i ) > - control%infinity ) data%Y_l( i ) = dufeas + nu
           IF ( data%C_u( i ) <   control%infinity ) data%Y_u( i ) = dufeas + nu
           IF ( ABS( control%elastic_type_inequalities ) == 1 ) THEN
             IF ( data%C_l( i ) > - control%infinity ) THEN
               IF ( data%C_u( i ) < control%infinity ) THEN
                 IF ( data%C( i ) >= data%C_l( i ) + prfeas .AND.              &
                      data%C( i ) <= data%C_u( i ) - prfeas ) THEN
                   data%S_type( i ) = 1
                 ELSE
                   data%S_type( i ) = 2
                 END IF
               ELSE
  !              WRITE(6,*) data%C( i ), data%C_l( i ) + prfeas
                 IF ( data%C( i ) >= data%C_l( i ) + prfeas ) THEN
                   data%S_type( i ) = 1
                 ELSE
                   data%S_type( i ) = 2
                 END IF
               END IF
             ELSE IF ( data%C_u( i ) < control%infinity ) THEN
               IF ( data%C( i ) <= data%C_u( i ) - prfeas ) THEN
                 data%S_type( i ) = 1
               ELSE
                 data%S_type( i ) = 2
               END IF
             ELSE
               data%S_type( i ) = 2
             END IF
           ELSE
             data%S_type( i ) = 2
           END IF
         END IF
       ELSE
         inform%status = GALAHAD_error_bad_bounds ; GO TO 990
       END IF
     END DO
!    write( out, "( ' data%S_type', /, ( 20I3 ) )" ) data%S_type

!  Record the indices of constraints that have elastic variables

     nelastic = 0
!    s_lower = .FALSE.
     DO i = 1, m
       IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
!        IF ( data%S_type( i ) /= 0 ) s_lower = .TRUE.
         nelastic = nelastic + 1
         data%SSTATE( i ) = nelastic
       ELSE
         data%SSTATE( i ) = 0
       END IF
     END DO

     IF ( printi ) WRITE( out,                                                 &
       "( /, I7, ' out of ', I7, ' constraints have elastics ')" ) nelastic, m
     IF ( printi .AND. nelastic > 0 .AND. control%scale_s > 0 )                &
       WRITE( out, "( '  Initial elastic variables will be scaled ' )" )

!  And yet further allocations ... 

     nfrpel = nfree + nelastic
     IF ( control%bound_elastics ) THEN
       len_s_u = nelastic
     ELSE
       len_s_u = 0
     END IF

     array_name = 'superb: data%ELASTICS'
     CALL SPACE_resize_array( nelastic, data%ELASTICS, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%S'
     CALL SPACE_resize_array( nelastic, data%S, inform%status,                 &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%SCALE_S'
     CALL SPACE_resize_array( nelastic, data%SCALE_S, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!    IF ( s_lower ) THEN
       array_name = 'superb: data%U'
       CALL SPACE_resize_array( nelastic, data%U, inform%status,               &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910

       array_name = 'superb: data%U_P'
       CALL SPACE_resize_array( nelastic, data%U_P, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 910
!    END IF

     array_name = 'superb: data%S_u'
     CALL SPACE_resize_array( len_s_u, data%S_u, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%U_u'
     CALL SPACE_resize_array( len_s_u, data%U_u, inform%status,                &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%U_u_P'
     CALL SPACE_resize_array( len_s_u, data%U_u_P, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%GRAD_b'
     CALL SPACE_resize_array( nfrpel, data%GRAD_b, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     nelastic = 0
     DO i = 1, m
       IF ( data%SSTATE( i ) > 0 ) THEN
         nelastic = nelastic + 1
         data%ELASTICS( nelastic ) = i
         IF ( data%C_l( i ) == data%C_u( i ) ) THEN
           IF ( data%S_type( i ) == 0 ) THEN
             infeas = MAX( zero, data%C_l( i ) - data%C( i ),                  &
                           data%C( i ) - data%C_u( i ) )
           ELSE IF ( data%S_type( i ) == 2 ) THEN
             infeas = MAX( zero, data%C_l( i ) - data%C( i ) )
           ELSE IF ( data%S_type( i ) == - 2 ) THEN
             infeas = MAX( zero, data%C( i ) - data%C_u( i ) )
           END IF
         ELSE
           IF ( data%S_type( i ) >= 2 ) THEN
             infeas = MAX( zero, data%C_l( i ) - data%C( i ),                  &
                           data%C( i ) - data%C_u( i ) )
           END IF
         END IF
         IF ( control%scale_s > 0 ) THEN
           data%SCALE_S( nelastic ) = MAX( one, infeas )
         ELSE
           data%SCALE_S( nelastic ) = one
         END IF
         data%S( nelastic ) = prfeas + infeas / data%SCALE_S( nelastic )
!        data%S( nelastic ) = ( prfeas + infeas ) / data%SCALE_S( nelastic )
!        write(6,"( ' elastic ', ES12.4 )") data%S( nelastic )
         IF ( data%S_type( i ) /= 0 ) data%U( nelastic ) = dufeas
         IF ( control%bound_elastics ) THEN
           data%S_u( nelastic ) = MAX( max_elastic, two * data%S( nelastic ) )
           data%U_u( nelastic ) = dufeas
         END IF
       END IF
     END DO

     IF ( control%scale_s > 0 ) THEN
       IF ( printd ) THEN
         WRITE( out, "( '  scale_s ', /, ( 3ES22.14 ) )" )                     &
           data%SCALE_S( 1 : nelastic )
       ELSE IF ( printi ) THEN
         WRITE( out, "( '  max scale_s ', /, ES22.14 )" )                      &
           MAXVAL( ABS( data%SCALE_S( 1 : nelastic ) ) )
       END IF
     END IF

!  Compute the new elastic

     IF ( control%magical_path ) THEN
       CALL SUPERB_magical( m, nelastic, data%S_type, data%SSTATE, mu, nu,     &
                            data%C_l, data%C_u, data%C, data%S( : nelastic ),  &
                            data%SCALE_S( : nelastic ), data%S_u, len_s_u,     &
                            control, invalid )
       IF ( invalid ) GO TO 980
     END IF

!    write(out,"( 'x', /, (5ES12.4 ))" ) X
!    write(out,"( 'f', ES12.4)" ) f
!    write(out,"( 'c', /, (5ES12.4 ))" ) C
!    write(out,"( 'norm', L1 )" ) one_norm

     IF ( printd ) THEN
       WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  data%X( : n )
       IF ( nelastic > 0 )                                                     &
         WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )  data%S( : nelastic )
     END IF

!  Obtain an automatic value if the user so desires

     IF ( mu <= zero .OR. nu <= zero ) THEN

!  Evaluate both the gradients of the general constraint functions
!  Also obtain the gradient of either the objective function or
!  the Lagrangian function. The data is stored in a sparse format.

       IF ( .NOT. g_recent ) THEN
         grlagf = .FALSE. ; J_len = J_ne
         IF ( scale_xcf ) THEN
           CALL PTRANS_csgr( n, m, grlagf, m, data%LAMBDA, data%X, J_ne, J_len,&
                             data%prob%A%val, data%prob%A%col, data%prob%A%row,&
                             data%ptrans_transform, data%ptrans_data,          &
                             inform%ptrans_inform )
         ELSE
           CALL CUTEST_csgr( cutest_status, n, m, data%X, data%LAMBDA, grlagf, &
                             J_ne, J_len, data%prob%A%val, data%prob%A%col,    &
                             data%prob%A%row )
           IF ( cutest_status /= 0 ) GO TO 930
         END IF     
         inform%g_eval = inform%g_eval + 1
       END IF     

       IF ( printi ) WRITE( out, "( ' ' )" )
       IF ( mu <= zero ) THEN
         mu = MAXVAL( ABS( data%prob%A%val( : J_ne ) ) )
         IF ( printi ) WRITE( out, "( '  ** Initial barrier parameter',        &
        &                     ES11.4, ' chosen automatically ** ' )" ) mu
       END IF
       IF ( nu <= zero ) THEN
         nu = MAXVAL( ABS( data%prob%A%val( : J_ne ) ) )
         IF ( printi ) WRITE( out, "( '  ** Initial penalty parameter',        &
        &                     ES11.4, ' chosen automatically ** ' )" ) nu
       END IF

!  Untangle A: separate the gradient terms from the constraint Jacobian

       data%prob%A%ne = 0 ; data%prob%G( : n ) = zero
       DO i = 1, J_ne
         IF ( data%prob%A%row( i ) == 0 ) THEN
           data%prob%G( data%prob%A%col( i ) ) = data%prob%A%val( i )
         ELSE
           data%prob%A%ne = data%prob%A%ne + 1
           data%prob%A%row( data%prob%A%ne ) = data%prob%A%row( i )
           data%prob%A%col( data%prob%A%ne ) = data%prob%A%col( i )
           data%prob%A%val( data%prob%A%ne ) = data%prob%A%val( i )
!          write(6,"(2I8,ES12.4)")                                             &
!            data%prob%A%row( data%prob%A%ne ),                                &
!            data%prob%A%col( data%prob%A%ne ),                                &
!            data%prob%A%val( data%prob%A%ne ) 
         END IF
       END DO

!      WRITE( out, "( ' g ', /, ( 5ES12.4 ) )" )  data%prob%G( : n )
!      WRITE( out, "( ' g_max ', ES12.4 )" )  MAXVAL( ABS( data%prob%G ) )

!  Now reorder A so that it is stored by rows

       CALL SORT_reorder_by_rows( m, n, data%prob%A%ne, data%prob%A%row,       &
         data%prob%A%col, J_ne, data%prob%A%val, data%prob%A%ptr, m + 1,       &
         data%IW, liw, control%error, control%out, inform_sort )

       g_recent = .TRUE.

       IF ( .NOT. g_recent ) THEN
         write(6,"( '     mu          nu        KKT       KKT2 ' )" ) 
         kkt_best = point1 * huge( one )
         IF ( control%initial_mu < 0 ) THEN
           i_mu_end = 5 ; ELSE ; i_mu_end = 1 ; END IF
         DO i_mu = 1, i_mu_end
           SELECT CASE( i_mu )
             CASE( 1 ) ; mu_try = mu
             CASE( 2 ) ; mu_try = point1 * mu
             CASE( 3 ) ; mu_try = point01 * mu
             CASE( 4 ) ; mu_try = ten * mu
             CASE( 5 ) ; mu_try = hundred * mu
           END SELECT
           IF ( control%initial_nu < 0 ) THEN
             i_nu_end = 5 ; ELSE ; i_nu_end = 1 ; END IF
           DO i_nu = 1, i_nu_end
             SELECT CASE( i_nu )
               CASE( 1 ) ; nu_try = nu
               CASE( 2 ) ; nu_try = ten * nu
               CASE( 3 ) ; nu_try = hundred * nu
               CASE( 4 ) ; nu_try = point1 * nu
               CASE( 5 ) ; nu_try = point01 * nu
             END SELECT

!  Compute primal Lagrange multiplier estimates

             IF ( .NOT. y_recent ) THEN
               DO l = 1, nfree
                 i = data%XFREE( l )
                 IF ( data%X_l( i ) > - control%infinity )                     &
                   data%Z_l_P( i ) = mu_try / ( data%X( i ) - data%X_l( i ) )
                 IF ( data%X_u( i ) <   control%infinity )                     &
                   data%Z_u_P( i ) = mu_try / ( data%X_u( i ) - data%X( i ) )
               END DO

               DO i = 1, m
                 j = data%SSTATE( i )
                 IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
                   IF ( data%S_type( i ) /= 0 )                                &
                     data%U_P( j ) = mu_try / data%S( j )
                   IF ( control%bound_elastics )                               &
                     data%U_u_P( j ) = mu_try / ( data%S_u( j ) - data%S( j ) )
                 END IF
                 IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 0 )
                     data%Y_l_P( i ) = mu_try / ( data%C( i ) - data%C_l( i )  &
                        + data%SCALE_S( j ) * data%S( j ) )
                     data%Y_u_P( i ) = mu_try / ( data%C_u( i ) - data%C( i )  &
                        + data%SCALE_S( j ) * data%S( j ) )
                   CASE( 1 )
                     data%Y_l_P( i ) = mu_try / ( data%C( i ) - data%C_l( i ) )
                   CASE( - 1 )
                     data%Y_u_P( i ) = mu_try / ( data%C_u( i ) - data%C( i ) )
                   CASE( 2 )
                     data%Y_l_P( i ) = mu_try / ( data%C( i ) - data%C_l( i )  &
                       + data%SCALE_S( j ) * data%S( j ) )
                   CASE( - 2 )
                     data%Y_u_P( i ) = mu_try / ( data%C_u( i ) - data%C( i )  &
                       + data%SCALE_S( j ) * data%S( j ) )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 ELSE
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 1 )
                     IF ( data%C_l( i ) > - control%infinity )                 &
                       data%Y_l_P( i ) = mu_try / ( data%C( i ) - data%C_l( i ))
                     IF ( data%C_u( i ) <   control%infinity )                 &
                       data%Y_u_P( i ) = mu_try / ( data%C_u( i ) - data%C( i ))
                   CASE( 2 )
                     IF ( data%C_l( i ) > - control%infinity )                 &
                       data%Y_l_P( i ) = mu_try / ( data%C( i )                &
                         - data%C_l( i ) + data%SCALE_S( j ) * data%S( j ) )
                     IF ( data%C_u( i ) <   control%infinity )                 &
                       data%Y_u_P( i ) = mu_try / ( data%C_u( i ) - data%C( i )&
                         + data%SCALE_S( j ) * data%S( j ) )
!                  CASE( 3 )
!                  CASE( - 3 )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 END IF
               END DO
             END IF

!  Compute the gradient of the barrier function

!  wrt x

             data%GRAD_b( : nfree ) = data%prob%G( data%XFREE( : nfree ) )
             DO l = 1, nfree
               i = data%XFREE( l )
               IF ( data%X_l( i ) > - control%infinity )                       &
                 data%GRAD_b( l ) = data%GRAD_b( l ) - data%Z_l_P( i )
               IF ( data%X_u( i ) <   control%infinity )                       &
                 data%GRAD_b( l ) = data%GRAD_b( l ) + data%Z_u_P( i )
             END DO

!  subtract Jacobian (transpose) times multiplers

             DO i = 1, m
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                   g_term = data%Y_l_P( i ) -  data%Y_u_P( i )
                 CASE( 1, 2 )
                   g_term = data%Y_l_P( i ) - nu_try
                 CASE( - 2, - 1 )
                   g_term = - ( data%Y_u_P( i ) - nu_try )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               ELSE
                 g_term = zero
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 1, 2 )
                   IF ( data%C_l( i ) > - control%infinity )                   &
                     g_term = g_term + data%Y_l_P( i )
                   IF ( data%C_u( i ) <   control%infinity )                   &
                     g_term = g_term - data%Y_u_P( i )
!                CASE( 3 )
!                CASE( - 3 )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
!                data%GRAD_b( nfree + i ) = g_term
               END IF

               DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
                 j = data%XSTATE( data%prob%A%col( l ) )
                 IF ( j > 0 ) data%GRAD_b( j )                                 &
                   = data%GRAD_b( j ) - data%prob%A%val( l ) * g_term
               END DO
             END DO

!  wrt s

             IF ( nelastic > 0 ) THEN
               DO j = 1, nelastic
                 i = data%ELASTICS( j )
                 data%GRAD_b( nfree + j ) = data%SCALE_S( j ) * nu_try
                 IF ( control%bound_elastics ) data%GRAD_b( nfree + j ) =      &
                    data%GRAD_b( nfree + j ) + data%U_u_P( j )
                 IF ( data%S_type( i ) /= 0 ) data%GRAD_b( nfree + j ) =       &
                   data%GRAD_b( nfree + j ) - data%U_P( j )
                 IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 0 )
                     data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -     &
                       data%SCALE_S( j ) * ( data%Y_l_P( i ) + data%Y_u_P( i ) )
                   CASE( 2 )
                     data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -     &
                       data%SCALE_S( j ) * ( data%Y_l_P( i ) - nu_try )
                   CASE( - 2 )
                     data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -     &
                       data%SCALE_S( j ) * ( data%Y_u_P( i ) - nu_try )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 ELSE
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 2 )
                     IF ( data%C_l( i ) > - control%infinity )                 &
                       data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -   &
                         data%SCALE_S( j ) * data%Y_l_P( i )
                     IF ( data%C_u( i ) <   control%infinity )                 &
                       data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -   &
                         data%SCALE_S( j ) * data%Y_u_P( i )
!                  CASE( 3 )
!                  CASE( - 3 )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 END IF
               END DO
             END IF

             val = MAXVAL( ABS( data%GRAD_b( : nfrpel ) ) )
             IF ( val < kkt_best ) THEN
               write(6,"(' new champion ', 3ES12.4)") mu_try, nu_try, val
               mu_best = mu_try ; nu_best = nu_try
               kkt_best = val
             END IF             
             write(6,"( 3ES12.4 )" ) mu_try, nu_try, val
!            write(6,"( 4ES12.4 )" ) mu_try, nu_try,                           &
!              MAXVAL( ABS( data%GRAD_b( : nfree ) ) ),                        &
!              MAXVAL( ABS( data%GRAD_b( nfree + 1 : nfrpel ) ) )
           END DO
         END DO
         mu = mu_best ; nu = nu_best

       END IF
     END IF

!  Set convergence tolerances, theta_c and theta_d

     theta_d = MIN( control%max_stop_d,                                        &
                    MAX( control%stop_d_factor * mu ** beta,                   &
                         point99 * control%stop_d ) )                       
     theta_c = MIN( control%max_stop_c,                                        &
                    MAX( control%stop_c_factor * mu ** beta,                   &
                         point99 * control%stop_c ) )                       

!  Compute the value of the merit function

     merit = SUPERB_merit( n, m, nfree, nelastic, data%XFREE, data%S_type,     &
               data%SSTATE, f, mu, nu, data%X, data%X_l, data%X_u, data%C,     &
               data%C_l, data%C_u, data%S, data%SCALE_S, data%S_u, len_s_u,    &
               inform%pr_feas, barrier, penalty, merit_error, print_level,     &
               out, control )
     IF ( merit_error == - 99 ) GO TO 980

!    write(out,"( 'merit, violation', 2ES12.4 )" ) merit, inform%pr_feas

     IF ( printi ) WRITE( out, "( /, ' initial objective value ', 16X, ' = ',  &
     & ES16.8, /, ' primal accuracy required                 = ', ES12.4,      &
     &         /, ' dual   accuracy required                 = ', ES12.4,      &
     &         /, ' complementarity required                 = ', ES12.4 )" )  &
         inform%obj, control%stop_p, control%stop_d, control%stop_c

     IF ( printi ) WRITE( control%out, 2050 ) mu, nu, theta_d, theta_c

!  If a non-monotone method is to be used, initialize counters

     IF ( nmhist > 0 ) THEN
       l_suc = 0
       merit_min = merit ; merit_ref = merit_min ; merit_current = merit_min
       sigma_r = zero ; sigma_c = zero
     END IF

!  Allocate sufficient space to hold the problem

     array_name = 'superb: data%GRAD_m'
     CALL SPACE_resize_array( nfrpel, data%GRAD_m, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%DV'
     CALL SPACE_resize_array( nfrpel + m, data%DV, inform%status,              &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%VECTOR'
     CALL SPACE_resize_array( nfrpel + m, data%VECTOR, inform%status,          &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%SOL'
     CALL SPACE_resize_array( nfrpel + m, data%SOL, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%RES'
     CALL SPACE_resize_array( nfrpel + m, data%RES, inform%status,             &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%BEST'
     CALL SPACE_resize_array( nfrpel + m, data%BEST, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%S_trial'
     CALL SPACE_resize_array( nelastic, data%S_trial, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%B_X'
     CALL SPACE_resize_array( nfree, data%B_X, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%B_C'
     CALL SPACE_resize_array( m, data%B_C, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%B_CM'
     CALL SPACE_resize_array( m, data%B_CM, inform%status,                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

!    IF ( control%bound_elastics .OR. s_lower ) THEN
     array_name = 'superb: data%B_S'
     CALL SPACE_resize_array( nelastic, data%B_S, inform%status,               &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910
!    END IF

     array_name = 'superb: data%prob%H%ptr'
     CALL SPACE_resize_array( n + 1, data%prob%H%ptr, inform%status,           &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%C_stat'
     CALL SPACE_resize_array( m, data%C_stat, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 910

     array_name = 'superb: data%B_stat'
     CALL SPACE_resize_array( n, data%B_stat, inform%status,                   &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     data%prob%m = m
     data%prob%n = n

!    data%prob%Z = one
!    data%prob%Y = Y

!  Set control parameters

     new_derivatives = .TRUE. ; refact = .FALSE.
     new_inner = .TRUE.
     got_ratio = .FALSE.
     step_max = half * infinity
     analyse = .TRUE.
!    violation = inform%pr_feas
     old_mu = mu
     itref_max = control%itref_max
!    theta_p = MAX( initial_target_min,                                        &
     theta_p = MAX( mu,                                                        &
       inform%pr_feas * SQRT( MIN( point1, control%barrier_decrease_factor ) ) )
     ratio = - one
     old_radius = initial_radius
     cg_iter = 0 ; nbacts = 0 ; mo = ' '
     zeta = zeta_tol
     n_pr_feas_increase = 0

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      O U T E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     DO

!  Intialize GLTR data

       control%gltr_control%stop_relative = control%inner_stop_relative
       control%gltr_control%stop_absolute = control%inner_stop_absolute
       control%gltr_control%fraction_opt = control%inner_fraction_opt
       control%gltr_control%unitm = .FALSE.
       control%gltr_control%steihaug_toint = .FALSE.
!      control%gltr_control%steihaug_toint = .TRUE.
       control%gltr_control%out = control%out
       control%gltr_control%print_level = print_level - 1
       control%gltr_control%itmax = control%cg_maxit
       control%gltr_control%lanczos_itmax = control%lanczos_itmax

       inform%radius = initial_radius

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      I N N E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

inner: DO
!        write(6,*) ' new_inner ', new_inner
         stop_inner_status = - 1
         pr_feas_old = inform%pr_feas
!        WRITE(6,*) ' data%Y_u = ', data%Y_u( 1 )

!  If required, print the distances to the constraint boundaries
 
         IF ( printd ) THEN
           DO i = 1, m
             j = data%SSTATE( i )
             IF ( data%C_l( i ) == data%C_u( i ) ) THEN
               SELECT CASE ( data%S_type( i ) )
               CASE( 0 )
                 data%C_feas( i ) =                                            &
                   MIN( data%C( i ) - data%C_l( i ) +                          &
                          data%SCALE_S( j ) * data%S( j ),                     &
                        data%C_u( i ) - data%C( i ) +                          &
                          data%SCALE_S( j ) * data%S( j ) )
               CASE( 1 )
                 data%C_feas( i ) = data%C( i ) - data%C_l( i )
               CASE( - 1 )
                 data%C_feas( i ) = data%C_u( i ) - data%C( i )
               CASE( 2 )
                 data%C_feas( i ) = data%C( i ) - data%C_l( i ) +              &
                   data%SCALE_S( j ) * data%S( j )
               CASE( - 2 )
                 data%C_feas( i ) = data%C_u( i ) - data%C( i ) +              &
                   data%SCALE_S( j ) * data%S( j )
               END SELECT
             ELSE
               SELECT CASE ( data%S_type( i ) )
               CASE( 1 )
                 IF ( data%C_l( i ) > - infinity ) THEN
                   IF ( data%C_u( i ) <   infinity ) THEN
                     data%C_feas( i ) = MIN( data%C( i ) - data%C_l( i ),      &
                                        data%C_u( i ) - data%C( i ) )
                   ELSE
                     data%C_feas( i ) = data%C( i ) - data%C_l( i )
                   END IF
                 ELSE
                   data%C_feas( i ) = data%C_u( i ) - data%C( i )
                 END IF
               CASE( 2 )
                 IF ( data%C_l( i ) > - infinity ) THEN
                   IF ( data%C_u( i ) <   infinity ) THEN
                     data%C_feas( i ) =                                        &
                       MIN( data%C( i ) - data%C_l( i ) +                      &
                              data%SCALE_S( j ) * data%S( j ),                 &
                            data%C_u( i ) - data%C( i ) +                      &
                              data%SCALE_S( j ) * data%S( j ) )
                   ELSE
                     data%C_feas( i ) = data%C( i ) - data%C_l( i ) +          &
                       data%SCALE_S( j ) * data%S( j )
                   END IF
                 ELSE
                   data%C_feas( i ) = data%C_u( i ) - data%C( i ) +            &
                     data%SCALE_S( j ) * data%S( j )
                 END IF
               END SELECT
             END IF
           END DO
           WRITE( out, "( ' c ', /, ( 5ES12.4 ) )" )  data%C_feas( : m )
!          IF ( nelastic > 0 ) WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )       &
!            data%S( : nelastic )
         END IF

!  ----------------------------------------------------------------------------
!                       COMPUTE DERIVATIVE VALUES
!  ----------------------------------------------------------------------------

!        WRITE(6,"('Y_u',ES12.4)") data%Y_u( 1 )
!        write(6,*) ' sucessful ', new_derivatives
!        WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  data%X( : n )
         IF ( new_derivatives .OR. refact ) THEN
           IF ( refact ) GO TO 50

!  Compute primal Lagrange multiplier estimates

           IF ( .NOT. y_recent ) THEN
             DO l = 1, nfree
               i = data%XFREE( l )
               IF ( data%X_l( i ) > - control%infinity )                       &
                 data%Z_l_P( i ) = mu / ( data%X( i ) - data%X_l( i ) )
               IF ( data%X_u( i ) <   control%infinity )                       &
                 data%Z_u_P( i ) = mu / ( data%X_u( i ) - data%X( i ) )
             END DO

             DO i = 1, m
               j = data%SSTATE( i )
               IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
                 IF ( data%S_type( i ) /= 0 ) data%U_P( j ) = mu / data%S( j )
                 IF ( control%bound_elastics )                                 &
                   data%U_u_P( j ) = mu / ( data%S_u( j ) - data%S( j ) )
               END IF
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                   data%Y_l_P( i ) = mu / ( data%C( i ) - data%C_l( i ) +      &
                     data%SCALE_S( j ) * data%S( j ))
                   data%Y_u_P( i ) = mu / ( data%C_u( i ) - data%C( i ) +      &
                     data%SCALE_S( j ) * data%S( j ))
                 CASE( 1 )
                   data%Y_l_P( i ) = mu / ( data%C( i ) - data%C_l( i ) )
                 CASE( - 1 )
                   data%Y_u_P( i ) = mu / ( data%C_u( i ) - data%C( i ) )
                 CASE( 2 )
                   data%Y_l_P( i ) = mu / ( data%C( i ) - data%C_l( i ) +      &
                     data%SCALE_S( j ) * data%S( j ))
                 CASE( - 2 )
                   data%Y_u_P( i ) = mu / ( data%C_u( i ) - data%C( i ) +      &
                     data%SCALE_S( j ) * data%S( j ))
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               ELSE
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 1 )
                   IF ( data%C_l( i ) > - control%infinity )                   &
                     data%Y_l_P( i ) = mu / ( data%C( i ) - data%C_l( i ) )
                   IF ( data%C_u( i ) <   control%infinity )                   &
                     data%Y_u_P( i ) = mu / ( data%C_u( i ) - data%C( i ) )
                 CASE( 2 )
                   IF ( data%C_l( i ) > - control%infinity )                   &
                     data%Y_l_P( i ) = mu / ( data%C( i ) - data%C_l( i ) +    &
                       data%SCALE_S( j ) * data%S( j ) )
                   IF ( data%C_u( i ) <   control%infinity )                   &
                     data%Y_u_P( i ) = mu / ( data%C_u( i ) - data%C( i ) +    &
                       data%SCALE_S( j ) * data%S( j ) )
!                CASE( 3 )
!                CASE( - 3 )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               END IF
             END DO
             IF ( printd ) THEN
               CALL SUPERB_write_y( out, m, .TRUE. , ' y_l_p ', data%S_type,   &
                 data%C_l, data%C_u, data%Y_l_P, data%Y_u_P, control, invalid )
               IF ( invalid ) GO TO 980
               CALL SUPERB_write_y( out, m, .FALSE., ' y_u_p ', data%S_type,   &
                 data%C_l, data%C_u, data%Y_l_P, data%Y_u_P, control, invalid )
               IF ( invalid ) GO TO 980
               CALL SUPERB_write_z( out, n, nfree, .TRUE. , ' z_l ',data%XFREE,&
                 data%X_l, data%X_u, data%Z_l_P, data%Z_u_P, control )
               CALL SUPERB_write_z( out, n, nfree, .FALSE., ' z_u ',data%XFREE,&
                 data%X_l, data%X_u, data%Z_l_P, data%Z_u_P, control )
             END IF
           END IF

!  If required, use primal multiplier estimates

           IF ( .NOT. use_primal_dual .AND. .NOT. new_inner ) THEN
             DO l = 1, nfree
               i = data%XFREE( l )
               IF ( data%X_l( i ) > - control%infinity )                       &
                 data%Z_l( i ) = data%Z_l_P( i )
               IF ( data%X_u( i ) <   control%infinity )                       &
                 data%Z_u( i ) = data%Z_u_P( i )
             END DO

             DO i = 1, m
               j = data%SSTATE( i )
               IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
                 IF ( data%S_type( i ) /= 0 ) data%U( j ) = data%U_P( j )
                 IF ( control%bound_elastics ) data%U_u( j ) = data%U_u_P( j )
               END IF
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                   data%Y_l( i ) = data%Y_l_P( i )
                   data%Y_u( i ) = data%Y_u_P( i )
                 CASE( 1, 2 )
                   data%Y_l( i ) = data%Y_l_P( i )
                 CASE( - 1, - 2 )
                   data%Y_u( i ) = data%Y_u_P( i )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               ELSE
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 1, 2 )
                   IF ( data%C_l( i ) > - control%infinity )                   &
                     data%Y_l( i ) = data%Y_l_P( i )
                   IF ( data%C_u( i ) <   control%infinity )                   &
                     data%Y_u( i ) = data%Y_u_P( i )
!                CASE( 3 )
!                CASE( - 3 )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               END IF
             END DO
           END IF
           IF ( printd ) THEN
             CALL SUPERB_write_y( out, m, .TRUE. , ' y_l ', data%S_type,       &
               data%C_l, data%C_u, data%Y_l, data%Y_u, control, invalid )
             IF ( invalid ) GO TO 980
             CALL SUPERB_write_y( out, m, .FALSE., ' y_u ', data%S_type,       &
               data%C_l, data%C_u, data%Y_l, data%Y_u, control, invalid )
             IF ( invalid ) GO TO 980
             CALL SUPERB_write_z( out, n, nfree, .TRUE. , ' z_l ', data%XFREE, &
               data%X_l, data%X_u, data%Z_l, data%Z_u, control )
             CALL SUPERB_write_z( out, n, nfree, .FALSE., ' z_u ', data%XFREE, &
               data%X_l, data%X_u, data%Z_l, data%Z_u, control )
           END IF

!  Compute the value of the current violation of (scaled) complementarity

!          comp_slack_primal = SUPERB_comp_slack( n, m, nfree, nelastic,       &
!            data%XFREE, data%S_type, data%SSTATE, mu, data%X, data%X_l,       &
!            data%X_u, data%C, data%C_l, data%C_u, data%S, data%SCALE_S,       &
!            data%S_u, len_s_u, data%Z_l_P, data%Z_u_P, data%U_P,              &
!            data%U_u_P, data%Y_l_P, data%Y_u_P, control, invalid )
!          IF ( invalid ) GO TO 980

           comp_slack_primal = zero
           IF ( use_primal_dual ) THEN
             inform%comp_slack = SUPERB_comp_slack( n, m, nfree, nelastic,     &
               data%XFREE, data%S_type, data%SSTATE, mu, data%X, data%X_l,     &
               data%X_u, data%C, data%C_l, data%C_u, data%S, data%SCALE_S,     &
               data%S_u, len_s_u, data%Z_l, data%Z_u, data%U, data%U_u,        &
               data%Y_l, data%Y_u, control, invalid )
             IF ( invalid ) GO TO 980
           ELSE
             inform%comp_slack = comp_slack_primal
           END IF

!  Compute the Lagrange multipliers

           DO i = 1, m
             IF ( data%C_l( i ) == data%C_u( i ) ) THEN
               SELECT CASE ( data%S_type( i ) )
               CASE( 0 )
                 data%LAMBDA( i ) = - ( data%Y_l( i ) - data%Y_u( i ) )
               CASE( 1, 2 )
                 data%LAMBDA( i ) = - ( data%Y_l( i ) - nu )
               CASE( - 1, - 2 )
                 data%LAMBDA( i ) = data%Y_u( i ) - nu
               CASE DEFAULT
                 GO TO 980
               END SELECT
             ELSE
               g_term = zero
               SELECT CASE ( data%S_type( i ) )
               CASE( 1, 2 )
                 IF ( data%C_l( i ) > - control%infinity )                     &
                   g_term = g_term + data%Y_l( i )
                 IF ( data%C_u( i ) <   control%infinity )                     &
                   g_term = g_term - data%Y_u( i )
!              CASE( 3 )
!              CASE( - 3 )
               CASE DEFAULT
                 GO TO 980
               END SELECT
               data%LAMBDA( i ) = - g_term
             END IF
           END DO
           IF ( printd )                                                       &
             WRITE( out, "( ' lambda ', /, ( 5ES12.4 ) )" ) data%LAMBDA( : m )

!          WRITE( out, "( ' complementary slackness ',ES12.4)")                &
!            inform%comp_slack

!  Check to see if there are potentially degenerate variables and constraints.
!  If so, reduce their multipliers accordingly

           IF ( degeneracy_check .AND. new_inner .AND. mu < 0.001_wp ) THEN
!          IF ( degeneracy_check .AND. new_inner .AND. mu <= point1 ) THEN
             val = SQRT( mu / old_mu )
             delta = old_mu ** 0.33
!            delta = old_mu ** 0.4
!            delta = old_mu ** 0.125

             blank = .TRUE.
             degen = 0
             DO l = 1, nfree
               i = data%XFREE( l )
               IF ( data%X_l( i ) > - control%infinity ) THEN
                 IF ( data%X( i ) - data%X_l( i ) <= delta .AND.               &
                      data%Z_l( i ) <= delta ) THEN
!                  WRITE( 6, * ) ' degen xl ', i
                   data%Z_l( i ) = val * data%Z_l( i )
                   degen = degen + 1
                 END IF
               END IF
               IF ( data%X_u( i ) <   control%infinity ) THEN
                 IF ( data%X_u( i ) - data%X( i ) <= delta .AND.               &
                      data%Z_u( i ) <= delta ) THEN
!                  WRITE( 6, * ) ' degen xu ', i
                   data%Z_u( i ) = val * data%Z_u( i )
                   degen = degen + 1
                 END IF
               END IF
             END DO
             IF ( degen > 0 .AND. printi ) THEN
               IF ( blank ) THEN 
                 WRITE( out, "( ' ' )" ) ; blank = .FALSE. ; END IF
               WRITE( out, "( I7, ' degenerate variable', A1 )" )              &
                 degen, STRING_pleural( degen )
             END IF

             degen = 0
             DO i = 1, m
               j = data%SSTATE( i )
               IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
                 IF ( data%S_type( i ) /= 0 ) THEN
                    IF ( data%S( j ) <= delta .AND. data%U( j ) <= delta ) THEN
!                    WRITE( 6, * ) ' degen s  ', j
!                    WRITE( 6, * ) data%S( j ), data%U( j ), delta
                     data%U( j ) = val * data%U( j )
                     degen = degen + 1
                   END IF
                 END IF
                 IF ( control%bound_elastics ) THEN
                   IF ( data%S_u( j ) - data%S( j ) <= delta .AND.             &
                        data%U_u( j ) <= delta ) THEN
 !                   WRITE( 6, * ) ' degen s_u  ', j
                     data%U_u( j ) = val * data%U_u( j )
                     degen = degen + 1
                   END IF
                 END IF
               END IF
             END DO
             IF ( degen > 0 .AND. printi ) THEN
               IF ( blank ) THEN 
                 WRITE( out, "( ' ' )" ) ; blank = .FALSE. ; END IF
               WRITE( out, "( I7, ' degenerate elastic', A1 )" )               &
                 degen, STRING_pleural( degen )
             END IF

             degen = 0
             DO i = 1, m
               j = data%SSTATE( i )
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                   IF ( data%C( i ) - data%C_l( i ) + data%SCALE_S( j ) *      &
                     data%S( j ) <= delta .AND. data%Y_l( i ) <= delta ) THEN
!                    WRITE( 6, * ) ' degen ce ', i
                     data%Y_l( i ) = val * data%Y_l( i ) 
                     degen = degen + 1
                   END IF
                   IF ( data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) *      &
                      data%S( j ) <= delta .AND. data%Y_u( i ) <= delta ) THEN
!                    WRITE( 6, * ) ' degen ce ', i
                     data%Y_u( i ) = val * data%Y_u( i ) 
                     degen = degen + 1
                   END IF
                 CASE( 1 )
                   IF ( data%C( i ) - data%C_l( i ) <= delta .AND.             &
                        data%Y_l( i ) <= delta ) THEN
!                    WRITE( 6, * ) ' degen ce ', i
                     data%Y_l( i ) = val * data%Y_l( i ) 
                     degen = degen + 1
                   END IF
                 CASE( - 1 )
                   IF ( data%C_u( i ) - data%C( i ) <= delta .AND.             &
                        data%Y_u( i ) <= delta ) THEN
!                    WRITE( 6, * ) ' degen ce ', i
                     data%Y_u( i ) = val * data%Y_u( i ) 
                     degen = degen + 1
                   END IF
                 CASE( 2 )
                   IF ( data%C( i ) - data%C_l( i ) + data%SCALE_S( j ) *      &
                     data%S( j ) <= delta .AND. data%Y_l( i ) <= delta ) THEN
!                    WRITE( 6, * ) ' degen ce ', i
                     data%Y_l( i ) = val * data%Y_l( i ) 
                     degen = degen + 1
                   END IF
                 CASE( - 2 )
                   IF ( data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) *      &
                     data%S( j ) <= delta .AND. data%Y_u( i ) <= delta ) THEN
!                    WRITE( 6, * ) ' degen ce ', i
                     data%Y_u( i ) = val * data%Y_u( i ) 
                     degen = degen + 1
                   END IF
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               ELSE
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 1 )
                   IF ( data%C_l( i ) > - control%infinity ) THEN
                     IF ( data%C( i ) - data%C_l( i ) <= delta .AND.           &
                          data%Y_l( i ) <= delta ) THEN
!                      WRITE( 6, * ) ' degen cl ', i
                       data%Y_l( i ) = val * data%Y_l( i ) 
                       degen = degen + 1
                     END IF
                   END IF
                   IF ( data%C_u( i ) <   control%infinity ) THEN
                     IF ( data%C_u( i ) - data%C( i ) <= delta .AND.           &
                          data%Y_u( i ) <= delta ) THEN
!                      WRITE( 6, * ) ' degen cu ', i
                       data%Y_u( i ) = val * data%Y_u( i ) 
                       degen = degen + 1
                     END IF
                   END IF
                 CASE( 2 )
                   IF ( data%C_l( i ) > - control%infinity ) THEN
                     IF ( data%C( i ) - data%C_l( i ) + data%SCALE_S( j ) *    &
                       data%S( j ) <= delta .AND. data%Y_l( i ) <= delta ) THEN
!                      WRITE( 6, * ) ' degen cl ', i
                       data%Y_l( i ) = val * data%Y_l( i ) 
                       degen = degen + 1
                     END IF
                   END IF
                   IF ( data%C_u( i ) <   control%infinity ) THEN
                     IF ( data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) *    &
                       data%S( j ) <= delta .AND. data%Y_u( i ) <= delta ) THEN
!                      WRITE( 6, * ) ' degen cu ', i
                       data%Y_u( i ) = val * data%Y_u( i ) 
                       degen = degen + 1
                     END IF
                   END IF
!                CASE( 3 )
!                CASE( - 3 )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               END IF
             END DO
             IF ( degen > 0 .AND. printi ) THEN
               IF ( blank ) THEN 
                 WRITE( out, "( ' ' )" ) ; blank = .FALSE. ; END IF
               WRITE( out, "( I7, ' degenerate constraint', A1 )" )            &
                 degen, STRING_pleural( degen )
             END IF
!            IF ( printi .AND. .NOT. blank ) WRITE( out, "( ' ' )" )
           END IF 

!          WRITE( out, "( ' y_norm ', ES12.4 )" )                              &
!            MAX( MAXVAL( ABS( Y ) ), MAXVAL( ABS( data%U ) ) )
!          WRITE( out, "( ' y ', /, ( 5ES12.4 ) )" )  data%LAMBDA( : m )
!          write(6,"( ' primal, primal dual = ', 2ES12.4 )" )                  &
!            data%Y_l_P( 1 ), data%Y_l( 1 )

!          WRITE( out, "( ' y_l_p ', /, ( 5ES12.4 ) )" )  data%Y_l_P( : m )
!          WRITE( out, "( ' y_u_p ', /, ( 5ES12.4 ) )" )  data%Y_u_P( : m )

!  Evaluate both the gradients of the general constraint functions
!  and the Hessian matrix of the Lagrangian function for the problem.
!  The Hessian is stored as a sparse matrix in "co-ordinate" format. 
!  Also obtain the gradient of either the objective function or
!  the Lagrangian function. The data is stored in a sparse format.

!          IF ( inform%iter == 0 .OR. .NOT. control%exact_linesearch ) THEN
           IF ( .NOT. g_recent ) THEN
             IF ( control%model == 1 ) THEN
               grlagf = .FALSE. ; J_len = J_ne ; H_len = H_ne
               IF ( scale_xcf ) THEN
                 CALL PTRANS_csgrsh( n, m, data%X, grlagf, m, data%LAMBDA,     &
                                     J_ne, J_len,                              &
                                     data%prob%A%val, data%prob%A%col,         &
                                     data%prob%A%row, data%prob%H%ne,          &
                                     H_len, data%prob%H%val, data%prob%H%row,  &
                                     data%prob%H%col, data%ptrans_transform,   &
                                     data%ptrans_data, inform%ptrans_inform )
               ELSE
                 CALL CUTEST_csgrsh( cutest_status, n, m, data%X, data%LAMBDA, &
                                     grlagf, J_ne, J_len,                      &
                                     data%prob%A%val, data%prob%A%col,         &
                                     data%prob%A%row, data%prob%H%ne,          &
                                     H_len, data%prob%H%val, data%prob%H%row,  &
                                     data%prob%H%col )
               END IF     
               IF ( cutest_status /= 0 ) GO TO 930
             ELSE
               grlagf = .FALSE. ; J_len = J_ne
               IF ( scale_xcf ) THEN
                 CALL PTRANS_csgr( n, m, grlagf, m, data%LAMBDA, data%X,       &
                                   J_ne, J_len,                                &
                                   data%prob%A%val, data%prob%A%col,           &
                                   data%prob%A%row, data%ptrans_transform,     &
                                   data%ptrans_data, inform%ptrans_inform )
               ELSE
                 CALL CUTEST_csgr( cutest_status, n, m, data%X, data%LAMBDA,   &
                                   grlagf, J_ne, J_len, data%prob%A%val,       &
                                   data%prob%A%col, data%prob%A%row )
                 IF ( cutest_status /= 0 ) GO TO 930
               END IF     
               IF ( control%model == 2 ) THEN
                 data%prob%H%ne = 0 
               ELSE
                 data%prob%H%ne = n
                 DO i = 1, n
                   data%prob%H%row( i ) = i ; data%prob%H%col( i ) = i
                   data%prob%H%val( i ) = h_diag
                 END DO
               END IF
             END IF
             inform%g_eval = inform%g_eval + 1

!  Untangle A: separate the gradient terms from the constraint Jacobian

             data%prob%A%ne = 0 ; data%prob%G( : n ) = zero
             DO i = 1, J_ne
               IF ( data%prob%A%row( i ) == 0 ) THEN
                 data%prob%G( data%prob%A%col( i ) ) = data%prob%A%val( i )
               ELSE
                 data%prob%A%ne = data%prob%A%ne + 1
                 data%prob%A%row( data%prob%A%ne ) = data%prob%A%row( i )
                 data%prob%A%col( data%prob%A%ne ) = data%prob%A%col( i )
                 data%prob%A%val( data%prob%A%ne ) = data%prob%A%val( i )
!                write(6,"(2I8,ES12.4)") data%prob%A%row( data%prob%A%ne ),    &
!                                        data%prob%A%col( data%prob%A%ne ),    &
!                                        data%prob%A%val( data%prob%A%ne ) 
               END IF
             END DO

!            WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  data%X( : n )
!            WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )  data%S( : m )
!            WRITE( out, "( ' g ', /, ( 5ES12.4 ) )" )  data%prob%G( : n )
!            WRITE( out, "( ' g_max ', ES12.4 )" )  MAXVAL( ABS( data%prob%G ) )

!  Now reorder A so that it is stored by rows

             CALL SORT_reorder_by_rows( m, n, data%prob%A%ne,                  &
               data%prob%A%row, data%prob%A%col, J_ne, data%prob%A%val,        &
               data%prob%A%ptr, m + 1, data%IW, liw, control%error,            &
               control%out, inform_sort )

             IF ( inform_sort /= 0 ) THEN
               WRITE( control%error, "( ' on exit from SORT_reorder_by_rows,', &
              &   ' inform = ', I8, '. Terminating ' )" ) inform_sort
               GO TO 990
             END IF
           ELSE

!  If the gradient has recently been computed, only obtain the Hessian

             IF ( control%model == 1 ) THEN
               H_len = H_ne
               IF ( scale_xcf ) THEN
                 CALL PTRANS_csh( n, m, data%X, m, data%LAMBDA,                &
                                  data%prob%H%ne, H_len,                       &
                                  data%prob%H%val, data%prob%H%row,            &
                                  data%prob%H%col,                             &
                                  data%ptrans_transform, data%ptrans_data,     &
                                  inform%ptrans_inform )
               ELSE
                 CALL CUTEST_csh( cutest_status, n, m, data%X, data%LAMBDA,    &
                                  data%prob%H%ne, H_len, data%prob%H%val,      &
                                  data%prob%H%row, data%prob%H%col )
                 IF ( cutest_status /= 0 ) GO TO 930
               END IF     
             ELSE IF ( control%model == 2 ) THEN
               data%prob%H%ne = 0
             ELSE
               data%prob%H%ne = n
               DO i = 1, n
                 data%prob%H%row( i ) = i ; data%prob%H%col( i ) = i
                 data%prob%H%val( i ) = h_diag
               END DO
             END IF
           END IF

!  Ensure that only entries from lower triangle of H are given

           data%VECTOR( : nfree ) = zero
           DO l = 1, data%prob%H%ne
             i = data%prob%H%row( l ) ; j = data%prob%H%col( l )
             IF ( i < j ) THEN
               data%prob%H%row( l ) = j ; data%prob%H%col( l ) = i
             END IF
             i = data%XSTATE( i ) ; j = data%XSTATE( j )
             IF ( i > 0 .AND. j > 0 ) THEN
               val = data%prob%H%val( l )
               data%VECTOR( i ) = data%VECTOR( i ) + ABS( val )
               IF ( i /= j ) data%VECTOR( j ) = data%VECTOR( j ) + ABS( val )
             END IF
           END DO
           h_norm = MAXVAL( data%VECTOR( : nfree ) ) + one

!  Find the largest components of A and H

!          IF ( data%prob%A%ne > 0 ) THEN
!            a_max = MAXVAL( ABS( data%prob%A%val( : data%prob%A%ne ) ) )
!          ELSE
!            a_max = zero
!          END IF

!          IF ( data%prob%H%ne > 0 ) THEN
!            h_max = MAX( h_min,                                               &
!                         MAXVAL( ABS( data%prob%H%val( : data%prob%H%ne ) ) ) )
!          ELSE
!            h_max = h_min
!          END IF

!          WRITE( out, "( ' a_max, h_max ', /, 2ES12.4 )" )  a_max, h_max

!  Compute diagonal barrier Hessian terms

           DO l = 1, nfree
             i = data%XFREE( l )
             b_term = zero
             IF ( data%X_l( i ) > - control%infinity )                         &
               b_term = b_term + data%Z_l( i ) / ( data%X( i ) - data%X_l( i ) )
             IF ( data%X_u( i ) <   control%infinity )                         &
               b_term = b_term + data%Z_u( i ) / ( data%X_u( i ) - data%X( i ) )
             data%B_X( l ) = b_term
           END DO
           IF ( printd2 )                                                      &
             WRITE( out, "( ' B_x ', /, ( 5ES12.4 ) )" )  data%B_X( : nfree )

           DO i = 1, m
             j = data%SSTATE( i )
             IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
               IF ( data%S_type( i ) /= 0 .AND. control%bound_elastics ) THEN
                 data%B_S( j ) = data%U( j ) / data%S( j ) + data%U_u( j ) /   &
                   ( data%S_u( j ) - data%S( j ) )
               ELSE IF ( data%S_type( i ) /= 0 ) THEN
                 data%B_S( j ) = data%U( j ) / data%S( j )
               ELSE IF ( control%bound_elastics ) THEN
                 data%B_S( j ) = data%U_u( j ) / ( data%S_u( j ) - data%S( j ) )
               ELSE
                 data%B_S( j ) = zero
               END IF
             END IF
             IF ( data%C_l( i ) == data%C_u( i ) ) THEN
               SELECT CASE ( data%S_type( i ) )
               CASE( 0 )
                 b_cl = data%Y_l( i ) / ( data%C( i ) - data%C_l( i ) +        &
                   data%SCALE_S( j ) * data%S( j ) )
                 b_cu = data%Y_u( i ) / ( data%C_u( i ) - data%C( i ) +        &
                   data%SCALE_S( j ) * data%S( j ) )
                 data%B_C( i ) = b_cl + b_cu ; data%B_CM( i ) = b_cl - b_cu
               CASE( 1 )
                 data%B_C( i ) = data%Y_l( i ) / ( data%C( i ) - data%C_l( i ) )
                 data%B_CM( i ) = data%B_C( i )
               CASE( - 1 )
                 data%B_C( i ) = data%Y_u( i ) / ( data%C_u( i ) - data%C( i ) )
                 data%B_CM( i ) = - data%B_C( i )
               CASE( 2 )
                 data%B_C( i ) = data%Y_l( i ) / ( data%C( i ) -               &
                   data%C_l( i ) + data%SCALE_S( j ) * data%S( j ) )
                 data%B_CM( i ) = data%B_C( i )
               CASE( - 2 )
                 data%B_C( i ) =                                               &
                   data%Y_u( i ) / ( data%C_u( i ) - data%C( i ) +             &
                     data%SCALE_S( j ) * data%S( j ) )
                 data%B_CM( i ) = - data%B_C( i )
               CASE DEFAULT
                 GO TO 980
               END SELECT
             ELSE
               SELECT CASE ( data%S_type( i ) )
               CASE( 1 )
                 b_cl = zero ; b_cu = zero
                 IF ( data%C_l( i ) > - control%infinity )                     &
                   b_cl = data%Y_l( i ) / ( data%C( i ) - data%C_l( i ) )
                 IF ( data%C_u( i ) <   control%infinity )                     &
                   b_cu = data%Y_u( i ) / ( data%C_u( i ) - data%C( i ) )
                 data%B_C( i ) = b_cl + b_cu ; data%B_CM( i ) = b_cl - b_cu
               CASE( 2 )
                 b_cl = zero ; b_cu = zero
                 IF ( data%C_l( i ) > - control%infinity ) b_cl =              &
                   data%Y_l( i ) / ( data%C( i ) - data%C_l( i ) +             &
                     data%SCALE_S( j ) * data%S( j ) )
                 IF ( data%C_u( i ) <   control%infinity )                     &
                   b_cu = data%Y_u( i ) / ( data%C_u( i ) - data%C( i ) +      &
                     data%SCALE_S( j ) * data%S( j ) )
                 data%B_C( i ) = b_cl + b_cu ; data%B_CM( i ) = b_cl - b_cu
!              CASE( 3 )
!              CASE( - 3 )
               CASE DEFAULT
                 GO TO 980
               END SELECT
             END IF
           END DO
           IF ( printd2 .AND. nelastic > 0 )                                   &
             WRITE( out, "( ' data%B_S ', /, ( 5ES12.4 ) )" )                  &
               data%B_S( : nelastic )
           IF ( printd2 ) THEN
             WRITE( out, "( ' data%B_C ', /, ( 5ES12.4 ) )" )  data%B_C( : m )
             WRITE( out, "( ' data%B_CM ', /, ( 5ES12.4 ) )" )  data%B_CM( : m )
           END IF
         
           IF ( use_primal_dual ) THEN

!  Compute the gradient of the Lagrangian function

!  wrt x

             data%GRAD_m( : nfree ) = data%prob%G( data%XFREE( : nfree ) )
             DO l = 1, nfree
               i = data%XFREE( l )
               IF ( data%X_l( i ) > - control%infinity )                       &
                 data%GRAD_m( l ) = data%GRAD_m( l ) - data%Z_l( i )
               IF ( data%X_u( i ) <   control%infinity )                       &
                 data%GRAD_m( l ) = data%GRAD_m( l ) + data%Z_u( i )
             END DO
!            WRITE(6,"( 'z', /, ( 6ES12.4) )" )                                &
!              data%GRAD_m( : n ) - data%prob%G(:n)

!  subtract Jacobian (transpose) times multiplers

             DO i = 1, m
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                   g_term = data%Y_l( i ) - data%Y_u( i )
                 CASE( 1, 2 )
                   g_term = data%Y_l( i ) - nu
                 CASE( -1, - 2 )
                   g_term = - ( data%Y_u( i ) - nu ) 
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               ELSE
                 g_term = zero
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 1, 2 )
                   IF ( data%C_l( i ) > - control%infinity )                   &
                     g_term = g_term + data%Y_l( i )
                   IF ( data%C_u( i ) <   control%infinity )                   &
                     g_term = g_term - data%Y_u( i )
!                CASE( 3 )
!                CASE( - 3 )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               END IF

               DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
                 j = data%XSTATE( data%prob%A%col( l ) )
                 IF ( j > 0 ) data%GRAD_m( j ) = data%GRAD_m( j ) -            &
                   data%prob%A%val( l ) * g_term
               END DO
             END DO

!  wrt s

             IF ( nelastic > 0 ) THEN
               DO j = 1, nelastic
                 i = data%ELASTICS( j )
                 data%GRAD_m( nfree + j ) = data%SCALE_S( j ) * nu
                 IF ( control%bound_elastics ) data%GRAD_m( nfree + j ) =      &
                    data%GRAD_m( nfree + j ) + data%U_u( j )
                 IF ( data%S_type( i ) /= 0 ) data%GRAD_m( nfree + j ) =       &
                    data%GRAD_m( nfree + j ) - data%U( j )
                 IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 0 )
                     data%GRAD_m( nfree + j ) = data%GRAD_m( nfree + j ) -     &
                       data%SCALE_S( j ) * ( data%Y_l( i ) + data%Y_u( i ) )
                   CASE( 2 )
                     data%GRAD_m( nfree + j ) = data%GRAD_m( nfree + j ) -     &
                       data%SCALE_S( j ) * ( data%Y_l( i ) - nu )
                   CASE( - 2 )
                     data%GRAD_m( nfree + j ) = data%GRAD_m( nfree + j ) -     &
                       data%SCALE_S( j ) * ( data%Y_u( i ) - nu )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 ELSE
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 2 )
                     IF ( data%C_l( i ) > - control%infinity )                 &
                       data%GRAD_m( nfree + j ) = data%GRAD_m( nfree + j ) -   &
                         data%SCALE_S( j ) * data%Y_l( i )
                     IF ( data%C_u( i ) <   control%infinity )                 &
                       data%GRAD_m( nfree + j ) = data%GRAD_m( nfree + j ) -   &
                         data%SCALE_S( j ) * data%Y_u( i )
!                  CASE( 3 )
!                  CASE( - 3 )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 END IF
               END DO
             END IF

!  Check for termination using the gradient of the Lagrangian 

             grad_x = MAXVAL( ABS( data%GRAD_m( : nfree ) ) )
             grad_l = MAXVAL( ABS( data%GRAD_m( : nfrpel ) ) )
             IF ( grad_l <= theta_d .AND. inform%pr_feas <= theta_p            &
                 .AND. inform%comp_slack <= theta_c .AND. .NOT. new_inner ) THEN
!                .AND. inform%comp_slack <= theta_c ) THEN
               inform%du_feas = grad_l
               stop_inner_status = 0 ; EXIT
             END IF
             IF ( printt )                                                     &
               WRITE( out,"( ' gradient Lagrangian = ', ES12.4 )" ) grad_l
!            WRITE( out, "( ' g_b ', /, ( 5ES12.4 ) )" ) data%GRAD_m( : nfree )
           END IF

!  Compute the gradient of the barrier function

!  wrt x

           data%GRAD_b( : nfree ) = data%prob%G( data%XFREE( : nfree ) )
           DO l = 1, nfree
             i = data%XFREE( l )
             IF ( data%X_l( i ) > - control%infinity )                         &
               data%GRAD_b( l ) = data%GRAD_b( l ) - data%Z_l_P( i )
             IF ( data%X_u( i ) <   control%infinity )                         &
               data%GRAD_b( l ) = data%GRAD_b( l ) + data%Z_u_P( i )
           END DO

!  subtract Jacobian (transpose) times multiplers

!          J_norm = zero
!          data%VECTOR( : nfree ) = zero
           DO i = 1, m
!            j = data%SSTATE( i )
             IF ( data%C_l( i ) == data%C_u( i ) ) THEN
               SELECT CASE ( data%S_type( i ) )
               CASE( 0 )
                 g_term = data%Y_l_P( i ) - data%Y_u_P( i )
               CASE( 1, 2 )
                 g_term = data%Y_l_P( i ) - nu
               CASE( - 1, - 2 )
                 g_term = - ( data%Y_u_P( i ) - nu )
               CASE DEFAULT
                 GO TO 980
               END SELECT
             ELSE
               g_term = zero
               SELECT CASE ( data%S_type( i ) )
               CASE( 1, 2 )
                 IF ( data%C_l( i ) > - control%infinity )                     &
                   g_term = g_term + data%Y_l_P( i )
                 IF ( data%C_u( i ) <   control%infinity )                     &
                   g_term = g_term - data%Y_u_P( i )
!              CASE( 3 )
!              CASE( - 3 )
               CASE DEFAULT
                 GO TO 980
               END SELECT
!              IF ( j > 0 ) data%GRAD_b( nfree + j ) = g_term
             END IF

!            a_norm = zero
             DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
               j = data%XSTATE( data%prob%A%col( l ) )
               IF ( j > 0 ) data%GRAD_b( j ) = data%GRAD_b( j ) -              &
                 data%prob%A%val( l ) * g_term
!              a_norm = A_norm + ABS( data%prob%A%val( l ) )
!              data%VECTOR( j ) = data%VECTOR( j ) + ABS( data%prob%A%val( l ) )
             END DO
!            J_norm = MAX( J_norm, a_norm )
           END DO
!          WRITE( out, "( ' y_norm_est ', ES12.4 )" )                          &
!            MAXVAL( ABS( data%prob%G( : n ) ) ) / MAXVAL( data%VECTOR( :nfree))
!          WRITE( out, "( ' g_norm, J_norm ', 3ES12.4 )" )                     &
!       MAXVAL( ABS( data%prob%G( : n ) ) ), J_norm, MAXVAL(data%VECTOR(:nfree))

!  wrt s

           IF ( nelastic > 0 ) THEN
             DO j = 1, nelastic
               i = data%ELASTICS( j )
               data%GRAD_b( nfree + j ) = data%SCALE_S( j ) * nu
               IF ( control%bound_elastics ) data%GRAD_b( nfree + j ) =        &
                 data%GRAD_b( nfree + j ) + data%U_u_P( j )
               IF ( data%S_type( i ) /= 0 ) data%GRAD_b( nfree + j ) =         &
                 data%GRAD_b( nfree + j ) - data%U_P( j )
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                   data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -       &
                     data%SCALE_S( j ) * ( data%Y_l_P( i ) + data%Y_u_P( i ) )
                 CASE( 2 )
                   data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -       &
                     data%SCALE_S( j ) * ( data%Y_l_P( i ) - nu )
                 CASE( - 2 )
                   data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -       &
                     data%SCALE_S( j ) * ( data%Y_u_P( i ) - nu )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               ELSE
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 2 )
                   IF ( data%C_l( i ) > - control%infinity )                   &
                     data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -     &
                       data%SCALE_S( j ) * data%Y_l_P( i )
                   IF ( data%C_u( i ) <   control%infinity )                   &
                     data%GRAD_b( nfree + j ) = data%GRAD_b( nfree + j ) -     &
                       data%SCALE_S( j ) * data%Y_u_P( i )
!                CASE( 3 )
!                CASE( - 3 )
                 CASE DEFAULT
                   GO TO 980
                 END SELECT
               END IF
             END DO
           END IF

!          WRITE(6,"( ' gradx, grad_s ', 2ES12.4 )" )                          &
!            MAXVAL( ABS( data%GRAD_b( : nfree ) ) ),                          &
!            MAXVAL( ABS( data%GRAD_b( nfree + 1 : nfrpel ) ) )
           inform%du_feas = MAXVAL( ABS( data%GRAD_b( : nfrpel ) ) )

!  Check for termination

           IF ( inform%du_feas <= theta_d .AND.                                &
                comp_slack_primal <= theta_c .AND. .NOT. new_inner ) THEN
!               comp_slack_primal <= theta_c ) THEN

!  If primal-dual multipliers are being used, reset these to the terminating
!  primal ones

             IF ( use_primal_dual ) THEN
               inform%comp_slack = comp_slack_primal
               DO l = 1, nfree
                 i = data%XFREE( l )
                 IF ( data%X_l( i ) > - control%infinity )                     &
                   data%Z_l( i ) = data%Z_l_P( i )
                 IF ( data%X_u( i ) <   control%infinity )                     &
                   data%Z_u( i ) = data%Z_u_P( i )
               END DO
               DO i = 1, m
                 j = data%SSTATE( i )
                 IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
                   IF ( data%S_type( i ) /= 0 ) data%U( j ) = data%U_P( j )
                   IF ( control%bound_elastics ) data%U_u( j ) = data%U_u_P( j )
                 END IF
                 IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 0 )
                     data%Y_l( i ) = data%Y_l_P( i )
                     data%Y_u( i ) = data%Y_u_P( i )
                   CASE( 1, 2 )
                     data%Y_l( i ) = data%Y_l_P( i )
                   CASE( - 1, - 2 )
                     data%Y_u( i ) = data%Y_u_P( i )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 ELSE
                   SELECT CASE ( data%S_type( i ) )
                   CASE( 1, 2 )
                     IF ( data%C_l( i ) > - control%infinity )                 &
                       data%Y_l( i ) = data%Y_l_P( i )
                     IF ( data%C_u( i ) <   control%infinity )                 &
                       data%Y_u( i ) = data%Y_u_P( i )
!                  CASE( 3 )
!                  CASE( - 3 )
                   CASE DEFAULT
                     GO TO 980
                   END SELECT
                 END IF
               END DO
             END IF
             IF ( printd ) THEN
               CALL SUPERB_write_y( out, m, .TRUE. , ' y_l_p ', data%S_type,   &
                 data%C_l, data%C_u, data%Y_l_P, data%Y_u_P, control, invalid )
               IF ( invalid ) GO TO 980
               CALL SUPERB_write_y( out, m, .FALSE., ' y_u_p ', data%S_type,   &
                 data%C_l, data%C_u, data%Y_l_P, data%Y_u_P, control, invalid )
               IF ( invalid ) GO TO 980
               CALL SUPERB_write_z( out, n, nfree, .TRUE. , ' z_l ',data%XFREE,&
                 data%X_l, data%X_u, data%Z_l_P, data%Z_u_P, control)
               CALL SUPERB_write_z( out, n, nfree, .FALSE., ' z_u ',data%XFREE,&
                  data%X_l, data%X_u, data%Z_l_P, data%Z_u_P, control )
             END IF
             stop_inner_status = 1 ; EXIT
           END IF
           
           IF ( printd2 ) WRITE( out,                                          &
                "( ' Grb ', /, ( 5ES12.4 ) )" )  data%GRAD_B( : nfrpel )

   40      CONTINUE

!  Compute a suitable preconditioner

           IF ( analyse ) THEN
             IF ( printw ) WRITE( out,                                         &
                  "( ' .............. analysis phase ................ ' )" )

!  ----------------------------------------------------------------------------
!                           ANALYSIS PHASE
!  ----------------------------------------------------------------------------

!            A_ne = data%prob%A%ne
             A_ne = COUNT( data%XSTATE( data%prob%A%col( : data%prob%A%ne ))> 0)

             Hfree_ne =                                                        &
               COUNT( data%XSTATE( data%prob%H%row(:data%prob%H%ne )) > 0 .AND.&
                      data%XSTATE( data%prob%H%col(:data%prob%H%ne ) ) > 0 )

!  Set up the data structures for the matrix

!         ( P + B_X            J^T   )
!     K = (          K_22    K_23^T  ),
!         (   J      K_23     K_33   )

!  or for the eliminated matrix

!   K_1 = ( P + B_X              J^T            )
!         (   J     K_33 - K_32 K_22^-1 K_23^T  ),
             
!  where P is a specified "preconditioner" for H,
!  K_22 = B_S + Theta ( B_C - B_CM B_C^-1 B_CM ) Theta,
!  K_23 = B_C^-1 B_CM Theta,
!  K_33 = -B_C^-1, and
!  Theta = diag(data%SCALE_S)

             IF ( printi ) THEN
               WRITE( out, "( ' ' )" )
               SELECT CASE( precon )
                 CASE( 1 ) ; WRITE( out, "( '  Identity Hessian ' )" )
                 CASE( 2 ) ; WRITE( out, "( '  Full Hessian ' )" )
                 CASE( 3 ) ; WRITE( out, "( '  Band (semi-bandwidth ', I3,     &
                            &               ') Hessian ' )" ) nsemib
                 CASE( 4 ) ; WRITE( out, "( '  Barrier Hessian ' )" ) 
               END SELECT
               WRITE( out, "( '  Augmented system method used ' )" )
             END IF

!  Compute the space required

!  ... for the eliminated matrix

             IF ( control%eliminate_elastics ) THEN
               data%K%n = nfree + m

               SELECT CASE( precon )
               CASE ( 1 )
                 lk = A_ne + data%K%n + nfree
               CASE ( 2 )
                 lk = A_ne + data%K%n + Hfree_ne
               CASE ( 3 )
                 lk = A_ne + data%K%n +                                        &
                   COUNT( data%XSTATE( data%prob%H%row(:data%prob%H%ne ) ) > 0 &
                   .AND. data%XSTATE( data%prob%H%col(:data%prob%H%ne ) ) > 0  &
                   .AND. ABS( data%XSTATE( data%prob%H%row(:data%prob%H%ne)) - &
                              data%XSTATE( data%prob%H%col(:data%prob%H%ne)))  &
                                <= nsemib )
               CASE DEFAULT
                 lk = A_ne + data%K%n
               END SELECT

!  ... for the uneliminated matrix

             ELSE
               data%K%n = nfrpel + m

               SELECT CASE( precon )
               CASE ( 1 )
                 lk = A_ne + nelastic + data%K%n + nfree
               CASE ( 2 )
                 lk = A_ne + nelastic + data%K%n + Hfree_ne
               CASE ( 3 )
                 lk = A_ne + nelastic + data%K%n +                             &
                   COUNT( data%XSTATE( data%prob%H%row(:data%prob%H%ne ) ) > 0 &
                   .AND. data%XSTATE( data%prob%H%col(:data%prob%H%ne ) ) > 0  &
                   .AND. ABS( data%XSTATE( data%prob%H%row(:data%prob%H%ne) ) -&
                              data%XSTATE( data%prob%H%col(:data%prob%H%ne) ) )&
                                <= nsemib )
               CASE DEFAULT
                 lk = A_ne + nelastic + data%K%n
               END SELECT
             END IF

!  Allocate the arrays for the analysis phase

             array_name = 'superb: data%K%row'
             CALL SPACE_resize_array( lk, data%K%row, inform%status,           &
                    inform%alloc_status, array_name = array_name,              &
                    deallocate_error_fatal = control%deallocate_error_fatal,   &
                    exact_size = control%space_critical,                       &
                    bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 910
        
             array_name = 'superb: data%K%col'
             CALL SPACE_resize_array( lk, data%K%col, inform%status,           &
                    inform%alloc_status, array_name = array_name,              &
                    deallocate_error_fatal = control%deallocate_error_fatal,   &
                    exact_size = control%space_critical,                       &
                    bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 910
        
             array_name = 'superb: data%K%val'
             CALL SPACE_resize_array( lk, data%K%val, inform%status,           &
                    inform%alloc_status, array_name = array_name,              &
                    deallocate_error_fatal = control%deallocate_error_fatal,   &
                    exact_size = control%space_critical,                       &
                    bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 910
        
!  Set the row and column indices

!  ... for the eliminated matrix

             IF ( control%eliminate_elastics ) THEN

!  Set the coordinates and value of 2,1 block, A, in K
             
               A_ne = 0
               DO i = 1, m
                 ii = nfree + i
                 DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
                   j = data%XSTATE( data%prob%A%col( l ) )
                   IF ( j > 0 ) THEN
                     A_ne = A_ne + 1
                     data%K%row( A_ne ) = ii
                     data%K%col( A_ne ) = j
                   END IF
                 END DO
               END DO

!  Set the coordinates and values of M in K

               nnzk = A_ne 
               SELECT CASE( precon )

!  * P is a diagonal matrix

               CASE ( 1 )
                 DO i = 1, nfree
                   nnzk = nnzk + 1 
                   data%K%row( nnzk ) = i ; data%K%col( nnzk ) = i
                   data%K%val( nnzk ) = k_diag
                 END DO

!  * P is the Hessian matrix

               CASE ( 2 )
                 nnzhs = nnzk
                 DO l = 1, data%prob%H%ne
                   i = data%XSTATE( data%prob%H%row( l ) )
                   j = data%XSTATE( data%prob%H%col( l ) )
                   IF ( i > 0 .AND. j > 0 ) THEN
                     nnzk = nnzk + 1
                     data%K%row( nnzk ) = i ; data%K%col( nnzk ) = j
                   END IF
                 END DO

!  * P is a band from the Hessian matrix

               CASE ( 3 )
                 nnzhs = nnzk
                 DO l = 1, data%prob%H%ne
                   i = data%XSTATE( data%prob%H%row( l ) )
                   j = data%XSTATE( data%prob%H%col( l ) )
                   IF ( i > 0 .AND. j > 0 .AND. ABS( j - i ) <= nsemib ) THEN
                     nnzk = nnzk + 1 
                     data%K%row( nnzk ) = i ; data%K%col( nnzk ) = j
                   END IF
                 END DO
   
!  * P is just the barrier terms

               CASE DEFAULT

               END SELECT

               nnzks = nnzk

!  Finally, include the co-ordinates of the barrier terms

               DO i = 1, data%K%n
                 data%K%row( nnzks + i ) = i ; data%K%col( nnzks + i ) = i
               END DO
               data%K%ne = nnzks + data%K%n

             ELSE

!  ... for the uneliminated matrix

!  Set the coordinates and value of 3,1 block, A, in K
             
               A_ne = 0
               DO i = 1, m
                 ii = nfrpel + i
                 DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
                   j = data%XSTATE( data%prob%A%col( l ) )
                   IF ( j > 0 ) THEN
                     A_ne = A_ne + 1
                     data%K%row( A_ne ) = ii
                     data%K%col( A_ne ) = j
                   END IF
                 END DO
               END DO

!  Set the coodinates corresponding to 3,2 block in K

               DO j = 1, nelastic
                 i = data%ELASTICS( j )
                 data%K%row( A_ne + j ) = nfrpel + i
                 data%K%col( A_ne + j ) = nfree + j
               END DO

!  Set the coordinates and values of M in K

               nnzk = A_ne + nelastic
               SELECT CASE( precon )

!  * P is a diagonal matrix

               CASE ( 1 )
                 DO i = 1, nfree
                   nnzk = nnzk + 1 
                   data%K%row( nnzk ) = i ; data%K%col( nnzk ) = i
                   data%K%val( nnzk ) = k_diag
                 END DO

!  * P is the Hessian matrix

               CASE ( 2 )
                 nnzhs = nnzk
                 DO l = 1, data%prob%H%ne
                   i = data%XSTATE( data%prob%H%row( l ) )
                   j = data%XSTATE( data%prob%H%col( l ) )
                   IF ( i > 0 .AND. j > 0 ) THEN
                     nnzk = nnzk + 1
                     data%K%row( nnzk ) = i ; data%K%col( nnzk ) = j
                   END IF
                 END DO

!  * P is a band from the Hessian matrix

               CASE ( 3 )
                 nnzhs = nnzk
                 DO l = 1, data%prob%H%ne
                   i = data%XSTATE( data%prob%H%row( l ) )
                   j = data%XSTATE( data%prob%H%col( l ) )
                   IF ( i > 0 .AND. j > 0 .AND. ABS( j - i ) <= nsemib ) THEN
                     nnzk = nnzk + 1 
                     data%K%row( nnzk ) = i ; data%K%col( nnzk ) = j
                   END IF
                 END DO
   
!  * P is just the barrier terms

               CASE DEFAULT

               END SELECT

               nnzks = nnzk

!  Finally, include the co-ordinates of the barrier terms

               DO i = 1, data%K%n
                 data%K%row( nnzks + i ) = i ; data%K%col( nnzks + i ) = i
               END DO
               data%K%ne = nnzks + data%K%n

             END IF

!  Analyse the sparsity pattern of the preconditioner

             CALL SILS_analyse( data%K, data%FACTORS, data%CNTL, AINFO )

!  Record the storage requested

             inform%factorization_integer = AINFO%nirnec 
             inform%factorization_real = AINFO%nrlnec

             data%CNTL%liw = MAX( 2 * inform%factorization_integer,            &
                                  control%indmin )
             data%CNTL%la  = MAX( 2 * inform%factorization_real,               &
                                  control%valmin )

!  Check for error returns

             inform%factorization_status = AINFO%flag
             IF ( AINFO%flag < 0 ) THEN
               IF ( printe ) WRITE( control%error, 2100 ) AINFO%flag 
               inform%status = GALAHAD_error_analysis ; GO TO 990
             ELSE IF ( AINFO%flag > 0 ) THEN 
               IF ( printt ) WRITE( out, 2060 ) AINFO%flag, 'SILS_analyse'
             END IF
        
             IF ( printt ) WRITE( out,                                         &
               "( ' real/integer space required for factors ', 2I10 )" )       &
                 AINFO%nrladu, AINFO%niradu

!  Analysis complete

             analyse = .FALSE.
           END IF 

!  ----------------------------------------------------------------------------
!                           FACTORIZATION PHASE
!  ----------------------------------------------------------------------------

!  Put remaining (changable) numerical values in K

!  ... for the eliminated matrix

           IF ( control%eliminate_elastics ) THEN

!  Contributions from J:
             
             A_ne = 0
             DO i = 1, m
               DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
                 j = data%XSTATE( data%prob%A%col( l ) )
                 IF ( j > 0 ) THEN
                   A_ne = A_ne + 1
                   data%K%val( A_ne ) = data%prob%A%val( l )
                 END IF
               END DO
             END DO

!  Contributions from H to P:

             SELECT CASE( precon )

!  * P is the Hessian matrix

             CASE ( 2 )
               nnzk = nnzhs
               DO l = 1, data%prob%H%ne
                 IF ( data%XSTATE( data%prob%H%row( l ) ) > 0 .AND.            &
                      data%XSTATE( data%prob%H%col( l ) ) > 0 ) THEN
                   nnzk = nnzk + 1 ; data%K%val( nnzk ) = data%prob%H%val( l )
                 END IF
               END DO

!  * P is a band from the Hessian matrix

             CASE ( 3 )
               nnzk = nnzhs
               DO l = 1, data%prob%H%ne
                 i = data%XSTATE( data%prob%H%row( l ) )
                 j = data%XSTATE( data%prob%H%col( l ) )
                 IF ( i > 0 .AND. j > 0 .AND. ABS( j - i ) <= nsemib ) THEN
                   nnzk = nnzk + 1  ; data%K%val( nnzk ) = data%prob%H%val( l ) 
                 END IF
               END DO
             END SELECT

!  Contributions from the barrier terms

!  From B_X

             nnzk = nnzks
             DO l = 1, nfree
!              i = data%XFREE( l ) ; 
               data%K%val( nnzk + l ) = data%B_X( l )
             END DO

!  From B_C

             nnzk = nnzk + nfree
             DO i = 1, m
               j = data%SSTATE( i )
               data%K%val( nnzk + i ) = - one / data%B_C( i )
               IF ( j > 0 ) data%K%val( nnzk + i ) = data%K%val( nnzk + i ) -  &
                 ( data%B_CM( i ) * data%SCALE_S( j ) / data%B_C( i ) ) ** 2 / &
                  ( data%B_S( j ) + ( data%B_C( i ) - data%B_CM( i ) ** 2 /    &
                    data%B_C( i ) ) * data%SCALE_S( j ) ** 2 )
             END DO

           ELSE

!  ... for the uneliminated matrix

!  Contributions from J:
             
             A_ne = 0
             DO i = 1, m
               DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
                 j = data%XSTATE( data%prob%A%col( l ) )
                 IF ( j > 0 ) THEN
                   A_ne = A_ne + 1
                   data%K%val( A_ne ) = data%prob%A%val( l )
                 END IF
               END DO
             END DO

!  Set the coodinates corresponding to 3,2 block in K

             DO j = 1, nelastic
               i = data%ELASTICS( j ) 
               data%K%val( A_ne + j ) =                                        &
                 data%B_CM( i ) * data%SCALE_S( j ) / data%B_C( i )
             END DO

!  Contributions from H to P:

             SELECT CASE( precon )

!  * P is the Hessian matrix

             CASE ( 2 )
               nnzk = nnzhs
               DO l = 1, data%prob%H%ne
                 IF ( data%XSTATE( data%prob%H%row( l ) ) > 0 .AND.            &
                      data%XSTATE( data%prob%H%col( l ) ) > 0 ) THEN
                   nnzk = nnzk + 1 ; data%K%val( nnzk ) = data%prob%H%val( l )
                 END IF
               END DO

!  * P is a band from the Hessian matrix

             CASE ( 3 )
               nnzk = nnzhs
               DO l = 1, data%prob%H%ne
                 i = data%XSTATE( data%prob%H%row( l ) )
                 j = data%XSTATE( data%prob%H%col( l ) )
                 IF ( i > 0 .AND. j > 0 .AND. ABS( j - i ) <= nsemib ) THEN
                   nnzk = nnzk + 1  ; data%K%val( nnzk ) = data%prob%H%val( l ) 
                 END IF
               END DO
             END SELECT

!  Contributions from the barrier terms

!  From B_X

             nnzk = nnzks
             DO l = 1, nfree
!              i = data%XFREE( l ) ; 
               data%K%val( nnzk + l ) = data%B_X( l )
             END DO

!  From B_S

             nnzk = nnzk + nfree
             DO j = 1, nelastic
               i = data%ELASTICS( j )
               data%K%val( nnzk + j ) = data%B_S( j ) + ( data%B_C( i ) -      &
                 data%B_CM( i ) ** 2 / data%B_C( i ) ) * data%SCALE_S( j ) ** 2
             END DO

!  From B_C

             nnzk = nnzk + nelastic
             DO i = 1, m
               data%K%val( nnzk + i ) = - one / data%B_C( i )
             END DO

           END IF

!  Factorize K

           IF ( control%print_matrix ) THEN
             WRITE( 6, "( ' n, nnz = ', 2I6, ' values ' )" ) data%K%n, data%K%ne
             WRITE( 6, "( 2 ( 2I6, ES24.16 ) )" ) ( data%K%row( i ),           &
               data%K%col( i ), data%K%val( i ), i = 1, data%K%ne ) 
           END IF

           IF ( printd4 ) THEN
             DO i = 1, data%K%ne
               WRITE( out, "( ( 2I6, ES24.16 ) )" )                            &
                 data%K%row( i ), data%K%col( i ), data%K%val( i )
             END DO
           END IF

           H_perturb = zero ; new_mo = ' '
   50      CONTINUE

           IF ( printw ) WRITE( out,                                           &
                "( ' ............. factorization phase ............... ' )" )

!          IF ( inform%iter == 25 ) THEN
!            WRITE( 6, " ( ' ----------- dumping -------- ' )" )
!            WRITE( 22, "( 2I6 )" ) data%K%n, data%K%ne
!            WRITE( 22, "( ( 10I6 ) )" ) data%K%row( : data%K%ne )
!            WRITE( 22, "( ( 10I6 ) )" ) data%K%col( : data%K%ne ) 
!            WRITE( 22, "( ( 3ES24.16 ) )" ) data%K%val( : data%K%ne ) 
!            WRITE( 22, "( ( 3ES24.16 ) )" ) data%SOL( : data%K%n )
!          END IF

           CALL SILS_factorize( data%K, data%FACTORS, data%CNTL, FINFO )

!  Record the storage required

           inform%nfacts = inform%nfacts + 1 
           inform%factorization_integer = FINFO%nirbdu 
           inform%factorization_real = FINFO%nrlbdu

!  Test that the factorization succeeded

           zeig = 0
           inform%factorization_status = FINFO%flag

!  The factorization failed. If possible, increase the pivot tolerance

           IF ( FINFO%flag < 0 ) THEN
             IF ( printe ) WRITE( control%error, 2100 ) FINFO%flag,            &
                                                        'SILS_factorize'
             IF ( FINFO%flag == - 5 .OR. FINFO%flag == - 6 ) THEN
!              IF ( data%CNTL%u < half ) THEN
               IF ( data%CNTL%u < half ) THEN
                 IF ( data%CNTL%u < point01 ) THEN
                   data%CNTL%u = point01
                 ELSE
                   IF ( data%CNTL%u < point1 ) THEN
                     data%CNTL%u = point1
                   ELSE
                     data%CNTL%u = half
                   END IF
                 END IF
                 IF ( printi ) THEN
                   WRITE( out, "( ' potentially zero pivot detected ' )" )
                   WRITE( out, 2070 ) data%CNTL%u
                 END IF
                 GO TO 50
               END IF
             ELSE IF ( FINFO%flag == - 3 ) THEN
               IF ( auto .AND. precon /= 4 ) THEN
                 precon = 4
!                fact_hist = 5
!                WRITE( out, "( ' change precon to ', I2 )" ) precon
                 analyse = .TRUE.
                 GO TO 40
               END IF
             END IF
             precon = - precon 
             GO TO 60
!            inform%status = GALAHAD_error_factorization ; GO TO 990

!  Record warning conditions

           ELSE IF ( FINFO%flag > 0 ) THEN
             IF ( printt ) WRITE( control%out, 2060 )                          &
                                  FINFO%flag, 'SILS_factorize'
             IF ( FINFO%flag == 4 ) THEN 
                zeig = data%K%n - FINFO%rank
                IF ( printt ) WRITE( control%out, "( ' ** Matrix has ', I7,    &
                             &       ' zero eigenvalues ' )" ) zeig
             END IF 
           END IF 

!  The problem is not convex on the null space. Modify the
!  preconditioner and refactorize

!          write(6,*) FINFO%neig, zeig, m, h_max
!          IF ( FINFO%neig + zeig > 2 * m ) THEN 
           IF ( FINFO%neig + zeig > m ) THEN 
             IF ( new_mo /= ' ' ) THEN 
               IF ( data%CNTL%u < half ) THEN
                 IF ( data%CNTL%u < point01 ) THEN
                   data%CNTL%u = point01
                 ELSE
                   IF ( data%CNTL%u < point1 ) THEN
                     data%CNTL%u = point1
                   ELSE
                     data%CNTL%u = half
                   END IF
                 END IF
                 IF ( printi ) THEN
                   WRITE( out, "( ' incorrect inertia detected ' )" )
                   WRITE( out, 2070 ) data%CNTL%u
                 END IF
                 H_perturb = zero ; new_mo = ' '
                 GO TO 50
               END IF
               precon = - precon ; GO TO 60
!              inform%status = GALAHAD_error_inertia ; GO TO 990
             END IF
             IF ( new_mo == ' ' ) inform%nmods = inform%nmods + 1
             new_mo = 'm'
             H_perturb = H_perturb + h_norm
!            H_perturb = H_perturb + n * h_max
!            write(6,*) n * h_max, h_norm
             IF ( printt )                                                     &
               WRITE( out, "( /, ' Preconditioner is inappropriate as it has ',&
              & /, I6, ' negative and ', I6, ' zero eigenvalues rather than',  &
              & /, I6, ' negative and ', I6, ' zero ones',                     &
              & /, ' Perturbing H', :, ' by ', ES12.4, ' and restarting ' )" ) &
               FINFO%neig, zeig, m, 0, H_perturb

!  Increase the diagonal of P

             DO i = 1, nfree
               data%K%val( nnzks + i ) = data%K%val( nnzks + i ) + H_perturb
             END DO

             GO TO 50
           END IF 

!  The factorization is complete

         END IF
   60    CONTINUE
         refact = .FALSE.

!  Print a summary of the last iteration

         IF ( printi ) THEN
           CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total
           IF ( inform%iter > 0 ) THEN
             IF ( print_level > 1 .OR. new_inner ) WRITE( out, 2030 )
             IF ( new_inner ) THEN
               WRITE( out, "( I6, ES12.4, 2ES8.1, '    -   ', ES8.1,           &
              &   '     -          -   -', F8.1 )" ) inform%iter, merit,       &
                inform%pr_feas, inform%du_feas, inform%radius, inform%time%total
             ELSE
               WRITE( out,                                                     &
                   "( I6, ES12.4, 4ES8.1, ES9.1, A1, I7, A1, I3, F8.1 )" )     &
                 inform%iter, merit, inform%pr_feas, inform%du_feas, step,     &
                 old_radius, ratio, restrict, cg_iter, mo, nbacts,             &
                 inform%time%total
             END IF
           ELSE
             WRITE( out, 2030 )
             WRITE( out, "( I6, ES12.4, ES8.1, '    -       -   ', ES8.1,      &
            &     '     -          -   -', F8.1 )" ) inform%iter, merit,       &
              inform%pr_feas, inform%radius, inform%time%total
           END IF
         END IF

!  Start the next iteration

         inform%iter = inform%iter + 1
         nbacts = 0 ; step = zero ; ratio = zero

         IF ( inform%iter > control%maxit ) THEN
           IF ( printi ) WRITE( out, "( /, ' Iteration limit exceeded ' )" )
           inform%status = GALAHAD_error_max_iterations ; GO TO 905
         END IF

         IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
           printe = set_printe ; printi = set_printi ; printt = set_printt
           printm = set_printm ; printw = set_printw ; printd = set_printd
           printd2 = set_printd2 ; printd4 = set_printd4
           print_level = control%print_level
         ELSE
           printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
           printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
           printd2 = .FALSE. ; printd4 = .FALSE.
           print_level = 0
         END IF

         IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
           print_level = control%print_level
           control%gltr_control%print_level = print_level - 1
           IF ( print_level > 1 ) THEN
             control%gltr_control%error = control%error
           ELSE
             control%gltr_control%error = 0
           END IF
         ELSE
           print_level = 0 ; control%gltr_control%print_level = 0
           control%gltr_control%error = 0
         END IF

         IF ( print_level > 10 ) THEN

!  Print debug details

           WHERE( data%X_l > - control%infinity )
             data%prob%X_l( : n ) = MAX( - inform%radius, data%X_l - data%X )
           ELSEWHERE
             data%prob%X_l( : n ) = - inform%radius
           END WHERE

           WHERE( data%X_u < control%infinity )
             data%prob%X_u( : n ) = MIN( inform%radius, data%X_u - data%X )
           ELSEWHERE
             data%prob%X_u( : n ) = inform%radius
           END WHERE

           WHERE( data%C_l > - control%infinity )
             data%prob%C_l( : m ) = data%C_l - data%C
           ELSEWHERE
             data%prob%C_l( : m ) = data%C_l
           END WHERE

           WHERE( data%C_u < control%infinity )
             data%prob%C_u( : m ) = data%C_u - data%C
           ELSEWHERE
             data%prob%C_u( : m ) = data%C_u
           END WHERE

           WRITE( out, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" ) n, m, f
           WRITE( out, "( ' g ', /, ( 5ES12.4 ) )" ) data%prob%G( : n )
           WRITE( out, "( ' x_l ', /, ( 5ES12.4 ) )" ) data%prob%X_l( : n )
           WRITE( out, "( ' x_u ', /, ( 5ES12.4 ) )" ) data%prob%X_u( : n )
           WRITE( out, "( ' c_l ', /, ( 5ES12.4 ) )" ) data%prob%C_l( : m )
           WRITE( out, "( ' c_u ', /, ( 5ES12.4 ) )" ) data%prob%C_u( : m )
           WRITE( out, "( ' A_row ', /, ( 10I6 ) )" )                          &
             data%prob%A%row( : data%prob%A%ne )
           WRITE( out, "( ' A_col ', /, ( 10I6 ) )" )                          &
             data%prob%A%col( : data%prob%A%ne )
           WRITE( out, "( ' A_val ', /, ( 5ES12.4 ) )" )                       &
             data%prob%A%val( : data%prob%A%ne )
           WRITE( out, "( ' H_row ', /, ( 10I6 ) )" )                          &
             data%prob%H%row( : data%prob%H%ne )
           WRITE( out, "( ' H_col ', /, ( 10I6 ) )" )                          &
             data%prob%H%col( : data%prob%H%ne )
           WRITE( out, "( ' H_val ', /, ( 5ES12.4 ) )" )                       &
             data%prob%H%val( : data%prob%H%ne )
         END IF

!  See if the M-norm of the gradient of the Lagrangian is small

!        IF ( use_primal_dual .AND. precon > 0 .AND.                           &
!             inform%pr_feas <= theta_p .AND.                                  &
!             inform%comp_slack <= theta_c ) THEN 
!          data%VECTOR( : nfree ) = data%GRAD_m( : nfree ) 
!          data%VECTOR( nfree + 1 : ) = zero
!          IF ( control%eliminate_elastics ) THEN
!            CALL SUPERB_block_refinement(                                     &
!               m, nfree, nelastic, data%ELASTICS, data%K, data%B_C,           &
!               data%B_CM, data%B_S, data%SCALE_S, data%FACTORS, data%CNTL,    &
!               data%SOL, data%VECTOR, data%RES, data%BEST,                    &
!               res_norm, big_res, itref_max, print_level, out )
!          ELSE
!            CALL SUPERB_iterative_refinement( data%K, data%FACTORS,           &
!               data%CNTL, data%SOL, data%VECTOR, data%RES, data%BEST,         &
!               res_norm, big_res, itref_max, print_level, out )
!          END IF
!          IF ( .NOT. big_res ) THEN
!            val = SQRT( ABS( DOT_PRODUCT( data%GRAD_m( : nfree ),             &
!              data%SOL( : nfree ) ) ) )
!            IF ( printt ) WRITE( out, "( ' M norm of grad_l ', ES12.4 )" ) val
!!           IF ( val <= theta_d .AND. inform%comp_slack <= theta_c ) THEN
!!             inform%du_feas  = val ; GO TO 800
!            END IF
!          END IF
!        END IF

!  ----------------------------------------------------------------------------
!                       SEARCH DIRECTION COMPUTATION
!  ----------------------------------------------------------------------------

!  Solve the model problem to find a search direction.
!  Use the GLTR algorithm to compute a suitable trial step

!  Set initial data

         cg_iter = 0
         control%gltr_control%boundary = .FALSE.
         control%gltr_control%unitm = precon < 0
         inform%gltr_inform%status = 1
         inform%gltr_inform%negative_curvature = .TRUE.

         data%GRAD_m( : nfrpel ) = data%GRAD_b( : nfrpel )

         IF ( printm ) WRITE( out,                                             &
          "(/, '   |------------------------------------------------------|',  &
        &   /, '   |        start to solve trust-region subproblem        |',  &
        &   / )" )

         CALL CPU_TIME( time )

! Inner-most (GLTR) loop

         gltr_iter_eq_0 = .TRUE.

         DO
           IF ( printd2 ) WRITE( out,                                          &
                "( ' Grm ', /, ( 5ES12.4 ) )" )  data%GRAD_m( : nfrpel )
           CALL GLTR_solve( nfrpel, inform%radius, model,                 &
                            data%DV( : nfrpel ), data%GRAD_m( : nfrpel ),      &
                            data%VECTOR( : nfrpel ), data%gltr_data,           &
                            control%gltr_control, inform%gltr_inform )

!  Check for error returns

           SELECT CASE( inform%gltr_inform%status )

!  Successful return

           CASE ( 0 )
             EXIT

!  Warnings

           CASE ( GALAHAD_error_max_iterations, GALAHAD_warning_on_boundary )
             IF ( printt ) WRITE( out, "( /,                                   &
            &  ' Warning return from GLTR, status = ', I6 )" )                 &
               inform%gltr_inform%status
             EXIT
          
!  Allocation errors

           CASE ( GALAHAD_error_allocate )
             inform%status = GALAHAD_error_allocate
             inform%alloc_status = inform%gltr_inform%alloc_status
             inform%bad_alloc = inform%gltr_inform%bad_alloc
             GO TO 920

!  Deallocation errors

           CASE ( GALAHAD_error_deallocate )
             inform%status = GALAHAD_error_deallocate
             inform%alloc_status = inform%gltr_inform%alloc_status
             inform%bad_alloc = inform%gltr_inform%bad_alloc
             GO TO 920

!  Error return

           CASE DEFAULT
             IF ( printt ) WRITE( out, "( /,                                   &
            &  ' Error return from GLTR, status = ', I6 )" )                   &
               inform%gltr_inform%status
             EXIT

!  Find the preconditioned gradient

           CASE ( 2, 6 )
             IF ( printw ) WRITE( out,                                         &
                "( '    ............... precondition  ............... ' )" )

!  Compute the search direction, taking care to get small residuals

!            write(6,"( 'before ', (5ES12.4))" ) data%VECTOR( : nfrpel )
             data%VECTOR( nfrpel + 1 : ) = zero
             IF ( control%eliminate_elastics ) THEN
               CALL SUPERB_block_refinement(                                   &
                  m, nfree, nelastic, data%ELASTICS, data%K, data%B_C,         &
                  data%B_CM, data%B_S, data%SCALE_S, data%FACTORS, data%CNTL,  &
                  data%SOL, data%VECTOR, data%RES, data%BEST,                  &
                  res_norm, big_res, itref_max, print_level, out )
             ELSE
               CALL SUPERB_iterative_refinement( data%K, data%FACTORS,         &
                  data%CNTL, data%SOL, data%VECTOR, data%RES, data%BEST,       &
                  res_norm, big_res, itref_max, print_level, out )
             END IF

!  Ensure that the residuals are small. If not, try to obtain a more
!  accurate factorization and try again

             IF ( big_res ) THEN
               IF ( data%CNTL%u < half ) THEN
                 IF ( data%CNTL%u < point01 ) THEN
                   data%CNTL%u = point01
                 ELSE
                   IF ( data%CNTL%u < point1 ) THEN
                     data%CNTL%u = point1
                   ELSE
                     data%CNTL%u = half
                   END IF
                 END IF
                 refact = .TRUE.
                 IF ( printi ) THEN
                   WRITE( out, "( ' residuals too large ' )" )
                   WRITE( out, 2070 ) data%CNTL%u
                 END IF
                 CYCLE inner
               ELSE
                 precon = - precon ; GO TO 60
!                inform%status = - 7 ; GO TO 990
               END IF
             END IF

!  See if the M-norm of the gradient of the barrier function is small

             IF ( gltr_iter_eq_0 ) THEN
               val = SQRT( ABS( DOT_PRODUCT( data%VECTOR( : nfrpel ),          &
                                             data%SOL( : nfrpel ) ) ) )

!              IF ( val <= theta_d .AND. comp_slack_primal <= theta_c          &
!                   .AND. .NOT. new_inner ) THEN
!              IF ( val <= theta_d .AND. comp_slack_primal <= theta_c ) THEN
               IF ( val <= theta_d .AND. comp_slack_primal <= theta_c          &
                    .AND. ( .NOT. new_inner .OR. inform%pr_feas <= theta_p ) ) &
                    THEN
                 IF ( printm ) WRITE( out,                                     &
          "(/, '   |           trust-region subproblem solved             |',  &
        &   /, '   |------------------------------------------------------|',  &
        &     / )" )
                 IF ( printt ) WRITE( out,                                     &
                   "( ' primal vs primal-dual feasibility', 2ES11.4 )" )       &
                  inform%du_feas, val

!  If primal-dual multipliers are being used, reset these to the terminating
!  primal ones

!                IF ( precon == -27 ) THEN
                 IF ( use_primal_dual ) THEN
                   inform%comp_slack = comp_slack_primal
                   DO l = 1, nfree
                     i = data%XFREE( l )
                     IF ( data%X_l( i ) > - control%infinity )                 &
                       data%Z_l( i ) = data%Z_l_P( i )
                     IF ( data%X_u( i ) <   control%infinity )                 &
                       data%Z_u( i ) = data%Z_u_P( i )
                   END DO
                   DO i = 1, m
                     j = data%SSTATE( i )
                     IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
                       IF ( data%S_type( i ) /= 0 ) data%U( j ) = data%U_P( j )
                       IF ( control%bound_elastics )                           &
                         data%U_u( j ) = data%U_u_P( j )
                     END IF
                     IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                       SELECT CASE ( data%S_type( i ) )
                       CASE( 0 )
                         data%Y_l( i ) = data%Y_l_P( i )
                         data%Y_u( i ) = data%Y_u_P( i )
                       CASE( 1, 2 )
                         data%Y_l( i ) = data%Y_l_P( i )
                       CASE( - 1, - 2 )
                         data%Y_u( i ) = data%Y_u_P( i )
                       CASE DEFAULT
                         GO TO 980
                       END SELECT
                     ELSE
                       SELECT CASE ( data%S_type( i ) )
                       CASE( 1, 2 )
                         IF ( data%C_l( i ) > - control%infinity )             &
                           data%Y_l( i ) = data%Y_l_P( i )
                         IF ( data%C_u( i ) <   control%infinity )             &
                           data%Y_u( i ) = data%Y_u_P( i )
!                      CASE( 3 )
!                      CASE( - 3 )
                       CASE DEFAULT
                         GO TO 980
                       END SELECT
                     END IF
                   END DO
                 END IF
                 inform%du_feas = val
                 stop_inner_status = 2 ; GO TO 800
               END IF
               gltr_iter_eq_0 = .FALSE.
             END IF

!  Replace the solution in VECTOR

             data%VECTOR( : nfrpel ) = data%SOL( : nfrpel )
!            write(6,"( 'after  ', (5ES12.4))" ) data%SOL( nfrpel + 1 : )

!  Form the product of VECTOR with H

           CASE ( 3, 7 )

             IF ( printw ) WRITE( out,                                         &
                  "( '    ............ matrix-vector product .......... ' )" )

!            CALL SUPERB_H_b( data%prob, nfree, nelastic, data%XSTATE,         &
!              data%SSTATE, data%B_X, data%B_S, data%SCALE_S, data%B_C,        &
!              data%B_CM, data%VECTOR( : nfrpel ), data%SOL, vTHv, .TRUE. )
!            write(6,*) ' vthv = ', vthv

             CALL SUPERB_H_b( data%prob, nfree, nelastic, data%XSTATE,         &
               data%SSTATE, data%B_X, data%B_S, data%SCALE_S, data%B_C,        &
               data%B_CM, data%VECTOR( : nfrpel ), data%SOL, vTHv, .FALSE. )

!            write(6,*) ' vthv = ', DOT_PRODUCT( data%VECTOR( : nfrpel ), &
!                                                data%SOL( : nfrpel ) )


!  Replace the product in VECTOR

             data%VECTOR( : nfrpel ) = data%SOL( : nfrpel )

!  Reform the initial residual

           CASE ( 5 )
           
             IF ( printw ) WRITE( out,                                         &
                  "( '    ................. restarting ................ ' )" )

             data%GRAD_m( : nfrpel ) = data%GRAD_b( : nfrpel )
           END SELECT
         END DO

         CALL CPU_TIME( dum ) ; dum = dum - time
         IF ( precon < 0 ) precon = - precon

         IF ( printm ) WRITE( out,                                             &
          "(/, '   |           trust-region subproblem solved             |',  &
        &   /, '   |------------------------------------------------------|',  &
        &     / )" )

         IF ( printw ) WRITE( out,                                             &
              "( ' ............... step computed ............... ' )" )

         IF ( printt ) WRITE( out, "( ' solve time = ', F10.2 ) " ) dum
         inform%time%solve = inform%time%solve + dum

         cg_iter = cg_iter + inform%gltr_inform%iter
         inform%cg_iter = inform%cg_iter + cg_iter

!  If the overall search direction is unlikely to make a significant
!  impact on the residual, exit

         IF ( inform%gltr_inform%mnormx <= teneps ) THEN
           stop_inner_status = 3 ; GO TO 800
         END IF
!        WRITE(6,"( ( 2ES12.4 ) )" ) ( data%LAMBDA(i), data%SOL(nfrpel+i),i=1,m)

!  ----------------------------------------------------------------------------
!                       STEP ACCEPTANCE TESTS
!  ----------------------------------------------------------------------------

         pred = - model
         model = model + merit

!  Compute the step size

!        step = SQRT( DOT_PRODUCT( data%DV( : nfrpel ), data%DV( : nfrpel ) ) )
         step = inform%gltr_inform%mnormx

!        IF ( step > inform%radius ) THEN 
!          data%DV( : nfrpel ) = inform%radius * data%DV( : nfrpel ) / step
!          step = inform%radius
!        END IF

!        WRITE(6,"( ( 2ES12.4 ) )" ) ( data%GRAD_b( i ), data%DV( i ),i=1,n+m)

! steepest descent:
!        norm_gb = SQRT( DOT_PRODUCT( data%GRAD_b, data%GRAD_b ) )
!        data%DV = - inform%radius * data%GRAD_b / norm_gb
!        model = merit - norm_gb * inform%radius

!  Compute the model value

         IF ( printm ) THEN
           WRITE( out, "( ' estimated model decrease ', ES12.4 )" ) merit-model
         END IF

         slope = DOT_PRODUCT( data%GRAD_b( : nfrpel ), data%DV( : nfrpel ) )
         IF ( slope >= zero ) write( 6, "( ' slope ', ES12.4 )" ) slope

         IF ( printm ) THEN
           CALL SUPERB_H_b( data%prob, nfree, nelastic, data%XSTATE,           &
             data%SSTATE, data%B_X, data%B_S, data%SCALE_S, data%B_C,          &
             data%B_CM, data%DV( : nfrpel ), data%RES, curv, .TRUE. )
           model = merit + slope + half * curv
           WRITE( out, "( ' true model decrease      ', ES12.4 )" )            &
             - slope - half * curv
         END IF

!        IF ( printi ) THEN
         IF ( print_debug .AND. printi ) THEN
           WRITE( out, "( '  x ', /, ( 3ES22.14 ) )" ) data%X( 1 : nfree )
           WRITE( out, "( '  s ', /, ( 3ES22.14 ) )" ) data%S( 1 : nelastic )
           WRITE( out, "( ' dx ', /, ( 3ES22.14 ) )" ) data%DV( 1 : nfree )
           WRITE( out, "( ' ds ', /, ( 3ES22.14 ) )" ) data%DV(nfree+1 : nfrpel)
         END IF

         IF ( printd ) THEN
           WRITE( out, "( ' dx ', /, ( 5ES12.4 ) )" ) data%DV( 1 : nfree )
           WRITE( out, "( ' ds ', /, ( 5ES12.4 ) )" ) data%DV( nfree+1 : nfrpel)
         END IF

!  Find the largest feasible step for x

         alpha_x = infinity
         DO l = 1, nfree
           i = data%XFREE( l ) 
           IF ( data%X_l( i ) > - control%infinity ) THEN
!            WRITE( 6, "( ' Xl ', ES12.4 )" )  data%X( i ) - data%X_l( i )
             IF( data%DV( l ) < zero ) alpha_x = MIN( alpha_x, -               &
               ( data%X( i ) - data%X_l( i ) ) / data%DV( l ) )
           END IF
           IF ( data%X_u( i ) <   control%infinity ) THEN
!            WRITE( 6, "( ' Xu ', ES12.4 )" )  data%X_u( i ) - data%X( i )
             IF( data%DV( l ) > zero ) alpha_x = MIN( alpha_x,                 &
               ( data%X_u( i ) - data%X( i ) ) / data%DV( l ) )
           END IF
         END DO

!  Find the largest feasible step for s

         alpha_s = infinity
         DO j = 1, nelastic
           IF ( data%DV( nfree + j ) < zero )                                  &
             alpha_s = MIN( alpha_s, - data%S( j ) / data%DV( nfree + j ) )
           IF ( control%bound_elastics .AND. data%DV( nfree + j ) > zero )     &
             alpha_s = MIN( alpha_s, ( data%S_u( j ) - data%S( j ) ) /         &
               data%DV( nfree + j ) )
         END DO

!  Find the largest feasible step for the linearized constraints

         alpha_c = infinity
         DO i = 1, m
           jTd = zero
           DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1
             j =  data%XSTATE( data%prob%A%col( l ) )
             IF ( j > 0 ) jTd = jTd + data%prob%A%val( l ) * data%DV( j )
           END DO 

           j = data%SSTATE( i )
!          WRITE( 6, * )  j, data%S_type( i )
           IF ( data%C_l( i ) == data%C_u( i ) ) THEN
             SELECT CASE ( data%S_type( i ) )
             CASE( 0 )
               IF ( jTd + data%SCALE_S( j ) * data%DV( nfree + j ) < zero )    &
                 alpha_c = MIN( alpha_c, - ( data%C( i ) - data%C_l( i ) +     &
                   data%SCALE_S( j ) * data%S( j ) ) /                         &
                   ( jTd + data%SCALE_S( j ) * data%DV( nfree + j ) ) )
               IF ( jTd - data%SCALE_S( j ) * data%DV( nfree + j ) > zero )    &
                 alpha_c = MIN( alpha_c, ( data%C_u( i ) - data%C( i ) +       &
                   data%SCALE_S( j ) * data%S( j ) ) /                         &
                   ( jTd - data%SCALE_S( j ) * data%DV( nfree + j ) ) )
             CASE( 1 )
               IF ( jTd < zero )                                               &
                 alpha_c = MIN( alpha_c, - ( data%C( i ) - data%C_l( i ) )/jTd )
             CASE( - 1 )
               IF ( jTd > zero )                                               &
                 alpha_c = MIN( alpha_c, ( data%C_u( i ) - data%C( i ) ) / jTd )
             CASE( 2 )
               jTd = jTd + data%SCALE_S( j ) * data%DV( nfree + j )
               IF ( jTd < zero )                                               &
                 alpha_c = MIN( alpha_c, - ( data%C( i ) - data%C_l( i ) +     &
                   data%SCALE_S( j ) * data%S( j ) ) / jTd )
             CASE( - 2 )
               jTd = jTd - data%SCALE_S( j ) * data%DV( nfree + j )
               IF ( jTd > zero )                                               &
                 alpha_c = MIN( alpha_c, ( data%C_u( i ) - data%C( i ) +       &
                   data%SCALE_S( j ) * data%S( j ) ) / jTd )
             CASE DEFAULT
               GO TO 980
             END SELECT
           ELSE
             SELECT CASE ( data%S_type( i ) )
             CASE( 1 )
               IF ( data%C_l( i ) > - control%infinity .AND. jTd < zero )      &
                 alpha_c = MIN( alpha_c, - ( data%C( i ) - data%C_l( i ) )/jTd )
               IF ( data%C_u( i ) <   control%infinity  .AND. jTd > zero )     &
                 alpha_c = MIN( alpha_c, ( data%C_u( i ) - data%C( i ) ) / jTd )
             CASE( 2 )
!              write(6,*) jTd - data%SCALE_S( j ) * data%DV( nfree + j ),      &
!              ( data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) * data%S( j ) )
               IF ( data%C_l( i ) > - control%infinity .AND.                   &
                    jTd + data%SCALE_S( j ) * data%DV( nfree + j ) < zero )    &
                      alpha_c = MIN( alpha_c, - ( data%C( i ) - &
                      data%C_l( i ) + data%SCALE_S( j ) * data%S( j ) ) /      &
                       ( jTd + data%SCALE_S( j ) * data%DV( nfree + j ) ) )
               IF ( data%C_u( i ) <   control%infinity  .AND.                  &
                    jTd - data%SCALE_S( j ) * data%DV( nfree + j ) > zero )    &
                      alpha_c = MIN( alpha_c, ( data%C_u( i ) -                &
                      data%C( i ) + data%SCALE_S( j ) * data%S( j ) ) /        &
                       ( jTd - data%SCALE_S( j ) * data%DV( nfree + j ) ) )
!            CASE( 3 )
!            CASE( - 3 )
             CASE DEFAULT
               GO TO 980
             END SELECT
           END IF
         END DO

!  A step of no larger than zeta of the distance to the nearest 
!  bound will be attempted

         IF ( printm ) WRITE( out, "( ' alpha_x, _s, _c ', 3ES12.4 )" )        &
!        WRITE( out, "( ' alpha_x, _s, _c ', 3ES12.4 )" )                      &
           alpha_x, alpha_s, alpha_c

         alpha = MIN( one / ( one - zeta ), alpha_x, alpha_s, alpha_c )
         ratio = - point1 * HUGE( one ) 

         IF ( alpha >= one / ( one - zeta ) ) THEN
           alpha = one

!  Find the new point

           data%X_trial( data%XFREE ) = data%X( data%XFREE ) + data%DV( : nfree)
!          WRITE( out, "( ' x_trial ', /, ( 4ES16.8 ) )" ) data%X_trial

!  Compute the new function and constraint values

!          write(out,"( ' x_trial ', /, ( 4ES16.8 ) )" ) data%X_trial

           IF ( scale_xcf ) THEN
             CALL PTRANS_cfn( n, m, data%X_trial, f_trial, m, data%C_trial,    &
                              data%ptrans_transform, data%ptrans_data,         &
                              inform%ptrans_inform )
           ELSE
             CALL CUTEST_cfn( cutest_status, n, m, data%X_trial,               &
                              f_trial, data%C_trial )
             IF ( cutest_status /= 0 ) GO TO 930
           END IF     
           inform%f_eval = inform%f_eval + 1

!          write(out,"( ' f_trial ', /, ES16.8 )" ) f_trial
!          write(out,"( ' c_trial ', /, ( 4ES16.8 ) )" ) data%C_trial

!  Compute the new elastic

           IF ( control%magical_path ) THEN
             IF ( SUPERB_chop_magical( m, data%S_type, data%SSTATE,            &
                  data%C_trial, data%C_l, data%C_u,                            &
                  data%SCALE_S, data%S_u, len_s_u, control ) ) THEN
               got_ratio = .FALSE. ; GO TO 110 ; END IF
             data%S_trial( : nelastic ) = data%S( : nelastic )
             CALL SUPERB_magical( m, nelastic, data%S_type, data%SSTATE, mu,   &
                nu, data%C_l, data%C_u, data%C_trial,                          &
                data%S_trial( : nelastic ), data%SCALE_S( : nelastic ),        &
                data%S_u, len_s_u, control, invalid )
             IF ( invalid ) GO TO 980
           ELSE
             data%S_trial( : nelastic ) =                                      &
               data%S( : nelastic ) + data%DV( nfree + 1 : nfrpel )
           END IF

!  Compute the value of the merit function

           merit_trial = SUPERB_merit( n, m, nfree, nelastic, data%XFREE,      &
              data%S_type, data%SSTATE, f_trial, mu, nu, data%X_trial,         &
              data%X_l, data%X_u, data%C_trial, data%C_l, data%C_u,            &
              data%S_trial, data%SCALE_S, data%S_u, len_s_u, inform%pr_feas,   &
              barrier, penalty, merit_error, print_level, out, control )
           IF ( merit_error == - 99 ) GO TO 980

!          IF ( merit_error == 0 ) THEN
!            write(6,"( ' merit ', ES12.4 )") merit_trial
!          ELSE
!            write(6,"( ' merit    infinity ' )") 
!          END IF

!  Ensure that the step is allowed

           got_ratio = merit_error == 0
         ELSE
           alpha = ( one - zeta ) * alpha
!          write(6,"( ' reduced step = ', ES12.4 )" ) alpha
           alpha = alpha / reduce_factor
           nbacts = - 1 
           got_ratio = .FALSE. 
         END IF

  110    CONTINUE
         g_recent = .FALSE. ; y_recent = .FALSE.

         IF ( got_ratio ) THEN
           restrict = ' '
         ELSE
           restrict = 'r'
         END IF

         IF ( got_ratio ) THEN

!  If the primal-dual model is used, comnpute the differece between
!  the primal and primal-dual model 

! To be done !!

           IF ( printm ) WRITE( out, "(' objective decrease vs value ',        &
          &                     2ES12.4 )")  merit - merit_trial, ABS( merit )
!          write(6,*) merit, merit_trial, model
           ared = merit - merit_trial
!          pred = merit - model
!          pred = ared
!          write(6,"( ' ared, pred = ', 2ES12.4 )" ) ared, pred
           IF ( ared == zero .AND. pred == zero ) THEN
             stop_inner_status = 5 ; GO TO 800
           END IF
           ared = ared + MAX( one, ABS( merit ) ) * teneps
           pred = pred + MAX( one, ABS( merit ) ) * teneps
           IF ( pred > zero ) THEN
             ratio = ared / pred 
           ELSE
             IF ( printi .AND. step > step_tiny )                              &
               WRITE( out, "( ' --> predicted reduction =', ES10.2 )" ) pred
           END IF
           IF ( printm ) WRITE( out, "(' actual, predicted reduction ',        &
          &                     2ES12.4 )")  ared, pred

!          write(out,"( ES16.8 )" ) ratio

!  Adjust ratio in the non-monotone case

           IF ( nmhist > 0 ) THEN
             ar_h = ( merit_ref - merit_trial )                                &
                      + MAX( one, ABS( merit_ref ) ) * teneps
             pr_h = sigma_r + pred
             IF ( ABS( ar_h ) < teneps .AND. ABS( merit_ref ) > teneps )       &
               ar_h = pr_h
             ratio = MAX( ratio, ar_h / pr_h )
           END IF
         END IF

!  ----------------------------------------------------------------------------
!                         SUCCESSFUL STEP 
!  ----------------------------------------------------------------------------

         IF ( got_ratio .AND.                                                  &
              ( ratio >= control%rho_successful .OR. new_inner ) ) THEN
!             ratio >= control%rho_successful ) THEN

           IF ( printw ) WRITE( out,                                           &
              "( ' ............... successful step ............... ' )" )

!  In the non-monotone case, update the sum of predicted models

           IF ( nmhist > 0 ) THEN
             sigma_c = sigma_c + merit - model 
             sigma_r = sigma_r + merit - model

!  If appropriate, update the best value found

             IF ( merit_trial < merit_min ) THEN
               merit_min = merit_trial ; merit_current = merit_min
               sigma_c = zero ; l_suc = 0
             ELSE
               l_suc = l_suc + 1

!  Check to see if there is a new candidate for the next reference value

               IF ( merit_trial > merit_current ) THEN
                 merit_current = merit_trial ; sigma_c = zero ; END IF

!  Check to see if the reference value needs to be reset

               IF ( l_suc == nmhist ) THEN
                 merit_ref = merit_current ; sigma_r = sigma_c ; END IF
             END IF
           END IF

!          WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" ) X
!          WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" ) S

!  ----------------------------------------------------------------------------
!                         UNSUCCESSFUL STEP 
!  ----------------------------------------------------------------------------

         ELSE

           IF ( printw ) WRITE( out,                                           &
                "( ' .............. unsuccessful step .............. ' )" )

!  As we have not achieved a sufficient reduction, use the Nocedal-Yuan 
!  technique to achieve one. To do this, perform a line-search to find
!  a point x + alpha s which sufficiently reduces the barrier function

           alpha = reduce_factor * alpha

!  :::::::::::::::::::::::::::::::::::::::::::::::::::::::
!   Use a backtracking line-search, starting from alpha_v
!  :::::::::::::::::::::::::::::::::::::::::::::::::::::::

           IF ( printt ) WRITE( out, "( /, ' value = ', ES12.4, ' slope = ',   &
          &     ES12.4 )" ) merit, slope
           IF ( printw ) WRITE( out,                                           &
                "( ' ................ linesearch ................ ' )" )

           IF ( control%exact_linesearch ) THEN
!            ared = alpha ; pred = slope
!            CALL SUPERB_Armijo_linesearch( n, m, nfree, nelastic, data%XFREE, &
!              data%S_type, data%SSTATE, mu, nu, data%X, data%X_l, data%X_u,   &
!              data%C_l, data%C_u, data%S, data%S_u, len_s_u, barrier, penalty,&
!              alpha, data%X_trial, data%C_trial, data%S_trial,                &
!              data%DV( : nfrpel ), merit, merit_trial, f_trial, slope, eta,   &
!              search_error, print_level, out, printt, nbacts, ratio_final,    &
!              scale_xcf, data%ptrans_transform, data%ptrans_data,             &
!              control, inform )
!            IF ( inform%status == - 99 ) GO TO 980
!            alpha = ared / reduce_factor ; slope = pred
             CALL SUPERB_exact_linesearch( n, m, nfree, nelastic, data%XFREE,  &
               data%S_type, data%XSTATE, data%SSTATE, data%ELASTICS, mu, nu,   &
               data%X, data%X_l, data%X_u, data%C_l, data%C_u, data%S,         &
               data%SCALE_S, data%S_u, len_s_u, merit, barrier, penalty,       &
               alpha, data%X_trial, data%C_trial, data%S_trial,                &
               data%DV( : nfrpel ), merit_trial, f_trial, slope, data%prob,    &
               data%Z_l_P, data%Z_u_P, data%U_P, data%U_u_P, data%Y_l_P,       &
               data%Y_u_P, data%LAMBDA, data%GRAD_b, data%IW, liw, J_ne,       &
               search_error, print_level, out, ratio_final, printt, scale_xcf, &
               data%ptrans_transform, data%ptrans_data, control, inform )
             IF ( inform%status == - 99 ) GO TO 980
             g_recent = .TRUE. ; y_recent = .TRUE.
           ELSE
             CALL SUPERB_Armijo_linesearch( n, m, nfree, nelastic, data%XFREE, &
               data%S_type, data%SSTATE, mu, nu, data%X, data%X_l, data%X_u,   &
               data%C_l, data%C_u, data%S, data%SCALE_S, data%S_u, len_s_u,    &
               barrier, penalty, alpha, data%X_trial, data%C_trial,            &
               data%S_trial, data%DV( : nfrpel ), merit, merit_trial, f_trial, &
               slope, eta, search_error, print_level, out, printt, nbacts,     &
               ratio_final, scale_xcf, data%ptrans_transform,                  &
               data%ptrans_data, control, inform )
             IF ( inform%status == - 99 ) GO TO 980
           END IF

!          WRITE(6,"( ' X ', ( 6ES12.4 ) )" ) data%X( : n )
           IF ( search_error /= 0 ) THEN
             stop_inner_status = 4 ; GO TO 810
           END IF

         END IF

!  If required, use primal-dual multiplier estimates

!        IF ( got_ratio .AND. ratio >= control%rho_successful .AND.            &
!             use_primal_dual ) THEN
         IF ( use_primal_dual ) THEN

           mult_max_eq = two * nu + mu ; mult_max = nu + mu
!          mult_max_eq = HUGE( one ) ; mult_max = HUGE( one )
!          alpha_z = one ; alpha_u = one ; alpha_y = one
           alpha_z = alpha ; alpha_u = alpha ; alpha_y = alpha

!  For debugging, print details of multiplier updates. For each multiplier,
!  give (a) the existing estimate, (b) the new primal-dual estimate, and (c) 
!  the new primal estimate

           IF ( printd ) THEN
!          IF ( printi ) THEN

!  Multipliers for simple bounds

             DO l = 1, nfree
               i = data%XFREE( l ) 
               IF ( data%X_l( i ) > - control%infinity ) THEN
                 WRITE( 6, "( ' data%Z_l  P, PD. P_new ', 3ES12.4 )" )         &
                   data%Z_l( i ), ( 1 - alpha_z ) * data%Z_l( i ) + alpha_z *  &
                   ( mu - data%Z_l( i ) * data%DV( l ) ) /                     &
                   ( data%X( i ) - data%X_l( i ) ),                            &
                   mu / ( data%X_trial( i ) - data%X_l( i ) )
               END IF                                                          
               IF ( data%X_u( i ) <   control%infinity ) THEN
                 WRITE( 6, "( ' data%Z_u  P, PD, P_new ', 3ES12.4 )" )         &
                   data%Z_u( i ), ( 1 - alpha_z ) * data%Z_u( i ) + alpha_z *  &
                   ( mu + data%Z_u( i ) * data%DV( l ) ) /                     &
                   ( data%X_u( i ) - data%X( i ) ),                            &
                   mu / ( data%X_u( i ) - data%X_trial( i ) )
               END IF                                                          
             END DO                                                            
                                                                             
!  Multipliers for general constraints

             DO i = 1, m                                                       
               jTd = zero                                                      
               DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1       
                 j =  data%XSTATE( data%prob%A%col( l ) )
                 IF ( j > 0 ) jTd = jTd + data%prob%A%val( l ) * data%DV( j )   
               END DO                                                          
               j = data%SSTATE( i )
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
   
!  Multipliers for elastic variables for equality constraints

                 IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
                  IF ( data%S_type( i ) /= 0 )                                 &
                    WRITE( 6, "( ' u    P, PD, P_new ', 3ES12.4 )" )           &
                     data%U( j ), ( 1 - alpha_u ) * data%U( j ) + alpha_u *    &
                     ( mu - data%U( j ) * data%DV( nfree + j ) ) / data%S( j ),&
                     mu / data%S_trial( j )
                  IF ( control%bound_elastics )                                &
                    WRITE( 6, "( ' u_u  P, PD, P_new ', 3ES12.4 )" )           &
                      data%U_u( j ), ( 1 - alpha_u ) * data%U_u( j ) +         &
                      alpha_u * ( mu - data%U_u( j ) *                         &
                      ( - data%DV( nfree + j ) ) ) / ( data%S_u( j ) -         &
                      data%S( j ) ), mu / ( data%S_u( j ) - data%S_trial( j ) )
                  END IF                                                        

!  Multipliers for equality constraints

                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                    WRITE( 6, "( ' y_el  P, PD, P_new ', 3ES12.4 )" )          &
                      data%Y_l( i ), ( 1 - alpha_y ) * data%Y_l( i ) +         &
                        alpha_y * ( mu - data%Y_l( i ) *                       &
                        ( jTd + data%SCALE_S( j ) * data%DV( nfree + j ) ) )   &
                        / ( data%C( i ) - data%C_l( i ) +                      &
                        data%SCALE_S( j ) * data%S( j ) ),                     &
                        mu / ( data%C_trial( i ) - data%C_l( i ) +             &
                        data%SCALE_S( j ) * data%S_trial( j ) )
                    WRITE( 6, "( ' y_eu  P, PD, P_new ', 3ES12.4 )" )          &
                      data%Y_u( i ),( 1 - alpha_y ) * data%Y_u( i ) +          &
                        alpha_y * ( mu - data%Y_u( i ) *                       &
                        ( - jTd + data%SCALE_S( j ) * data%DV( nfree + j ) ) ) &
                        / ( data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) *  &
                        data%S( j ) ), mu / ( data%C_u( i ) -                  &
                        data%C_trial( i ) + data%SCALE_S( j ) *                &
                        data%S_trial( j ) )
                 CASE( 1 )                                                     
                    WRITE( 6, "( ' y_e  P, PD, P_new ', 3ES12.4 )" )           &
                      data%Y_l( i ), ( 1 - alpha_y ) * data%Y_l( i ) +         &
                      alpha_y * ( mu - data%Y_l( i ) * ( jTd ) ) /             &
                      ( data%C( i ) - data%C_l( i ) ),                         &
                       mu / ( data%C_trial( i ) - data%C_l( i ) )
                 CASE( - 1 )                                                   
                    WRITE( 6, "( ' y_e  P, PD, P_new ', 3ES12.4 )" )           &
                      data%Y_u( i ),( 1 - alpha_y ) * data%Y_u( i ) +          &
                      alpha_y * ( mu - data%Y_u( i ) * ( - jTd ) ) /           &
                      ( data%C_u( i ) - data%C( i ) ),                         &
                      mu / ( data%C_u( i ) - data%C_trial( i ) )
                 CASE( 2 )                                                     
                    WRITE( 6, "( ' y_e  P, PD, P_new ', 3ES12.4 )" )           &
                      data%Y_l( i ), ( 1 - alpha_y ) * data%Y_l( i ) +         &
                      alpha_y * ( mu - data%Y_l( i ) *                         &
                      ( jTd + data%SCALE_S( j ) * data%DV( nfree + j ) ) )     &
                      / ( data%C( i ) - data%C_l( i ) + data%SCALE_S( j ) *    &
                      data%S( j ) ), mu / ( data%C_trial( i ) - data%C_l( i )  &
                      + data%SCALE_S( j ) * data%S_trial( j ) )
                 CASE( - 2 )                                                   
                    WRITE( 6, "( ' y_e  P, PD, P_new ', 3ES12.4 )" )           &
                      data%Y_u( i ),( 1 - alpha_y ) * data%Y_u( i ) +          &
                      alpha_y *( mu - data%Y_u( i ) *                          &
                      ( - jTd + data%SCALE_S( j ) * data%DV( nfree + j ) ) )   &
                      / ( data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) *    &
                      data%S( j ) ), mu / ( data%C_u( i ) - data%C_trial( i )  &
                      + data%SCALE_S( j ) * data%S_trial( j ) )
                 END SELECT                                                    
               ELSE                                                            

!  Multipliers for elastic variables for inequality constraints

                 IF ( ABS( data%S_type( i ) ) /= 1 ) THEN   
                   IF ( data%S_type( i ) /= 0 )                                &
                     WRITE( 6, "( ' u    P, PD, P_new ', 3ES12.4 )" )          &
                       data%U( j ),( 1 - alpha_u ) * data%U( j ) + alpha_u *   &
                       ( mu - data%U( j ) * data%DV( nfree + j ) ) /           &
                        data%S( j ), mu / data%S_trial( j )
                   IF ( control%bound_elastics )                               &
                     WRITE( 6, "( ' u_u  P, PD, P_new ', 3ES12.4 )" )          &
                       data%U_u( j ),( 1 - alpha_u ) * data%U_u( j ) + alpha_u &
                       * ( mu - data%U_u( j ) * ( - data%DV( nfree + j ) ) ) / &
                       ( data%S_u( j ) - data%S( j ) ),                        &
                       mu / ( data%S_u( j ) - data%S_trial( j ) )
                 END IF 

!  Multipliers for inequality constraints

                 SELECT CASE ( data%S_type( i ) )
                 CASE( 1 )
                   IF ( data%C_l( i ) > - control%infinity ) THEN
                     WRITE( 6, "( ' y_il P, PD, P_new ', 3ES12.4 )" )          &
                       data%Y_l( i ), ( 1 - alpha_y ) * data%Y_l( i ) +        &
                       alpha_y * ( mu - data%Y_l( i ) * ( jTd ) ) /            &
                       ( data%C( i ) - data%C_l( i ) ), mu /                   &
                       ( data%C_trial( i ) - data%C_l( i ) )
                   END IF                                                      
                   IF ( data%C_u( i ) <   control%infinity ) THEN  
                     WRITE( 6, "( ' y_iu P, PD, P_new ', 3ES12.4 )" )          &
                        data%Y_u( i ), ( 1 - alpha_y ) * data%Y_u( i ) +       &
                        alpha_y *( mu - data%Y_u( i ) * ( - jTd ) ) /          &
                        ( data%C_u( i ) - data%C( i ) ),                       &
                         mu / ( data%C_u( i ) - data%C_trial( i ) )
                   END IF                                                      
                 CASE( 2 )                                                     
                   IF ( data%C_l( i ) > - control%infinity ) THEN
                     WRITE( 6, "( ' y_il P, PD, P_new ', 3ES12.4 )" )          &
                       data%Y_l( i ), ( 1 - alpha_y ) * data%Y_l( i ) +        &
                       alpha_y * ( mu - data%Y_l( i ) *                        &
                       ( jTd + data%SCALE_S( j ) * data%DV( nfree + j ) ) )    &
                       / ( data%C( i ) - data%C_l( i ) + data%SCALE_S( j ) *   &
                       data%S( j ) ), mu / ( data%C_trial( i ) - data%C_l( i ) &
                       + data%SCALE_S( j ) * data%S_trial( j ) )
                   END IF                                                      
                   IF ( data%C_u( i ) <   control%infinity ) THEN
                     WRITE( 6, "( ' y_iu P, PD, P_new ', 3ES12.4 )" )          &
                       data%Y_u( i ), ( 1 - alpha_y ) * data%Y_u( i ) +        &
                       alpha_y * ( mu - data%Y_u( i ) *                        &
                       ( - jTd  + data%SCALE_S( j ) * data%DV( nfree + i ) ) ) &
                       / ( data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) *   &
                       data%S( i ) ), mu / ( data%C_u( i ) - data%C_trial( i ) &
                       + data%SCALE_S( j ) * data%S_trial( i ) )
                   END IF                                                      
                 END SELECT                                                    
               END IF                                                          
             END DO                                                            

           END IF

!  Compute multiplier updates. Each multiplier is chosen to be the
!  new primal-dual value if possible, but if this is too small it will
!  be reset to the larger of a tiny constrant (mult_min) and a fraction
!  (nu_1) of the new primal value. In addition, multipliers for general
!  constraints will be no(t much) larger than a known upper bound 
!  obtained from dual-feasibility requirements for the elastic variables. 

!  Multipliers for simple bounds

           DO l = 1, nfree
             i = data%XFREE( l ) 
             IF ( data%X_l( i ) > - control%infinity ) THEN
               data%Z_l( i ) = MAX( mult_min,                                  &
                 ( 1 - alpha_z ) * data%Z_l( i ) + alpha_z *                   &
                 ( mu - data%Z_l( i ) * data%DV( l ) ) / ( data%X( i ) -       &
                 data%X_l( i ) ), nu_1 * MIN( one, mu / ( data%X_trial( i ) -  &
                 data%X_l( i ) ) ) )
             END IF                                                          
             IF ( data%X_u( i ) <   control%infinity ) THEN
               data%Z_u( i ) = MAX( mult_min,                                  &
                 ( 1 - alpha_z ) * data%Z_u( i ) + alpha_z *                   &
                 ( mu + data%Z_u( i ) * data%DV( l ) ) / ( data%X_u( i ) -     &
                 data%X( i ) ), nu_1 * MIN( one, mu / ( data%X_u( i ) -        &
                 data%X_trial( i ) ) ) )
             END IF                                                          
           END DO                                                            
                                                                             
!  Multipliers for general constraints

           DO i = 1, m                                                       
             jTd = zero                                                      
             DO l = data%prob%A%ptr( i ), data%prob%A%ptr( i + 1 ) - 1       
               j =  data%XSTATE( data%prob%A%col( l ) )
               IF ( j > 0 ) jTd = jTd + data%prob%A%val( l ) * data%DV( j )
             END DO                                                          
             j = data%SSTATE( i )
             IF ( data%C_l( i ) == data%C_u( i ) ) THEN

!  Multipliers for elastic variables for equality constraints

               IF ( ABS( data%S_type( i ) ) /= 1 ) THEN   
                 IF ( data%S_type( i ) /= 0 )                                  &
                   data%U( j ) = MIN( mult_max_eq, MAX( mult_min,              &
                     ( 1 - alpha_u ) * data%U( j ) + alpha_u *                 &
                     ( mu - data%U( j ) * data%DV( nfree + j ) ) / data%S( j ),&
                     nu_1 * MIN( one, mu / data%S_trial( j ) ) ) )
                 IF ( control%bound_elastics )                                 &
                   data%U_u( j ) = MAX( mult_min,                              &
                     ( 1 - alpha_u ) * data%U_u( j ) + alpha_u *               &
                     ( mu - data%U_u( j ) * ( - data%DV( nfree + j ) ) )       &
                     / ( data%S_u( j ) - data%S( j ) ), nu_1 * MIN( one, mu    &
                     / ( data%S_u( j ) - data%S_trial( j ) ) ) )
               END IF                                                          

!  Multipliers for equality constraints

               SELECT CASE ( data%S_type( i ) )
               CASE( 0 )
                   data%Y_l( i ) = MIN( mult_max_eq, MAX( mult_min,            &
                     ( 1 - alpha_y ) * data%Y_l( i ) + alpha_y *               &
                     ( mu - data%Y_l( i ) * ( jTd + data%SCALE_S( j ) *        &
                     data%DV( nfree + j ) ) ) / ( data%C( i ) - data%C_l( i )  &
                     + data%SCALE_S( j ) * data%S( j ) ), nu_1 * MIN( one,     &
                     mu / ( data%C_trial( i ) - data%C_l( i ) +                &
                            data%SCALE_S( j ) * data%S_trial( j ) ) ) ) )
                   data%Y_u( i ) = MIN( mult_max_eq, MAX( mult_min,            &
                     ( 1 - alpha_y ) * data%Y_u( i ) + alpha_y *               &
                     ( mu - data%Y_u( i ) * ( - jTd + data%SCALE_S( j ) *      &
                     data%DV( nfree + j ) ) ) / ( data%C_u( i ) - data%C( i )  &
                     + data%SCALE_S( j ) * data%S( j ) ),  nu_1 * MIN( one,    &
                     mu / ( data%C_u( i ) - data%C_trial( i ) +                &
                     data%SCALE_S( j ) * data%S_trial( j ) ) ) ) )
               CASE( 1 )                                                     
                   data%Y_l( i ) = MAX( mult_min,                              &
                     ( 1 - alpha_y ) * data%Y_l( i ) + alpha_y *               &
                     ( mu - data%Y_l( i ) * jTd ) / ( data%C( i ) -            &
                     data%C_l( i ) ), nu_1 * MIN( one, mu /                    &
                     ( data%C_trial( i ) - data%C_l( i ) ) ) )
               CASE( - 1 )                                                   
                   data%Y_u( i ) = MAX( mult_min,                              &
                     ( 1 - alpha_y ) * data%Y_u( i ) + alpha_y *               &
                     ( mu - data%Y_u( i ) * ( - jTd ) ) / ( data%C_u( i ) -    &
                     data%C( i ) ), nu_1 * MIN( one, mu / ( data%C_u( i ) -    &
                     data%C_trial( i ) ) ) )
               CASE( 2 )                                                     
                   data%Y_l( i ) = MIN( mult_max_eq, MAX( mult_min,            &
                     ( 1 - alpha_y ) * data%Y_l( i ) + alpha_y *               &
                     ( mu - data%Y_l( i ) * ( jTd + data%SCALE_S( j ) *        &
                     data%DV( nfree + j ) ) ) / ( data%C( i ) - data%C_l( i )  &
                     + data%SCALE_S( j ) * data%S( j ) ), nu_1 * MIN( one,     &
                     mu / ( data%C_trial( i ) - data%C_l( i ) +                &
                     data%SCALE_S( j ) * data%S_trial( j ) ) ) ) )
               CASE( - 2 )                                                   
                   data%Y_u( i ) = MIN( mult_max_eq, MAX( mult_min,            &
                     ( 1 - alpha_y ) * data%Y_u( i ) + alpha_y * ( mu -        &
                     data%Y_u( i ) * ( - jTd + data%SCALE_S( j ) *             &
                     data%DV( nfree + j ) ) ) / ( data%C_u( i ) - data%C( i )  &
                     + data%SCALE_S( j ) * data%S( j ) ), nu_1 * MIN( one,     &
                     mu / ( data%C_u( i ) - data%C_trial( i ) +                &
                     data%SCALE_S( j ) * data%S_trial( j ) ) ) ) )
               CASE DEFAULT                                                  
                 GO TO 980
               END SELECT                                                    
             ELSE                                                            

!  Multipliers for elastic variables for inequality constraints

               IF ( ABS( data%S_type( i ) ) /= 1 ) THEN 
                 data%U( j ) = MIN( mult_max, MAX( mult_min,                   &
                   ( 1 - alpha_u ) * data%U( j ) + alpha_u *                   &
                   ( mu - data%U( j ) * data%DV( nfree + j ) ) / data%S( j ),  &
                   nu_1 * MIN( one, mu / data%S_trial( j ) ) ) )
                 IF ( control%bound_elastics )                                 &
                   data%U_u( j ) = MAX( mult_min,                              &
                     ( 1 - alpha_u ) * data%U_u( j ) + alpha_u *               &
                     ( mu - data%U_u( j ) * ( - data%DV( nfree + j ) ) )       &
                     / ( data%S_u( j ) - data%S( j ) ), nu_1 * MIN( one,       &
                     mu / ( data%S_u( j ) - data%S_trial( j ) ) ) )
               END IF

!  Multipliers for inequality constraints

               SELECT CASE ( data%S_type( i ) )
               CASE( 1 )                                                     
                 IF ( data%C_l( i ) > - control%infinity ) THEN
                   data%Y_l( i ) = MAX( mult_min,                              &
                     ( 1 - alpha_y ) * data%Y_l( i ) + alpha_y *               &
                     ( mu - data%Y_l( i ) * ( jTd ) )                          &
                     / ( data%C( i ) - data%C_l( i ) ), nu_1 * MIN( one,       &
                     mu / ( data%C_trial( i ) - data%C_l( i ) ) ) )
                 END IF                                                      
                 IF ( data%C_u( i ) < control%infinity ) THEN
                   data%Y_u( i ) = MAX( mult_min,                              &
                     ( 1 - alpha_y ) * data%Y_u( i ) + alpha_y *               &
                     ( mu - data%Y_u( i ) * ( - jTd ) )                        &
                     / ( data%C_u( i ) - data%C( i ) ),  nu_1 * MIN( one,      &
                     mu / ( data%C_u( i ) - data%C_trial( i ) ) ) )
                 END IF                                                      
               CASE( 2 )                                                     
                 IF ( data%C_l( i ) > - control%infinity ) THEN
                   data%Y_l( i ) = MIN( mult_max, MAX( mult_min,               &
                      ( 1 - alpha_y ) * data%Y_l( i ) + alpha_y *              &
                      ( mu - data%Y_l( i ) * ( jTd + data%SCALE_S( j ) *       &
                      data%DV( nfree + j ) ) ) / ( data%C( i ) -               &
                      data%C_l( i ) + data%SCALE_S( j ) * data%S( j ) ), nu_1  &
                      * MIN( one, mu / ( data%C_trial( i ) - data%C_l( i ) +   &
                      data%SCALE_S( j ) * data%S_trial( j ) ) ) ) )
                 END IF                                                      
                 IF ( data%C_u( i ) <   control%infinity ) THEN
                   data%Y_u( i ) = MIN( mult_max, MAX( mult_min,               &
                     ( 1 - alpha_y ) * data%Y_u( i ) + alpha_y *               &
                     ( mu - data%Y_u( i ) * ( - jTd  + data%SCALE_S( j ) *     &
                     data%DV( nfree + j ) ) ) / ( data%C_u( i ) - data%C( i )  &
                     + data%SCALE_S( j ) * data%S( j ) ), nu_1 * MIN( one,     &
                     mu / ( data%C_u( i ) - data%C_trial( i ) +                &
                     data%SCALE_S( j ) * data%S_trial( j ) ) ) ) )
                 END IF                                                      
!              CASE( 3 )                                                     
!              CASE( - 3 )                                                   
               CASE DEFAULT                                                  
                 GO TO 980
                END SELECT                                                    
             END IF                                                          
           END DO                                                            
                                                                             
         END IF                                                              
                                                                             
!  Update the values of the function and variables                           
                                                                             
         merit = merit_trial ; f = f_trial
         IF ( control%scale_f == 0 ) THEN
           inform%obj = f        
         ELSE
           CALL PTRANS_s_untrans( data%ptrans_transform%f_scale,               &
                                  data%ptrans_transform%f_shift, f, inform%obj )
         END IF
         data%X = data%X_trial ; data%C = data%C_trial 
         data%S( : nelastic ) = data%S_trial( : nelastic )
         new_derivatives = .TRUE.

!  ----------------------------------------------------------------------------
!                       TRUST-REGION RADIUS UPDATES
!  ----------------------------------------------------------------------------

         old_radius = inform%radius

!  If ratio >= rho_very_successful, possibly increase radius

         IF ( got_ratio .AND. ratio >= control%rho_very_successful ) THEN
           inform%radius = MIN( control%maximum_radius, inform%radius *        &
             MIN( control%radius_increase_factor,                              &
                 half * ( step_max + one ) ), MAX( inform%radius, two * step ) )

!  If rho_very_successful > ratio >= rho_successful, replace radius by 
!  something in [gamma_2 radius, radius] 

         ELSE IF ( got_ratio .AND. ratio >= control%rho_successful ) THEN
           inform%radius = MIN(control%maximum_radius,                         &
             inform%radius * MIN( control%radius_small_increase_factor,        &
               half * ( step_max + one ) ), MAX( inform%radius, two * step ) )

!  If rho_successful > ratio, replace radius by something
!  in [gamma_1 radius, gamma_2 radius] 

         ELSE
           delta = one
           DO
             delta = control%radius_decrease_factor * delta
             IF ( delta < alpha ) EXIT
           END DO
           inform%radius = delta * inform%radius
         END IF

!  ----------------------------------------------------------------------------
!                               MAGICAL STEP
!  ----------------------------------------------------------------------------

         IF ( control%magical_steps .AND. nelastic > 0 ) THEN

!  If required, print the distances to the constraint boundaries

           IF ( printd ) THEN
             DO i = 1, m
               j = data%SSTATE( i )
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                   data%C_feas( i ) =                                          &
                     MIN( data%C( i ) - data%C_l( i ) + data%SCALE_S( j ) *    &
                            data%S( j ),                                       &
                          data%C_u( i ) - data%C( i ) + data%SCALE_S( j ) *    &
                            data%S( j ) )
                 CASE( 1 )
                   data%C_feas( i ) = data%C( i ) - data%C_l( i )
                 CASE( - 1 )
                   data%C_feas( i ) = data%C_u( i ) - data%C( i )
                 CASE( 2 )
                   data%C_feas( i ) = data%C( i ) - data%C_l( i ) +            &
                     data%SCALE_S( j ) * data%S( j )
                 CASE( - 2 )
                   data%C_feas( i ) = data%C_u( i ) - data%C( i ) +            &
                     data%SCALE_S( j ) * data%S( j )
                 END SELECT
               ELSE
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 1 )
                   IF ( data%C_l( i ) > - infinity ) THEN
                     IF ( data%C_u( i ) <   infinity ) THEN
                       data%C_feas( i ) = MIN( data%C( i ) - data%C_l( i ),    &
                                          data%C_u( i ) - data%C( i ) )
                     ELSE
                       data%C_feas( i ) = data%C( i ) - data%C_l( i )
                     END IF
                   ELSE
                     data%C_feas( i ) = data%C_u( i ) - data%C( i )
                   END IF
                 CASE( 2 )
                   IF ( data%C_l( i ) > - infinity ) THEN
                     IF ( data%C_u( i ) <   infinity ) THEN
                       data%C_feas( i ) =                                      &
                         MIN( data%C( i ) - data%C_l( i ) +                    &
                                data%SCALE_S( j ) * data%S( j ),               &
                              data%C_u( i ) - data%C( i ) +                    &
                                data%SCALE_S( j ) * data%S( j ) )
                     ELSE
                       data%C_feas( i ) = data%C( i ) - data%C_l( i ) +        &
                         data%SCALE_S( j ) * data%S( j )
                     END IF
                   ELSE
                     data%C_feas( i ) = data%C_u( i ) - data%C( i ) +          &
                       data%SCALE_S( j ) * data%S( j )
                   END IF
                 END SELECT
               END IF
             END DO
!            WRITE( out, "( ' Before magic step ... ' )" ) 
             WRITE( out, "( ' c ', /, ( 5ES12.4 ) )" )  data%C_feas( : m )
!            IF ( nelastic > 0 ) WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )     &
!              data%S( : nelastic )
           END IF

           IF ( printw ) WRITE( out,                                           &
                "( ' .......... magical step computation ............. ' )" )

           CALL SUPERB_magical( m, nelastic, data%S_type, data%SSTATE, mu, nu, &
              data%C_l, data%C_u, data%C, data%S, data%SCALE_S, data%S_u,      &
              len_s_u, control, invalid )
           IF ( invalid ) GO TO 980

           delta = merit
           merit = SUPERB_merit( n, m, nfree, nelastic, data%XFREE,            &
              data%S_type, data%SSTATE, f, mu, nu, data%X, data%X_l, data%X_u, &
              data%C, data%C_l, data%C_u, data%S, data%SCALE_S, data%S_u,      &
              len_s_u, inform%pr_feas, barrier, penalty, merit_error,          &
              print_level, out, control )
           IF ( merit_error == - 99 ) GO TO 980
!          write(6,"( ' merit ', ES12.4 )" ) delta - merit
         END IF  
         zeta = zeta_tol

         IF ( printd ) THEN
           WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  data%X( : n )
           IF ( nelastic > 0 )                                                &
             WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )  data%S( : nelastic )
         END IF

!  Check to see if the primal infeasibility is continuing to grow

!        IF ( inform%pr_feas > control%max_pr_feas_growth * pr_feas_old ) THEN
         IF ( inform%pr_feas > MAX( control%max_pr_feas_growth * pr_feas_old,  &
                                    theta_p ) ) THEN
           n_pr_feas_increase = n_pr_feas_increase + 1
           IF (  n_pr_feas_increase > control%n_pr_feas_increase_max ) THEN
             n_pr_feas_increase = 0
             stop_inner_status = 6 ; GO TO 810
           END IF
         ELSE
           IF ( inform%pr_feas > control%max_pr_feas_growth * pr_feas_old ) THEN
             n_pr_feas_increase = n_pr_feas_increase + 1
           ELSE
             n_pr_feas_increase = 0
           END IF
         END IF

         mo = new_mo
         new_inner = .FALSE.
         IF ( printw ) WRITE( out,                                             &
              "( ' .............. end of iteration .............. ' )" )
       END DO inner

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    I N N E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 800   CONTINUE
       IF ( printi ) THEN
         CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total
         IF ( inform%iter > 0 ) THEN
           IF ( print_level > 1 .OR. ( new_inner .AND.                         &
                stop_inner_status == 0 ) ) WRITE( out, 2030 )
!          IF ( stop_inner_status == 1 .OR. stop_inner_status == 2 .OR.        &
!               new_inner ) THEN
!          IF ( new_inner ) THEN
           IF ( new_inner .OR. stop_inner_status == 2 ) THEN
             WRITE( out, "( I6, ES12.4, 2ES8.1, '    -   ', ES8.1,             &
            &   '     -          -   -', F8.1 )" ) inform%iter, merit,         &
               inform%pr_feas, inform%du_feas, inform%radius, inform%time%total
           ELSE
             WRITE( out, "( I6, ES12.4, 4ES8.1, ES9.1, A1, I7, A1, I3, F8.1 )")&
               inform%iter, merit, inform%pr_feas, inform%du_feas,             &
               step, old_radius, ratio, restrict, cg_iter, mo, nbacts,         &
               inform%time%total
           END IF
         ELSE
           WRITE( out, 2030 )
           WRITE( out, "( I6, ES12.4, ES8.1, '    -       -   ', ES8.1,        &
          &     '     -          -   -', F8.1 )" ) inform%iter, merit,         &
            inform%pr_feas, inform%radius, inform%time%total
         END IF
       END IF

!  Indicate why the inner iteration terminated

 810   CONTINUE
       IF ( printd ) THEN
         WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  data%X( : n )
         IF ( nelastic > 0 )                                                   &
           WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )  data%S( : nelastic )
       END IF
       IF ( printi ) THEN
         WRITE( out, 2080 ) stop_inner_status
         SELECT CASE ( stop_inner_status )
         CASE ( 0 )
           WRITE( out, "( '   stopping based on gradient of Lagrangian ' )" )
           WRITE( out, 2090 )                                                  &
             inform%du_feas, theta_d, inform%comp_slack, theta_c
         CASE ( 1 )
           WRITE( out, "( '   stopping based on gradient of barrier function')")
           WRITE( out, 2090 )                                                  &
             inform%du_feas, theta_d, inform%comp_slack, theta_c
         CASE ( 2 )
           WRITE( out, "( '   stopping based on preconditioned gradient of',   &
          &               ' barrier function ' )" )
           WRITE( out, 2090 )                                                  &
             inform%du_feas, theta_d, inform%comp_slack, theta_c
         CASE ( 3 )
           WRITE( out, "( '   step too small' )" )
         CASE ( 4 )
           WRITE( out, "( '   step too small following linesearch' )" )
         CASE (  5 )
           WRITE( out, "( '   actual and predicted reductions both zero ' )" )
         CASE ( 6 )
           WRITE( out, "( '   primal infeasibility growing significantly ' )" )
         CASE DEFAULT
           WRITE( out, "( '   ???? why stop inner ???? ' )" )
         END SELECT
       END IF

!  Change the penalty or barrier parameter if not optimal

!      WRITE( out, "( ' y_norm ', ES12.4 )" )                                  &
!           MAX( MAXVAL( ABS( Y ) ), MAXVAL( ABS( data%U ) ) )

!  Compute the value of the current violation of (scaled) complementarity

       inform%comp_slack = SUPERB_comp_slack( n, m, nfree, nelastic,data%XFREE,&
         data%S_type, data%SSTATE, zero, data%X, data%X_l, data%X_u, data%C,   &
         data%C_l, data%C_u, data%S, data%SCALE_S, data%S_u, len_s_u,          &
         data%Z_l, data%Z_u, data%U, data%U_u, data%Y_l, data%Y_u, control,    &
         invalid )
       IF ( invalid ) GO TO 980

       IF ( inform%pr_feas > control%stop_p .OR.                               &
            inform%du_feas > control%stop_d .OR.                               &
            ( inform%comp_slack > control%stop_c .AND.                         &
              inform%status /= - 9 ) ) THEN
!        IF ( inform%du_feas > control%stop_d ) THEN
!          mu = control%barrier_decrease_factor * mu
!          IF ( inform%pr_feas > two * ( mu / old_mu ) * violation )           &
!            nu = control%penalty_increase_factor * nu
!        ELSE
!          mu = control%barrier_decrease_factor * mu
!        END IF
!        violation = inform%pr_feas

!        IF ( printi ) WRITE( out, "( /, ' objective value ', 24X, ' = ',      &
!       &  ES16.8, /, ' infeasibility           (actual, target) = ', 2ES12.4, &
!       &   /, ' dual feasibility        (actual, target) = ', 2ES12.4,        &
!       &   /, ' complementary slackness (actual, target) = ', 2ES12.4 )" )    &
         IF ( printi ) WRITE( out, "( /,                                       &
        &      ' objective value         =', ES16.8,                           &
        &   /, ' infeasibility           =', ES12.4,                           &
        &      ' (actual),', ES12.4, ' (target)',                              &
        &   /, ' dual feasibility        =', ES12.4,                           &
        &      ' (actual),', ES12.4, ' (target)',                              &
        &   /, ' complementary slackness =', ES12.4,                           &
        &      ' (actual),', ES12.4, ' (target)' )" )                          &
 
             inform%obj, inform%pr_feas, control%stop_p, inform%du_feas,       &
             control%stop_d, inform%comp_slack, control%stop_c
!        write(6,*) control%get_feasible_first,inform%pr_feas, control%stop_p, &
!             theta_p

!  Increase the penalty parameter if not feasible

         IF ( ( control%get_feasible_first .AND.                               &
                inform%pr_feas > control%stop_p ) .OR.                         &
                inform%pr_feas > theta_p ) THEN
!             ( inform%pr_feas > theta_p .AND. .NOT. new_inner ) ) THEN
           old_nu = nu
           nu = control%penalty_increase_factor * nu

!  If the penalty parameter exceeds its maximum permitted value,
!  check to see if this is because the problem is likely infeasible

           IF ( nu > control%maximum_nu ) THEN
             IF (  inform%comp_slack / old_nu < control%stop_c .AND.           &
                   grad_x / old_nu < control%stop_d ) THEN
               IF ( printi ) WRITE( out,                                       &
              &   "( /, ' ** Stopping at suspected critical point',            &
              &   ' of infeasibility ' )" )
               inform%du_feas = grad_x / old_nu
               inform%comp_slack = inform%comp_slack / old_nu
               inform%status = GALAHAD_error_primal_infeasible ; GO TO 905
             ELSE
               IF ( printi ) WRITE( out,                                       &
              &   "( /, ' ** Stopping because penalty paramter too large ' )" )
               inform%status = GALAHAD_error_primal_infeasible ; GO TO 990
             END IF
           END IF

           mu = control%initial_mu
!          IF ( mu < point1 ) mu = ten * mu
!          IF ( inform%pr_feas > theta_p ) THEN
!            theta_c = SQRT( point1 ) * theta_c
!            theta_d = SQRT( point1 ) * theta_d
!            theta_c = SQRT( point1 ) * inform%comp_slack
!            theta_d = SQRT( point1 ) * inform%du_feas
!          END IF
           IF ( use_primal_dual ) THEN
             DO i = 1, m
               IF ( data%C_l( i ) == data%C_u( i ) ) THEN
                 SELECT CASE ( data%S_type( i ) )
                 CASE( 0 )
                 CASE( 1, 2 )
                   data%Y_l( i ) = data%Y_l( i ) + ( nu - old_nu )
                 CASE( -1, - 2 )
                   data%Y_u( i ) = data%Y_u( i ) + ( nu - old_nu )
                 END SELECT
               END IF
             END DO
           END IF
           theta_d = MIN( control%max_stop_d,                                  &
                          MAX( control%stop_d_factor * mu ** beta,             &
                               point99 * control%stop_d ) )
           theta_c = MIN( control%max_stop_c,                                  &
                          MAX( control%stop_c_factor * mu ** beta,             &
                               point99 * control%stop_c ) )

!  Reduce the barrier parameter if not optimal

         ELSE
!          inform%comp_slack = SUPERB_comp_slack( n, m, nfree, nelastic,       &
!            data%XFREE, data%S_type, data%SSTATE, mu, data%X, data%X_l,       &
!            data%X_u, data%C, data%C_l, data%C_u, data%S, data%SCALE_S,       &
!            data%S_u, len_s_u, data%Z_l, data%Z_u, data%U, data%U_u,          &
!            data%Y_l, data%Y_u, control, invalid )
!          IF ( invalid ) GO TO 980
!          write(6,*) 'cs-', inform%comp_slack
           DO 
             old_mu = mu
             IF ( control%superlinear_decrease ) THEN
               mu = MIN( control%barrier_decrease_factor * mu, mu ** 1.5 )
             ELSE
               mu = control%barrier_decrease_factor * mu
             END IF
             IF ( mu < control%minimum_mu ) THEN
               IF ( printi ) WRITE( out,                                       &
              &   "( /, ' ** Stopping because barrier paramter too small ' )" )
               inform%status = GALAHAD_error_primal_infeasible ; GO TO 990
             END IF
             zeta = MIN( zeta_tol, point1 * mu / old_mu )

             theta_p = MAX( control%stop_p, mu * MIN( one, mu ),               &
                                  inform%pr_feas * SQRT( MIN( point1,          &
                                  control%barrier_decrease_factor ) ) )
!            theta_p = MAX( mu, inform%pr_feas * SQRT( MIN( point1,            &
!                                 control%barrier_decrease_factor ) ) )
!            theta_p = inform%pr_feas *                                        &
!              SQRT( MIN( point1, control%barrier_decrease_factor ) )
!            theta_p = theta_p *                                   &
!              SQRT( MIN( point1, control%barrier_decrease_factor ) )

!  Compute the value of the current violation of (scaled) complementarity

             inform%comp_slack = SUPERB_comp_slack( n, m, nfree, nelastic,     &
               data%XFREE, data%S_type, data%SSTATE, mu, data%X, data%X_l,     &
               data%X_u, data%C, data%C_l, data%C_u, data%S, data%SCALE_S,     &
               data%S_u, len_s_u, data%Z_l, data%Z_u, data%U, data%U_u,        &
               data%Y_l, data%Y_u, control, invalid )
             IF ( invalid ) GO TO 980

!  Update convergence tolerances, theta_d and theta_c

             theta_d = MIN( control%max_stop_d,                                &
                            MAX( control%stop_d_factor * mu ** beta,           &
                                 point99 * control%stop_d ) )
             theta_c = MIN( control%max_stop_c,                                &
                            MAX( control%stop_c_factor * mu ** beta,           &
                                 point99 * control%stop_c ) )
             IF ( inform%du_feas > theta_d .OR.                                &
                  inform%comp_slack > theta_c .OR.                             &
                  inform%pr_feas > theta_p ) EXIT
           END DO
         END IF

!  Re-evaluate the value of the merit function to account for changes to
!  the barrier and penalty parameters

         merit = f + mu * barrier + nu * penalty

!  If desired, try to reduce the number of elastic variables

         IF ( nelastic > 0 .AND. ( control%elastic_type_equations < 0 .OR.     &
              control%elastic_type_inequalities < 0 ) ) THEN

!  Record the indices of constraints that have elastic variables

           nelastic_old = nelastic ; nelastic = 0
           DO j = 1, nelastic_old
             i = data%ELASTICS( j )

             IF ( data%C_l( i ) == data%C_u( i ) .AND.                         &
                  control%elastic_type_equations < 0 ) THEN
               IF ( data%S_type( i ) == 2 .AND. data%C( i ) >=                 &
                 data%C_l( i ) + prfeas ) data%S_type( i ) = 1
               IF ( data%S_type( i ) == - 2 .AND. data%C( i ) <=               &
                 data%C_u( i ) - prfeas ) data%S_type( i ) = - 1
             ELSE IF ( data%C_l( i ) < data%C_u( i ) .AND.                     &
                       control%elastic_type_inequalities < 0) THEN
               IF ( data%S_type( i ) == 2 ) THEN
                 IF ( data%C_l( i ) > - control%infinity ) THEN
                   IF ( data%C_u( i ) < control%infinity ) THEN
                     IF ( data%C( i ) >= data%C_l( i ) + prfeas .AND.          &
                          data%C( i ) <= data%C_u( i ) - prfeas )              &
                       data%S_type( i ) = 1
                   ELSE
                     IF ( data%C( i ) >= data%C_l( i ) + prfeas )              &
                       data%S_type( i ) = 1
                   END IF
                 ELSE IF ( data%C_u( i ) < control%infinity ) THEN
                   IF ( data%C( i ) <= data%C_u( i ) - prfeas )                &
                     data%S_type( i ) = 1
                 END IF
               END IF
             END IF

             IF ( ABS( data%S_type( i ) ) /= 1 ) THEN
               nelastic = nelastic + 1
               data%SSTATE( i ) = nelastic
               data%ELASTICS( nelastic ) = i
               data%S( nelastic ) = data%S( j )
               data%SCALE_S( nelastic ) = data%SCALE_S( j )
               IF ( data%S_type( i ) /= 0 ) THEN
                 data%U_P( nelastic ) = data%U_P( j )
                 data%U( nelastic ) = data%U( j )
               END IF
               IF ( control%bound_elastics ) THEN
                 data%S_u( nelastic ) = data%S_u( j )
                 data%U_u_P( nelastic ) = data%U_u_P( j )
                 data%U_u( nelastic ) = data%U_u( j )
               END IF
             ELSE
               data%SSTATE( i ) = 0
             END IF
           END DO

           nfrpel = nfree + nelastic

!  If necessary, re-evaluate the merit function

           IF ( nelastic /= nelastic_old ) THEN
             IF ( printi ) WRITE( out,                                         &
          &    "( /, ' ** Number of elastics reduced by ', I7, /, '    Now ',  &
          &       I7, ' out of ', I7, ' constraints have elastics ')" )        &
               nelastic_old - nelastic, nelastic, m    
             merit = SUPERB_merit( n, m, nfree, nelastic, data%XFREE,          &
               data%S_type, data%SSTATE, f, mu, nu, data%X, data%X_l,          &
               data%X_u, data%C, data%C_l, data%C_u, data%S, data%SCALE_S,     &
               data%S_u, len_s_u, inform%pr_feas, barrier, penalty,            &
               merit_error, print_level, out, control )
             IF ( merit_error == - 99 ) GO TO 980
             analyse = .TRUE.
             CALL GLTR_terminate( data%gltr_data, control%gltr_control,        &
                                  inform%gltr_inform )

           END IF
         END IF

         IF ( control%magical_steps .AND. nelastic > 0 ) THEN

!  ----------------------------------------------------------------------------
!                               MAGICAL STEP
!  ----------------------------------------------------------------------------

           IF ( printw ) WRITE( out,                                           &
                "( ' .......... magical step computation ............. ' )" )

           CALL SUPERB_magical( m, nelastic, data%S_type, data%SSTATE, mu, nu, &
             data%C_l, data%C_u, data%C, data%S, data%SCALE_S, data%S_u,       &
             len_s_u, control, invalid )
           IF ( invalid ) GO TO 980

           delta = merit
           merit = SUPERB_merit( n, m, nfree, nelastic, data%XFREE,            &
             data%S_type, data%SSTATE, f, mu, nu, data%X, data%X_l, data%X_u,  &
             data%C, data%C_l, data%C_u, data%S, data%SCALE_S, data%S_u,       &
             len_s_u, inform%pr_feas, barrier, penalty, merit_error,           &
             print_level, out, control )
           IF ( merit_error == - 99 ) GO TO 980
!          write(6,"( ' merit ', ES12.4 )" ) delta - merit
         END IF  

         IF ( printi ) THEN
           WRITE( out, 2050 ) mu, nu, theta_d, theta_c
!          IF ( inform%iter > 0 ) WRITE( out, 2030 )
         END IF
         new_inner = .TRUE.
       ELSE
         EXIT
       END IF

     END DO

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    O U T E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Normal return

     inform%status = GALAHAD_ok

 905 CONTINUE

!  Set the multiplier estimates

     DO i = 1, n
       g_term = zero
       IF ( data%X_l( i ) > - control%infinity )                               &
         g_term = g_term + data%Z_l( i )
       IF ( data%X_u( i ) <   control%infinity )                               &
         g_term = g_term - data%Z_u( i )
       data%Z( i ) = g_term
     END DO

     DO i = 1, m
!      IF ( ABS( data%S_type( i ) ) /= 1 ) data%U_P( i ) = data%U( i )
       IF ( data%C_l( i ) == data%C_u( i ) ) THEN
         SELECT CASE ( data%S_type( i ) )
         CASE( 0 )
           data%LAMBDA( i ) = - ( data%Y_l( i ) - data%Y_u( i ) )
         CASE( 1, 2 )
           data%LAMBDA( i ) = - ( data%Y_l( i ) - nu )
         CASE( - 1, - 2 )
           data%LAMBDA( i ) = data%Y_u( i ) - nu
         CASE DEFAULT
           GO TO 980
         END SELECT
       ELSE
         g_term = zero
         SELECT CASE ( data%S_type( i ) )
         CASE( 1, 2 )
           IF ( data%C_l( i ) > - control%infinity )                           &
             g_term = g_term + data%Y_l( i )
           IF ( data%C_u( i ) <   control%infinity )                           &
             g_term = g_term - data%Y_u( i )
!        CASE( 3 )
!        CASE( - 3 )
         CASE DEFAULT
           GO TO 980
         END SELECT
         data%LAMBDA( i ) = g_term
       END IF
     END DO

!  If the point is infeasible, scale the multipliers

     IF ( inform%status == GALAHAD_error_primal_infeasible ) THEN
       data%Z( : n ) = data%Z( : n ) / old_nu
       data%LAMBDA( : m ) = data%LAMBDA( : m ) / old_nu
     END IF

!  If the problem has been scaled, unscale it

     IF ( scale_xcf ) THEN
       CALL PTRANS_untrans( n, m, data%ptrans_transform, control%infinity,     &
                            f = f, X = data%X, X_l = data%X_l, X_u = data%X_u, &
                            Z_l = data%Z_l, Z_u = data%Z_u, C = data%C,        &
                            C_l = data%C_l, C_u = data%C_u,                    &
                            Y_l = data%Y_l, Y_u = data%Y_u, V_m = data%LAMBDA )
     END IF

!  Print the solution

     l = 4
     IF ( control%fulsol ) l = n 
     IF ( control%print_level >= 10 ) l = n

     WRITE( out, 2000 )
     DO j = 1, 2 
       IF ( j == 1 ) THEN 
         ir = 1 ; ic = MIN( l, n ) 
       ELSE 
         IF ( ic < n - l ) WRITE( out, 2040 ) 
         ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
       END IF 
       DO i = ir, ic 
         WRITE( out, 2020 ) i, data%X_name( i ), data%X( i ), data%X_l( i ),   &
           data%X_u( i ), data%Z( i )
       END DO
     END DO

     IF ( m > 0 ) THEN
       l = 4
       IF ( control%fulsol ) l = m
       IF ( control%print_level >= 10 ) l = m

       WRITE( out, 2010 )
       DO j = 1, 2 
         IF ( j == 1 ) THEN 
           ir = 1 ; ic = MIN( l, m ) 
         ELSE 
           IF ( ic < m - l ) WRITE( out, 2040 ) 
           ir = MAX( ic + 1, m - ic + 1 ) ; ic = m
         END IF 
         DO i = ir, ic 
           WRITE( out, 2020 ) i, data%C_name( i ), data%C( i ), data%C_l( i ), &
             data%C_u( i ), data%LAMBDA( i )
         END DO
       END DO
     END IF

     CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total
     inform%obj = f

     WRITE( out, "( /, ' Problem: ', 16X, A10,                                 &
    &          '     Solver: ', 15X, ' SUPERB', /,                             &
    &  ' n              =     ',bn, I12, '       m               = ',bn, I12,/,&
    &  ' Objective      = ', ES16.8, '       Complementarity = ', ES12.4, /,   &
    &  ' Violation      =     ',ES12.4, '       Dual infeas     = ', ES12.4, /,&
    &  ' Max multiplier =     ',ES12.4, '       Max dual var.   = ', ES12.4, /,&
    &  ' Iterations     =     ',bn, I12, '       Time            = ', F12.2 )")&
      inform%pname, n, m, inform%obj, inform%comp_slack, inform%pr_feas,       &
      inform%du_feas, MAXVAL( ABS( data%LAMBDA ) ), MAXVAL( ABS( data%Z ) ),   &
      inform%iter, inform%time%total
     IF ( nmhist > 0 ) WRITE( out,                                             &
       "( ' Non-monotone descent strategy ( history =', I3,                    &
      &     ' ) used ' )" ) nmhist
     IF ( control%model == 1 ) THEN
       WRITE( out, "( ' Newton model used ' )" )
     ELSE IF ( control%model == 2 ) THEN
       WRITE( out, "( ' Linear model used ' )" )
     ELSE
       WRITE( out, "( ' Linear plus model used ' )" )
     END IF
     IF ( use_primal_dual ) THEN
       WRITE( out, "( ' Primal-dual updates used ' )" )
     ELSE
       WRITE( out, "( ' Primal updates used ' )" )
     END IF
     WRITE( out, "( '' )" )

     IF ( inform%status /= GALAHAD_ok ) THEN
       WRITE( control%error, "( ' ** Message from -SUPERB_solve- ',            &
      &  '    Error exit (status = ', I6, ')', / )" ) inform%status
     END IF
     CALL CUTEST_cterminate( cutest_status )

     RETURN

!  Allocation or deallocation errors

 910 CONTINUE

!  Compute total time

 920 CONTINUE
     CALL CPU_TIME( time_new ); inform%time%total = time_new - time_total
     RETURN

!  CUTEst errors

 930 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")           &
       cutest_status
     inform%status = GALAHAD_error_cutest
     RETURN

!  Other errors

 980 CONTINUE
     IF ( control%error > 0 ) WRITE( control%error,                           &
       "( ' error return from SUPERB: invalid default case for data%S_type' )" )
     inform%status = GALAHAD_error_technical
     RETURN

!  Other errors

 990 CONTINUE
     CALL CPU_TIME( time_new ); inform%time%total = time_new - inform%time%total
     inform%obj = f
     WRITE( control%error, "( /, ' ** Message from -SUPERB_solve-',            &
    &  '    Error exit (status = ', I6, ')', / )" ) inform%status
     RETURN

!  Non-executable statements

 2000 FORMAT( /,' Solution: ', /,'                        ',                   &
                '        <------ Bounds ------> ', /                           &
                '      # name          value   ',                              &
                '    Lower       Upper       Dual ' ) 
 2010 FORMAT( /,' Constraints: ', /, '                        ',               &
                '        <------ Bounds ------> ', /                           &
                '      # name           value   ',                             &
                '    Lower       Upper    Multiplier ' ) 
 2020 FORMAT( I7, 1X, A10, 4ES12.4 ) 
 2030 FORMAT( /, '  iter   merit fun pr_feas du_feas  step ',                  &
                 '  radius ared/pred  cg its  bt     CPU')
 2040 FORMAT( 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2050 FORMAT( /, 1X, 9( '-=' ), '     mu = ', ES10.2, ' nu =     ', ES10.2,    &
                 1X, 9( '=-' ),                                                &
              /, 9X, '  required stop_d = ', ES10.2, ' stop_c = ', ES10.2 )
 2060 FORMAT( '   **  Warning ', I6, ' from ', A15 ) 
 2070 FORMAT( '   ==>  increasing pivot tolerance to ', ES12.4 )
 2080 FORMAT( /, '  End of inner iteration (status = ', I1, '): ' )
 2090 FORMAT( '   dual feasibility ', ES10.4, ' smaller than required ',       &
              ES10.4, /, '   compl. slackness ', ES10.4,                       &
              ' smaller than required ', ES10.4 )
 2100 FORMAT( '   **  Error ', I6, ' from ', A15 ) 

!  End of subroutine SUPERB_solve

     END SUBROUTINE SUPERB_solve

!-*-*-*-*  G A L A H A D -  SUPERB_terminate  S U B R O U T I N E -*-*-*-*

     SUBROUTINE SUPERB_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SUPERB_data_type ), INTENT( INOUT ) :: data
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     TYPE ( SUPERB_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate arrays set for GLTR

     CALL GLTR_terminate( data%gltr_data, control%gltr_control,                &
                          inform%gltr_inform )
     IF ( inform%gltr_inform%status == GALAHAD_error_deallocate ) THEN
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = inform%gltr_inform%alloc_status
       inform%bad_alloc = inform%gltr_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate arrays set for PTRANS

     IF ( control%scale_x > 0 .OR. control%scale_c > 0 .OR.                    &
          control%scale_f > 0 ) THEN
       CALL PTRANS_terminate( data%ptrans_transform, data%ptrans_data,         &
                              inform%ptrans_inform )
       IF ( inform%ptrans_inform%status /= 0 ) THEN
         inform%status = inform%ptrans_inform%status 
         inform%alloc_status = inform%ptrans_inform%alloc_status
         inform%bad_alloc = inform%ptrans_inform%bad_alloc
         IF ( control%deallocate_error_fatal ) RETURN
       END IF
     END IF

!  Deallocate all remaining allocated arrays

     array_name = 'superb: data%X'
     CALL SPACE_dealloc_array( data%X,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%X_l'
     CALL SPACE_dealloc_array( data%X_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%X_u'
     CALL SPACE_dealloc_array( data%X_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%LAMBDA'
     CALL SPACE_dealloc_array( data%LAMBDA,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Z'
     CALL SPACE_dealloc_array( data%Z,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%C_l'
     CALL SPACE_dealloc_array( data%C_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%C_u'
     CALL SPACE_dealloc_array( data%C_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%C_feas'
     CALL SPACE_dealloc_array( data%C_feas,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%XSTATE'
     CALL SPACE_dealloc_array( data%XSTATE,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%EQUATN'
     CALL SPACE_dealloc_array( data%EQUATN,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%LINEAR'
     CALL SPACE_dealloc_array( data%LINEAR,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%EQUATN'
     CALL SPACE_dealloc_array( data%EQUATN,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%XFREE'
     CALL SPACE_dealloc_array( data%XFREE,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%C_name'
     CALL SPACE_dealloc_array( data%C_name,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%X_name'
     CALL SPACE_dealloc_array( data%X_name,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%X_trial'
     CALL SPACE_dealloc_array( data%X_trial,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%C'
     CALL SPACE_dealloc_array( data%C,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%C_trial'
     CALL SPACE_dealloc_array( data%C_trial,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%G'
     CALL SPACE_dealloc_array( data%prob%G,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%C'
     CALL SPACE_dealloc_array( data%prob%C,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%A%row'
     CALL SPACE_dealloc_array( data%prob%A%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%A%col'
     CALL SPACE_dealloc_array( data%prob%A%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%A%val'
     CALL SPACE_dealloc_array( data%prob%A%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%H%row'
     CALL SPACE_dealloc_array( data%prob%H%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%H%col'
     CALL SPACE_dealloc_array( data%prob%H%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%H%val'
     CALL SPACE_dealloc_array( data%prob%H%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%IW'
     CALL SPACE_dealloc_array( data%IW,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%S_type'
     CALL SPACE_dealloc_array( data%S_type,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%SSTATE'
     CALL SPACE_dealloc_array( data%SSTATE,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Y_l'
     CALL SPACE_dealloc_array( data%Y_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Y_u'
     CALL SPACE_dealloc_array( data%Y_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Z_l'
     CALL SPACE_dealloc_array( data%Z_l,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Z_u'
     CALL SPACE_dealloc_array( data%Z_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%A%ptr'
     CALL SPACE_dealloc_array( data%prob%A%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Y_l_P'
     CALL SPACE_dealloc_array( data%Y_l_P,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Y_u_P'
     CALL SPACE_dealloc_array( data%Y_u_P,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Z_l_P'
     CALL SPACE_dealloc_array( data%Z_l_P,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%Z_u_P'
     CALL SPACE_dealloc_array( data%Z_u_P,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%ELASTICS'
     CALL SPACE_dealloc_array( data%ELASTICS,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%SCALE_S'
     CALL SPACE_dealloc_array( data%SCALE_S,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%U_P'
     CALL SPACE_dealloc_array( data%U_P,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%S_u'
     CALL SPACE_dealloc_array( data%S_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%U_u'
     CALL SPACE_dealloc_array( data%U_u,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%U_u_P'
     CALL SPACE_dealloc_array( data%U_u_P,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%GRAD_b'
     CALL SPACE_dealloc_array( data%GRAD_b,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%GRAD_m'
     CALL SPACE_dealloc_array( data%GRAD_m,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%DV'
     CALL SPACE_dealloc_array( data%DV,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%VECTOR'
     CALL SPACE_dealloc_array( data%VECTOR,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%SOL'
     CALL SPACE_dealloc_array( data%SOL,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%RES'
     CALL SPACE_dealloc_array( data%RES,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%BEST'
     CALL SPACE_dealloc_array( data%BEST,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%S_trial'
     CALL SPACE_dealloc_array( data%S_trial,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%B_X'
     CALL SPACE_dealloc_array( data%B_X,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%B_C'
     CALL SPACE_dealloc_array( data%B_C,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%B_CM'
     CALL SPACE_dealloc_array( data%B_CM,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%B_S'
     CALL SPACE_dealloc_array( data%B_S,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%prob%H%ptr'
     CALL SPACE_dealloc_array( data%prob%H%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%C_stat'
     CALL SPACE_dealloc_array( data%C_stat,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%B_stat'
     CALL SPACE_dealloc_array( data%B_stat,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%K%row'
     CALL SPACE_dealloc_array( data%K%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%K%col'
     CALL SPACE_dealloc_array( data%K%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'superb: data%K%val'
     CALL SPACE_dealloc_array( data%K%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine SUPERB_terminate

     END SUBROUTINE SUPERB_terminate

!-*-*-*-*  G A L A H A D  -  S U P E R B _ m e r i t   F U N C T I O N -*-*-*-*

     FUNCTION SUPERB_merit( n, m, nfree, nelastic, XFREE, S_type, SSTATE, f,   &
                            mu, nu, X, X_l, X_u, C, C_l, C_u, S, SCALE_S,      &
                            S_u, len_s_u, violation, barrier, penalty,         &
                            merit_error, print_level, out, control )
     REAL ( KIND = wp ) :: SUPERB_merit

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute the value of the merit function

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, m, nfree, nelastic, len_s_u, print_level, out
     INTEGER, INTENT( OUT ) :: merit_error
     REAL ( KIND = wp ), INTENT( IN ) :: f, mu, nu
     REAL ( KIND = wp ), INTENT( OUT ) :: violation, barrier, penalty
     INTEGER, INTENT( IN ), DIMENSION( nfree ) :: XFREE
     INTEGER, INTENT( IN ), DIMENSION( m ) :: S_type, SSTATE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, C_l, C_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nelastic ) :: S, SCALE_S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( len_s_u ) :: S_u
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, l
     REAL ( KIND = wp ) :: x_min, s_min, c_min
 
     merit_error = 0
     violation = zero ; barrier = zero ; penalty = zero
     x_min = control%infinity ; s_min = control%infinity 
     c_min = control%infinity

     DO l = 1, nfree
       i = XFREE( l )
       IF ( X_l( i ) > - control%infinity ) THEN
         IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )               &
           write(6,"( 'x_l', ES12.4 )" ) X( i ) - X_l( i )
         IF ( X( i ) <= X_l( i ) ) GO TO 900
         barrier = barrier - LOG( X( i ) - X_l( i ) )
         x_min = MIN( x_min, X( i ) - X_l( i ) )
       END IF
       IF ( X_u( i ) <   control%infinity ) THEN
        IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )                &
           write(6,"( 'x_u', ES12.4 )" ) X_u( i ) - X( i )
         IF ( X( i ) >= X_u( i ) ) GO TO 900
         barrier = barrier - LOG( X_u( i ) - X( i ) )
         x_min = MIN( x_min, X_u( i ) - X( i ) )
       END IF
     END DO

     DO i = 1, m
       j = SSTATE( i )
       IF ( ABS( S_type( i ) ) /= 1 ) THEN
         IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )               &
            write(6,"( 's', ES22.14 )" ) S( j )
         IF ( S_type( i ) /= 0 ) THEN
           IF ( S( j ) <= zero ) GO TO 900
           barrier = barrier - LOG( S( j ) )
           s_min = MIN( s_min, S( j ) )
         END IF
         IF ( control%bound_elastics ) THEN
           IF ( S_u( j ) - S( j ) <= zero ) GO TO 900
           barrier = barrier - LOG( S_u( j ) - S( j ) )
!          write(6,"( ' s, smax ', 2ES12.4 )" ) S( j ), S_u( j )
           s_min = MIN( s_min, S_u( j ) - S( j ) )
         END IF
       END IF
       IF ( C_l( i ) == C_u( i ) ) THEN
         SELECT CASE ( S_type( i ) )
         CASE( 0 )
           penalty = penalty + SCALE_S( j ) * S( j )
           IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )             &
             WRITE(6,"('c', ES22.14 )" ) C( i ) - C_l( i ) + SCALE_S( j ) * S(j)
           IF ( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) <= zero ) GO TO 900
           IF ( control%bound_elastics .AND.                                   &
                C( i ) - C_l( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 910
           barrier = barrier - LOG( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
           violation = MAX( violation, C_l( i ) - C( i ) )
           c_min = MIN( c_min, C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
           IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )             &
             WRITE(6,"('c', ES22.14 )" ) C_u( i ) - C( i ) + SCALE_S( j ) * S(j)
           IF ( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) <= zero ) GO TO 900
           IF ( control%bound_elastics .AND.                                   &
                C_u( i ) - C( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 910
           barrier = barrier - LOG( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
           violation = MAX( violation, C( i ) - C_u( i ) )
           c_min = MIN( c_min, C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
         CASE( 1 )
         IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )               &
           WRITE(6,"('c', ES22.14 )" ) C( i ) - C_l( i )
           IF ( C( i ) - C_l( i ) <= zero ) GO TO 900
           penalty = penalty + C( i ) - C_l( i )
           barrier = barrier - LOG( C( i ) - C_l( i ) )
           violation = MAX( violation, ABS( C( i ) - C_l( i ) ) )
           c_min = MIN( c_min, C( i ) - C_l( i ) )
         CASE( - 1 )
         IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )               &
           WRITE(6,"('c', ES22.14 )" ) C_u( i ) - C( i )
           IF ( C_u( i ) - C( i ) <= zero ) GO TO 900
           penalty = penalty + C_u( i ) - C( i ) 
           barrier = barrier - LOG( C_u( i ) - C( i ) )
           violation = MAX( violation, ABS( C_u( i ) - C( i ) ) )
           c_min = MIN( c_min, C_u( i ) - C( i ) )
         CASE( 2 )
         IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )               &
           WRITE(6,"('c', ES22.14 )" ) C( i ) - C_l( i ) + SCALE_S( j ) * S( j )
           IF ( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) <= zero ) GO TO 900
           IF ( control%bound_elastics .AND.                                   &
                C( i ) - C_l( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 910
           penalty = penalty + C( i ) - C_l( i ) + two * SCALE_S( j ) * S( j )
           barrier = barrier - LOG( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
           violation = MAX( violation, ABS( C( i ) - C_l( i ) ) )
           c_min = MIN( c_min, C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
         CASE( - 2 )
         IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )               &
           WRITE(6,"('c', ES22.14 )" ) C_u( i ) - C( i ) + SCALE_S( j ) * S( j )
           IF ( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) <= zero ) GO TO 900
           IF ( control%bound_elastics .AND.                                   &
                C_u( i ) - C( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 910
           penalty = penalty + C_u( i ) - C( i ) + two * SCALE_S( j ) * S( j )
           barrier = barrier - LOG( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
           violation = MAX( violation, ABS( C_u( i ) - C( i ) ) )
           c_min = MIN( c_min, C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
         CASE DEFAULT
           GO TO 980
         END SELECT
!        write(6,"( 'c_e', ES22.14 )" ) C( i ) - C_l( i ) + SCALE_S( j ) * S(j)
       ELSE
         SELECT CASE ( S_type( i ) )
         CASE( 1 )
           IF ( C_l( i ) > - control%infinity ) THEN
             IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )           &
               WRITE(6,"('c', ES22.14 )" ) C( i ) - C_l( i )
             IF ( C( i ) - C_l( i ) <= zero ) GO TO 900
             barrier = barrier - LOG( C( i ) - C_l( i ) )
             violation = MAX( violation, C_l( i ) - C( i ) )
             c_min = MIN( c_min, C( i ) - C_l( i ) )
           END IF
           IF ( C_u( i ) <   control%infinity ) THEN
             IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )           &
               WRITE(6,"('c', ES22.14 )" ) C_u( i ) - C( i )
             IF ( C_u( i ) - C( i ) <= zero ) GO TO 900
             barrier = barrier - LOG( C_u( i ) - C( i ) )
             violation = MAX( violation, C( i ) - C_u( i ) )
             c_min = MIN( c_min, C_u( i ) - C( i ) )
           END IF
         CASE( 2 )
           penalty = penalty + SCALE_S( j ) * S( j )
           IF ( C_l( i ) > - control%infinity ) THEN
             IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )           &
               WRITE(6,"('c', ES22.14 )" )                                     &
                 C( i ) - C_l( i ) + SCALE_S( j ) * S( j )
             IF ( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) <= zero ) GO TO 900
             IF ( control%bound_elastics .AND.                                 &
                C( i ) - C_l( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 910
             barrier =                                                         &
               barrier - LOG( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
             violation = MAX( violation, C_l( i ) - C( i ) )
             c_min = MIN( c_min, C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
           END IF
           IF ( C_u( i ) <   control%infinity ) THEN
             IF ( out > 0 .AND. print_debug .AND. print_level >= 1 )           &
               WRITE(6,"('c', ES22.14 )" )                                     &
                 C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) 
             IF ( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) <= zero ) GO TO 900
             IF ( control%bound_elastics .AND.                                 &
                C_u( i ) - C( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 910
             barrier =                                                         &
               barrier - LOG( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
             violation = MAX( violation, C( i ) - C_u( i ) )
             c_min = MIN( c_min, C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
           END IF
!        CASE( 3 )
!        CASE( - 3 )
         CASE DEFAULT
           GO TO 980
           STOP
         END SELECT
       END IF
     END DO
     IF ( out > 0 .AND. print_level >= 3 ) WRITE( out,                         &
       "( ' dist to x, s & c boundaries = ', 3ES12.4 )" ) x_min, s_min, c_min

!    write(6,"(5ES12.4)") f, mu, barrier, nu, penalty
     SUPERB_merit = f + mu * barrier + nu * penalty
     RETURN

!  Error returns

 900 CONTINUE
     merit_error = 1
     SUPERB_merit = HUGE( one )
     RETURN

 910 CONTINUE
     merit_error = 2
     SUPERB_merit = HUGE( one )
     RETURN

!  Other errors

 980 CONTINUE
     merit_error = - 99
     RETURN

!  End of function SUPERB_merit

     END FUNCTION SUPERB_merit

!-*-  G A L A H A D  -  S U P E R B _ c o m p _ s l a c k  F U N C T I O N  -*-

     FUNCTION SUPERB_comp_slack( n, m, nfree, nelastic, XFREE, S_type, SSTATE, &
                                 mu, X, X_l, X_u, C, C_l, C_u, S, SCALE_S,     &
                                 S_u, len_s_u, Z_l, Z_u, U, U_u, Y_l, Y_u,     &
                                 control, invalid )
     REAL ( KIND = wp ) :: SUPERB_comp_slack

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the value of the current violation of (scaled) complementarity

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, m, nfree, nelastic, len_s_u
     REAL ( KIND = wp ), INTENT( IN ) :: mu
     INTEGER, INTENT( IN ), DIMENSION( nfree ) :: XFREE
     INTEGER, INTENT( IN ), DIMENSION( m ) :: S_type, SSTATE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: Z_l, Z_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, C_l, C_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y_l, Y_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nelastic ) :: S, SCALE_S, U
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( len_s_u ) :: S_u, U_u
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     LOGICAL, INTENT( OUT ) :: invalid

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, l
     REAL ( KIND = wp ) :: scale
!    REAL ( KIND = wp ) :: normalize
     LOGICAL, PARAMETER :: comp_scale = .TRUE.
!    LOGICAL, PARAMETER :: comp_scale = .FALSE.

     SUPERB_comp_slack = zero
!    normalize = one
     IF ( .NOT. comp_scale ) scale = one
     DO l = 1, nfree
       i = XFREE( l )
       IF ( X_l( i ) > - control%infinity ) THEN
         IF ( comp_scale ) scale = MAX( Z_l( i ), X( i ) - X_l( i ) )
         SUPERB_comp_slack = MAX( SUPERB_comp_slack,                           &
           ABS( Z_l( i ) * ( X( i ) - X_l( i ) ) - mu ) / MAX( one, scale ) )
!        normalize = MAX( normalize, scale )
       END IF
       IF ( X_u( i ) <   control%infinity ) THEN
         IF ( comp_scale ) scale = MAX( Z_u( i ), X_u( i ) - X( i ) )
         SUPERB_comp_slack = MAX( SUPERB_comp_slack,                           &
           ABS( Z_u( i ) * ( X_u( i ) - X( i ) ) - mu ) / MAX( one, scale ) )
!        normalize = MAX( normalize, scale )
       END IF
     END DO

     DO i = 1, m
       j = SSTATE( i )
       IF ( ABS( S_type( i ) ) /= 1 ) THEN
         IF ( S_type( i ) /= 0 ) THEN
           IF ( comp_scale ) scale = MAX( U( j ), S( j ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( U( j ) * S( j ) - mu ) / MAX( one,  scale ) )
!          normalize = MAX( normalize, scale )
         END IF
         IF ( control%bound_elastics ) THEN
           IF ( comp_scale ) scale = MAX( U_u( j ), S_u( j ) - S( j ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( U_u( j ) * ( S_u( j ) - S( j ) ) - mu ) / MAX( one, scale ) )
!          normalize = MAX( normalize, scale )
         END IF
       END IF
       IF ( C_l( i ) == C_u( i ) ) THEN
         SELECT CASE ( S_type( i ) )
         CASE( 0 )
           IF ( comp_scale )                                                   &
             scale = MAX( Y_l( i ), C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( Y_l( i ) * ( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )     &
                  - mu ) / MAX( one, scale ) )
!          normalize = MAX( normalize, scale )
           IF ( comp_scale )                                                   &
             scale = MAX( Y_u( i ), C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( Y_u( i ) * ( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )     &
                  - mu ) / MAX( one, scale ) )
!          normalize = MAX( normalize, scale )
         CASE( 1 )
           IF ( comp_scale ) scale = MAX( Y_l( i ), C( i ) - C_l( i ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( Y_l( i ) * ( C( i ) - C_l( i ) ) - mu ) / MAX( one, scale ) )
!          normalize = MAX( normalize, scale )
         CASE( - 1 )
           IF ( comp_scale ) scale = MAX( Y_u( i ), C_u( i ) - C( i ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( Y_u( i ) * ( C_u( i ) - C( i ) ) - mu ) / MAX( one, scale ) )
!          normalize = MAX( normalize, scale )
         CASE( 2 )
           IF ( comp_scale )                                                   &
             scale = MAX( Y_l( i ), C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( Y_l( i ) * ( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )     &
                  - mu ) / MAX( one, scale ) )
!          normalize = MAX( normalize, scale )
         CASE( - 2 )
           IF ( comp_scale )                                                   &
             scale = MAX( Y_u( i ), C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
           SUPERB_comp_slack = MAX( SUPERB_comp_slack,                         &
             ABS( Y_u( i ) * ( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )     &
                  - mu ) / MAX( one, scale ) )
!          normalize = MAX( normalize, scale )
         CASE DEFAULT
           GO TO 980
         END SELECT
       ELSE
         SELECT CASE ( S_type( i ) )
         CASE( 1 )
           IF ( C_l( i ) > - control%infinity ) THEN
             IF ( comp_scale ) scale = MAX( Y_l( i ), C( i ) - C_l( i ) )
             SUPERB_comp_slack = MAX( SUPERB_comp_slack,                       &
               ABS( Y_l( i ) * ( C( i ) - C_l( i ) )                           &
                    - mu ) / MAX( one, scale ) )
!            normalize = MAX( normalize, scale )
           END IF
           IF ( C_u( i ) <   control%infinity ) THEN
             IF ( comp_scale ) scale = MAX( Y_u( i ), C_u( i ) - C( i ) )
             SUPERB_comp_slack = MAX( SUPERB_comp_slack,                       &
               ABS( Y_u( i ) * ( C_u( i ) - C( i ) )                           &
                    - mu ) / MAX( one, scale ) )
!            normalize = MAX( normalize, scale )
           END IF
         CASE( 2 )
           IF ( C_l( i ) > - control%infinity ) THEN
             IF ( comp_scale )                                                 &
               scale = MAX(Y_l( i ), C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )
             SUPERB_comp_slack = MAX( SUPERB_comp_slack,                       &
               ABS( Y_l( i ) * ( C( i ) - C_l( i ) + SCALE_S( j ) * S( j ) )   &
                    - mu ) / MAX( one, scale ) )
!            normalize = MAX( normalize, scale )
           END IF
           IF ( C_u( i ) <   control%infinity ) THEN
             IF ( comp_scale )                                                 &
               scale = MAX(Y_u( i ), C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )
             SUPERB_comp_slack = MAX( SUPERB_comp_slack,                       &
               ABS( Y_u( i ) * ( C_u( i ) - C( i ) + SCALE_S( j ) * S( j ) )   &
                    - mu ) / MAX( one, scale ) )
!            normalize = MAX( normalize, scale )
           END IF
!        CASE( 3 )
!        CASE( - 3 )
         CASE DEFAULT
           GO TO 980
         END SELECT
       END IF
     END DO
!    write(6,*) ' normalization ', normalize

     invalid = .FALSE.
     RETURN

!  Error returns

 980 CONTINUE
     invalid = .TRUE.
     RETURN

!  End of function SUPERB_comp_slack

     END FUNCTION SUPERB_comp_slack

!-*-*-  G A L A H A D  -  S U P E R B _ m a g i c a l  S U B R O U T I N E -*-*-

     SUBROUTINE SUPERB_magical( m, nelastic, S_type, SSTATE, mu, nu, C_l, C_u, &
                                C, S, SCALE_S, S_u, len_s_u, control, invalid )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute "magical" improvements for the elastic variables

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m, nelastic, len_s_u
     REAL ( KIND = wp ), INTENT( IN ) :: mu, nu
     INTEGER, INTENT( IN ), DIMENSION( m ) :: S_type, SSTATE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u, C
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nelastic ) :: SCALE_S
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nelastic ) :: S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( len_s_u ) :: S_u
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     LOGICAL, INTENT( OUT ) :: invalid
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j
     REAL ( KIND = wp ) :: delta

     IF ( control%bound_elastics ) THEN
       DO i = 1, m
         j = SSTATE( i )
         IF ( j <= 0 ) CYCLE
         IF ( C_l( i ) == C_u( i ) ) THEN
           SELECT CASE ( S_type( i ) )
           CASE( 0 )
              delta = SUPERB_magical_step( 6, mu, nu, SCALE_S( j ),            &
                        C( i ) - C_l( i ), S( j ),                             &
                        dc2 = C_u( i ) - C( i ), smax = S_u( j ) )
           CASE( 1, - 1 )
           CASE( 2 )
             delta = SUPERB_magical_step( 3, mu, two * nu, SCALE_S( j ),       &
                       C( i ) - C_l( i ), S( j ), smax = S_u( j ) )
           CASE( - 2 )
             delta = SUPERB_magical_step( 3, mu, two * nu, SCALE_S( j ),       &
                       C_u( i ) - C( i ), S( j ), smax = S_u( j ) )
           CASE DEFAULT
             GO TO 980
           END SELECT
         ELSE
           SELECT CASE ( S_type( i ) )
           CASE( 1 )
           CASE( 2 )
             IF ( C_l( i ) > - control%infinity ) THEN
               IF ( C_u( i ) < control%infinity ) THEN
                 delta = SUPERB_magical_step( 4, mu, nu, SCALE_S( j ),         &
                            C( i ) - C_l( i ),  S( j ),                        &
                            dc2 = C_u( i ) - C( i ), smax = S_u( j ) )
               ELSE
                 delta = SUPERB_magical_step( 3, mu, nu,SCALE_S( j ),          &
                            C( i ) - C_l( i ), S( j ), smax = S_u( j ) )
               END IF
             ELSE IF ( C_u( i ) < control%infinity ) THEN
               delta = SUPERB_magical_step( 3, mu, nu, SCALE_S( j ),           &
                         C_u( i ) - C( i ), S( j ), smax = S_u( j ) )
             END IF
!          CASE( 3 )
!          CASE( - 3 )
           CASE DEFAULT
             GO TO 980
           END SELECT
         END IF
!        write(6,"( 'old, new s', 2ES12.4 )" ) S( j ), delta
         S( j ) = delta
       END DO
     ELSE
       DO i = 1, m
         j = SSTATE( i )
         IF ( j <= 0 ) CYCLE
         IF ( C_l( i ) == C_u( i ) ) THEN
           SELECT CASE ( S_type( i ) )
           CASE( 0 )
             delta = SUPERB_magical_step( 5, mu, nu, SCALE_S( j ),             &
                       C( i ) - C_l( i ), S( j ), dc2 = C_u( i ) - C( i ) )
           CASE( 1, - 1 )
           CASE( 2 )
             delta = SUPERB_magical_step( 1, mu, two * nu, SCALE_S( j ),       &
                       C( i ) - C_l( i ), S( j ) )
           CASE( - 2 )
             delta = SUPERB_magical_step( 1, mu, two * nu, SCALE_S( j ),       &
                       C_u( i ) - C( i ), S( j ) )
           CASE DEFAULT
             GO TO 980
           END SELECT
         ELSE
           SELECT CASE ( S_type( i ) )
           CASE( 1 )
           CASE( 2 )
             IF ( C_l( i ) > - control%infinity ) THEN
               IF ( C_u( i ) < control%infinity ) THEN
                 delta = SUPERB_magical_step( 2, mu, nu, SCALE_S( j ),         &
                           C( i ) - C_l( i ), S( j ), dc2 = C_u( i ) - C( i ) )
               ELSE
                 delta = SUPERB_magical_step( 1, mu, nu, SCALE_S( j ),         &
                           C( i ) - C_l( i ), S( j ) )
               END IF
             ELSE IF ( C_u( i ) < control%infinity ) THEN
               delta = SUPERB_magical_step( 1, mu, nu, SCALE_S( j ),           &
                         C_u( i ) - C( i ), S( j ) )
             END IF
!          CASE( 3 )
!          CASE( - 3 )
           CASE DEFAULT
             GO TO 980
           END SELECT
         END IF
!        write(6,"( 'old, new s', 2ES12.4 )" ) S( j ), delta
         S( j ) = delta
       END DO
     END IF

    invalid = .FALSE.
    RETURN
 
!  Error returns

 980 CONTINUE
     invalid = .TRUE.
     RETURN

!  Internal functions

     CONTAINS

!-  G A L A H A D  -  S U P E R B _ m a g i c a l _ s t e p   F U N C T I O N  -
 
       FUNCTION SUPERB_magical_step( phi_type, mu, nu, scale_s, dc1,           &
                                     s_initial, dc2, smax )
       REAL ( KIND = wp ) :: SUPERB_magical_step

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute the smallest value of the univariate function phi 
!   subject to constraints. Phi and its constraints are (phi_type):

!   (1): phi(s) =  nu s - mu log( dc1 + s ) - mu log s
!        such that   s > max( 0, - dc1 )
!   (2): phi(s) =  nu s - mu log( dc1 + s ) - mu log( dc2 + s ) - mu log s
!        such that s > max( 0, - dc1, - dc2 )
!   (3): phi(s) =  nu s - mu log( dc1 + s ) - mu log s - mu log( smax - s )
!        such that smax > s > max( 0, - dc1 )
!   (4): phi(s) =  nu s - mu log( dc1 + s ) - mu log( dc2 + s ) - mu log s 
!                  - mu log( smax - s )
!        such that smax > s > max( 0, - dc1, - dc2 )
!   (5): phi(s) =  nu s - mu log( dc1 + s ) - mu log( dc2 + s )
!        such that s > max( 0, - dc1, - dc2 )
!   (6): phi(s) =  nu s - mu log( dc1 + s ) - mu log( dc2 + s )
!                  - mu log( smax - s )
!        such that smax > s > max( 0, - dc1, - dc2 )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER, INTENT( IN ) :: phi_type
       REAL ( KIND = wp ), INTENT( IN ) :: mu, nu, s_initial, scale_s, dc1
       REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: dc2, smax
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

       REAL ( KIND = wp ) :: stop_s
       TYPE ( SUPERB_phi_data_type ) :: data
 
!  Special case using exact formula

       IF ( phi_type == 1 .AND.                                                &
            mu / nu > ten * SQRT( epsmch ) * half * dc1 ) THEN
!        write(6,*) ' explicit '
         SUPERB_magical_step = ( mu / nu - half * dc1 +                        &
           SQRT( ( half * dc1 ) ** 2 + ( mu / nu ) ** 2 ) ) / scale_s
         RETURN
       END IF

!  General case using safguarded Newton iteration

!      write(6,*) ' iteration '

!  Set data

       data%mu = mu ; data%nu = nu ; data%dc1 = dc1 ; data%scale_s = scale_s
       SELECT CASE ( phi_type )
       CASE ( 1 )
       CASE ( 2, 5 )
         data%dc2 = dc2
       CASE ( 3 )
         data%smax = smax
       CASE ( 4, 6 )
         data%dc2 = dc2 ; data%smax = smax
       END SELECT

!  Set stopping tolerance

       stop_s = ten * epsmch

!  Perform iteration

       SUPERB_magical_step                                                     &
         = SUPERB_magical_newton( phi_type, s_initial, stop_s, data )

       RETURN

!  End of function SUPERB_magical_step

       END FUNCTION SUPERB_magical_step

!- G A L A H A D - S U P E R B _ m a g i c a l _ n e w t o n  F U N C T I O N -
 
       FUNCTION SUPERB_magical_newton( phi_type, s_initial, stop_s, data )
       REAL ( KIND = wp ) :: SUPERB_magical_newton

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Use a safeguarded Newton iteration to minimize a function
!   phi of a single variable s. Possible functions phi are (phi_type):

!    (1) phi(s) =  nu scale_s s 
!                    - mu log( dc1 + scale_s s ) 
!                    - mu log( s )
!    (2) phi(s) =  nu  scale_ss 
!                    - mu log( dc1 + scale_s s ) - mu log( dc2 + scale_s s ) 
!                    - mu log s
!    (3) phi(s) =  nu scale_s s 
!                    - mu log( dc1 + scale_s s ) 
!                    - mu log( s ) - mu log( smax - s )
!    (4) phi(s) =  nu scale_s s 
!                    - mu log( dc1 + scale_s s ) - mu log( dc2 + scale_s s ) 
!                    - mu log s - mu log( smax - s )
!    (5) phi(s) =  nu scale_s s 
!                    - mu log( dc1 + scale_s s ) - mu log( dc2 + scale_s s ) 
!    (6) phi(s) =  nu scale_s s 
!                    - mu log( dc1 + scale_s s ) - mu log( dc2 + scale_s s )
!                    - mu log( smax - s )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

       INTEGER, INTENT( IN ) :: phi_type
       REAL ( KIND = wp ), INTENT( IN ) :: s_initial, stop_s
       TYPE ( SUPERB_phi_data_type ), INTENT( IN ) :: data
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

       INTEGER :: iter
       REAL ( KIND = wp ) :: lower, upper, s, s_new, phi_s, phi_ss
!      REAL ( KIND = wp ) :: phi
       LOGICAL, PARAMETER :: debug_newton = .FALSE.
!      LOGICAL, PARAMETER :: debug_newton = .TRUE.

       s = s_initial

       SELECT CASE( phi_type )
       CASE( 1 )
         IF ( data%scale_s * s <= MAX( zero, - data%dc1 ) )                    &
           s = MAX( zero, - data%dc1 ) / data%scale_s + one
       CASE( 2, 5 )
         IF ( data%scale_s * s <= MAX( zero, - data%dc1, - data%dc2 ) )        &
           s = MAX( zero, - data%dc1, - data%dc2 ) / data%scale_s + one
       CASE( 3 )
         IF ( data%scale_s * s <= MAX( zero, - data%dc1 ) .OR. s >= data%smax )&
           s = half * ( MAX( zero, - data%dc1 ) / data%scale_s + data%smax )
       CASE( 4, 6 )
         IF ( data%scale_s * s <= MAX( zero, - data%dc1, - data%dc2 ) .OR.     &
              s >= data%smax ) s = half * ( MAX( zero, - data%dc1,             &
                 - data%dc2 ) / data%scale_s + data%smax )
       END SELECT

!  Safeguarded Newton iteration

       iter = 0
       IF (  debug_newton )                                                    &
         WRITE( 6, "( ' phi_type ', I1 )" ) phi_type
       IF (  debug_newton )                                                    &
         WRITE( 6, "( '    lower         s         upper     phi_s' )" )
       DO

!  Compute the current slope ! and function value

         SELECT CASE( phi_type )
         CASE( 1 )
!          phi = data%nu * data%scale_s * s                                    &
!                  - data%mu * LOG( data%dc1 + data%scale_s * s )              &
!                  - data%mu * LOG( s )
           phi_s = data%nu * data%scale_s                                      &
                   - data%mu * data%scale_s / ( data%dc1 + data%scale_s * s )  &
                   - data%mu / s
         CASE( 2 )
!          phi = data%nu * data%scale_s * s                                    &
!                  - data%mu * LOG( data%dc1 + data%scale_s * s )              &
!                  - data%mu * LOG( data%dc2 + data%scale_s * s )              &
!                  - data%mu * LOG( s )
           phi_s = data%nu * data%scale_s                                      &
                   - data%mu * data%scale_s / ( data%dc1 + data%scale_s * s )  &
                   - data%mu * data%scale_s / ( data%dc2 + data%scale_s * s )  &
                   - data%mu / s
         CASE( 3 )
!          phi = data%nu * data%scale_s * s                                    &
!                  - data%mu * LOG( data%dc1 + data%scale_s * s )              &
!                  - data%mu * LOG( s ) - data%mu * LOG( data%smax - s )
           phi_s = data%nu * data%scale_s                                      &
                   - data%mu * data%scale_s / ( data%dc1 + data%scale_s * s )  &
                   - data%mu / s + data%mu / ( data%smax - s )
         CASE( 4 )
!          phi = data%nu * data%scale_s * s                                    &
!                  - data%mu * LOG( data%dc1 + data%scale_s * s )              &
!                  - data%mu * LOG( data%dc2 + data%scale_s * s )              &
!                  - data%mu * LOG( s ) - data%mu * LOG( data%smax - s )
           phi_s = data%nu * data%scale_s                                      &
                   - data%mu * data%scale_s / ( data%dc1 + data%scale_s * s )  &
                   - data%mu * data%scale_s / ( data%dc2 + data%scale_s * s )  &
                   - data%mu / s + data%mu / ( data%smax - s )
         CASE( 5 )
!          phi = data%nu * data%scale_s * s                                    &
!                  - data%mu * LOG( data%dc1 + data%scale_s * s )              &
!                  - data%mu * LOG( data%dc2 + data%scale_s * s )              &
           phi_s = data%nu * data%scale_s                                      &
                   - data%mu * data%scale_s / ( data%dc1 + data%scale_s * s )  &
                   - data%mu * data%scale_s / ( data%dc2 + data%scale_s * s )
         CASE( 6 )
!          phi = data%nu * data%scale_s * s                                    &
!                  - data%mu * LOG( data%dc1 + data%scale_s * s )              &
!                  - data%mu * LOG( data%dc2 + data%scale_s * s )              &
!                  - data%mu * LOG( data%smax - s )
           phi_s = data%nu * data%scale_s                                      &
                   - data%mu * data%scale_s / ( data%dc1 + data%scale_s * s )  &
                   - data%mu * data%scale_s / ( data%dc2 + data%scale_s * s )  &
                   + data%mu / ( data%smax - s )
         END SELECT

!  Test for convergence

         IF ( ABS( phi_s ) <= stop_s ) THEN
           EXIT

!  Refine the safeguard interval

         ELSE IF ( phi_s > zero ) THEN
           upper = s
           IF ( iter == 0 ) THEN
             SELECT CASE( phi_type )
             CASE( 1, 3 )
               lower = MAX( zero, - data%dc1 ) / data%scale_s
             CASE( 2, 4, 5, 6 )
               lower = MAX( zero, - data%dc1, - data%dc2 ) / data%scale_s
             END SELECT
           END IF
         ELSE
           lower = s
           IF ( iter == 0 ) THEN
             SELECT CASE( phi_type )
             CASE( 1 )
               upper = ( two * data%mu / data%nu + MAX( zero, - data%dc1 ) )   &
                         / data%scale_s
             CASE( 2 )
               upper = ( three * data%mu / data%nu                             &
                         + MAX( zero, - data%dc1, - data%dc2 ) )               &
                         / data%scale_s
             CASE( 3 )
               upper = MIN( data%smax, ( two * data%mu / data%nu               &
                              + MAX( zero, - data%dc1 ) ) / data%scale_s )
             CASE( 4 )
               upper = MIN( data%smax, ( three * data%mu / data%nu             &
                              + MAX( zero, - data%dc1, - data%dc2 ) )          &
                            / data%scale_s )
             CASE( 5 )
               upper = ( three * data%mu / data%nu                             &
                         + MAX( zero, - data%dc1, - data%dc2 ) ) / data%scale_s
             CASE( 6 )
               upper = MIN( data%smax, ( three * data%mu / data%nu             &
                         + MAX( zero, - data%dc1, - data%dc2 ) )               &
                        / data%scale_s )
             END SELECT
           END IF
         END IF

         iter = iter + 1
         IF (  debug_newton ) WRITE( 6, "( 4ES12.4 )" ) lower, s, upper, phi_s
!        WRITE( 6, "( 2ES12.4 )" ) ABS( upper - lower ), stop_s
         IF ( ABS( upper - lower ) < stop_s ) EXIT
!        IF ( iter > 100 ) STOP

!  Compute the current line curvature

         SELECT CASE( phi_type )
         CASE( 1 )
           phi_ss = data%mu *                                                  &
                      ( data%scale_s / ( data%dc1 + data%scale_s * s ) ) ** 2  &
                    + data%mu / s ** 2
         CASE( 2 )
           phi_ss = data%mu *                                                  &
                      ( data%scale_s / ( data%dc1 + data%scale_s * s ) ) ** 2  &
                    + data%mu *                                                &
                      ( data%scale_s / ( data%dc2 + data%scale_s * s ) ) ** 2  &
                    + data%mu / s ** 2
         CASE( 3 )
           phi_ss = data%mu *                                                  &
                      ( data%scale_s / ( data%dc1 + data%scale_s * s ) ) ** 2  &
                    + data%mu / s ** 2 + data%mu / ( data%smax - s ) ** 2
         CASE( 4 )
           phi_ss = data%mu *                                                  &
                      ( data%scale_s / ( data%dc1 + data%scale_s * s ) ) ** 2  &
                    + data%mu *                                                &
                      ( data%scale_s / ( data%dc2 + data%scale_s * s ) ) ** 2  &
                    + data%mu / s ** 2 + data%mu / ( data%smax - s ) ** 2
         CASE( 5 )
           phi_ss = data%mu *                                                  &
                      ( data%scale_s / ( data%dc1 + data%scale_s * s ) ) ** 2  &
                    + data%mu *                                                &
                      ( data%scale_s / ( data%dc2 + data%scale_s * s ) ) ** 2
         CASE( 6 )
           phi_ss = data%mu *                                                  &
                      ( data%scale_s / ( data%dc1 + data%scale_s * s ) ) ** 2  &
                    + data%mu *                                                &
                      ( data%scale_s / ( data%dc2 + data%scale_s * s ) ) ** 2  &
                    + data%mu / ( data%smax - s ) ** 2
         END SELECT

!  Form the Newton correction

         s_new = s - phi_s / phi_ss

!  Safeguard the correction

         IF ( s_new <= lower .OR. s_new >= upper )                             &
            s_new = half * ( lower + upper )
!        WRITE(6,"( ' s - snew ', ES12.4 )" ) s - s_new
         IF ( ABS( s - s_new ) < stop_s ) EXIT
         s = s_new
       END DO
       SUPERB_magical_newton = s

       IF (  debug_newton )                                                    &
         WRITE( 6, "( '  Final s = ', ES12.4, '  phi_s = ', ES12.4 )" ) s, phi_s

       RETURN

!  End of function SUPERB_magical_newton

     END FUNCTION SUPERB_magical_newton

!  End of subroutine SUPERB_magical

     END SUBROUTINE SUPERB_magical

!-*- G A L A H A D  -  S U P E R B _ c h o p _ m a g i c a l  F U N C T I O N -*

     FUNCTION SUPERB_chop_magical( m, S_type, SSTATE, C, C_l, C_u,             &
                                   SCALE_S, S_u, len_s_u, control )
     LOGICAL :: SUPERB_chop_magical

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Check if the upper bound on the elastic variables are consistent

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m, len_s_u
     INTEGER, INTENT( IN ), DIMENSION( m ) :: S_type, SSTATE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, C_l, C_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( len_s_u ) :: SCALE_S, S_u
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j
 
     SUPERB_chop_magical = .FALSE.
     IF ( .NOT. control%bound_elastics ) RETURN
     DO i = 1, m
       j = SSTATE( i )
       IF ( C_l( i ) == C_u( i ) ) THEN
         IF ( S_type( i ) == 0 ) THEN
           IF ( C( i ) - C_l( i ) + SCALE_S( j ) * S_u( j ) <= zero .OR.       &
                C_u( i ) - C( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 900
         ELSE IF ( S_type( i ) == 2 ) THEN
           IF ( C( i ) - C_l( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 900
         ELSE IF ( S_type( i ) == - 2 ) THEN
           IF ( C_u( i ) - C( i ) + SCALE_S( j ) * S_u( j ) <= zero ) GO TO 900
         END IF
       ELSE
         IF ( S_type( i ) == 2 ) THEN
           IF ( C_l( i ) > - control%infinity ) THEN
             IF ( C( i ) - C_l( i ) + SCALE_S( j ) * S_u( j ) <= zero )        &
               GO TO 900
           END IF
           IF ( C_u( i ) <   control%infinity ) THEN
             IF ( C_u( i ) - C( i ) + SCALE_S( j ) * S_u( j ) <= zero )        &
               GO TO 900
           END IF
         END IF
       END IF
     END DO
     RETURN

 900 CONTINUE
     SUPERB_chop_magical = .TRUE.
     RETURN

!  End of function SUPERB_chop_magical

     END FUNCTION SUPERB_chop_magical

!-*-  G A L A H A D  -  S U P E R B _ Armijo_linesearch  S U B R O U T I N E -*-

     SUBROUTINE SUPERB_Armijo_linesearch( n, m, nfree, nelastic,               &
                                          XFREE, S_type, SSTATE,               &
                                          mu, nu, X, X_l, X_u, C_l, C_u, S,    &
                                          SCALE_S, S_u, len_s_u, barrier,      &
                                          penalty, alpha, X_trial,             &
                                          C_trial, S_trial, DV, merit,         &
                                          merit_trial, f_trial, slope, eta,    &
                                          search_error, print_level, out,      &
                                          printt, nbacts, ratio,               &
                                          scale_xcf, ptrans_transform,         &
                                          ptrans_data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute a point along the line v + alpha * dv for which
!         phi(v+alpha*dv) - phi(v) < eta * linear model of change in phi
!   by backtracking from the initial alpha

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, m, nfree, nelastic, len_s_u, print_level, out
     INTEGER, INTENT( OUT ) :: search_error
     INTEGER, INTENT( INOUT ) :: nbacts
     LOGICAL, INTENT( IN ) :: printt, scale_xcf
     REAL ( KIND = wp ), INTENT( IN ) :: mu, nu, merit, slope, eta
     REAL ( KIND = wp ), INTENT( OUT ) :: barrier, penalty, merit_trial, f_trial
     REAL ( KIND = wp ), INTENT( INOUT ) :: alpha, ratio
     INTEGER, INTENT( IN ), DIMENSION( nfree ) :: XFREE
     INTEGER, INTENT( IN ), DIMENSION( m ) :: S_type, SSTATE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X_trial
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nelastic ) :: S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( len_s_u ) :: S_u, SCALE_S
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C_trial
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( nelastic ) :: S_trial
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nfree + nelastic ) :: DV
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: ptrans_transform
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: ptrans_data
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     TYPE ( SUPERB_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: merit_error, cutest_status
     REAL ( KIND = wp ) :: linear_model, slope_used
     LOGICAL :: invalid

!    write(6,"( ' slope = ', ES12.4 )" ) slope
     IF ( printt ) WRITE( out, "( ' ',/,'       ***  Linesearch    ',          &
    &                 ' step      trial value           model value ' )" )
     IF ( printt ) WRITE( out, "( 10X, ES22.14, 2ES22.14 )" )                  &
           zero, merit, merit

     DO
       nbacts = nbacts + 1 

!  Compute the trial point x + alpha dx

       X_trial( XFREE ) = X( XFREE ) + alpha * DV( : nfree )

!  Compute the new function and constraint values

!      WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  X_trial( : n )
!      WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )  S_trial( : m )

       IF ( scale_xcf ) THEN
         CALL PTRANS_cfn( n, m, X_trial, f_trial, m, C_trial,                  &
                          ptrans_transform, ptrans_data, inform%ptrans_inform )
       ELSE
         CALL CUTEST_cfn( cutest_status, n, m, X_trial, f_trial, C_trial )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF     
       inform%f_eval = inform%f_eval + 1

!  Compute the trial point for s

       IF ( control%magical_path ) THEN
         IF ( SUPERB_chop_magical( m, S_type, SSTATE, C_trial, C_l, C_u,       &
              SCALE_S, S_u, len_s_u, control ) ) THEN
           merit_error = 2 ; GO TO 110 ; END IF
         S_trial = S
         CALL SUPERB_magical( m, nelastic, S_type, SSTATE, mu, nu,             &
                              C_l, C_u, C_trial, S_trial, SCALE_S,             &
                              S_u, len_s_u, control, invalid )
         IF ( invalid ) GO TO 980
       ELSE
         S_trial = S + alpha * DV( nfree + 1 : )
       END IF
       slope_used = slope

!  The barrier value should be smaller than a linear model

       linear_model = merit + alpha * eta * slope_used

!      write(out,"( ' f_trial ', /, ES16.8 )" ) f_trial
!      write(out,"( ' c_trial ', /, ( 4ES16.8 ) )" ) C_trial

!  Compute the value of the merit function

       merit_trial = SUPERB_merit( n, m, nfree, nelastic, XFREE, S_type,       &
         SSTATE, f_trial, mu, nu, X_trial, X_l, X_u, C_trial, C_l, C_u,        &
         S_trial, SCALE_S, S_u, len_s_u, inform%pr_feas, barrier, penalty,     &
         merit_error, print_level, out, control )
       IF ( merit_error == - 99 ) GO TO 980

       IF ( ABS( merit_trial - merit ) > epsmch )                              &
         ratio = alpha * slope_used / ( merit_trial - merit )

 110   CONTINUE
       IF ( merit_error == 0 ) THEN
         IF ( printt ) WRITE( out, "( 10X, ES22.14, 2ES22.14 )" )              &
           alpha, merit_trial, linear_model 

!  Check to see if the Armijo criterion is satisfied. If not, halve the 
!  steplength

         IF ( merit_trial <= linear_model ) EXIT
       ELSE
         IF ( printt ) WRITE( out, "( 10X, ES22.14,                            &
        &  '  infinite logarithm  ', ES22.14 )" ) alpha, linear_model 
       END IF
       alpha = reduce_factor * alpha
       IF ( alpha < epsmch ) THEN ; search_error = 1 ; RETURN ; END IF
     END DO
     search_error = 0

     inform%status = 0
     RETURN

 930 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")           &
       cutest_status
     inform%status = GALAHAD_error_cutest
     RETURN

 980 CONTINUE
     inform%status = GALAHAD_error_technical
     RETURN

!  End of subroutine SUPERB_Armijo_linesearch

     END SUBROUTINE SUPERB_Armijo_linesearch

!-*-  G A L A H A D  -  S U P E R B _ exact_linesearch   S U B R O U T I N E -*-

     SUBROUTINE SUPERB_exact_linesearch( n, m, nfree, nelastic, XFREE, S_type, &
                                         XSTATE, SSTATE, ELASTICS, mu, nu,     &
                                         X, X_l, X_u, C_l, C_u, S, SCALE_S,    &
                                         S_u, len_s_u, merit,                  &
                                         barrier, penalty, alpha,              &
                                         X_trial, C_trial, S_trial, DV,        &
                                         merit_trial, f_trial, slope, prob,    &
                                         Z_l_P, Z_u_P, U_P, U_u_P,             &
                                         Y_l_P, Y_u_P, LAMBDA,                 &
                                         GRAD_b, IW, liw, J_ne, search_error,  &
                                         print_level, out, ratio, printt,      &
                                         scale_xcf, ptrans_transform,          &
                                         ptrans_data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute a point along the line v + alpha * dv for which
!         phi(v+alpha*dv) is (locally) smallest
!   by backtracking from the initial alpha

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
     INTEGER, INTENT( IN ) :: n, m, nfree, nelastic
     INTEGER, INTENT( IN ) :: len_s_u, print_level, out, liw
     INTEGER, INTENT( OUT ) :: search_error, J_ne
     LOGICAL, INTENT( IN ) :: printt, scale_xcf
     REAL ( KIND = wp ), INTENT( IN ) :: mu, nu, merit
     REAL ( KIND = wp ), INTENT( OUT ) :: barrier, penalty, merit_trial, f_trial
     REAL ( KIND = wp ), INTENT( INOUT ) :: alpha, slope, ratio
     INTEGER, INTENT( IN ), DIMENSION( n ) :: XSTATE
     INTEGER, INTENT( IN ), DIMENSION( nfree ) :: XFREE
     INTEGER, INTENT( IN ), DIMENSION( m ) :: S_type, SSTATE
     INTEGER, INTENT( IN ), DIMENSION( nelastic ) :: ELASTICS
     INTEGER, INTENT( OUT ), DIMENSION( liw ) :: IW
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: Z_l_P, Z_u_P
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X_trial
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u, LAMBDA
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nelastic ) :: S, SCALE_S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( len_s_u ) :: S_u
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( nelastic ) :: S_trial
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C_trial, Y_l_P, Y_u_P
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( nelastic ) :: U_P
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( len_s_u ) :: U_u_P
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nfree + nelastic ) :: DV
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( nfree + nelastic ) :: GRAD_b
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: ptrans_transform
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: ptrans_data
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     TYPE ( SUPERB_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: merit_error, i, j, l, J_len, inform_sort, cutest_status
     REAL ( KIND = wp ) :: alpha_max, alpha_min, g_term, alpha_old, slope_old
     REAL ( KIND = wp ) :: stop_g, stop_a, alpha_save, slope_init
     LOGICAL :: grlagf, got_slope_old, invalid

     alpha_min = zero
     alpha_max = alpha

     alpha_old = zero
     slope_old = slope
     got_slope_old = .TRUE.

     slope_init = slope
     stop_g = SQRT( epsmch ) ; stop_a = epsmch

     IF ( printt ) WRITE( out, "( ' ',/,'       ***  Linesearch    ',          &
    &                 ' step      trial value           slope ' )" )

     IF ( printt ) WRITE( out, "( 10X, ES22.14, 2ES22.14 )" )                  &
            zero, merit, slope
     DO

!  Compute the trial point x + alpha dx

       X_trial( XFREE ) = X( XFREE ) + alpha * DV( : nfree )

!  Compute the new function and constraint values

       IF ( scale_xcf ) THEN
         CALL PTRANS_cfn( n, m, X_trial, f_trial, m, C_trial,                  &
                          ptrans_transform, ptrans_data, inform%ptrans_inform )
       ELSE
         CALL CUTEST_cfn( cutest_status,  n, m, X_trial, f_trial, C_trial )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF     
       inform%f_eval = inform%f_eval + 1

!      write(out,"( ' f_trial ', /, ES16.8 )" ) f_trial
!      write(out,"( ' c_trial ', /, ( 4ES16.8 ) )" ) C_trial

!  Compute the trial point for s

       IF ( control%magical_path ) THEN
         IF ( SUPERB_chop_magical( m, S_type, SSTATE, C_trial, C_l, C_u,       &
              SCALE_S, S_u, len_s_u, control ) ) THEN
           merit_error = 2 ; GO TO 110 ; END IF
         S_trial = S
         CALL SUPERB_magical( m, nelastic, S_type, SSTATE, mu, nu,             &
                              C_l, C_u, C_trial, S_trial, SCALE_S,             &
                              S_u, len_s_u, control, invalid )
         IF ( invalid ) GO TO 980
       ELSE
         S_trial = S + alpha * DV( nfree + 1 : )
       END IF

!  Compute the value of the merit function

       merit_trial = SUPERB_merit( n, m, nfree, nelastic, XFREE, S_type,       &
         SSTATE, f_trial, mu, nu, X_trial, X_l, X_u, C_trial, C_l, C_u,        &
         S_trial, SCALE_S, S_u, len_s_u, inform%pr_feas, barrier, penalty,     &
         merit_error, print_level, out, control )
       IF ( merit_error == - 99 ) GO TO 980

 110   CONTINUE
       IF ( merit_error == 0 ) THEN

!  Compute primal Lagrange multiplier estimates

         DO l = 1, nfree
           i = XFREE( l )
           IF ( X_l( i ) > - control%infinity )                                &
             Z_l_P( i ) = mu / ( X_trial( i ) - X_l( i ) )
           IF ( X_u( i ) <   control%infinity )                                &
             Z_u_P( i ) = mu / ( X_u( i ) - X_trial( i ) )
         END DO

         DO i = 1, m
           j = SSTATE( i )
           IF ( ABS( S_type( i ) ) /= 1 ) THEN
             IF ( S_type( i ) /= 0 ) U_P( j ) = mu / S_trial( j )
             IF ( control%bound_elastics )                                     &
               U_u_P( j ) = mu / ( S_u( j ) - S_trial( j ) )
           END IF
           IF ( C_l( i ) == C_u( i ) ) THEN
             SELECT CASE ( S_type( i ) )
             CASE( 0 )
               Y_l_P( i ) = mu / ( C_trial( i ) - C_l( i ) +                   &
                                   SCALE_S( j ) * S_trial( j ) )
               Y_u_P( i ) = mu / ( C_l( i ) - C_trial( i ) +                   &
                                   SCALE_S( j ) * S_trial( j ) )
             CASE( 1 )
               Y_l_P( i ) = mu / ( C_trial( i ) - C_l( i ) )
             CASE( - 1 )
               Y_u_P( i ) = mu / ( C_l( i ) - C_trial( i ) )
             CASE( 2 )
               Y_l_P( i ) = mu / ( C_trial( i ) - C_l( i ) +                   &
                                   SCALE_S( j ) * S_trial( j ) )
             CASE( - 2 )
               Y_u_P( i ) = mu / ( C_l( i ) - C_trial( i ) +                   &
                                   SCALE_S( j ) * S_trial( j ) )
             CASE DEFAULT
               GO TO 980
             END SELECT
           ELSE
             SELECT CASE ( S_type( i ) )
             CASE( 1 )
               IF ( C_l( i ) > - control%infinity )                            &
                 Y_l_P( i ) = mu / ( C_trial( i ) - C_l( i ) )
               IF ( C_u( i ) <   control%infinity )                            &
                 Y_u_P( i ) = mu / ( C_u( i ) - C_trial( i ) )
             CASE( 2 )
               IF ( C_l( i ) > - control%infinity )                            &
                 Y_l_P( i ) = mu / ( C_trial( i ) - C_l( i ) +                 &
                                     SCALE_S( j ) * S_trial( j ) )
               IF ( C_u( i ) <   control%infinity )                            &
                 Y_u_P( i ) = mu / ( C_u( i ) - C_trial( i ) +                 &
                                     SCALE_S( j ) * S_trial( j ) )
!            CASE( 3 )
!            CASE( - 3 )
             CASE DEFAULT
               GO TO 980
             END SELECT
           END IF
         END DO

!  Evaluate both the gradients of the general constraint functions
!  and the Hessian matrix of the Lagrangian function for the problem.
!  The Hessian is stored as a sparse matrix in "co-ordinate" format. 
!  Also obtain the gradient of either the objective function or
!  the Lagrangian function. The data is stored in a sparse format.

         grlagf = .FALSE. ; J_len = J_ne
         IF ( scale_xcf ) THEN
           CALL PTRANS_csgr( n, m, grlagf, m, LAMBDA, X_trial, J_ne, J_len,    &
                              prob%A%val, prob%A%col, prob%A%row,              &
                              ptrans_transform, ptrans_data,                   &
                              inform%ptrans_inform )
         ELSE
           CALL CUTEST_csgr( cutest_status, n, m, X_trial, LAMBDA, grlagf,     &
                             J_ne, J_len, prob%A%val, prob%A%col, prob%A%row )
           IF ( cutest_status /= 0 ) GO TO 930
         END IF     
         inform%g_eval = inform%g_eval + 1

!  Untangle A: separate the gradient terms from the constraint Jacobian

         prob%A%ne = 0 ; prob%G( : n ) = zero
         DO i = 1, J_ne
           IF ( prob%A%row( i ) == 0 ) THEN
             prob%G( prob%A%col( i ) ) = prob%A%val( i )
           ELSE
             prob%A%ne = prob%A%ne + 1
             prob%A%row( prob%A%ne ) = prob%A%row( i )
             prob%A%col( prob%A%ne ) = prob%A%col( i )
             prob%A%val( prob%A%ne ) = prob%A%val( i )
!            write(6,"(2I8,ES12.4)")                                           &
!              prob%A%row( prob%A%ne ), prob%A%col( prob%A%ne ),               &
!              prob%A%val( prob%A%ne ) 
           END IF
         END DO

!        WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" )  X( : n )
!        WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" )  S( : m )
!        WRITE( out, "( ' g ', /, ( 5ES12.4 ) )" )  prob%G( : n )
!        WRITE( out, "( ' g_max ', ES12.4 )" )  MAXVAL( ABS( prob%G ) )

!  Now reorder A so that it is stored by rows

         CALL SORT_reorder_by_rows( m, n, prob%A%ne, prob%A%row, prob%A%col,   &
           J_ne, prob%A%val, prob%A%ptr, m + 1, IW, liw, control%error,        &
           control%out, inform_sort )

!  Compute the gradient of the barrier function

!  wrt x

         GRAD_b( : nfree ) = prob%G( XFREE( : nfree ) )
         DO l = 1, nfree
           i = XFREE( l )
           IF ( X_l( i ) > - control%infinity )                                &
             GRAD_b( l ) = GRAD_b( l ) - Z_l_P( i )
           IF ( X_u( i ) <   control%infinity )                                &
             GRAD_b( l ) = GRAD_b( l ) + Z_u_P( i )
         END DO

!  subtract Jacobian (transpose) times multiplers

         DO i = 1, m
           IF ( C_l( i ) == C_u( i ) ) THEN
             SELECT CASE ( S_type( i ) )
             CASE( 0 )
               g_term = Y_l_P( i ) - Y_u_P( i )
             CASE( 1, 2 )
               g_term = Y_l_P( i ) - nu
             CASE( - 1, - 2 )
               g_term = - ( Y_u_P( i ) - nu )
             CASE DEFAULT
               GO TO 980
             END SELECT
           ELSE
             g_term = zero
             SELECT CASE ( S_type( i ) )
             CASE( 1, 2 )
               IF ( C_l( i ) > - control%infinity ) g_term = g_term + Y_l_P( i )
               IF ( C_u( i ) <   control%infinity ) g_term = g_term - Y_u_P( i )
!            CASE( 3 )
!            CASE( - 3 )
             CASE DEFAULT
               GO TO 980
             END SELECT
           END IF
!          GRAD_b( nfree + i ) = g_term

           DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
             j = XSTATE( prob%A%col( l ) )
             IF ( j > 0 ) GRAD_b( j ) = GRAD_b( j ) - prob%A%val( l ) * g_term
           END DO
         END DO

!  wrt s

         IF ( nelastic > 0 ) THEN
           DO j = 1, nelastic
             i = ELASTICS( j )
             GRAD_b( nfree + j ) = SCALE_S( j ) * nu
             IF ( control%bound_elastics )                                     &
               GRAD_b( nfree + j ) = GRAD_b( nfree + j ) + U_u_P( j )
             IF ( S_type( i ) /= 0 )                                           &
               GRAD_b( nfree + j ) = GRAD_b( nfree + j ) - U_P( j )
             IF ( C_l( i ) == C_u( i ) ) THEN
               SELECT CASE ( S_type( i ) )
               CASE( 0 )
                 GRAD_b( nfree + j ) = GRAD_b( nfree + j ) -                   &
                   SCALE_S( j ) * ( Y_l_P( i ) + Y_u_P( i ) )
               CASE( 2 )
                 GRAD_b( nfree + j ) = GRAD_b( nfree + j ) -                   &
                   SCALE_S( j ) * ( Y_l_P( i ) - nu )
               CASE( - 2 )
                 GRAD_b( nfree + j ) = GRAD_b( nfree + j ) -                   &
                   SCALE_S( j ) * ( Y_u_P( i ) - nu )
               CASE DEFAULT
                 GO TO 980
               END SELECT
             ELSE
               SELECT CASE ( S_type( i ) )
               CASE( 2 )
                 IF ( C_l( i ) > - control%infinity ) GRAD_b( nfree + j ) =    &
                   GRAD_b( nfree + j ) - SCALE_S( j ) * Y_l_P( i )
                 IF ( C_u( i ) <   control%infinity ) GRAD_b( nfree + j ) =    &
                   GRAD_b( nfree + j ) - SCALE_S( j ) * Y_u_P( i )
!              CASE( 3 )
!              CASE( - 3 )
               CASE DEFAULT
                 GO TO 980
               END SELECT
             END IF
           END DO
         END IF

         IF ( control%magical_path ) THEN
!          slope = DOT_PRODUCT( GRAD_b( : nfree ), DV( : nfree ) ) +           &
!                  DOT_PRODUCT( GRAD_b( nfree + 1 : ), S_trial - S )
           slope = DOT_PRODUCT( GRAD_b, DV ) 
         ELSE
           slope = DOT_PRODUCT( GRAD_b, DV ) 
         END IF

         IF ( printt ) WRITE( out, "( 10X, ES22.14, 2ES22.14 )" )              &
           alpha, merit_trial, slope

!  Check to see if the exact criterion is satisfied. If not, halve the 
!  steplength

         IF ( ABS( merit_trial - merit ) > epsmch )                            &
            ratio = alpha * slope_init / ( merit_trial - merit )
         IF ( ABS( slope ) <= stop_g ) THEN ; EXIT
         ELSE IF ( slope < zero ) THEN ; alpha_min = alpha
         ELSE ; alpha_max = alpha ; END IF
         IF ( alpha_max - alpha_min <= stop_a ) EXIT

         alpha_save = alpha
         IF ( got_slope_old ) THEN
           IF ( ABS( slope - slope_old ) > epsmch ) THEN
             alpha = alpha - slope * ( alpha - alpha_old )                     &
                     / ( slope - slope_old ) 
             IF ( alpha <= alpha_min + stop_a .OR.                             &
                  alpha >= alpha_max - stop_a ) THEN
               alpha = half * ( alpha_max + alpha_min )
             END IF
           ELSE
             alpha = half * ( alpha_max + alpha_min )
           END IF
         ELSE
            alpha = half * ( alpha_max + alpha_min )
         END IF
         alpha_old = alpha_save
         slope_old = slope
         got_slope_old = .TRUE.
       ELSE
         IF ( printt ) WRITE( out, "( 10X, ES22.14,                            &
        &  '  infinite logarithm  ', ES22.14 )" ) alpha
         alpha_max = alpha
         alpha = half * ( alpha_max + alpha_min )
         ratio = - point1 * HUGE( one ) 
         got_slope_old = .FALSE.
       END IF
       IF ( alpha < stop_a ) THEN ; search_error = 1 ; RETURN ; END IF
     END DO
     search_error = 0

     inform%status = 0
     RETURN

 930 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )")           &
       cutest_status
     inform%status = GALAHAD_error_cutest
     RETURN

 980 CONTINUE
     inform%status = GALAHAD_error_technical
     RETURN

!  End of subroutine SUPERB_exact_linesearch

     END SUBROUTINE SUPERB_exact_linesearch

!-*- G A L A H A D  -  S U P E R B _ i t e r a t i v e _ r e f i n e m e n t -*-

     SUBROUTINE SUPERB_iterative_refinement( K, FACTORS, CNTL, SOL, RHS, RES,  &
                                             BEST, res_norm, big_res,          &
                                             itref_max, print_level, out )
                                        
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute the solution to the preconditioned system 

!      ( P + B_X            J^T   ) (d_x)   (r_x)
!      (          K_22    K_23^T  ) (d_s) = (r_s)
!      (   J      K_23     K_33   ) (d_y)   (r_y)

!  where P is a specified "preconditioner" for H,
!  K_22 = B_S + Theta ( B_C - B_CM B_C^-1 B_CM ) Theta,
!  K_23 = B_C^-1 B_CM Theta,
!  K_33 = -B_C^-1, and
!  Theta = diag(SCALE_S)

!  using iterative refinement

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SMT_type ), INTENT( IN ) :: K
     TYPE ( SILS_factors ), INTENT( IN ) :: FACTORS
     TYPE ( SILS_control ), INTENT( IN ) :: CNTL
     INTEGER, INTENT( IN ) :: itref_max, print_level, out
     REAL ( KIND = wp ), INTENT( OUT ) :: res_norm
     LOGICAL, INTENT( OUT ) :: big_res
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( K%n ) :: RHS
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( K%n ) :: SOL, RES, BEST
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, it, j, l
     REAL ( KIND = wp ) ::  old_res, val, res_stop
     TYPE ( SILS_sinfo ) :: SINFO

     big_res = .FALSE.
     old_res = NRM2( K%n, RHS, 1 )
     res_stop = MAX( res_large, old_res )

     SOL = RHS
     CALL SILS_solve( K, FACTORS, SOL, CNTL, SINFO )

!  Perform the iterative refinement

     DO it = 1, itref_max

!  Compute the residual

       RES = RHS
       DO l = 1, K%ne
         i = K%row( l ) ; j = K%col( l ) ; val = K%val( l )
         RES( i ) = RES( i ) - val * SOL( j )
         IF ( i /= j ) RES( j ) = RES( j ) - val * SOL( i )
       END DO
       res_norm = NRM2( K%n, RES, 1 )
       IF ( out > 0 .AND. print_level >= 3 ) WRITE( out, 2000 ) res_norm
       IF ( res_norm > res_stop ) THEN
         write(6, "( ' res, tol ', 2ES12.4 )" ) res_norm, res_stop
         big_res = .TRUE.
         RETURN
       END IF

!  If the norm has increased, quit

       IF ( it > 1 ) THEN
         IF ( res_norm < old_res ) THEN
           old_res = res_norm
           BEST = SOL
         ELSE
           res_norm = old_res
           SOL = BEST
           GO TO 100
         END IF
       ELSE
         BEST = SOL
       END IF

!  Obtain a new correction

       CALL SILS_solve( K, FACTORS, RES, CNTL, SINFO )
       SOL = SOL + RES

!  End of iterative refinement

     END DO

!  Obtain final residuals if required

     IF ( it >= itref_max .OR. itref_max == 0 ) THEN
       RES = RHS
       DO l = 1, K%ne
         i = K%row( l ) ; j = K%col( l ) ; val = K%val( l )
         RES( i ) = RES( i ) - val * SOL( j )
         IF ( i /= j ) RES( j ) = RES( j ) - val * SOL( i )
       END DO
       res_norm = NRM2( K%n, RES, 1 )
       IF ( res_norm > res_stop ) THEN
         write(6, "( ' res, tol ', 2ES12.4 )" ) res_norm, res_stop
         big_res = .TRUE.
         RETURN
       END IF
       IF ( out > 0 .AND. print_level >= 3 ) WRITE( out, 2000 ) res_norm
     END IF

 100 CONTINUE
     RETURN

!  Non-executable statements

 2000 FORMAT( '    residual = ', ES12.4 )

!  End of subroutine SUPERB_iterative_refinement

     END SUBROUTINE SUPERB_iterative_refinement

!-*- G A L A H A D  -  S U P E R B _ i t e r a t i v e _ r e f i n e m e n t -*-

     SUBROUTINE SUPERB_block_refinement( m, nfree, nelastic, ELASTICS,         &
                                         K, B_C, B_CM, B_S, SCALE_S,           &
                                         FACTORS, CNTL, SOL, RHS, RES, BEST,   &
                                         res_norm, big_res, itref_max,         &
                                         print_level, out )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the solution to the preconditioned system 

!      ( P + B_X            J^T   ) (d_x)   (r_x)
!      (          K_22    K_23^T  ) (d_s) = (r_s)
!      (   J      K_23     K_33   ) (d_y)   (r_y)

!  by solving the block system

!     ( P + B_X              J^T            )(d_x) = (          r_x         )
!     (   J     K_33 - K_32 K_22^-1 K_23^T  )(d_y)   (r_y - K_23 K_22^-1 r_s)

!  followed by 

!       K_22 d_s = r_s - K_23^T d_y
             
!  where P is a specified "preconditioner" for H,
!  K_22 = B_S + Theta ( B_C - B_CM B_C^-1 B_CM ) Theta,
!  K_23 = B_C^-1 B_CM Theta,
!  K_33 = -B_C^-1, and
!  Theta = diag(SCALE_S)

!  using block iterative refinement

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m, nfree, nelastic
     INTEGER, INTENT( IN ), DIMENSION( nelastic ) :: ELASTICS
     TYPE ( SMT_type ), INTENT( IN ) :: K
     TYPE ( SILS_factors ), INTENT( IN ) :: FACTORS
     TYPE ( SILS_control ), INTENT( IN ) :: CNTL
     INTEGER, INTENT( IN ) :: itref_max, print_level, out
     REAL ( KIND = wp ), INTENT( OUT ) :: res_norm
     LOGICAL, INTENT( OUT ) :: big_res
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( K%n + nelastic ) :: RHS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: B_C, B_CM
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nelastic ) :: B_S, SCALE_S
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( K%n + nelastic ) :: SOL
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( K%n ) :: RES, BEST
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, it, j, l, nfreep1, nfreepm
     REAL ( KIND = wp ) ::  old_res, val, res_stop
     TYPE ( SILS_sinfo ) :: SINFO

     nfreep1 = nfree + 1 ; nfreepm = nfree + m
     big_res = .FALSE.
     old_res = NRM2( K%n, RHS, 1 )
     res_stop = MAX( res_large, old_res )

     SOL( : nfree ) = RHS( : nfree )
     SOL( nfreep1 : nfreepm ) = RHS( nfreep1 + nelastic : nfreepm + nelastic )

     DO j = 1, nelastic
       i = ELASTICS( j )
       SOL( nfree + i ) = SOL( nfree + i ) -                                   &
         ( B_CM( i ) * SCALE_S( j ) / B_C( i ) ) * RHS( nfree + j ) /          &
           ( B_S( j ) + ( B_C( i ) - B_CM( i ) ** 2 / B_C( i ) )               &
             * SCALE_S( j ) ** 2 )
     END DO

     CALL SILS_solve( K, FACTORS, SOL( : nfreepm ), CNTL, SINFO )

!  Perform the iterative refinement

     DO it = 1, itref_max

!  Compute the residual

       RES( : nfree ) = RHS( : nfree )
       RES( nfreep1 : nfreepm ) = RHS( nfreep1 + nelastic : nfreepm + nelastic )
       DO j = 1, nelastic
         i = ELASTICS( j )
         RES( nfree + i ) = RES( nfree + i ) -                                 &
           ( B_CM( i ) * SCALE_S( j ) / B_C( i ) ) * RHS( nfree + j ) /        &
             ( B_S( j ) + ( B_C( i ) - B_CM( i ) ** 2 / B_C( i ) )             &
               * SCALE_S( j ) ** 2 )
       END DO

       DO l = 1, K%ne
         i = K%row( l ) ; j = K%col( l ) ; val = K%val( l )
         RES( i ) = RES( i ) - val * SOL( j )
         IF ( i /= j ) RES( j ) = RES( j ) - val * SOL( i )
       END DO

!      WRITE(6, "(2ES12.4)") MAXVAL( ABS( RES( : nfree ) ) ),                  &
!        MAXVAL( ABS( RES( nfree + 1 : nfreepm ) ) )
       res_norm = NRM2( K%n, RES( : nfreepm ), 1 )
       IF ( out > 0 .AND. print_level >= 3 ) WRITE( out, 2000 ) res_norm

       IF ( res_norm > res_stop ) THEN
         write(6, "( ' res, tol ', 2ES12.4 )" ) res_norm, res_stop
         big_res = .TRUE.
         RETURN
       END IF

!  If the norm has increased, quit

       IF ( it > 1 ) THEN
         IF ( res_norm < old_res ) THEN
           old_res = res_norm
           BEST( : nfreepm ) = SOL( : nfreepm )
         ELSE
           res_norm = old_res
           SOL( : nfreepm ) = BEST( : nfreepm )
           GO TO 100
         END IF
       ELSE
         BEST( : nfreepm ) = SOL( : nfreepm )
       END IF

!  Obtain a new correction

       CALL SILS_solve( K, FACTORS, RES( : nfreepm ), CNTL, SINFO )
       SOL( : nfreepm ) = SOL( : nfreepm ) + RES( : nfreepm )

!  End of iterative refinement

     END DO

!  Obtain final residuals if required

     IF ( it >= itref_max .OR. itref_max == 0 ) THEN
       RES( : nfree ) = RHS( : nfree )
       RES( nfreep1 : nfreepm ) = RHS( nfreep1 + nelastic : nfreepm + nelastic )
       DO j = 1, nelastic
         i = ELASTICS( j )
         RES( nfree + i ) = RES( nfree + i ) -                                 &
           ( B_CM( i )  * SCALE_S( j ) / B_C( i ) ) * RHS( nfree + j ) /       &
           ( B_S( j ) + ( B_C( i ) - B_CM( i ) ** 2 / B_C( i ) )               &
             * SCALE_S( j ) ** 2 )
       END DO
       DO l = 1, K%ne
         i = K%row( l ) ; j = K%col( l ) ; val = K%val( l )
         RES( i ) = RES( i ) - val * SOL( j )
         IF ( i /= j ) RES( j ) = RES( j ) - val * SOL( i )
       END DO
       res_norm = NRM2( K%n, RES( : nfreepm ), 1 )
       IF ( res_norm > res_stop ) THEN
         write(6, "( ' res, tol ', 2ES12.4 )" ) res_norm, res_stop
         big_res = .TRUE.
         RETURN
       END IF
       IF ( out > 0 .AND. print_level >= 3 ) WRITE( out, 2000 ) res_norm
     END IF

 100 CONTINUE
     IF ( nelastic /= 0 ) THEN

!  Move the d_y components to their correct positions

       DO i = m, 1, - 1
         SOL( nfree + nelastic + i ) = SOL( nfree + i )
       END DO

!  Compute the d_s components

       DO j = 1, nelastic
          i = ELASTICS( j )
          SOL( nfree + j ) = ( RHS( nfree + j ) -                              &
            ( B_CM( i ) * SCALE_S( j ) / B_C( i ) )                            &
              * SOL( nfree + nelastic + i ) ) /                                &
              ( B_S( j ) + ( B_C( i ) - B_CM( i ) ** 2 / B_C( i ) )            &
                * SCALE_S( j ) ** 2 )
       END DO

     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( '    residual = ', ES12.4 )

!  End of subroutine SUPERB_block_refinement

     END SUBROUTINE SUPERB_block_refinement

!-*-*-*-*  G A L A H A D -  S U P E R B _ H _ b   S U B R O U T I N E -*-*-*-*

     SUBROUTINE SUPERB_H_b( prob, nfree, nelastic, XSTATE, SSTATE, B_X, B_S,   &
                            SCALE_S, B_C, B_CM, V, HV, vTHv, inner_prod )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Form either H_b * v or v^T * H_b * v, where H_b is the Hessian of the
!  barrier function

!   H_b = (  H + B_X + J^T B_C J      J^T B_CM Theta     ),
!         (     Theta B_CM J       Theta B_C Theta + B_S )

!   Theta = diag( scale_s ) and v = ( x,  s )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( QPT_problem_type ), INTENT( IN ) :: prob
     INTEGER, INTENT( IN ) :: nfree, nelastic
     REAL ( KIND = wp ), INTENT( OUT ) :: vTHV
     LOGICAL, INTENT( IN ) :: inner_prod
     INTEGER, INTENT( IN ), DIMENSION( prob%n ) :: XSTATE
     INTEGER, INTENT( IN ), DIMENSION( prob%m ) :: SSTATE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nfree ) :: B_X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( prob%m ) :: B_C, B_CM
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nelastic ) :: B_S, SCALE_S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nfree + nelastic ) :: V
     REAL ( KIND = wp ), INTENT( OUT ),                                        &
       DIMENSION( nfree + nelastic + prob%m ):: HV

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, l, m, nfrpel
     REAL ( KIND = wp ) :: val

     m = prob%m
     nfrpel = nfree + nelastic

!  First, form ( B_x * x,   B_s * s )

     DO i = 1, nfree
       HV( i ) = B_X( i ) * V( i )
     END DO

     DO i = 1, nelastic
       HV( nfree + i ) = B_S( i ) * V( nfree + i )
     END DO

!  Also form J * x

     HV( nfrpel + 1 : ) = zero
     DO l = 1, prob%A%ne
       i = nfrpel + prob%A%row( l )
       j = XSTATE( prob%A%col( l ) )
       IF ( j > 0 ) HV( i ) = HV( i ) + prob%A%val( l ) * V( j )
     END DO

!  Now add H * x to B_x * x

     DO l = 1, prob%H%ne
       i = XSTATE( prob%H%row( l ) ) ; j = XSTATE( prob%H%col( l ) )
       val = prob%H%val( l )
       IF ( i > 0 .AND. j > 0 ) THEN
         HV( i ) = HV( i ) + val * V( j )
         IF ( j /= i ) HV( j ) = HV( j ) + val * V( i )
       END IF
     END DO

     IF ( inner_prod ) THEN

!  The required inner product is 
!    x^T * ( H * x + B_x * x ) + s^T*  B_s * s
!    + ( J * x )^T B_c ( J * x ) + 2 s^T * Theta * B_cm * J * x 
!    + s^T * Theta B_c * Theta s

       vTHv = DOT_PRODUCT( V( : nfrpel ), HV( : nfrpel ) )
       DO i = 1, m
         j = SSTATE( i )
         vTHv = vTHv + B_C( i ) * HV( nfrpel + i ) ** 2 
         IF ( j > 0 ) vTHv = vTHv                                              &
           + B_C( i ) * ( SCALE_S( j ) * V( nfree + j ) ) ** 2                 &
           + two * V( nfree + j ) * SCALE_S( j ) * B_CM( i ) *  HV( nfrpel + i )
       END DO
     ELSE

!  The required matrix-vector product is
!   ( H * x + B_x * x ) + J^T (          B_c * J * x + B_cm * Theta * s        )
!   (     B_s * s     )       ( Theta * B_cm * J * x + Theta * B_c * Theta * s )

       DO i = 1, m
         j = SSTATE( i )
         IF ( j > 0 ) HV( nfree + j ) = HV( nfree + j )                        &
           + SCALE_S( j ) * ( B_CM( i ) * HV( nfrpel + i )                     &
             + B_C( i ) * SCALE_S( j ) * V( nfree + j ) )
         HV( nfrpel + i ) = B_C( i ) * HV( nfrpel + i )
         IF ( j > 0 ) HV( nfrpel + i ) = HV( nfrpel + i )                      &
           + B_CM( i ) * SCALE_S( j ) * V( nfree + j )
       END DO
       DO l = 1, prob%A%ne
         j = XSTATE( prob%A%col( l ) )
         IF ( j > 0 ) HV( j ) = HV( j )                                        &
             + prob%A%val( l ) * HV( nfrpel + prob%A%row( l ) )
       END DO
     END IF
     RETURN

!  End of subroutine SUPERB_H_b

     END SUBROUTINE SUPERB_H_b

!-*-  G A L A H A D  -  S U P E R B _ w r i t e _ y _ S U B R O U T I N E  -*-

     SUBROUTINE SUPERB_write_y( out, m, low, string_name, S_type, C_l, C_u,    &
                                Y_l_P, Y_u_P, control, invalid )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the value of the current violation of (scaled) complementarity

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m, out
     LOGICAL, INTENT( IN ) :: low
     CHARACTER ( len = * ), INTENT( IN ) :: string_name
     INTEGER, INTENT( IN ), DIMENSION( m ) :: S_type
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y_l_P, Y_u_P
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     LOGICAL, INTENT( OUT ) :: invalid
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, l
     INTEGER, PARAMETER :: n_strings = 5
     INTEGER, PARAMETER :: len_string = 12
     INTEGER, PARAMETER :: len_full_string = n_strings * len_string
     CHARACTER ( len = len_string ) :: string, null_string
     CHARACTER ( len = len_full_string ) :: full_string, null_full_string 
     null_string = repeat( ' ', len_string )
     null_full_string = repeat( null_string, n_strings )

     WRITE( out, "( A )" ) string_name
     DO l = 1, m, n_strings
       full_string = null_full_string
       DO i = l, l + min( n_strings - 1, m - l )
         string = null_string
         IF ( C_l( i ) == C_u( i ) ) THEN
           SELECT CASE ( S_type( i ) )
           CASE( 0 )
             IF ( low ) THEN
               WRITE( string, "( ES12.4 )" ) Y_l_P( i )
             ELSE 
               WRITE( string, "( ES12.4 )" ) Y_u_P( i )
             END IF
           CASE( 1, 2 )
             IF ( low ) WRITE( string, "( ES12.4 )" ) Y_l_P( i )
           CASE( - 1, - 2 )
             IF ( .NOT. low ) WRITE( string, "( ES12.4 )" ) Y_u_P( i )
           CASE DEFAULT
             GO TO 980
           END SELECT
         ELSE
           SELECT CASE ( S_type( i ) )
           CASE( 1, 2 )
             IF ( C_l( i ) > - control%infinity .AND. low )                    &
               WRITE( string, "( ES12.4 )" ) Y_l_P( i )
             IF ( C_u( i ) <   control%infinity .AND. .NOT. low )              &
               WRITE( string, "( ES12.4 )" ) Y_u_P( i )
!          CASE( 3 )
!          CASE( - 3 )
           CASE DEFAULT
             GO TO 980
           END SELECT
         END IF
         full_string( 1 + len_string * ( i - l ) :                             &
                      len_string * ( i + 1 - l ) ) = string
       END DO
       WRITE( out, "( A )" ) full_string
     END DO

     invalid = .FALSE.
     RETURN

 980 CONTINUE
     invalid = .TRUE.
     RETURN

!  End of subroutine SUPERB_write_y

     END SUBROUTINE SUPERB_write_y

!-*-*  G A L A H A D  -  S U P E R B _ w r i t e _ z _ S U B R O U T I N E  *-*-

     SUBROUTINE SUPERB_write_z( out, n, nfree, low, string_name, XFREE,        &
                                X_l, X_u, Z_l_P, Z_u_P, control )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Compute the value of the current violation of (scaled) complementarity

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nfree, out
     LOGICAL, INTENT( IN ) :: low
     CHARACTER ( len = * ), INTENT( IN ) :: string_name
     INTEGER, INTENT( IN ), DIMENSION( nfree ) :: XFREE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: Z_l_P, Z_u_P
     TYPE ( SUPERB_control_type ), INTENT( IN ) :: control
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, k, l
     INTEGER, PARAMETER :: n_strings = 5
     INTEGER, PARAMETER :: len_string = 12
     INTEGER, PARAMETER :: len_full_string = n_strings * len_string
     CHARACTER ( len = len_string ) :: string, null_string
     CHARACTER ( len = len_full_string ) :: full_string, null_full_string 
     null_string = repeat( ' ', len_string )
     null_full_string = repeat( null_string, n_strings )

     WRITE( out, "( A )" ) string_name
     DO l = 1, nfree, n_strings
       full_string = null_full_string
       DO k = l, l + min( n_strings - 1, nfree - l )
         i = XFREE( l )
         string = null_string
         IF ( X_l( i ) > - control%infinity .AND. low )                        &
           WRITE( string, "( ES12.4 )" ) Z_l_P( i )
         IF ( X_u( i ) <   control%infinity .AND. .NOT. low )                  &
           WRITE( string, "( ES12.4 )" ) Z_u_P( i )
         full_string( 1 + len_string * ( k - l ) :                             &
                      len_string * ( k + 1 - l ) ) = string
       END DO
       WRITE( out, "( A )" ) full_string
     END DO

     RETURN

!  End of subroutine SUPERB_write_z

     END SUBROUTINE SUPERB_write_z

!  End of module GALAHAD_SUPERB

   END MODULE GALAHAD_SUPERB_double
