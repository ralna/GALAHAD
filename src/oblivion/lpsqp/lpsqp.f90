! THIS VERSION: GALAHAD 2.5 - 09/02/2013 AT 16:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D -  L P S Q P  M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started August 15th 2002
!   originally released GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!A MODULE GALAHAD_LPSQPA_double
!B MODULE GALAHAD_LPSQP_double

!      --------------------------------------------------
!     |                                                  |
!     | Use an l_p SQP approach to solve general         |
!     | nonlinear programmimg problems                   |
!     |                                                  |
!      --------------------------------------------------!  

     USE CUTEst_interface_double
     USE GALAHAD_SYMBOLS
!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SPECFILE_double
!A   USE GALAHAD_QPA_double
!B   USE GALAHAD_LPQPB_double
     USE GALAHAD_SMT_double, ONLY: SMT_put

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: LPSQP_initialize, LPSQP_read_specfile, LPSQP_solve,             &
               LPSQP_terminate

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19

!  ===================================
!  The LPSQP_data_type derived type
!  ===================================

     TYPE, PUBLIC :: LPSQP_data_type
!A     TYPE ( QPA_data_type ) :: QPA_data
!B     TYPE ( LPQPB_data_type ) :: LPQPB_data
       TYPE ( QPT_problem_type ) :: prob
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_stat, C_stat
    END TYPE LPSQP_data_type

!  ======================================
!  The LPSQP_control_type derived type
!  ======================================

     TYPE, PUBLIC :: LPSQP_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit
       INTEGER :: start_print, stop_print, print_gap, linear_solver
       INTEGER :: icfact, semibandwidth, max_sc, io_buffer, more_toraldo
       INTEGER :: non_monotone, first_derivatives, second_derivatives
       REAL ( KIND = wp ) :: stopc, stopg, acccg, initial_radius, maximum_radius
       REAL ( KIND = wp ) :: eta_successful, eta_very_successful
       REAL ( KIND = wp ) :: eta_extremely_successful
       REAL ( KIND = wp ) :: gamma_smallest, gamma_decrease, gamma_increase
       REAL ( KIND = wp ) :: mu_meaningful_model, mu_meaningful_group
       REAL ( KIND = wp ) :: initial_mu, mu_tol, firstg, firstc, infinity
       LOGICAL :: quadratic_problem, two_norm_tr, exact_gcp, magical_steps
       LOGICAL :: accurate_bqp, structured_tr, print_max
       CHARACTER ( LEN = 30 ) :: alive_file
!A     TYPE ( QPA_control_type ) :: QPA_control
!B     TYPE ( LPQPB_control_type ) :: LPQPB_control
     END TYPE LPSQP_control_type

!  =====================================
!  The LPSQP_inform_type derived type
!  =====================================

     TYPE, PUBLIC :: LPSQP_inform_type
       INTEGER :: status, alloc_status, iter, itercg, itcgmx
       INTEGER :: ncalcf, ncalcg, nvar, ngeval, iskip, ifixed, nsemib
       REAL ( KIND = wp ) :: aug, obj, pjgnrm, pr_feas, du_feas
       REAL ( KIND = wp ) :: ratio, mu, radius, ciccg
       LOGICAL :: newsol
       CHARACTER ( LEN = 24 ) :: bad_alloc
!A     TYPE ( QPA_inform_type ) :: QPA_inform
!B     TYPE ( LPQPB_inform_type ) :: LPQPB_inform
     END TYPE LPSQP_inform_type
   CONTAINS

!-*-*  G A L A H A D -  L P S Q P _i n i t i a l i z e  S U B R O U T I N E -*-*

     SUBROUTINE LPSQP_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for LPSQP controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LPSQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( LPSQP_control_type ), INTENT( OUT ) :: control
     TYPE ( LPSQP_inform_type ), INTENT( OUT ) :: inform

!    INTEGER, PARAMETER :: lmin = 1
     INTEGER, PARAMETER :: lmin = 10000

     inform%status = GALAHAD_ok
 
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

!  linear_solver gives the method to be used for solving the
!                linear system. 1=CG, 2=diagonal preconditioned CG,
!                3=user-provided preconditioned CG, 4=expanding band
!                preconditioned CG, 5=Munksgaard's preconditioned CG,
!                6=Schnabel-Eskow modified Cholesky preconditioned CG,
!                7=Gill-Murray-Ponceleon-Saunders modified Cholesky
!                preconditioned CG, 8=band matrix preconditioned CG, 
!                9=Lin-More' preconditioned CG, 11=multifrontal direct
!                method, 12=direct modified multifrontal method

     control%linear_solver = 8

!  The number of vectors allowed in Lin and More's incomplete factorization

     control%icfact = 5

!  The semi-bandwidth of the band factorization

     control%semibandwidth = 5

!   The maximum dimension of the Schur complement 

     control%max_sc = 100

!  Unit number of i/o buffer for writing temporary files (if needed)

     control%io_buffer = 75

!  more_toraldo >= 1 gives the number of More'-Toraldo projected searches 
!                to be used to improve upon the Cauchy point, anything
!                else is for the standard add-one-at-a-time CG search

     control%more_toraldo = 0

!  non-monotone <= 0 monotone strategy used, anything else non-monotone
!                strategy with this history length used.

     control%non_monotone = 1

!  first_derivatives = 0 if exact first derivatives are given, = 1 if forward
!             finite difference approximations are to be calculated, and 
!             = 2 if central finite difference approximations are to be used

     control%first_derivatives = 0

!  second_derivatives specifies the approximation to the second derivatives
!                used. 0=exact, 1=BFGS, 2=DFP, 3=PSB, 4=SR1

     control%second_derivatives = 0

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of violation of the constraints (the "primal infeasibility") is smaller than 
!  control%stopc and the norm of the gradient of the Lagrangian function (the
!  "dual infeasibility") is smaller than control%stopg

     control%stopc = tenm5
     control%stopg = tenm5
     
!  Require a relative reduction in the resuiduals from CG of at least acccg

     control%acccg = 0.01_wp

!  The initial trust-region radius - a non-positive value allows the
!  package to choose its own

     control%initial_radius = - one

!  The largest possible trust-region radius

     control%maximum_radius = ten ** 20

!  Parameters that define when to decrease/increase the trust-region 
!  (specialists only!)

     control%eta_successful = 0.01_wp
     control%eta_very_successful = 0.9_wp
     control%eta_extremely_successful = 0.95_wp
     
     control%gamma_smallest = 0.0625_wp
     control%gamma_decrease = 0.25_wp
     control%gamma_increase = 2.0_wp
     
     control%mu_meaningful_model = 0.01_wp
     control%mu_meaningful_group = 0.1_wp

!  The initial value of the penalty parameter

     control%initial_mu = point1

!  The value of the penalty parameter above which the algorithm
!  will not attempt to update the estimates of the Lagrange multipliers

     control%mu_tol = point1

!  The required accuracy of the norm of the projected gradient at the end
!  of the first major iteration

     control%firstg = point1

!  The required accuracy of the norm of the constraints at the end
!  of the first major iteration

     control%firstc = point1

!  Any bound larger than infinity in absolute value is infinite

     control%infinity = infinity

!  Is the function quadratic ? 

     control%quadratic_problem = .FALSE.

!  two_norm_tr is true if a 2-norm trust-region is to be used, and false 
!                for the infinity norm

     control%two_norm_tr = .FALSE.

!  exact_gcp is true if the exact Cauchy point is required, and false if an
!                approximation suffices

     control%exact_gcp = .TRUE.

!  magical_steps is true if magical steps are to be used to improve upon
!                already accepted points, and false otherwise

     control%magical_steps = .FALSE.

!  accurate_bqp is true if the the minimizer of the quadratic model within
!                the intersection of the trust-region and feasible box
!                is to be sought (to a prescribed accuracy), and false 
!                if an approximation suffices

     control%accurate_bqp = .FALSE.

!  structured_tr is true if a structured trust region will be used, and false
!                if a standard trust-region suffices

     control%structured_tr = .FALSE.

!  For printing, if we are maximizing rather than minimizing, print_max
!  should be .TRUE.

     control%print_max = .FALSE.

!  Initialize QPA or LPQPB control parameters

!A   CALL QPA_initialize( data%QPA_data, control%QPA_control,                  &
!A                        inform%QPA_inform )
!B   CALL LPQPB_initialize( data%LPQPB_data, control%LPQPB_control,            &
!B                          inform%LPQPB_inform )

     RETURN

!  End of subroutine LPSQP_initialize

     END SUBROUTINE LPSQP_initialize

!-*-*-   L P S Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE LPSQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by LPSQP_initialize could (roughly) 
!  have been set as:

! BEGIN LPSQP SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  alive-device                                   60
!  print-level                                    0
!  maximum-number-of-iterations                   1000
!  start-print                                    -1 
!  stop-print                                     -1
!  linear-solver-used                             BAND_CG
!  number-of-lin-more-vectors-used                5
!  semi-bandwidth-for-band-preconditioner         5
!  maximum-dimension-of-schur-complement          100
!  unit-number-for-temporary-io                   75
!  more-toraldo-search-length                     0
!  history-length-for-non-monotone-descent        0
!  first-derivative-approximations                EXACT
!  second-derivative-approximations               SR1
!  primal-accuracy-required                       1.0D-5
!  dual-accuracy-required                         1.0D-5
!  inner-iteration-relative-accuracy-required     0.01
!  initial-trust-region-radius                    -1.0
!  maximum-radius                                 1.0D+20
!  eta-successful                                 0.01
!  eta-very-successful                            0.9
!  eta-extremely-successful                       0.95
!  gamma-smallest                                 0.0625
!  gamma-decrease                                 0.25
!  gamma-increase                                 2.0
!  mu-meaningful-model                            0.01
!  mu-meaningful-group                            0.1
!  initial-penalty-parameter                      0.1
!  no-dual-updates-until-penalty-parameter-below  0.1
!  initial-dual-accuracy-required                 0.1
!  initial-primal-accuracy-required               0.1
!  infinity-value                                 1.0D+19
!  quadratic-problem                              NO
!  two-norm-trust-region-used                     NO
!  exact-GCP-used                                 YES
!  magical-steps-allowed                          NO
!  subproblem-solved-accuractely                  NO
!  structured-trust-region-used                   NO
!  print-for-maximimization                       NO
!  alive-filename                                 ALIVE.d
! END LPSQP SPECIFICATIONS

!  Dummy arguments

     TYPE ( LPSQP_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: lspec = 43
     CHARACTER( LEN = 5 ), PARAMETER :: specname = 'LPSQP'
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
     spec(  9 )%keyword = 'linear-solver-used'
     spec( 10 )%keyword = 'number-of-lin-more-vectors-used'
     spec( 11 )%keyword = 'semi-bandwidth-for-band-preconditioner'
     spec( 12 )%keyword = 'maximum-dimension-of-schur-complement'
     spec( 13 )%keyword = 'unit-number-for-temporary-io'
     spec( 14 )%keyword = 'more-toraldo-search-length'
     spec( 15 )%keyword = 'history-length-for-non-monotone-descent'
     spec( 16 )%keyword = 'first-derivative-approximations'
     spec( 17 )%keyword = 'second-derivative-approximations'

!  Real key-words

     spec( 18 )%keyword = 'primal-accuracy-required'
     spec( 19 )%keyword = 'dual-accuracy-required'
     spec( 20 )%keyword = 'inner-iteration-relative-accuracy-required'
     spec( 21 )%keyword = 'initial-trust-region-radius'
     spec( 22 )%keyword = 'maximum-radius'
     spec( 23 )%keyword = 'eta-successful'
     spec( 24 )%keyword = 'eta-very-successful'
     spec( 25 )%keyword = 'eta-extremely-successful'
     spec( 26 )%keyword = 'gamma-smallest'
     spec( 27 )%keyword = 'gamma-decrease'
     spec( 28 )%keyword = 'gamma-increase'
     spec( 29 )%keyword = 'mu-meaningful-model'
     spec( 30 )%keyword = 'mu-meaningful-group'
     spec( 31 )%keyword = 'initial-penalty-parameter'
     spec( 32 )%keyword = 'no-dual-updates-until-penalty-parameter-below'
     spec( 33 )%keyword = 'initial-dual-accuracy-required'
     spec( 34 )%keyword = 'initial-primal-accuracy-required'
     spec( 35 )%keyword = 'infinity-value'

!  Logical key-words

     spec( 36 )%keyword = 'quadratic-problem'
     spec( 37 )%keyword = 'two-norm-trust-region-used'
     spec( 38 )%keyword = 'exact-GCP-used'
     spec( 39 )%keyword = 'magical-steps-allowed'
     spec( 40 )%keyword = 'subproblem-solved-accuractely'
     spec( 41 )%keyword = 'structured-trust-region-used'
     spec( 42 )%keyword = 'print-for-maximimization'

!  Character key-words

     spec( 43 )%keyword = 'alive-filename'

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
     CALL SPECFILE_assign_symbol( spec( 9 ), control%linear_solver,            &
                                  control%error )                           
     CALL SPECFILE_assign_integer( spec( 10 ), control%icfact,                 &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 11 ), control%semibandwidth,          &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 12 ), control%max_sc,                 &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 13 ), control%io_buffer,              &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 14 ), control%more_toraldo,           &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 15 ), control%non_monotone,           &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 16 ), control%first_derivatives,       &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 17 ), control%second_derivatives,      &
                                  control%error )                           

!  Set real values

     CALL SPECFILE_assign_real( spec( 18 ), control%stopc,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 19 ), control%stopg,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 20 ), control%acccg,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 21 ), control%initial_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 22 ), control%maximum_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 23 ), control%eta_successful,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 24 ), control%eta_very_successful,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 25 ), control%eta_extremely_successful,  &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 26 ), control%gamma_smallest,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 27 ), control%gamma_decrease,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 28 ), control%gamma_increase,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 29 ), control%mu_meaningful_model,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 30 ), control%mu_meaningful_group,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 31 ), control%initial_mu,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 32 ), control%mu_tol,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 33 ), control%firstg,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 34 ), control%firstc,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 35 ), control%infinity,                  &
                                control%error )                           

!  Set logical values

     CALL SPECFILE_assign_logical( spec( 36 ), control%quadratic_problem,      &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 37 ), control%two_norm_tr,            &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 38 ), control%exact_gcp,              &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 39 ), control%magical_steps,          &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 40 ), control%accurate_bqp,           &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 41 ), control%structured_tr,          &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 42 ), control%print_max,              &
                                   control%error )                           

!  Set character values

     CALL SPECFILE_assign_string( spec( 43 ), control%alive_file,              &
                                  control%error )                           

!  Assign values for QPA or LPQPB control parameters

     IF ( PRESENT( alt_specname ) ) THEN
!A     CALL QPA_read_specfile( control%QPA_control, device,                    &
!A                             alt_specname = TRIM( alt_specname ) // '-QPA' )
!B     CALL LPQPB_read_specfile( control%LPQPB_control, device,                &
!B                             alt_specname = TRIM( alt_specname ) // '-LPQPB')
     ELSE
!A     CALL QPA_read_specfile( control%QPA_control, device )
!B     CALL LPQPB_read_specfile( control%LPQPB_control, device )
     END IF

     RETURN

     END SUBROUTINE LPSQP_read_specfile

!-*-*-*-*-*  G A L A H A D -  LPSQP_solve  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE LPSQP_solve( input, io_buffer, rho, one_norm, control, inform, &
                             data )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  LPSQP_solve, a method for finding a local minimizer of a function subject 
!  to general constraints and simple bounds on the sizes of the variables.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: input
     REAL ( KIND = wp ), INTENT( INOUT ) :: rho
     LOGICAL, INTENT( IN ) :: one_norm
     TYPE ( LPSQP_control_type ), INTENT( INOUT ) :: control
     TYPE ( LPSQP_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( LPSQP_data_type ), INTENT( INOUT ) :: data

!  Local variables

     INTEGER :: m, n, H_ne, J_ne, J_len, H_len, print_level_lpqps, ir, ic, l_suc
     INTEGER :: alloc_status, i, j, l, start_print, stop_print, print_level, out
     INTEGER :: nfacts, nmhist, maxit_qp, prob_m, cutest_status, io_buffer
     REAL ( KIND = wp ) :: merit, merit_trial, f_trial, step, prfeas
     REAL ( KIND = wp ) :: ared, pred, ratio, old_radius, violation_trial
     REAL ( KIND = wp ) :: model, ar_h, pr_h, epsmch, teneps
     REAL ( KIND = wp ) :: merit_min, merit_ref, merit_current, sigma_r, sigma_c
!    REAL ( KIND = wp ) :: first_radius
     REAL ( KIND = wp ), PARAMETER :: rho_u = 0.01
     REAL ( KIND = wp ), PARAMETER :: rho_s = 0.9
     REAL ( KIND = wp ), PARAMETER :: step_tiny = ten ** ( - 6 )
     REAL ( KIND = wp ), PARAMETER :: stop_tiny = ten ** ( - 6 )
!    LOGICAL, PARAMETER :: fulsol = .TRUE.
     LOGICAL, PARAMETER :: fulsol = .FALSE.
     REAL :: time, time_new
     LOGICAL :: grlagf, successful, new_inner
!    LOGICAL :: set_first_radius
     CHARACTER ( LEN = 10 ) :: pname
     CHARACTER ( LEN = 20 ) :: bad_alloc
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, X_l, X_u, X_trial
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, C_l, C_u, C_trial, Y
     LOGICAL, ALLOCATABLE, DIMENSION( : ) :: EQUATN, LINEAR
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: X_name, C_name

!  Initialize

     CALL CPU_TIME( time )

     inform%iter = 0
     IF ( control%initial_radius > zero ) THEN
       inform%radius = control%initial_radius
     ELSE
       inform%radius = one
     END IF
!    first_radius = inform%radius
!    set_first_radius = .FALSE.

     epsmch = EPSILON( one ) ; teneps = ten * epsmch

!  nmhist is the length of the history if a non-monotone strategy is to be used
  
     nmhist = control%non_monotone

!  ===========================
!  Control the output printing
!  ===========================

     out = 6

!A   print_level_lpqps = control%QPA_control%print_level
!B   print_level_lpqps = control%LPQPB_control%print_level

     IF ( control%start_print < 0 ) THEN
       start_print = 0
     ELSE
       start_print = control%start_print
     END IF

     IF ( control%stop_print < 0 ) THEN
       stop_print = control%maxit
     ELSE
       stop_print = control%stop_print
     END IF

     IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
       print_level = control%print_level
!A     control%QPA_control%print_level = print_level_lpqps
!B     control%LPQPB_control%print_level = print_level_lpqps
!B     control%LPQPB_control%QPB_control%print_level = print_level_lpqps
     ELSE
       print_level = 0
!A     control%QPA_control%print_level = 0
!B     control%LPQPB_control%print_level = 0
!B     control%LPQPB_control%QPB_control%print_level = 0
     END IF

!  Discover how many variables and constraints are involved in the problem

     CALL CUTEST_cdimen( cutest_status, input, n, m )
     IF ( cutest_status /= 0 ) GO TO 930
     prob_m = m + 1

!  Allocate sufficient space to hold the problem

     ALLOCATE( X( n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'X' ; GO TO 910; END IF

     ALLOCATE( X_l( n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'X_l' ; GO TO 910; END IF

     ALLOCATE( X_u( n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'X_u' ; GO TO 910; END IF

     ALLOCATE( Y( prob_m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'Y' ; GO TO 910; END IF

     ALLOCATE( C_l( prob_m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'C_l' ; GO TO 910; END IF

     ALLOCATE( C_u( prob_m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'C_u' ; GO TO 910; END IF

     ALLOCATE( EQUATN( prob_m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'EQUATN' ; GO TO 910; END IF

     ALLOCATE( LINEAR( prob_m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'LINEAR' ; GO TO 910; END IF

     data%prob%n = n
     data%prob%m = prob_m

!  Set up the correct data structures for subsequent computations.

     CALL CUTEST_csetup( cutest_status, input, control%error, io_buffer,       &
                         data%prob%n, data%prob%m, X, X_l, X_u,                &
                         Y, C_l, C_u, EQUATN, LINEAR, 0, 0, 0 )
     IF ( cutest_status /= 0 ) GO TO 930

     Y( prob_m ) = zero
     C_l( prob_m ) = - ten * infinity
!    C_u( prob_m ) = zero
     C_u( prob_m ) = ten * infinity

     data%prob%m = prob_m

!    write(out,"('cl',/,(5ES12.4))") C_l
!    write(out,"('cu',/,(5ES12.4))") C_u
     DEALLOCATE( EQUATN, LINEAR, STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'EQUATN' ; GO TO 910; END IF

!  Determine how many nonzeros are required to store the matrix of 
!  gradients of the objective function and constraints, when the matrix 
!  is stored in sparse format.

     CALL CUTEST_cdimsj( cutest_status, J_ne )
     IF ( cutest_status /= 0 ) GO TO 930

!  Determine how many nonzeros are required to store the Hessian matrix of the
!  Lagrangian, when the matrix is stored as  sparse matrix in "co-ordinate" 
!  format
 
     CALL CUTEST_cdimsh( cutest_status, H_ne )
     IF ( cutest_status /= 0 ) GO TO 930

!  Allocate further space to hold the problem

     ALLOCATE( C_name( prob_m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'C_name' ; GO TO 910; END IF

     ALLOCATE( X_name( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'X_name' ; GO TO 910; END IF

     ALLOCATE( C( prob_m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'C' ; GO TO 910; END IF

     ALLOCATE( data%prob%G( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%G' ; GO TO 910; END IF

     ALLOCATE( data%prob%A%row( J_ne ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%A%row' ; GO TO 910; END IF

     ALLOCATE( data%prob%A%col( J_ne ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%A%col' ; GO TO 910; END IF

     ALLOCATE( data%prob%A%val( J_ne ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%A%val' ; GO TO 910; END IF

     ALLOCATE( data%prob%H%row( H_ne ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%H%row' ; GO TO 910; END IF

     ALLOCATE( data%prob%H%col( H_ne ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%H%col' ; GO TO 910; END IF

     ALLOCATE( data%prob%H%val( H_ne ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%H%val' ; GO TO 910; END IF

!  Ensure that the initial point is feasible with respect to its simple bounds

     prfeas = zero
     DO i = 1, data%prob%n
       X( i ) = MIN( MAX( X( i ), X_l( i ) + prfeas ), X_u( i ) - prfeas ) 
     END DO

!  Obtain the names of the problem, its variables and general constraints

     CALL CUTEST_cnames( cutest_status, data%prob%n, m, pname, X_name,         &
                         C_name( : m ) )
     IF ( cutest_status /= 0 ) GO TO 930
     C_name( prob_m ) = 'slope'
     C( prob_m ) = zero

!  Evaluate the objective and general constraint function values
   
     CALL CUTEST_cfn( cutest_status, data%prob%n, m, X, data%prob%f, C( : m ) )
     IF ( cutest_status /= 0 ) GO TO 930

!    write(out,"( 'x', /, (5ES12.4 ))" ) X
!    write(out,"( 'f', ES12.4)" ) data%prob%f
!    write(out,"( 'c', /, (5ES12.4 ))" ) C
!    write(out,"( 'norm', L1 )" ) one_norm

!  Compute the value of the merit function

     merit = LPSQP_merit( m, data%prob%f, rho, one_norm, C( : m ),             &
                          C_l( : m ), C_u( : m ), inform%pr_feas )

!    write(out,"( 'rho', ES12.4 )" ) rho
!    write(out,"( 'merit, violation', 2ES12.4 )" ) merit, inform%pr_feas
     IF ( control%out > 0 .AND. print_level > 0 ) THEN
       WRITE( control%out, 2050 ) rho
     END IF

!  If a non-monotone method is to be used, initialize counters

     IF ( nmhist > 0 ) THEN
       l_suc = 0
       merit_min = merit ; merit_ref = merit_min ; merit_current = merit_min
       sigma_r = zero ; sigma_c = zero
     END IF

!  Allocate sufficient space to hold the problem

     ALLOCATE( X_trial( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'X_trial' ; GO TO 910; END IF

     ALLOCATE( C_trial( data%prob%m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'C_trial' ; GO TO 910; END IF

     ALLOCATE( data%prob%X( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%X' ; GO TO 910; END IF

     ALLOCATE( data%prob%X_l( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%X_l' ; GO TO 910; END IF

     ALLOCATE( data%prob%X_u( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%X_u' ; GO TO 910; END IF

     ALLOCATE( data%prob%Y( data%prob%m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%Y' ; GO TO 910; END IF

     ALLOCATE( data%prob%Z( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%Z' ; GO TO 910; END IF

     ALLOCATE( data%prob%C( data%prob%m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%C' ; GO TO 910; END IF

     ALLOCATE( data%prob%C_l( data%prob%m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%C_l' ; GO TO 910; END IF

     ALLOCATE( data%prob%C_u( data%prob%m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%C_u' ; GO TO 910; END IF

     ALLOCATE( data%prob%A%ptr( data%prob%m + 1 ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%A%ptr' ; GO TO 910; END IF

     ALLOCATE( data%prob%H%ptr( data%prob%n + 1 ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'data%prob%H%ptr' ; GO TO 910; END IF

     ALLOCATE( data%C_stat( data%prob%m ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'C_stat' ; GO TO 910; END IF

     ALLOCATE( data%B_stat( data%prob%n ), STAT = alloc_status ) 
     IF ( alloc_status /= 0 ) THEN 
       bad_alloc = 'B_stat' ; GO TO 910; END IF

     data%prob%Z = one
     data%prob%new_problem_structure = .TRUE.
     successful = .TRUE.
     new_inner = .TRUE.
!A   maxit_qp = control%QPA_control%maxit
!B   maxit_qp = control%LPQPB_control%QPB_control%maxit

!A   data%prob%rho_g = rho
!A   data%prob%rho_b = ten ** 6
!A   control%QPA_control%cold_start = 1
!A   control%QPA_control%solve_within_bounds = .TRUE.

     data%prob%Y = Y

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      O U T E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  10 CONTINUE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      I N N E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  20   CONTINUE

!  If required, compute derivative values

       IF ( successful ) THEN

!  Evaluate both the gradients of the general constraint functions
!  and the Hessian matrix of the Lagrangian function for the problem.
!  The Hessian is stored as a sparse matrix in "co-ordinate" format. 
!  Also obtain the gradient of either the objective function or
!  the Lagrangian function. The data is stored in a sparse format.

         grlagf = .FALSE. ; J_len = J_ne ; H_len = H_ne
         CALL CUTEST_csgrsh( cutest_status, data%prob%n, m, X, - Y( : m ),     &
                             grlagf, J_ne, J_len, data%prob%A%val,             &
                             data%prob%A%col, data%prob%A%row, data%prob%H%ne, &
                             H_len, data%prob%H%val, data%prob%H%row,          &
                             data%prob%H%col )
         IF ( cutest_status /= 0 ) GO TO 930

!  Untangle A: separate the gradient terms from the constraint Jacobian

         data%prob%A%ne = 0 ; data%prob%G( : data%prob%n ) = zero
         DO i = 1, J_ne
           IF ( data%prob%A%row( i ) == 0 ) THEN
             data%prob%G( data%prob%A%col( i ) ) = data%prob%A%val( i )
!  Hold the extra constraint g^T s <= 0
             data%prob%A%ne = data%prob%A%ne + 1
             data%prob%A%row( data%prob%A%ne ) = prob_m
             data%prob%A%col( data%prob%A%ne ) = data%prob%A%col( i )
             data%prob%A%val( data%prob%A%ne ) = data%prob%A%val( i )
           ELSE
             data%prob%A%ne = data%prob%A%ne + 1
             data%prob%A%row( data%prob%A%ne ) = data%prob%A%row( i )
             data%prob%A%col( data%prob%A%ne ) = data%prob%A%col( i )
             data%prob%A%val( data%prob%A%ne ) = data%prob%A%val( i )
           END IF
         END DO

         IF ( ALLOCATED( data%prob%H%type ) ) DEALLOCATE( data%prob%H%type )
         CALL SMT_put( data%prob%H%type, 'COORDINATE', alloc_status )
         IF ( ALLOCATED( data%prob%A%type ) ) DEALLOCATE( data%prob%A%type )
         CALL SMT_put( data%prob%A%type, 'COORDINATE', alloc_status )

!  Ensure that only entries from lower triangle of H are given

         DO l = 1, data%prob%H%ne
           i = data%prob%H%row( l ) ; j = data%prob%H%col( l )
           IF ( i < j ) THEN
             data%prob%H%row( l ) = j ; data%prob%H%col( l ) = i
           END IF
         END DO

!  Compute the gradient of the Lagrangian (put it in data%prob%X)

!        WRITE( out, "( ' g ', /, ( 5ES12.4 ) )" ) data%prob%G( : n )
!        WRITE( out, "( ' z ', /, ( 5ES12.4 ) )" ) data%prob%Z( : n )
!        WRITE( out, "( ' y ', /, ( 5ES12.4 ) )" ) Y( : m )
         data%prob%X( : n ) = data%prob%G( : n ) - data%prob%Z( : n )
         DO i = 1, data%prob%A%ne
           data%prob%X( data%prob%A%col( i ) ) =                               &
             data%prob%X( data%prob%A%col( i ) ) - data%prob%A%val( i ) *      &
               Y( data%prob%A%row( i ) )
         END DO
         inform%du_feas = MAXVAL( ABS( data%prob%X( : n ) ) )
!        WRITE(out,"( ' KKT violation ', ES12.4 )" ) inform%du_feas
!        IF ( inform%du_feas < step_tiny ) GO TO 800
         
!B       control%LPQPB_control%reformulate = .TRUE.
       ELSE
!B       control%LPQPB_control%reformulate = .FALSE.
       END IF

!  Print a summary of the last iteration

       IF ( control%out > 0 .AND. print_level > 0 ) THEN
         IF ( inform%iter > 0 ) THEN
!A         IF ( control%QPA_control%print_level > 0 .OR.                       &
!B         IF ( control%LPQPB_control%print_level > 0 .OR.                     &
                print_level > 1 ) WRITE( control%out, 2030 )
           IF ( new_inner ) THEN
             WRITE( control%out,                                               &
             "( I6, ES12.4, 2ES9.2, '     -   ', ES9.2,                        &
            &   '      -          -' )" )                                      &
               inform%iter, merit, inform%pr_feas, inform%du_feas, inform%radius
           ELSE
             WRITE( control%out, "( I6, ES12.4, 4ES9.2, ES10.2, I8 )" )        &
               inform%iter, merit, inform%pr_feas, inform%du_feas,             &
               step, old_radius, ratio, nfacts
           END IF
         ELSE
           WRITE( control%out, 2030 )
           WRITE( control%out,                                                 &
             "( I6, ES12.4, ES9.2, 2( '     -   ' ), ES9.2,                    &
           &    '      -          - ' )" )                                     &
               inform%iter, merit, inform%pr_feas, inform%radius
         END IF
       END IF

!  Start the next iteration

       inform%iter = inform%iter + 1

       IF ( inform%iter > control%maxit ) THEN
         inform%status = - 10
         RETURN 
       END IF

       IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
         print_level = control%print_level
!A       control%QPA_control%print_level = print_level_lpqps
!B       control%LPQPB_control%print_level = print_level_lpqps
!B       control%LPQPB_control%QPB_control%print_level = print_level_lpqps
       ELSE
         print_level = 0
!A       control%QPA_control%print_level = 0
!B       control%LPQPB_control%print_level = 0
!B       control%LPQPB_control%QPB_control%print_level = 0
       END IF

       new_inner = .FALSE.

!  Start inner loop to minimize penalty function for the current value of mu

       data%prob%X = zero
       data%prob%C = zero
!      data%prob%X( 1 ) = 0.0
!      data%prob%X( 2 ) = 0.0000D+00
!      data%prob%X( 3 ) = -8.3333333333D-01
       WHERE( X_l > - control%infinity )
         data%prob%X_l( : n ) = MAX( - inform%radius, X_l - X )
       ELSEWHERE
         data%prob%X_l( : n ) = - inform%radius
       END WHERE

       WHERE( X_u < control%infinity )
         data%prob%X_u( : n ) = MIN( inform%radius, X_u - X )
       ELSEWHERE
         data%prob%X_u( : n ) = inform%radius
       END WHERE

       WHERE( C_l > - control%infinity )
         data%prob%C_l( : prob_m ) = C_l - C
       ELSEWHERE
         data%prob%C_l( : prob_m ) = C_l
       END WHERE

       WHERE( C_u < control%infinity )
         data%prob%C_u( : prob_m ) = C_u - C
       ELSEWHERE
         data%prob%C_u( : prob_m ) = C_u
       END WHERE

!  Solve the l_p QP to find a search direction

       IF ( print_level > 10 ) THEN
         WRITE( out, "( 'n, m = ', 2I6, ' obj = ', ES12.4 )" )                 &
           data%prob%n, data%prob%m, data%prob%f
         WRITE( out, "( ' g ', /, ( 5ES12.4 ) )" ) data%prob%G( : n )
         WRITE( out, "( ' x_l ', /, ( 5ES12.4 ) )" ) data%prob%X_l( : n )
         WRITE( out, "( ' x_u ', /, ( 5ES12.4 ) )" ) data%prob%X_u( : n )
         WRITE( out, "( ' c_l ', /, ( 5ES12.4 ) )" ) data%prob%C_l( : m )
         WRITE( out, "( ' c_u ', /, ( 5ES12.4 ) )" ) data%prob%C_u( : m )
         WRITE( out, "( ' A_row ', /, ( 10I6 ) )" )                            &
           data%prob%A%row( : data%prob%A%ne )
         WRITE( out, "( ' A_col ', /, ( 10I6 ) )" )                            &
           data%prob%A%col( : data%prob%A%ne )
         WRITE( out, "( ' A_val ', /, ( 5ES12.4 ) )" )                         &
           data%prob%A%val( : data%prob%A%ne )
         WRITE( out, "( ' H_row ', /, ( 10I6 ) )" )                            &
           data%prob%H%row( : data%prob%H%ne )
         WRITE( out, "( ' H_col ', /, ( 10I6 ) )" )                            &
           data%prob%H%col( : data%prob%H%ne )
         WRITE( out, "( ' H_val ', /, ( 5ES12.4 ) )" )                         &
           data%prob%H%val( : data%prob%H%ne )
       END IF

   110 CONTINUE
!A     CALL QPA_solve( data%prob, data%C_stat, data%B_stat,                    &
!A                     data%QPA_data, control%QPA_control, inform%QPA_inform )
!B     CALL LPQPB_solve( data%prob, rho, one_norm, data%LPQPB_data,            &
!B                       control%LPQPB_control, inform%LPQPB_inform )

!A     nfacts = inform%QPA_inform%nfacts
!B     nfacts = inform%LPQPB_inform%QPB_inform%nfacts

!A     IF ( inform%QPA_inform%status /= 0 .AND.                                &
!A       inform%QPA_inform%status /= GALAHAD_error_max_iterations  .AND.       &
!A       inform%QPA_inform%status /= GALAHAD_error_tiny_step ) THEN
!A       inform%status = inform%QPA_inform%status
!A       WRITE( control%out, "( ' On exit from QPA_solve, status = ', I6 )" )  &
!A         inform%QPA_inform%status
!B     IF ( inform%LPQPB_inform%status /= 0 .AND.                              &
!B       inform%LPQPB_inform%status /= GALAHAD_error_max_iterations  .AND.     &
!B       inform%LPQPB_inform%status /= GALAHAD_error_tiny_step ) THEN
!B       inform%status = inform%LPQPB_inform%status
!B       WRITE( control%out, "( ' On exit from LPQPB_solve, status = ', I6 )" )&
!B         inform%LPQPB_inform%status
         RETURN
       END IF

       step = MAXVAL( ABS( data%prob%X( : n ) ) )

!A     IF ( step < step_tiny .AND.                                             &
!A          inform%QPA_inform%status == GALAHAD_error_max_iterations  ) THEN
!B     IF ( step < step_tiny .AND.                                             &
!B          inform%LPQPB_inform%status == GALAHAD_error_max_iterations  ) THEN
!A       control%QPA_control%maxit = control%QPA_control%maxit                 &
!B       control%LPQPB_control%QPB_control%maxit =                             &
!B         control%LPQPB_control%QPB_control%maxit                             &
           + maxit_qp
         WRITE( out, "( ' doubling maxit ' )" )
!A       control%QPA_control%cold_start = 0
!B       control%LPQPB_control%reformulate = .FALSE.
         GO TO 110
       END IF

!  Find the new point

       X_trial = X + data%prob%X( : n )
!      WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" ) X( : n )
!      WRITE( out, "( ' step ', /, ( 5ES12.4 ) )" ) data%prob%X( : n )
!      WRITE( out, "( ' y    ', /, ( 5ES12.4 ) )" ) data%prob%Y( : m )
!      WRITE( out, "( ' x_trial ', /, ( 4ES16.8 ) )" ) X_trial

       old_radius = inform%radius
       ratio = - one

!  Compute the new function and gradient values

       CALL CUTEST_cfn( cutest_status, data%prob%n, m, X_trial, f_trial,       &
                        C_trial( : m ) )
       IF ( cutest_status /= 0 ) GO TO 930
!      write(out,"( ' f_trial ', /, ES16.8 )" ) f_trial
!      write(out,"( ' c_trial ', /, ( 4ES16.8 ) )" ) C_trial( : m )

!  Compute the value of the merit function

       merit_trial = LPSQP_merit( m, f_trial, rho, one_norm,                   &
                                  C_trial( : m ), C_l( : m ) , C_u( : m ),     &
                                  violation_trial )

!A     model = inform%QPA_inform%merit
!B     model = inform%LPQPB_inform%QPB_inform%obj
       ared = merit - merit_trial
       pred = merit - model
       IF ( ared == zero .AND. pred == zero ) GO TO 800
       ared = ared + MAX( one, ABS( merit ) ) * teneps
       pred = pred + MAX( one, ABS( merit ) ) * teneps
       IF ( pred > zero ) THEN
         ratio = ared / pred 
       ELSE
         IF ( control%out > 0 .AND. print_level > 0 .AND. step > stop_tiny )   &
           WRITE( control%out, "( ' --> predicted reduction =', ES10.2 )" )    &
             pred
       END IF

!!A    write(out,"(3ES12.4)") merit, inform%QPA_inform%merit, merit_trial
!!B    write(out,"(3ES12.4)") merit, inform%LPQPB_inform%QPB_inform%obj,       &
!!B       merit_trial
!      write(out,"( ES16.8 )" ) ratio

!  Adjust ratio in the non-monotone case

       IF ( nmhist > 0 ) THEN
         ar_h = ( merit_ref - merit_trial )                                    &
                  + MAX( one, ABS( merit_ref ) ) * teneps
         pr_h = sigma_r + pred
         IF ( ABS( ar_h ) < teneps .AND. ABS( merit_ref ) > teneps ) ar_h = pr_h
         ratio = MAX( ratio, ar_h / pr_h )
       END IF

!  - - - - - - - - - - unsuccessful step - - - - - - - - - - - - - - - -

!      IF ( ratio < rho_u .OR. pred <= zero ) THEN
!      IF ( ratio < rho_u .AND. inform%iter < 18 ) THEN
       IF ( ratio < rho_u ) THEN
         successful = .FALSE.
         DO
           inform%radius = half * inform%radius
           IF ( inform%radius < step ) EXIT
         END DO
         IF ( pred <= zero .AND. ( step <= step_tiny .OR.                     &
              MAX( inform%pr_feas, inform%du_feas ) <= stop_tiny ) ) THEN
           merit = merit_trial ; data%prob%f = f_trial
           inform%pr_feas = violation_trial
           X = X_trial ; C = C_trial
           Y = data%prob%Y( : data%prob%m )
           successful = .TRUE.
!A         control%QPA_control%maxit = maxit_qp
!B         control%LPQPB_control%QPB_control%maxit = maxit_qp
!A         control%QPA_control%cold_start = 0
         ELSE
!A         control%QPA_control%cold_start = 1
         END IF
       ELSE

!  - - - - - - - - - - - successful step - - - - - - - - - - - - - - - -

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

         merit = merit_trial ; data%prob%f = f_trial
         inform%pr_feas = violation_trial
         X = X_trial ; C( : m ) = C_trial( : m )
         Y = data%prob%Y( : data%prob%m )

!        WRITE( out, "( ' x ', /, ( 5ES12.4 ) )" ) X( : n )
!        WRITE( out, "( ' c ', /, ( 5ES12.4 ) )" ) C( : m )
!        WRITE( out, "( ' y ', /, ( 5ES12.4 ) )" ) Y( : m )

         successful = .TRUE.
!A       control%QPA_control%maxit = maxit_qp
!B       control%LPQPB_control%QPB_control%maxit = maxit_qp
!A       control%QPA_control%cold_start = 0
!        IF ( .NOT. set_first_radius ) THEN
!          set_first_radius = .TRUE.
!          first_radius = inform%radius
!        END IF
!        IF ( ratio > rho_s .OR. inform%du_feas < 0.01 )                       &
         IF ( ratio > rho_s )                                                  &
           inform%radius = two * inform%radius

       END IF

!!A     control%QPA_control%randomize =                                        & 
!!A        MAX( inform%pr_feas, inform%du_feas ) > 0.001

       IF ( step > step_tiny .AND.                                             &
            MAX( inform%pr_feas, inform%du_feas ) > stop_tiny )  GO TO 20

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    I N N E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 800 CONTINUE
     Y = data%prob%Y( : data%prob%m )
     data%prob%X( : n ) = data%prob%G( : n ) - data%prob%Z( : n )
     DO i = 1, data%prob%A%ne
       data%prob%X( data%prob%A%col( i ) ) =                                   &
         data%prob%X( data%prob%A%col( i ) ) - data%prob%A%val( i ) *          &
           Y( data%prob%A%row( i ) )
     END DO
     inform%du_feas = MAXVAL( ABS( data%prob%X( : n ) ) )
!    WRITE(out,"( ' KKT violation ', ES12.4 )" ) inform%du_feas


     IF ( control%out > 0 .AND. print_level > 0 ) THEN
!A     IF ( control%QPA_control%print_level > 0 .OR.                           &
!B     IF ( control%LPQPB_control%print_level > 0 .OR.                         &
         print_level > 1 ) WRITE( control%out, 2030 )
       WRITE( control%out, "( I6, ES12.4, 4ES9.2, ES10.2, I8 )" )              &
         inform%iter, merit, inform%pr_feas, inform%du_feas,                   &
         step, old_radius, ratio, nfacts
     END IF

!  Increase the penalty parameter if not feasible

     IF ( inform%pr_feas > ten ** ( - 6 ) ) THEN

       rho = ten * rho
!A     data%prob%rho_g = rho ; data%prob%rho_b = ten ** 6
!!     inform%radius = first_radius

!  Compute the value of the merit function

       merit = LPSQP_merit( m, data%prob%f, rho, one_norm, C( : m ),           &
                            C_l( : m ), C_u( : m ), inform%pr_feas )

       IF ( control%out > 0 .AND. print_level > 0 ) THEN
         WRITE( control%out, 2050 ) rho
         WRITE( control%out, 2030 )
       END IF
       new_inner = .TRUE.
       GO TO 10
     END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    O U T E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

     l = 4
     IF ( fulsol ) l = data%prob%n 
     IF ( control%print_level >= 10 ) l = data%prob%n

     WRITE( out, 2000 )
     DO j = 1, 2 
       IF ( j == 1 ) THEN 
         ir = 1 ; ic = MIN( l, data%prob%n ) 
       ELSE 
         IF ( ic < data%prob%n - l ) WRITE( out, 2040 ) 
         ir = MAX( ic + 1, data%prob%n - ic + 1 ) ; ic = data%prob%n
       END IF 
       DO i = ir, ic 
         WRITE( out, 2020 ) i, X_name( i ), X( i ), X_l( i ), X_u( i ),        &
           data%prob%Z( i )
       END DO
     END DO

     IF ( data%prob%m > 0 ) THEN
       l = 4
       IF ( fulsol ) l = data%prob%m
       IF ( control%print_level >= 10 ) l = data%prob%m

       WRITE( out, 2010 )
       DO j = 1, 2 
         IF ( j == 1 ) THEN 
           ir = 1 ; ic = MIN( l, data%prob%m ) 
         ELSE 
           IF ( ic < data%prob%m - l ) WRITE( out, 2040 ) 
           ir = MAX( ic + 1, data%prob%m - ic + 1 ) ; ic = data%prob%m
         END IF 
         DO i = ir, ic 
           WRITE( out, 2020 ) i, C_name( i ), C( i ), C_l( i ), C_u( i ), Y( i )
         END DO
       END DO
     END IF

     CALL CPU_TIME( time_new ) ; time = time_new - time

!A   WRITE( out, "( /, ' Solver:        LPSQPA', /, ' Problem: ', 6X, A10, /,  &
!B   WRITE( out, "( /, ' Solver:        LPSQP',  /, ' Problem: ', 6X, A10, /,  &
    &   ' Objective  = ', ES16.8, /, ' Violation  = ', ES12.4, /,              &
    &  ' Iterations = ', bn, I12, /, ' Time       = ', F12.2 )" )              &
      pname, data%prob%f, inform%pr_feas, inform%iter, time
     IF ( nmhist > 0 ) WRITE( out,                                             &
       "( ' Non-monotone descent strategy ( history =', I3,                    &
      &     ' ) used ' )" ) nmhist
     inform%status = 0

!  Normal return

     RETURN

!  Allocation errors

 910 CONTINUE
     inform%status = - 2
     WRITE( control%error, "( ' ** Message from -LPSQP_solve-', /,             &
    &               ' Allocation error (status = ', I6, ') for ', A20 )" )     &
       alloc_status, bad_alloc
     RETURN

 930 CONTINUE
     WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )          &
       cutest_status
     inform%status = - 98
     RETURN


!  Non-executable statements

 2000 FORMAT( /,' Solution : ', /,'                        ',                  &
                '        <------ Bounds ------> ', /                           &
                '      # name          value   ',                              &
                '    Lower       Upper       Dual ' ) 
 2010 FORMAT( /,' Constraints : ', /, '                        ',              &
                '        <------ Bounds ------> ', /                           &
                '      # name           value   ',                             &
                '    Lower       Upper    Multiplier ' ) 
 2020 FORMAT( I7, 1X, A10, 4ES12.4 ) 
 2030 FORMAT( /, '  iter   merit fun  pr_feas  du_feas   step ',               &
                 '   radius  ared/pred   facts ')
 2040 FORMAT( 6X, '. .', 9X, 4( 2X, 10( '.' ) ) )
 2050 FORMAT( /, 1X, 10( '-=' ), ' rho = ', ES10.2, 1X, 20( '=-' ) )

!  End of subroutine LPSQP_solve

     END SUBROUTINE LPSQP_solve

!-*-*-*-*  G A L A H A D -  LPSQP_terminate  S U B R O U T I N E -*-*-*-*

     SUBROUTINE LPSQP_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LPSQP_data_type ), INTENT( INOUT ) :: data
     TYPE ( LPSQP_control_type ), INTENT( IN ) :: control
     TYPE ( LPSQP_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!    INTEGER :: alloc_status

     inform%status = 0

!  Non-executable statement

!2990  FORMAT( ' ** Message from -LPSQP_terminate-', /,                     &
!              ' Deallocation error (status = ', I6, ') for ', A24 )

!  End of subroutine LPSQP_terminate

     END SUBROUTINE LPSQP_terminate

!-*-*-*-*  G A L A H A D  -  L P S Q P _ m e r i t   F U N C T I O N -*-*-*-*

     FUNCTION LPSQP_merit( m, f, rho, one_norm, C, C_l, C_u, violation )
     REAL ( KIND = wp ) :: LPSQP_merit

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Compute the value of the merit function

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m
     REAL ( KIND = wp ), INTENT( IN ) :: f, rho
     REAL ( KIND = wp ), INTENT( OUT ) :: violation
     LOGICAL, INTENT( IN ) :: one_norm
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, C_l, C_u
     
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i

!    write(6,*) ' one norm ', one_norm
     violation = zero

     IF ( one_norm ) THEN

!  l_1-norm merit function

!      write(6,"(ES12.4)") violation
       DO i = 1, m
         IF ( C( i ) < C_l( i ) )                                              &
           violation = violation + ( C_l( i ) - C( i ) )
!      write(6,"(3ES12.4)") violation, C_l( i ), C( i )
         IF ( C( i ) > C_u( i ) )                                              &
           violation = violation + ( C( i ) - C_u( i ) )
!      write(6,"(3ES12.4)") violation, C( i ), C_u( i )
       END DO
     ELSE

!  l_infinity-norm merit function

       DO i = 1, m
         IF ( C( i ) < C_l( i ) )                                              &
           violation = MAX( violation, ( C_l( i ) - C( i ) ) )
         IF ( C( i ) > C_u( i ) )                                              &
           violation = MAX( violation, ( C( i ) - C_u( i ) ) )
       END DO
     END IF

!    write(6,"(3ES12.4)") f, rho, violation
     LPSQP_merit = f + rho * violation

     RETURN

!  End of function LPSQP_merit

     END FUNCTION LPSQP_merit

!  End of module LPSQP

!A END MODULE GALAHAD_LPSQPA_double
!B END MODULE GALAHAD_LPSQP_double
