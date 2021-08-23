! THIS VERSION: GALAHAD 3.3 - 20/05/2021 AT 10:30 GMT.

!-*-*-  L A N C E L O T  -B-  LANCELOT _ S T E E R I N G _  M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   ( based on Conn-Gould-Toint fortran 77 version LANCELOT A, ~1992 )
!   originally released pre GALAHAD Version 1.0. February 7th 1995
!   update released with GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE LANCELOT_steering_double

!  |------------------------------------------------------------------|
!  |                                                                  |
!  |  Find a local minimizer of a smooth (group partially separable)  |
!  |  objective function subject to (partially separable) constraints |
!  |  and simple bounds                                               |
!  |                                                                  |
!  |  ** Version with Curtis-Jiang-Robinson feasibiliy steering **    |
!  |                                                                  |
!  |------------------------------------------------------------------|

!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SPACE_double
     USE LANCELOT_TYPES_double
     USE LANCELOT_INITW_double
     USE LANCELOT_OTHERS_double
     USE LANCELOT_HSPRD_double
     USE LANCELOT_CAUCHY_double
     USE LANCELOT_CG_double
     USE LANCELOT_PRECN_double
     USE LANCELOT_FRNTL_double
     USE LANCELOT_STRUTR_double
     USE GALAHAD_SMT_double
     USE GALAHAD_SILS_double
     USE GALAHAD_SCU_double, ONLY : SCU_matrix_type, SCU_data_type,            &
       SCU_inform_type, SCU_factorize, SCU_terminate
     USE LANCELOT_ASMBL_double, ONLY : ASMBL_save_type
     USE GALAHAD_EXTEND_double, ONLY : EXTEND_save_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: LANCELOT_initialize, LANCELOT_read_specfile, LANCELOT_solve,    &
               LANCELOT_terminate, LANCELOT_problem_type, LANCELOT_save_type,  &
               LANCELOT_control_type, LANCELOT_inform_type, LANCELOT_data_type

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point99 = 0.99_wp
     REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundrd = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenten = ten ** 10
     REAL ( KIND = wp ), PARAMETER :: tenm2 = 0.01_wp
     REAL ( KIND = wp ), PARAMETER :: tenm4 = 0.0001_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: tenm10 = ten ** ( - 10 )
                            
     REAL ( KIND = wp ), PARAMETER :: wmin = point1
     REAL ( KIND = wp ), PARAMETER :: theta = point1
     REAL ( KIND = wp ), PARAMETER :: stptol = point1

   CONTAINS

!-*-*-*-*  L A N C E L O T -B- LANCELOT_initialize  S U B R O U T I N E -*-*-*-*

     SUBROUTINE LANCELOT_initialize( data, control )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for LANCELOT controls
!   (see LANCELOT_types for more details)

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_data_type ), INTENT( INOUT ) :: data
     TYPE ( LANCELOT_control_type ), INTENT( OUT ) :: control
 
!    INTEGER, PARAMETER :: lmin = 1
     INTEGER, PARAMETER :: lmin = 10000

!  Error and ordinary output unit numbers

     data%S%error = control%error ; data%S%out = control%out

!  Set initial array lengths for EXTEND arrays

     data%S%EXTEND%lirnh = lmin
     data%S%EXTEND%ljcnh = lmin
     data%S%EXTEND%llink_min = lmin
     data%S%EXTEND%lirnh_min = lmin
     data%S%EXTEND%ljcnh_min = lmin
     data%S%EXTEND%lh_min = lmin
     data%S%EXTEND%lwtran_min = lmin
     data%S%EXTEND%litran_min = lmin
     data%S%EXTEND%lh = lmin
     data%S%ASMBL%ptr_status = .FALSE.

     CALL SILS_initialize( data%SILS_data, control%SILS_cntl )
     control%SILS_cntl%ordering = 3
!57V2 control%SILS_cntl%ordering = 2
!57V3 control%SILS_cntl%ordering = 5
!57V2 control%SILS_cntl%scaling = 0
!57V2 control%SILS_cntl%static_tolerance = zero
!57V2 control%SILS_cntl%static_level = zero

     RETURN

!  End of subroutine LANCELOT_initialize

     END SUBROUTINE LANCELOT_initialize

!-*-*-   L A N C E L O T _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE LANCELOT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by LANCELOT_initialize could (roughly) 
!  have been set as:

! BEGIN LANCELOT SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  alive-device                                   60
!  print-level                                    0
!  maximum-number-of-iterations                   1000
!  start-print                                    -1 
!  stop-print                                     -1
!  iterations-between-printing                    1
!  linear-solver-used                             BAND_CG
!  number-of-lin-more-vectors-used                5
!  semi-bandwidth-for-band-preconditioner         5
!  maximum-dimension-of-schur-complement          100
!  unit-number-for-temporary-io                   75
!  more-toraldo-search-length                     0
!  history-length-for-non-monotone-descent        0
!  penalty-parameter-decreased-limit              1000000
!  penalty-parameter-decreased-limit-per-iter     1000000
!  first-derivative-approximations                EXACT
!  second-derivative-approximations               SR1
!  primal-accuracy-required                       1.0D-5
!  dual-accuracy-required                         1.0D-5
!  minimum-merit-value                            -1.0D+300
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
!  penalty-parameter-decrease                     0.1
!  penalty-parameter-steering-decrease            0.7
!  no-dual-updates-until-penalty-parameter-below  0.1
!  initial-dual-accuracy-required                 0.1
!  initial-primal-accuracy-required               0.1
!  pivot-tolerance-used                           0.1
!  steering-kappa-3                               1.0D-5
!  steering-kappa-t                               0.9
!  steering-mu-min                                0.0
!  maximum-cpu-time-limit                         -1.0
!  quadratic-problem                              NO
!  steer-towards-feasibility                      NO
!  two-norm-trust-region-used                     NO
!  exact-GCP-used                                 YES
!  Gauss-Newton-model-used                        NO
!  Gauss-Newton-model-used-after-Cauchy-step      NO
!  magical-steps-allowed                          NO
!  subproblem-solved-accurately                   NO
!  structured-trust-region-used                   NO
!  print-for-maximization                         NO
!  space-critical                                 NO
!  deallocate-error-fatal                         NO
!  print-full-solution                            YES
!  alive-filename                                 ALIVE.d
! END LANCELOT SPECIFICATIONS

!  Dummy arguments

     TYPE ( LANCELOT_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: alive_unit = out + 1
     INTEGER, PARAMETER :: print_level = alive_unit + 1
     INTEGER, PARAMETER :: maxit = print_level + 1
     INTEGER, PARAMETER :: start_print = maxit + 1
     INTEGER, PARAMETER :: stop_print = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print + 1
     INTEGER, PARAMETER :: linear_solver = print_gap + 1
     INTEGER, PARAMETER :: icfact = linear_solver + 1
     INTEGER, PARAMETER :: semibandwidth = icfact + 1
     INTEGER, PARAMETER :: max_sc = semibandwidth + 1
     INTEGER, PARAMETER :: io_buffer = max_sc + 1
     INTEGER, PARAMETER :: more_toraldo = io_buffer + 1
     INTEGER, PARAMETER :: non_monotone = more_toraldo + 1
     INTEGER, PARAMETER :: first_derivatives = non_monotone + 1
     INTEGER, PARAMETER :: second_derivatives = first_derivatives + 1
     INTEGER, PARAMETER :: num_mudec = second_derivatives + 1
     INTEGER, PARAMETER :: num_mudec_per_iteration = num_mudec + 1
     INTEGER, PARAMETER :: stopc = num_mudec_per_iteration + 1
     INTEGER, PARAMETER :: stopg = stopc + 1
     INTEGER, PARAMETER :: min_aug = stopg + 1
     INTEGER, PARAMETER :: acccg = min_aug + 1
     INTEGER, PARAMETER :: initial_radius = acccg + 1
     INTEGER, PARAMETER :: maximum_radius = initial_radius + 1
     INTEGER, PARAMETER :: eta_successful = maximum_radius + 1
     INTEGER, PARAMETER :: eta_very_successful = eta_successful + 1
     INTEGER, PARAMETER :: eta_extremely_successful = eta_very_successful + 1
     INTEGER, PARAMETER :: gamma_smallest = eta_extremely_successful + 1
     INTEGER, PARAMETER :: gamma_decrease = gamma_smallest + 1
     INTEGER, PARAMETER :: gamma_increase = gamma_decrease + 1
     INTEGER, PARAMETER :: mu_meaningful_model = gamma_increase + 1
     INTEGER, PARAMETER :: mu_meaningful_group = mu_meaningful_model + 1
     INTEGER, PARAMETER :: initial_mu = mu_meaningful_group + 1
     INTEGER, PARAMETER :: mu_decrease = initial_mu + 1
     INTEGER, PARAMETER :: mu_steering_decrease = mu_decrease + 1
     INTEGER, PARAMETER :: mu_tol = mu_steering_decrease + 1
     INTEGER, PARAMETER :: firstg = mu_tol + 1
     INTEGER, PARAMETER :: firstc = firstg + 1
     INTEGER, PARAMETER :: pivtol = firstc + 1
     INTEGER, PARAMETER :: kappa_3 = pivtol + 1
     INTEGER, PARAMETER :: kappa_t = kappa_3 + 1
     INTEGER, PARAMETER :: mu_min = kappa_t + 1
     INTEGER, PARAMETER :: cpu_time_limit = mu_min + 1
     INTEGER, PARAMETER :: quadratic_problem = cpu_time_limit + 1
     INTEGER, PARAMETER :: steering = quadratic_problem + 1
     INTEGER, PARAMETER :: two_norm_tr = steering + 1
     INTEGER, PARAMETER :: exact_gcp = two_norm_tr + 1
     INTEGER, PARAMETER :: gn_model = exact_gcp + 1
     INTEGER, PARAMETER :: gn_model_after_cauchy = gn_model + 1
     INTEGER, PARAMETER :: magical_steps = gn_model_after_cauchy + 1
     INTEGER, PARAMETER :: accurate_bqp = magical_steps + 1
     INTEGER, PARAMETER :: structured_tr = accurate_bqp + 1
     INTEGER, PARAMETER :: print_max = structured_tr + 1
     INTEGER, PARAMETER :: full_solution = print_max + 1
     INTEGER, PARAMETER :: space_critical = full_solution + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 8 ), PARAMETER :: specname = 'LANCELOT'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( alive_unit )%keyword = 'alive-device'
     spec( print_level )%keyword = 'print-level' 
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( linear_solver )%keyword = 'linear-solver-used'
     spec( icfact )%keyword = 'number-of-lin-more-vectors-used'
     spec( semibandwidth )%keyword = 'semi-bandwidth-for-band-preconditioner'
     spec( max_sc )%keyword = 'maximum-dimension-of-schur-complement'
     spec( io_buffer )%keyword = 'unit-number-for-temporary-io'
     spec( more_toraldo )%keyword = 'more-toraldo-search-length'
     spec( non_monotone )%keyword = 'history-length-for-non-monotone-descent'
     spec( first_derivatives )%keyword = 'first-derivative-approximations'
     spec( second_derivatives )%keyword = 'second-derivative-approximations'
     spec( num_mudec )%keyword = 'penalty-parameter-decreased-limit'
     spec( num_mudec_per_iteration )%keyword                                   &
       = 'penalty-parameter-decreased-limit-per-iter'

!  Real key-words

     spec( stopc )%keyword = 'primal-accuracy-required'
     spec( stopg )%keyword = 'dual-accuracy-required'
     spec( min_aug )%keyword = 'minimum-merit-value'
     spec( acccg )%keyword = 'inner-iteration-relative-accuracy-required'
     spec( initial_radius )%keyword = 'initial-trust-region-radius'
     spec( maximum_radius )%keyword = 'maximum-radius'
     spec( eta_successful )%keyword = 'eta-successful'
     spec( eta_very_successful )%keyword = 'eta-very-successful'
     spec( eta_extremely_successful )%keyword = 'eta-extremely-successful'
     spec( gamma_smallest )%keyword = 'gamma-smallest'
     spec( gamma_decrease )%keyword = 'gamma-decrease'
     spec( gamma_increase )%keyword = 'gamma-increase'
     spec( mu_meaningful_model )%keyword = 'mu-meaningful-model'
     spec( mu_meaningful_group )%keyword = 'mu-meaningful-group'
     spec( initial_mu )%keyword = 'initial-penalty-parameter'
     spec( mu_tol )%keyword = 'no-dual-updates-until-penalty-parameter-below'
     spec( mu_decrease )%keyword = 'penalty-parameter-decrease'
     spec( mu_steering_decrease )%keyword                                      &
       = 'penalty-parameter-steering-decrease'
     spec( firstg )%keyword = 'initial-dual-accuracy-required'
     spec( firstc )%keyword = 'initial-primal-accuracy-required'
     spec( kappa_3 )%keyword = 'steering-kappa-3'
     spec( kappa_t )%keyword = 'steering-kappa-t'
     spec( mu_min )%keyword = 'steering-mu-min'
     spec( pivtol )%keyword = 'pivot-tolerance-used'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'

!  Logical key-words

     spec( quadratic_problem )%keyword = 'quadratic-problem'
     spec( steering )%keyword = 'steer-towards-feasibility'
     spec( two_norm_tr )%keyword = 'two-norm-trust-region-used'
     spec( exact_gcp )%keyword = 'exact-GCP-used'
     spec( gn_model )%keyword = 'Gauss-Newton-model-used'
     spec( gn_model_after_cauchy )%keyword                                     &
       = 'Gauss-Newton-model-used-after-Cauchy-step'
     spec( magical_steps )%keyword = 'magical-steps-allowed'
     spec( accurate_bqp )%keyword = 'subproblem-solved-accurately'
     spec( structured_tr )%keyword = 'structured-trust-region-used'
     spec( print_max )%keyword = 'print-for-maximization'
     spec( full_solution )%keyword = 'print-full-solution'

!  Character key-words

     spec( alive_file )%keyword = 'alive-filename'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ), control%error,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ), control%out,                     &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( alive_unit ), control%out,              &
                                 control%alive_unit )                         
     CALL SPECFILE_assign_value( spec( print_level ), control%print_level,     &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( maxit ), control%maxit,                 &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( start_print ), control%start_print,     &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( stop_print ), control%stop_print,       &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( print_gap ), control%print_gap,         &
                                 control%error )                           
     CALL SPECFILE_assign_symbol( spec( linear_solver ), control%linear_solver,&
                                  control%error )                           
     CALL SPECFILE_assign_value( spec( icfact ), control%icfact,               &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( semibandwidth ), control%semibandwidth, &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( max_sc ), control%max_sc,               &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( io_buffer ), control%io_buffer,         &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( more_toraldo ), control%more_toraldo,   &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( non_monotone ), control%non_monotone,   &
                                 control%error )                           
     CALL SPECFILE_assign_symbol( spec( first_derivatives ),                   &
                                  control%first_derivatives, control%error )  
     CALL SPECFILE_assign_symbol( spec( second_derivatives ),                  &
                                  control%second_derivatives, control%error )
     CALL SPECFILE_assign_value( spec( num_mudec ), control%num_mudec,         &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( num_mudec_per_iteration ),              &
                                 control%num_mudec_per_iteration,              &
                                 control%error )                           

!  Set real values

     CALL SPECFILE_assign_value( spec( stopc ), control%stopc,                 &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( stopg ), control%stopg,                 &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( min_aug ), control%min_aug,             &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( acccg ), control%acccg,                 &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( initial_radius ),                       &
                                 control%initial_radius, control%error )
     CALL SPECFILE_assign_value( spec( maximum_radius ),                       &
                                 control%maximum_radius, control%error )
     CALL SPECFILE_assign_value( spec( eta_successful ),                       &
                                 control%eta_successful, control%error )
     CALL SPECFILE_assign_value( spec( eta_very_successful ),                  &
                                 control%eta_very_successful, control%error )
     CALL SPECFILE_assign_value( spec( eta_extremely_successful ),             &
                                 control%eta_extremely_successful,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_smallest ),                       &
                                 control%gamma_smallest, control%error )
     CALL SPECFILE_assign_value( spec( gamma_decrease ),                       &
                                 control%gamma_decrease, control%error )
     CALL SPECFILE_assign_value( spec( gamma_increase ),                       &
                                 control%gamma_increase, control%error )
     CALL SPECFILE_assign_value( spec( mu_meaningful_model ),                  &
                                 control%mu_meaningful_model, control%error )
     CALL SPECFILE_assign_value( spec( mu_meaningful_group ),                  &
                                 control%mu_meaningful_group, control%error )
     CALL SPECFILE_assign_value( spec( initial_mu ), control%initial_mu,       &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( mu_tol ), control%mu_tol,               &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( firstg ), control%firstg,               &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( firstc ), control%firstc,               &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( mu_decrease ), control%mu_decrease,     &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( mu_steering_decrease ),                 &
                                 control%mu_steering_decrease,                 &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( kappa_3 ), control%kappa_3,             &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( kappa_t ), control%kappa_t,             &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( mu_min ), control%mu_min,               &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( pivtol ), control%SILS_cntl%u,          &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )                           

!  Set logical values

     CALL SPECFILE_assign_value( spec( quadratic_problem ),                    &
                                 control%quadratic_problem, control%error )
     CALL SPECFILE_assign_value( spec( steering ), control%steering,           &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( two_norm_tr ), control%two_norm_tr,     &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( exact_gcp ), control%exact_gcp,         &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( gn_model ), control%gn_model,           &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( gn_model_after_cauchy ),                &
                                 control%gn_model_after_cauchy,                &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( magical_steps ), control%magical_steps, &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( accurate_bqp ), control%accurate_bqp,   &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( structured_tr ), control%structured_tr, &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( print_max ), control%print_max,         &
                                 control%error )                           
     CALL SPECFILE_assign_value( spec( full_solution ), control%full_solution, &
                                 control%error ) 
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
!  Set character values

     CALL SPECFILE_assign_string( spec( alive_file ), control%alive_file,      &
                                  control%error )                           

     RETURN

     END SUBROUTINE LANCELOT_read_specfile

!-*-*-*-*-*  L A N C E L O T -B- LANCELOT_solve  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE LANCELOT_solve( prob, RANGE , GVALS, FT, XT, FUVALS, lfuval,   &
                                ICALCF, ICALCG, IVAR, Q, DGRAD, control,       &
                                inform, data, ELFUN, GROUP, ELFUN_flexible,    &
                                ELDERS )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  LANCELOT_solve, a method for finding a local minimizer of a function 
!  subject to general constraints and simple bounds on the sizes of the 
!  variables. The method is described in the paper 
!
!  'A globally convergent augmented Lagrangian algorithm for optimization 
!   with general constraints and simple bounds', A. R. Conn, N. I. M. Gould 
!  and Ph. L. Toint, SIAM J. Num. Anal. 28 (1991) PP.545-572

!  See LANCELOT_solve_main for more details

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_control_type ), INTENT( INOUT ) :: control
     TYPE ( LANCELOT_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( LANCELOT_data_type ), INTENT( INOUT ) :: data
     TYPE ( LANCELOT_problem_type ), INTENT( INOUT ), TARGET :: prob

     INTEGER, INTENT( IN ) :: lfuval
     INTEGER, INTENT( INOUT ), DIMENSION( prob%n  ) :: IVAR
     INTEGER, INTENT( INOUT ), DIMENSION( prob%nel ) :: ICALCF
     INTEGER, INTENT( INOUT ), DIMENSION( prob%ng ) :: ICALCG
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( prob%ng, 3 ) :: GVALS 
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( prob%n ) :: Q, XT, DGRAD
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%ng ) :: FT
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( lfuval ) :: FUVALS

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for RANGE

       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN 

!  Interface block for ELFUN_flexible 

       SUBROUTINE ELFUN_flexible(                                              &
                          FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, llders, ifflag, ELDERS,      &
                          ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( IN ) :: llders
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf), ELDERS(2,llders)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN_flexible

!  Interface block for GROUP

       SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,              &
                          ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,              &
                          lcalcg, lfvalu, lgpvlu, derivs, igstat )
       INTEGER, INTENT( IN ) :: lgvalu, ncalcg
       INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
       INTEGER, INTENT( OUT ) :: igstat
       LOGICAL, INTENT( IN ) :: derivs
       INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
       INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
       INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lfvalu ) :: FVALUE
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lgpvlu ) :: GPVALU
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ),                        &
                                       DIMENSION ( lgvalu, 3 ) :: GVALUE
       END SUBROUTINE GROUP

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( 2, prob%nel ) :: ELDERS
     OPTIONAL :: ELFUN, ELFUN_flexible, GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig, k1, k2, scu_status, alloc_status
     REAL ( KIND = wp ) :: epsmch
     LOGICAL :: alive, internal_el, internal_gr, use_elders
     CHARACTER ( LEN = 80 ) :: bad_alloc
     LOGICAL, POINTER, DIMENSION( : ) :: GXEQX_used
     CHARACTER ( LEN = 80 ) :: array_name

!-----------------------------------------------
!   A l l o c a t a b l e   A r r a y s
!-----------------------------------------------

     epsmch = EPSILON( one )
     internal_el = PRESENT( ELFUN ) .OR. PRESENT( ELFUN_flexible )
     internal_gr = PRESENT( GROUP )
     use_elders = PRESENT( ELDERS )

     IF ( inform%status > 0 .AND. inform%status /= 14 ) RETURN

! Initial entry: set up data

     IF ( inform%status == 0 ) THEN

!  Record time at which subroutine initially called
  
        CALL CPU_TIME( data%S%time )
  
!  Initialize integer inform parameters
  
!  iter gives the number of iterations performed
!  itercg gives the total numbr of CG iterations performed
!  itcgmx is the maximum number of CG iterations permitted per inner iteration
!  ncalcf gives the number of element functions that must be re-evaluated
!  ncalcg gives the number of group functions that must be re-evaluated
!  nvar gives the current number of free variables
!  ngeval is the number of derivative evaluations made
!  iskip gives the total number of secant updates that are skipped
!  ifixed is the variable that most recently encountered on of its bounds
!  nsemib is the bandwidth used with the expanding-band preconditioner
  
       inform%iter = 0 ; inform%itercg = 0 ; inform%itcgmx = 0
       inform%ncalcf = 0 ; inform%ncalcg = 0 ; inform%nvar = 0
       inform%ngeval = 0 ; inform%iskip = 0 ; inform%ifixed = 0
       inform%nsemib = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
  
!  Initialize real inform parameters
  
!  aug gives the value of the augmented Lagrangian merit function
!  obj gives the value of the objective function
!  pjgnrm is the norm of the projected gradient of the merit function
!  cnorm gives the norm of the equality constraints
!  ratio gives the current ratio of predicted to achieved merit fn. reduction
!  mu is the current value of the penalty parameter
!  radius is the current value of the trust-region radius
!  ciccg gives the pivot tolerance used when ICCG is used for preconditioning
  
       inform%aug = HUGE( one ) ; inform%obj = HUGE( one ) 
       inform%pjgnrm = HUGE( one )
       inform%cnorm = zero ; inform%ratio = zero ; inform%mu = zero
       inform%radius = zero ; inform%ciccg = zero
  
!  Initialize logical inform parameter
  
!  newsol is true if a major iteration has just been completed
  
       inform%newsol = .FALSE.
  
!  Check problem dimensions

       IF ( prob%n <= 0 .OR. prob%ng <= 0 .OR. prob%nel < 0 ) THEN
         inform%status = 15 ; RETURN ; END IF

!  Set output character strings
  
       data%S%STATE = (/ ' FREE', 'LOWER', 'UPPER', 'FIXED', 'DEGEN' /)
       data%S%ISYS = (/ 0, 0, 0, 0, 0 /)
!      data%S%CGENDS = (/ ' CONVR', ' MAXIT', ' BOUND', ' -CURV', ' S<EPS' /)
!      data%S%LSENDS = (/ ' PSDEF', ' INDEF', ' SINGC', ' SINGI', ' PRTRB' /)
       data%S%CGENDS1 = (/ 'C', 'M', 'B', 'N', 'T' /)
       data%S%LSENDS1 = (/ 'P', 'I', 'C', 'S', 'M' /)

!  Initialize floating-point parameters
  
!  epstlp and epstln are tolerances on how far a variable may lie away from
!         its bound and still be considered active
!  radtol is the smallest value that the trust region radius is allowed
!  stpmin is the smallest allowable step between consecutive iterates
!  teneps is 10 times the machine precision(!)
!  epsrcg is the smallest that the CG residuals will be required to be
!  vscmax is the largest specified variable scaling
!  fill   is the amount of fill-in when a direct method is used to find an
!         approximate solution to the model problem
  
       data%S%epstlp = epsmch ; data%S%epstln = epsmch
       data%S%epsrcg = hundrd * epsmch ** 2 ; data%S%teneps = ten * epsmch
       data%S%radtol = point1 * epsmch ; data%S%smallh = epsmch ** 0.3333
       data%S%stpmin = epsmch ** 0.75 ; data%S%vscmax = zero
       data%S%fill = zero 

!  Timing parameters: tca, tls, tmv and tup are, respectively, the times spent
!  in finding the Cauchy step, finding the approximate minimizer of the model,
!  forming the product of the Hessian with a specified vector and in updating
!  the second derivative approximations. time gives the clock time on initial
!  entry to the subroutine. t and time give the instantaneous clock time
  
       data%S%tmv = 0.0 ; data%S%tca = 0.0 ; data%S%tls = 0.0 ; data%S%tup = 0.0
  
!  number is used to control which negative eigenvalue is picked when the
!         negative curvature exploiting multifrontal scheme is used
  
       data%S%number = 0
  
!  Initialize logical parameters
  
!  full_solution is .TRUE. if all components of the solution and constraints 
!  are to be printed on termination, and .FALSE. if only the first and last
!  (representative) few are required

       data%S%full_solution = control%full_solution

!  S%firsup is .TRUE. if initial second derivative approximations are
!  to be rescaled using the Shanno-Phua scalings
  
       data%S%firsup = .FALSE. ; data%S%next = .FALSE.
  
!  alllin is .TRUE. if there are no nonlinear elements and .FALSE. otherwise
  
       data%S%alllin = prob%nel== 0
  
!  new_major is true during the first minor iteration of a new major iteration

       data%S%new_major = .TRUE.

!  p_type indicates the type of problem: 
!    1  (unconstrained or bound constrained)
!    2  (feasibility, maybe bound constrained)
!    3  (generally constrained)
  
       IF ( ALLOCATED( prob%KNDOFG ) ) THEN
         IF ( SIZE( prob%KNDOFG ) < prob%ng ) THEN
           inform%status = 9 ; RETURN ; END IF
         IF ( COUNT( prob%KNDOFG( : prob%ng ) <= 1 ) == prob%ng ) THEN
           data%S%p_type = 1
         ELSE
           IF ( ALLOCATED( prob%C ) ) THEN
             IF ( SIZE( prob%C ) < prob%ng ) THEN
               inform%status = 9 ; RETURN ; END IF
           ELSE
             inform%status = 9 ; RETURN
           END IF
           IF ( ALLOCATED( prob%Y ) ) THEN
             IF ( SIZE( prob%Y ) < prob%ng ) THEN
               inform%status = 9 ; RETURN ; END IF
           ELSE
             inform%status = 9 ; RETURN
           END IF
           data%S%p_type = 2
           DO i = 1, prob%ng
             IF ( prob%KNDOFG( i ) == 1 ) THEN 
               data%S%p_type = 3 ; EXIT ; END IF
           END DO
         END IF

!  See if any of the groups are to be skipped

         data%S%skipg = COUNT( prob%KNDOFG == 0 ) > 0
       ELSE
         data%S%p_type = 1 
         data%S%skipg = .FALSE.
       END IF
       data%S%steering = control%steering .AND. data%S%p_type > 2
     END IF
  
     IF ( inform%status == 0 .OR. inform%status == 14 ) THEN
  
!  Record the print level and output channel
  
       data%S%out = control%out
  
!  Only print between iterations start_print and stop_print

       IF ( control%start_print < inform%iter ) THEN
         data%S%start_print = 0
       ELSE
         data%S%start_print = control%start_print
       END IF
 
       IF ( control%stop_print < inform%iter ) THEN
         data%S%stop_print = control%maxit
       ELSE
         data%S%stop_print = control%stop_print
       END IF
 
       IF ( control%print_gap < 2 ) THEN
         data%S%print_gap = 1
       ELSE
         data%S%print_gap = control%print_gap
       END IF
 
!  Print warning and error messages
  
       data%S%set_printe = data%S%out > 0 .AND. control%print_level >= 0
  
       IF ( data%S%start_print <= 0 .AND. data%S%stop_print > 0 ) THEN
         data%S%printe = data%S%set_printe
         data%S%print_level = control%print_level
       ELSE
         data%S%printe = .FALSE.
         data%S%print_level = 0
       END IF

!  Test whether the maximum allowed number of iterations has been reached
  
       IF ( control%maxit < 0 ) THEN
         IF ( data%S%printe ) WRITE( data%S%out,                               &
           "( /, ' LANCELOT_solve : maximum number of iterations reached ' )" )
         inform%status = 1 ; RETURN
       END IF
  
!  Basic single line of output per iteration
  
       data%S%set_printi = data%S%out > 0 .AND. control%print_level >= 1 
  
!  As per printi, but with additional timings for various operations
  
       data%S%set_printt = data%S%out > 0 .AND. control%print_level >= 2 
  
!  As per printm, but with checking of residuals, etc
  
       data%S%set_printm = data%S%out > 0 .AND. control%print_level >= 3 
  
!  As per printm but also with an indication of where in the code we are
  
       data%S%set_printw = data%S%out > 0 .AND. control%print_level >= 4
  
!  Full debugging printing with significant arrays printed
  
       data%S%set_printd = data%S%out > 0 .AND. control%print_level >= 10
  
       IF ( data%S%start_print <= 0 .AND. data%S%stop_print > 0 ) THEN
         data%S%printi = data%S%set_printi
         data%S%printt = data%S%set_printt
         data%S%printm = data%S%set_printm
         data%S%printw = data%S%set_printw
         data%S%printd = data%S%set_printd
       ELSE
         data%S%printi = .FALSE.
         data%S%printt = .FALSE.
         data%S%printm = .FALSE.
         data%S%printw = .FALSE.
         data%S%printd = .FALSE.
       END IF

!  Create a file which the user may subsequently remove to cause
!  immediate termination of a run

       IF ( control%alive_unit > 0 ) THEN
         INQUIRE( FILE = control%alive_file, EXIST = alive )
        IF ( .NOT. alive ) THEN
           OPEN( control%alive_unit, FILE = control%alive_file,                &
                 FORM = 'FORMATTED', STATUS = 'NEW' )
           REWIND control%alive_unit
           WRITE( control%alive_unit, "( ' LANCELOT rampages onwards ' )" )
           CLOSE( control%alive_unit )
         END IF
       END IF

       IF ( control%print_max ) THEN
         data%S%findmx = - one ; ELSE ; data%S%findmx = one ; END IF
  
!  twonrm is .TRUE. if the two-norm trust region is to be used, and is .FALSE.
!  if the infinity-norm trust region is required
  
       data%S%twonrm = control%two_norm_tr
       data%S%maximum_radius = MAX( one, control%maximum_radius )
  
!  direct is .TRUE. if the linear system is to be solved using a direct method
!  (MA27). Otherwise, the linear system will be solved using conjugate gradients
  
       data%S%direct = control%linear_solver >= 11
  
!  modchl is .TRUE. if the Hessian is to be forced to be positive definite
!  prcond is .TRUE. if preconditioning is to be used in the conjugate
!         gradient iteration
  
       data%S%modchl = control%linear_solver == 12
       data%S%prcond =  .NOT. data%S%direct .AND. control%linear_solver >= 2
  
!  dprcnd is .TRUE. if the user wishes to use a diagonal preconditioner
  
       data%S%dprcnd = control%linear_solver == 2 
       data%S%calcdi = data%S%dprcnd
  
!  myprec is .TRUE. if the user is to take responsibility for providing the
!  preconditioner
  
       data%S%myprec = control%linear_solver == 3
  
!  iprcnd is .TRUE. if the user wishes to use a positive definite
!  perturbation of the inner band of the true matrix as a preconditioner
  
       data%S%iprcnd = control%linear_solver == 4
  
!  munks is .TRUE. if the Munksgaard preconditioner is to be used
  
       data%S%munks = control%linear_solver == 5
  
!  seprec is .TRUE. if the user wishes to use the Schnabel-Eskow positive
!  definite perturbation of the complete matrix as a preconditioner
  
       data%S%seprec = control%linear_solver == 6
  
!  gmpspr is .TRUE. if the user wishes to use the Gill-Murray-Ponceleon-
!  Saunders positive definite perturbation of the complete matrix as a
!  preconditioner
  
       data%S%gmpspr = control%linear_solver == 7
  
!  use_band is .TRUE. if the user wishes to use a bandsolver as a
!    preconditioner
  
       data%S%use_band = control%linear_solver == 8
  
!  icfs is .TRUE. if the user wishes to use Lin and More's incomplete Cholesky
!  factorization as a preconditioner
  
       data%S%icfs = control%linear_solver == 9
       data%S%icfact = MAX( control%icfact, 0 )
  
!  fdgrad is .FALSE. if the user provides exact first derivatives of the
!  nonlinear element functions and .TRUE. otherwise
  
       IF ( use_elders ) THEN
         ELDERS( 1 , : ) = MAX( MIN( ELDERS( 1 , : ), 2 ), 0 )
         data%S%first_derivatives = MAXVAL( ELDERS( 1 , : ) )
         i = COUNT( ELDERS( 1 , : ) <= 0 )
         data%S%fdgrad = i /= prob%nel
         data%S%getders = i /= 0
       ELSE
         data%S%first_derivatives = MIN( control%first_derivatives, 2 )
         data%S%fdgrad = data%S%first_derivatives >= 1
         data%S%getders = .NOT. data%S%fdgrad
       END IF
  
!  second is .TRUE. if the user provides exact second derivatives
!  of the nonlinear element functions and .FALSE. otherwise
  
       IF ( use_elders ) THEN
         DO i = 1, prob%nel
           ELDERS( 2 , i ) = MAX( MIN( ELDERS( 2 , i ), 4 ), 0 )
           IF ( ELDERS( 1 , i ) > 0 ) ELDERS( 2 , i ) = 4
         END DO
         data%S%second = COUNT( ELDERS( 2 , : ) <= 0 ) == prob%nel
       ELSE
         data%S%second_derivatives = MIN( control%second_derivatives, 4 )
         data%S%second = data%S%second_derivatives <= 0 
         IF ( data%S%fdgrad .AND. data%S%second ) THEN
           data%S%second_derivatives = 4 ; data%S%second = .FALSE.
         END IF
       END IF
  
!  xactcp is .TRUE, if the user wishes to calculate the exact Cauchy
!  point in the fashion of Conn, Gould and Toint ( 1988 ). If an
!  approximation suffices, xactcp will be .FALSE.
  
       data%S%xactcp = control%exact_gcp
  
!  slvbqp is .TRUE. if a good approximation to the minimum of the quadratic
!  model is to be sought at each iteration, while slvbqp is .FALSE. if a less
!  accurate solution is desired
  
       data%S%slvbqp = control%accurate_bqp
  
!  strctr is .TRUE. if a structured trust-region is to be used
           
       data%S%strctr = control%structured_tr
  
!  S%mortor is .TRUE. if the More-Toraldo projected search is to be used
           
       data%S%msweep = control%more_toraldo ; data%S%mortor = data%S%msweep /= 0
  
!  unsucc is .TRUE. if the last attempted step proved unsuccessful
  
       data%S%unsucc = .FALSE.
  
!  nmhist is the length of the history if a non-monotone strategy is to be used
  
       data%S%nmhist = control%non_monotone

!  The problem is generally constrained
  
       IF ( data%S%p_type == 3 ) THEN
  
!  Set initial real values
  
         data%S%tau = control%mu_decrease
         data%S%tau_steering = control%mu_steering_decrease
         data%S%gamma1 = point1
         data%S%alphae = point1 ; data%S%alphao = one
         data%S%betae = point9 ; data%S%betao = one
         data%S%epstol = epsmch ** 0.75
         inform%mu = MAX( epsmch, control%initial_mu )
         inform%cnorm = HUGE( one ) ; data%S%cnorm_major = inform%cnorm
         data%S%omega_min = control%stopg ; data%S%eta_min = control%stopc
         data%S%epsgrd = control%stopg
         data%S%omega0 = control%firstg                                        &
           / MIN( inform%mu, data%S%gamma1 ) ** data%S%alphao
         data%S%eta0 = control%firstc                                          &
           / MIN( inform%mu, data%S%gamma1 ) ** data%S%alphae
         data%S%icrit = 0 ; data%S%ncrit = 9
         data%S%itzero = .TRUE.
  
!  Set the convergence tolerances
  
         data%S%alphak = MIN( inform%mu, data%S%gamma1 )
         data%S%etak   = MAX( data%S%eta_min,                                  &
                              data%S%eta0 * data%S%alphak ** data%S%alphae )
         data%S%omegak = MAX( data%S%omega_min,                                &
                              data%S%omega0 * data%S%alphak ** data%S%alphao )
         IF ( data%S%printi )                                                  &
           WRITE( data%S%out, 2010 ) inform%mu, data%S%omegak, data%S%etak
       ELSE
         data%S%omegak = control%stopg
       END IF
     END IF
  
     IF ( inform%status == 0 ) THEN

!  Check that ELFUN has not been provided when ELDERS is present

       IF ( use_elders .AND. PRESENT( ELFUN ) ) THEN
         inform%status = 16 ; RETURN ; END IF

!  Check that if ELFUN_flexible is present, then so is ELDERS

       IF ( PRESENT( ELFUN_flexible ) .AND. .NOT. use_elders ) THEN
         inform%status = 17 ; RETURN ; END IF

!  If the element functions are to be evaluated internally, check that
!  the user has supplied appropriate information

       IF ( internal_el ) THEN
         IF ( ALLOCATED( prob%ISTEPA ) .AND. ALLOCATED( prob%EPVALU ) ) THEN
           IF ( SIZE( prob%ISTEPA ) < prob%nel + 1 ) THEN
             inform%status = 10 ; RETURN ; END IF
           IF ( SIZE( prob%EPVALU ) < prob%ISTEPA( prob%nel + 1 ) - 1 ) THEN
             inform%status = 10 ; RETURN ; END IF
         ELSE
           inform%status = 10 ; RETURN
         END IF
       END IF

!  Do the same if the group functions are to be evaluated internally.

       IF ( internal_gr ) THEN
         IF ( ALLOCATED( prob%ISTGPA ) .AND. ALLOCATED( prob%ITYPEG ) .AND.    &
              ALLOCATED( prob%GPVALU ) ) THEN
           IF ( SIZE( prob%ISTGPA ) < prob%ng + 1 .OR.                         &
                SIZE( prob%ITYPEG ) < prob%ng ) THEN
             inform%status = 11 ; RETURN ; END IF
           IF ( SIZE( prob%GPVALU ) < prob%ISTGPA( prob%ng + 1 ) - 1 ) THEN
             inform%status = 11 ; RETURN ; END IF
         ELSE
           inform%status = 11 ; RETURN
         END IF
       END IF

!  Allocate extra local workspace when there are constraints

       IF ( data%S%p_type == 2 .OR. data%S%p_type == 3 ) THEN
         ALLOCATE( data%GROUP_SCALING( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%GROUP_SCALING' ; GO TO 980 ; END IF
         ALLOCATE( data%GXEQX_AUG( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           bad_alloc = 'data%GXEQX_AUG' ; GO TO 980 ; END IF
       END IF
  
!  The problem is generally constrained
  
       IF ( data%S%p_type == 3 ) THEN

!  Set initial integer values
  
         data%S%m = 0 ; data%S%nobjgr = 0
         DO ig = 1, prob%ng

!  KNDOFG = 1 correspomd to objective groups, while KNDOFG = 0
!  are groups which are to be excluded from the problem solved.
!  KNDOFG > 1 corresponds to constraint groups. More specifically,
!  KNDOFG = 2 is a general equality constraint, while KNDOFG = 3,4
!  are general equalities resulting after appending a slack variable to 
!  less-than-or-equal or greater-than-or-equal inequalities respectively

           IF ( prob%KNDOFG( ig ) >= 2 ) THEN
             IF ( prob%KNDOFG( ig ) > 4 ) THEN
               inform%status = 7
               RETURN
             ELSE
               data%S%m = data%S%m + 1
             END IF
           ELSE
             data%S%nobjgr = data%S%nobjgr + 1
           END IF
         END DO
  
!  Set initial values for the internal group scalings, GROUP_scaling, 
!  and the array, GXEQX_aug, which tells if each group is trivial
  
         IF ( prob%ng > 0 ) THEN
           WHERE ( prob%KNDOFG > 1 )
             data%GROUP_SCALING = one ; data%GXEQX_AUG = .FALSE.
           ELSEWHERE
             data%GROUP_SCALING = prob%GSCALE ; data%GXEQX_AUG = prob%GXEQX
           END WHERE
         END IF
         GXEQX_used => data%GXEQX_AUG
  
!  The problem is un-/bound-constrained
  
       ELSE IF ( data%S%p_type == 2 ) THEN
         data%S%m = prob%ng ; data%S%nobjgr = 0
         data%GROUP_SCALING = one ; data%GXEQX_AUG = .FALSE.
         GXEQX_used => data%GXEQX_AUG
  
!  The problem is un-/bound-constrained
  
       ELSE
         data%S%m = 0 ; data%S%nobjgr = prob%ng
         GXEQX_used => prob%GXEQX
       END IF
       data%S%violation = zero

       IF ( data%S%printi ) WRITE( data%S%out, 2000 )

!  Print details of the objective function characteristics
  
       IF ( data%S%printi ) WRITE( data%S%out,                                 &
         "( /, ' There are ', I8, ' variables', /,                             &
      &        ' There are ', I8, ' groups', /,                                &
      &        ' There are ', I8, ' nonlinear elements ' )" )                  &
               prob%n, prob%ng, prob%nel
  
       IF ( data%S%printm ) THEN
         WRITE( data%S%out, "( /, ' ------- Group information ------ ' )" )
         IF ( data%S%printd .OR. prob%ng <= 100 ) THEN
           DO ig = 1, prob%ng
             k1 = prob%ISTADG( ig ) ; k2 = prob%ISTADG( ig + 1 ) - 1
  
!  Print details of the groups
  
             IF ( k1 <= k2 ) THEN
               IF ( k1 == k2 ) THEN
                 WRITE( data%S%out, "( /, ' Group ', I5, ' contains ', I5,     &
                &  ' nonlinear element.  This  is  element ', I5 )" )          &
                   ig, 1, prob%IELING( k1 )
               ELSE
                 WRITE( data%S%out, "( /, ' Group ', I5, ' contains ', I5,     &
                &  ' nonlinear element( s ). These are element( s )', 2I5,     &
                &  /, ( 16I5 ) )" ) ig, k2 - k1 + 1, prob%IELING( k1 : k2 )
               END IF
             ELSE
               WRITE( data%S%out, "( /, ' Group ', I5,                         &
              &  ' contains     no nonlinear', ' elements. ')" ) ig
             END IF
             IF ( .NOT. prob%GXEQX( ig ) )                                     &
               WRITE( data%S%out, "( '  * The group function is non-trivial')" )
  
!  Print details of the nonlinear elements
  
             WRITE( data%S%out,                                                &
               "( :, '  * The group has a linear element with variable( s )',  &
            &       ' X( i ), i =', 3I5, /, ( 3X, 19I5 ) )" )                  &
               prob%ICNA( prob%ISTADA( ig ) : prob%ISTADA( ig + 1 ) - 1 )
           END DO
         END IF
         IF ( .NOT. data%S%alllin ) THEN
           WRITE( data%S%out, "( /, ' ------ Element information ----- ' )" )
           IF ( data%S%printd .OR. prob%nel<= 100 ) THEN
             DO iel = 1, prob%nel
               k1 = prob%ISTAEV( iel ) ; k2 = prob%ISTAEV( iel + 1 ) - 1
  
!  Print details of the nonlinear elements
  
               IF ( k1 <= k2 ) THEN
                 WRITE( data%S%out, "( /, ' Nonlinear element', I5, ' has ',   &
                &  I4, ' internal and ', I4, ' elemental variable( s ), ', /,  &
                &  ' X( i ), i =   ', 13I5, /, ( 16I5 ) )" )                   &
                  iel, prob%INTVAR( iel ), k2 - k1 + 1, prob%IELVAR( k1 : k2 )
               ELSE
                 WRITE( data%S%out, "( /, ' Nonlinear element', I5,            &
                &  ' has   no internal',                                       &
                &  ' or       elemental variables.' )" ) iel
               END IF
             END DO
           END IF
         END IF
       END IF
  
!  Partition the workspace array FUVALS and initialize other workspace
!  arrays

       data%S%ntotel = prob%ISTADG( prob%ng  + 1 ) - 1
       data%S%nvrels = prob%ISTAEV( prob%nel + 1 ) - 1
       data%S%nnza   = prob%ISTADA( prob%ng  + 1 ) - 1

       IF ( ALLOCATED( prob%KNDOFG ) ) THEN
       CALL INITW_initialize_workspace(                                        &
             prob%n, prob%ng, prob%nel,                                        &
             data%S%ntotel, data%S%nvrels, data%S%nnza, prob%n,                &
             data%S%nvargp, prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,&
             prob%INTVAR, prob%ISTADH, prob%ICNA, prob%ISTADA, prob%ITYPEE,    &
             GXEQX_used, prob%INTREP, data%S%altriv, data%S%direct,            &
             data%S%fdgrad, data%S%lfxi, data%S%lgxi, data%S%lhxi,             &
             data%S%lggfx, data%S%ldx, data%S%lnguvl, data%S%lnhuvl,           &
             data%S%ntotin, data%S%ntype, data%S%nsets , data%S%maxsel,        &
             RANGE, data%S%print_level, data%S%out, control%io_buffer,         &
!  workspace
             data%S%EXTEND%lwtran, data%S%EXTEND%litran,                       &
             data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,               &
             data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,              &
             data%ITRANS, data%LINK_elem_uses_var, data%WTRANS,                &
             data%ISYMMD, data%ISWKSP, data%ISTAJC, data%ISTAGV,               &
             data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR,               &
             data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,               &
             data%ISET  , data%ISVSET, data%INVSET, data%LIST_elements,        &
             data%ISYMMH, data%IW_asmbl, data%NZ_comp_w, data%W_ws,            &
             data%W_el, data%W_in, data%H_el, data%H_in,                       &
             inform%status, alloc_status, bad_alloc,                           &
             data%S%skipg, KNDOFG = prob%KNDOFG )
       ELSE
       CALL INITW_initialize_workspace(                                        &
             prob%n, prob%ng, prob%nel,                                        &
             data%S%ntotel, data%S%nvrels, data%S%nnza, prob%n,                &
             data%S%nvargp, prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,&
             prob%INTVAR, prob%ISTADH, prob%ICNA, prob%ISTADA, prob%ITYPEE,    &
             GXEQX_used, prob%INTREP, data%S%altriv, data%S%direct,            &
             data%S%fdgrad, data%S%lfxi, data%S%lgxi, data%S%lhxi,             &
             data%S%lggfx, data%S%ldx, data%S%lnguvl, data%S%lnhuvl,           &
             data%S%ntotin, data%S%ntype, data%S%nsets , data%S%maxsel,        &
             RANGE, data%S%print_level, data%S%out, control%io_buffer,         &
!  workspace
             data%S%EXTEND%lwtran, data%S%EXTEND%litran,                       &
             data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,               &
             data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,              &
             data%ITRANS, data%LINK_elem_uses_var, data%WTRANS,                &
             data%ISYMMD, data%ISWKSP, data%ISTAJC, data%ISTAGV,               &
             data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR,               &
             data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,               &
             data%ISET  , data%ISVSET, data%INVSET, data%LIST_elements,        &
             data%ISYMMH, data%IW_asmbl, data%NZ_comp_w, data%W_ws,            &
             data%W_el, data%W_in, data%H_el, data%H_in,                       &
             inform%status, alloc_status, bad_alloc,                           &
             data%S%skipg )
       END IF

       IF ( inform%status == 12 ) THEN
         inform%alloc_status = alloc_status
         inform%bad_alloc = bad_alloc
       END IF
       IF ( inform%status /= 0 ) RETURN                              
                                                                              
!  Allocate arrays                                                          
                                                                              
       FUVALS = - HUGE( one )  ! needed for epcf90 debugging compiler

       array_name = 'lancelot: data%P'
       CALL SPACE_resize_array( prob%n, data%P,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%XCP'
       CALL SPACE_resize_array( prob%n, data%XCP,                              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%X0'
       CALL SPACE_resize_array( prob%n, data%X0,                               &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%GX0'
       CALL SPACE_resize_array( prob%n, data%GX0, inform%status,               &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%DELTAX'
       CALL SPACE_resize_array( prob%n, data%DELTAX,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%QGRAD'
       CALL SPACE_resize_array( MAX( prob%n, data%S%ntotin ), data%QGRAD,      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%GRJAC'
       CALL SPACE_resize_array( data%S%nvargp, data%GRJAC,                     &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%BND'
       CALL SPACE_resize_array( prob%n, 2, data%BND,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%BREAKP'
       CALL SPACE_resize_array( prob%n, data%BREAKP,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980


       IF ( data%S%xactcp ) THEN
         array_name = 'lancelot: data%GRAD'
         CALL SPACE_resize_array( 0, data%GRAD,                                &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       ELSE       
         array_name = 'lancelot: data%GRAD'
         CALL SPACE_resize_array( prob%n, data%GRAD,                           &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       END IF

       data%S%nbnd = prob%n
       IF ( data%S%mortor .AND. .NOT. data%S%twonrm ) THEN
         array_name = 'lancelot: data%BND_radius'
         CALL SPACE_resize_array( data%S%nbnd, 2, data%BND_radius,             &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       ELSE IF ( data%S%strctr ) THEN
         array_name = 'lancelot: data%BND_radius'
         CALL SPACE_resize_array( data%S%nbnd, 1, data%BND_radius,             &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       ELSE
         data%S%nbnd = 0
         array_name = 'lancelot: data%BND_radius'
         CALL SPACE_resize_array( data%S%nbnd, 2, data%BND_radius,             &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       END IF
       
       IF ( data%S%strctr ) THEN
!        ALLOCATE( data%D_model( prob%ng ), STAT = alloc_status )
!        IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%D_model' ; GO TO 980
!        END IF
         
!        ALLOCATE( data%D_function( prob%ng ), STAT = alloc_status )
!        IF ( alloc_status /= 0 ) THEN 
!          bad_alloc = 'data%D_function' ; GO TO 980
!        END IF
         
         array_name = 'lancelot: data%RADII'
         CALL SPACE_resize_array( prob%ng, data%RADII,                         &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'lancelot: data%GV_old'
         CALL SPACE_resize_array( prob%ng, data%GV_old,                        &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       ELSE
         array_name = 'lancelot: data%RADII'
         CALL SPACE_resize_array( 0, data%RADII,                               &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'lancelot: data%GV_old'
         CALL SPACE_resize_array( 0, data%GV_old,                              &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       END IF
  
       IF ( data%S%steering ) THEN
         array_name = 'lancelot: data%CDASH'
         CALL SPACE_resize_array( prob%ng, data%CDASH,                         &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980

         array_name = 'lancelot: data%C2DASH'
         CALL SPACE_resize_array( prob%ng, data%C2DASH,                        &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       END IF

!  Store the free variables as the the first nfree components of IFREE
  
       array_name = 'lancelot: data%IFREE'
       CALL SPACE_resize_array( prob%n, data%IFREE,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
  
!  INDEX( j ), j = 1, ..., n, will contain the status of the
!  j-th variable as the current iteration progresses. Possible values
!  are 0 if the variable lies away from its bounds, 1 and 2 if it lies
!  on its lower or upper bounds (respectively) - these may be problem
!  bounds or trust-region bounds, and 3 if the variable is fixed
  
       array_name = 'lancelot: data%INDEX'
       CALL SPACE_resize_array( prob%n, data%INDEX,                            &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

  
!  IFREEC( j ), j = 1, ..., n will give the indices of the
!  variables which are considered to be free from their bounds at the
!  current generalized cauchy point
  
       array_name = 'lancelot: data%IFREEC'
       CALL SPACE_resize_array( prob%n, data%IFREEC,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

  
!  INNONZ( j ), j = 1, ..., nnnonz will give the indices of the nonzeros
!  in the vector obtained as a result of the matrix-vector product from
!  subroutine HSPRD
  
       array_name = 'lancelot: data%INNONZ'
       CALL SPACE_resize_array( prob%n, data%INNONZ,                           &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

  
!  Make space for finite-difference values if required
  
       IF ( data%S%fdgrad .AND. .NOT. data%S%alllin ) THEN
         array_name = 'lancelot: data%FUVALS_temp'
         CALL SPACE_resize_array( prob%nel, data%FUVALS_temp,                  &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       ELSE
         array_name = 'lancelot: data%FUVALS_temp'
         CALL SPACE_resize_array( 0, data%FUVALS_temp,                         &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       END IF

!  Space required for the Schur complement

       data%SCU_matrix%m = 0
       data%SCU_matrix%n = 1
       data%SCU_matrix%m_max = MAX( control%max_sc, 1 )
       data%SCU_matrix%class = 4
     
       array_name = 'lancelot: data%SCU_matrix%BD_col_start'
       CALL SPACE_resize_array( data%SCU_matrix%m_max + 1,                     &
              data%SCU_matrix%BD_col_start,                                    &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%SCU_matrix%BD_row'
       CALL SPACE_resize_array( data%SCU_matrix%m_max, data%SCU_matrix%BD_row, &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'lancelot: data%SCU_matrix%BD_val'
       CALL SPACE_resize_array( data%SCU_matrix%m_max, data%SCU_matrix%BD_val, &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
      
!  Space required for the factors of the Schur complement

       data%SCU_matrix%BD_col_start( 1 ) = 1
       scu_status = 1
       CALL SCU_factorize( data%SCU_matrix, data%SCU_data, data%P, scu_status, &
                           inform%SCU_info )
       IF ( scu_status /= 0 ) THEN
         WRITE( data%S%out, "( ' SCU_factorize: status = ', I2 )" ) scu_status
         inform%status = 12
         inform%alloc_status = inform%SCU_info%alloc_status
         inform%bad_alloc = 'SCU_factorize array'
         RETURN
       END IF
     END IF

!  ===============================================
!  Call the solver to perform the bulk of the work
!  ===============================================

!  Both internal element and group evaluations will be performed
!  -------------------------------------------------------------

     IF ( internal_el .AND. internal_gr ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

        IF ( data%S%skipg ) THEN
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          END IF

!  Use all groups

        ELSE
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          END IF
        END IF
 
!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
        IF ( use_elders ) THEN
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ENDIF
       END IF

!  Just internal element evaluations will be performed
!  ---------------------------------------------------

     ELSE IF ( internal_el ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

        IF ( data%S%skipg ) THEN

          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          END IF

!  Use all groups

        ELSE
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          END IF
        END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
        IF ( use_elders ) THEN
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        END IF
       END IF

!  Just internal group evaluations will be performed
!  -------------------------------------------------

     ELSE IF ( internal_gr ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

         IF ( data%S%skipg ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )

!  Use all groups

         ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
         END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
       END IF

!  Element and group evaluations will be performed via reverse communication
!  -------------------------------------------------------------------------

     ELSE

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

         IF ( data%S%skipg ) THEN

           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional argument
             ELDERS = ELDERS )

!  Use all groups

         ELSE

           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &
             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional argument
             ELDERS = ELDERS )
         END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &

             data%S%EXTEND%lrowst, data%S%EXTEND%lpos, data%S%EXTEND%lused,    &
             data%S%EXTEND%lfilled, data%ITRANS, data%ROW_start,               &
             data%POS_in_H, data%USED, data%FILLED,                            &

             data%LINK_elem_uses_var, data%WTRANS,                             &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y, ELDERS = ELDERS,    &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )

       END IF
     END IF

     RETURN

!  Unsuccessful returns

 980 CONTINUE
     inform%status = 12
     inform%alloc_status = alloc_status
     inform%bad_alloc = bad_alloc
     WRITE( data%S%error, 2990 ) alloc_status, TRIM( bad_alloc )

     NULLIFY( GXEQX_used )

     IF ( ASSOCIATED( data%GXEQX_AUG ) ) THEN
       DEALLOCATE( data%GXEQX_AUG, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GXEQX_AUG'
         WRITE( data%S%error, 2990 ) alloc_status, TRIM( inform%bad_alloc )
       END IF
     END IF

     IF ( ASSOCIATED( data%GROUP_SCALING ) ) THEN
       DEALLOCATE( data%GROUP_SCALING, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GROUP_SCALING'
         WRITE( data%S%error, 2990 ) alloc_status, TRIM( inform%bad_alloc )
       END IF
     END IF

     RETURN

!  Non-executable statements

 2000  FORMAT( /, ' *********  Starting optimization  ************** ' )
 2010  FORMAT( /, ' Penalty parameter ', ES12.4,                               &
                  ' Required projected gradient norm = ', ES12.4, /,           &
                  '                   ', 12X,                                  &
                  ' Required constraint         norm = ', ES12.4 )             
 2990  FORMAT( ' ** Message from -LANCELOT_solve-', /,                         &
               ' Allocation error (status = ', I0, ') for ', A )

!  End of subroutine LANCELOT_solve

     END SUBROUTINE LANCELOT_solve

!-*-*-*-  L A N C E L O T -B- LANCELOT_solve_main  S U B R O U T I N E  -*-*-*-

     SUBROUTINE LANCELOT_solve_main(                                           &
                      n, ng, nel, lfuval,                                      &
                      IELING, ISTADG, IELVAR, ISTAEV, INTVAR, ISTADH,          &
                      ICNA  , ISTADA, A , B , BL, BU, GSCALE, ESCALE, VSCALE,  &
                      GXEQX , INTREP, RANGE , X ,     GVALS , FT, XT, FUVALS,  &
                      ICALCF, ICALCG, IVAR  , Q     , DGRAD , VNAMES, GNAMES,  &
                      ITYPEE, control, inform, S    ,                          &
!  workspace
                      lirnh, ljcnh, lirnh_min, ljcnh_min,                      &
                      lh, lh_min, lrowst, lpos, lused, lfilled,                &
                      ITRANS, ROW_start, POS_in_H, USED, FILLED,               &
                      LINK_elem_uses_var    , WTRANS,                          &
                      DIAG  , OFFDIA, IW    , IKEEP  , IW1  ,                  &
                      IVUSE , H_col_ptr, L_col_ptr,                            &
                      W , W1, RHS   , RHS2   , P2, G,                          &
                      IW_asmbl, NZ_comp_w, W_ws, W_el, W_in, H_el, H_in,       &
                      ISYMMD, ISWKSP, ISTAJC, ISTAGV, ISVGRP, ISLGRP,          &
                      IGCOLJ, IVALJR, IUSED , ITYPER, ISSWTR, ISSITR,          &
                      ISET  , ISVSET, INVSET, IFREE , INDEX , IFREEC,          &
                      INNONZ, LIST_elements , ISYMMH, FUVALS_temp   ,          &
                      P , X0, XCP   , GX0   , RADII , DELTAX, QGRAD ,          &
                      GRJAC , GV_old, BND   , BND_radius    ,                  &
                      BREAKP, GRAD, SCU_matrix, SCU_data,                      &
                      matrix, SILS_data,                                       &
!  optional arguments
                      KNDOFG, C, Y  ,                                          &
                      ELDERS, ELFUN_flexible, ELFUN , ISTEPA , EPVALU,         &
                      GROUP , ISTGPA, ITYPEG, GPVALU, GROUP_SCALING, GXEQX_AUG )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  LANCELOT_solve, a method for finding a local minimizer of a
!  function subject to general constraints and simple bounds on the sizes of
!  the variables. The method is described in the paper 'A globally convergent
!  augmented Lagrangian algorithm for optimization with general constraints
!  and simple bounds' by A. R. Conn, N. I. M. Gould and Ph. L. Toint,
!  SIAM J. Num. Anal. 28 (1991) PP.545-572
!
!  The objective function is assumed to be of the form
!
!                             ISTADG(obj+1)-1
!      F( X ) = SUM GS   * G(   SUM    ES * F( X ) + A (TRANS) X - B )
!               obj   obj   obj  j=      j   j        m+1           m+1
!            IN OBJSET        ISTADG(obj)
!
!  and the constraints (i = 1, .... , ng, i .NE. objset) of the form
!
!                        ISTADG(i+1)-1
!      CI( X ) = GS * G(   SUM     ES * F ( X ) + A (TRANS) X - B ) = 0
!                  i   i j=ISTADG(i) j   j         i             i
!
!  Where the Fj( X ) are known as nonlinear element functions, the
!  Ai(trans) X + Bi are the linear element functions, the GSi are group
!  weights and the ESi are element weights. Each Fj is expected only to
!  involve a few 'internal' variables, that is there is a linear
!  transformation from the problem variables to the variables actually needed
!  to define the element (and its derivatives) whose range space is very small

!  Contents of the array FUVALS:
!  -----------------------------

!     <-nel-><-- ntotin --> <-- nhel --> <---- n ---> <-

!    ---------------------------------------------------------
!    |  Fj(X)  |  grad Fj(X)  | Hess Fj(X) | grad F(X)) | ....
!    ---------------------------------------------------------
!   |         |              |            |            |
!  lfxi     lgxi           lhxi         lggfx         ldx
!  (=0)
!        -> <------- n ------> <-- nvargp ->

!       ------------------------
!      ... | Diag scaling F(X) |
!       ------------------------
!         |                   | 
!        ldx                lend

!  Only the upper triangular part of each element Hessian is stored;
!  the storage is by columns

!  Contents of the arrays ISTADG, ESCALE AND IELING:
!  ------------------------------------------------

!           <---------------------- S%ntotel ------------------------>

!           --------------------------------------------------------
!  ESCALE:  | el.scales | el.scales | ............... | el.scales  |
!           | group 1   | group 2   |                 | group ng   |
!           --------------------------------------------------------
!  IELING:  | elements  | elements  | ............... | elements   |
!           | group 1   | group 2   |                 | group ng   |
!           --------------------------------------------------------
!            |           |                             |            |
!            | |--- > ---|                             |            |
!            | |   |-------------------- > ------------|            |
!            | |   | |------------------------- > ------------------|
!            ---------
!  ISTADG:   | ..... |    pointer to the position of the 1st element
!            ---------    of each group within the array
!            <-ng+1->

!  Contents of the arrays IELVAR and ISTAEV:
!  ----------------------------------------

!          <--------------------- nelvar -------------------------->

!          ---------------------------------------------------------
!          | variables | variables | ............... |  variables  |
!  IELVAR: | element 1 | element 2 |                 | element nel |
!          ---------------------------------------------------------
!           |           |                             |             |
!           | |--- > ---|                             |             |
!           | |    |------------------- > ------------|             |
!           | |    | |----------------- > --------------------------|
!           ----------
!  ISTAEV:  | ...... |    pointer to the position of the 1st variable
!           ----------    in each element (including one to the end).
!          <- nel+1 ->

!  Contents of the array INTVAR:
!  -----------------------------

!  On initial entry, INTVAR( i ), i = 1, ... , nel, gives the number of
!  internal variables for element i. Thereafter, INTVAR provides pointers to
!  the start of each element gradient with respect to its internal variables
!  as follows:

!         -> <---------------------- ntotin -----------------------> <-

!         -------------------------------------------------------------
!  part of  | gradient  | gradient  | ............... |  gradient   | .
!  FUVALS   | element 1 | element 2 |                 | element nel |
!         -------------------------------------------------------------
!          | |           |                             |           | |
!       lgxi | |--- > ---|                             |         lhxi|
!            | |   |-------------------- > ------------|             |
!            | |   | |------------------------- > -------------------|
!            ---------
!  INTVAR:   | ..... |    pointer to the position of the 1st entry of
!            ---------    the gradient for each element
!            <-nel+1->

!  Contents of the array ISTADH:
!  -----------------------------

!         -> <---------------------- nhel -------------------------> <-

!         -------------------------------------------------------------
!  part of  | Hessian   | Hessian   | ............... |  Hessian    | .
!  FUVALS   | element 1 | element 2 |                 | element nel |
!         -------------------------------------------------------------
!          | |           |                             |           |
!       lhxi | |--- > ---|                             |         lggfx
!            | |    |------------------- > ------------|
!            | |    |
!            ---------
!  ISTADH:   | ..... |    pointer to the position of the 1st entry of the
!            ---------    Hessian for each element, with respect to its
!                         internal variables
!            <- nel ->

!  Contents of the arrays A, ICNA AND ISTADA:
!  ------------------------------------------

!          <--------------------- na ----------------------------->

!          ---------------------------------------------------------
!          |   values  |   values  | ............... |    values   |
!  A:      |    A(1)   |    A(2)   |                 |     A(ng)   |
!          ---------------------------------------------------------
!          | variables | variables | ............... |  variables  |
!  ICNA:   |    A(1)   |    A(2)   |                 |     A(ng)   |
!          ---------------------------------------------------------
!           |           |                             |             |
!           | |--- > ---|                             |             |
!           | |    |------------------- > ------------|             |
!           | |    | |----------------- > --------------------------|
!           ----------
!  ISTADA:  | ...... |    pointer to the position of the 1st variable in the
!           ----------    linear element for each group (including one to the
!                         end)

!           <- ng+1 ->

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  If the routine terminates with a negative value of inform%status, the user is
!  asked to re-enter the subroutine with further information

!  If status  = - 1, the user must supply the function and derivative values of
!                    each Fj at the point XT.
!  If status  = - 2, the user must supply the function and derivative of each
!                    Gi for the argument FT(i) .
!  If status  = - 3, the user must supply the value, alone, of each function
!                    Fj evaluated at the point XT.
!  If status  = - 4, the user must supply the value of each function Gi, alone,
!                    for the argument FT(i).
!  If status  = - 5, the user must supply the derivatives, alone, of the
!                    functions Fj and Gi at the point XT and argument FT(i)
!                    respectively.
!  If status  = - 6, the user must supply the derivatives, alone, of the
!                    functions Fj at the point XT.
!  If status  = - 7, the user must supply the value, alone, of each function
!                    Fj evaluated at the point XT.
!  If status  = - 8, 9, 10, the user must provide the product of the inverse
!                    of the preconditioning matrix and the vector GRAD. The
!                    nonzero components of GRAD occur in positions IVAR(i),
!                    i = 1,..,nvar and have the values DGRAD(i). The product
!                    must be returned in the vector Q. This return is only
!                    possible if ICHOSE( 2 ) is 3.
!  If status  = - 12, the user must supply the derivatives, alone, of the
!                    functions Fj at the point XT.
!  If status  = - 13, the user must supply the derivative valuse of each 
!                     function Gi, alone, for the argument FT(i).

!  If the user does not wish to compute an element or group function at
!  a particular argument returned from the inner iteration, she may
!  reset status to -11 and re-enter. The routine will treat such a re-entry as
!  if the current iteration had been unsuccessful and reduce the trust region.
!  This facility is useful when, for instance, the user is asked to evaluate
!  a function at a point outside its domain of definition.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE( LANCELOT_control_type ), INTENT( INOUT ) :: control
     TYPE( LANCELOT_inform_type ), INTENT( INOUT ) :: inform
     TYPE( LANCELOT_save_type ), INTENT( INOUT ) :: S
     INTEGER, INTENT( IN ) :: n, ng, nel, lfuval
     INTEGER, INTENT( IN ), DIMENSION( ng + 1 ) :: ISTADA
     INTEGER, INTENT( IN ), DIMENSION( ISTADA( ng + 1 ) - 1 ) :: ICNA
     INTEGER, INTENT( INOUT ), DIMENSION( ng + 1 ) :: ISTADG
     INTEGER, INTENT( IN ), DIMENSION( ISTADG( ng + 1 ) - 1 ) :: IELING
     INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( ISTAEV( nel + 1 ) - 1 ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: ISTADH, INTVAR
     INTEGER, INTENT( INOUT ), DIMENSION( n  ) :: IVAR
     INTEGER, INTENT( INOUT ), DIMENSION( nel ) :: ICALCF
     INTEGER, INTENT( INOUT ), DIMENSION( ng ) :: ICALCG
     REAL ( KIND = wp ), INTENT( IN  ),                                        &
                               DIMENSION( ISTADA( ng + 1 ) - 1 ) :: A
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: BL, BU
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: B
     REAL ( KIND = wp ), INTENT( IN  ),                                        &
            DIMENSION( ISTADG( ng + 1 ) - 1 ) :: ESCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng, 3 ) :: GVALS 
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
            DIMENSION( n ) :: X, Q, XT, DGRAD, VSCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng ) :: FT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     REAL ( KIND = wp ), INTENT( IN  ), TARGET, DIMENSION( ng ) :: GSCALE
     LOGICAL, INTENT( IN ), TARGET, DIMENSION( ng ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     CHARACTER ( LEN = 10 ), INTENT( IN ), DIMENSION( n ) :: VNAMES
     CHARACTER ( LEN = 10 ), INTENT( IN ), DIMENSION( ng ) :: GNAMES

!--------------------------------------------------------------
!   D u m m y   A r g u m e n t s  f o r   W o r k s p a c e 
!--------------------------------------------------------------

     INTEGER, INTENT( INOUT ) :: lirnh, ljcnh, lirnh_min, ljcnh_min, lh, lh_min
     INTEGER, INTENT( INOUT ) :: lrowst, lpos, lused, lfilled
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITRANS 
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW_start
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: POS_in_H
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: USED
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: FILLED

     INTEGER, ALLOCATABLE, DIMENSION( : ) :: LINK_elem_uses_var
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WTRANS
 
     INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: IKEEP, IW1
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, IVUSE
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_col_ptr, L_col_ptr
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) ::                        &
       W, RHS, RHS2, P2, G , DIAG
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: W1, OFFDIA
     
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISYMMD, ISWKSP, ISTAJC
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISTAGV, ISVGRP, ISLGRP
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IGCOLJ, IVALJR, IUSED 
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ITYPER, ISSWTR, ISSITR
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISET  , ISVSET, INVSET
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IFREE , INDEX , IFREEC
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: INNONZ, LIST_elements
     INTEGER, INTENT( INOUT ), DIMENSION( : , : ) :: ISYMMH
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: FUVALS_temp
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: P, X0
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: XCP
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: GX0, RADII
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: DELTAX
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: QGRAD, GRJAC
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: GV_old
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : , : ) :: BND, BND_radius
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: BREAKP, GRAD
     
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IW_asmbl
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: NZ_comp_w
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: W_ws
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: H_el
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: H_in

     TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_matrix
     TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
     TYPE ( SMT_type ), INTENT( INOUT ) :: matrix
     TYPE ( SILS_factors ), INTENT( INOUT ) :: SILS_data

!  local arrays (move to data later)

     REAL ( KIND = wp ), DIMENSION( ng ) :: C_best, CDASH, C2DASH, GVALS2_GN

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for RANGE

       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                          lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN 

!  Interface block for ELFUN_flexible 

       SUBROUTINE ELFUN_flexible(                                              &
                          FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, llders, ifflag, ELDERS,      &
                          ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( IN ) :: llders
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf), ELDERS(2,llders)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN_flexible

!  Interface block for GROUP

       SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,              &
                          ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,              &
                          lcalcg, lfvalu, lgpvlu, derivs, igstat )
       INTEGER, INTENT( IN ) :: lgvalu, ncalcg
       INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
       INTEGER, INTENT( OUT ) :: igstat
       LOGICAL, INTENT( IN ) :: derivs
       INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
       INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
       INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lfvalu ) :: FVALUE
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lgpvlu ) :: GPVALU
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ),                        &
                                       DIMENSION ( lgvalu, 3 ) :: GVALUE
       END SUBROUTINE GROUP

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( nel + 1 ) :: ISTEPA
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng + 1 ) :: ISTGPA
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: ITYPEG
     INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( 2, nel ) :: ELDERS
     REAL ( KIND = wp ), INTENT( INOUT ), OPTIONAL,                            &
            DIMENSION( ng ) :: C, Y
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL,                               &
            DIMENSION( : ) :: EPVALU
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL,                               &
            DIMENSION( : ) :: GPVALU
     REAL ( KIND = wp ), INTENT( IN  ), OPTIONAL, TARGET,                      &
            DIMENSION( ng ) :: GROUP_SCALING
     LOGICAL, INTENT( IN ), OPTIONAL, TARGET, DIMENSION( ng ) :: GXEQX_AUG
     OPTIONAL :: ELFUN, ELFUN_flexible, GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ig, ic, is, j, lgfx, ifixd, k, k1, k2, l, ifflag
     INTEGER :: ipdgen, iddgen, istate, ir, nvar1, alloc_status
     REAL ( KIND = KIND( 1.0E0 ) ) :: tim
     REAL ( KIND = wp ) :: hmuinv, yiui, scaleg, epsmch, epslam, hdash, ctt
     REAL ( KIND = wp ) :: ftt, xi, gi, bli, bui, dltnrm, val
     REAL ( KIND = wp ) :: gnorm, distan, ar_h, pr_h, slope
     LOGICAL :: external_el, external_gr, start_p, alive, use_elders, reduced_mu
     CHARACTER ( LEN = 7 ) :: atime
     CHARACTER ( LEN = 6 ) :: citer, cngevl, citcg, cfree
     CHARACTER ( LEN = 80 ) :: bad_alloc

!---------------------------------
!   L o c a l   P o i n t e r s
!---------------------------------

     REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: GSCALE_used
     LOGICAL, POINTER, DIMENSION( : ) :: GXEQX_used

     epsmch = EPSILON( one )
     external_el = .NOT. ( PRESENT( ELFUN ) .OR. PRESENT( ELFUN_flexible ) )
     external_gr = .NOT. PRESENT( GROUP )
     use_elders = PRESENT( ELDERS )

!  If the run is being continued after the "alive" file has been reinstated
!  jump to the appropriate place in the code

     IF ( inform%status == 14 ) THEN
       IF ( S%inform_status < 0 ) THEN
         inform%status = S%inform_status
         GO TO 700
       ELSE
         inform%status = - S%inform_status
         GO TO 800
       END IF
     END IF

!  Branch to the interior of the code if a re-entry is being made

!write(6,*) ' --------------------------------------- ststus ', inform%status
     IF ( inform%status < 0 ) GO TO 810

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

!  Start inner loop to minimize penalty function for the current values of
!  mu and Y

         IF ( S%p_type == 1 ) THEN
           GXEQX_used => GXEQX ; GSCALE_used => GSCALE
         ELSE
           GXEQX_used => GXEQX_AUG ; GSCALE_used => GROUP_SCALING
         END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  The basic algorithm used for the inner iteration is described in the paper 
!  'Testing a class of methods for solving minimization problems with 
!  simple bounds on their variables' by A. R. Conn, N. I. M. Gould and 
!  Ph. L. Toint, Mathematics of Computation, 50 (1988) pp. 399-430

!  The objective function is assumed to be of the form

!               ng
!     F( X ) = sum  GS * G(   sum     ES  * F ( X ) + A (trans) X - B )
!              i=1    i   i j in J(i)   ij   j         i             i

!  Where the F(j)( X ) are known as nonlinear element functions, the
!  A(i)(trans) X - Bi are the linear element functions, the GS(i) are group
!  weights, the ES(ij) are element weights, the G(i) are called group
!  functions and each set J(i) is a subset of the set of the first nel
!  integers. Each F(j) is expected only to involve a few 'internal' variables,
!  that is there is a linear transformation from the problem variables to the
!  variables actually needed to define the element (and its derivatives) whose
!  range space is very small. Each group function is a differentiable
!  univariate function

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Branch to different parts of the code depending on the input value of status

         SELECT CASE ( inform%status )
         CASE ( -  1 ) ; GO TO 40
         CASE ( -  2 ) ; GO TO 70
         CASE ( -  3 ) ; GO TO 430
         CASE ( -  4 ) ; GO TO 470
         CASE ( - 13 : - 12, - 7 : - 5 ) ; GO TO 540
         CASE ( -  8 ) ; GO TO 90
         CASE ( -  9 ) ; GO TO 320
         CASE ( - 10 ) ; GO TO 570
         CASE ( - 11 ) ; GO TO 590
         END SELECT

!  -----------------------------------
!  Step 0 of the algorithm (see paper)
!  -----------------------------------

         DO i = 1, n

!  Project user supplied starting point into the feasible box

           bli = BL( i )
           bui = BU( i )
           IF ( bli > bui ) THEN
             IF ( S%printe ) WRITE( S%out,                                     &
               "( /, ' Lower bound ', ES12.4, ' on variable ', I8,             &
            &        ' larger than upper bound ', ES12.4, //,                  &
            &        ' Execution terminating ' )" ) bli, i, bui
             inform%status = 8 ; GO TO 820
           END IF
           xi = MAX( bli, MIN( bui, X( i ) ) )

!  Find the maximum variable scale factor

           S%vscmax = MAX( S%vscmax, VSCALE( i ) )

!  Find initial active set

           is = 0
           IF ( xi <= bli * ( one + SIGN( S%epstlp, bli ) ) ) is = 1
           IF ( xi >= bui * ( one - SIGN( S%epstln, bui ) ) ) is = 2
           IF ( bui * ( one - SIGN( S%epstln, bui ) ) <=                       &
                bli * ( one + SIGN( S%epstlp, bli ) ) ) is = 3
           INDEX( i ) = is
           IF ( is == 3 ) xi = half * ( bli + bui )

!  Copy the initial point into XT prior to calculating function values

           X( i ) = xi ; XT( i ) = xi
         END DO

!  Ensure that all the element functions are evaluated at the initial point

         inform%ncalcf = nel
         DO i = 1, inform%ncalcf ; ICALCF( i ) = i ; END DO

!  Return to the calling program to obtain the element function
!  and, if possible, derivative values at the initial point

         IF ( S%fdgrad ) S%igetfd = 0
         inform%ngeval = inform%ngeval + 1
         inform%status = - 1
         IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF

!  If finite-difference gradients are used, compute their values

   40    CONTINUE
         IF ( S%fdgrad .AND. .NOT. S%alllin ) THEN

!  Store the values of the nonlinear elements for future use

           IF ( S%igetfd == 0 ) THEN
             FUVALS_temp( : nel ) = FUVALS( : nel )
             S%centrl = S%first_derivatives == 2
           END IF

!  Obtain a further set of differences

           IF ( use_elders ) THEN
             CALL OTHERS_fdgrad_flexible(                                      &
                                 n, nel, lfuval, S%ntotel, S%nvrels, S%nsets,  &
                                 IELVAR, ISTAEV, IELING, ICALCF, inform%ncalcf,&
                                 INTVAR, S%ntype , X , XT, FUVALS, S%centrl,   &
                                 S%igetfd, S%OTHERS, ISVSET, ISET, INVSET,     &
                                 ISSWTR, ISSITR, ITYPER, LIST_elements,        &
                                 LINK_elem_uses_var, WTRANS, ITRANS,           &
                                 ELDERS( 1, : ) )
           ELSE
             CALL OTHERS_fdgrad( n, nel, lfuval, S%ntotel, S%nvrels, S%nsets,  &
                                 IELVAR, ISTAEV, IELING, ICALCF, inform%ncalcf,&
                                 INTVAR, S%ntype , X , XT, FUVALS, S%centrl,   &
                                 S%igetfd, S%OTHERS, ISVSET, ISET, INVSET,     &
                                 ISSWTR, ISSITR, ITYPER, LIST_elements,        &
                                 LINK_elem_uses_var, WTRANS, ITRANS )
           END IF
           IF ( S%igetfd > 0 ) THEN
             IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
           END IF

!  Restore the values of the nonlinear elements at X

           S%igetfd = S%nsets + 1
           FUVALS( : nel ) = FUVALS_temp( : nel )
         END IF

!  The convergence tolerance is modified to reflect the scaling

         S%epscns = S%omegak * S%vscmax

!  Compute the norm of the residual that is to be required when obtaining the
!  approximate minimizer of the model problem

         S%resmin = MIN( tenm4, MAX( S%epsrcg, S%epscns ** 2.02 ) )

!  set the steering counter to zero

         S%n_steering = 0

!  Compute the group argument values FT

         DO ig = 1, ng

!  Include the contribution from the linear element

!          ftt = SUM( A( ISTADA( ig ): ISTADA( ig + 1 ) - 1 ) *                &
!            X( ICNA( ISTADA( ig ) : ISTADA( ig + 1 ) - 1 ) ) ) - B( ig )
           ftt = - B( ig )
           DO i = ISTADA( ig ), ISTADA( ig + 1 ) - 1
             ftt = ftt + A( i ) * X( ICNA( i ) )
           END DO

!  Include the contributions from the nonlinear elements

!          ftt = ftt + SUM( ESCALE( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) *    &
!            FUVALS( IELING( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) ) )
           DO i = ISTADG( ig ), ISTADG( ig + 1 ) - 1
             ftt = ftt + ESCALE( i ) * FUVALS( IELING( i ) )
           END DO
           FT( ig ) = ftt
         END DO

!  Compute the group function values

         IF ( S%altriv ) THEN
!          inform%aug = DOT_PRODUCT( GSCALE_used, FT )
           inform%aug = zero
           IF ( S%skipg ) THEN
             DO ig = 1, ng
               IF ( KNDOFG( ig ) > 0 ) THEN
                 inform%aug = inform%aug + GSCALE_used( ig ) * FT( ig )
                 GVALS( ig, 1 ) = FT( ig )
                 GVALS( ig, 2 ) = one
               END IF
             END DO
           ELSE
             DO ig = 1, ng
               inform%aug = inform%aug + GSCALE_used( ig ) * FT( ig ) ; END DO
             GVALS( : , 1 ) = FT
             GVALS( : , 2 ) = one
           END IF

!  If necessary, return to the calling program to obtain the group function
!  and derivative values at the initial point. Ensure that all the group
!  functions are evaluated at the initial point

         ELSE
           inform%ncalcg = ng
           DO ig = 1, ng
             ICALCG( ig ) = ig
             IF ( GXEQX_used( ig ) ) THEN
               GVALS( ig, 1 ) = FT( ig )
               GVALS( ig, 2 ) = one
             END IF
           END DO
           inform%status = - 2
           IF ( external_gr ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
         END IF

   70    CONTINUE
         S%reusec = .FALSE.
!        IF ( .NOT. S%altriv )                                                 &
!          inform%aug = DOT_PRODUCT( GSCALE_used, GVALS( : , 1 ) )
         IF ( .NOT. S%altriv ) THEN
           inform%aug = zero
           IF ( S%skipg ) THEN
             DO ig = 1, ng
!              write(6,*)  ' kndofg ', KNDOFG( ig ), GSCALE_used( ig ),        &
!                           GVALS( ig, 1 )
               IF ( KNDOFG( ig ) > 0 )                                         &
                inform%aug = inform%aug + GSCALE_used( ig ) * GVALS( ig, 1 )
             END DO
           ELSE
             DO ig = 1, ng 
               inform%aug = inform%aug + GSCALE_used( ig ) * GVALS( ig, 1 )
             END DO
           END IF
         END IF

!  If a structured trust-region is to be used, store the current values
!  of the group functions

         IF ( S%strctr ) THEN
           DO ig = 1, ng
             IF ( GXEQX_used( ig ) ) THEN
               GV_old( ig ) = FT( ig )
             ELSE
               GV_old( ig ) = GVALS( ig, 1 )
             END IF
           END DO
         END IF

!  If a secant method is to be used, initialize the second
!  derivatives of each element as a scaled identity matrix

         CALL CPU_TIME( S%t )
         IF ( .NOT. S%second .AND. .NOT. S%alllin ) THEN
           IF ( use_elders ) THEN
             CALL OTHERS_scaleh_flexible(                                      &
                 .TRUE., n, nel, lfuval, S%nvrels, S%ntotin,                   &
                 inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR, IELVAR,        &
                 ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el, H_in,         &
                 ELDERS( 2, : ), RANGE )
           ELSE
             CALL OTHERS_scaleh( .TRUE., n, nel, lfuval, S%nvrels, S%ntotin,   &
                 inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR, IELVAR,        &
                 ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el, H_in, RANGE )
           END IF
         END IF

!  If a two-norm trust region is to be used, initialize the vector P

         IF ( S%twonrm ) P = zero
         CALL CPU_TIME( tim )
         S%tup = S%tup + tim - S%t

!  Compute the gradient value

         CALL LANCELOT_form_gradients(                                         &
             n, ng, nel, S%ntotel, S%nvrels, S%nnza,                           &
             S%nvargp, .TRUE., ICNA, ISTADA, IELING, ISTADG, ISTAEV,           &
             IELVAR, INTVAR, A, GVALS( : , 2 ), FUVALS( : S%lnguvl ),          &
             S%lnguvl, FUVALS( S%lggfx  + 1 : S%lggfx + n ),                   &
             GSCALE_used, ESCALE, GRJAC, GXEQX_used, INTREP, ISVGRP, ISTAGV,   &
             ITYPEE, ISTAJC, W_ws, W_el, RANGE, KNDOFG )

!  Find the initial projected gradient and its norm

         CALL LANCELOT_projected_gradient(                                     &
             n, X, FUVALS( S%lggfx + 1 : S%lggfx + n ),                        &
             VSCALE, BL, BU, DGRAD, IVAR, inform%nvar, inform%pjgnrm )
         S%nfree = inform%nvar

!  Find the norm of the projected gradient

         IF ( S%prcond .AND. inform%nvar > 0 .AND. S%myprec ) THEN

!  Use the users preconditioner

           inform%status = - 8 ; GO TO 800
         END IF

   90    CONTINUE

!  Find the norm of the 'preconditioned' projected gradient. Also, find the
!  diagonal elements of the assembled Hessian as scalings, if required

         CALL LANCELOT_norm_proj_grad(                                         &
             n , ng, nel, S%ntotel, S%nvrels, S%nvargp,      &
             inform%nvar, S%smallh, inform%pjgnrm, S%calcdi, S%dprcnd,         &
             S%myprec, IVAR(:inform%nvar ), ISTADH, ISTAEV, IELVAR, INTVAR,    &
             IELING, DGRAD( : inform%nvar ), Q, GVALS( : , 2 ), GVALS( : , 3 ),&
             FUVALS( S%ldx + 1 : S%ldx + n ), GSCALE_used, ESCALE,       &
             GRJAC, FUVALS( : S%lnhuvl ), S%lnhuvl, S%qgnorm, GXEQX_used,      &
             INTREP, ISYMMD, ISYMMH, ISTAGV, ISLGRP, ISVGRP, IVALJR, ITYPEE,   &
             W_el, W_in, H_in, RANGE, KNDOFG )

!  If a non-monotone method is to be used, initialize counters

         IF ( S%nmhist > 0 ) THEN
           S%l_suc = 0
           S%f_min = inform%aug ; S%f_r = S%f_min ; S%f_c = S%f_min
           IF ( S%p_type > 2 ) THEN
             S%f_min_viol = S%violation
             S%f_min_lag = S%f_min - S%f_min_viol / inform%mu
             S%f_r_viol = S%f_min_viol ; S%f_r_lag = S%f_min_lag
             S%f_c_viol = S%f_min_viol ; S%f_c_lag = S%f_min_lag
           END IF
           S%sigma_r = zero ; S%sigma_c = zero
         END IF

!  Set initial trust-region radius

         S%print_header = .TRUE.
         inform%radius = control%initial_radius

         IF ( inform%radius > zero ) THEN
           S%oldrad = inform%radius
           IF ( S%strctr ) RADII = inform%radius
         ELSE

!  An unsophisticated method is to be used. Ensure that the initial Cauchy 
!  step is of order unity
       
!          gnorm = SQRT( SUM( FUVALS( S%lggfx + 1 : S%lggfx + n ) ** 2 ) )
           gnorm = zero
           DO i = 1, n ; gnorm = gnorm + FUVALS( S%lggfx + i ) ** 2 ; END DO
           gnorm = SQRT( gnorm )
           inform%radius = MIN( S%maximum_radius, point1 * gnorm )
           S%oldrad = inform%radius
           IF ( S%strctr ) RADII = inform%radius
         END IF 
!write(6,"( ' c = ', ES12.4, ', x = ', /, ( 5ES12.4 ) )" ) inform%cnorm, X

         S%lisend1 = ' ' ; S%cgend1 = ' '

!  ------------------------------------------------
!  Main iteration loop of the algorithm (see paper)
!  ------------------------------------------------

  120    CONTINUE

!  If required, print one line of details of the current iteration

           IF ( S%out > 0 .AND.                                                &
                ( S%print_level == 1 .OR. S%print_level == 2 ) ) THEN

!  If needed, print the iteration header

             IF ( S%print_header .OR. S%print_level == 2 ) THEN
               IF ( S%direct ) THEN
                 WRITE( S%out, "( /, ' Iter g.ev fill-in    obj    ||c||',     &
                &  '   proj.g    rho    radius   step  free  time' )" )
               ELSE
                 WRITE( S%out, "( /, ' Iter g.ev cg.it      obj   ||c|| ',     &
                &  '  proj.g    rho   radius   step   free  time' )" )
               END IF
             END IF

!  Print the iteration details
            
             CALL CPU_TIME( tim )
             atime = OTHERS_time6( tim - S%time )
             citer = OTHERS_iter5( inform%iter )
             cngevl = OTHERS_iter5( inform%ngeval )
             cfree = OTHERS_iter5( S%nfree )
             IF ( S%direct ) THEN
               IF ( S%print_header ) THEN
                 WRITE( S%out, "( 2A5, 7X, ES10.2, 2ES8.1, '     -       -  ', &
                &  '     -   ', A5, A6 )" ) citer, cngevl,                     &
                     inform%aug * S%findmx, inform%cnorm, inform%pjgnrm,       &
                     cfree, atime
               ELSE IF ( inform%status == - 11 ) THEN
                 WRITE( S%out, "( 2A5, F6.1, A1, ES10.2, 2ES8.1, '     -   ',  &
                &  2ES8.1, A5, A6 )" ) citer, cngevl, S%fill, S%lisend1,       &
                     inform%aug * S%findmx, inform%cnorm, inform%pjgnrm,       &
                     S%oldrad, S%step, cfree, atime
               ELSE
                 WRITE( S%out,                                                 &
                   "( 2A5, F6.1, A1, ES10.2, 2ES8.1, ES9.1, 2ES8.1, A5, A6 )" )&
                     citer, cngevl, S%fill, S%lisend1,                         &
                     inform%aug * S%findmx, inform%cnorm, inform%pjgnrm,       &
                     S%rho, S%oldrad, S%step, cfree, atime
               END IF
             ELSE
               citcg = OTHERS_iter5( inform%itercg )
               IF ( S%print_header ) THEN
                 WRITE( S%out, "( 3A5, 1X,  ES10.2, 2ES8.1, '     -       - ', &
                &  '      -   ', A5, A6 )" ) citer, cngevl, citcg,             &
                    inform%aug * S%findmx, inform%cnorm, inform%pjgnrm,        &
                    cfree, atime
               ELSE IF ( inform%status == - 11 ) THEN
                 WRITE( S%out, "( 3A5, A1, ES10.2, 2ES8.1, '     -   ',        &
                &  2ES8.1, A5, A6 )" ) citer, cngevl, citcg, S%cgend1,         &
                     inform%aug * S%findmx, inform%cnorm, inform%pjgnrm,       &
                     S%oldrad, S%step, cfree, atime
               ELSE
                 WRITE( S%out,                                                 &
                   "( 3A5, A1, ES10.2, 2ES8.1, ES9.1, 2ES8.1, A5, A6 )" )      &
                      citer, cngevl, citcg, S%cgend1, inform%aug * S%findmx,   &
                      inform%cnorm, inform%pjgnrm, S%rho, S%oldrad, S%step,    &
                      cfree, atime
               END IF
             END IF
             S%print_header = .FALSE.
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, ' Required gradient accuracy ', ES8.1 )" ) S%epscns
           END IF

!  -----------------------------------
!  Step 1 of the algorithm (see paper)
!  -----------------------------------

!  If required, print more thorough details of the current iteration

           IF ( S%printm ) THEN
             IF ( inform%iter == 0 ) THEN
               WRITE( S%out, 2570 ) inform%iter, inform%aug *  S%findmx,       &
                 inform%ngeval, inform%pjgnrm, inform%itercg, inform%iskip
             ELSE
               WRITE( S%out, 2550 ) inform%iter, inform%aug *  S%findmx,       &
                 inform%ngeval, inform%pjgnrm, inform%itercg, S%oldrad,        &
                 inform%iskip
             END IF
             WRITE( S%out, 2500 ) X
             IF ( S%printw ) THEN
               WRITE( S%out, 2510 )                                            &
                 FUVALS( S%lggfx + 1 : S%lggfx + n ) * S%findmx
               IF ( S%print_level >= 5 ) THEN
                 WRITE( S%out, "( /, ' Element values ', / (  6ES12.4 ) )" )   &
                   FUVALS( : nel )
                 WRITE( S%out, "( /, ' Group values ', / (  6ES12.4 ) )" )     &
                   GVALS( : , 1 )
                 IF ( S%calcdi ) WRITE( S%out,                                 &
                   "( /, ' Diagonals of second derivatives', / ( 6ES12.4 ) )") &
                   FUVALS( S%ldx + 1 : S%ldx + n ) * S%findmx
                 IF ( S%print_level >= 6 ) THEN
                   WRITE( S%out, "( :, /, ' Element gradients ', /             &
                  &  ( 6ES12.4 ) )" ) FUVALS( S%lgxi + 1 : S%lhxi )
                   WRITE( S%out, "( /, ' Group derivatives ', /                &
                  &  ( 6ES12.4 ) )" ) GVALS( : , 2 )
                   WRITE( S%out,                                               &
                     "( :, /, ' Element hessians ', / (  6ES12.4 ) )" )        &
                     FUVALS( S%lhxi + 1 : S%lggfx )
                  END IF
               END IF
             END IF
           END IF

!  Test for convergence

           IF ( inform%pjgnrm <= S%epscns ) THEN
             inform%status = 0
             GO TO 600
           END IF

           IF ( inform%aug < control%min_aug ) THEN
             inform%status = 18
             GO TO 600
           END IF

!  Test whether the maximum allowed number of iterations has been reached

           IF ( inform%iter >= control%maxit ) THEN
             IF ( S%printe ) WRITE( S%out,                                     &
               "( /, ' LANCELOT_solve : maximum number of iterations reached')")

             inform%status = 1 ; GO TO 600
           END IF

!  Test whether the trust-region radius is too small for progress

           IF ( inform%radius < S%radtol ) THEN
             IF ( S%printe ) WRITE( S%out, 2540 )
             inform%status = 2 ; GO TO 600
           END IF
           inform%iter = inform%iter + 1

!  Test whether the CPU time limit has been exceeded

           CALL CPU_TIME( tim )
           IF ( control%cpu_time_limit >= zero .AND.                           &
                REAL( tim - S%time, wp ) > control%cpu_time_limit ) THEN
             IF ( S%printe ) WRITE( S%out, 2520 )
             inform%status = 19 ; GO TO 600
           END IF

!  Check that the print status remains unchanged

           IF ( ( inform%iter >= S%start_print .AND.                           &
                  inform%iter < S%stop_print ) .AND.                           &
                MOD( inform%iter - S%start_print, S%print_gap ) == 0 ) THEN
             S%printe = S%set_printe
             S%printi = S%set_printi
             S%printt = S%set_printt
             S%printm = S%set_printm
             S%printw = S%set_printw
             S%printd = S%set_printd
             S%print_level = control%print_level
           ELSE
             S%printe = .FALSE.
             S%printi = .FALSE.
             S%printt = .FALSE.
             S%printm = .FALSE.
             S%printw = .FALSE.
             S%printd = .FALSE.
             S%print_level = 0
           END IF
           S%save_c = .FALSE.

!  --------------------------------------------------------------
!  Step 2- (find the Cauchy point for the feasibility subproblem)
!  ** Curtis-Jiang-Robinson steering additon **
!  --------------------------------------------------------------

           IF ( S%steering ) THEN
             reduced_mu = .FALSE.

!  Use ISWKSP to indicate which elements are needed for the matrix-vector
!  product B * P. If ISWKSP( I ) = nbprod, the I-th element is used

             S%nbprod = 0
             IF ( .NOT. S%alllin ) ISWKSP( : S%ntotel ) = 0

!  Estimate the norm of the preconditioning matrix by computing its smallest
!  and largest (in magnitude) diagonals

             S%diamin = HUGE( one ) ; S%diamax = zero
             IF ( S%calcdi ) THEN
               DO i = 1, n
                 IF ( INDEX( i ) == 0 ) THEN
                  S%diamin = MIN( S%diamin, FUVALS( S%ldx + i ) )
                  S%diamax = MAX( S%diamax, FUVALS( S%ldx + i ) )
                 END IF
               END DO
             END IF

!  If all the diagonals are small, the norm will be estimated as one

             IF ( S%diamax <= epsmch ) THEN
               S%diamin = one ; S%diamax = one
             END IF

!  Initialize values for the generalized Cauchy point calculation

             S%stepmx = zero ; S%f0 = zero ; S%ibqpst = 1

!  Calculate the radius bounds for the structured trust region

             IF ( S%strctr ) THEN
               BND_radius( : , 1 ) = S%maximum_radius
               DO ig = 1, ng
                 k1 = ISTAGV( ig ) ; k2 = ISTAGV( ig + 1 ) - 1
                 BND_radius( ISVGRP( k1 : k2 ), 1 ) =                          &
                   MIN( RADII( ig ), BND_radius( ISVGRP( k1 : k2 ), 1 ) )
               END DO
             END IF

!  compute J(transpose) c in GX0

!write(6,"('C', /, (6ES12.4))" ) C_best
             CALL LANCELOT_JTc( n, ng, S%nvargp, GX0, C_best, CDASH, GRJAC,    &
                                GSCALE, GXEQX, ISVGRP, ISTAGV, IVALJR, KNDOFG )
!write(6,"('GX0', /, (6ES12.4))" ) GX0

!DIR$ IVDEP
             DO i = 1, n

!  Set the bounds on the variables for the model problem. If a two-norm
!  trust region is to be used, the bounds are just the box constraints

               IF ( S%twonrm ) THEN
                 BND( i, 1 ) = BL( i ) ; BND( i, 2 ) = BU( i )

!  If an infinity-norm trust region is to be used, the bounds are the
!  intersection of the trust region with the box constraints

               ELSE
                 IF ( S%strctr ) THEN
                   S%rad = BND_radius( i, 1 )
                 ELSE
                   S%rad = inform%radius
                 END IF
                 IF ( S%calcdi ) THEN
                   distan = S%rad / SQRT( FUVALS( S%ldx + i ) )
                 ELSE
                   distan = S%rad * VSCALE( i )
                 END IF
                 BND( i, 1 ) = MAX( X( i ) - distan, BL( i ) )
                 BND( i, 2 ) = MIN( X( i ) + distan, BU( i ) )
                 IF ( S%mortor ) THEN
                   BND_radius( i, 1 ) = X( i ) - distan
                   BND_radius( i, 2 ) = X( i ) + distan
                 END IF
               END IF

!  Compute the steering Cauchy direction, DGRAD, as - J(transpose) c
!  direction. Normalize the diagonal scalings if necessary

               X0( i ) = X( i )
               DELTAX( i ) = zero
               P( i ) = zero
               IF ( S%reusec ) CYCLE
               DGRAD( i ) = - GX0( i )

!  If an approximation to the Cauchy point is to be used, calculate a
!  suitable initial estimate of the line minimum, stepmx

               IF ( S%xactcp ) THEN
                 S%stepmx = HUGE( one )
               ELSE  
                 IF ( DGRAD( i ) /= zero ) THEN
                   IF ( DGRAD( i ) > zero ) THEN
                     S%stepmx = MAX( S%stepmx,                                 &
                                     ( BU( i ) - X( i ) ) / DGRAD( i ) )
                   ELSE
                     S%stepmx = MAX( S%stepmx,                                 &
                                     ( BL( i ) - X( i ) ) / DGRAD( i ) )
                   END IF
                 END IF
               END IF

!  Release any artificially fixed variables from their bounds

               IF ( INDEX( i ) == 4 ) INDEX( i ) = 0
             END DO

!  ** temporraily normalize 

!         val =  SQRT( DOT_PRODUCT( DGRAD, DGRAD ) )
!         DGRAD = DGRAD / val
!         IF ( .NOT. S%twonrm .AND. .NOT. S%xactcp ) S%stepmx = S%stepmx * val
           
!  The value of the integer S%ifactr controls whether a new factorization
!  of the Hessian of the model is obtained (S%ifactr = 1) or whether a
!  Schur-complement update to an existing factorization is required
!  (S%ifactr = 2) when forming the preconditioner

             S%ifactr = 1 ; S%refact = .FALSE.

!  Evaluate the generalized Cauchy point, XT

             S%jumpto = 1
             S%rad = inform%radius
!write(6,*) ' rad ', S%rad
!write(6,"('GSCALE', /, (6ES12.4))" ) GSCALE
!write(6,"('initial g', /, (6ES12.4))" ) DGRAD

             IF ( S%printt ) WRITE( S%out,                                   &
               "( /, '    Find Cauchy point for the feasibility' )" )

  140        CONTINUE
             CALL CPU_TIME( S%t )

!  The exact generalized Cauchy point is required

             IF ( S%xactcp ) THEN
               CALL CAUCHY_get_exact_gcp(                                      &
                   n, X0, XT, GX0, BND, INDEX, S%f0, S%stepmx, S%epstlp,       &
                   S%twonrm, S%dxsqr, S%rad, S%fmodel, DGRAD, Q, IVAR,         &
                   inform%nvar, nvar1, S%nvar2, S%nnonnz, INNONZ, S%out,       &
                   S%jumpto, S%print_level, S%findmx, BREAKP, S%CAUCHY )

!  An approximation to the Cauchy point suffices

             ELSE
               CALL CAUCHY_get_approx_gcp(                                     &
                   n, X0, XT, GX0, BND, INDEX, S%f0, S%epstlp, S%stepmx,       &
                   point1, S%twonrm, S%rad, S%fmodel, DGRAD, Q, IVAR,          &
                   inform%nvar, nvar1, S%nvar2, S%out, S%jumpto, S%print_level,&
                   S%findmx, BREAKP, GRAD, S%CAUCHY )
             END IF
             CALL CPU_TIME( tim )
             S%tca = S%tca + tim - S%t

!  Scatter the nonzeros in DGRAD onto P

             P( IVAR( nvar1 : S%nvar2 ) ) = DGRAD( IVAR( nvar1 : S%nvar2 ) )
!write(6,"('p', /, (6ES12.4))" ) P

!  A further matrix-vector product is required

             IF ( S%jumpto > 0 ) THEN
               CALL CPU_TIME( S%t )
                S%nbprod = S%nbprod + 1

!  Calculate the product of J^T J with the vector P (set alllin = .TRUE. )

               S%densep = S%jumpto == 2 .OR. ( S%xactcp .AND. S%jumpto == 4 )

!write(6,*) ' densep ', S%densep

               CALL LANCELOT_JTJ_times_vector(                                 &
                      n, ng, S%nvargp, inform%nvar, nvar1, S%nvar2, S%nnonnz,  &
                      IVAR, INNONZ( : n ), P, Q , CDASH, GRJAC, GSCALE,        &
                      GXEQX, S%densep, IGCOLJ, ISVGRP, ISTAGV, IVALJR,         &
                      ISTAJC, IUSED, NZ_comp_w, W_ws, KNDOFG )

               IF ( S%printd .AND. S%jumpto == 3 ) WRITE( S%out,               &
                 "( ' Nonzeros of J^T J * P are in positions', /, ( 24I3 ))" ) &
                   INNONZ( : S%nnonnz )
!write(6,"('JTJp', /, (6ES12.4))" ) Q
               CALL CPU_TIME( tim )
               S%tmv = S%tmv + tim - S%t

!  Reset the components of P that have changed to zero

               P( IVAR( nvar1 : S%nvar2 ) ) = zero

!  If required, print a list of the nonzeros of P

               IF ( S%jumpto == 3 .AND. S%printd .AND. .NOT. S%alllin )        &
                 WRITE( S%out, 2560 ) S%nbprod, ISWKSP( : S%ntotel )

!  Continue the Cauchy point calculation

               GO TO 140
             END IF

!  Check to see if there are any remaining free variables

             IF ( nvar1 > S%nvar2 ) THEN
               IF ( S%printt ) WRITE( S%out,                                   &
                 "( /, '    No free variables - search direction complete ' )" )
               GO TO 400
             ELSE 
               IF ( S%printt ) WRITE( S%out,                                   &
                 "( /, '    There are now ', I7, ' free variables ' )" )       &
                   S%nvar2 - nvar1 + 1
             END IF
  
!  Store the value of the Cauchy point for future use

             S%delta_qv_steering = - S%fmodel
             P = XT - X0
!write(6,"('DX steering', /, (6ES12.4))" ) P

!  If required, print the active set at the generalized Cauchy point

             IF ( S%printw ) THEN
               WRITE( S%out, "( / )" )
               DO i = 1, n
                 IF ( INDEX( i ) == 2 .AND.  XT( i ) >=                        &
                   BU( i ) - ABS( BU( i ) ) * S%epstln ) WRITE( S%out,         &
                 "( ' The variable number ', I3, ' is at its upper bound' )" ) i
                 IF ( INDEX( i ) == 1 .AND. XT( i ) <=                         &
                   BL( i ) + ABS( BL( i ) ) * S%epstlp ) WRITE( S%out,         &
                 "( ' The variable number ', I3, ' is at its lower bound' )" ) i
                 IF ( INDEX( i ) == 4 ) WRITE( S%out,                          &
                 "( ' The variable number ', I3, ' is temporarily fixed ' )" ) i
               END DO
             END IF
             S%n_steering_this_iteration = 0
           END IF

!  -----------------------------------
!  Step 2 of the algorithm (see paper)
!  -----------------------------------

  200      CONTINUE

!  Use ISWKSP to indicate which elements are needed for the matrix-vector
!  product B * P. If ISWKSP( I ) = nbprod, the I-th element is used

           S%nbprod = 0
           IF ( .NOT. S%alllin ) ISWKSP( : S%ntotel ) = 0

!  Estimate the norm of the preconditioning matrix by computing its smallest
!  and largest (in magnitude) diagonals

           S%diamin = HUGE( one ) ; S%diamax = zero
           IF ( S%calcdi ) THEN
             DO i = 1, n
               IF ( INDEX( i ) == 0 ) THEN
                S%diamin = MIN( S%diamin, FUVALS( S%ldx + i ) )
                S%diamax = MAX( S%diamax, FUVALS( S%ldx + i ) )
               END IF
             END DO
           END IF

!  If all the diagonals are small, the norm will be estimated as one

           IF ( S%diamax <= epsmch ) THEN
             S%diamin = one ; S%diamax = one
           END IF

!  Initialize values for the generalized Cauchy point calculation

           S%stepmx = zero ; S%f0 = inform%aug ; S%ibqpst = 1

!  Calculate the radius bounds for the structured trust region

           IF ( S%strctr ) THEN
             BND_radius( : , 1 ) = S%maximum_radius
             DO ig = 1, ng
               k1 = ISTAGV( ig ) ; k2 = ISTAGV( ig + 1 ) - 1
               BND_radius( ISVGRP( k1 : k2 ), 1 ) =                            &
                 MIN( RADII( ig ), BND_radius( ISVGRP( k1 : k2 ), 1 ) )
             END DO
           END IF

!DIR$ IVDEP
           DO i = 1, n

!  Set the bounds on the variables for the model problem. If a two-norm
!  trust region is to be used, the bounds are just the box constraints

             IF ( S%twonrm ) THEN
               BND( i, 1 ) = BL( i ) ; BND( i, 2 ) = BU( i )

!  If an infinity-norm trust region is to be used, the bounds are the
!  intersection of the trust region with the box constraints

             ELSE
               IF ( S%strctr ) THEN
                 S%rad = BND_radius( i, 1 )
               ELSE
                 S%rad = inform%radius
               END IF
               IF ( S%calcdi ) THEN
                 distan = S%rad / SQRT( FUVALS( S%ldx + i ) )
               ELSE
                 distan = S%rad * VSCALE( i )
               END IF
               BND( i, 1 ) = MAX( X( i ) - distan, BL( i ) )
               BND( i, 2 ) = MIN( X( i ) + distan, BU( i ) )
               IF ( S%mortor ) THEN
                 BND_radius( i, 1 ) = X( i ) - distan
                 BND_radius( i, 2 ) = X( i ) + distan
               END IF
             END IF

!  Compute the Cauchy direction, DGRAD, as a scaled steepest-descent
!  direction. Normalize the diagonal scalings if necessary

             X0( i ) = X( i )

             DELTAX( i ) = zero
             GX0( i ) = FUVALS( S%lggfx + i )
             P( i ) = zero
             IF ( S%reusec ) CYCLE
             IF ( S%calcdi ) THEN
               j = S%ldx + i
               DGRAD( i ) = - FUVALS( S%lggfx + i ) / FUVALS( j )
               FUVALS( j ) = FUVALS( j ) / S%diamax
             ELSE
               DGRAD( i ) = - FUVALS( S%lggfx + i ) *                          &
                            ( VSCALE( i ) / S%vscmax ) ** 2
             END IF

!  If an approximation to the Cauchy point is to be used, calculate a
!  suitable initial estimate of the line minimum, stepmx

             IF ( S%xactcp ) THEN
               S%stepmx = HUGE( one )
             ELSE  
               IF ( DGRAD( i ) /= zero ) THEN
                 IF ( DGRAD( i ) > zero ) THEN
                   S%stepmx = MAX( S%stepmx, ( BU( i ) - X( i ) ) / DGRAD( i ) )
                 ELSE
                   S%stepmx = MAX( S%stepmx, ( BL( i ) - X( i ) ) / DGRAD( i ) )
                 END IF
               END IF
             END IF

!  Release any artificially fixed variables from their bounds

             IF ( INDEX( i ) == 4 ) INDEX( i ) = 0
           END DO

!  ** temporraily normalize 

!          val =  SQRT( DOT_PRODUCT( DGRAD, DGRAD ) )
!          DGRAD = DGRAD / val
!          IF ( .NOT. S%twonrm .AND. .NOT. S%xactcp ) S%stepmx = S%stepmx * val
           
!  The value of the integer S%ifactr controls whether a new factorization
!  of the Hessian of the model is obtained (S%ifactr = 1) or whether a
!  Schur-complement update to an existing factorization is required
!  (S%ifactr = 2) when forming the preconditioner

           S%ifactr = 1 ; S%refact = .FALSE.

!  If a previously calculated generalized Cauchy point still lies
!  within the trust-region bounds, it will be reused

           IF ( S%reusec ) THEN

!  Retrieve the Cauchy point

             XT( : n ) = XCP( : n )

!  Retrieve the set of free variables

             inform%nvar = S%nfreec
             IVAR( : inform%nvar ) = IFREEC( : inform%nvar )
             INDEX( IVAR( : inform%nvar ) ) = 0

!  Skip the remainder of step 2

             S%reusec = .FALSE.
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, ' Reusing previous generalized Cauchy point ' )" )
             GO TO 290
           END IF

!  Evaluate the generalized Cauchy point, XT

           S%jumpto = 1
           S%firstc = .TRUE.
           S%mortor_its = 0
           S%rad = inform%radius
!write(6,*) ' rad ', S%rad
!IF ( S%steering ) write(6,"('initial g', /, (6ES12.4))" ) DGRAD

           IF ( S%printt ) WRITE( S%out,                                       &
             "( /, '    Find Cauchy point for the objective' )" )

  240      CONTINUE
!write(6,*) ' nvar ', inform%nvar
           CALL CPU_TIME( S%t )

!  The exact generalized Cauchy point is required

           IF ( S%xactcp ) THEN
             CALL CAUCHY_get_exact_gcp(                                        &
                 n, X0, XT, GX0, BND, INDEX, S%f0, S%stepmx, S%epstlp,         &
                 S%twonrm, S%dxsqr, S%rad, S%fmodel, DGRAD, Q, IVAR,           &
                 inform%nvar, nvar1, S%nvar2, S%nnonnz, INNONZ, S%out,         &
                 S%jumpto, S%print_level, S%findmx, BREAKP, S%CAUCHY )

!  An approximation to the Cauchy point suffices

           ELSE
             CALL CAUCHY_get_approx_gcp(                                       &
                 n, X0, XT, GX0, BND, INDEX, S%f0, S%epstlp, S%stepmx,         &
                 point1, S%twonrm, S%rad, S%fmodel, DGRAD, Q, IVAR,            &
                 inform%nvar, nvar1, S%nvar2, S%out, S%jumpto, S%print_level,  &
                 S%findmx, BREAKP, GRAD, S%CAUCHY )
           END IF
           CALL CPU_TIME( tim )
           S%tca = S%tca + tim - S%t

!  Scatter the nonzeros in DGRAD onto P

           P( IVAR( nvar1 : S%nvar2 ) ) = DGRAD( IVAR( nvar1 : S%nvar2 ) )
!write(6,"('p ', /, (6ES12.4))" ) P
!  A further matrix-vector product is required

           IF ( S%jumpto > 0 ) THEN
             CALL CPU_TIME( S%t )
             S%nbprod = S%nbprod + 1

!  Calculate the product of the Hessian with the vector P

             S%densep = S%jumpto == 2 .OR. ( S%xactcp .AND. S%jumpto == 4 )

!write(6,*) ' densep ', S%densep

             IF (  S%steering ) THEN
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR , ISTAEV, ISTADH, INTVAR, IELING,  &
                   IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                &
                   P , Q , GVALS2_GN, GVALS( : , 3 ),                          &
                   GRJAC, GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl, &
                   GXEQX_used , INTREP, S%densep,                              &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             ELSE
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR , ISTAEV, ISTADH, INTVAR, IELING,  &
                   IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                &
                   P , Q , GVALS( : , 2 ), GVALS( : , 3 ),                     &
                   GRJAC, GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl, &
                   GXEQX_used , INTREP, S%densep,                              &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             END IF

             IF ( S%printd .AND. S%jumpto == 3 ) WRITE( S%out,                 &
               "( ' Nonzeros of Hessian * P are in positions', /, ( 24I3 ))" ) &
                 INNONZ( : S%nnonnz )
!write(6,"('Hp*mu', /, (6ES12.4))" ) Q * ( inform%mu ) ** 1
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t

!  Reset the components of P that have changed to zero

             P( IVAR( nvar1 : S%nvar2 ) ) = zero

!  If required, print a list of the nonzeros of P

             IF ( S%jumpto == 3 .AND. S%printd .AND. .NOT. S%alllin )          &
               WRITE( S%out, 2560 ) S%nbprod, ISWKSP( : S%ntotel )

!  Continue the Cauchy point calculation

             GO TO 240
           END IF

!  ---------------------------------------------------
!  Step 2+ (check that find this Cauchy point performs 
!   well relative to that the feasibility subproblem)
!  ** Curtis-Jiang-Robinson steering additon **
!  ---------------------------------------------------

           IF ( S%steering ) THEN
!write(6,"('KNDOFG', /, (6I12))" ) KNDOFG
!write(6,"('GSCALE', /, (6ES12.4))" ) GSCALE
!write(6,"('C_best', /, (6ES12.4))" ) C_best
!            P = zero
!            val = LANCELOT_violation( n, ng, S%nvargp, C_best, CDASH, GRJAC,  &
!                    P, GSCALE, GXEQX, IGCOLJ, ISTAJC, KNDOFG, W_ws, gi )
! write(6,"(' violation_0 violation_calc')" )
! write(6,"(A, 2ES12.4)") '+++++++++++++++++++++ ', S%violation, val
             P = XT - X0
!write(6,"('C', /, (6ES12.4))" ) C_best
!write(6,"('DX', /, (6ES12.4))" ) P
             val = LANCELOT_violation( n, ng, S%nvargp, C_best, CDASH, GRJAC,  &
                     P, GSCALE, GXEQX, IGCOLJ, ISTAJC, KNDOFG, W_ws, gi )
             S%delta_qv = S%violation - val

!write(6,*) ' viol_0 ', gi
!if(inform%mu<ten**(-10))stop

!  if the violation at the Cauchy point is significantly larger than that
!  at the corresponding Cauchy point for feasibility, reduce the penalty 
!  parameter

! write(6,"(' violation_0 violation_1 d_violation d_vio_steer  3rd term ')" )
! write(6,"(5ES12.4)") S%violation, val, S%delta_qv, &
!                     control%kappa_3 * S%delta_qv_steering,     &
!                     S%violation - half * ( control%kappa_t * S%etak ) ** 2 

!write(6,*) ' dqv, dqv_steering', S%delta_qv, S%delta_qv_steering
             IF ( ( S%delta_qv < MIN( control%kappa_3 * S%delta_qv_steering,   &
                    S%violation - half * ( control%kappa_t * S%etak ) ** 2 ) ) &
                   .AND. S%n_steering <= control%num_mudec                     &
                   .AND. S%n_steering_this_iteration <=                        &
                           control%num_mudec_per_iteration                     &
                   .AND. inform%mu >= control%mu_min ) THEN
               S%n_steering = S%n_steering + 1
               S%n_steering_this_iteration = S%n_steering_this_iteration + 1
               reduced_mu = .TRUE.

!  reduce the penalty parameter and adjust the convergence tolerences

               inform%mu = S%tau_steering * inform%mu
!if ( inform%mu < ten ** ( - 9 ) ) stop
               S%alphak = MIN( inform%mu, S%gamma1 )
               S%etak   = MAX( S%eta_min, S%eta0 * S%alphak ** S%alphae )
               S%omegak = MAX( S%omega_min, S%omega0 * S%alphak ** S%alphao )

               IF ( S%printt ) WRITE( S%out,                                   &
                 "( /, ' ** Reducing penalty parameter to', ES12.4 )") inform%mu

!  recompute the terms involving the constraints for the augmented Lagrangian
!  function

               hmuinv = half / inform%mu ; inform%aug = zero
               DO ig = 1, ng

!  constraint terms psi = 1/2mu ( g(e) + mu y )^2, where c = s g(e),
!  g is the group function and s is the group scaling

                 IF ( KNDOFG( ig ) > 1 ) THEN
                   scaleg = GSCALE( ig )
                   yiui = scaleg * C_best( ig ) + inform%mu * Y( ig )
!WRITE(6,"(I2, 4es12.4)") IG, GSCALE( ig ), C_best( ig ), inform%mu, Y( ig )
                   GVALS( ig, 1 ) = ( hmuinv * yiui ) * yiui

!  derivatives of constraint terms with trivial groups (g(e) = e):
!  psi'(e) = 1/mu (s g(e) + mu y) s and
!  psi''(e) = 1/mu s^2

                   IF ( GXEQX( ig ) ) THEN
                     GVALS( ig, 2 ) = scaleg * ( C_best( ig ) *                &
                                    ( scaleg / inform%mu ) + Y( ig ) )
                     GVALS( ig, 3 ) = scaleg * ( scaleg / inform%mu )

!  if a Gauss Newton model is required, psi'(e) = y s

                     IF ( control%gn_model ) THEN
                       GVALS2_GN( ig ) = scaleg * Y( ig )
                     ELSE
                       GVALS2_GN( ig ) = GVALS( ig, 2 )
                     END IF

!  derivatives of constraint terms with non-trivial groups:
!  psi'(e) = 1/mu (s g(e) + mu y) s g'(e) and
!  psi''(e) = 1/mu (s g(e) + mu y) s g''(e) + 1/mu (sg'(e))^2

                   ELSE
                     hdash = scaleg * ( Y( ig ) + C_best( ig ) *               &
                                      ( scaleg / inform%mu ) )
                     GVALS( ig, 2 ) = hdash * CDASH( ig )
                     GVALS( ig, 3 ) = hdash * C2DASH( ig ) +                   &
                                ( scaleg * CDASH( ig ) ) ** 2 / inform%mu

!  if a Gauss Newton model is required, psi'(e) = y s g'(e)

                     IF ( control%gn_model ) THEN
                       GVALS2_GN( ig ) = scaleg * Y( ig ) * CDASH( ig )
                     ELSE
                       GVALS2_GN( ig ) = GVALS( ig, 2 )
                     END IF
                   END IF

!  objective terms psi = g(e), where c = s g(e)

                 ELSE
                   GVALS( ig, 1 ) = C_best( ig )
                 END IF
                 inform%aug = inform%aug + GSCALE_used( ig ) * GVALS( ig, 1 )
!write(6,"(I2, 2ES12.4)") ig, GSCALE_used( ig ), GVALS( ig, 1 )
               END DO
!write(6,*) 'ng ', ng
!write(6,*) 'gscale_used ', GSCALE_used
!write(6,*) 'gscale      ',  GSCALE

!  If a non-monotone method is to be used, reset reference values (heuristic)

               IF ( S%nmhist > 0 ) THEN
!                IF ( .FALSE. ) THEN
                 IF ( .TRUE. ) THEN
                   IF ( S%new_major ) THEN
                     S%f_min = inform%aug ; S%f_r = S%f_min ; S%f_c = S%f_min
                   ELSE
                     S%f_min = S%f_min_lag + S%f_min_viol / inform%mu
                     S%f_r = S%f_r_lag + S%f_r_viol / inform%mu
                     S%f_c = S%f_c_lag + S%f_c_viol / inform%mu
                     S%sigma_r = zero
                     S%sigma_c = S%sigma_r 
                   END IF

!  If a non-monotone method is to be used, initialize counters

                 ELSE
                   S%l_suc = 0
                   S%f_min = inform%aug ; S%f_r = S%f_min ; S%f_c = S%f_min
                   IF ( S%p_type > 2 ) THEN
                     S%f_min_viol = S%violation
                     S%f_min_lag = S%f_min - S%f_min_viol / inform%mu
                     S%f_r_viol = S%f_min_viol ; S%f_r_lag = S%f_min_lag
                     S%f_c_viol = S%f_min_viol ; S%f_c_lag = S%f_min_lag
                   END IF
                   S%sigma_r = zero ; S%sigma_c = zero
                 END IF
               END IF

!  Compute the gradient value

               CALL LANCELOT_form_gradients(                                   &
                   n, ng, nel, S%ntotel, S%nvrels, S%nnza,                     &
                   S%nvargp, .FALSE., ICNA, ISTADA, IELING, ISTADG, ISTAEV,    &
                   IELVAR, INTVAR, A, GVALS( : , 2 ), FUVALS( : S%lnguvl ),    &
                   S%lnguvl, FUVALS( S%lggfx  + 1 : S%lggfx + n ),             &
                   GSCALE_used, ESCALE, GRJAC, GXEQX_used, INTREP,             &
                   ISVGRP, ISTAGV, ITYPEE, ISTAJC, W_ws, W_el, RANGE, KNDOFG )

!  find a new Generalized Cauchy point

               GO TO 200
             END IF

!  If needed, print the new penalty parameter and iteration header

             IF ( reduced_mu ) THEN
               IF ( S%printi ) WRITE( S%out, 2200 )
               IF ( S%printi ) WRITE( S%out, 2000 ) inform%mu, S%omegak, S%etak

               IF ( S%out > 0 .AND.                                            &
                    ( S%print_level == 1 .OR. S%print_level == 2 ) ) THEN
                 IF ( S%direct ) THEN
                   WRITE( S%out, "( /, ' Iter g.ev fill-in    obj    ||c||',   &
                  &  '   proj.g    rho    radius   step  free  time' )" )
                 ELSE
                   WRITE( S%out, "( /, ' Iter g.ev cg.it      obj   ||c|| ',   &
                  &  '  proj.g    rho   radius   step   free  time' )" )
                 END IF
               END IF
             END IF

!  if a Newton model is required after steering, prepare to recompute the 
!  Cauchy point

             IF ( .NOT. control%gn_model_after_cauchy ) THEN
               S%steering = .FALSE.
               IF ( S%printt ) WRITE( S%out,                                   &
             "( /, '    Turn off steering until next successful iteration' )" )

!  reset the Hessian weight psi'(e)

               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 1 ) GVALS2_GN( ig ) = GVALS( ig, 2 )
               END DO
               GO TO 200
             END IF
           END IF

!  Check to see if there are any remaining free variables

           IF ( nvar1 > S%nvar2 ) THEN
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, '    No free variables - search direction complete ' )" )
             GO TO 400
           ELSE 
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, '    There are now ', I7, ' free variables ' )" )         &
                 S%nvar2 - nvar1 + 1
           END IF
  
!          IF ( S%mortor_its >= 1 ) GO TO 400
           IF ( S%msweep > 0 .AND. S%mortor_its > S%msweep ) GO TO 400

!  Store the Cauchy point and its gradient for future use

           XCP( : n ) = XT( : n )
           S%fcp = S%fmodel

!  Store the set of free variables at Cauchy point for future use

           S%nfreec = inform%nvar
           IFREEC( : S%nfreec ) = IVAR( : S%nfreec )

!          IF ( S%mortor ) THEN
!            WHERE( INDEX == 1 .OR. INDEX == 2 )
!              INDEX = 4 ; BND( : , 1 ) = XT ; BND( : , 2 ) = XT
!            END WHERE
!          END IF

!  See if an accurate approximation to the minimum of the quadratic
!  model is to be sought

           IF ( S%slvbqp ) THEN

!  Fix the variables which the Cauchy point predicts are active at the solution

             IF ( S%firstc ) THEN
               S%firstc = .FALSE.
               WHERE( INDEX == 1 .OR. INDEX == 2 )
                 INDEX = 4 ; BND( : , 1 ) = XT ; BND( : , 2 ) = XT
               END WHERE

!  Update the step taken and the set of variables which are considered free

             ELSE
               inform%nvar = 0
               DO i = 1, n
                 P( i ) = P( i ) + DELTAX( i )
                 IF ( P( i ) /= zero .OR. INDEX( i ) == 0 ) THEN
                   inform%nvar = inform%nvar + 1
                   IVAR( inform%nvar ) = i
                 END IF
               END DO
               S%nvar2 = inform%nvar
             END IF
           END IF
  
           IF ( S%mortor ) P = XT - X0

!  If required, print the active set at the generalized Cauchy point

  290      CONTINUE
           IF ( S%printw ) THEN
             WRITE( S%out, "( / )" )
             DO i = 1, n
               IF ( INDEX( i ) == 2 .AND.  XT( i ) >=                          &
                 BU( i ) - ABS( BU( i ) ) * S%epstln ) WRITE( S%out,           &
                 "( ' The variable number ', I3, ' is at its upper bound' )" ) i
               IF ( INDEX( i ) == 1 .AND. XT( i ) <=                           &
                 BL( i ) + ABS( BL( i ) ) * S%epstlp ) WRITE( S%out,           &
                 "( ' The variable number ', I3, ' is at its lower bound' )" ) i
               IF ( INDEX( i ) == 4 ) WRITE( S%out,                            &
                 "( ' The variable number ', I3, ' is temporarily fixed ' )" ) i
             END DO
           END IF

!  -----------------------------------
!  Step 3 of the algorithm (see paper)
!  -----------------------------------

           S%jumpto = 1

!  If an iterative method is to be used, set up convergence tolerances

           S%cgstop = MAX( S%resmin, MIN( control%acccg, S%qgnorm )            &
                           * S%qgnorm * S%qgnorm ) * S%diamin / S%diamax
!          IF ( S%twonrm .AND. .NOT. S%direct ) S%dxsqr = DOT_PRODUCT( P, P )
           IF ( S%twonrm .AND. .NOT. S%direct ) THEN
             S%dxsqr = zero
             DO i = 1, n ; S%dxsqr = S%dxsqr + P( i ) ** 2 ; END DO
           END IF
           IF ( S%printw .AND. S%twonrm .AND. .NOT. S%direct )                 &
             WRITE( S%out,                                                     &
               "( /, ' Two-norm of step to Cauchy point = ', ES12.4 )" )       &
                 SQRT( S%dxsqr )
           S%step = inform%radius

!  If an incomplete factorization preconditioner is to be used, decide
!  on the semi-bandwidth, nsemib, of the preconditioner. For the expanding
!  band method, the allowable bandwidth increases as the solution is approached

           IF ( S%iprcnd ) THEN
             inform%nsemib = n / 5
             IF ( inform%pjgnrm <= point1 ) inform%nsemib = n / 2
             IF ( inform%pjgnrm <= tenm2 ) inform%nsemib = n
           ELSE
             IF ( S%use_band ) THEN
               inform%nsemib = control%semibandwidth
             ELSE
               inform%nsemib = n
             END IF
           END IF

!  If Munksgaards preconditioner is to be used, set the stability factor
!  required by MA61 to be more stringent as the solution is approached

           IF ( S%munks ) THEN
             inform%ciccg = point1
             IF ( inform%pjgnrm <= point1 ) inform%ciccg = tenm2
             IF ( inform%pjgnrm <= tenm2 ) inform%ciccg = zero
           ELSE
             inform%ciccg = zero
           END IF

!  Set a limit on the number of CG iterations that are to be allowed

           inform%itcgmx = n
           IF ( S%iprcnd .OR. S%use_band )                                     &
             inform%itcgmx = MAX( 5, n / ( inform%nsemib + 1 ) )
           IF ( S%seprec .OR. S%gmpspr ) inform%itcgmx = 5
           S%nobnds = S%mortor .AND. S%twonrm

!  If required, compare the recurred and calculated model values

           IF ( S%printw ) WRITE( S%out,                                       &
             "( ' *** Calculated quadratic at X0 ', ES22.14 )" )               &
                 inform%aug * S%findmx

!  Calculate an approximate minimizer of the model within the specified bounds

  300      CONTINUE
           IF ( S%jumpto == 4 ) GO TO 320

!  The product of the Hessian with the vector P is required

           IF ( S%jumpto /= 2 ) THEN
             S%nbprod = S%nbprod + 1
             nvar1 = 1
             CALL CPU_TIME( S%t )

!  Set the required components of Q to zero

             IF ( S%jumpto == 1 ) THEN
               Q = zero
             ELSE
               Q( IVAR( : S%nvar2 ) ) = zero
             END IF

!  Compute the matrix-vector product with the dense vector P

             S%densep = .TRUE.
             IF ( S%steering ) THEN
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR,         &
                   IELING, IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),        &
                   P , Q , GVALS2_GN, GVALS( : , 3 ),                          &
                   GRJAC, GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl, &
                   GXEQX_used , INTREP, S%densep,                              &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             ELSE
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR,         &
                   IELING, IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),        &
                   P , Q , GVALS( : , 2 )  , GVALS( : , 3 ),                   &
                   GRJAC, GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl, &
                   GXEQX_used , INTREP, S%densep,                              &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             END IF
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t
!            IF ( S%jumpto == 1 ) THEN
!              WRITE(6,"( 'XT-X0 ', 5ES12.4 )" ) XT - X0
!              WRITE(6,"( 'P   ', 5ES12.4 )" ) P
!              WRITE(6,"( 'Q   ', 5ES12.4 )" ) Q
!            END IF

!  If required, print a list of the nonzeros of P

             IF ( S%printd .AND. .NOT. S%alllin )                              &
               WRITE( S%out, 2560 ) S%nbprod, ISWKSP( : S%ntotel )

!  If required, print the step taken

             IF ( S%jumpto == 1 ) THEN
               IF ( S%out > 0 .AND. S%print_level >= 20 )                      &
                 WRITE( S%out, 2530 ) P( : n )

!  Compute the value of the model at the generalized Cauchy point and then
!  reset P to zero

               S%fnew = S%fmodel
!              S%fmodel = inform%aug
!!DIR$ IVDEP  
!              DO j = 1, S%nvar2
!                i = IVAR( j )
!                S%fmodel = S%fmodel + ( FUVALS( S%lggfx + i ) +               &
!                                      half * Q( i ) ) * P( i )
!                P( i ) = zero
!              END DO
               S%fmodel = S%f0
!DIR$ IVDEP  
               DO j = 1, S%nvar2
                 i = IVAR( j )
                 S%fmodel = S%fmodel + ( GX0( i ) + half * Q( i ) ) * P( i )
                 P( i ) = zero
               END DO

!  If required, compare the recurred and calculated model values

               IF ( S%printw ) WRITE( S%out,                                   &
                 "( ' *** Calculated quadratic at CP ', ES22.14, /,            &
              &     ' *** Recurred   quadratic at CP ', ES22.14 )" )           &
                 S%fmodel * S%findmx, S%fnew * S%findmx
             END IF

!  Evaluate the 'preconditioned' gradient. If the user has supplied a
!  preconditioner, return to the calling program

           ELSE
             IF ( S%myprec ) THEN
               inform%status = - 9 ; GO TO 800
             ELSE
               IF ( S%iprcnd .OR. S%munks .OR. S%icfs .OR. S%seprec .OR.       &
                    S%gmpspr .OR. S%use_band ) THEN

!  If required, use a preconditioner

                 CALL CPU_TIME( S%t )
                 IF ( control%gn_model_after_cauchy ) THEN
                   CALL PRECN_use_preconditioner(                              &
                       S%ifactr, S%munks, S%use_band, S%seprec, S%icfs,        &
                       n, ng, nel, S%ntotel, S%nnza, S%maxsel,                 &
                       S%nadd, S%nvargp, S%nfreef, S%nfixed,                   &
                       control%io_buffer, S%refact, S%nvar2,                   &
                       IVAR, ISTADH, ICNA, ISTADA, INTVAR, IELVAR, S%nvrels,   &
                       IELING, ISTADG, ISTAEV, IFREE,  A, FUVALS,              &
                       S%lnguvl, FUVALS, S%lnhuvl, GVALS2_GN,                  &
                       GVALS( : , 3 ), DGRAD , Q, GSCALE_used, ESCALE,         &
                       GXEQX_used , INTREP, RANGE , S%icfact, inform%ciccg,    &
                       inform%nsemib, inform%ratio, S%print_level,             &
                       S%error, S%out, S%infor, alloc_status, bad_alloc,       &
                       ITYPEE, DIAG, OFFDIA, IW, IKEEP, IW1, IVUSE,            &
                       H_col_ptr, L_col_ptr, W, W1, RHS, RHS2, P2,             &
                       G, ISTAGV, ISVGRP,                                      &
                       lirnh, ljcnh, lh, lirnh_min, ljcnh_min, lh_min,         &
                       ROW_start, POS_in_H, USED, FILLED,                      &
                       lrowst, lpos, lused, lfilled,                           &
                       IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                 &
                       matrix, SILS_data, control%SILS_cntl,                   &
                       inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,&
                       S%PRECN, SCU_matrix, SCU_data, inform%SCU_info,         &
                       S%ASMBL, S%skipg, KNDOFG )
                 ELSE
                   CALL PRECN_use_preconditioner(                              &
                       S%ifactr, S%munks, S%use_band, S%seprec, S%icfs,        &
                       n, ng, nel, S%ntotel, S%nnza, S%maxsel,                 &
                       S%nadd, S%nvargp, S%nfreef, S%nfixed,                   &
                       control%io_buffer, S%refact, S%nvar2,                   &
                       IVAR, ISTADH, ICNA, ISTADA, INTVAR, IELVAR, S%nvrels,   &
                       IELING, ISTADG, ISTAEV, IFREE,  A, FUVALS,              &
                       S%lnguvl, FUVALS, S%lnhuvl, GVALS( : , 2 ),             &
                       GVALS( : , 3 ), DGRAD , Q, GSCALE_used, ESCALE,         &
                       GXEQX_used , INTREP, RANGE , S%icfact, inform%ciccg,    &
                       inform%nsemib, inform%ratio, S%print_level,             &
                       S%error, S%out, S%infor, alloc_status, bad_alloc,       &
                       ITYPEE, DIAG, OFFDIA, IW, IKEEP, IW1, IVUSE,            &
                       H_col_ptr, L_col_ptr, W, W1, RHS, RHS2, P2,             &
                       G, ISTAGV, ISVGRP,                                      &
                       lirnh, ljcnh, lh, lirnh_min, ljcnh_min, lh_min,         &
                       ROW_start, POS_in_H, USED, FILLED,                      &
                       lrowst, lpos, lused, lfilled,                           &
                       IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                 &
                       matrix, SILS_data, control%SILS_cntl,                   &
                       inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,&
                       S%PRECN, SCU_matrix, SCU_data, inform%SCU_info,         &
                       S%ASMBL, S%skipg, KNDOFG )
                 END IF
                 CALL CPU_TIME( tim )
                 S%tls = S%tls + tim - S%t
                 S%ifactr = 0

!  Check for error returns

                 IF ( S%infor == 10 ) THEN
                   inform%status = 4 ; GO TO 820 ; END IF
                 IF ( S%infor == 11 ) THEN
                   inform%status = 5 ; GO TO 820 ; END IF
                 IF ( S%infor == 12 ) GO TO 990
               ELSE

!  If required, use a diagonal preconditioner

                 IF ( S%dprcnd ) THEN
                   Q( IVAR( : S%nvar2 ) ) = DGRAD( : S%nvar2 ) /               &
                      FUVALS( S%ldx + IVAR( : S%nvar2 ) )
                 ELSE

!  No preconditioner is required

                   Q( IVAR( : S%nvar2 ) ) = DGRAD( : S%nvar2 ) *               &
                      VSCALE( IVAR( : S%nvar2 ) )
                 END IF
               END IF
             END IF
           END IF
  320      CONTINUE

!  The minimization will take place over all variables which are not on the
!  trust-region boundary with negative gradients pushing over the boundary

           IF ( S%direct ) THEN

!  - - - - - - - - - - - - direct method - - - - - - - - - - - - - - - -

!  Minimize the quadratic using a direct method. The method used is a
!  multifrontal symmetric indefinite factorization scheme. Evaluate the
!  gradient of the quadratic at XT

             inform%nvar = 0
             S%gmodel = zero
             DO i = 1, n
               IF ( INDEX( i ) == 0 ) THEN
                 inform%nvar = inform%nvar + 1
                 IVAR( inform%nvar ) = i
!                gi = FUVALS( S%lggfx + i ) + Q( i )
                 gi = GX0( i ) + Q( i )
                 DGRAD( inform%nvar ) = gi
                 S%gmodel = MAX( S%gmodel, ABS( gi ) )
               ELSE
                 gi = zero
               END IF
               P( i ) = zero ; QGRAD( i ) = gi
             END DO
             S%nvar2 = inform%nvar

!  Check if the gradient of the model at the generalized Cauchy point
!  is already small enough. Compute the ( scaled ) step moved from the
!  previous to the current iterate

             S%step =                                                          &
               LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE, .TRUE. )

!  If the step taken is small relative to the trust-region radius,
!  ensure that an accurate approximation to the minimizer of the
!  model is found

             IF ( S%step <= stptol * inform%radius ) THEN
               IF ( MAX( S%resmin, S%step * S%cgstop /                         &
                 ( inform%radius * stptol ) ) >= S%gmodel ) GO TO 400
             ELSE
               IF ( S%gmodel * S%gmodel < S%cgstop ) GO TO 400
             END IF

!  Factorize the matrix and obtain the solution to the linear system, a
!  direction of negative curvature or a descent direction for the quadratic
!  model

             CALL CPU_TIME( S%t )
             IF ( control%gn_model_after_cauchy ) THEN
               IF ( S%mortor ) THEN
                 CALL FRNTL_get_search_direction(                              &
                     n, ng, nel, S%ntotel, S%nnza, S%maxsel,                   &
                     S%nvargp, control%io_buffer, INTVAR, IELVAR, S%nvrels,    &
                     INTREP, IELING, ISTADG, ISTAEV, A     , ICNA  , ISTADA,   &
                     FUVALS, S%lnguvl, FUVALS, S%lnhuvl, ISTADH, GXEQX_used,   &
                     GVALS2_GN, GVALS( : , 3 ), IVAR, S%nvar2,                 &
                     QGRAD , P, XT, BND_radius, S%fmodel, GSCALE_used,         &
                     ESCALE, X0, S%twonrm, S%nobnds, S%dxsqr , S%rad,          &
                     S%cgstop, S%number, S%next  , S%modchl, RANGE ,           &
                     inform%nsemib, inform%ratio, S%print_level, S%error,      &
                     S%out, S%infor, alloc_status, bad_alloc,                  &
                     ITYPEE, DIAG, OFFDIA, IVUSE, RHS, RHS2, P2,               &
                     ISTAGV, ISVGRP, lirnh, ljcnh, lh,                         &
                     ROW_start, POS_in_H, USED, FILLED,                        &
                     lrowst, lpos, lused, lfilled,                             &
                     IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                   &
                     matrix, SILS_data, control%SILS_cntl,                     &
                     inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,  &
                     SCU_matrix, SCU_data, inform%SCU_info, S%ASMBL,           &
                     S%skipg, KNDOFG )
               ELSE
                 CALL FRNTL_get_search_direction(                              &
                     n, ng, nel, S%ntotel, S%nnza, S%maxsel,                   &
                     S%nvargp, control%io_buffer, INTVAR, IELVAR, S%nvrels,    &
                     INTREP, IELING, ISTADG, ISTAEV, A     , ICNA  , ISTADA,   &
                     FUVALS, S%lnguvl, FUVALS, S%lnhuvl, ISTADH, GXEQX_used,   &
                     GVALS2_GN, GVALS( : , 3 ), IVAR, S%nvar2,                 &
                     QGRAD , P, XT, BND, S%fmodel, GSCALE_used,                &
                     ESCALE, X0, S%twonrm, S%nobnds, S%dxsqr , S%rad,          &
                     S%cgstop, S%number, S%next  , S%modchl, RANGE ,           &
                     inform%nsemib, inform%ratio, S%print_level, S%error,      &
                     S%out, S%infor, alloc_status, bad_alloc,                  &
                     ITYPEE, DIAG, OFFDIA, IVUSE, RHS, RHS2, P2,               &
                     ISTAGV, ISVGRP, lirnh, ljcnh, lh,                         &
                     ROW_start, POS_in_H, USED, FILLED,                        &
                     lrowst, lpos, lused, lfilled,                             &
                     IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                   &
                     matrix, SILS_data, control%SILS_cntl,                     &
                     inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,  &
                     SCU_matrix, SCU_data, inform%SCU_info, S%ASMBL,           &
                     S%skipg, KNDOFG )
               END IF
             ELSE
               IF ( S%mortor ) THEN
                 CALL FRNTL_get_search_direction(                              &
                     n, ng, nel, S%ntotel, S%nnza, S%maxsel,                   &
                     S%nvargp, control%io_buffer, INTVAR, IELVAR, S%nvrels,    &
                     INTREP, IELING, ISTADG, ISTAEV, A     , ICNA  , ISTADA,   &
                     FUVALS, S%lnguvl, FUVALS, S%lnhuvl, ISTADH, GXEQX_used,   &
                     GVALS( : , 2 ), GVALS( : , 3 ), IVAR, S%nvar2,            &
                     QGRAD , P, XT, BND_radius, S%fmodel, GSCALE_used,         &
                     ESCALE, X0, S%twonrm, S%nobnds, S%dxsqr , S%rad,          &
                     S%cgstop, S%number, S%next  , S%modchl, RANGE,            &
                     inform%nsemib, inform%ratio, S%print_level, S%error,      &
                     S%out, S%infor, alloc_status, bad_alloc,                  &
                     ITYPEE, DIAG, OFFDIA, IVUSE, RHS, RHS2, P2,               &
                     ISTAGV, ISVGRP, lirnh, ljcnh, lh,                         &
                     ROW_start, POS_in_H, USED, FILLED,                        &
                     lrowst, lpos, lused, lfilled,                             &
                     IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                   &
                     matrix, SILS_data, control%SILS_cntl,                     &
                     inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,  &
                     SCU_matrix, SCU_data, inform%SCU_info, S%ASMBL,           &
                     S%skipg, KNDOFG )
               ELSE
                 CALL FRNTL_get_search_direction(                              &
                     n, ng, nel, S%ntotel, S%nnza, S%maxsel,                   &
                     S%nvargp, control%io_buffer, INTVAR, IELVAR, S%nvrels,    &
                     INTREP, IELING, ISTADG, ISTAEV, A     , ICNA  , ISTADA,   &
                     FUVALS, S%lnguvl, FUVALS, S%lnhuvl, ISTADH, GXEQX_used,   &
                     GVALS( : , 2 ), GVALS( : , 3 ), IVAR, S%nvar2,            &
                     QGRAD , P, XT, BND   , S%fmodel, GSCALE_used,             &
                     ESCALE, X0, S%twonrm, S%nobnds, S%dxsqr , S%rad,          &
                     S%cgstop, S%number, S%next  , S%modchl, RANGE,            &
                     inform%nsemib, inform%ratio, S%print_level, S%error,      &
                     S%out, S%infor, alloc_status, bad_alloc,                  &
                     ITYPEE, DIAG, OFFDIA, IVUSE, RHS, RHS2, P2,               &
                     ISTAGV, ISVGRP, lirnh, ljcnh, lh,                         &
                     ROW_start, POS_in_H, USED, FILLED,                        &
                     lrowst, lpos, lused, lfilled,                             &
                     IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                   &
                     matrix, SILS_data, control%SILS_cntl,                     &
                     inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,  &
                     SCU_matrix, SCU_data, inform%SCU_info, S%ASMBL,           &
                     S%skipg, KNDOFG )
               END IF
             END IF
             CALL CPU_TIME( tim )
             S%tls = S%tls + tim - S%t
             inform%nvar = S%nvar2

!  Check for error returns

             IF ( S%infor == 10 ) THEN ; inform%status = 4 ; GO TO 820 ; END IF
             IF ( S%infor == 11 ) THEN ; inform%status = 5 ; GO TO 820 ; END IF
             IF ( S%infor == 12 ) GO TO 990
             IF ( S%infor >= 6 ) THEN
               inform%status = S%infor ; GO TO 820 ; END IF

!  Save details of the system solved

             S%fill = MAX( S%fill, inform%ratio )
             S%ISYS( S%infor ) = S%ISYS( S%infor ) + 1
             S%lisend1 = S%LSENDS1( S%infor )

!  Compute the ( scaled ) step from the previous to the current iterate
!  in the appropriate norm

             S%step =                                                          &
               LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE, .TRUE. )

!  For debugging, compute the directional derivative and curvature
!  along the direction P

             IF ( S%printm ) THEN
               IF ( .NOT. S%alllin ) ISWKSP( : S%ntotel ) = 0
               nvar1 = 0
               DO i = 1, S%nvar2
                 IF ( IVAR( i ) > 0 ) THEN
                   nvar1 = nvar1 + 1
                   IVAR( nvar1 ) = IVAR( i )
                 END IF
               END DO
               S%nvar2 = nvar1 ; inform%nvar = S%nvar2
               nvar1 = 1 ; S%nbprod = 1

!  Evaluate the product of the Hessian with the dense vector P

               CALL CPU_TIME( S%t )
               Q = zero
               S%densep = .TRUE.
               IF ( control%gn_model_after_cauchy ) THEN
                 CALL HSPRD_hessian_times_vector(                              &
                     n , ng, nel, S%ntotel, S%nvrels,                          &
                     S%nvargp, inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,      &
                     S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR,       &
                     IELING, IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),      &
                     P , Q , GVALS2_GN, GVALS( : , 3 ),                        &
                     GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),        &
                     S%lnhuvl, GXEQX_used , INTREP, S%densep,                  &
                     IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,   &
                     ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,         &
                     NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
               ELSE
                 CALL HSPRD_hessian_times_vector(                              &
                     n , ng, nel, S%ntotel, S%nvrels,                          &
                     S%nvargp, inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,      &
                     S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR,       &
                     IELING, IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),      &
                     P , Q , GVALS( : , 2 ), GVALS( : , 3 ),                   &
                     GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),        &
                     S%lnhuvl, GXEQX_used , INTREP, S%densep,                  &
                     IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,   &
                     ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,         &
                     NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
               END IF
               CALL CPU_TIME( tim )
               S%tmv = S%tmv + tim - S%t

!  Compute the curvature

!              S%curv =                                                        &
!                DOT_PRODUCT( Q( IVAR( : S%nvar2 ) ), P( IVAR( : S%nvar2 ) ) )
               S%curv = zero
               DO i = 1, S%nvar2
                 S%curv = S%curv + Q( IVAR( i ) ) * P( IVAR( i ) )
               END DO

!  Compare the calculated and recurred curvature

               WRITE( S%out, "( ' curv  = ', ES12.4 )" ) S%curv
               WRITE( S%out, "( ' FRNTL - infor = ', I1 )" ) S%infor
               IF ( S%infor == 1 .OR. S%infor == 3 .OR. S%infor == 5 ) THEN
                 DO j = 1, S%nvar2
                   i = IVAR( j )
                   WRITE( S%out, "( ' P, H * P( ', I6, ' ), RHS( ', I6,        &
                  &  ' ) = ', 3ES15.7 )" ) i, i, P( i ), Q( i ), QGRAD( i )
                 END DO
               END IF
             END IF

!  - - - - - - - - - - - - iterative method - - - - - - - - - - - - - -

!   Minimize the quadratic using an iterative method. The method used
!   is a safeguarded preconditioned conjugate gradient scheme

           ELSE
             CALL CPU_TIME( S%t )
             IF ( S%mortor ) THEN
               inform%itcgmx = COUNT( INDEX == 0 )
               IF ( S%mortor_its > 0 )                                         &
                 inform%itcgmx = MAX( 10, inform%itcgmx / 2 )
!              IF ( S%mortor_its > - 1 ) inform%itcgmx = 10
               CALL CG_solve(                                                  &
                   n, X0, XT, GX0,  BND_radius, S%nbnd,                        &
                   INDEX, S%cgstop, S%fmodel, DGRAD, inform%status, P, Q,      &
                   IVAR, inform%nvar, S%nvar2, S%twonrm, S%rad, S%nobnds,      &
                   S%gmodel, S%dxsqr, S%out, S%jumpto, S%print_level,          &
                   S%findmx, inform%itercg, inform%itcgmx,                     &
                   inform%ifixed, W_ws, S%CG, XSCALE = VSCALE )
             ELSE
               inform%itcgmx = 3 * COUNT( INDEX == 0 )
               CALL CG_solve(                                                  &
                   n, X0, XT, GX0, BND, n,                                     &
                   INDEX, S%cgstop, S%fmodel, DGRAD, inform%status, P, Q,      &
                   IVAR, inform%nvar, S%nvar2, S%twonrm, S%rad, S%nobnds,      &
                   S%gmodel, S%dxsqr, S%out, S%jumpto, S%print_level,          &
                   S%findmx, inform%itercg, inform%itcgmx,                     &
                   inform%ifixed, W_ws, S%CG, XSCALE = VSCALE )
             END IF
             CALL CPU_TIME( tim )
             S%tls = S%tls + tim - S%t
             IF ( S%jumpto == 0 .OR. S%jumpto == 4 .OR. S%jumpto == 5 )        &
               S%step =                                                        &
                 LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE, .TRUE. )

!  The norm of the gradient of the quadratic model is smaller than
!  cgstop. Perform additional tests to see if the current iterate
!  is acceptable

             S%nvar2 = inform%nvar

!  If the (scaled) step taken is small relative to the trust-region radius,
!  ensure that an accurate approximation to the minimizer of the model is found

             IF ( S%jumpto == 4 ) THEN
               IF ( S%step <= stptol * inform%radius .AND.                     &
                   .NOT. control%quadratic_problem .AND. .NOT. S%slvbqp ) THEN
                 IF ( MAX( S%resmin, S%step * S%cgstop /                       &
                   ( inform%radius * stptol ) ) >=  S%gmodel ) THEN
                   IF ( S%printw ) WRITE( S%out,                               &
                     "( ' Norm of trial step ', ES12.4 )" ) S%step
                   S%jumpto = 0
                 ELSE
                   gi = S%step * S%cgstop / ( inform%radius * stptol )
                   IF ( S%printw ) WRITE( S%out,                               &
                     "( /, ' C.G. tolerance of ', ES12.4, ' has not been',     &
                  &        ' achieved. ', /, ' Actual step length = ', ES12.4, &
                  &        ' Radius = ', ES12.4 )" ) gi, S%step, inform%radius
                   S%jumpto = 4
                 END IF
               ELSE
                 S%jumpto = 0
               END IF
             END IF

!  A bound has been encountered in CG. If the bound is a trust-region bound,
!  stop the minimization

             IF ( S%jumpto == 5 ) THEN
               S%ifactr = 2
               S%nadd = 1
               IF ( S%twonrm ) THEN
                 S%jumpto = 2
               ELSE
                 IF ( S%slvbqp ) THEN
                   S%jumpto = 2
                 ELSE
                   S%jumpto = 0
                   IF ( S%mortor ) THEN
 
!  The bound encountered is an upper bound

!                    IF ( inform%ifixed > 0 ) THEN
!                      IF ( BU( inform%ifixed ) <                              &
!                        BND_radius( inform%ifixed, 2 ) ) S%jumpto = 2
!                    ELSE

!  The bound encountered is a lower bound

!                      IF ( BL( - inform%ifixed ) >                            &
!                        BND_radius( - inform%ifixed, 1 ) ) S%jumpto = 2
!                    END IF
                   ELSE 
                     IF ( S%strctr ) THEN
                       S%rad = BND_radius( i, 1 )
                     ELSE
                       S%rad = inform%radius
                     END IF

!  The bound encountered is an upper bound

                     IF ( inform%ifixed > 0 ) THEN
                       IF ( S%calcdi ) THEN
                         IF ( BU( inform%ifixed ) < X( inform%ifixed ) +       &
                           S%rad / SQRT( FUVALS( S%ldx + inform%ifixed ) ) )   &
                             S%jumpto = 2
                       ELSE
                         IF ( BU( inform%ifixed ) < X( inform%ifixed ) +       &
                           S%rad * VSCALE( inform%ifixed ) ) S%jumpto = 2
                       END IF
                     ELSE

!  The bound encountered is a lower bound

                       IF ( S%calcdi ) THEN
                         IF ( BL( - inform%ifixed ) > X( - inform%ifixed ) -   &
                           S%rad / SQRT( FUVALS( S%ldx - inform%ifixed ) ) )   &
                             S%jumpto = 2
                       ELSE
                         IF ( BL( - inform%ifixed ) > X( - inform%ifixed ) -   &
                           S%rad * VSCALE( - inform%ifixed ) ) S%jumpto = 2
                       END IF
                     END IF
                   END IF
                 END IF
               END IF
               IF ( S%printw .AND. S%jumpto == 2 ) WRITE( S%out,               &
                 "( /, ' Restarting the conjugate gradient iteration ' )" )
             END IF

!  If the bound encountered was a problem bound, continue minimizing the model

             IF ( S%jumpto > 0 ) GO TO 300
             S%cgend1 = S%CGENDS1( inform%status - 9 )
           END IF

!  If required, compute the value of the model from first principles

           IF ( S%printw ) THEN
             S%nbprod = S%nbprod + 1
             inform%nvar = n ; nvar1 = 1 ; S%nvar2 = inform%nvar

!  Compute the step taken, P

             DO i = 1, n
               IVAR( i ) = i
               P( i ) = XT( i ) - X( i )
             END DO

!  Evaluate the product of the Hessian with the dense vector P

             CALL CPU_TIME( S%t )
             Q = zero
             S%densep = .TRUE.
             IF ( control%gn_model_after_cauchy ) THEN
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR, IELING, &
                   IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                &
                   P , Q , GVALS2_GN, GVALS( : , 3 ),                          &
                   GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl,&
                   GXEQX_used , INTREP, S%densep,                              &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             ELSE
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR, IELING, &
                   IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                &
                   P , Q , GVALS( : , 2 ), GVALS( : , 3 ),                     &
                   GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl,&
                   GXEQX_used , INTREP, S%densep,                              &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             END IF
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t

!  If required, print the step taken

             IF ( S%out > 0 .AND. S%print_level >= 20 ) WRITE( S%out, 2530 ) P

!  Compute the model value, fnew, and reset P to zero

             S%fnew = inform%aug
!DIR$ IVDEP  
             DO j = 1, S%nvar2
               i = IVAR( j )
               S%fnew =                                                        &
                 S%fnew + ( FUVALS( S%lggfx + i ) + half * Q( i ) ) * P( i )
               P( i ) = zero
             END DO
             WRITE( S%out, "( ' *** Calculated quadratic at end CG ', ES22.14, &
            &   /, ' *** Recurred   quadratic at end CG ', ES22.14 )" )        &
               S%fnew * S%findmx, S%fmodel * S%findmx
           END IF

!  ------------------------------------------------------
!  Step 3.25 of the algorithm - More'-Toraldo projections
!  ------------------------------------------------------

           IF ( S%mortor ) THEN
             j = 0
             DO i = 1, n
               IF ( XT( i ) < BL( i ) .OR. XT( i ) > BU( i ) ) THEN
!                WRITE(6,"(3ES12.4)" ) BL(i), XT( i ), BU( i )
                 IF ( S%printt ) WRITE( S%out,                                 &
                   "( /, '    Problem bound would be violated so .... ' )" )
                 j = 1
                 EXIT 
               END IF
             END DO
             
!  Compute P, the step taken to the Cauchy point

             IF ( j == 1 ) THEN
               inform%nvar = n
               S%nvar2 = inform%nvar
             
               DO i = 1, n
                 IVAR( i ) = i
                 P( i ) = XCP( i ) - X( i )
               END DO

!  Evaluate the product of the Hessian with the dense vector P

               CALL CPU_TIME( S%t )
               Q = zero
               S%densep = .TRUE.
               IF ( control%gn_model_after_cauchy ) THEN
                 CALL HSPRD_hessian_times_vector(                              &
                     n , ng, nel, S%ntotel, S%nvrels,                          &
                     S%nvargp, inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,      &
                     S%nbprod, S%alllin, IVAR, ISTAEV, ISTADH, INTVAR, IELING, &
                     IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),              &
                     P , Q , GVALS2_GN, GVALS( : , 3 ),                        &
                     GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),        &
                     S%lnhuvl, GXEQX_used , INTREP, S%densep,                  &
                     IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,   &
                     ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,         &
                     NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
               ELSE
                 CALL HSPRD_hessian_times_vector(                              &
                     n , ng, nel, S%ntotel, S%nvrels,                          &
                     S%nvargp, inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,      &
                     S%nbprod, S%alllin, IVAR, ISTAEV, ISTADH, INTVAR, IELING, &
                     IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),              &
                     P , Q , GVALS( : , 2 ), GVALS( : , 3 ),                   &
                     GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),        &
                     S%lnhuvl, GXEQX_used , INTREP, S%densep,                  &
                     IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,   &
                     ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,         &
                     NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
               END IF
               CALL CPU_TIME( tim )
               S%tmv = S%tmv + tim - S%t

!  Recover the Cauchy point and its function and gradient values

               X0( : n ) = XCP( : n )
               S%f0 = S%fcp
               GX0 = FUVALS( S%lggfx + 1 : S%lggfx + n ) + Q
               
!              WRITE(6,"( 'P   ', 5ES12.4 )" ) P
!              WRITE(6,"( 'Q   ', 5ES12.4 )" ) Q
!              WRITE(6,"( ' gx0 ', 5ES12.4 )" ) GX0

!  Recover the set of free variables at the Cauchy point

!              inform%nvar = S%nfreec
!              IVAR( : S%nfreec ) = IFREEC( : S%nfreec )

!  Set the Cauchy direction

               DGRAD = XT - XCP
               S%stepmx = MIN( S%stepmx, one )
               P = zero
               
               IF ( S%twonrm ) S%rad = SQRT( DOT_PRODUCT( DGRAD, DGRAD ) )

!  If possible, use the existing preconditioner

!              IF ( S%refact ) THEN
!                S%ifactr = 1
!              ELSE

!  Ensure that a new Schur complement is calculated. Restore the complete
!  list of variables that were free when the factorization was calculated

!                S%ifactr = 2 ; S%nadd = 1 ; S%nfixed = 0
!                IFREE( : S%nfreef ) = ABS( IFREE( : S%nfreef ) )
!              END IF
               S%jumpto = 1
               S%mortor_its = S%mortor_its + 1
               GO TO 240
             END IF
           END IF

!  -------------------------------------
!  Step 3.5 of the algorithm (SEE PAPER)
!  -------------------------------------

!  An accurate approximation to the minimum of the quadratic model is to be
!  sought

           IF ( S%slvbqp .AND. S%ibqpst <= 2 ) THEN

!  Compute the gradient value

             inform%nvar = n
             S%nvar2 = inform%nvar

!  Compute the step taken

             DO i = 1, n
               IVAR( i ) = i
               DELTAX( i ) = XT( i ) - X( i )
             END DO

!  Evaluate the product of the Hessian with the dense step vector

             CALL CPU_TIME( S%t )
             Q = zero
             S%densep = .TRUE.
             IF ( control%gn_model_after_cauchy ) THEN
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR,         &
                   IELING, IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),        &
                   DELTAX, Q, GVALS2_GN, GVALS( : , 3 ),                       &
                   GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),          &
                   S%lnhuvl, GXEQX_used , INTREP, S%densep,                    &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             ELSE
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                  &
                   inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                  &
                   S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR,         &
                   IELING, IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),        &
                   DELTAX, Q, GVALS( : , 2 ), GVALS( : , 3 ),                  &
                   GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),          &
                   S%lnhuvl, GXEQX_used , INTREP, S%densep,                    &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             END IF
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t

!  Compute the model gradient at XT

             GX0 = FUVALS( S%lggfx + 1 : S%lggfx + n ) + Q
             dltnrm = MAX( zero, MAXVAL( ABS( Q ) ) )

!  Save the values of the nonzero components of the gradient

             k = 0
             DO j = 1, S%nfreef
               i = IFREE( j )
               IF ( i > 0 ) THEN
                 k = k + 1
                 GX0( i ) = DGRAD( k )
               END IF
             END DO

!  Find the projected gradient of the model and its norm

             CALL LANCELOT_projected_gradient(                                 &
                 n, XT, GX0, VSCALE, BND( : , 1 ), BND( : , 2 ), DGRAD,  &
                 IVAR, inform%nvar, S%gmodel )

!  Check for convergence of the inner iteration

             IF ( S%printt )                                                   &
               WRITE( S%out, "( /, '    ** Model gradient is ', ES12.4,        &
              &  ' Required accuracy is ', ES12.4 )" ) S%gmodel, SQRT( S%cgstop)
             IF ( S%gmodel * S%gmodel > S%cgstop .AND. dltnrm > epsmch ) THEN

!  The approximation to the minimizer of the quadratic model is not yet
!  good enough. Perform another iteration

!  Store the function value at the starting point for the Cauchy search

               S%f0 = S%fmodel

!  Set the staring point for the Cauchy step

               X0 = XT

!  Set the Cauchy direction

               DGRAD = - GX0 * ( VSCALE / S%vscmax ) ** 2
               P = zero

!  If possible, use the existing preconditioner

               IF ( S%refact ) THEN
                 S%ifactr = 1
               ELSE

!  Ensure that a new Schur complement is calculated. Restore the complete
!  list of variables that were free when the factorization was calculated

                 S%ifactr = 2 ; S%nadd = 1 ; S%nfixed = 0
                 IFREE( : S%nfreef ) = ABS( IFREE( : S%nfreef ) )
               END IF
               S%jumpto = 1
               GO TO 240
             END IF
           END IF

!  -----------------------------------
!  Step 4 of the algorithm (see paper)
!  -----------------------------------

!  Test for acceptance of new point and trust-region management

  400      CONTINUE
!          WRITE(6,"( 4ES12.4 )") ( BND( i, 1 ), X( i ),                       &
!            XT( i ) - X( i ), BND( i, 2 ), i = 1, n )

!  Determine which nonlinear elements and non-trivial groups need to
!  be re-evaluated by considering which of the variables have changed

           CALL OTHERS_which_variables_changed(                                &
               S%unsucc, n, ng, nel, inform%ncalcf, inform%ncalcg, ISTAEV,     &
               ISTADG, IELING, ICALCF, ICALCG, X, XT, ISTAJC, IGCOLJ,          &
               LIST_elements, LINK_elem_uses_var )

!  If required, print a list of the nonlinear elements and groups
!  which have changed

           IF ( S%printw .AND. .NOT. S%alllin ) THEN
             WRITE( S%out,                                                     &
               "( /, ' Functions for the following elements need to be',       &
            &      ' re-evaluated ', /, ( 12I6 ) )" ) ICALCF( : inform%ncalcf )
             WRITE( S%out,                                                     &
               "( /, ' Functions for the following groups need to be',         &
            &      ' re-evaluated ', /, ( 12I6 ) )" ) ICALCG( : inform%ncalcg )
           END IF

!  If the step taken is ridiculously small, exit

           IF ( S%step <= S%stpmin ) THEN
             inform%status = 3
             GO TO 600
           END IF

!IF ( inform%iter == 16 ) then
!write(6,*) ' &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& trying xt = x '
!XT = X
!end if

!  Return to the calling program to obtain the function value at the new point

           inform%status = - 3
           IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF

!  Compute the group argument values FT

  430      CONTINUE
           DO ig = 1, ng

!  Include the contribution from the linear element

!            ftt = SUM( A( ISTADA( ig ) : ISTADA( ig + 1 ) - 1 ) *             &
!              XT( ICNA( ISTADA( ig ) : ISTADA( ig + 1 ) - 1 ) ) ) - B( ig )
             ftt = - B( ig )
             DO i = ISTADA( ig ), ISTADA( ig + 1 ) - 1
               ftt = ftt + A( i ) * XT( ICNA( i ) )
             END DO
  
!  Include the contributions from the nonlinear elements
  
!            ftt = ftt + SUM( ESCALE( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) *  &
!              FUVALS( IELING( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) ) )
             DO i = ISTADG( ig ), ISTADG( ig + 1 ) - 1
               ftt = ftt + ESCALE( i ) * FUVALS( IELING( i ) )
             END DO
             FT( ig ) = ftt
           END DO

!  Compute the group function values

           IF ( S%altriv ) THEN
!            S%fnew = DOT_PRODUCT( GSCALE_used, FT )
             S%fnew = zero
             IF ( S%skipg ) THEN
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 0 )                                       &
                   S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
                END DO
             ELSE
               DO ig = 1, ng 
                 S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
               END DO
             END IF
           ELSE

!  If necessary, return to the calling program to obtain the group
!  function and derivative values at the initial point

             inform%status = - 4
             IF ( external_gr ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
           END IF
  470      CONTINUE
           IF ( .NOT. S%altriv ) THEN
             S%fnew = zero
             IF ( S%p_type == 2 ) THEN
               IF ( S%skipg ) THEN
                 DO ig = 1, ng
                   IF ( KNDOFG( ig ) > 0 )                                     &
                     S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                 END DO
               ELSE
                 DO ig = 1, ng
                   S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                 END DO
               END IF
             ELSE
               IF ( S%skipg ) THEN
                 DO ig = 1, ng
                   IF ( KNDOFG( ig ) > 0 ) THEN
                     IF ( GXEQX_used( ig ) ) THEN
                        S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
!write(6,"('n', I2, 2ES12.4)") ig, GSCALE_used( ig ), FT( ig )
                    ELSE
!write(6,"('y', I2, 2ES12.4)") ig, GSCALE_used( ig ), GVALS( ig, 1 )
                        S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                     END IF
                   END IF
                 END DO
               ELSE
                 DO ig = 1, ng
                   IF ( GXEQX_used( ig ) ) THEN
!write(6,"('n', I2, 2ES12.4)") ig, GSCALE_used( ig ), FT( ig )
                      S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
                   ELSE
!write(6,"('y', I2, 2ES12.4)") ig, GSCALE_used( ig ), GVALS( ig, 1 )
                      S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                   END IF
                 END DO
               END IF
             END IF
           END IF

!  Compute the actual and predicted reductions in the function value.
!  Ensure that rounding errors do not dominate

           IF ( S%printm ) WRITE( S%out,                                       &
             "( /, ' f_current    = ', ES20.12, /,                             &
          &        ' f_new        = ', ES20.12, /                              &
          &        ' model_new    = ', ES20.12 )" ) inform%aug, S%fnew, S%fmodel
           S%ared =                                                            &
             ( inform%aug - S%fnew ) + MAX( one, ABS( inform%aug ) ) * S%teneps
           S%prered =                                                          &
            ( inform%aug - S%fmodel ) + MAX( one, ABS( inform%aug ) ) * S%teneps
!          write(6,"(A,3ES12.4)") ' orig, new_f, new_m ', inform%aug, S%fnew,  &
!            S%fmodel
           IF ( ABS( S%ared ) < S%teneps .AND. ABS( inform%aug ) > S%teneps )  &
             S%ared = S%prered
           IF ( control%quadratic_problem ) THEN
             S%rho = one
           ELSE
             S%rho = S%ared / S%prered
           END IF
           IF ( S%out > 0 .AND. S%print_level >= 100 ) WRITE( S%out,           &
             "( /, ' Old f = ', ES20.12, ' New   f = ', ES20.12, /,            &
          &        ' Old f = ', ES20.12, ' Model f = ', ES20.12 )" )           &
             inform%aug, S%fnew, inform%aug, S%fmodel
           IF ( S%printm ) WRITE( S%out,                                       &
             "( /, ' Actual change    = ', ES20.12, /,                         &
          &        ' Predicted change = ', ES20.12, /                          &
          &        ' Ratio ( rho )    = ', ES20.12 )" ) S%ared, S%prered, S%rho

!  Adjust rho in the non-monotone case

           IF ( S%nmhist > 0 ) THEN
!write(6,*) ' hist, f_r, sigma_r ',  S%nmhist, S%f_r, S%sigma_r
             ar_h = ( S%f_r - S%fnew ) + MAX( one, ABS( S%f_r ) ) * S%teneps
             pr_h = S%sigma_r + S%prered
             IF ( ABS( ar_h ) < S%teneps .AND. ABS( S%f_r ) > S%teneps )       &
               ar_h = pr_h
             S%rho = MAX( S%rho, ar_h / pr_h )
             IF ( S%printm ) WRITE( S%out,                                     &
               "( /, ' Nonmonotone actual change    = ', ES20.12, /,           &
            & ' Nonmonotone predicted change = ', ES20.12, /                   &
            & ' Nonmonotone ratio ( rho )    = ', ES20.12 )" ) ar_h, pr_h, S%rho
           END IF

!  Compute the actual and predicted reductions in each of the
!  group values in the structured trust-region case

!          IF ( S%strctr ) THEN
!            CALL STRUTR_changes( DIMEN , D_model, D_function, XT - X,         &
!                IELING, ISTADG, IELVAR, ISTAEV, INTVAR, ISTADH, ISTADA, ICNA, &
!                A, ESCALE, GSCALE_used, FT, GXEQX_used, INTREP,               &
!                FUVALS, S%lnhuvl, GV_old, GVALS( : , 1 ),                     &
!                GVALS( : , 2 ), GVALS( : , 3 ), GRJAC , S%nvargp, RANGE )
!          END IF

!  -----------------------------------
!  Step 5 of the algorithm (see paper)
!  -----------------------------------

           S%oldrad = inform%radius

!  - - - - step management when the iteration has proved unsuccessful -

           IF ( S%rho < control%eta_successful .OR. S%prered <= zero ) THEN

!  unsuccessful step. Calculate the radius which would just include the newly
!  found point, XT

             S%unsucc = .TRUE.
             IF ( S%rho >= zero .AND. S%prered > zero ) THEN
               S%radmin = S%step
             ELSE

!  Very unsuccessful step. Obtain an estimate of the radius required to obtain
!  a successful step along the step taken, radmin, if such a step were taken
!  at the next iteration

!              slope =                                                         &
!                DOT_PRODUCT( FUVALS( S%lggfx + 1: S%lggfx + n ), XT - X )
               slope = zero
               DO i = 1, n
                 slope = slope + FUVALS( S%lggfx + i ) * ( XT( i ) - X( i ) )
               END DO
               S%curv = S%fmodel - inform%aug - slope
               S%radmin = S%step * ( control%eta_very_successful - one ) *     &
                 slope / ( S%fnew - inform%aug - slope -                       &
                 control%eta_very_successful * S%curv )
             END IF

!  Update the trust-region radius/radii

             IF ( S%strctr ) THEN

!  Structured trust-region case:

!              CALL STRUTR_radius_update(                                      &
!                  DIMEN, D_model, D_function, S%ared, S%prered, control,      &
!                  RADII )
              
               CALL STRUTR_radius_update(                                      &
                   n, ng, nel, XT - X, IELING, ISTADG, IELVAR, ISTAEV, INTVAR, &
                   ISTADH, ISTADA, ICNA, A, ESCALE, GSCALE_used, FT,           &
                   GXEQX_used, ITYPEE, INTREP, FUVALS, S%lnhuvl, GV_old,       &
                   GVALS( : , 1 ),  GVALS( : , 2 ), GVALS( : , 3 ), GRJAC,     &
                   S%nvargp, S%ared, S%prered, RADII, S%maximum_radius,        &
                   control%eta_successful, control%eta_very_successful,        &
                   control%eta_extremely_successful, control%gamma_decrease,   &
                   control%gamma_increase, control%mu_meaningful_model,        &
                   control%mu_meaningful_group, ISTAGV, ISVGRP, IVALJR,        &
                   ISYMMH, W_el, W_in, H_in, W_ws, RANGE )
               inform%radius = MAXVAL( RADII )

!  Unstructured trust-region case:

             ELSE

!  Compute an upper bound on the new trust-region radius. Radmin, the actual
!  radius will be the current radius multiplied by the largest power of 
!  gamma_decrease for which the product is smaller than radmin.

               S%radmin = MIN( S%step, MAX(                                    &
                 inform%radius * control%gamma_smallest, S%radmin ) )

!  If the trust-region radius has shrunk too much, exit. this may indicate a
!  derivative bug or that the user is asking for too much accuracy in the
!  final gradient

               IF ( S%radmin < S%radtol ) THEN
                 IF ( S%printe ) WRITE( S%out, 2540 )
                 inform%status = 2 ; GO TO 600
               END IF

!  Continue reducing the radius by the factor gamma_decrease until it is 
!  smaller than radmin

               DO
                 inform%radius = control%gamma_decrease * inform%radius
                 IF ( inform%radius < S%radmin ) EXIT
               END DO
             
             END IF

!  Compute the distance of the generalized Cauchy point from the
!  initial point

             IF ( S%calcdi ) THEN
               IF ( n > 0 ) QGRAD( : n ) = one /                   &
                 SQRT( FUVALS( S%ldx + 1 : S%ldx + n ) )
               S%step =                                                        &
                 LANCELOT_norm_diff( n, XT, X, S%twonrm, QGRAD, .TRUE. )
             ELSE
               S%step =                                                        &
                 LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE,.TRUE. )
             END IF

!  If the generalized Cauchy point lies within the new trust region,
!  it may be reused

!            S%reusec = S%step < inform%radius

!  Start a further iteration using the newly reduced trust region

             IF ( S%direct ) S%next = .FALSE.
             GO TO 120

!  - - - - - - - - - - - successful step - - - - - - - - - - - - - - - -

           ELSE
             S%unsucc = .FALSE.
             S%n_steering = 0

!  In the non-monotone case, update the sum of predicted models

             IF ( S%nmhist > 0 ) THEN
               S%sigma_c = S%sigma_c + ( inform%aug - S%fmodel )
               S%sigma_r = S%sigma_r + ( inform%aug - S%fmodel )

!  If appropriate, update the best value found

               IF ( S%fnew < S%f_min ) THEN
                 S%f_min = S%fnew ; S%f_c = S%f_min
                 IF ( S%p_type > 2 ) THEN
                   S%f_min_viol = S%violation
                   S%f_min_lag = S%f_min - S%f_min_viol / inform%mu
                   S%f_c_viol = S%f_min_viol ; S%f_c_lag = S%f_min_lag
                 END IF
                 S%sigma_c = zero ; S%l_suc = 0

!  Otherwise, increment l_min by one

               ELSE
                 S%l_suc = S%l_suc + 1

!  Check to see if there is a new candidate for the next reference value

                 IF ( S%fnew > S%f_c ) THEN
                   S%f_c = S%fnew ; S%sigma_c = zero
                   IF ( S%p_type > 2 ) THEN
                     S%f_c_viol = S%violation
                     S%f_c_lag = S%f_c - S%f_c_viol / inform%mu
                   END IF
                 END IF

!  Check to see if the reference value needs to be reset

                 IF ( S%l_suc == S%nmhist ) THEN
                   S%f_r = S%f_c
                   IF ( S%p_type > 2 ) THEN
                     S%f_r_viol = S%f_c_viol ; S%f_r_lag = S%f_c_lag
                   END IF
                   S%sigma_r = S%sigma_c
                 END IF
               END IF
             END IF

!  Update the trust-region radius/radii

             IF ( S%strctr ) THEN

!  Structured trust-region case:

!              CALL STRUTR_radius_update( DIMEN, D_model, D_function, S%ared,  &
!                                         S%prered, control, RADII )
               CALL STRUTR_radius_update(                                      &
                   n, ng, nel, XT - X, IELING, ISTADG, IELVAR, ISTAEV, INTVAR, &
                   ISTADH, ISTADA, ICNA, A, ESCALE, GSCALE_used, FT,           &
                   GXEQX_used, ITYPEE, INTREP, FUVALS, S%lnhuvl, GV_old,       &
                   GVALS( : , 1 ), GVALS( : , 2 ), GVALS( : , 3 ), GRJAC,      &
                   S%nvargp, S%ared, S%prered, RADII, S%maximum_radius,        &
                   control%eta_successful, control%eta_very_successful,        &
                   control%eta_extremely_successful, control%gamma_decrease,   &
                   control%gamma_increase, control%mu_meaningful_model,        &
                   control%mu_meaningful_group, ISTAGV, ISVGRP, IVALJR,        &
                   ISYMMH, W_el, W_in, H_in, W_ws, RANGE )
               inform%radius = MAXVAL( RADII )

!  Unstructured trust-region case:
!  Increase the trust-region radius. Note that we require the step taken to be
!  at least a certain multiple of the distance to the trust-region boundary

             ELSE
               IF ( S%rho >= control%eta_very_successful )                     &
                 inform%radius = MIN( MAX( inform%radius,                      &
                   control%gamma_increase * S%step ), S%maximum_radius )
             END IF 

!  - - derivative evaluations when the iteration has proved successful -

!  Evaluate the gradient and approximate Hessian. Firstly, save the
!  old element gradients if approximate Hessians are to be used

             CALL CPU_TIME( S%t )
             IF ( .NOT. S%second .AND. .NOT. S%alllin ) THEN
               IF ( use_elders ) THEN
                 QGRAD( : S%ntotin ) = FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin )
               ELSE
                 QGRAD( : S%ntotin ) = FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin )
               END IF

!  If they are used, update the second derivative approximations.
!  Form the differences in the iterates, P

               P = XT - X
             END IF
             CALL CPU_TIME( tim )
             S%tup = S%tup + tim - S%t

!  Accept the computed point and function value

             inform%aug = S%fnew ; X = XT

!write(6,"(A)") '+*+*+*+*+*+*+*+ ACCEPTING NEW POINT +*+*+*+*+*+*+*+ '

             IF ( S%p_type > 2 ) THEN
               S%violation = zero
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 1 ) THEN
                   S%violation = S%violation + ( GSCALE( ig ) * C( ig ) ) ** 2
                 END IF
               END DO
!              WRITE( 6, "( ' violation =', ES11.4 )" ) SQRT( S%violation )
             END IF
             S%violation = half * S%violation

!  record the violation

               IF ( S%printt .AND. control%steering .AND. S%p_type > 2 .AND.   &
                   .NOT. S%steering ) WRITE( S%out,                            &
             "( /, '    Turn steering on until next unsuccessful iteration' )" )

             S%steering = control%steering .AND. S%p_type > 2
             IF ( S%steering ) C_best = C

!  Return to the calling program to obtain the derivative
!  values at the new point

             IF ( S%fdgrad ) S%igetfd = 0
             IF ( .NOT. ( S%altriv .AND. S%alllin ) ) THEN
               inform%ngeval = inform%ngeval + 1
               IF ( S%altriv ) THEN
                 inform%status = - 6
                 IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
               ELSE
                 inform%status = - 5
                 IF ( external_el .AND. external_gr ) THEN
                   GO TO 800 ; ELSE ; GO TO 700 ; END IF
               END IF
             END IF
           END IF

!  If a structured trust-region is being used, store the current values
!  of the group functions

  540      CONTINUE
           IF ( .NOT. S%unsucc ) THEN
             IF ( S%strctr ) THEN
               DO ig = 1, ng
                 IF ( GXEQX_used( ig ) ) THEN
                   GV_old( ig ) = FT( ig )
                 ELSE
                   GV_old( ig ) = GVALS( ig, 1 )
                 END IF
               END DO
             END IF
           END IF

!  If finite-difference gradients are used, compute their values

           IF ( S%fdgrad .AND. .NOT. S%alllin ) THEN

!  Store the values of the nonlinear elements for future use

             IF ( S%igetfd == 0 ) THEN
               FUVALS_temp( : nel ) = FUVALS( : nel )
               S%centrl =                                                      &
                 S%first_derivatives == 2 .OR. inform%pjgnrm < epsmch ** 0.25
             END IF

!  Obtain a further set of differences

             IF ( use_elders ) THEN
               CALL OTHERS_fdgrad_flexible(                                    &
                                   n, nel, lfuval, S%ntotel, S%nvrels,         &
                                   S%nsets, IELVAR, ISTAEV, IELING,            &
                                   ICALCF, inform%ncalcf, INTVAR,              &
                                   S%ntype, X, XT, FUVALS, S%centrl, S%igetfd, &
                                   S%OTHERS, ISVSET, ISET, INVSET, ISSWTR,     &
                                   ISSITR, ITYPER, LIST_elements,              &
                                   LINK_elem_uses_var, WTRANS, ITRANS,         &
                                   ELDERS( 1, : ) )
             ELSE
               CALL OTHERS_fdgrad( n, nel, lfuval, S%ntotel, S%nvrels,         &
                                   S%nsets, IELVAR, ISTAEV, IELING,            &
                                   ICALCF, inform%ncalcf, INTVAR,              &
                                   S%ntype, X, XT, FUVALS, S%centrl, S%igetfd, &
                                   S%OTHERS, ISVSET, ISET, INVSET, ISSWTR,     &
                                   ISSITR, ITYPER, LIST_elements,              &
                                   LINK_elem_uses_var, WTRANS, ITRANS )
             END IF
             IF ( S%igetfd > 0 ) THEN
               inform%status = - 7
               IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
             END IF

!  Restore the values of the nonlinear elements at X

             S%igetfd = S%nsets + 1
             FUVALS( : nel ) = FUVALS_temp( : nel )
           END IF

!  Compute the gradient value

           CALL CPU_TIME( S%t )
           CALL LANCELOT_form_gradients(                                       &
               n, ng, nel, S%ntotel, S%nvrels, S%nnza,                         &
               S%nvargp, .FALSE., ICNA, ISTADA, IELING, ISTADG, ISTAEV,        &
               IELVAR, INTVAR, A, GVALS( : , 2 ), FUVALS( : S%lnguvl ),        &
               S%lnguvl, FUVALS( S%lggfx  + 1 : S%lggfx + n ),                 &
               GSCALE_used, ESCALE, GRJAC, GXEQX_used, INTREP,                 &
               ISVGRP, ISTAGV, ITYPEE, ISTAJC, W_ws, W_el, RANGE, KNDOFG )

!  If they are used, update the second derivative approximations

           IF ( .NOT. S%second .AND. .NOT.S%alllin ) THEN
             IF ( use_elders ) THEN

!  Form the differences in the gradients, QGRAD

               QGRAD( : S%ntotin ) =                                           &
                 FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin ) - QGRAD( : S%ntotin )
               IF ( S%firsup ) THEN

!  If a secant method is to be used, scale the initial second derivative
!  matrix for each element so as to satisfy the weak secant condition

                 CALL OTHERS_scaleh_flexible(                                  &
                     .FALSE., n, nel, lfuval, S%nvrels,                        &
                     S%ntotin, inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR,  &
                     IELVAR, ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el,   &
                     H_in, ELDERS( 2, : ), RANGE )
                 S%firsup = .FALSE.
               END IF

!  Update the second derivative approximations using one of four
!  possible secant updating formulae, BFGS, DFP, PSB and SR1.

               CALL OTHERS_secant_flexible(                                    &
                   n, nel, lfuval, S%nvrels, S%ntotin, IELVAR, ISTAEV, INTVAR, &
                   ITYPEE, INTREP, ISTADH, FUVALS, ICALCF, inform%ncalcf, P,   &
                   QGRAD, inform%iskip, S%print_level, S%out, W_el, W_in,      &
                   H_in, ELDERS( 2, : ), RANGE )
             ELSE

!  Form the differences in the gradients, QGRAD

               QGRAD( : S%ntotin ) =                                           &
                 FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin ) - QGRAD( : S%ntotin )
               IF ( S%firsup ) THEN

!  If a secant method is to be used, scale the initial second derivative
!  matrix for each element so as to satisfy the weak secant condition

                 CALL OTHERS_scaleh( .FALSE., n, nel, lfuval, S%nvrels,        &
                     S%ntotin, inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR,  &
                     IELVAR, ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el,   &
                     H_in, RANGE )
                 S%firsup = .FALSE.
               END IF

!  Update the second derivative approximations using one of four
!  possible secant updating formulae, BFGS, DFP, PSB and SR1.

               CALL OTHERS_secant(                                             &
                   n, nel, lfuval, S%nvrels, S%ntotin, IELVAR, ISTAEV, INTVAR, &
                   ITYPEE, INTREP, ISTADH, FUVALS, ICALCF, inform%ncalcf, P,   &
                   QGRAD, S%second_derivatives, inform%iskip,                  &
                   S%print_level, S%out, W_el, W_in, H_in, RANGE )
             END IF
           END IF
           CALL CPU_TIME( tim )
           S%tup = S%tup + tim - S%t

!  Compute the projected gradient and its norm

           CALL LANCELOT_projected_gradient(                                   &
               n, X, FUVALS( S%lggfx + 1 : S%lggfx + n ), VSCALE,              &
               BL, BU, DGRAD, IVAR, inform%nvar, inform%pjgnrm )
           S%nfree = inform%nvar

!  If required, use the users preconditioner

           IF ( S%prcond .AND. inform%nvar > 0 .AND. S%myprec ) THEN
             inform%status = - 10 ; GO TO 800
           END IF
  570      CONTINUE

!  Find the norm of the 'preconditioned' projected gradient. Also,
!  if required, find the diagonal elements of the assembled Hessian

           CALL LANCELOT_norm_proj_grad(                                       &
               n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                      &
               inform%nvar, S%smallh, inform%pjgnrm, S%calcdi, S%dprcnd,       &
               S%myprec, IVAR(:inform%nvar ), ISTADH, ISTAEV, IELVAR, INTVAR,  &
               IELING, DGRAD( : inform%nvar ), Q, GVALS( : , 2 ),              &
               GVALS( : , 3 ), FUVALS( S%ldx + 1 : S%ldx + n ),                &
               GSCALE_used, ESCALE, GRJAC, FUVALS( : S%lnhuvl ), S%lnhuvl,     &
               S%qgnorm, GXEQX_used, INTREP, ISYMMD, ISYMMH, ISTAGV, ISLGRP,   &
               ISVGRP, IVALJR, ITYPEE, W_el, W_in, H_in, RANGE, KNDOFG )

           IF ( S%direct ) S%next = S%step < tenten * epsmch .AND. S%infor == 2
           GO TO 120

!  If the user's computed group function values are inadequate, reduce
!  the trust-region radius

  590      CONTINUE
           S%unsucc = .TRUE.
           S%oldrad = inform%radius 
           inform%radius = control%gamma_decrease * inform%radius
           IF ( S%strctr ) RADII = control%gamma_decrease * RADII

!  If the trust-region radius has shrunk too much, exit. this may indicate a
!  derivative bug or that the user is asking for too much accuracy in the
!  final gradient

           IF ( inform%radius < S%radtol ) THEN
             IF ( S%printe ) WRITE( S%out, 2540 )
             inform%status = 2 ; GO TO 600
           END IF
           GO TO 120

! ---------------------
!
!   End the main loop
!
! ---------------------

  600    CONTINUE

!  Print details of the solution

         IF ( S%printi ) THEN
           inform%iter = MIN0( control%maxit, inform%iter )
           IF ( inform%iter == 0 ) THEN
             WRITE( S%out, 2570 ) inform%iter, inform%aug * S%findmx,          &
               inform%ngeval, inform%pjgnrm, inform%itercg, inform%iskip
           ELSE
             WRITE( S%out,2550 ) inform%iter, inform%aug * S%findmx,           &
               inform%ngeval, inform%pjgnrm, inform%itercg, S%oldrad,          &
               inform%iskip
           END IF
           k = COUNT( X <= BL * ( one + SIGN( S%epstlp, BL ) ) .OR.            &
                      X >= BU * ( one - SIGN( S%epstln, BU ) ) )
           WRITE( S%out, "( /, ' There are ', I0, ' variables and ', I0,       &
          &  ' active bounds')" ) n, k
           IF ( S%printm ) THEN
             WRITE( S%out, 2500 ) X
             WRITE( S%out, 2510 )                                              &
               FUVALS( S%lggfx + 1 : S%lggfx + n ) * S%findmx
           END IF
           WRITE( S%out, "( /, ' Times for Cauchy, systems, products and',     &
          &   ' updates', 0P, 4F8.2 )" ) S%tca, S%tls, S%tmv, S%tup

           IF ( S%xactcp ) THEN
             WRITE( S%out, "( /, ' Exact Cauchy step computed' )" )
           ELSE
             WRITE( S%out, "( /, ' Approximate Cauchy step computed' )" )
           END IF
           IF ( S%steering ) WRITE( S%out, "( ' Feasibility steering used' )" )
           IF ( control%gn_model_after_cauchy ) THEN
             WRITE( S%out, "( ' Gauss-Newton model used' )" )
           ELSE IF ( control%gn_model .AND. S%steering ) THEN
             WRITE( S%out, "( ' Gauss-Newton model used for Cauchy step' )" )
           END IF
           IF ( S%slvbqp )                                                     &
             WRITE( S%out, "( ' Accuarate solution of BQP computed' )" )
           IF ( S%mortor ) WRITE( S%out,                                       &
             "( ' More''-Toraldo projected search technique used' )" )
           IF ( control%linear_solver ==  1 ) WRITE( S%out,                    &
             "( ' Conjugate gradients without preconditioner used' )" )
           IF ( control%linear_solver ==  2 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with diagonal preconditioner used' )" )
           IF ( control%linear_solver ==  3 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with user-supplied',                     &
          &     ' preconditioner used' )" )
           IF ( control%linear_solver ==  4 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with band inverse',                      &
          &     ' preconditioner used' )" )
           IF ( control%linear_solver ==  5 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with Munksgaards',                       &
          &     ' preconditioner used' )" )
           IF ( control%linear_solver ==  6 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with Schnabel-Eskow ',                   &
          &     ' modified Cholesky preconditioner used' )" )
           IF ( control%linear_solver ==  7 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with GMPS modified Cholesky',            &
          &     ' preconditioner used' )" )
           IF ( control%linear_solver ==  8 ) WRITE( S%out,                    &
             "( ' Bandsolver preconditioned C.G.',                             &
          &     ' (semi-bandwidth = ', I0, ') used' )" ) inform%nsemib
           IF ( control%linear_solver ==  9 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with Lin and More`s',                    &
          &     ' preconditioner used (memory = ', I0,')' )" ) S%icfact
           IF ( control%linear_solver == 11 ) WRITE( S%out,                    &
             "( ' Exact matrix factorization used' )" )
           IF ( control%linear_solver == 12 ) WRITE( S%out,                    &
             "( ' Modified matrix factorization used' )" )
           IF ( S%twonrm ) THEN
             WRITE( S%out, "( ' Two-norm trust region used' )" )
           ELSE
             IF ( S%strctr ) THEN
               WRITE( S%out, "( ' Structured infinty-norm trust region used')" )
             ELSE
               WRITE( S%out, "( ' Infinity-norm trust region used' )" )
             END IF
           END IF
           IF ( S%nmhist > 0 ) WRITE( S%out,                                   &
             "( ' Non-monotone descent strategy (history = ', I0,              &
          &     ') used' )" ) S%nmhist
           IF ( S%first_derivatives >= 1 )  WRITE( S%out,                      &
             "( ' Finite-difference approximations to',                        &
          &     ' nonlinear-element gradients used' )" )
           IF ( S%second_derivatives <= 0 ) WRITE( S%out,                      &
             "( ' Exact second derivatives used' )" )
           IF ( S%second_derivatives == 1 ) WRITE( S%out,                      &
             "( ' B.F.G.S. approximation to second derivatives used' )" )
           IF ( S%second_derivatives == 2 ) WRITE( S%out,                      &
             "( ' D.F.P. approximation to second derivatives used' )" )
           IF ( S%second_derivatives == 3 ) WRITE( S%out,                      &
             "( ' P.S.B. approximation to second derivatives used' )" )
           IF ( S%second_derivatives >= 4 ) WRITE( S%out,                      &
             "( ' S.R.1 Approximation to second derivatives used' )" )
           IF ( S%direct ) THEN
             IF ( S%modchl ) THEN
               WRITE( S%out, "( ' No. pos. def. systems = ', I4,               &
              &  ' No. indef. systems = ', I4, /, ' Ratio ( fill-in ) = ',     &
              &  ES11.2 )" ) S%ISYS( 1 ), S%ISYS( 5 ), S%fill
             ELSE
               WRITE( S%out, "( ' Positive definite   = ', I6,                 &
              &  ' indefinite', 12X, '= ', I6, /, ' Singular consistent = ',   &
              &  I6, ' singular inconsistent = ', I6,                          &
              &  /, ' Ratio ( fill-in ) = ', ES11.2 )" ) S%ISYS( : 4 ), S%fill
             END IF
           END IF
         END IF
         GO TO 820

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!     F U N C T I O N   A N D   D E R I V A T I V E   E V A L U A T I O N S
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  ======================
!
!   Internal evaluations
!
!  ======================

  700    CONTINUE

!  Check to see if we are still "alive"

         IF ( control%alive_unit > 0 ) THEN
           INQUIRE( FILE = control%alive_file, EXIST = alive )
           IF ( .NOT. alive ) THEN
             S%inform_status = inform%status
             inform%status = 14
             RETURN
           ELSE
             S%inform_status = inform%status
           END IF
         ELSE
           S%inform_status = inform%status
         END IF

!        WRITE( 6, "( ' internal evaluation ' )" )

!  Further problem information is required

         IF ( inform%status == - 1 .OR. inform%status == - 3 .OR.              &
              inform%status == - 7 ) THEN
           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating element functions ' )" )

!  Evaluate the element function values

           i = 0
           IF ( use_elders ) THEN
             CALL ELFUN_flexible(                                              &
                          FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE,           &
                          ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel, &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1, nel, &
                          1, ELDERS, i )
           ELSE
             CALL ELFUN ( FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE,           &
                          ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel, &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1, 1, i )
           END IF
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 1 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF
         END IF

         IF ( ( inform%status == - 1 .OR. inform%status == - 6 .OR.            &
              ( inform%status == - 5 .AND. .NOT. external_el ) )               &
              .AND. S%getders ) THEN
           ifflag = 2
           IF ( S%second ) ifflag = 3

!  Evaluate the element function derivatives

           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating derivatives of element functions ' )" )
           i = 0
           IF ( use_elders ) THEN
             CALL ELFUN_flexible(                                              &
                          FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE, ISTAEV,   &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel,         &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1,      &
                          nel, ifflag, ELDERS, i )
           ELSE
             CALL ELFUN ( FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE, ISTAEV,   &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel,         &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1,      &
                          ifflag, i )
           END IF
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 1 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF
         END IF

!  Evaluate the group function values

         IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN
           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating group functions ' )" )
           IF ( S%out > 0 .AND. S%print_level >= 100 ) WRITE( S%out,           &
             "( /, ' Group values ', /, ( 6ES12.4 ) )" ) FT( 1 : ng )
           IF ( S%p_type == 2 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 1 ) = FT( i )
             END DO
           END IF
           i = 0
           CALL GROUP ( GVALS , ng, FT, GPVALU, inform%ncalcg, ITYPEG,         &
                        ISTGPA, ICALCG, ng, ng + 1, ng, ng,                    &
                        ISTGPA( ng + 1 ) - 1, .FALSE., i )
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 2 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF
           IF ( S%p_type == 2 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               Y( i ) = GVALS( i, 1 )
               GVALS( i, 1 ) = GVALS( i, 1 ) ** 2
             END DO
!            IF ( S%out > 0  ) WRITE( S%out,                                   &
             IF ( S%out > 0 .AND. S%print_level >= 100 ) WRITE( S%out,         &
               "( /, ' Group function values ', /, ( 6ES12.4 ) )" )            &
                  GVALS( ICALCG(  1 : inform%ncalcg ) , 1 )
           END IF
         END IF

         IF ( inform%status == - 2 .OR.                                        &
              ( inform%status == - 5 .AND. .NOT. external_gr ) ) THEN

!  Evaluate the group function derivatives

           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating derivatives of group functions ' )" )
           IF ( S%p_type == 2 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 2 ) = one
               GVALS( i, 3 ) = zero
             END DO
           END IF
           i = 0
           CALL GROUP ( GVALS , ng, FT, GPVALU, inform%ncalcg, ITYPEG,         &
                        ISTGPA, ICALCG, ng, ng + 1, ng, ng,                    &
                        ISTGPA( ng + 1 ) - 1, .TRUE., i )
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 2 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF

!  group terms psi = ( g(e) )^2, where y = s g(e), g is the group 
!  function and s is the group scaling and their derivatives:
!  psi'(e) = 2 (s g(e) ) s g'(e) and
!  psi''(e) = 2 (s g(e) ) s g''(e) + 2 (sg'(e))^2

           IF ( S%p_type == 2 ) THEN
             DO j = 1,  inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 3 ) = two * Y( i ) * GVALS( i, 3 ) +                  &
                               two * GVALS( i, 2 ) ** 2
               GVALS( i, 2 ) = two * Y( i ) * GVALS( i, 2 )
               GVALS2_GN( i ) = GVALS( i, 2 )
             END DO
           END IF
         END IF

!  Rejoin the iteration

         IF ( inform%status == - 5 .AND. ( external_el .OR. external_gr ) ) THEN
           GO TO 800
         ELSE IF ( S%p_type == 3 ) THEN
           GO TO 810
         ELSE  
           GO TO 20
         END IF

!  =======================================
!
!   Evaluations via reverse communication
!
!  =======================================

 800     CONTINUE

!  Check to see if we are still "alive"

         IF ( control%alive_unit > 0 ) THEN
           INQUIRE( FILE = control%alive_file, EXIST = alive )
           IF ( .NOT. alive ) THEN
             S%inform_status = - inform%status
             inform%status = 14
             RETURN
           ELSE
             S%inform_status = inform%status
           END IF
         ELSE
           S%inform_status = inform%status
         END IF

!  First, make sure that the data is correct for feasibility problems

         IF ( S%p_type == 2 ) THEN
           IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 1 ) = FT( i )
             END DO
           END IF
           IF ( inform%status == - 2 .OR.                                      &
              ( inform%status == - 5 .AND. external_gr ) ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 2 ) = one
               GVALS( i, 3 ) = zero
               GVALS2_GN( i ) = GVALS( i, 2 )
             END DO
           END IF
         END IF

!  Return to the user to obtain problem dependent information

         RETURN
!        IF ( inform%status <= - 1 .AND. inform%status >= - 13 ) RETURN

!  =============================
!
!   R E - E N T R Y   P O I N T
!
!  =============================

 810     CONTINUE
         IF ( inform%status == - 11 ) THEN
           IF ( S%inform_status == - 1 .OR. S%inform_status == - 2 ) THEN
             inform%status = 13 ; RETURN
           ELSE
             GO TO 590
           END IF
         END IF

!  For constrained problems:

         IF ( S%p_type == 3 ) THEN

!  Calculate problem related information

           IF ( inform%status == - 1 ) THEN

!  If there are slack variables, initialize  them to minimize the
!  infeasibility of their associated constraints

             IF ( S%itzero ) THEN
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) >= 3 ) THEN

!  Calculate the constraint value for the inequality constraints. It is
!  assumed that the slack variable occurs last in the list of variables in the
!  linear element

!  Include the contribution from the linear element

!                  ctt = SUM( A( ISTADA( ig ) : ISTADA( ig + 1 ) - 2 ) *       &
!                             X( ICNA( ISTADA( ig ) :                          &
!                                      ISTADA( ig + 1 ) - 2 ) ) ) - B( ig )
                   ctt = - B( ig )
                   DO i = ISTADA( ig ), ISTADA( ig + 1 ) - 2
                     ctt = ctt + A( i ) * X( ICNA( i ) )
                   END DO

!  Include the contributions from the nonlinear elements

!                  ctt = ctt + SUM( ESCALE( ISTADG( ig ) :                     &
!                                           ISTADG( ig + 1 ) - 1 ) *           &
!                                   FUVALS( IELING( ISTADG( ig ) :             &
!                                           ISTADG( ig + 1 ) - 1 ) ) )
                   DO i = ISTADG( ig ), ISTADG( ig + 1 ) - 1
                     ctt = ctt + ESCALE( i ) * FUVALS( IELING( i ) )
                   END DO

!  The slack variable corresponds to a less-than-or-equal-to constraint. Set
!  its value as close as possible to the constraint value

                   j = ISTADA( ig + 1 ) - 1
                   ic = ICNA( j )
                   IF ( KNDOFG( ig ) == 3 ) THEN
                     X( ic ) = MIN( MAX( BL( ic ), - ctt ), BU( ic ) )
  
!  The slack variable corresponds to a greater-than-or-equal-to constraint.
!  Set its value as close as possible to the constraint value
  
                   ELSE
                     X( ic ) = MIN( MAX( BL( ic ), ctt ), BU( ic ) )
                   END IF

!  Compute a suitable scale factor for the slack variable

                   IF ( X( ic ) > one ) THEN
                     VSCALE( ic ) = ten ** ANINT( LOG10( X( ic ) ) )
                   ELSE
                     VSCALE( ic ) = one
                   END IF
                 END IF
               END DO
               S%itzero = .FALSE.
             END IF
           END IF

!  Record the unscaled constraint values in C

           IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN
             WHERE ( GXEQX )
               C = FT
             ELSEWHERE
               C = GVALS( : , 1 )
             END WHERE
!write(6,"( ' /////////////////////////// X = ', /, (6ES12.4))" ) XT
!write(6,"( ' /////////////////////////// C = ', /, (6ES12.4))" ) C

!  record the violation

             inform%cnorm = zero
             DO ig = 1, ng
               IF ( KNDOFG( ig ) > 1 ) THEN
                 inform%cnorm                                                  &
                   = MAX( inform%cnorm, ABS( GSCALE( ig ) * C( ig ) ) )
               END IF
             END DO
!write(6,"( ' c = ', ES12.4, ', x = ', /, ( 5ES12.4 ) )" ) inform%cnorm, X

             IF ( inform%cnorm <= S%etak .AND. S%printt ) WRITE( S%out,        &
               "( ' warning: violation', ES12.4, ' smaller than required ',    &
            &     ES12.4 )" ) inform%cnorm, S%etak

             IF ( S%m > 0 ) THEN
               IF ( inform%iter == 0 ) THEN

!  Print the constraint values on the first iteration

                 IF ( S%printi ) THEN
                   inform%obj = zero ; j = 1
                   IF ( S%printm ) WRITE( S%out, 2120 )
                   DO i = 1, ng
                     IF ( KNDOFG( i ) <= 1 ) THEN
                       IF ( i - 1 >= j .AND. S%printm )                        &
                         WRITE( S%out, 2090 ) ( GNAMES( ig ), ig,              &
                           C( ig ) * GSCALE( ig ), ig = j, i - 1 )
                       j = i + 1
                       IF ( KNDOFG( i ) == 1 )                                 &
                         inform%obj = inform%obj + C( i ) * GSCALE( i )
                     END IF
                   END DO
                   IF ( ng >= j .AND. S%printm )                               &
                     WRITE( S%out, 2090 )  ( GNAMES( ig ), ig,                 &
                       C( ig ) * GSCALE( ig ), ig = j, ng )

!  Print the objective function value on the first iteration

                   IF ( S%nobjgr > 0 ) THEN
                     WRITE( S%out, 2010 ) inform%obj * S%findmx
                   ELSE
                     WRITE( S%out, 2020 )
                   END IF
                 END IF

!  Calculate the constraint norm

                 inform%cnorm = zero
                 DO ig = 1, ng
                   IF ( KNDOFG( ig ) > 1 ) THEN
                     inform%cnorm                                              &
                       = MAX( inform%cnorm, ABS( GSCALE( ig ) * C( ig ) ) )
                   END IF
                 END DO

!  record the violation

                 IF ( S%steering ) THEN
                   C_best = C
!write(6,"( ' /////////////////////////// setting C_best = ', /, (6ES12.4))" ) C
                   S%violation = zero
                   DO ig = 1, ng
                     IF ( KNDOFG( ig ) > 1 ) THEN
                       S%violation                                             &
                         = S%violation + ( GSCALE( ig ) * C( ig ) ) ** 2
                     END IF
                   END DO
                   S%violation = half * S%violation
!write(6,"('KNDOFG', /, (6I12))" ) KNDOFG
!write(6,"('GSCALE', /, (6ES12.4))" ) GSCALE
!write(6,"('C', /, (6ES12.4))" ) C
!write(6,"('C_best', /, (6ES12.4))" ) C_best
!write(6,"(A, ES12.4)") '+*+*+*+*+*+*+*+ ', S%violation

                 END IF
                 IF ( S%out > 0 .AND. S%print_level == 1 ) THEN
                   IF ( inform%iter == 0 ) THEN
                     WRITE( S%out,                                             &
                  &  "( ' Constraint norm           ', ES22.14 )" ) inform%cnorm
                   ELSE
                     WRITE( S%out, 2180 ) inform%mu, S%omegak, inform%cnorm,   &
                       S%etak
                   END IF
                 END IF
               END IF

!  Calculate the terms involving the constraints for the augmented Lagrangian
!  function

               hmuinv = half / inform%mu
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 1 ) THEN
!WRITE(6,"(I2, 4es12.4)") IG, GSCALE( ig ), C( ig ), inform%mu, Y( ig )
                   yiui = GSCALE( ig ) * C( ig ) + inform%mu * Y( ig )
                   GVALS( ig, 1 ) = ( hmuinv * yiui ) * yiui
                 END IF
               END DO
             END IF
           END IF

!  Record the unscaled scalar derivatives of the constraint values in CDASH

           IF ( inform%status == - 2 .OR. inform%status == - 5 ) THEN
             IF ( S%m > 0 ) THEN
               S%violation = zero
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 1 ) THEN
                   S%violation = S%violation + ( GSCALE( ig ) * C( ig ) ) ** 2
                 END IF
               END DO
               S%violation = half * S%violation

               IF ( S%steering ) THEN
!write(6,*) ' =============================== save c is true ============== '
                 IF ( S%save_c ) THEN
                   C_best = C
!write(6,"( ' /////////////////////////// setting C_best = ', /, (6ES12.4))" ) C
!write(6,"('KNDOFG', /, (6I12))" ) KNDOFG
!write(6,"('GSCALE', /, (6ES12.4))" ) GSCALE
!write(6,"('C', /, (6ES12.4))" ) C
!write(6,"('C_best', /, (6ES12.4))" ) C_best
!write(6,"(A, ES12.4)") '+*+*+*+*+*+*+*+ ', S%violation

                 END IF
                 WHERE ( GXEQX )
                   CDASH = one
                   C2DASH = zero
                 ELSEWHERE
                   CDASH = GVALS( : , 2 )
                   C2DASH = GVALS( : , 3 )
                 END WHERE
               END IF

!  Calculate the derivatives of the terms psi = 1/2mu ( g(e) + mu y )^2, 
!  where c = s g(e), g is the group function and s is the group scaling
!  for the augmented Lagrangian function

               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 1 ) THEN
                   scaleg = GSCALE( ig )

!  derivatives of constraint terms with trivial groups (g(e) = e):
!  psi'(e) = 1/mu (s g(e) + mu y) s and
!  psi''(e) = 1/mu s^2

                   IF ( GXEQX( ig ) ) THEN
                     GVALS( ig, 3 ) = scaleg * ( scaleg / inform%mu )
                     GVALS( ig, 2 ) = scaleg * ( FT( ig ) *                    &
                                    ( scaleg / inform%mu ) + Y( ig ) )

!  if a Gauss Newton model is required, psi'(e) = y s

                     IF ( control%gn_model ) THEN
!                      GVALS2_GN( ig ) = scaleg * FT( ig )
                       GVALS2_GN( ig ) = scaleg * Y( ig )
                     ELSE
                       GVALS2_GN( ig ) = GVALS( ig, 2 )
                     END IF

!  derivatives of constraint terms with non-trivial groups:
!  psi'(e) = 1/mu (s g(e) + mu y) s g'(e) and
!  psi''(e) = 1/mu (s g(e) + mu y) s g''(e) + 1/mu (sg'(e))^2

                   ELSE
                     hdash = scaleg * ( Y( ig ) + C( ig ) *                    &
                                      ( scaleg / inform%mu ) )
                     GVALS( ig, 3 ) = hdash * GVALS( ig, 3 ) +                 &
                                ( scaleg * GVALS( ig, 2 ) ) ** 2 / inform%mu

!  if a Gauss Newton model is required, psi'(e) = y s g'(e)

                     IF ( control%gn_model ) THEN
                       GVALS2_GN( ig ) = scaleg * Y( ig ) * GVALS( ig, 2 )
                       GVALS( ig, 2 ) = hdash * GVALS( ig, 2 )
                     ELSE
                       GVALS( ig, 2 ) = hdash * GVALS( ig, 2 )
                       GVALS2_GN( ig ) = GVALS( ig, 2 )
                     END IF
                   END IF
                 END IF
               END DO
             END IF
           END IF

!  For feasibility problems, the group terms psi = ( g(e) )^2, where y = s g(e),
!  g is the group function and s is the group scaling and their derivatives:
!  psi'(e) = 2 (s g(e) ) s g'(e) and
!  psi''(e) = 2 (s g(e) ) s g''(e) + 2 (sg'(e))^2

         ELSE IF ( S%p_type == 2 ) THEN
           IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               Y( i ) = GVALS( i, 1 )
               GVALS( i, 1 ) = GVALS( i, 1 ) ** 2
             END DO
           END IF
           IF ( inform%status == - 2 .OR.                                      &
              ( inform%status == - 5 .AND. external_gr ) ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 3 ) = two * Y( i ) * GVALS( i, 3 ) +                  &
                               two * GVALS( i, 2 ) ** 2
               GVALS( i, 2 ) = two * Y( i ) * GVALS( i, 2 )
               GVALS2_GN( i ) = GVALS( i, 2 )
             END DO
           END IF
         END IF
         S%n_steering = 0
         S%new_major = .FALSE.
         GO TO 20

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    I N N E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 820   CONTINUE

!  Compute the residuals for feasibility problems

       IF ( S%p_type == 2 ) THEN
         WHERE ( GXEQX )
           C = FT
         ELSEWHERE
           C = GVALS( : , 1 )
         END WHERE
       END IF

!  Test whether the maximum allowed number of iterations has been reached

       IF ( inform%status > 3 ) RETURN

!  Print the values of the constraint functions

       IF ( S%p_type == 3 ) THEN
         IF ( S%out > 0 ) THEN
           inform%obj = zero ; j = 1
           IF ( S%printm ) WRITE( S%out, 2120 )
           DO i = 1, ng
             IF ( KNDOFG( i ) == 1 ) THEN
               IF ( i - 1 >= j .AND. S%printm )                                &
                 WRITE( S%out, 2090 ) ( GNAMES( ig ), ig,                      &
                        C( ig ) * GSCALE( ig ), ig = j, i - 1 )
               j = i + 1
               IF ( GXEQX( i ) ) THEN
                 inform%obj = inform%obj + FT( i ) * GSCALE( i )
               ELSE
                 inform%obj = inform%obj + GVALS( i, 1 ) * GSCALE( i )
               END IF
             END IF
           END DO
           IF ( ng >= j .AND. S%printm )                                       &
             WRITE( S%out, 2090 ) ( GNAMES( ig ), ig,                          &
               C( ig ) * GSCALE( ig ), ig = j, ng )

!  Print the objective function value

           IF ( S%printi ) THEN
             IF ( S%nobjgr > 0 ) THEN
               WRITE( S%out, 2010 ) inform%obj * S%findmx
             ELSE
               WRITE( S%out, 2020 )
             END IF
           END IF
         END IF
       END IF

!  Calculate the constraint norm

       IF ( S%p_type == 3 ) THEN
         S%ocnorm = S%cnorm_major ; inform%cnorm = zero
         DO ig = 1, ng
           IF ( KNDOFG( ig ) > 1 ) THEN
             inform%cnorm                                                      &
               = MAX( inform%cnorm, ABS( C( ig ) * GSCALE( ig ) ) )
           END IF
         END DO
         S%cnorm_major = inform%cnorm 
         IF ( inform%status == 1 ) GO TO 900
         IF ( S%printi )                                                       &
           WRITE( S%out, 2060 ) inform%mu, inform%pjgnrm, S%omegak,            &
                                inform%cnorm, S%etak
       ELSE
         GO TO 900
       END IF

!  Test for convergence of the outer iteration

       IF ( ( S%omegak <= control%stopg .OR. inform%pjgnrm <= control%stopg )  &
              .AND. inform%cnorm <= control%stopc ) GO TO 900

!  Test to see if the merit function has become too small

       IF ( inform%aug < control%min_aug ) THEN
         inform%status = 18
         GO TO 900
       END IF

!  Compute the ratio of successive norms of constraint violations. If this
!  ratio is not substantially decreased over NCRIT iterations, exit with the
!  warning that no feasible point can be found

!write(6,*) ' cnorm, 0.99ocnorm ',  inform%cnorm, point99 * S%ocnorm, S%icrit
       IF ( inform%cnorm > point99 * S%ocnorm ) THEN
         S%icrit = S%icrit + 1
         IF ( S%icrit >= S%ncrit ) THEN
           inform%status = 8
           IF ( S%printi ) WRITE( S%out, 2160 ) S%ncrit
           GO TO 900
         END IF
       ELSE
         S%icrit = 0
       END IF

!  Record that an approximate minimizer of the augmented Lagrangian function
!  has been found

       inform%newsol = .TRUE.
       IF ( S%printm ) WRITE( S%out, 2070 ) ( X( i ), i = 1, n )

!  Another iteration will be performed

       inform%status = - 1
       S%new_major = .TRUE.

!  Check to see if the constraint has been sufficiently reduced

       IF ( inform%cnorm < S%etak .AND. inform%mu <= control%mu_tol ) THEN
!write(6,"( ' c = ', ES12.4, ', x = ', /, ( 5ES12.4 ) )" ) inform%cnorm, X
         IF ( S%ocnorm > tenm10 .AND. S%printm ) WRITE( S%out, 2080 )          &
           inform%cnorm / S%ocnorm, S%alphak ** S%betae

!  The constraint norm has been reduced sufficiently. Update the Lagrange
!  multiplier estimates, Y

         IF ( S%m > 0 ) THEN
           WHERE ( KNDOFG( : ng ) > 1 )                                        &
             Y( : ng ) =  Y( : ng ) + C( : ng ) * ( GSCALE( : ng ) / inform%mu )
         END IF
         IF ( S%printi ) WRITE( S%out, 2040 )
         IF ( S%printm ) THEN
           j = 1
           DO i = 1, ng
             IF ( KNDOFG( i ) == 1 ) THEN
               IF ( i - 1 >= j ) WRITE( S%out, 2030 )                          &
                    ( Y( ig ), ig = j, i - 1 )
               j = i + 1
             END IF
           END DO
           IF ( ng >= j ) WRITE( S%out, 2030 ) Y( j : ng )
         END IF

!  Decrease the convergence tolerances

         S%alphak = MIN( inform%mu, S%gamma1 )
         S%etak   = MAX( S%eta_min, S%etak * S%alphak ** S%betae )
         S%omegak = MAX( S%omega_min, S%omegak * S%alphak ** S%betao )

!  Prepare for the next outer iteration

         IF ( S%printi ) WRITE( S%out, 2000 ) inform%mu, S%omegak, S%etak

!  Move variables which are close to their bounds onto the bound

         lgfx = ISTADH( nel + 1 ) - 1
         S%reeval = .FALSE.
         DO i = 1, n
           XT( i ) = X( i )
           IF ( X( i ) /= BL( i ) .AND.                                        &
                X( i ) - BL( i ) <= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BL( i )
           END IF
           IF ( X( i ) /= BU( i ) .AND.                                        &
                X( i ) - BU( i ) >= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BU( i )
           END IF
         END DO
         
!  Reduce the penalty parameter and reset the convergence tolerances
         
       ELSE
         inform%mu = S%tau * inform%mu
         S%alphak = MIN( inform%mu, S%gamma1 )
         S%etak   = MAX( S%eta_min, S%eta0 * S%alphak ** S%alphae )
         S%omegak = MAX( S%omega_min, S%omega0 * S%alphak ** S%alphao )

!  Prepare for the next outer iteration

         IF ( S%printi ) WRITE( S%out, 2150 )
         IF ( S%printi ) WRITE( S%out, 2000 ) inform%mu, S%omegak, S%etak

!  Move variables which are close to their bounds onto the bound

         S%reeval = .FALSE.
         lgfx = ISTADH( nel + 1 ) - 1
         DO i = 1, n
           XT( i ) = X( i )
           IF ( X( i ) /= BL( i ) .AND.                                        &
                X( i ) - BL( i ) <= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BL( i )
           END IF
           IF ( X( i ) /= BU( i ) .AND.                                        &
                X( i ) - BU( i ) >= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BU( i )
           END IF
         END DO

!  If finite-difference gradients are used, use central differences
!  whenever the penalty parameter is small

         IF ( S%first_derivatives >= 1 .AND. inform%mu < epsmch ** 0.25 )      &
           S%first_derivatives = 2
       END IF

!  See if we need to re-evaluate the problem functions

!      S%reeval = .FALSE.
       IF ( S%reeval ) THEN
         IF ( S%printi ) WRITE( S%out, 2190 )
         inform%ngeval = inform%ngeval + 1
         IF ( S%first_derivatives >= 1 ) S%igetfd = 0
         CALL OTHERS_which_variables_changed(                                  &
                     S%unsucc, n, ng, nel, inform%ncalcf, inform%ncalcg,       &
                     ISTAEV, ISTADG, IELING, ICALCF, ICALCG, X, XT,            &
                     ISTAJC, IGCOLJ, LIST_elements, LINK_elem_uses_var )
         X = XT
         S%save_c = .TRUE.
         IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
         GO TO 800
       END IF
       GO TO 10

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    O U T E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 900 CONTINUE

!  Compute the final, unweighted, multiplier estimates

     ipdgen = 0 ; epslam = control%stopc
     IF ( ( inform%status == 0 .OR. inform%status == 18 ) .AND. S%m > 0 ) THEN
       IF ( S%p_type == 3 ) THEN
         DO ig = 1, ng
           IF ( KNDOFG( ig ) >  1 ) THEN
             scaleg = GSCALE( ig )
             Y( ig ) = scaleg * ( Y( ig ) + C( ig ) * ( scaleg / inform%mu ) )
             IF ( ABS( Y( ig ) ) <= epslam ) ipdgen = ipdgen + 1
           END IF
         END DO
       ELSE
         DO ig = 1, ng
           Y( ig ) = GSCALE( ig ) * C( ig ) 
           IF ( ABS( Y( ig ) ) <= epslam ) ipdgen = ipdgen + 1
         END DO
       END IF
     END IF

     IF ( S%printi ) THEN
       lgfx = ISTADH( nel + 1 ) - 1
       ifixd = COUNT( X >= BU - S%epstol * MAX( one, ABS( BU ) ) .OR.          &
                      X <= BL + S%epstol * MAX( one, ABS( BL ) ) .OR.          &
                      BU - BL <= two * epsmch )      
       IF ( control%maxit >= 0 ) iddgen = COUNT(                               &
               ( X >= BU - S%epstol * MAX( one, ABS( BU ) ) .OR.               &
                 X <= BL + S%epstol * MAX( one, ABS( BL ) ) ) .AND.            &
                 ABS( FUVALS( lgfx + 1 : lgfx + n ) ) <= S%epsgrd )
       IF ( S%printi ) THEN
         WRITE( S%out, 2100 )
!        IF ( S%printt ) THEN ; l = n ; ELSE ; l = 2 ; END IF
         IF ( S%full_solution ) THEN ; l = n ; ELSE ; l = 2 ; END IF
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1
             ic = MIN( l, n )
           ELSE
             IF ( ic < n - l ) WRITE( S%out, 2220 )
             ir = MAX( ic + 1, n - ic + 1 )
             ic = n
           END IF
           DO i = ir, ic
             istate = 1
             IF ( X( i ) >= BU( i ) - S%epstol *                               &
                            MAX( one, ABS( BU( i ) ) ) ) istate = 3
             IF ( X( i ) <= BL( i ) + S%epstol *                               &
                            MAX( one, ABS( BL( i ) ) ) ) istate = 2
             IF ( BU( i ) - BL( i ) <= two * epsmch ) istate = 4
             IF ( control%maxit >= 0 ) THEN
               IF ( ( istate == 2 .OR. istate == 3 ) .AND.                     &
                    ABS( FUVALS( lgfx + i ) ) <= S%epsgrd ) istate = 5
             END IF
             IF ( control%maxit >= 0 ) THEN
               WRITE( S%out, 2110 ) VNAMES( i ), i, S%STATE( istate ),         &
                 X( i ), BL( i ), BU( i ), FUVALS( lgfx + i )
             ELSE
               WRITE( S%out, 2210 ) VNAMES( i ), i, S%STATE( istate ),         &
                 X( i ), BL( i ), BU( i )
             END IF
           END DO
         END DO
       END IF
     END IF

!  Compute the objective function value

     IF ( S%p_type == 3 ) THEN
       inform%obj = zero
       DO ig = 1, ng
         IF ( KNDOFG( ig ) == 1 ) THEN
           IF ( GXEQX( ig ) ) THEN
             inform%obj = inform%obj + FT( ig ) * GSCALE( ig )
           ELSE
             inform%obj = inform%obj + GVALS( ig, 1 ) * GSCALE( ig )
           END IF
         END IF
       END DO
     ELSE
       inform%obj = inform%aug
     END IF
     IF ( S%p_type >= 2 ) THEN
       IF ( S%printi )THEN
         WRITE( S%out, 2130 )
         start_p = .FALSE.
!        IF ( S%printt )THEN ; l = ng ; ELSE ; l = 2 ; END IF
         IF ( S%full_solution ) THEN
           l = ng 
         ELSE 
           j = 0
           DO l = 1, ng
             IF ( KNDOFG( l ) > 1 ) j = j + 1
             IF ( j == 2 ) EXIT
           END DO
         END IF
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1
             ic = MIN( l, ng )
           ELSE
             IF ( ic < ng - l .AND. start_p ) WRITE( S%out, 2230 )
             ir = MAX( ic + 1, ng - ic + 1 )
             ic = ng
           END IF
           DO ig = ir, ic
             IF ( KNDOFG( ig ) > 1 ) THEN
               WRITE( S%out, 2140 )                                            &
                 GNAMES( ig ), ig, C( ig ), GSCALE( ig ), Y( ig )
               start_p = .TRUE.
             END IF
           END DO
         END DO
         IF ( S%nobjgr > 0 ) THEN
           WRITE( S%out, 2010 ) inform%obj * S%findmx
         ELSE
           WRITE( S%out, 2020 )
         END IF
         WRITE( S%out, 2170 ) n, S%m, ipdgen, ifixd, iddgen
       END IF
     END IF

     NULLIFY( GXEQX_used, GSCALE_used )

     RETURN

!  Unsuccessful returns

 990 CONTINUE
     inform%status = 12
     inform%alloc_status = alloc_status
     inform%bad_alloc = bad_alloc

!  Non-executable statements

 2000  FORMAT( /, ' Penalty parameter ', ES12.4,                               &
                  ' Required projected gradient norm = ', ES12.4, /,           &
                  '                   ', 12X,                                  &
                  ' Required constraint         norm = ', ES12.4 )             
 2010  FORMAT( /, ' Objective function value  ', ES22.14 )                     
 2020  FORMAT( /, ' There is no objective function ' )                         
 2030  FORMAT( /, ' Multiplier values ', /, ( 5ES12.4 ) )                       
 2040  FORMAT( /, ' ******** Updating multiplier estimates ********** ' )      
 2060  FORMAT( /, ' Penalty parameter       = ', ES12.4, /,                    &
                  ' Projected gradient norm = ', ES12.4,                       &
                  ' Required gradient   norm = ', ES12.4, /,                   &
                  ' Constraint         norm = ', ES12.4,                       &
                  ' Required constraint norm = ', ES12.4 )                     
 2070  FORMAT( /, ' Solution   values ', /, ( 5ES12.4 ) )                      
 2080  FORMAT( /, ' ||c|| / ||c( old )|| = ', ES12.4,                          &
                  ' vs ALPHA ** betae = ', ES12.4 )                            
 2090  FORMAT( ( 4X, A10, I7, 6X, ES22.14 ) )                                  
 2100  FORMAT( /, ' Variable name Number Status     Value',                    &
                  '   Lower bound Upper bound  |  Dual value ', /,             &
                  ' ------------- ------ ------     -----',                    &
                  '   ----------- -----------  |  ----------' )                
 2110  FORMAT( 2X, A10, I7, 4X, A5, 3ES12.4, '  |', ES12.4 )                   
 2120  FORMAT( /, ' Constraint name Number        Value ' )                    
 2130  FORMAT( /, ' Constraint name Number    Value    Scale factor ',         &
                  '| Lagrange multiplier', /,                                  &
                  ' --------------- ------    -----    ----- ------ ',         &
                  '| -------------------' )                                    
 2140  FORMAT( 4X, A10, I7, 2X, 2ES12.4, '  |   ', ES12.4 )                    
 2150  FORMAT( /, ' ***********    Reducing mu    *************** ' )          
 2160  FORMAT( /, ' Constraint violations have not decreased',                 &
                  ' substantially over ', I0, ' major iterations. ', /,        &
                  ' Problem possibly infeasible, terminating run. ' )          
 2170  FORMAT( /, ' There are ', I7, ' variables in total. ', /,               &
                  ' There are ', I7, ' equality constraints. ', /,             &
                  ' Of these  ', I7, ' are primal degenerate. ', /,            &
                  ' There are ', I7, ' variables on their bounds. ', /,        &
                  ' Of these  ', I7, ' are dual degenerate. ' )                
 2180  FORMAT( /, ' Penalty parameter       = ', ES12.4, /,                    &
                  '                           ', 12X,                          &
                  ' Required gradient norm   = ', ES12.4, /,                   &
                  ' Constraint norm         = ', ES12.4,                       &
                  ' Required constraint norm = ', ES12.4 )                     
 2190  FORMAT( /, ' Using the shifted starting point. ' )                      
 2200  FORMAT( /, ' *******    Reducing mu for steering   ******* ' )          
 2210  FORMAT( 2X, A10, I7, 4X, A5, 3ES12.4, '  |      - ' )                   
 2220  FORMAT( '  .               .    .....  ..........  ..........',         &
               '  ..........  |  ..........' )
 2230  FORMAT( '    .               .   ........... ...........',              &
               '  |    ........... ' )                                         
 2500  FORMAT( /, ' X = ', / (  6ES12.4 ) )
 2510  FORMAT( /, ' G = ', / (  6ES12.4 ) )
 2520  FORMAT( /, ' LANCELOT_solve : CPU time limit exceeded' )
 2530  FORMAT( /, ' Change in X = ', / ( 6ES12.4 ) )
 2540  FORMAT( /, ' LANCELOT_solve : trust-region radius too small ' )
 2550  FORMAT( /, ' Iteration number      ', I10,                              &
                  '  Merit function value    = ',  ES19.11, /,                 &
                  ' No. derivative evals  ', I10,                              &
                  '  Projected gradient norm = ',  ES19.11, /,                 &
                  ' C.G. iterations       ', I10,                              &
                  '  Trust-region radius     = ',  ES19.11, /,                 &
                  ' Number of updates skipped  ', I5 )
 2560  FORMAT( /, ' The matrix-vector product used elements', ' marked ',      &
               I5, ' in the following list ', /, ( 20I4 ) )   
 2570  FORMAT( /, ' Iteration number      ', I10,                              &
                  '  Merit function value    = ',  ES19.11, /,                 &
                  ' No. derivative evals  ', I10,                              &
                  '  Projected gradient norm = ',  ES19.11, /,                 &
                  ' C.G. iterations       ', I10, /,                           &
                  ' Number of updates skipped  ', I5 )
!               , '  Correct active set after', I5, ' iteration( s ) ' )

!  End of subroutine LANCELOT_solve_main

     END SUBROUTINE LANCELOT_solve_main

!-*-*-*-*  L A N C E L O T -B- LANCELOT_terminate  S U B R O U T I N E -*-*-*-*

     SUBROUTINE LANCELOT_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_data_type ), INTENT( INOUT ) :: data
     TYPE ( LANCELOT_control_type ), INTENT( IN ) :: control
     TYPE ( LANCELOT_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: alloc_status
     LOGICAL:: alive

     inform%status = 0
     IF ( ASSOCIATED( data%GXEQX_AUG ) ) THEN
       DEALLOCATE( data%GXEQX_AUG, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GXEQX_AUG'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ASSOCIATED( data%GROUP_SCALING ) ) THEN
       DEALLOCATE( data%GROUP_SCALING, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GROUP_SCALING'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ITRANS ) ) THEN
       DEALLOCATE( data%ITRANS, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ITRANS'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%LINK_elem_uses_var ) ) THEN
       DEALLOCATE( data%LINK_elem_uses_var, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%LINK_elem_uses_var'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%WTRANS ) ) THEN
       DEALLOCATE( data%WTRANS, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%WTRANS'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISYMMD ) ) THEN
       DEALLOCATE( data%ISYMMD, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISYMMD'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISWKSP ) ) THEN
       DEALLOCATE( data%ISWKSP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISWKSP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISTAJC ) ) THEN
       DEALLOCATE( data%ISTAJC, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISTAJC'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISTAGV ) ) THEN
       DEALLOCATE( data%ISTAGV, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISTAGV'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISVGRP ) ) THEN
       DEALLOCATE( data%ISVGRP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISVGRP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISLGRP ) ) THEN
       DEALLOCATE( data%ISLGRP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISLGRP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IGCOLJ ) ) THEN
       DEALLOCATE( data%IGCOLJ, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IGCOLJ'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IVALJR ) ) THEN
       DEALLOCATE( data%IVALJR, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IVALJR'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IUSED  ) ) THEN
       DEALLOCATE( data%IUSED , STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IUSED '
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ITYPER ) ) THEN
       DEALLOCATE( data%ITYPER, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ITYPER'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISSWTR ) ) THEN
       DEALLOCATE( data%ISSWTR, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISSWTR'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISSITR ) ) THEN
       DEALLOCATE( data%ISSITR, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISSITR'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISET   ) ) THEN
       DEALLOCATE( data%ISET  , STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISET  '
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISVSET ) ) THEN
       DEALLOCATE( data%ISVSET, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISVSET'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%INVSET ) ) THEN
       DEALLOCATE( data%INVSET, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%INVSET'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%LIST_elements ) ) THEN
       DEALLOCATE( data%LIST_elements, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%LIST_elements'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISYMMH ) ) THEN
       DEALLOCATE( data%ISYMMH, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISYMMH'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IW_asmbl ) ) THEN
       DEALLOCATE( data%IW_asmbl, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IW_asmbl'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%NZ_comp_w ) ) THEN
       DEALLOCATE( data%NZ_comp_w, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%NZ_comp_w'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W_ws ) ) THEN
       DEALLOCATE( data%W_ws, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W_ws'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W_el ) ) THEN
       DEALLOCATE( data%W_el, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W_el'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W_in ) ) THEN
       DEALLOCATE( data%W_in, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W_in'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%H_el ) ) THEN
       DEALLOCATE( data%H_el, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%H_el'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%H_in ) ) THEN
       DEALLOCATE( data%H_in, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%H_in'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%DIAG ) ) THEN
       DEALLOCATE( data%DIAG, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%DIAG'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%OFFDIA ) ) THEN
       DEALLOCATE( data%OFFDIA, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%OFFDIA'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IW ) ) THEN
       DEALLOCATE( data%IW, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IW'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IKEEP ) ) THEN
       DEALLOCATE( data%IKEEP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IKEEP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IW1 ) ) THEN
       DEALLOCATE( data%IW1, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IW1'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IVUSE ) ) THEN
       DEALLOCATE( data%IVUSE, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IVUSE'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%H_col_ptr ) ) THEN
       DEALLOCATE( data%H_col_ptr, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%H_col_ptr'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%L_col_ptr ) ) THEN
       DEALLOCATE( data%L_col_ptr, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%L_col_ptr'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W ) ) THEN
       DEALLOCATE( data%W, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W1 ) ) THEN
       DEALLOCATE( data%W1, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W1'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%RHS ) ) THEN
       DEALLOCATE( data%RHS, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%RHS'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%RHS2 ) ) THEN
       DEALLOCATE( data%RHS2, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%RHS2'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%XCP ) ) THEN
       DEALLOCATE( data%XCP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%XCP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

    IF ( ALLOCATED( data%CDASH ) ) THEN
       DEALLOCATE( data%CDASH, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%CDASH'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

    IF ( ALLOCATED( data%C2DASH ) ) THEN
       DEALLOCATE( data%C2DASH, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%C2DASH'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

!    IF ( ALLOCATED( data%XCP_steering ) ) THEN
!      DEALLOCATE( data%XCP_steering, STAT = alloc_status )
!      IF ( alloc_status /= 0 ) THEN
!        inform%status = 12
!        inform%alloc_status = alloc_status
!        inform%bad_alloc = 'data%XCP'
!        WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
!      END IF
!    END IF

     IF ( ALLOCATED( data%X0 ) ) THEN
       DEALLOCATE( data%X0, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%X0'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%BND ) ) THEN
       DEALLOCATE( data%BND, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%BND'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%BND_radius ) ) THEN
       DEALLOCATE( data%BND_radius, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%BND_radius'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%BREAKP ) ) THEN
       DEALLOCATE( data%BREAKP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%BREAKP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%DELTAX ) ) THEN
       DEALLOCATE( data%DELTAX, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%DELTAX'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%GRAD ) ) THEN
       DEALLOCATE( data%GRAD, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GRAD'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%GRJAC ) ) THEN
       DEALLOCATE( data%GRJAC, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GRJAC'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%INDEX ) ) THEN
       DEALLOCATE( data%INDEX, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%INDEX'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%QGRAD ) ) THEN
       DEALLOCATE( data%QGRAD, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%QGRAD'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IFREE ) ) THEN
       DEALLOCATE( data%IFREE, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IFREE'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%RADII ) ) THEN
       DEALLOCATE( data%RADII, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%RADII'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%GX0 ) ) THEN
       DEALLOCATE( data%GX0, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GX0'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IFREEC ) ) THEN
       DEALLOCATE( data%IFREEC, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IFREEC'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%SCU_matrix%BD_row ) ) THEN
       DEALLOCATE( data%SCU_matrix%BD_row, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%SCU_matrix%BD_row'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%SCU_matrix%BD_val ) ) THEN
       DEALLOCATE( data%SCU_matrix%BD_val, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%SCU_matrix%BD_val'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%SCU_matrix%BD_col_start ) ) THEN
       DEALLOCATE( data%SCU_matrix%BD_col_start, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%SCU_matrix%BD_col_start'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%FUVALS_temp ) ) THEN
       DEALLOCATE( data%FUVALS_temp, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%FUVALS_temp'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%INNONZ ) ) THEN
       DEALLOCATE( data%INNONZ, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%INNONZ'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%GV_old ) ) THEN
       DEALLOCATE( data%GV_old, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GV_old'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF
 
     IF ( ALLOCATED( data%P ) ) THEN
       DEALLOCATE( data%P, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%P'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%P2 ) ) THEN
       DEALLOCATE( data%P2, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%P2'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%G ) ) THEN
       DEALLOCATE( data%G, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%G'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%BREAKP ) ) THEN
       DEALLOCATE( data%BREAKP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%BREAKP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%GRAD ) ) THEN
       DEALLOCATE( data%GRAD, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GRAD'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ROW_start ) ) THEN
       DEALLOCATE( data%ROW_start, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ROW_start'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%POS_in_H ) ) THEN
       DEALLOCATE( data%POS_in_H, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%POS_in_H'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%USED ) ) THEN
       DEALLOCATE( data%USED, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%USED'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%FILLED ) ) THEN
       DEALLOCATE( data%FILLED, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%FILLED'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%matrix%row ) ) THEN
       DEALLOCATE( data%matrix%row, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%matrix%row'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%matrix%col ) ) THEN
       DEALLOCATE( data%matrix%col, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%matrix%col'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%matrix%val ) ) THEN
       DEALLOCATE( data%matrix%val, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%matrix%val'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     CALL SCU_terminate( data%SCU_data, inform%status, inform%SCU_info )
     CALL SILS_finalize( data%SILS_data, control%SILS_cntl, inform%status )

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

!  Non-executable statement

 2990  FORMAT( ' ** Message from -LANCELOT_terminate-', /,                     &
               ' Deallocation error (status = ', I6, ') for ', A24 )

!  End of subroutine LANCELOT_terminate

     END SUBROUTINE LANCELOT_terminate

!-*-*-  L A N C E L O T  -B-  LANCELOT_form_gradients  S U B R O U T I N E -*-*

     SUBROUTINE LANCELOT_form_gradients(                                       &
                       n , ng, nel   , ntotel, nvrels, nnza  , nvargp,         &
                       firstg, ICNA  , ISTADA, IELING, ISTADG, ISTAEV,         &
                       IELVAR, INTVAR, A     , GVALS2, GUVALS, lguval,         &
                       GRAD  , GSCALE, ESCALE, GRJAC , GXEQX , INTREP,         &
                       ISVGRP, ISTAGV, ITYPEE, ISTAJC, GRAD_el, W_el ,         &
                       RANGE , KNDOFG )

!  Calculate the the gradient, GRAD, of the objective function and the
!  Jacobian matrix of gradients, GRJAC, of each group

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN    ) :: n , ng, nel   , ntotel, nnza, nvargp
     INTEGER, INTENT( IN    ) :: nvrels, lguval
     LOGICAL, INTENT( IN    ) :: firstg
     INTEGER, INTENT( IN    ), DIMENSION( ng  + 1 ) :: ISTADA, ISTADG
     INTEGER, INTENT( IN    ), DIMENSION( nel + 1 ) :: ISTAEV, INTVAR
     INTEGER, INTENT( IN    ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN    ), DIMENSION( nnza    ) :: ICNA
     INTEGER, INTENT( IN    ), DIMENSION( ntotel  ) :: IELING
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( nnza ) :: A
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: GVALS2
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( lguval ) :: GUVALS
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ntotel ) :: ESCALE
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GRAD
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nvargp ) :: GRJAC
     LOGICAL, INTENT( IN ), DIMENSION( ng  ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISTAJC
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: GRAD_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig, ii, k, ig1, j, jj, l , ll
     INTEGER :: nin   , nvarel, nelow , nelup, istrgv, iendgv
     REAL ( KIND = wp ) :: gi, scalee
     LOGICAL :: nontrv

!  Initialize the gradient as zero

     GRAD = zero

!  Consider the IG-th group

     DO ig = 1, ng
       IF ( PRESENT( KNDOFG ) ) THEN
         IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
       ig1 = ig + 1
       istrgv = ISTAGV( ig ) ; iendgv = ISTAGV( ig1 ) - 1
       nelow = ISTADG( ig ) ; nelup = ISTADG( ig1 ) - 1
       nontrv = .NOT. GXEQX( ig )

!  Compute the first derivative of the group

       IF ( nontrv ) THEN
         gi = GSCALE( ig ) * GVALS2( ig )
       ELSE
         gi = GSCALE( ig )
       END IF

!  This is the first gradient evaluation or the group has nonlinear elements

       IF ( firstg .OR. nelow <= nelup ) THEN
         GRAD_el( ISVGRP( istrgv : iendgv ) ) = zero

!  Loop over the group's nonlinear elements

         DO ii = nelow, nelup
           iel = IELING( ii )
           k = INTVAR( iel ) ; l = ISTAEV( iel )
           nvarel = ISTAEV( iel + 1 ) - l
           scalee = ESCALE( ii )
           IF ( INTREP( iel ) ) THEN

!  The IEL-th element has an internal representation

             nin = INTVAR( iel + 1 ) - k
             CALL RANGE ( iel, .TRUE., GUVALS( k : k + nin - 1 ),              &
                          W_el( : nvarel ), nvarel, nin, ITYPEE( iel ),        &
                          nin, nvarel )
!DIR$ IVDEP
             DO i = 1, nvarel
               j = IELVAR( l )
               GRAD_el( j ) = GRAD_el( j ) + scalee * W_el( i )
               l = l + 1
             END DO
           ELSE

!  The IEL-th element has no internal representation

!DIR$ IVDEP
             DO i = 1, nvarel
               j = IELVAR( l )
               GRAD_el( j ) = GRAD_el( j ) + scalee * GUVALS( k )
               k = k + 1
               l = l + 1
             END DO
           END IF
         END DO

!  Include the contribution from the linear element

!DIR$ IVDEP
         DO k = ISTADA( ig ), ISTADA( ig1 ) - 1
           GRAD_el( ICNA( k ) ) = GRAD_el( ICNA( k ) ) + A( k )
         END DO

!  Find the gradient of the group

         IF ( nontrv ) THEN

!  The group is non-trivial

!DIR$ IVDEP
           DO i = istrgv, iendgv
             ll = ISVGRP( i )
             GRAD( ll ) = GRAD( ll ) + gi * GRAD_el( ll )

!  As the group is non-trivial, also store the nonzero entries of the
!  gradient of the function in GRJAC

             jj = ISTAJC( ll )
             GRJAC( jj ) = GRAD_el( ll )

!  Increment the address for the next nonzero in the column of
!  the Jacobian for variable LL

             ISTAJC( ll ) = jj + 1
           END DO
         ELSE

!  The group is trivial

!DIR$ IVDEP
           DO i = istrgv, iendgv
             ll = ISVGRP( i )
             GRAD( ll ) = GRAD( ll ) + gi * GRAD_el( ll )
           END DO
         END IF

!  This is not the first gradient evaluation and there is only a linear element

       ELSE

!  Add the gradient of the linear element to the overall gradient

!DIR$ IVDEP
         DO k = ISTADA( ig ), ISTADA( ig1 ) - 1
           GRAD( ICNA( k ) ) = GRAD( ICNA( k ) ) + gi * A( k )
         END DO

!  The group is non-trivial; increment the starting addresses for
!  the groups used by each variable in the (unchanged) linear
!  element to avoid resetting the nonzeros in the Jacobian

         IF ( nontrv ) THEN
!DIR$ IVDEP
           DO i = istrgv, iendgv
             ISTAJC( ISVGRP( i ) ) = ISTAJC( ISVGRP( i ) ) + 1
           END DO
         END IF
       END IF
     END DO

!  Reset the starting addresses for the lists of groups using each variable to
!  their values on entry

     DO i = n, 2, - 1
       ISTAJC( i ) = ISTAJC( i - 1 )
     END DO
     ISTAJC( 1 ) = 1

     RETURN

!  End of subroutine LANCELOT_form_gradients

     END SUBROUTINE LANCELOT_form_gradients

!-*-  L A N C E L O T  -B-  LANCELOT_projected_gradient  S U B R O U T I N E -*-

     SUBROUTINE LANCELOT_projected_gradient( n, X, G, XSCALE, BL, BU, GRAD,    &
                                             IVAR, nvar, pjgnrm )

!  Compute the projection of the gradient into the feasible box and its norm

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) ::  n
     INTEGER, INTENT( OUT ) ::  nvar
     REAL ( KIND = wp ), INTENT( OUT ) :: pjgnrm
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: IVAR
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: X, G, XSCALE
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: BL, BU
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GRAD

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: gi, epsmch

     epsmch = EPSILON( one )

     nvar = 0
     pjgnrm = zero
     DO i = 1, n
       gi = G( i ) * XSCALE( i )
       IF ( gi == zero ) CYCLE

!  Compute the projection of the gradient within the box

       IF ( gi < zero ) THEN
         gi = - MIN( ABS( BU( i ) - X( i ) ), - gi )
       ELSE
         gi = MIN( ABS( BL( i ) - X( i ) ), gi )
       END IF

!  Record the nonzero components of the Cauchy direction in GRAD

       IF ( ABS( gi ) > epsmch ) THEN
         nvar = nvar + 1
         pjgnrm = MAX( pjgnrm, ABS( gi ) )
         ivar( nvar ) = i
         GRAD( nvar ) = gi
       END IF
     END DO

     RETURN

!  End of LANCELOT_projected_gradient

     END SUBROUTINE LANCELOT_projected_gradient

!-*-*-*- L A N C E L O T -B- LANCELOT_norm_proj_grad S U B R O U T I N E -*-*-*-

     SUBROUTINE LANCELOT_norm_proj_grad(                                       &
                        n , ng, nel   , ntotel, nvrels,                        &
                        nvargp, nvar  , smallh, pjgnrm, calcdi, dprcnd,        &
                        myprec, IVAR  , ISTADH, ISTAEV, IELVAR, INTVAR,        &
                        IELING, DGRAD , Q     , GVALS2, GVALS3, DIAG  ,        &
                        GSCALE, ESCALE, GRJAC , HUVALS, lnhuvl, qgnorm,        &
                        GXEQX , INTREP, ISYMMD, ISYMMH, ISTAGV, ISLGRP,        &
                        ISVGRP, IVALJR, ITYPEE, W_el  , W_in  , H_in  ,        &
                        RANGE , KNDOFG )

!  Find the norm of the projected gradient, scaled if desired.
!  If required, also find the diagonal elements of the Hessian matrix

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n , ng, nel   , ntotel, nvar
     INTEGER, INTENT( IN ) :: nvrels, nvargp, lnhuvl
     REAL ( KIND = wp ), INTENT( IN    ) :: smallh, pjgnrm
     REAL ( KIND = wp ), INTENT( OUT   ) :: qgnorm
     LOGICAL, INTENT( IN ) :: calcdi, dprcnd, myprec
     INTEGER, INTENT( IN ), DIMENSION( nvar    ) :: IVAR
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTADH, ISTAEV, INTVAR
     INTEGER, INTENT( IN ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( ntotel  ) :: IELING
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: Q
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvar ) :: DGRAD
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS2, GVALS3, GSCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotel ) :: ESCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvargp ) :: GRJAC
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lnhuvl ) :: HUVALS
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: DIAG
     LOGICAL, INTENT( IN ), DIMENSION( ng  ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP

     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISYMMD
     INTEGER, INTENT( IN ), DIMENSION( : , : ) :: ISYMMH
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISLGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IVALJR
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_in

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig , j , irow  , ijhess, k , kk, ll
     INTEGER :: iell  , nin    , nvarel, jcol  , ielhst
     REAL ( KIND = wp ) :: gdash, g2dash, temp
     
     IF ( myprec ) THEN
!      qgnorm = SQRT( DOT_PRODUCT( DGRAD( : nvar ), Q( IVAR( : nvar ) ) ) )
       qgnorm = zero
       DO i = 1, nvar
         qgnorm = qgnorm + DGRAD( i ) * Q( IVAR( i ) )
       END DO
       qgnorm = SQRT( qgnorm )
     ELSE
       IF ( calcdi ) THEN

!  Obtain the diagonal elements of the Hessian.
!  Initialize the diagonals as zero

         DIAG = zero ; W_el = zero

!  Obtain the contributions from the second derivatives of the elements

         DO iell = 1, ntotel
           iel = IELING( iell )
           ig = ISLGRP( iell )
           IF ( PRESENT( KNDOFG ) ) THEN
             IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
           nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
           ll = ISTAEV( iel )
           IF ( GXEQX( ig ) ) THEN
             gdash = ESCALE( iell ) * GSCALE( ig )
           ELSE
             gdash = ESCALE( iell ) * GSCALE( ig ) * GVALS2( ig )
           END IF
           IF ( INTREP( iel ) ) THEN
             nin = INTVAR( iel + 1 ) - INTVAR( iel )
             DO kk = 1, nvarel

!  The IEL-th element Hessian has an internal representation.
!  Set W_el as the KK-th column of the identity matrix

               W_el( kk ) = one

!  Gather W_el into its internal variables, W_in

               CALL RANGE ( iel, .FALSE., W_el, W_in, nvarel, nin,             &
                            ITYPEE( iel ), nvarel, nin )
               W_el( kk ) = zero

!  Multiply the internal variables by the element Hessian.
!  Consider the first column of the Hessian

               ielhst = ISTADH( iel )
               H_in( : nin ) = W_in( 1 ) * HUVALS( ISYMMH( 1, : nin ) + ielhst )

!  Now consider the remaining columns of the Hessian

               DO jcol = 2, nin
                 H_in( : nin ) = H_in( : nin ) + W_in( jcol ) *                &
                   HUVALS( ISYMMH( jcol, : nin ) + ielhst )
               END DO

!  Add the KK-th diagonal of the IEL-th element Hessian

               j = IELVAR( ll )
               ll = ll + 1
!              DIAG( j ) =                                                     &
!                DIAG( j ) + gdash * DOT_PRODUCT( W_in( : nin ), H_in( : nin ) )
               temp = zero
               DO i = 1, nin
                  temp = temp + W_in( i ) * H_in( i )
               END DO
               DIAG( j ) = DIAG( j ) + gdash * temp
             END DO
           ELSE

!  The IEL-th element Hessian has no internal representation

             ielhst = ISTADH( iel )
!DIR$ IVDEP
             DO irow = 1, nvarel
               ijhess = ISYMMD( irow ) + ielhst
               j = IELVAR( ll )
               ll = ll + 1
               DIAG( j ) = DIAG( j ) + gdash * HUVALS( ijhess )
             END DO
           END IF
         END DO

!  If the group is non-trivial, add on rank-one first order terms

         DO ig = 1, ng
           IF ( PRESENT( KNDOFG ) ) THEN
             IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
           IF ( .NOT. GXEQX( ig ) ) THEN
             g2dash = GSCALE( ig ) * GVALS3( ig )
!DIR$ IVDEP
             DO k = ISTAGV( ig ), ISTAGV( ig + 1 ) - 1
               DIAG( ISVGRP( k ) ) = DIAG( ISVGRP( k ) ) + g2dash *            &
                 GRJAC( IVALJR( k ) ) ** 2
             END DO
           END IF
         END DO

!  Take the absolute values of all the diagonal entries, ensuring that all
!  entries are larger than the tolerance SMALLH

         DIAG( : n ) = MAX( smallh, ABS( DIAG( : n ) ) )
       END IF

!  Use the diagonals to calculate a scaled norm of the gradient

       IF ( dprcnd ) THEN
!        qgnorm = SQRT( DOT_PRODUCT( DGRAD( : nvar ),                          &
!          ( DGRAD( : nvar ) / DIAG( IVAR( : nvar ) ) ) ) )
         qgnorm = zero
         DO i = 1, nvar
           qgnorm = qgnorm + ( DGRAD( i ) ** 2 ) /  DIAG( IVAR( i ) )
         END DO
         qgnorm = SQRT( qgnorm )
       ELSE
         qgnorm = pjgnrm
       END IF
     END IF

     RETURN

!  End of subroutine LANCELOT_norm_proj_grad

     END SUBROUTINE LANCELOT_norm_proj_grad

!-*-*-*-*  L A N C E L O T  -B-   LANCELOT_norm_diff   F U N C T I O N -*-*-*-*

     FUNCTION LANCELOT_norm_diff( n, X, Y, twonrm, RSCALE, scaled )
     REAL ( KIND = wp ) :: LANCELOT_norm_diff

!  Compute the scaled (or unscaled) two (or infinity) norm distance
!  between the vectors X and Y

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     LOGICAL, INTENT( IN ) :: twonrm, scaled
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, Y
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: RSCALE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i

     IF ( scaled ) THEN

!  Compute the scaled two-norm distance between X and Y

       IF ( twonrm ) THEN
!        LANCELOT_norm_diff = SQRT( SUM( ( ( X - Y ) / RSCALE ) ** 2 ) )
         LANCELOT_norm_diff = zero
         DO i = 1, n
           LANCELOT_norm_diff =                                                &
             LANCELOT_norm_diff + ( ( X( i ) - Y( i ) ) / RSCALE( i ) ) ** 2 
         END DO
         LANCELOT_norm_diff = SQRT( LANCELOT_norm_diff )

!  Compute the scaled infinity-norm distance between X and Y

       ELSE
         LANCELOT_norm_diff = MAXVAL( ABS( ( X - Y ) / RSCALE ) )
       END IF
     ELSE

!  Compute the two-norm distance between X and Y

       IF ( twonrm ) THEN
!        LANCELOT_norm_diff = SQRT( SUM( ( X - Y ) ** 2 ) )
         LANCELOT_norm_diff = zero
         DO i = 1, n
            LANCELOT_norm_diff = LANCELOT_norm_diff + ( X( i ) - Y( i ) ) ** 2
         END DO
         LANCELOT_norm_diff = SQRT( LANCELOT_norm_diff )

!  Compute the infinity-norm distance between X and Y

       ELSE
         LANCELOT_norm_diff = MAXVAL( ABS( X - Y ) )
       END IF
     END IF
     RETURN

!  End of function LANCELOT_norm_diff

     END FUNCTION LANCELOT_norm_diff

!-*-  L A N C E L O T -B- LANCELOT_JTJ_times_vector  S U B R O U T I N E  -*-

     SUBROUTINE LANCELOT_JTJ_times_vector(                                     &
                      n, ng, nvargp, nfree, nvar1, nvar2 , nnonnz,             &
                      IVAR, INONNZ, P, Q, CDASH, GRJAC, GSCALE, GXEQX,         &
                      densep, IGCOLJ, ISVGRP, ISTAGV, IVALJR,                  &
                      ISTAJC, IUSED, NZ_comp_w, AP, KNDOFG )

!  Evaluate Q, the product of J(trans) * CDASH ** 2 * J from a groups 
!  partially separable function with the vector P

!  The nonzero components of P have indices IVAR( I ), I = NVAR1, ..., NVAR2.
!  The nonzero components of the product Q have indices INNONZ( I ),
!  I = 1, ..., NNONNZ

!  The elements of the array IUSED must be set to zero on entry; they will have
!  been reset to zero on exit

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, ng, nfree, nvar1 , nvar2, nvargp
     INTEGER, INTENT( INOUT ) :: nnonnz
     LOGICAL, INTENT( IN ) :: densep
     INTEGER, INTENT( IN ), DIMENSION( n ) :: IVAR
     INTEGER, INTENT( INOUT ), DIMENSION( n ) :: INONNZ
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: P
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: CDASH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvargp ) :: GRJAC
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: Q
     LOGICAL, INTENT( IN ), DIMENSION( ng ) :: GXEQX
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IGCOLJ
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IVALJR
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAJC
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IUSED 
     INTEGER, INTENT( OUT ), DIMENSION( : ) :: NZ_comp_w
     INTEGER, INTENT( IN ), DIMENSION( ng ) :: KNDOFG
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: AP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ig, j, k, l, nnz_comp_w
     REAL ( KIND = wp ) :: pi

!  If the IG-th group is non-trivial, form the product of p with the
!  sum of rank-one first order terms, J(trans) * CDASH ** 2 * J. 
!  J is stored by both rows and columns. For maximum efficiency, the
!  product is formed in different ways if p is sparse or dense

!  -----------------  Case 1. P is not sparse -----------------------

     IF ( densep ) THEN

!  Initialize AP and Q as zero

       AP( : ng ) = zero ; Q = zero

!  Form the matrix-vector product AP = J * p, using the column-wise
!  storage of J

       DO l = nvar1, nvar2
         i = IVAR( l )
         pi = P( i )
!DIR$ IVDEP
         DO k = ISTAJC( i ), ISTAJC( i + 1 ) - 1
           ig = IGCOLJ( k )
           IF ( KNDOFG( ig ) > 1 ) AP( ig ) = AP( ig ) + pi * GRJAC( k )
         END DO
       END DO

!  Multiply AP by the diagonal matrix CDASH ** 2

       DO ig = 1, ng
         IF ( KNDOFG( ig ) > 1 ) THEN
           IF ( GXEQX( ig ) ) THEN
             AP( ig ) = AP( ig ) * GSCALE( ig ) ** 2 
           ELSE
             AP( ig ) = AP( ig ) * ( GSCALE( ig ) * CDASH( ig ) ) ** 2
           END IF
         END IF
       END DO

!  Form the matrix-vector product Q = J(trans) * AP, once again using the
!  column-wise storage of J

       nnonnz = 0
       DO l = 1, nfree
         i = IVAR( l )
         pi = zero
         DO k = ISTAJC( i ), ISTAJC( i + 1 ) - 1
           ig = IGCOLJ( k )
           IF ( KNDOFG( ig ) > 1 ) pi = pi + AP( ig ) * GRJAC( k )
         END DO
         Q( i ) = pi
       END DO

!  ------------------- Case 2. P is sparse --------------------------

     ELSE
       nnz_comp_w = 0
       Q( IVAR( : nfree ) ) = zero

!  Form the matrix-vector product AP = J * p, using the column-wise
!  storage of J. Keep track of the nonzero components of W in NZ_comp_w.
!  Only store components corresponding to non trivial groups

       DO j = nvar1, nvar2
         i = IVAR( j ); pi = P( i )
         DO k = ISTAJC( i ), ISTAJC( i + 1 ) - 1
           ig = IGCOLJ( k )
           IF ( KNDOFG( ig ) > 1 ) THEN
             IF ( IUSED( ig ) == 0 ) THEN
               AP( ig ) = pi * GRJAC( k )
               IUSED( ig ) = 1
               nnz_comp_w = nnz_comp_w + 1
               NZ_comp_w( nnz_comp_w ) = ig
             ELSE
               AP( ig ) = AP( ig ) + pi * GRJAC( k )
             END IF
           END IF
         END DO
       END DO

!  Reset IUSED to zero

       IUSED( NZ_comp_w( : nnz_comp_w ) ) = 0

!  Form the matrix-vector product Q = J( TRANS ) * CDASH ** 2 * AP, using 
!  the row-wise storage of J

       nnonnz = 0
       DO j = 1, nnz_comp_w
         ig = NZ_comp_w( j )
         IF ( KNDOFG( ig ) >= 2 ) THEN
           pi = AP( ig ) * ( GSCALE( ig ) * CDASH( ig ) ) ** 2 
           DO k = ISTAGV( ig ), ISTAGV( ig + 1 ) - 1
             l = ISVGRP( k )

!  If Q has a nonzero in position l, store its index in INONNZ

             IF ( IUSED( l ) == 0 ) THEN
               Q( l ) = pi * GRJAC( IVALJR( k ) )
               IUSED( l ) = 1
               nnonnz = nnonnz + 1
               INONNZ( nnonnz ) = l
             ELSE
               Q( l ) = Q( l ) + pi * GRJAC( IVALJR( k ) )
             END IF
           END DO
         END IF
       END DO
       IUSED( INONNZ( : nnonnz ) ) = 0
     END IF

     RETURN

!  End of subroutine JTJhessian_times_vector

     END SUBROUTINE LANCELOT_JTJ_times_vector

!-*-*-*-*-*-  L A N C E L O T -B- LANCELOT_JTc  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE LANCELOT_JTc(                                                  &
                      n, ng, nvargp, JTc, C, CDASH, GRJAC, GSCALE, GXEQX,      &
                      ISVGRP, ISTAGV, IVALJR, KNDOFG )

!  Evaluate JTc, the product of J(trans) * CDASH * c from a groups 
!  partially separable function

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN    ) :: n, ng, nvargp
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: C, CDASH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvargp ) :: GRJAC
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: JTc
     LOGICAL, INTENT( IN ), DIMENSION( ng ) :: GXEQX
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IVALJR
     INTEGER, INTENT( IN ), DIMENSION( ng ) :: KNDOFG

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ig, k
     REAL ( KIND = wp ) :: pi

!  If the IG-th group is non-trivial, form the product of p with the
!  sum of rank-one first order terms, J(trans) * CDASH ** 2 * J. 

!  Initialize JTc as zero

     JTc = zero

!  Multiply C by the diagonal matrix CDASH

     DO ig = 1, ng
       IF ( KNDOFG( ig ) >= 2 ) THEN
         IF ( GXEQX( ig ) ) THEN
           pi = C( ig ) * GSCALE( ig ) ** 2 
         ELSE
           pi = C( ig ) * CDASH( ig ) * ( GSCALE( ig ) ) ** 2
         END IF

!  Form the matrix-vector product JTc = J( TRANS ) * CDASH * C, using
!  the row-wise storage of J

         DO k = ISTAGV( ig ), ISTAGV( ig + 1 ) - 1
           i = ISVGRP( k )
           JTc( i ) = JTc( i ) + pi * GRJAC( IVALJR( k ) )
         END DO
       END IF
     END DO

     RETURN

!  End of subroutine JTc

     END SUBROUTINE LANCELOT_JTc

!-*-*-*-*-*-  L A N C E L O T -B- LANCELOT_violation  F U N C T I O N  -*-*-*-*-

     FUNCTION LANCELOT_violation(                                              &
                      n, ng, nvargp, C, CDASH, GRJAC, P, GSCALE, GXEQX,        &
                      IGCOLJ, ISTAJC, KNDOFG, AP, ss2 )
     REAL ( KIND = wp ) :: LANCELOT_violation

!  Evaluate  the "violation" 1/2 || C + CDASH * J p ||^2 from a groups 
!  partially separable function

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN    ) :: n, ng, nvargp
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: C, CDASH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: P
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvargp ) :: GRJAC
     LOGICAL, INTENT( IN ), DIMENSION( ng ) :: GXEQX
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IGCOLJ
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAJC
     INTEGER, INTENT( IN ), DIMENSION( ng ) :: KNDOFG
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: AP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ig, k
     REAL ( KIND = wp ) :: pi, ss, ss2

!  Form the matrix-vector product AP = J * p, using the column-wise
!  storage of J

     AP( : ng ) = zero
     DO i = 1, n
       pi = P( i )
!DIR$ IVDEP
       DO k = ISTAJC( i ), ISTAJC( i + 1 ) - 1
         ig = IGCOLJ( k )
         IF ( KNDOFG( ig ) > 1 ) AP( ig ) = AP( ig ) + pi * GRJAC( k )
       END DO
     END DO

!  Multiply AP by the diagonal matrix CDASH and add to C

     ss = zero ; ss2 = zero
     DO ig = 1, ng
       IF ( KNDOFG( ig ) > 1 ) THEN
         IF ( GXEQX( ig ) ) THEN
!          ss = ss + ( C( ig ) + AP( ig ) * GSCALE( ig ) ) ** 2
           ss = ss + ( GSCALE( ig ) * ( C( ig ) + AP( ig ) ) ) ** 2
           ss2 = ss2 + ( GSCALE( ig ) * C( ig ) ) ** 2
         ELSE
!          ss = ss + ( C( ig ) + AP( ig ) * GSCALE( ig ) * CDASH( ig ) ) ** 2
           ss = ss +                                                          & 
                  ( GSCALE( ig ) * ( C( ig ) + AP( ig ) * CDASH( ig ) ) ) ** 2
           ss2 = ss2 + ( GSCALE( ig ) * C( ig ) ) ** 2
         END IF
       END IF
     END DO

!  record the violation

     LANCELOT_violation = half * ss
     ss2 = half * ss2 

     RETURN

!  End of subroutine LANCELOT_violation

     END FUNCTION LANCELOT_violation

!  End of module LANCELOT

   END MODULE LANCELOT_steering_double




