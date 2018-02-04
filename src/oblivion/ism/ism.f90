! THIS VERSION: GALAHAD 3.0 - 18/01/2018 AT 08:30 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ I S M   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.4. February 27th 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_ISM_double

!     -------------------------------------------------------
!    |                                                       |
!    | ISM, an iterated-subspace minimization algorithm for  |
!    |  unconstrained optimization                           |
!    |                                                       |
!    |      Aim: find a (local) minimizer of the problem     |
!    |                                                       |
!    |                minimize   f(x)                        |
!    |                                                       |
!     -------------------------------------------------------

!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_PSLS_double
     USE GALAHAD_SLS_double
     USE GALAHAD_SCU_double
     USE GALAHAD_GLTR_double
     USE GALAHAD_TRS_double
     USE GALAHAD_TRU_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_MOP_double, ONLY: mop_Ax
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: ISM_initialize, ISM_read_specfile, ISM_solve,                 &
               ISM_terminate, NLPT_problem_type, NLPT_userdata_type,         &
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
     REAL ( KIND = wp ), PARAMETER :: sixteen = 16.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
     REAL ( KIND = wp ), PARAMETER :: tenm8 = ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

     REAL ( KIND = wp ), PARAMETER :: gamma_1 = sixteenth
     REAL ( KIND = wp ), PARAMETER :: gamma_2 = half
     REAL ( KIND = wp ), PARAMETER :: gamma_3 = two
     REAL ( KIND = wp ), PARAMETER :: gamma_4 = sixteen
     REAL ( KIND = wp ), PARAMETER :: nu_g = ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: mu_1 = one - ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: mu_2 = point1
     REAL ( KIND = wp ), PARAMETER :: theta = half

!--------------------------
!  Derived type definitions
!--------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: ISM_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

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

!   specify the model used. Possible values are
!
!      0  dynamic (*not yet implemented*)
!      1  first-order (no Hessian)
!      2  second-order (exact Hessian)
!      3  barely second-order (identity Hessian)
!      4  secant second-order (sparsity-based) (*not yet implemented*)
!      5  secant second-order (limited-memory BFGS) (*not yet implemented*)
!      6  secant second-order (limited-memory SR1) (*not yet implemented*)

       INTEGER :: model = 2

!   specify the norm used. The norm is defined via ||v||^2 = v^T P v,
!    and will define the preconditioner used for iterative methods.
!    Possible values for P are
!
!     -3  users own norm
!     -2  P = limited-memory BFGS matrix (with %lbfgs_vectors history)
!     -1  identity (= Euclidan two-norm)
!      0  automatic (*not yet implemented*)
!      1  diagonal, P = diag( max( Hessian, %min_diagonal ) )
!      2  banded, P = band( Hessian ) with semi-bandwidth %semi_bandwidth
!      3  re-ordered band, P=band(order(A)) with semi-bandwidth %semi_bandwidth
!      4  full factorization, P = Hessian, Schnabel-Eskow modification (*nyi*)
!      5  full factorization, P = Hessian, GMPS modification (*not yet*)
!      6  incomplete factorization of Hessian, Lin-More' (*not yet*)
!      7  incomplete factorization of Hessian, HSL_MI28 (*not yet*)
!      8  incomplete factorization of Hessian, Munskgaard (*not yet*)
!      9  expanding band of Hessian (*not yet implemented*)

       INTEGER :: preconditioner = 1

!   specify the semi-bandwidth of the band matrix P if required

       INTEGER :: semi_bandwidth = 5

!   number of vectors used by the L-BFGS matrix P if required

       INTEGER :: lbfgs_vectors = 10

!   number of vectors used by the Lin-More' incomplete factorization
!    matrix P if required

       INTEGER :: icfs_vectors = 10

!   number of vectors used by the sparsity-based secant Hessian if required

       INTEGER :: max_dxg = 100

!   dimension of the search subspace

       INTEGER :: subspace_dimension = 10

!   overall convergence tolerances. The iteration will terminate when the
!     norm of the gradient of the objective function is smaller than
!       MAX( %stop_g_absolute, %stop_g_relative * norm of the initial gradient
!     or if the step is less than %stop_s

       REAL ( KIND = wp ) :: stop_g_absolute = tenm5
       REAL ( KIND = wp ) :: stop_g_relative = tenm8
       REAL ( KIND = wp ) :: stop_s = epsmch

!   convergence of the subspace minimization occurs as sson as the subspace
!    gradient relative to the gradient is smaller than %stop_relative_subspace_g

       REAL ( KIND = wp ) :: stop_relative_subspace_g = SQRT( epsmch )

!   value for the trust-region radius

       REAL ( KIND = wp ) :: radius = one

!   reset the maximum subspace dimension to 1 when the gradient is smaller than
!   switch_to_1_d

       REAL ( KIND = wp ) :: switch_to_1_d = ten ** ( - 2 )

!   the smallest value the onjective function may take before the problem
!    is marked as unbounded

       REAL ( KIND = wp ) :: obj_unbounded = - epsmch ** ( - 2 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

!   should we include the gradient in the search subspace?

       LOGICAL :: include_gradient_in_subspace = .TRUE.

!   is the Hessian matrix of second derivatives available or is access only
!    via matrix-vector products?

       LOGICAL :: hessian_available = .TRUE.

!   if %space_critical true, every effort will be made to use as little
!    space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix  = '""                            '

!  control parameters for TRU

       TYPE ( TRU_control_type ) :: TRU_control

!  control parameters for TRS

       TYPE ( TRS_control_type ) :: TRS_control

!  control parameters for GLTR

       TYPE ( GLTR_control_type ) :: GLTR_control

!  control parameters for PSLS

       TYPE ( PSLS_control_type ) :: PSLS_control
     END TYPE ISM_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: ISM_time_type

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

     TYPE, PUBLIC :: ISM_inform_type

!  return status. See TRU_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER :: iter = 0

!  the total number of CG iterations performed

       INTEGER :: cg_iter = 0

!  the total number of evaluations of the objection function

       INTEGER :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER :: g_eval = 0

!  the total number of evaluations of the Hessian of the objection function

       INTEGER :: h_eval = 0

!  the maximum number of factorizations in a sub-problem solve

       INTEGER :: factorization_max = 0

!  the return status from the factorization

       INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

       INTEGER :: factorization_integer = - 1

!  the total real workspace required for the factorization

       INTEGER :: factorization_real = - 1

!  the return status from the factorization

       INTEGER :: scu_status = 0

!   the maximum number of entries in the factors

       INTEGER ( KIND = long ) :: max_entries_factors = 0

!  the value of the objective function at the best estimate of the solution
!   determined by TRU_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the norm of the gradient of the objective function at the best estimate
!   of the solution determined by TRU_solve

       REAL ( KIND = wp ) :: norm_g = HUGE( one )

!  the average number of factorizations per sub-problem solve

       REAL ( KIND = wp ) :: factorization_average = zero

!  timings (see above)

       TYPE ( ISM_time_type ) :: time

!  inform parameters for TRS

       TYPE ( TRS_inform_type ) :: TRS_inform

!  inform parameters for TRU

       TYPE ( TRU_inform_type ) :: TRU_inform

!  inform parameters for GLTR

       TYPE ( GLTR_info_type ) :: GLTR_inform

!  inform parameters for PSLS

       TYPE ( PSLS_inform_type ) :: PSLS_inform

!  inform parameters for SCU

       TYPE ( SCU_info_type ) :: SCU_inform

     END TYPE ISM_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: ISM_data_type
       INTEGER :: eval_status, out, start_print, stop_print
       INTEGER :: print_level, print_level_gltr, print_level_trs, ref( 1 )
       INTEGER :: len_history, branch, ibound, ipoint, icp, max_hist
       INTEGER :: nprec, nsemib, nskip_prec, n_subspace, n_subspace_max
       INTEGER :: n_subspace_pos
       REAL :: time, time_new, time_total
       REAL ( KIND = wp ) :: f_ref, f_trial, f_best, m_best, model, stop_g
       REAL ( KIND = wp ) :: radius, radius_trial, stg_min
       REAL ( KIND = wp ) :: dxtdg, dgtdg, df, stg, hstbs, s_norm, radius_max
       LOGICAL :: printi, printt, printd, printm
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printd, set_printm
       LOGICAL :: new_h, got_f, got_g, significant_gradient
       LOGICAL :: reverse_f, reverse_g, reverse_h, reverse_hprod, reverse_prec
       CHARACTER ( LEN = 1 ) :: negcur, perturb, hard
       TYPE ( TRS_history_type ), DIMENSION( history_max ) :: history
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_best
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHO
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ALPHA
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: F_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SV
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SCU_X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SCU_RHS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: HS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BANDH
       TYPE ( SMT_type ) :: A
       TYPE ( NLPT_problem_type ) :: TRU_nlp
       TYPE ( ISM_control_type ) :: control
       TYPE ( SCU_matrix_type ) :: SCU_matrix
       TYPE ( PSLS_data_type ) :: PSLS_data
       TYPE ( SCU_data_type ) :: SCU_data
       TYPE ( GLTR_data_type ) :: GLTR_data
       TYPE ( TRS_data_type ) :: TRS_data
       TYPE ( TRU_data_type ) :: TRU_data
       TYPE ( NLPT_userdata_type ) :: TRU_userdata
     END TYPE ISM_data_type

   CONTAINS

!-*-*-  G A L A H A D -  I S M _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE ISM_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for ISM controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ISM_data_type ), INTENT( INOUT ) :: data
     TYPE ( ISM_control_type ), INTENT( OUT ) :: control
     TYPE ( ISM_inform_type ), INTENT( OUT ) :: inform

     inform%status = GALAHAD_ok

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!    INTEGER :: status, alloc_status

!  Initalize PSLS components

     CALL PSLS_initialize( data%PSLS_data, control%PSLS_control,               &
                           inform%PSLS_inform )
     control%PSLS_control%prefix = '" - PSLS:"                    '

!  Initalize GLTR components

     CALL GLTR_initialize( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )
     control%GLTR_control%prefix = '" - GLTR:"                    '

!  Initalize TRS components

     CALL TRS_initialize( data%TRS_data, control%TRS_control,                  &
                          inform%TRS_inform )
     control%TRS_control%force_Newton = .TRUE.
     control%TRS_control%prefix = '" - TRS:"                     '

!  Initalize TRU components

     CALL TRU_initialize( data%TRU_data, control%TRU_control,                  &
                          inform%TRU_inform )
     control%TRU_control%prefix = '" - TRU:"                     '
     control%TRU_control%hessian_available = .TRUE.
!    control%TRU_control%hessian_available = .FALSE.
!    control%TRU_control%preconditioner = - 1
     control%TRU_control%subproblem_direct = .TRUE.

     data%branch = 1

     RETURN

!  End of subroutine ISM_initialize

     END SUBROUTINE ISM_initialize

!-*-*-*-*-   I S M _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE ISM_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by ISM_initialize could (roughly)
!  have been set as:

! BEGIN ISM SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  maximum-number-of-iterations                    100
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  model-used                                      2
!  preconditioner-used                             1
!  semi-bandwidth-for-band-preconditioner          5
!  number-of-lbfgs-vectors                         5
!  number-of-lin-more-vectors                      5
!  max-number-of-secant-vectors                    100
!  subspace-dimension                              10
!  absolute-gradient-accuracy-required             1.0D-5
!  relative-gradient-reduction-required            1.0D-5
!  minimum-step-allowed                            2.0D-16
!  relative-subspace-gradient-accuracy-required    1.0D-8
!  switch-to-1-D                                   1.0D-2
!  trust-region-radius                             1.0D+0
!  minimum-objective-before-unbounded              -1.0D+32
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  include-gradient-in-subspace                    yes
!  hessian-available                               yes
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END ISM SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ISM_control_type ), INTENT( INOUT ) :: control
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
     INTEGER, PARAMETER :: model = alive_unit + 1
     INTEGER, PARAMETER :: preconditioner = model + 1
     INTEGER, PARAMETER :: semi_bandwidth = preconditioner + 1
     INTEGER, PARAMETER :: lbfgs_vectors = semi_bandwidth + 1
     INTEGER, PARAMETER :: icfs_vectors = lbfgs_vectors + 1
     INTEGER, PARAMETER :: max_dxg = icfs_vectors + 1
     INTEGER, PARAMETER :: subspace_dimension = max_dxg + 1
     INTEGER, PARAMETER :: stop_g_absolute = subspace_dimension + 1
     INTEGER, PARAMETER :: stop_g_relative = stop_g_absolute + 1
     INTEGER, PARAMETER :: stop_s = stop_g_relative + 1
     INTEGER, PARAMETER :: stop_relative_subspace_g = stop_s + 1
     INTEGER, PARAMETER :: radius = stop_relative_subspace_g + 1
     INTEGER, PARAMETER :: switch_to_1_d = radius + 1
     INTEGER, PARAMETER :: obj_unbounded = switch_to_1_d + 1
     INTEGER, PARAMETER :: cpu_time_limit = obj_unbounded + 1
     INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER, PARAMETER :: include_gradient_in_subspace = clock_time_limit + 1
     INTEGER, PARAMETER :: hessian_available = include_gradient_in_subspace + 1
     INTEGER, PARAMETER :: space_critical = hessian_available + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 3 ), PARAMETER :: specname = 'ISM'
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
     spec( model )%keyword = 'model-used'
     spec( preconditioner )%keyword = 'preconditioner-used'
     spec( semi_bandwidth )%keyword = 'semi-bandwidth-for-band-norm'
     spec( lbfgs_vectors )%keyword = 'number-of-lbfgs-vectors'
     spec( icfs_vectors )%keyword = 'number-of-lin-more-vectors'
     spec( max_dxg )%keyword = 'max-number-of-secant-vectors'
     spec( subspace_dimension )%keyword = 'subspace-dimension'

!  Real key-words

     spec( stop_g_absolute )%keyword = 'absolute-gradient-accuracy-required'
     spec( stop_g_relative )%keyword = 'relative-gradient-reduction-required'
     spec( stop_s )%keyword = 'minimum-step-allowed'
     spec( stop_relative_subspace_g )%keyword                                  &
       = 'relative-subspace-gradient-accuracy-required'
     spec( radius )%keyword = 'trust-region-radius'
     spec( switch_to_1_d )%keyword = 'switch-to-1-d'
     spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

     spec( include_gradient_in_subspace )%keyword                              &
       = 'include-gradient-in-subspace'
     spec( hessian_available )%keyword = 'hessian-available'
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
     CALL SPECFILE_assign_value( spec( max_dxg ),                              &
                                 control%max_dxg,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( subspace_dimension ),                   &
                                 control%subspace_dimension,                   &
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
     CALL SPECFILE_assign_value( spec( stop_relative_subspace_g ),             &
                                 control%stop_relative_subspace_g,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( radius ),                               &
                                 control%radius,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( switch_to_1_d ),                        &
                                 control%switch_to_1_d,                        &
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

     CALL SPECFILE_assign_value( spec( include_gradient_in_subspace ),         &
                                 control%include_gradient_in_subspace,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( hessian_available ),                    &
                                 control%hessian_available,                    &
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

!  read the controls for the preconditioner and iterative solver

     IF ( PRESENT( alt_specname ) ) THEN
       CALL PSLS_read_specfile( control%PSLS_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-PSLS' )
       CALL GLTR_read_specfile( control%GLTR_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-GLTR' )
       CALL TRS_read_specfile( control%TRS_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-TRS' )
       CALL TRU_read_specfile( control%TRU_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-TRU' )
     ELSE
       CALL PSLS_read_specfile( control%PSLS_control, device )
       CALL GLTR_read_specfile( control%GLTR_control, device )
       CALL TRS_read_specfile( control%TRS_control, device )
       CALL TRU_read_specfile( control%TRU_control, device )
     END IF

     RETURN

     END SUBROUTINE ISM_read_specfile

!-*-*-*-  G A L A H A D -  I S M _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE ISM_solve( nlp, control, inform, data, userdata,               &
                           eval_F, eval_G, eval_H, eval_HPROD, eval_PREC )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  ISM_solve, an iterated-subspace minimization method for finding a local
!    unconstrained minimizer of a given function

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_ISM.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_ISM_double
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
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
!  X is a rank-one allocatable array of dimension n and type default real,
!   that holds the values x of the optimization variables. The j-th component of
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
! control is a scalar variable of type ISM_control_type. See ISM_initialize
!  for details
!
! inform is a scalar variable of type ISM_inform_type. On initial entry,
!  inform%status should be set to 1. On exit, the following components will have
!  been set:
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
!    -9. The analysis phase of the factorization failed; the return status from
!        the factorization package is given in the component
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
!        value should be set in nlp%f, and data%eval_status should be set to 0
!        If the user is unable to evaluate f(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%f, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value of the i-th component of the gradient should be
!        set in nlp%G(i), for i = 1, ..., n and data%eval_status should be set
!        to 0. If the user is unable to evaluate a component of nabla_x f(x) -
!        for instance if a component of the gradient is undefined at x - the
!        user need not set nlp%G, but should then set data%eval_status to a
!        non-zero value.
!     4. The user should compute the Hessian of the objective function
!        nabla_xx f(x) at the point x indicated in nlp%X and then re-enter the
!        subroutine. The value l-th component of the Hessian stored according
!        to  scheme input in the remainder of nlp%H should be set in
!        nlp%H%val(l), for l = 1, ..., nlp%H%ne and data%eval_status should
!        be set to 0. If the user is unable to evaluate a component of
!        nabla_xx f(x) - for instance, if a component of the Hessian is
!        undefined at x - the user eed not set nlp%H%val, but should then set
!        data%eval_status to a non-zero alue.
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
!        resulting vector u = P(x)v should be set in data%U and data%eval_status
!        should be set to 0. If the user is unable to evaluate the product -
!        for instance, if a component of the preconditioner is undefined at x -
!        the user need not set data%U, but should then set data%eval_status to
!        a n-zero value.
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
!  norm_g is a scalar variable of type default real, that holds the
!   value of the norm of the objective function gradient at the best estimate
!   of the solution found.
!
!  time is a scalar variable of type ISM_time_type whose components are used
!   to hold elapsed CPU times for the various parts of the calculation.
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total time spent in the package.
!
!    preprocess is a scalar variable of type default real, that gives
!      the time spent reordering the problem to standard form prior to solution.
!
!    analyse is a scalar variable of type default real, that gives
!      the time spent analysing required matrices prior to factorization.
!
!    factorize is a scalar variable of type default real, that gives
!      the time spent factorizing the required matrices.
!
!    solve is a scalar variable of type default real, that gives
!     the time spent using the factors to solve relevant linear equations.
!
! data is a scalar variable of type ISM_data_type used for internal data.
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
! eval_F is an optional subroutine which if present must have the arguments
!  given below (see the interface blocks). The value of the objective
!  function f(x) evaluated at x=X must be returned in f, and the status
!  variable set to 0. If the evaluation is impossible at X, status should
!  be set to a nonzero value. If eval_F is not present, ISM_solve will
!  return to the user with inform%status = 2 each time an evaluation is
!  required.
!
! eval_G is an optional subroutine which if present must have the arguments
!  given below (see the interface blocks). The components of the gradient
!  nabla_x f(x) of the objective function evaluated at x=X must be returned in
!  G, and the status variable set to 0. If the evaluation is impossible at X,
!  status should be set to a nonzero value. If eval_G is not present, ISM_solve
!  will return to the user with inform%status = 3 each time an evaluation is
!  required.
!
! eval_H is an optional subroutine which if present must have the arguments
!  given below (see the interface blocks). The nonzeros of the Hessian
!  nabla_xx f(x) of the objective function evaluated at x=X must be returned in
!  H in the same order as presented in nlp%H, and the status variable set to 0.
!  If the evaluation is impossible at X, status should be set to a nonzero
!  value. If eval_H is not present, ISM_solve will return to the user with
!  inform%status  = 4 each time an evaluation is required.
!
! eval_HPROD is an optional subroutine which if present must have the arguments
!  given below (see the interface blocks). The sum u + nabla_xx f(x) v of the
!  product of the Hessian nabla_xx f(x) of the objective function evaluated
!  at x=X with the vector v=V and the vector u=U must be returned in U, and the
!  status variable set to 0. If the evaluation is impossible at X, status should
!  be set to a nonzero value. If eval_HPROD is not present, ISM_solve will
!  return to the user with inform%status = 5 each time an evaluation is required
!
! eval_PREC is an optional subroutine which if present must have the arguments
!  given below (see the interface blocks). The product u = P(x) v of the
!  user's preconditioner P(x) evaluated at x=X with the vector v=V, the result u
!  must be retured in U, and the status variable set to 0. If the evaluation is
!  impossible at X, status should be set to a nonzero value. If eval_PREC is
!  not present, ISM_solve will return to the user with inform%status = 6 each
!  time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( ISM_control_type ), INTENT( IN ) :: control
     TYPE ( ISM_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( ISM_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_F, eval_G, eval_H, eval_HPROD, eval_PREC

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
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_H( status, X, userdata, Hval )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_H
     END INTERFACE

     INTERFACE
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
     END INTERFACE

     INTERFACE
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

     INTEGER :: i, j, ic, ir, l
     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name
     REAL ( KIND = wp ) :: val

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_TIME( data%time ) ; GO TO 990 ; END IF
     IF ( inform%status == 1 ) data%branch = 1

     SELECT CASE ( data%branch )
     CASE ( 1 )  ! initialization
       GO TO 100
     CASE ( 2 )  ! initial objective evaluation
       GO TO 200
     CASE ( 3 )  ! initial gradient evaluation
       GO TO 300
     CASE ( 4 )  ! Hessian evaluation
       GO TO 400
     CASE ( 5 )  ! Hessian-vector product
       GO TO 430
     CASE ( 6 )  ! objective, gradient or Hessian evaluation
       GO TO 500
     END SELECT

!  =================
!  0. Initialization
!  =================

 100 CONTINUE
     CALL CPU_TIME( data%time )

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  record the problem dimensions

     nlp%H%n = nlp%n ; nlp%H%m = nlp%n

!  set initial values

     inform%iter = 0 ; inform%cg_iter = 0
     inform%f_eval = 0 ; inform%g_eval = 0 ; inform%h_eval = 0
     inform%obj = HUGE( one ) ; inform%norm_g = HUGE( one )
     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     inform%factorization_average = zero
     inform%factorization_max = 0

     data%n_subspace_max = MIN( MAX( 1, control%subspace_dimension ), nlp%n )
     IF ( control%include_gradient_in_subspace ) THEN
       data%n_subspace_pos = data%n_subspace_max + 1
     ELSE
       data%n_subspace_pos = data%n_subspace_max
     END IF

     CALL SMT_put( data%tru_nlp%H%type, 'DENSE', inform%status )

!  allocate sufficient space for the problem

     array_name = 'ism: data%X_current'
     CALL SPACE_resize_array( nlp%n, data%X_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ism: data%G_current'
     CALL SPACE_resize_array( nlp%n, data%G_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ism: data%tru_nlp%G'
     CALL SPACE_resize_array( data%n_subspace_pos,                             &
            data%tru_nlp%G, inform%status,                                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ism: data%tru_nlp%X'
     CALL SPACE_resize_array( data%n_subspace_pos,                             &
            data%tru_nlp%X, inform%status,                                     &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ism: data%tru_nlp%H%val'
     CALL SPACE_resize_array( data%n_subspace_pos *                            &
           ( data%n_subspace_pos + 1 ) / 2, data%tru_nlp%H%val, inform%status, &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ism: data%SV'
     CALL SPACE_resize_array( data%n_subspace_pos, data%SV,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ism: data%S'
     CALL SPACE_resize_array( nlp%n, data%n_subspace_pos, data%S,              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ism: data%HS'
     CALL SPACE_resize_array( nlp%n, data%n_subspace_pos, data%HS,             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

       array_name = 'ism: data%U'
       CALL SPACE_resize_array( nlp%n, data%U, inform%status,                  &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'ism: data%V'
       CALL SPACE_resize_array( nlp%n, data%V, inform%status,                  &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

     IF ( control%preconditioner >= 0 .OR.                                     &
          .NOT. control%hessian_available ) THEN
       array_name = 'ism: data%VECTOR'
       CALL SPACE_resize_array( nlp%n, data%VECTOR, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'ism: data%SCU_X'
       CALL SPACE_resize_array( nlp%n + data%n_subspace_max - 1, data%SCU_X,   &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'ism: data%SCU_RHS'
       CALL SPACE_resize_array( nlp%n + data%n_subspace_max - 1, data%SCU_RHS, &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'ism: data%SCU_data%BD_val'
       CALL SPACE_resize_array( nlp%n * ( data%n_subspace_max - 1 ),           &
              data%SCU_matrix%BD_val, inform%status,                           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'ism: data%SCU_data%BD_row'
       CALL SPACE_resize_array( nlp%n * ( data%n_subspace_max - 1 ),           &
              data%SCU_matrix%BD_row, inform%status,                           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'ism: data%SCU_data%BD_col_start'
       CALL SPACE_resize_array( data%n_subspace_max,                           &
              data%SCU_matrix%BD_col_start, inform%status,                     &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

     IF ( control%preconditioner < 0 ) THEN
       IF ( data%n_subspace_max > 0 )                                          &
          CALL SMT_put( data%A%type, 'DENSE', inform%status )
       array_name = 'ism: data%A'
       CALL SPACE_resize_array( nlp%n * data%n_subspace_max, data%A%val,       &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  ensure that the data is consistent

     data%control = control
     data%control%TRS_control%initial_multiplier = zero

     data%radius = data%control%radius
     data%nsemib = MAX( 0, MIN( data%control%semi_bandwidth, nlp%n - 1 ) )
     data%negcur = ' ' ; data%hard = ' '

!  decide how much reverse communication is required

     data%reverse_f = .NOT. PRESENT( eval_F )
     data%reverse_g = .NOT. PRESENT( eval_G )
     IF ( data%control%model == 2 ) THEN
       IF ( data%control%hessian_available ) THEN
         data%reverse_h = .NOT. PRESENT( eval_H )
       ELSE
         IF ( data%control%preconditioner >= 0 )                               &
           data%control%preconditioner = - 1
         data%reverse_h = .FALSE.
       END IF
       data%reverse_hprod = .NOT. PRESENT( eval_HPROD )
     ELSE
       data%control%preconditioner = - 1
       data%control%hessian_available = .FALSE.
       data%reverse_h = .FALSE.
       data%reverse_hprod = .FALSE.
       IF ( data%control%model /= 1 .AND. data%control%model /= 3 )            &
         data%control%model = 3
       IF ( data%control%model == 1 .OR. data%control%model == 3 )             &
         data%control%GLTR_control%steihaug_toint = .TRUE.
     END IF
     data%reverse_prec = .NOT. PRESENT( eval_PREC )

     data%nprec = data%control%preconditioner
     data%control%GLTR_control%unitm = data%nprec == - 1
     data%control%PSLS_control%preconditioner = data%nprec
     data%control%PSLS_control%semi_bandwidth = data%control%semi_bandwidth
     data%control%PSLS_control%icfs_vectors = data%control%icfs_vectors

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

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
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

! evaluate the objective function at the initial point

     IF ( data%reverse_f ) THEN
       data%branch = 2 ; inform%status = 2 ; RETURN
     ELSE
       CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, inform%obj )
     END IF

!  return from reverse communication to obtain the objective value

 200 CONTINUE
     IF ( data%reverse_f ) inform%obj = nlp%f
     inform%f_eval = inform%f_eval + 1

!  test to see if the objective appears to be unbounded from below

     IF ( inform%obj < control%obj_unbounded ) THEN
       inform%status = GALAHAD_error_unbounded ; GO TO 990
     END IF

!  evaluate the gradient of the objective function

     IF ( data%reverse_g ) THEN
       data%branch = 3 ; inform%status = 3 ; RETURN
     ELSE
       CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,              &
                    nlp%G( : nlp%n ) )
     END IF

!  return from reverse communication to obtain the gradient

 300 CONTINUE
     inform%g_eval = inform%g_eval + 1
     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )

!  compute the stopping tolerance

     data%stop_g = MAX( control%stop_g_absolute,                               &
                        control%stop_g_relative * inform%norm_g )

!    data%new_h = data%control%hessian_available
     data%new_h = .TRUE.

      IF ( data%printi ) WRITE( data%out, "( A, '  Problem: ', A,              &
    &   ' (n = ', I0, '): ISM stopping tolerance =', ES11.4, / )" )            &
        prefix, TRIM( nlp%pname ), nlp%n, data%stop_g

!  =======================
!  Start of main iteration
!  =======================

 310 CONTINUE
       data%got_f = .TRUE. ; data%got_g = .TRUE.

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = - 40
           RETURN
         END IF
       END IF

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print ) THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%control%GLTR_control%print_level = data%print_level_gltr
         data%control%TRS_control%print_level = data%print_level_trs
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%GLTR_control%print_level = 0
         data%control%TRS_control%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         data%control%TRS_control%print_level > 0 .OR.                         &
         data%control%TRU_control%print_level > 0

!  =======================
!  1. Test for convergence
!  =======================

       inform%iter = inform%iter + 1
       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header )         &
            WRITE( data%out, 2100 ) prefix
         data%print_1st_header = .FALSE.
         CALL CPU_TIME( data%time_total )
         data%time_total = data%time_total - data%time
         IF ( inform%iter > 1 ) THEN
            WRITE( data%out, 2120 ) prefix, inform%iter, data%hard,            &
              data%negcur, inform%obj, inform%norm_g,                          &
              inform%tru_inform%f_eval, inform%tru_inform%g_eval,              &
              inform%tru_inform%h_eval, inform%TRS_inform%factorizations,      &
              data%n_subspace, data%time_total
         ELSE
           WRITE( data%out, 2140 ) prefix, inform%iter, inform%obj,            &
             inform%norm_g
         END IF
       END IF
       IF ( inform%norm_g <= data%stop_g ) THEN
         inform%status = 0 ; GO TO 910
       END IF

!  debug printing

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2040 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2000 ) prefix, inform%f_eval, prefix, inform%g_eval,&
           prefix, inform%h_eval, prefix, inform%iter, prefix, inform%cg_iter, &
           prefix, inform%obj, prefix, inform%norm_g
         WRITE ( data%out, 2010 ) prefix
!        l = nlp%n
         l = 2
         DO j = 1, 2
            IF ( j == 1 ) THEN
               ir = 1 ; ic = MIN( l, nlp%n )
            ELSE
               IF ( ic < nlp%n - l ) WRITE( data%out, 2050 ) prefix
               ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
            END IF
            IF ( ALLOCATED( nlp%vnames ) ) THEN
              DO i = ir, ic
                 WRITE( data%out, 2020 ) prefix, nlp%vnames( i ), nlp%X( i ),  &
                  nlp%G( i )
              END DO
            ELSE
              DO i = ir, ic
                 WRITE( data%out, 2030 ) prefix, i, nlp%X( i ), nlp%G( i )
              END DO
            END IF
         END DO
       END IF

!  check to see if the iteration limit has not been exceeded

       IF ( inform%iter > data%control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  ================================
!  2. Calculate the search space, S
!  ================================

!  recompute the Hessian if it has changed

       IF ( data%new_h ) data%nskip_prec = data%nskip_prec + 1
       IF ( data%new_h .AND. data%control%hessian_available ) THEN

!  form the Hessian or a preconditioner based on the Hessian

         IF ( data%nskip_prec > nskip_prec_max ) THEN
           IF ( data%reverse_h ) THEN
             data%branch = 4 ; inform%status = 4 ; RETURN
           ELSE
             CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                  &
                          userdata, nlp%H%val( : nlp%H%ne ) )
           END IF
         END IF
       END IF

!  return from reverse communication to obtain the Hessian

 400   CONTINUE
       IF ( data%new_h .AND. data%control%hessian_available ) THEN
         IF ( data%nskip_prec > nskip_prec_max ) THEN
           inform%h_eval = inform%h_eval + 1

!  print the Hessian if desired

           IF ( data%printd ) THEN
             WRITE( data%out, "( A, ' Hessian ' )" ) prefix
             DO l = 1, nlp%H%ne
               WRITE( data%out, "( A, 2I7, ES24.16 )" ) prefix,                &
                 nlp%H%row( l ), nlp%H%col( l ), nlp%H%val( l )
             END DO
           END IF
         END IF
         IF ( data%nprec > 0 .AND. data%control%hessian_available ) THEN

!  form and factorize the preconditioner

           IF ( data%printt ) WRITE( data%out,                                 &
                 "( A, ' Computing preconditioner' )" ) prefix
           CALL PSLS_form_and_factorize( nlp%H, data%PSLS_data,                &
                     data%control%PSLS_control, inform%PSLS_inform )

!  check for error returns

           IF ( inform%PSLS_inform%status /= 0 ) THEN
             inform%status = inform%PSLS_inform%status  ; GO TO 900
           END IF
           IF ( inform%PSLS_inform%perturbed ) data%perturb = 'p'
         END IF
       END IF

       data%negcur = ' ' ; data%hard = ' '
!      data%radius = one
!      data%radius = ten ** 10

!  Loop to generate at most n_subspace_max orthogonal vectors in the
!  search space

       data%n_subspace = 0
       data%significant_gradient = .FALSE.
       data%stg_min = nu_g * inform%norm_g

       IF ( inform%norm_g <= data%control%switch_to_1_d ) THEN
         data%n_subspace_max = 1
         data%control%GLTR_control%itmax = 4 * control%GLTR_control%itmax
       END IF

  410  CONTINUE
         IF ( data%n_subspace >= data%n_subspace_max ) GO TO 460
         data%n_subspace = data%n_subspace + 1

!  Solve the trust-region subproblem
!  .................................

!  Compute the best trust-region solution orthogonal to the existing set

!  Direct method
!  -------------

!write(6,*) ' radius ', data%radius
         IF ( data%nprec <= 0 .AND. data%control%hessian_available ) THEN
           data%A%m = data%n_subspace - 1
           data%A%ne = nlp%n * data%A%m

           data%model = zero
           IF ( data%n_subspace > 1 ) THEN
             CALL TRS_solve( nlp%n, data%radius, data%model, nlp%G( : nlp%n ), &
                             nlp%H, data%S( : nlp%n, data%n_subspace ),        &
                             data%TRS_data, data%control%TRS_control,          &
                             inform%TRS_inform, A = data%A )
           ELSE
             CALL TRS_solve( nlp%n, data%radius, data%model, nlp%G( : nlp%n ), &
                             nlp%H, data%S( : nlp%n, data%n_subspace ),        &
                             data%TRS_data, data%control%TRS_control,          &
                             inform%TRS_inform )

!  if the problem is convex, use the existing factorization to compute
!  all of the required subspace matrix in one go

             IF ( inform%TRS_inform%multiplier == zero ) THEN
               write(6,"( ' definite Hessian, generate S in one go' )" )

               CALL TRS_orthogonal_solve( nlp%n, nlp%G( : nlp%n ), nlp%H,      &
                               data%S( : nlp%n, : data%n_subspace_max ),       &
                               data%n_subspace_max, data%TRS_data,             &
                               data%control%TRS_control, inform%TRS_inform )


!  normalise all of the new directions, and compute their products with the
!  gradient

               DO i = 1, data%n_subspace_max
                 data%s_norm = TWO_NORM( data%S( : nlp%n, i ) )
!write(6,*) '||s|| ', data%s_norm
!                 DO j = 1, i - 1
!                   write(6,*) 'i, j, si^s_j = ', i, j, &
!                     DOT_PRODUCT( data%S( : nlp%n, i ), data%S( : nlp%n, j ) )
!                 END DO
                 IF ( data%s_norm > zero ) THEN
                   data%S( : nlp%n, i ) =                                      &
                     data%S( : nlp%n, i ) / data%s_norm
                   data%SV( i ) =                                              &
                     DOT_PRODUCT( data%S( : nlp%n, i ), nlp%G( : nlp%n ) )
                   IF ( ABS( data%SV( i ) ) >= data%stg_min )                  &
                     data%significant_gradient = .TRUE.
                 ELSE
                   data%n_subspace = i - 1
                   GO TO 460
                 END IF
               END DO
               data%n_subspace = data%n_subspace_max
               GO TO 460
             END IF

           END IF
!write(6,*) ' norm solution ', TWO_NORM( data%S( : nlp%n, data%n_subspace ) )
!  Check for error returns

           IF ( inform%TRS_inform%status < 0 .AND.                             &
                inform%TRS_inform%status /= GALAHAD_error_ill_conditioned ) THEN
              IF ( data%printt ) WRITE( data%out, "( /,                        &
             &  A, ' Error return from TRS, status = ', I0 )" ) prefix,        &
               inform%TRS_inform%status
             inform%status = inform%TRS_inform%status
             GO TO 990
           END IF
!write(6,*) ' m, dm, mult, lambda', data%n_subspace, inform%TRS_inform%obj, &
! inform%TRS_inform%multiplier, inform%TRS_inform%pole

!  Accumulate statistics

           IF ( inform%TRS_inform%pole > zero ) data%negcur = 'n'
           IF ( inform%TRS_inform%hard_case ) data%hard = 'h'
           inform%factorization_average = ( inform%factorization_average *     &
            ( inform%iter - 1 ) + inform%TRS_inform%factorizations )/inform%iter
           inform%factorization_max =                                          &
             MAX( inform%factorization_max, inform%TRS_inform%factorizations )
           GO TO 440
         END IF

!  Iterative method
!  ----------------

!  Modify the preconditioner so that iterates are orthogonal to the existing set

         IF ( data%n_subspace > 1 ) THEN
!write(6,*) ' update scu'
           IF ( data%n_subspace == 2 ) data%SCU_matrix%m = 1
           l = data%SCU_matrix%BD_col_start( data%n_subspace - 1 )
           DO j = 1, nlp%n
             data%SCU_matrix%BD_val( l ) = data%S( j, data%n_subspace - 1 )
             data%SCU_matrix%BD_row( l ) = j
             l = l + 1
           END DO
           data%SCU_matrix%BD_col_start( data%n_subspace ) = l
!write(6,*) ' *************** new col ', ( data%SCU_matrix%BD_val( l ), &
! l = data%SCU_matrix%BD_col_start( data%n_subspace - 1 ), &
! data%SCU_matrix%BD_col_start( data%n_subspace ) - 1 )
           inform%scu_status = 1
           DO
             IF ( data%n_subspace == 2 ) THEN
               CALL SCU_factorize( data%SCU_matrix, data%SCU_data,             &
                                   data%VECTOR( : nlp%n ),                     &
                                   inform%scu_status, inform%SCU_inform )
             ELSE
               CALL SCU_append( data%SCU_matrix, data%SCU_data,                &
                                data%VECTOR( : nlp%n ),                        &
                                inform%scu_status, inform%SCU_inform )
             END IF
!write(6,*) ' scu_status ', inform%scu_status, data%SCU_matrix%m
             IF ( inform%scu_status <= 0 ) EXIT
!write(6,*) ' vec', data%VECTOR( : nlp%n )
             IF ( data%nprec > 0 ) THEN
               CALL PSLS_solve( data%VECTOR( : nlp%n ), data%PSLS_data,        &
                                data%control%PSLS_control,                     &
                                inform%PSLS_inform )
             END IF
!write(6,*) ' vec', data%VECTOR( : nlp%n )
           END DO
           data%control%GLTR_control%unitm = .FALSE.
         ELSE
           data%SCU_matrix%n = nlp%n
           data%SCU_matrix%m = 0
           data%SCU_matrix%m_max = data%n_subspace_max - 1
           data%SCU_matrix%class = 4
           data%SCU_matrix%BD_col_start( 1 ) = 1
!write(6,*) ' restart scu'
           CALL SCU_restart_m_eq_0( data%SCU_data, inform%SCU_inform )
           data%control%GLTR_control%unitm = data%nprec <= 0
         END IF

!  Start of the generalized Lanczos iteration
!  ..........................................

         data%model = zero ; data%S( : nlp%n, data%n_subspace ) = zero
         data%G_current( : nlp%n ) = nlp%G( : nlp%n )
         inform%GLTR_inform%status = 1
   420   CONTINUE

!  perform a generalized Lanczos iteration

           CALL GLTR_solve( nlp%n, data%radius, data%model,                    &
                            data%S( : nlp%n, data%n_subspace ),                &
                            data%G_current( : nlp%n ),                         &
                            data%V( : nlp%n ), data%GLTR_data,                 &
                            data%control%GLTR_control, inform%GLTR_inform )

           SELECT CASE( inform%GLTR_inform%status )

!  form the preconditioned gradient

           CASE ( 2 )

!  use the factors obtained from PSLS

!write(6,*) ' ----------- n subspace ',  data%n_subspace
             IF ( data%n_subspace == 1 ) THEN
               IF ( data%nprec > 0 ) THEN
                 CALL PSLS_solve( data%V( : nlp%n ), data%PSLS_data,           &
                                  data%control%PSLS_control,                   &
                                  inform%PSLS_inform )
               END IF
             ELSE
               inform%scu_status = 1
!write(6,*) '  rhs ', data%V( : nlp%n )
               data%SCU_RHS( : nlp%n ) = data%V( : nlp%n )
               data%SCU_RHS( nlp%n + 1 : nlp%n + data%n_subspace - 1 ) = zero
               DO
                 CALL SCU_solve( data%SCU_matrix, data%SCU_data,               &
                                 data%SCU_RHS( : nlp%n + data%n_subspace - 1 ),&
                                 data%SCU_X( : nlp%n + data%n_subspace - 1 ),  &
                                 data%VECTOR( : nlp%n ), inform%scu_status )
!write(6,*) ' solve scu_status ', inform%scu_status, data%SCU_matrix%m
                 IF ( inform%scu_status <= 0 ) EXIT
!write(6,*) ' vec', data%VECTOR( : nlp%n )
                 IF ( data%nprec > 0 ) THEN
                   CALL PSLS_solve( data%VECTOR( : nlp%n ), data%PSLS_data,    &
                                    data%control%PSLS_control,                 &
                                    inform%PSLS_inform )
                 END IF
!write(6,*) ' vec', data%VECTOR( : nlp%n )
               END DO
               data%V( : nlp%n ) = data%SCU_X( : nlp%n )
!write(6,*) ' || v ||', TWO_NORM( data%V( : nlp%n ) )
!write(6,*) ' || y ||', TWO_NORM( data%SCU_X( nlp%n + 1 : nlp%n + data%n_subspace - 1 ) )
             END IF

!  form the Hessian-vector product

           CASE ( 3 )
             SELECT CASE( data%control%model )

!  linear model

             CASE ( 1 )
               data%V( : nlp%n ) = zero

!  quadratic model with true Hessian

             CASE ( 2 )

!  if the Hessian has been calculated, form the product directly

               IF ( data%control%hessian_available ) THEN
                 data%U( : nlp%n ) = zero
                 DO l = 1, nlp%H%ne
                   i = nlp%H%row( l ) ; j = nlp%H%col( l )
                   val = nlp%H%val( l )
                   data%U( i ) = data%U( i ) + val * data%V( j )
                   IF ( i /= j ) data%U( j ) = data%U( j ) + val * data%V( i )
                 END DO
                 data%V( : nlp%n ) = data%U( : nlp%n )

!  if the Hessian is unavailable, obtain a matrix-free product

               ELSE
                 data%U( : nlp%n ) = zero
                 IF ( data%reverse_hprod ) THEN
                   data%branch = 5 ; inform%status = 5 ; RETURN
                 ELSE
                   CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),        &
                                    userdata, data%U( : nlp%n ),               &
                                    data%V( : nlp%n ) )
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
!write(6,*) ' --------- m, dm, -ve, it', data%n_subspace, data%model, &
! inform%GLTR_inform%negative_curvature, inform%GLTR_inform%iter
             GO TO 440

!  error returns

           CASE DEFAULT
             IF ( data%printt ) WRITE( data%out, "( /,                         &
             &  A, ' Error return from GLTR, status = ', I0 )" ) prefix,       &
               inform%GLTR_inform%status
             inform%status = inform%GLTR_inform%status
             GO TO 990
           END SELECT

!  return from reverse communication to obtain the Hessian-vector product
!  or preconditioned vector

  430      CONTINUE
           IF ( .NOT. data%control%hessian_available ) THEN
             IF ( inform%GLTR_inform%status == 3 .OR.                          &
                  inform%GLTR_inform%status == 7 ) THEN
               inform%h_eval = inform%h_eval + 1
               data%V( : nlp%n ) = data%U( : nlp%n )
             END IF
           END IF
           IF ( ( inform%GLTR_inform%status == 2 .OR.                          &
                  inform%GLTR_inform%status == 6 ) .AND.                       &
                  data%nprec == - 3 .AND. data%reverse_prec ) THEN
             data%V( : nlp%n ) = data%U( : nlp%n )
           END IF
           GO TO 420

!  End of the generalized Lanczos iteration
!  ........................................

  440    CONTINUE

!  normalise the new direction, and compute its product with the gradient

         data%s_norm = TWO_NORM( data%S( : nlp%n, data%n_subspace ) )
         IF ( data%s_norm > zero ) THEN
           data%S( : nlp%n, data%n_subspace ) =                                &
             data%S( : nlp%n, data%n_subspace ) / data%s_norm
           data%SV( data%n_subspace ) =                                        &
             DOT_PRODUCT( data%S( : nlp%n, data%n_subspace ), nlp%G( : nlp%n ) )
           IF ( ABS( data%SV( data%n_subspace ) ) >= data%stg_min )            &
             data%significant_gradient = .TRUE.
         ELSE
           data%n_subspace = data%n_subspace - 1
           GO TO 460
         END IF

!  Define the new basis matrix

!        IF ( data%nprec <= 0 )                                                &
         IF ( data%nprec <= 0 .AND. data%control%hessian_available )           &
           data%A%val( data%A%ne + 1 : data%A%ne + nlp%n ) =                   &
             data%S( : nlp%n, data%n_subspace )
!write(6,*) ' new s ', data%S( : nlp%n, data%n_subspace )
!  End of loop

         GO TO 410
  460  CONTINUE

!  If require add the gradient to the subspace. Orthgonalize with respect
!  to the current vectors

!if ( data%n_subspace > 1 ) write(6,*) '<s1,s2>',  &
! DOT_PRODUCT( data%S( : nlp%n, 1 ), data%S( : nlp%n, 2 ))

       IF ( data%control%include_gradient_in_subspace .AND.                    &
            .NOT.  data%significant_gradient ) THEN
         j = data%n_subspace + 1
         data%S( : nlp%n, j ) = nlp%G( : nlp%n )
         DO i = 1,  data%n_subspace
           data%S( : nlp%n, j ) =                                              &
             data%S( : nlp%n, j ) - data%S( : nlp%n, i ) * data%SV( i )
         END DO
         data%n_subspace = j

!  Normalise the final direction against the existing set

         data%s_norm = TWO_NORM( data%S( : nlp%n, data%n_subspace ) )
         IF ( data%s_norm > zero ) THEN
           data%S( : nlp%n, data%n_subspace ) =                                &
             data%S( : nlp%n, data%n_subspace ) / data%s_norm
         ELSE
           data%n_subspace = data%n_subspace - 1
         END IF
       END IF

!  check orthogonality
!      DO i = 1, data%n_subspace
!        DO j = 1, i - 1
!!       DO j = 1, i
!          WRITE(6,"( 2I3, ' product: ', ES12.4 )" ) i, j,                     &
!            DOT_PRODUCT( data%S( : nlp%n, i ), data%S( : nlp%n, j ) )
!        END DO
!      END DO

!      DO j = 1, data%n_subspace
!        WRITE(6,"( ' S_', I0, ': ', /, 5ES12.4 )" ) j,                        &
!         ( data%S( i, j ), i = 1, nlp%n )
!      END DO
! STOP


!  ====================================
!  3. perform the subspace minimization
!  ====================================

       data%tru_nlp%n = data%n_subspace
       data%X_current( : nlp%n )  = nlp%X( : nlp%n )
       data%tru_nlp%X( : data%n_subspace ) = zero

!  loop to perform subpsapce minimization

       inform%tru_inform%status = 1
       inform%tru_inform%iter = 0
       data%control%TRU_control%maxit                                          &
         = control%TRU_control%maxit + MAX( inform%tru_inform%iter, 0 )

 480   CONTINUE
!write(6,*) ' in tru '
         CALL TRU_solve( data%tru_nlp, data%control%TRU_control,               &
                         inform%tru_inform, data%tru_data, data%tru_userdata )
!write(6,*) ' out tru, status ', inform%tru_inform%status

!  test for reasonable termination

         IF ( inform%tru_inform%status == 0 .OR.                               &
              inform%tru_inform%status == GALAHAD_error_unbounded .OR.         &
              inform%tru_inform%status == GALAHAD_error_ill_conditioned .OR.   &
              inform%tru_inform%status == GALAHAD_error_tiny_step .OR.         &
              inform%tru_inform%status == GALAHAD_error_max_iterations ) THEN
           inform%obj = data%tru_nlp%f
           GO TO 510
         END IF

!  test for failure

         IF ( inform%tru_inform%status < 0 ) THEN
           WRITE( 6, "(  ' status = ', I0, ' on return from tru_solve'  )" )   &
             inform%tru_inform%status
           STOP
         END IF

!  obtain further information

!write(6,"( ' tru_status ', I0 )" ) inform%tru_inform%status
         SELECT CASE ( inform%tru_inform%status )

         CASE ( 2 )

!  compute the new point if necessary

           IF ( .NOT. ( data%got_f .OR. data%got_g ) ) THEN
!write(6,*) ' new x '
!write(6,"( ' x_tru ', ( 5ES12.4 ) )" ) data%tru_nlp%X( 1 : data%n_subspace )
             nlp%X( : nlp%n ) = data%X_current( : nlp%n )
             DO i = 1, data%n_subspace
                nlp%X( : nlp%n ) =  nlp%X( : nlp%n )                           &
                   + data%S( 1 : nlp%n, i ) * data%tru_nlp%X( i )
             END DO
           END IF

!  obtain the objective function

           IF ( data%got_f ) THEN
             nlp%f = inform%obj
             data%got_f = .FALSE.
           ELSE
             inform%f_eval = inform%f_eval + 1
             IF ( data%reverse_f ) THEN
               data%branch = 6 ; inform%status = 2 ; RETURN
             ELSE
               CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata,      &
                            nlp%f )
             END IF
           END IF
!          WRITE( 6, "( ' f_new ', ES22.14 )" ) nlp%f

!  obtain the gradient

         CASE ( 3 )
           IF ( data%got_g ) THEN
             data%got_g = .FALSE.
           ELSE
             inform%g_eval = inform%g_eval + 1
             IF ( data%reverse_g ) THEN
               data%branch = 6 ; inform%status = 3 ; RETURN
             ELSE
               CALL eval_G( data%eval_status, nlp%X( : nlp%n ),                &
                            userdata, nlp%G( : nlp%n ) )
             END IF
           END IF

!  obtain the Hessian

         CASE ( 4 )
!write(6,*) ' x ',  nlp%X( : nlp%n )
           inform%h_eval = inform%h_eval + 1
           IF ( data%reverse_h ) THEN
             data%branch = 6 ; inform%status = 4 ; RETURN
           ELSE
             CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                  &
                          userdata, nlp%H%val( : nlp%H%ne ) )
           END IF

!  obtain a Hessian-vector product

         CASE ( 5 )
!write(6,*) ' v_s ', data%tru_data%V( : data%n_subspace )
           data%V( : nlp%n ) =                                                 &
             MATMUL( data%S( 1 : nlp%n, 1 : data%n_subspace ),                 &
                     data%tru_data%V( : data%n_subspace ) )
           data%U( : nlp%n ) = zero
!write(6,*) ' u ', data%U( : nlp%n )
!write(6,*) ' v ', data%V( : nlp%n )
           IF ( data%reverse_hprod ) THEN
              data%branch = 6 ; inform%status = 5 ; RETURN
           ELSE
              CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),             &
                               userdata, data%U( : nlp%n ),                    &
                               data%V( : nlp%n ) )
           END IF
!write(6,*) ' Hv ', data%U( : nlp%n )
         END SELECT

!  obtain further information

  500    CONTINUE
         SELECT CASE ( inform%tru_inform%status )

!  obtain the objective function

         CASE ( 2 )
           data%tru_nlp%f = nlp%f
           data%tru_data%eval_status = 0

!  obtain the gradient

         CASE ( 3 )
!write(6,*) 'g', nlp%G( : nlp%n )
           DO i = 1, data%n_subspace
             data%tru_nlp%G( i ) =                                             &
               DOT_PRODUCT( nlp%G( : nlp%n ), data%S( 1 : nlp%n, i ) )
           END DO
!write(6,*) 'subspace g',  data%tru_nlp%G( : data%n_subspace )

!  see if the relative gradient is small

!          IF ( TWO_NORM( data%tru_nlp%G( : data%n_subspace ) ) <              &
!               data%control%stop_relative_subspace_g *                        &
!                 TWO_NORM( nlp%G( : nlp%n ) ) ) THEN
!            inform%obj = data%tru_nlp%f
!            GO TO 510
!          END IF
           data%tru_data%eval_status = 0

!  obtain the Hessian

         CASE ( 4 )

!  form H * S

           DO i = 1, data%n_subspace
             CALL mop_Ax( one, nlp%H,  data%S( 1 : nlp%n, i ), zero,           &
                          data%HS( : nlp%n, i ), data%out, data%control%error, &
                          0, symmetric = .TRUE. )
!                         data%print_level, symmetric = .TRUE. )
           END DO

!  form S^T * H * S

           l = 0
           DO i = 1, data%n_subspace
             DO j = 1, i
               l = l + 1
               data%tru_nlp%H%val( l ) = DOT_PRODUCT( data%S( 1 : nlp%n, i ),  &
                                                      data%HS( : nlp%n, j ) )
!write(6,*) ' h ', i, j, data%tru_nlp%H%val( l )
             END DO
           END DO
           data%tru_data%eval_status = 0

!  obtain a Hessian-vector product

         CASE ( 5 )
           DO i = 1, data%n_subspace
             data%tru_data%U( i ) = data%tru_data%U( i ) +                     &
               DOT_PRODUCT( data%U( : nlp%n ), data%S( 1 : nlp%n, i ) )
           END DO
!write(6,*) ' S^T H S v_s ', data%tru_data%U( : data%n_subspace )
           data%tru_data%eval_status = 0
         END SELECT
       GO TO 480

!  end of subspace iteration loop

  510  CONTINUE

!  ========================================
!  4. check for acceptance of the new point
!  ========================================

       data%X_current( : nlp%n )  = nlp%X( : nlp%n )

!  test to see if the objective appears to be unbounded from below

       IF ( inform%obj < control%obj_unbounded ) THEN
         inform%status = GALAHAD_error_unbounded ; GO TO 990
       END IF

       inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )
       data%new_h = .TRUE.
     GO TO 310

!  =========================
!  End of the main iteration
!  =========================

 900 CONTINUE
     IF ( data%printi ) WRITE( data%out, "( A, ' Inform = ', I0,' Stopping')") &
       prefix, inform%status

 910 CONTINUE

!  print details of solution

     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )

     CALL CPU_TIME( data%time_new )
     inform%time%total = data%time_new - data%time

     IF ( data%printi ) THEN
       IF ( data%printi ) WRITE( data%out, "( /, A, '  Problem: ', A,          &
      &     '  (n = ', I0, ')' )") prefix, TRIM( nlp%pname ), nlp%n
      IF ( data%nprec <= 0 .AND. data%control%hessian_available ) THEN
        WRITE( data%out, "( A, '  Direct solution of subspace-selection',      &
       &  ' trust-region sub-problems')") prefix
      ELSE
        WRITE( data%out, "( A, '  Iterative solution of subspace-selection',   &
       &  ' trust-region sub-problems')") prefix
      END IF
      WRITE( data%out, "( A, '  radius =', ES11.4 )") prefix, data%radius
      WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, ' seconds', / )" )  &
         prefix, inform%time%total
     END IF
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     CALL CPU_TIME( data%time_new )
     inform%time%total = data%time_new - data%time
     RETURN

 990 CONTINUE
     CALL CPU_TIME( data%time_new )
     inform%time%total = data%time_new - data%time
     IF ( data%printi ) WRITE( data%out, "( A, ' Inform = ', I0, ' Stopping')")&
       prefix, inform%status
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' # function evaluations  = ', I10,                        &
              /, A, ' # gradient evaluations  = ', I10,                        &
              /, A, ' # Hessian evaluations   = ', I10,                        &
              /, A, ' # major  iterations     = ', I10,                        &
              /, A, ' # minor (cg) iterations = ', I10,                        &
             //, A, ' Final objective value   = ', ES22.14,                    &
              /, A, ' Final gradient norm     = ', ES12.4 )
 2010 FORMAT( /, A, ' name             X         G ' )
 2020 FORMAT(  A, 1X, A10, 2ES12.4 )
 2030 FORMAT(  A, 1X, I10, 2ES12.4 )
 2040 FORMAT( /, A, ' Problem: ', A, ' n = ', I8 )
 2050 FORMAT( A, ' .          ........... ...........' )
 2100 FORMAT( A, '    It         f         grad       # f # grad # Hess',      &
             ' # fact n_sub        time' )
 2120 FORMAT( A, I6, 2A1, 2ES12.4, 4I7, I6, F12.2 )
 2140 FORMAT( A, I6, 2X, 2ES12.4 )

 !  End of subroutine ISM_solve

     END SUBROUTINE ISM_solve

!-*-*-  G A L A H A D -  I S M _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE ISM_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ISM_data_type ), INTENT( INOUT ) :: data
     TYPE ( ISM_control_type ), INTENT( IN ) :: control
     TYPE ( ISM_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'ism: data%X_best'
     CALL SPACE_dealloc_array( data%X_best,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%X_current'
     CALL SPACE_dealloc_array( data%X_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%G_current'
     CALL SPACE_dealloc_array( data%G_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%SV'
     CALL SPACE_dealloc_array( data%SV,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%S'
     CALL SPACE_dealloc_array( data%S,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%HS'
     CALL SPACE_dealloc_array( data%HS,                                        &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%RHO'
     CALL SPACE_dealloc_array( data%RHO,                                       &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%ALPHA'
     CALL SPACE_dealloc_array( data%ALPHA,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%D_hist'
     CALL SPACE_dealloc_array( data%D_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%F_hist'
     CALL SPACE_dealloc_array( data%F_hist,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%SCU_X'
     CALL SPACE_dealloc_array( data%SCU_X,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%SCU_RHS'
     CALL SPACE_dealloc_array( data%SCU_RHS,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%VECTOR'
     CALL SPACE_dealloc_array( data%VECTOR,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ism: data%BANDH'
     CALL SPACE_dealloc_array( data%BANDH,                                     &
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

!  Deallocate all arrays allocated within SCU

     CALL SCU_terminate( data%SCU_data, inform%scu_status, inform%SCU_inform )
     inform%status = inform%scu_status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%SCU_inform%alloc_status
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

!  Deallocate all arraysn allocated within TRU

     CALL TRU_terminate( data%TRU_data, data%control%TRU_control,              &
                          inform%TRU_inform )
     inform%status = inform%TRU_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%TRU_inform%alloc_status
       inform%bad_alloc = inform%TRU_inform%bad_alloc
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

!  End of subroutine ISM_terminate

     END SUBROUTINE ISM_terminate

!  End of module GALAHAD_ISM

   END MODULE GALAHAD_ISM_double

