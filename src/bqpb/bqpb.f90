! THIS VERSION: GALAHAD 3.3 - 20/07/2021 AT 08:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ B Q P B   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.4. January 1st 2010
!   completely revised as a wrapper to CQP, GALAHAD Version 3.3. July 20th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_BQPB_double

!     ---------------------------------------------------------
!     |                                                       |
!     | Solve the convex bound-constrained quadratic program  |
!     |                                                       |
!     |    minimize     1/2 x(T) H x + g(T) x + f             |
!     |    subject to       x_l <= x <= x_u                   |
!     |                                                       |
!     | using an infeasible-point primal-dual method          |
!     |                                                       |
!     ---------------------------------------------------------

!  ** This is essentially a wrapper for GALAHAD_CQP with m = a_ne = 0 **

      USE GALAHAD_CQP_double, BQPB_control_type => CQP_control_type,           &
                              BQPB_time_type => CQP_time_type,                 &
                              BQPB_inform_type => CQP_inform_type
      USE GALAHAD_QPD_double, BQPB_data_type => QPD_data_type
      USE GALAHAD_CLOCK
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SBLS_double, ONLY: SBLS_read_specfile
      USE GALAHAD_CRO_double, ONLY: CRO_read_specfile
      USE GALAHAD_FIT_double, ONLY: FIT_read_specfile
      USE GALAHAD_ROOTS_double, ONLY:  ROOTS_read_specfile

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: BQPB_initialize, BQPB_read_specfile, BQPB_solve,               &
                BQPB_terminate, BQPB_control_type, BQPB_data_type,             &
                BQPB_time_type, BQPB_inform_type, QPT_problem_type,            &
                SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp

    CONTAINS

!-*-*-*-*-*-   B Q P B _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE BQPB_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for BQPB. This routine should be called before
!  BQPB_solve
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

      TYPE ( BQPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( BQPB_control_type ), INTENT( OUT ) :: control
      TYPE ( BQPB_inform_type ), INTENT( OUT ) :: inform

      CALL CQP_initialize( data, control, inform )
      RETURN

!  End of BQPB_initialize

      END SUBROUTINE BQPB_initialize

!-*-*-*-*-   B Q P B _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE BQPB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BQPB_initialize could (roughly)
!  have been set as:

! BEGIN BQPB SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  maximum-number-of-iterations                      1000
!  maximum-number-of-pcg-iterations                  1000
!  maximum-poor-iterations-before-infeasible         200
!  barrier-fixed-until-iteration                     1
!  indicator-type-used                               3
!  arc-used                                          1
!  series-order                                      5
!  restore-problem-on-output                         2
!  sif-file-device                                   52
!  qplib-file-device                                 53
!  infinity-value                                    1.0D+19
!  absolute-primal-accuracy                          1.0D-5
!  relative-primal-accuracy                          1.0D-5
!  absolute-dual-accuracy                            1.0D-5
!  relative-dual-accuracy                            1.0D-5
!  absolute-complementary-slackness-accuracy         1.0D-5
!  relative-complementary-slackness-accuracy         1.0D-5
!  perturb-hessian-by                                0.0
!  mininum-initial-primal-feasibility                1000.0
!  mininum-initial-dual-feasibility                  1000.0
!  initial-barrier-parameter                         -1.0
!  feasibility-vs-complementarity-weight             1.0
!  balance-complentarity-factor                      1.0D-5
!  balance-feasibility-factor                        1.0D-5
!  poor-iteration-tolerance                          0.98
!  minimum-objective-before-unbounded                -1.0D+32
!  minimum-potential-before-unbounded                -10.0
!  identical-bounds-tolerance                        1.0D-15
!  barrier-rqeuired-before-final-lunge               1.0D-5
!  primal-indicator-tolerance                        1.0D-5
!  primal-dual-indicator-tolerance                   1.0
!  tapia-indicator-tolerance                         0.9
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  treat-separable-as-general                        F
!  just-find-feasible-point                          F
!  balance-initial-complentarity                     F
!  get-advanced-dual-variables                       F
!  puiseux-series                                    T
!  try-every-order-of-series                         T
!  move-final-solution-onto-bound                    F
!  cross-over-solution                               T
!  array-syntax-worse-than-do-loop                   F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  generate-qplib-file                               F
!  sif-file-name                                     BQPBPROB.SIF
!  qplib-file-name                                   BQPBPROB.qplib
!  output-line-prefix                                ""
! END BQPB SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( BQPB_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: start_print = print_level + 1
      INTEGER, PARAMETER :: stop_print = start_print + 1
      INTEGER, PARAMETER :: maxit = stop_print + 1
      INTEGER, PARAMETER :: infeas_max = maxit + 1
      INTEGER, PARAMETER :: muzero_fixed = infeas_max + 1
      INTEGER, PARAMETER :: restore_problem = muzero_fixed + 1
      INTEGER, PARAMETER :: indicator_type = restore_problem + 1
      INTEGER, PARAMETER :: arc = indicator_type + 1
      INTEGER, PARAMETER :: series_order = arc + 1
      INTEGER, PARAMETER :: sif_file_device = series_order + 1
      INTEGER, PARAMETER :: qplib_file_device = sif_file_device + 1
      INTEGER, PARAMETER :: infinity = qplib_file_device + 1
      INTEGER, PARAMETER :: stop_abs_p = infinity + 1
      INTEGER, PARAMETER :: stop_rel_p = stop_abs_p + 1
      INTEGER, PARAMETER :: stop_abs_d = stop_rel_p + 1
      INTEGER, PARAMETER :: stop_rel_d = stop_abs_d + 1
      INTEGER, PARAMETER :: stop_abs_c = stop_rel_d + 1
      INTEGER, PARAMETER :: stop_rel_c = stop_abs_c + 1
      INTEGER, PARAMETER :: perturb_h = stop_rel_c + 1
      INTEGER, PARAMETER :: prfeas = perturb_h + 1
      INTEGER, PARAMETER :: dufeas = prfeas + 1
      INTEGER, PARAMETER :: muzero = dufeas + 1
      INTEGER, PARAMETER :: tau = muzero + 1
      INTEGER, PARAMETER :: gamma_c = tau + 1
      INTEGER, PARAMETER :: gamma_f = gamma_c + 1
      INTEGER, PARAMETER :: reduce_infeas = gamma_f + 1
      INTEGER, PARAMETER :: obj_unbounded = reduce_infeas + 1
      INTEGER, PARAMETER :: potential_unbounded =obj_unbounded + 1
      INTEGER, PARAMETER :: identical_bounds_tol = potential_unbounded + 1
      INTEGER, PARAMETER :: mu_lunge = identical_bounds_tol + 1
      INTEGER, PARAMETER :: indicator_tol_p = mu_lunge + 1
      INTEGER, PARAMETER :: indicator_tol_pd = indicator_tol_p + 1
      INTEGER, PARAMETER :: indicator_tol_tapia = indicator_tol_pd + 1
      INTEGER, PARAMETER :: cpu_time_limit = indicator_tol_tapia + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: remove_dependencies = clock_time_limit + 1
      INTEGER, PARAMETER :: treat_zero_bounds_as_general =                     &
                              remove_dependencies + 1
      INTEGER, PARAMETER :: treat_separable_as_general =                       &
                              treat_zero_bounds_as_general + 1
      INTEGER, PARAMETER :: just_feasible = treat_separable_as_general + 1
      INTEGER, PARAMETER :: getdua = just_feasible + 1
      INTEGER, PARAMETER :: puiseux = getdua + 1
      INTEGER, PARAMETER :: every_order = puiseux + 1
      INTEGER, PARAMETER :: feasol = every_order + 1
      INTEGER, PARAMETER :: balance_initial_complentarity = feasol + 1
      INTEGER, PARAMETER :: crossover = balance_initial_complentarity + 1
      INTEGER, PARAMETER :: space_critical = crossover + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: generate_qplib_file = generate_sif_file + 1
      INTEGER, PARAMETER :: sif_file_name = generate_qplib_file + 1
      INTEGER, PARAMETER :: qplib_file_name = sif_file_name + 1
      INTEGER, PARAMETER :: prefix = qplib_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'BQPB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( start_print )%keyword = 'start-print'
      spec( stop_print )%keyword = 'stop-print'
      spec( maxit )%keyword = 'maximum-number-of-iterations'
      spec( infeas_max )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( muzero_fixed )%keyword = 'barrier-fixed-until-iteration'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( indicator_type )%keyword = 'indicator-type-used'
      spec( arc )%keyword = 'arc-used'
      spec( series_order )%keyword = 'series-order'
      spec( sif_file_device )%keyword = 'sif-file-device'
      spec( qplib_file_device )%keyword = 'qplib-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
      spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
      spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
      spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
      spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
      spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
      spec( perturb_h )%keyword = 'perturb-hessian-by'
      spec( prfeas )%keyword = 'mininum-initial-primal-feasibility'
      spec( dufeas )%keyword = 'mininum-initial-dual-feasibility'
      spec( muzero )%keyword = 'initial-barrier-parameter'
      spec( tau )%keyword = 'feasibility-vs-complementarity-weight'
      spec( gamma_c )%keyword = 'balance-complentarity-factor'
      spec( gamma_f )%keyword = 'balance-feasibility-factor'
      spec( reduce_infeas )%keyword = 'poor-iteration-tolerance'
      spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
      spec( potential_unbounded )%keyword = 'minimum-potential-before-unbounded'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( mu_lunge )%keyword = 'minimum-barrier-before-final-extrapolation'
      spec( indicator_tol_p )%keyword = 'primal-indicator-tolerance'
      spec( indicator_tol_pd )%keyword = 'primal-dual-indicator-tolerance'
      spec( indicator_tol_tapia )%keyword = 'tapia-indicator-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( treat_separable_as_general )%keyword = 'treat-separable-as-general'
      spec( just_feasible )%keyword = 'just-find-feasible-point'
      spec( getdua )%keyword = 'get-advanced-dual-variables'
      spec( puiseux )%keyword = 'puiseux-series'
      spec( every_order )%keyword = 'try-every-order-of-series'
      spec( feasol )%keyword = 'move-final-solution-onto-bound'
      spec( balance_initial_complentarity )%keyword =                          &
        'balance-initial-complentarity'
      spec( crossover )%keyword = 'cross-over-solution'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'
      spec( generate_qplib_file )%keyword = 'generate-qplib-file'

!  Character key-words

      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( qplib_file_name )%keyword = 'qplib-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' bqpb: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( infeas_max ),                           &
                                 control%infeas_max,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( muzero_fixed ),                         &
                                 control%muzero_fixed,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( restore_problem ),                      &
                                 control%restore_problem,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_type ),                       &
                                 control%indicator_type,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( arc ),                                  &
                                 control%arc,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( series_order ),                         &
                                 control%series_order,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_device ),                    &
                                 control%qplib_file_device,                    &
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
     CALL SPECFILE_assign_value( spec( perturb_h ),                            &
                                 control%perturb_h,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prfeas ),                               &
                                 control%prfeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dufeas ),                               &
                                 control%dufeas,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( muzero ),                               &
                                 control%muzero,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tau ),                                  &
                                 control%tau,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_c ),                              &
                                 control%gamma_c,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( gamma_f ),                              &
                                 control%gamma_f,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( reduce_infeas ),                        &
                                 control%reduce_infeas,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( obj_unbounded ),                        &
                                 control%obj_unbounded,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( potential_unbounded ),                  &
                                 control%potential_unbounded,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mu_lunge ),                             &
                                 control%mu_lunge,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_p ),                      &
                                 control%indicator_tol_p,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_pd ),                     &
                                 control%indicator_tol_pd,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( indicator_tol_tapia ),                  &
                                 control%indicator_tol_tapia,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( remove_dependencies ),                  &
                                 control%remove_dependencies,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_zero_bounds_as_general ),         &
                                 control%treat_zero_bounds_as_general,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( just_feasible ),                        &
                                 control%just_feasible,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_separable_as_general ),           &
                                 control%treat_separable_as_general,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( getdua ),                               &
                                 control%getdua,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( puiseux ),                              &
                                 control%puiseux,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( every_order ),                          &
                                 control%every_order,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasol ),                               &
                                 control%feasol,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( balance_initial_complentarity ),        &
                                 control%balance_initial_complentarity,        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( crossover ),                            &
                                 control%crossover,                            &
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
     CALL SPECFILE_assign_value( spec( generate_qplib_file ),                  &
                                 control%generate_qplib_file,                  &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_name ),                      &
                                 control%qplib_file_name,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Read the specfile for SBLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-SBLS')
      ELSE
        CALL SBLS_read_specfile( control%SBLS_control, device )
      END IF

!  Read the specfile for FIT

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FIT_read_specfile( control%FIT_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-FIT' )
      ELSE
        CALL FIT_read_specfile( control%FIT_control, device )
      END IF

!  Read the specfile for CRO

      IF ( PRESENT( alt_specname ) ) THEN
        CALL CRO_read_specfile( control%CRO_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CRO' )
      ELSE
        CALL CRO_read_specfile( control%CRO_control, device )
      END IF

!  Read the specfile for ROOTS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL ROOTS_read_specfile( control%ROOTS_control, device,               &
                              alt_specname = TRIM( alt_specname ) // '-ROOTS' )
      ELSE
        CALL ROOTS_read_specfile( control%ROOTS_control, device )
      END IF

      RETURN

      END SUBROUTINE BQPB_read_specfile

!-*-*-*-*-*-*-*-*-*-   B Q P B _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE BQPB_solve( prob, data, control, inform, X_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the quadratic objective
!
!        1/2 x^T H x + g^T x + f
!
!  or the linear/separable objective
!
!        1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  where (x_l)_i <= x_i <= (x_u)_i, for i = 1, .... , n,
!
!  x is a vector of n components ( x_1, .... , x_n ), and any of the bounds 
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to BQPB_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. %WEIGHT (see below) need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the positive semi-definite Hessian H will be used
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H (except for the L-BFGS case). Eight storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', H%type )
!       H%val( : )   the values of the components of H, stored row by row
!       H%col( : )   the column indices of the components of H
!       H%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = TRANSFER( 'DENSE', H%type )
!       H%val( : )   the values of the components of H, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!   v) scaled identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 15) = 'SCALED-IDENTITY'
!       H%val( 1 )  the value assigned to each diagonal of H
!
!   vi) identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = 'IDENTITY'
!
!   vii) no Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 4 ) = 'ZERO' or 'NONE'
!
!   ix) L-BFGS Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = 'LBFGS'
!
!       The Hessian in this case is available via the component H_lm below
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above,
!    except for scheme (ix), for which a permutation will be set within H_lm.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %H_lm is a structure of type LMS_data_type, whose components hold the
!     L-BFGS Hessian. Access to this structure is via the module GALAHAD_LMS,
!     and this component needs only be set if %H%type( 1 : 5 ) = 'LBFGS.'
!
!   %WEIGHT is a REAL array, which need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is not 1 or 2.
!    If this is so, it must be of length at least %n, and contain the
!    weights X^0 for the objective function.
!
!   %G is a REAL array, which need only be set if %gradient_kind is not 0
!    or 1. If this is so, it must be of length at least %n, and contain the
!    linear terms g for the objective function.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %X_l, %X_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %X_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %X_u to a value larger than
!    control%infinity. On exit, %X_l and %X_u will most likely have been
!    reordered.
!
!   %Z is a REAL array of length %n, which must be set by the user to
!    appropriate estimates of the values of the dual variables
!    (Lagrange multipliers corresponding to the simple bound constraints
!    x_l <= x <= x_u). On successful exit, it will contain
!   the required vector of dual variables.
!
!  data is a structure of type BQPB_data_type which holds private internal data
!
!  control is a structure of type BQPB_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to BQPB_initialize. See the preamble
!   to GALAHAD_CQP for details
!
!  inform is a structure of type BQPB_inform_type that provides
!    information on exit from QP_solve. The component status
!    has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -4 The constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -8 The analytic center appears to be unbounded.
!
!    -9 The analysis phase of the factorization failed; the return status
!       from the factorization package is given in the component factor_status.
!
!   -10 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!
!   -11 The solve of a required linear system failed; the return status from
!       the factorization package is given in the component factor_status.
!
!   -16 The problem is so ill-conditoned that further progress is impossible.
!
!   -17 The step is too small to make further impact.
!
!   -18 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!   -19 Too much time has passed. This may happen if control%cpu_time_limit or
!       control%clock_time_limit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!  On exit from BQPB_solve, other components of inform are given in the 
!   preamble to GALAHAD_CQP
!
!  X_stat is an optional  INTEGER array of length n, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   X_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( BQPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( BQPB_control_type ), INTENT( IN ) :: control
      TYPE ( BQPB_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_stat

!  local variables

      INTEGER :: status, alloc_status
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now
      CHARACTER ( LEN = 80 ) :: array_name, bad_alloc
      INTEGER, DIMENSION( 0 ) :: C_stat

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  set up a null constraint matrix A

      prob%m = 0
      prob%A%ne = 0 ; prob%A%m = 0 ; prob%A%n = prob%n

      CALL SMT_put( prob%A%type, 'COORDINATE', inform%alloc_status )
      IF ( inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_allocate ; GO TO 910
      END IF

!  allocate space for the null A and the ancillary c and y

      array_name = 'bqpb: prob%A%row'
      CALL SPACE_resize_array( prob%A%ne, prob%A%row, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bqpb: prob%A%col'
      CALL SPACE_resize_array( prob%A%ne, prob%A%col, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bqpb: prob%A%val'
      CALL SPACE_resize_array( prob%A%ne, prob%A%val, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bqpb: prob%C'
      CALL SPACE_resize_array( prob%m, prob%C, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bqpb: prob%C_l'
      CALL SPACE_resize_array( prob%m, prob%C_l, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bqpb: prob%C_u'
      CALL SPACE_resize_array( prob%m, prob%C_u, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'bqpb: prob%Y'
      CALL SPACE_resize_array( prob%m, prob%Y, inform%status,                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  now call the generic interior-point convex QP solver

      IF ( PRESENT( X_stat ) ) THEN
        CALL CQP_solve( prob, data, control, inform,                           &
                        B_stat = X_stat, C_stat = C_stat )
      ELSE
        CALL CQP_solve( prob, data, control, inform )
      END IF

!  save status values while arrays are deallocated

      status = inform%status 
      alloc_status = inform%alloc_status
      bad_alloc = inform%bad_alloc

!  deallocate space used for null A and its associates

      array_name = 'bqpb: prob%A%row'
      CALL SPACE_dealloc_array( prob%A%row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpb: prob%A%col'
      CALL SPACE_dealloc_array( prob%A%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpb: prob%A%val'
      CALL SPACE_dealloc_array( prob%A%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpb: prob%C'
      CALL SPACE_dealloc_array( prob%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpb: prob%C_l'
      CALL SPACE_dealloc_array( prob%C_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpb: prob%C_u'
      CALL SPACE_dealloc_array( prob%C_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpb: prob%Y'
      CALL SPACE_dealloc_array( prob%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bqpb: prob%A%type'
      CALL SPACE_dealloc_array( prob%A%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  restore status values

      inform%status = status
      inform%alloc_status = alloc_status
      inform%bad_alloc = bad_alloc

      RETURN

!  error returns

  910 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = REAL( time_now - time_start, wp )
      inform%time%clock_total = clock_now - clock_start
      RETURN

!  end of SUBROUTINE BQPB_solve

      END SUBROUTINE BQPB_solve

!-*-*-*-*-*-   B Q P B _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE BQPB_terminate( data, control, inform )

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
!   data    see Subroutine BQPB_initialize
!   control see Subroutine BQPB_initialize
!   inform  see Subroutine BQPB_solve
!   reverse see Subroutine BQPB_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( BQPB_data_type ), INTENT( INOUT ) :: data
     TYPE ( BQPB_control_type ), INTENT( IN ) :: control
     TYPE ( BQPB_inform_type ), INTENT( INOUT ) :: inform

     CALL CQP_terminate( data, control, inform )

     RETURN

!  End of subroutine BQPB_terminate

     END SUBROUTINE BQPB_terminate

!  End of module BQPB

   END MODULE GALAHAD_BQPB_double
