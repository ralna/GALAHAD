! THIS VERSION: GALAHAD 2.6 - 15/10/2014 AT 13:20 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ L P B   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Cartis/Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Coralia Cartis and Nick Gould

!  History -
!   planned for release in GALAHAD Version 2.2

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

! ***** NB. Just a simplified shell for QPB at present ******

   MODULE GALAHAD_LPB_double

!     ------------------------------------------
!     |                                        |
!     | Solve the linear program               |
!     |                                        |
!     |    minimize       g(T) x + f           |
!     |    subject to     c_l <= A x <= c_u    |
!     |                   x_l <=  x  <= x_u    |
!     |                                        |
!     | using an interior-point  approach      |
!     |                                        |
!     ------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_NORMS_double
      USE GALAHAD_SILS_double
      USE GALAHAD_QPT_double
      USE GALAHAD_QPP_double
      USE GALAHAD_ROOTS_double, ONLY : ROOTS_quadratic
      USE GALAHAD_QPD_double, LPB_data_type => QPD_data_type, LPB_AX => QPD_AX

      USE GALAHAD_LSQP_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_FDC_double
      USE GALAHAD_GLTR_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LPB_initialize, LPB_read_specfile, LPB_solve, LPB_terminate,   &
                LPB_solve_main, LPB_barrier_value, LPB_iterative_refinement,   &
                LPB_block_solve, LPB_block_iterative_refinement,               &
                LPB_analyse, LPB_cond, LPB_data_type,                          &
                LPB_optimal_for_SBLP, QPT_problem_type,                        &
                SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: LPB_time_type
        REAL :: total, preprocess, find_dependent, analyse, factorize, solve
        REAL :: phase1_total, phase1_analyse, phase1_factorize, phase1_solve
      END TYPE

      TYPE, PUBLIC :: LPB_control_type
        INTEGER :: error, out, print_level, start_print, stop_print, maxit 
        INTEGER :: factor, max_col, indmin, valmin, itref_max, infeas_max 
        INTEGER :: cg_maxit, precon, indicator_type, restore_problem
        REAL ( KIND = wp ) :: infinity, stop_p, stop_d, stop_c, prfeas, dufeas
        REAL ( KIND = wp ) :: muzero, reduce_infeas, obj_unbounded
        REAL ( KIND = wp ) :: pivot_tol, pivot_tol_for_dependencies, zero_pivot
        REAL ( KIND = wp ) :: identical_bounds_tol, indicator_tol_tapia
        REAL ( KIND = wp ) :: indicator_tol_p, indicator_tol_pd
        REAL ( KIND = wp ) :: inner_stop_relative, inner_stop_absolute
        REAL ( KIND = wp ) :: initial_radius, inner_fraction_opt, cpu_time_limit
        LOGICAL :: remove_dependencies, treat_zero_bounds_as_general
        LOGICAL :: center, primal, feasol, array_syntax_worse_than_do_loop
        CHARACTER ( LEN = 30 ) :: prefix
        TYPE ( LSQP_control_type ) :: LSQP_control
      END TYPE

      TYPE, PUBLIC :: LPB_inform_type
        INTEGER :: status, alloc_status, iter, cg_iter, factorization_status
        INTEGER ( KIND = long ) :: factorization_integer, factorization_real
        INTEGER :: nfacts, nbacts, nmods
        REAL ( KIND = wp ) :: obj, non_negligible_pivot
        LOGICAL :: feasible
        CHARACTER ( LEN = 80 ) :: bad_alloc
        TYPE ( LPB_time_type ) :: time
      END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      INTEGER, PARAMETER :: max_real_store_ratio = 100
      INTEGER, PARAMETER :: max_integer_store_ratio = 100
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
      REAL ( KIND = wp ), PARAMETER :: point99 = 0.99_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: tenm2 = ten ** ( - 2 )
      REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: k_diag = one
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: res_large = one
      REAL ( KIND = wp ), PARAMETER :: remote = ten ** 10
      REAL ( KIND = wp ), PARAMETER :: bar_min = zero
      REAL ( KIND = wp ), PARAMETER :: z_min = ten ** ( - 12 )

!-------------------------------
!   I n t e r f a c e  B l o c k
!-------------------------------

!      INTERFACE TWO_NORM
!
!        FUNCTION SNRM2( n, X, incx )
!        REAL :: SNRM2
!        INTEGER, INTENT( IN ) :: n, incx
!        REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!!       REAL, INTENT( IN ), DIMENSION( : ) :: X
!        END FUNCTION SNRM2
!
!        FUNCTION DNRM2( n, X, incx )
!        DOUBLE PRECISION :: DNRM2
!        INTEGER, INTENT( IN ) :: n, incx
!        DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!!       DOUBLE PRECISION, INTENT( IN ), DIMENSION( : ) :: X
!        END FUNCTION DNRM2
!        
!      END INTERFACE

   CONTAINS

!-*-*-*-*-*-   L P B _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LPB_initialize( data, control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LPB. This routine should be called before
!  LPB_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. Components are -
!
!  INTEGER control parameters:
!
!   error. Error and warning diagnostics occur on stream error 
!   
!   out. General output occurs on stream out
!   
!   print_level. The level of output required is specified by print_level
!   
!   maxit. At most maxit inner iterations are allowed 
!   
!   start_print. Any printing will start on this iteration
!
!   stop_print. Any printing will stop on this iteration
!
!   factor. The factorization to be used.
!    Possible values are
!
!      0  automatic 
!      1  Schur-complement factorization
!      2  augmented-system factorization
!
!   max_col. The maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization
!
!   indmin. An initial guess as to the integer workspace required by SILS
!
!   valmin. An initial guess as to the real workspace required by SILS
! 
!   itref_max. The maximum number of iterative refinements allowed
!
!   infeas_max. The number of iterations for which the overall infeasibility
!     of the problem is not reduced by at least a factor control%reduce_infeas
!     before the problem is flagged as infeasible (see reduce_infeas)
!
!   cg_maxit. The maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the dimension of the system + 1
!
!   precon. The preconditioner to be used for the CG is defined by precon. 
!    Possible values are
!
!      0  automatic 
!      1  no preconditioner, i.e, the identity within full factorization
!      2  diagonal using the barrier terms within full factorization
!
!   restore_problem. Indicates whether and how much of the input problem
!    should be restored on output. Possible values are
!
!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters
!
!   indicator_type. Specifies the type of indicator function used.
!    Pssible values are
!
!     1 primal indicator: constraint active <=> distance to nearest bound 
!         <= %indicator_p_tol
!     2 primal-dual indicator: constraint active <=> distance to nearest bound 
!        <= %indicator_tol_pd * size of corresponding multiplier
!     3 primal-dual indicator: constraint active <=> distance to nearest bound 
!        <= %indicator_tol_tapia * distance to same bound at previous iteration
!
!  REAL control parameters:
!
!   infinity. Any bound larger than infinity in modulus will be regarded as 
!    infinite 
!   
!   stop_p. The required accuracy for the primal infeasibility
!   
!   stop_d. The required accuracy for the dual infeasibility
!   
!   stop_c. The required accuracy for the complementarity
!   
!   prfeas. The initial primal variables will not be closer than prfeas 
!    from their bounds 
!   
!   dufeas. The initial dual variables will not be closer than dufeas from 
!    their bounds 
!   
!   muzero. The initial value of the barrier parameter. If muzero is
!    not positive, it will be reset to an appropriate value
!           
!   reduce_infeas. If the overall infeasibility of the problem is not reduced 
!    by at least a factor reduce_infeas over control%infeas_max iterations,
!    the problem is flagged as infeasible (see infeas_max)
!
!   obj_unbounded. If the objective function value is smaller than
!    obj_unbounded, it will be flagged as unbounded from below.
!
!   pivot_tol. The threshold pivot used by the matrix factorization.
!    See the documentation for SILS for details
!
!   pivot_tol_for_dependencies. The threshold pivot used by the matrix 
!    factorization when attempting to detect linearly dependent constraints.
!    See the documentation for SILS for details
!
!   zero_pivot. Any pivots smaller than zero_pivot in absolute value will 
!    be regarded to be zero when attempting to detect linearly dependent 
!    constraints
!
!   identical_bounds_tol. Any pair of constraint bounds (c_l,c_u) or (x_l,x_u)
!    that are closer than identical_bounds_tol will be reset to the average
!    of their values
!
!   initial_radius. The initial trust-region radius
!
!   inner_fraction_opt. a search direction which gives at least 
!    inner_fraction_opt times the optimal model decrease will be found
!
!   inner_stop_relative and inner_stop_absolute. The search direction is
!    considered as an acceptable approximation to the minimizer of the
!    model if the gradient of the model in the preconditioning(inverse) 
!    norm is less than 
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )
!
!   indicator_tol_p. If %indicator_type = 1, a constraint/bound will be 
!    deemed to be active <=> distance to nearest bound <= %indicator_p_tol
!
!   indicator_tol_pd. If %indicator_type = 2, a constraint/bound will be 
!    deemed to be active <=> distance to nearest bound 
!        <= %indicator_tol_pd * size of corresponding multiplier
!
!   indicator_tol_tapia. If %indicator_type = 3, a constraint/bound will be 
!    deemed to be active <=> distance to nearest bound 
!        <= %indicator_tol_tapia * distance to same bound at previous iteration
! 
!   cpu_time_limit. The maximum CPU time allowed
! 
!  LOGICAL control parameters:
!
!   remove_dependencies. If true, the equality constraints will be preprocessed
!    to remove any linear dependencies
!
!   treat_zero_bounds_as_general. If true, any problem bound with the value
!    zero will be treated as if it were a general value
!
!   center. If center is .TRUE., the algorithm will use the analytic center
!    of the feasible set as its initial feasible point. Otherwise, a feasible
!    point as close as possible to the initial point will be used. We recommend
!    using the analytic center
!
!   primal. If primal, is .TRUE., a primal barrier method will be used in
!    place of the primal-dual method
!   
!   feasol. If feasol is true, the final solution obtained will be perturbed 
!    so that variables close to their bounds are moved onto these bounds
!
!   array_syntax_worse_than_do_loop. If array_syntax_worse_than_do_loop is
!    true, f77-style do loops will be used rather than 
!    f90-style array syntax for vector operations
!
!  CHARACTER control parameters:
!
!  prefix (len=30). All output lines will be prefixed by 
!    %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( LPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPB_control_type ), INTENT( OUT ) :: control        

!  Set control parameters

      CALL LSQP_initialize( data, control%LSQP_control )

!  Integer parameters

      control%error  = 6
      control%out  = 6
      control%print_level = 0
      control%maxit  = 1000
      control%start_print = - 1
      control%stop_print = - 1
      control%factor = 0
      control%max_col = 35
      control%indmin = 1000
      control%valmin = 1000
      control%itref_max = 1
      control%infeas_max = 200
!     control%cg_maxit = - 1
      control%cg_maxit = 200
      control%precon = 0
      control%restore_problem = 2
      control%indicator_type = 3

!  Real parameters

      control%infinity = ten ** 19
      control%stop_p  = epsmch ** 0.33
      control%stop_c  = epsmch ** 0.33
      control%stop_d  = epsmch ** 0.33
      control%prfeas = one
      control%dufeas = one
      control%muzero = - one
      control%reduce_infeas = one - point01
      control%obj_unbounded = - epsmch ** ( - 2 )
!     control%pivot_tol = data%CNTL%u
      control%pivot_tol = epsmch ** 0.75
      control%pivot_tol_for_dependencies = half
      control%zero_pivot = epsmch ** 0.75
      control%identical_bounds_tol = epsmch
      control%initial_radius = - one
      control%inner_fraction_opt = point1
!     control%inner_stop_relative = zero
      control%inner_stop_relative = point01
      control%inner_stop_absolute = SQRT( epsmch )
      control%indicator_tol_p = control%stop_p
      control%indicator_tol_pd = 1.0_wp
      control%indicator_tol_tapia = 0.9_wp
      control%cpu_time_limit = - one

!  Logical parameters

      control%remove_dependencies = .TRUE.
      control%treat_zero_bounds_as_general = .FALSE.
      control%center = .TRUE.
      control%primal = .FALSE.
!     control%feasol = .TRUE.
      control%feasol = .FALSE.
      control%array_syntax_worse_than_do_loop = .FALSE.

!  Character parameters

      control%prefix = '""                            '

!  Reset relevant LSQP control parameters

      control%LSQP_control%indicator_type = control%indicator_type
      control%LSQP_control%indicator_tol_p = control%indicator_tol_p
      control%LSQP_control%indicator_tol_pd = control%indicator_tol_pd
      control%LSQP_control%indicator_tol_tapia = control%indicator_tol_tapia
      control%LSQP_control%feasol = .FALSE.
      control%LSQP_control%prefix = '" - LSQP:"                    '

      RETURN  

!  End of LPB_initialize

      END SUBROUTINE LPB_initialize

!-*-*-*-*-   L P B _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LPB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LPB_initialize could (roughly) 
!  have been set as:

! BEGIN LPB SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      1000
!  start-print                                       -1
!  stop-print                                        -1
!  factorization-used                                0
!  maximum-column-nonzeros-in-schur-complement       35
!  initial-integer-workspace                         1000
!  initial-real-workspace                            1000
!  maximum-refinements                               1
!  maximum-poor-iterations-before-infeasible         200
!  maximum-number-of-cg-iterations                   200
!  preconditioner-used                               0
!  semi-bandwidth-for-band-preconditioner            5
!  indicator-type-used                               3
!  restore-problem-on-output                         0
!  infinity-value                                    1.0D+19
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  mininum-initial-primal-feasibility                1.0
!  mininum-initial-dual-feasibility                  1.0
!  initial-barrier-parameter                         -1.0
!  poor-iteration-tolerance                          0.98
!  minimum-objective-before-unbounded                -1.0D+32
!  pivot-tolerance-used                              1.0D-12
!  pivot-tolerance-used-for-dependencies             0.5
!  zero-pivot-tolerance                              1.0D-12
!  identical-bounds-tolerance                        1.0D-15
!  initial-trust-region-radius                       -1.0
!  inner-iteration-fraction-optimality-required      0.1
!  inner-iteration-relative-accuracy-required        0.01
!  inner-iteration-absolute-accuracy-required        1.0E-8
!  primal-indicator-tolerance                        1.0D-5
!  primal-dual-indicator-tolerance                   1.0
!  tapia-indicator-tolerance                         0.9
!  maximum-cpu-time-limit                            -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  start-at-analytic-center                          T
!  primal-barrier-used                               F
!  move-final-solution-onto-bound                    F
!  array-syntax-worse-than-do-loop                   F
! END LPB SPECIFICATIONS

!  Dummy arguments

      TYPE ( LPB_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 44
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'LPB'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level' 
      spec(  4 )%keyword = 'maximum-number-of-iterations'
      spec(  5 )%keyword = 'start-print'
      spec(  6 )%keyword = 'stop-print'
      spec(  7 )%keyword = 'factorization-used'
      spec(  8 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec(  9 )%keyword = 'initial-integer-workspace'
      spec( 10 )%keyword = 'initial-real-workspace'
      spec( 11 )%keyword = 'maximum-refinements'
      spec( 12 )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( 13 )%keyword = 'maximum-number-of-cg-iterations'
      spec( 14 )%keyword = 'preconditioner-used'
      spec( 15 )%keyword = ''
      spec( 16 )%keyword = 'restore-problem-on-output'
      spec( 40 )%keyword = 'indicator-type-used'

!  Real key-words

      spec( 17 )%keyword = 'infinity-value'
      spec( 18 )%keyword = 'primal-accuracy-required'
      spec( 19 )%keyword = 'dual-accuracy-required'
      spec( 20 )%keyword = 'complementary-slackness-accuracy-required'
      spec( 21 )%keyword = 'mininum-initial-primal-feasibility'
      spec( 22 )%keyword = 'mininum-initial-dual-feasibility'
      spec( 23 )%keyword = 'initial-barrier-parameter'
      spec( 24 )%keyword = 'poor-iteration-tolerance'
      spec( 25 )%keyword = 'minimum-objective-before-unbounded'
      spec( 26 )%keyword = 'pivot-tolerance-used'
      spec( 27 )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( 28 )%keyword = 'zero-pivot-tolerance'
      spec( 29 )%keyword = 'initial-trust-region-radius'
      spec( 30 )%keyword = 'inner-iteration-fraction-optimality-required'
      spec( 31 )%keyword = 'inner-iteration-relative-accuracy-required'
      spec( 32 )%keyword = 'inner-iteration-absolute-accuracy-required'
      spec( 33 )%keyword = 'identical-bounds-tolerance'
      spec( 41 )%keyword = 'primal-indicator-tolerance'
      spec( 42 )%keyword = 'primal-dual-indicator-tolerance'
      spec( 43 )%keyword = 'tapia-indicator-tolerance'
      spec( 44 )%keyword = 'maximum-cpu-time-limit'

!  Logical key-words

      spec( 34 )%keyword = 'remove-linear-dependencies'
      spec( 35 )%keyword = 'treat-zero-bounds-as-general'
      spec( 36 )%keyword = 'start-at-analytic-center'
      spec( 37 )%keyword = 'primal-barrier-used'
      spec( 38 )%keyword = 'move-final-solution-onto-bound'
      spec( 39 )%keyword = 'array-syntax-worse-than-do-loop'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_integer( spec( 1 ), control%error,                  &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%out,                    &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 3 ), control%print_level,            &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 4 ), control%maxit,                  &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 5 ), control%start_print,            &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 6 ), control%stop_print,             &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 7 ), control%factor,                 &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 8 ), control%max_col,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 9 ), control%indmin,                 &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 10 ), control%valmin,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 11 ), control%itref_max,             &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 12 ), control%infeas_max,            &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 13 ), control%cg_maxit,              &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 14 ), control%precon,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 16 ), control%restore_problem,       &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 40 ), control%indicator_type,        &
                                    control%error )

!  Set real values

      CALL SPECFILE_assign_real( spec( 17 ), control%infinity,                 &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 18 ), control%stop_p,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 19 ), control%stop_d,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 20 ), control%stop_c,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 21 ), control%prfeas,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 22 ), control%dufeas,                   &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 23 ), control%muzero,                   &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 24 ), control%reduce_infeas,            &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 25 ), control%obj_unbounded,            &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 26 ), control%pivot_tol,                &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 27 ),                                   &
                                 control%pivot_tol_for_dependencies,           &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 28 ), control%zero_pivot,               &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 29 ), control%initial_radius,           &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 30 ), control%inner_fraction_opt,       &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 31 ), control%inner_stop_relative,      &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 32 ), control%inner_stop_absolute,      &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 33 ), control%identical_bounds_tol,     &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 41 ), control%indicator_tol_p,          &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 41 ), control%indicator_tol_pd,         &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 43 ), control%indicator_tol_tapia,      &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 44 ), control%cpu_time_limit,           &
                                 control%error )

!  Set logical values

      CALL SPECFILE_assign_logical( spec( 34 ), control%remove_dependencies,   &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 35 ),                                &
                                    control%treat_zero_bounds_as_general,      &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 36 ), control%center,                &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 37 ), control%primal,                &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 38 ), control%feasol,                &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 39 ),                                &
                                    control%array_syntax_worse_than_do_loop,   &
                                    control%error )

!  Read the specfile for LSQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL LSQP_read_specfile( control%LSQP_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-LSQP')
      ELSE
        CALL LSQP_read_specfile( control%LSQP_control, device )
      END IF

!  Reset relevant LSQP control parameters

      control%LSQP_control%indicator_type = control%indicator_type
      control%LSQP_control%indicator_tol_p = control%indicator_tol_p
      control%LSQP_control%indicator_tol_pd = control%indicator_tol_pd
      control%LSQP_control%indicator_tol_tapia = control%indicator_tol_tapia
      control%LSQP_control%feasol = .FALSE.
    
      RETURN

      END SUBROUTINE LPB_read_specfile

!-*-*-*-*-*-*-*-*-   L P B _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE LPB_solve( prob, data, control, inform, C_stat, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, A is an m by n matrix, and any of the bounds 
!  (c_l)_i, (c_u)_i, (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A is sparse
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
!    to be solved since the last call to LPB_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!                 
!   %G is a REAL array of length %n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.  
!    On exit, G will most likely have been reordered.
!   
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %A is a structure of type SMT_type used to hold the matrix A. 
!    Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of 
!                    increasing column indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row will contain
!    the row numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).   
! 
!   %C is a REAL array of length %m, which is used to store the values of 
!    A x. It need not be set on entry. On exit, it will have been filled 
!    with appropriate values.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %C_l, %C_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to control%infinity in 
!    absolute value will be regarded as being infinite (see the entry 
!    control%infinity). Thus, an infinite lower bound may be specified by 
!    setting the appropriate component of %C_l to a value smaller than 
!    -control%infinity, while an infinite upper bound can be specified by 
!    setting the appropriate element of %C_u to a value larger than 
!    control%infinity. On exit, %C_l and %C_u will most likely have been 
!    reordered.
!   
!   %Y is a REAL array of length %m, which must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers 
!    corresponding to the general constraints c_l <= A x <= c_u. 
!    On successful exit, it will contain the required vector of Lagrange 
!    multipliers.
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
!  data is a structure of type LPB_data_type which holds private internal data
!
!  control is a structure of type LPB_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LPB_initialize. See LPB_initialize 
!   for details
!
!  inform is a structure of type LPB_inform_type that provides 
!    information on exit from LPB_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!   - 1 one of the restrictions 
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -2 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 One of more of the components A_val,  A_row,  A_col is not
!       large enough to hold the given matrices.
!
!    -4 an entry from the strict upper triangle of H has been input.
!
!    -5 The constraints are inconsistent.
!
!    -6 The constraints appear to have no feasible point.
!
!    -7 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!      
!    -8 The problem is so ill-conditoned that further progress is impossible.  
!
!    -9 The step is too small to make further impact.
!
!   -10 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of 
!       a badly scaled problem.
!
!    -11 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -12 Too much CPU time has passed. This may happen if control%cpu_time_limit 
!        is too small, but may also be symptomatic of a badly scaled problem.
!
!  On exit from LPB_solve, other components of inform give the 
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation 
!     iter   = The total number of iterations required.
!     cg_iter = The total number of conjugate gradient iterations required.
!     factorization_integer = The total integer workspace required for the 
!       factorization.
!     factorization_real = The total real workspace required for the 
!       factorization.
!     nfacts = The total number of factorizations performed.
!     nbacts = The total number of "wasted" function evaluations during the 
!       linesearch.
!     nmods  = The total number of factorizations which were modified to 
!       ensure that the matrix was an appropriate preconditioner. 
!     factorization_status = the return status from the matrix factorization
!       package.   
!     obj = the value of the objective function at the best estimate of the 
!       solution determined by LPB_solve.
!     non_negligible_pivot = the smallest pivot which was not judged to be
!       zero when detecting linearly dependent constraints
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package.
!     time%preprocess = the time spent preprocessing the problem.
!     time%find_dependent = the time spent detecting linear dependencies
!     time%analyse = the time spent analysing the required matrices prior to
!       factorization.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!     time%phase1_total = the total time spent in the initial-point phase of the
!      package.
!     time%phase1_analyse = the time spent analysing the required matrices 
!       prior to factorization in the inital-point phase.
!     time%phase1_factorize = the time spent factorizing the required matrices
!       in the inital-point phase.
!     time%phase1_solve = the time spent computing the search direction
!       in the inital-point phase.
!
!  C_stat is an optional INTEGER array of length m, which if present will be 
!   set on exit to indicate the likely ultimate status of the constraints. 
!   Possible values are 
!   C_stat( i ) < 0, the i-th constraint is likely in the active set, 
!                    on its lower bound, 
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set
!
!  B_stat is an optional  INTEGER array of length m, which if present will be 
!   set on exit to indicate the likely ultimate status of the simple bound 
!   constraints. Possible values are 
!   B_stat( i ) < 0, the i-th bound constraint is likely in the active set, 
!                    on its lower bound, 
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPB_control_type ), INTENT( IN ) :: control
      TYPE ( LPB_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER :: a_ne, i, j, tiny_x, tiny_c, n_depen, n_more_depen, nzc
      REAL :: dum, time, time_start
      REAL ( KIND = wp ) :: tol, f, av_bnd
      LOGICAL :: reallocate, printi, first_pass, center, reset_bnd
      LOGICAL :: remap_fixed, remap_freed, remap_more_freed, stat_required
      TYPE ( FDC_data_type ) :: FDC_data
      TYPE ( FDC_control_type ) :: FDC_control        
      TYPE ( FDC_inform_type ) :: FDC_inform
      TYPE ( LSQP_control_type ) :: LSQP_control
      TYPE ( LSQP_inform_type ) :: LSQP_inform

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering LPB_solve ' )" )

!  Initialize time

      CALL CPU_TIME( time_start )

!  Set initial timing breakdowns

      inform%time%total = 0.0     ; inform%time%phase1_total = 0.0    
      inform%time%analyse = 0.0   ; inform%time%phase1_analyse = 0.0  
      inform%time%factorize = 0.0 ; inform%time%phase1_factorize = 0.0
      inform%time%solve = 0.0     ; inform%time%phase1_solve = 0.0    
      inform%time%preprocess = 0.0 ; inform%time%find_dependent = 0.0

!  Initialize counts

      inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%iter = - 1 ; inform%nfacts = - 1 ; inform%cg_iter = - 1
      inform%nbacts = 0 ; inform%nmods = 0
      inform%factorization_integer = 0 ; inform%factorization_real = 0
      inform%obj = prob%f ; inform%non_negligible_pivot = zero
      inform%feasible = .FALSE.
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )

!  Basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1 

!  Ensure that input parameters are within allowed ranges

      IF ( prob%n <= 0 .OR. prob%m < 0 .OR.                                    &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) inform%status 
        CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
        GO TO 800
      END IF 

!  If required, write out problem 

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', 2I8 )" ) prob%n, prob%m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) prob%f
        WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
        WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_l( : prob%n )
        WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_u( : prob%n )
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' A (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne )
        END IF
        WRITE( control%out, "( ' C_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_l( : prob%m )
        WRITE( control%out, "( ' C_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_u( : prob%m )
      END IF

!  Check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

      reset_bnd = .FALSE.
      DO i = 1, prob%n
        IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status 
          CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
          GO TO 800 
        ELSE IF ( prob%X_u( i ) == prob%X_l( i ) ) THEN
        ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
          prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO   
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( ' ', /, '   **  Warning: one or more variable bounds reset ' )" )

      reset_bnd = .FALSE.
      DO i = 1, prob%m
        IF ( prob%C_l( i ) - prob%C_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status 
          CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
          GO TO 800 
        ELSE IF ( prob%C_u( i ) == prob%C_l( i ) ) THEN
        ELSE IF ( prob%C_u( i ) - prob%C_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%C_l( i ) + prob%C_u( i ) )
          prob%C_l( i ) = av_bnd ; prob%C_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO   
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( ' ', /, '   **  Warning: one or more constraint bounds reset ' )" )

!  Set gradient type to generic - this may change in future

      prob%gradient_kind = - 1
      prob%Hessian_kind = 0

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF
      IF ( prob%new_problem_structure ) THEN
        CALL QPP_initialize( data%QPP_map, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                       &
          control%treat_zero_bounds_as_general

!  Store the problem dimensions

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne 
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions before preprocessing: ', /,          &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

!  Perform the preprocessing

        CALL CPU_TIME( time ) 
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                     &
                          data%QPP_inform, data%dims, prob,                   &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( dum ) ; dum = dum - time
        inform%time%preprocess = inform%time%preprocess + dum
  
!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= 0 ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )               &
            WRITE( control%out, "( ' status ', I3, ' after QPP_reorder ')" )  &
             data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status 
          IF ( control%out > 0 .AND. control%print_level > 0 .AND.             &
               inform%status == - 4 ) WRITE( control%error, 2240 ) 
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
          GO TO 800 
        END IF 

!  Record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne 
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions after preprocessing: ', /,           &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  Recover the problem dimensions after preprocessing

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time ) 
          CALL QPP_apply( data%QPP_map, data%QPP_inform, prob,                 &
                          get_all = .TRUE. )
          CALL CPU_TIME( dum ) ; dum = dum - time
          inform%time%preprocess = inform%time%preprocess + dum
 
!  Test for satisfactory termination

          IF ( data%QPP_inform%status /= 0 ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( ' status ', I3, ' after QPP_apply ')" )   &
               data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) inform%status 
            CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform)
            CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
            GO TO 800 
          END IF 
        END IF 
        data%trans = data%trans + 1

!  Record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne 
        END IF

      END IF

!  Special case: no free variables

      IF ( prob%n == 0 ) THEN
        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 1, prob%m ; prob%Y( i ) = zero ; END DO
        ELSE
          prob%Y( : prob%m ) = zero
        END IF
        remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
        inform%obj = prob%f
        i = COUNT( ABS( prob%C_l( : data%dims%c_equality ) ) >                 &
              control%stop_p ) +                                               &
            COUNT( prob%C_l( data%dims%c_l_start : data%dims%c_l_end ) >       &
              control%stop_p ) +                                               &
            COUNT( prob%C_u( data%dims%c_u_start : data%dims%c_u_end ) <       &
              - control%stop_p )
        IF ( i == 0 ) THEN
          inform%status = GALAHAD_ok
        ELSE
          inform%status = GALAHAD_error_primal_infeasible
        END IF
        GO TO 700
      END IF

!  =================================================================
!  Check to see if the equality constraints are linearly independent
!  =================================================================

      IF ( prob%m > 0 .AND.                                                    &
           ( .NOT. data%tried_to_remove_deps .AND.                             &
              control%remove_dependencies ) ) THEN

        CALL CPU_TIME( time ) 
        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out,                                                  &
            "( /, 1X, I0, ' equalities from ', I0, ' constraints ' )" )        &
            data%dims%c_equality, prob%m

!  Set control parameters

        CALL FDC_initialize( FDC_data, FDC_control, FDC_inform )
        FDC_control%error = control%error
        FDC_control%out = control%out
        FDC_control%print_level = control%print_level
        FDC_control%indmin = control%indmin
        FDC_control%valmin = control%valmin
        FDC_control%zero_pivot = control%zero_pivot
        FDC_control%pivot_tol = control%pivot_tol_for_dependencies
        FDC_control%max_infeas = control%stop_p
        FDC_control%prefix = '" - FDC:"                     '

!  Find any dependent rows

        nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1 
        CALL FDC_find_dependent( prob%n, data%dims%c_equality,                 &
                                 prob%A%val( : nzc ),                          &
                                 prob%A%col( : nzc ),                          &
                                 prob%A%ptr( : data%dims%c_equality + 1 ),     &
                                 prob%C_l, n_depen, data%Index_C_freed,        &
                                 FDC_data, FDC_control, FDC_inform )
        inform%status = FDC_inform%status
        inform%non_negligible_pivot = FDC_inform%non_negligible_pivot

!  Record output parameters

        inform%alloc_status = FDC_inform%alloc_status
        inform%factorization_status = FDC_inform%factorization_status
        inform%factorization_integer = FDC_inform%factorization_integer
        inform%factorization_real = FDC_inform%factorization_real
        inform%bad_alloc = FDC_inform%bad_alloc
        inform%time%find_dependent = FDC_inform%time%total
        inform%time%analyse = inform%time%analyse + FDC_inform%time%analyse
        inform%time%factorize = inform%time%factorize + FDC_inform%time%factorize
        inform%nfacts = 1

        CALL CPU_TIME( dum )
        IF ( control%cpu_time_limit >= zero .AND.                              &
             dum - time_start > control%cpu_time_limit ) THEN
          inform%status = GALAHAD_error_cpu_limit
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status 
          inform%time%total = dum - time_start ; GO TO 800 
        END IF 
        inform%time%preprocess = inform%time%preprocess + dum - time
        IF ( printi .AND. inform%non_negligible_pivot <                        &
             thousand * control%zero_pivot ) WRITE( control%out, "(            &
       &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /,                       & 
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, ' ***  perhaps increase control%zero_pivot from', ES11.4,'  ***', &
       &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ) )" )                      &
           inform%non_negligible_pivot, control%zero_pivot

!  Check for error exits

        IF ( inform%status /= 0 ) THEN

!  Allocate arrays to hold the matrix vector product

          reallocate = .TRUE.
          IF ( ALLOCATED(  data%HX ) ) THEN
            IF ( SIZE(  data%HX ) < prob%n ) THEN
              DEALLOCATE(  data%HX ) ; ELSE ; reallocate = .FALSE.
            END IF
          END IF
          IF ( reallocate ) THEN 
            ALLOCATE(  data%HX( prob%n ), STAT = inform%alloc_status )
            IF ( inform%alloc_status /= 0 ) THEN 
              inform%bad_alloc = 'lpb: data%HX' ; GO TO 900
            END IF
          END IF

          inform%obj = DOT_PRODUCT( prob%X( : prob%n ),                        &
                                    prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= 0 )                                          &
              WRITE( control%error, 2040 ) inform%status, 'LSQP_dependent'
          END IF
          GO TO 750
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
          WRITE( control%out, "(/, ' The following ',I0,' constraints appear', &
       &         ' to be dependent', /, ( 8I8 ) )" ) n_depen, data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  Special case: no free variables

        IF ( prob%n == 0 ) THEN
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, prob%m ; prob%Y( i ) = zero ; END DO
            DO i = 1, prob%n ; prob%Z( i ) = prob%G( i ) ; END DO
          ELSE
            prob%Y( : prob%m ) = zero
            prob%Z( : prob%n ) = prob%G( : prob%n )
          END IF
          prob%C( : prob%m ) = zero
          CALL LPB_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
                        prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,              &
                        prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
          GO TO 700
        END IF
        data%tried_to_remove_deps = .TRUE.
      ELSE
        remap_freed = .FALSE.
      END IF

      IF ( remap_freed ) THEN

!  Some of the current constraints will be removed by freeing them

        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE( control%out, "( /, ' -> ', i6, ' constraints are',            &
         & ' dependent and will be temporarily removed' )" ) n_depen

!  Allocate arrays to indicate which constraints have been freed

        reallocate = .TRUE.
        IF ( ALLOCATED( data%C_freed ) ) THEN
          IF ( SIZE( data%C_freed ) < n_depen ) THEN
            DEALLOCATE( data%C_freed ) ; ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( data%C_freed( n_depen ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb:data%C_freed' ; GO TO 900
          END IF
        END IF
        
!  Free the constraint bounds as required

        DO i = 1, n_depen
          j = data%Index_C_freed( i )
          data%C_freed( i ) = prob%C_l( j )
          prob%C_l( j ) = - control%infinity
          prob%C_u( j ) = control%infinity
          prob%Y( j ) = zero
        END DO

        CALL QPP_initialize( data%QPP_map_freed, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                       &
          control%treat_zero_bounds_as_general

!  Store the problem dimensions

        data%dims_save_freed = data%dims
        a_ne = prob%A%ne 

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions before removal of dependencies:', /, &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

!  Perform the preprocessing

        CALL CPU_TIME( time ) 
        CALL QPP_reorder( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( dum ) ; dum = dum - time
        inform%time%preprocess = inform%time%preprocess + dum
  
        data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1 
        data%dims%x_s = 1 ; data%dims%x_e = prob%n
        data%dims%c_s = data%dims%x_e + 1 
        data%dims%c_e = data%dims%x_e + data%dims%nc  
        data%dims%c_b = data%dims%c_e - prob%m
        data%dims%y_s = data%dims%c_e + 1 
        data%dims%y_e = data%dims%c_e + prob%m
        data%dims%y_i = data%dims%c_s + prob%m 
        data%dims%v_e = data%dims%y_e

!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= 0 ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( ' status ', I3, ' after QPP_reorder ')" )   &
             data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status 
          IF ( control%out > 0 .AND. control%print_level > 0 .AND.             &
               inform%status == - 4 ) WRITE( control%error, 2240 ) 
          CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,            &
                              data%QPP_inform )
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
          GO TO 800 
        END IF 

!  Record revised array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne 
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions after removal of dependencies: ', /, &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

      END IF

!  Special case: Bound-constrained LP

      IF ( a_ne == 0 ) THEN

        IF ( printi ) WRITE( control%out,                                    &
          "( /, ' Solving separable bound-constrained QP ' )" )
        IF ( PRESENT( B_stat ) ) THEN
          CALL LPB_optimal_for_SBLP( prob, control, inform,                  &
                                     B_stat = B_stat( : prob%n ) )
          IF ( printi ) WRITE( control%out,                                  &
              "( ' on exit from LPB_optimal_for_SBLP: status = ', I0,        &
           &   ', time = ', F0.2, /, ' objective value =', ES12.4,           &
           &   /, ' active bounds: ', I0, ' from ', I0 )" )                  &
              inform%status, inform%time%total, inform%obj,                  &
              COUNT( B_stat( : prob%n ) /= 0 ), prob%n
        ELSE
          CALL LPB_optimal_for_SBLP( prob, control, inform )
          IF ( printi ) WRITE( control%out,                                  &
              "( ' on exit from LPB_optimal_for_SBLP: status = ', I0,        &
           &   ', time = ', F0.2, /, ' objective value =', ES12.4 )" )       &
              inform%status, inform%time%total, inform%obj
        END IF
        GO TO 700

!  General case: LP

      ELSE
        first_pass = .TRUE.
        center = control%center
        f = prob%f
    
        prob%gradient_kind = 0

 10     CONTINUE

!  ==============================
!  Find an initial feasible point
!  ==============================

        LSQP_control = control%LSQP_control
        LSQP_control%factor = control%factor
        LSQP_control%max_col = control%max_col
        LSQP_control%itref_max = control%itref_max
        LSQP_control%infeas_max = control%infeas_max
        LSQP_control%restore_problem = control%restore_problem 
        LSQP_control%infinity = control%infinity
        LSQP_control%stop_p = control%stop_p
        LSQP_control%stop_c = control%stop_c
        LSQP_control%stop_d = control%stop_d
        LSQP_control%muzero = control%muzero
        LSQP_control%reduce_infeas = control%reduce_infeas
        LSQP_control%pivot_tol = control%pivot_tol
        LSQP_control%pivot_tol_for_dependencies =                              &
          control%pivot_tol_for_dependencies
        LSQP_control%zero_pivot = control%zero_pivot
        LSQP_control%identical_bounds_tol = control%identical_bounds_tol
        LSQP_control%remove_dependencies = control%remove_dependencies
        LSQP_control%treat_zero_bounds_as_general =                            &
          control%treat_zero_bounds_as_general
        LSQP_control%feasol = .FALSE.
        LSQP_control%array_syntax_worse_than_do_loop =                         &
          control%array_syntax_worse_than_do_loop
        LSQP_control%identical_bounds_tol =                                    &
          control%LSQP_control%identical_bounds_tol
        LSQP_control%prefix = '" - LSQP:"                    '
        IF ( printi ) THEN
          LSQP_control%print_level = control%print_level
          LSQP_control%start_print = control%start_print
          LSQP_control%stop_print = control%stop_print
          LSQP_control%out = control%out
          LSQP_control%error = control%error
        END IF

!  Either find the solution to the LP ...

        IF ( center ) THEN
          LSQP_control%just_feasible = .FALSE.
          LSQP_control%prfeas = control%prfeas
          LSQP_control%dufeas = control%dufeas
          prob%Hessian_kind = 0
          prob%f = zero

!  .. or minimize the distance to the nearest feasible point

        ELSE
          LSQP_control%just_feasible = .TRUE.
          LSQP_control%prfeas = control%prfeas
          LSQP_control%dufeas = control%dufeas
          prob%Hessian_kind = 1

          reallocate = .TRUE.
          IF ( ALLOCATED( data%X0 ) ) THEN
            IF ( SIZE( data%X0 ) < prob%n ) THEN
              DEALLOCATE( data%X0 ) ; ELSE ; reallocate = .FALSE.
            END IF
          END IF
          IF ( reallocate ) THEN 
            ALLOCATE( data%X0( prob%n ), STAT = inform%alloc_status )
            IF ( inform%alloc_status /= 0 ) THEN 
              inform%bad_alloc = 'lpb: data%X0' ; GO TO 900
            END IF
          END IF
          data%X0 = prob%X( : prob%n )
          prob%X0 = data%X0
          prob%f = zero
        END IF

        CALL LSQP_solve( prob, data, LSQP_control, LSQP_inform )
     
        inform%status = LSQP_inform%status 
        IF ( inform%status == - 2 .OR. inform%status == - 3 ) THEN
             inform%alloc_status = LSQP_inform%alloc_status
             inform%bad_alloc = LSQP_inform%bad_alloc
          GO TO 920
        END IF

        inform%feasible = LSQP_inform%feasible
        IF ( inform%status == - 4                                              &
             .OR. ( inform%status <= - 7 .AND. inform%status >= - 10 ) ) THEN
          IF ( inform%feasible ) THEN
            inform%status = GALAHAD_ok
          ELSE
            IF ( first_pass ) THEN
              center = .NOT. center
              first_pass = .FALSE.
              IF ( printi ) WRITE( control%out,                                &
              "( /, ' .... have a second attempt at getting feasible .... ' )" )
              GO TO 10
            END IF
          END IF
        END IF
        prob%f = f
        prob%Hessian_kind = 0 ; prob%gradient_kind = - 1
  
        inform%alloc_status = LSQP_inform%alloc_status 
        inform%factorization_status = LSQP_inform%factorization_status 

!  Record times for phase1

        CALL CPU_TIME( dum )
        inform%time%phase1_total = dum - time_start - inform%time%preprocess
  
        inform%time%phase1_analyse = LSQP_inform%time%analyse
        inform%time%phase1_factorize = LSQP_inform%time%factorize
        inform%time%phase1_solve = LSQP_inform%time%solve
   
        inform%time%analyse = inform%time%phase1_analyse
        inform%time%factorize = inform%time%phase1_factorize
        inform%time%solve = inform%time%phase1_solve
        inform%time%find_dependent =                                           &
          LSQP_inform%time%find_dependent + inform%time%find_dependent
   
        IF ( printi ) THEN
          WRITE( control%out, "( /, I10, ' integer and ', I10,                 &
        &  ' real words required for the factorization' )" )                   &
           LSQP_inform%factorization_integer, LSQP_inform%factorization_real
          WRITE( control%out, 2050 )
        END IF

!  Check for error exits

        IF ( inform%status /= 0 ) THEN

!  On error exit, compute the current objective function value

          IF ( inform%status /= - 1 .AND. inform%status /= - 2 ) THEN
            inform%obj = DOT_PRODUCT( prob%X( : prob%n ),                      &
                                      prob%G( : prob%n ) ) + prob%f
          END IF

!  Print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= 0 )                                          &
              WRITE( control%error, 2040 ) inform%status, 'LSQP_solve'
          END IF
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
          GO TO 700
        END IF

!  ============================
!  Initial feasible point found
!  ============================

!  Check to see if any variables/constraints are flagged as being fixed

        tol = MIN( control%stop_p / ten, SQRT( epsmch ) )
        tiny_x = COUNT( prob%X( data%dims%x_free + 1 : data%dims%x_l_start - 1)&
                        < tol ) +                                              &
                 COUNT( prob%X( data%dims%x_l_start: data%dims%x_l_end ) -     &
                        prob%X_l( data%dims%x_l_start : data%dims%x_l_end )    &
                        < tol ) +                                              &
                 COUNT( prob%X( data%dims%x_u_start: data%dims%x_u_end ) -     &
                        prob%X_u( data%dims%x_u_start : data%dims%x_u_end )    &
                        > - tol ) +                                            &
                 COUNT( prob%X( data%dims%x_u_end + 1 : prob%n )          &
                        > - tol )
  
        tiny_c = COUNT( prob%C( data%dims%c_l_start: data%dims%c_l_end ) -     &
                        prob%C_l( data%dims%c_l_start : data%dims%c_l_end )    &
                        < tol ) +                                              &
                 COUNT( prob%C( data%dims%c_u_start: data%dims%c_u_end ) -     &
                        prob%C_u( data%dims%c_u_start : data%dims%c_u_end )    &
                        > - tol )
  
        remap_fixed = tiny_x > 0 .OR. tiny_c > 0
        IF ( remap_fixed ) THEN

!  Some of the current variables/constraints will be fixed

          IF ( control%error > 0 .AND. control%print_level >= 1 )              &
            WRITE( control%out, "( /, ' -> ', i6, ' further variables and ',   &
           &       i6, ' further constraints will be fixed' )" ) tiny_x, tiny_c

!  Allocate arrays to record the bounds which will be altered

          IF ( tiny_x > 0 ) THEN
            reallocate = .TRUE.
            IF ( ALLOCATED( data%X_fixed ) ) THEN
              IF ( SIZE( data%X_fixed ) < tiny_x ) THEN
                DEALLOCATE( data%X_fixed ) ; ELSE ; reallocate = .FALSE.
              END IF
            END IF
            IF ( reallocate ) THEN 
              ALLOCATE( data%X_fixed( tiny_x ), STAT = inform%alloc_status )
              IF ( inform%alloc_status /= 0 ) THEN 
                inform%bad_alloc = 'lpb:data%X_fixed' ; GO TO 900
              END IF
            END IF
  
            reallocate = .TRUE.
            IF ( ALLOCATED( data%Index_X_fixed ) ) THEN
              IF ( SIZE( data%Index_X_fixed ) < tiny_x ) THEN
                DEALLOCATE( data%Index_X_fixed ) ; ELSE ; reallocate = .FALSE.
              END IF
            END IF
            IF ( reallocate ) THEN 
              ALLOCATE( data%Index_X_fixed( tiny_x ),                          &
                        STAT = inform%alloc_status )
              IF ( inform%alloc_status /= 0 ) THEN 
                inform%bad_alloc = 'lpb:data%Index_X_fixed' ; GO TO 900
              END IF
            END IF
          END IF
  
          IF ( tiny_c > 0 ) THEN
            reallocate = .TRUE.
            IF ( ALLOCATED( data%C_fixed ) ) THEN
              IF ( SIZE( data%C_fixed ) < tiny_c ) THEN
                DEALLOCATE( data%C_fixed ) ; ELSE ; reallocate = .FALSE.
              END IF
            END IF
            IF ( reallocate ) THEN 
              ALLOCATE( data%C_fixed( tiny_c ), STAT = inform%alloc_status )
              IF ( inform%alloc_status /= 0 ) THEN 
                inform%bad_alloc = 'lpb:data%C_fixed' ; GO TO 900
              END IF
            END IF
  
            reallocate = .TRUE.
            IF ( ALLOCATED( data%Index_C_fixed ) ) THEN
              IF ( SIZE( data%Index_C_fixed ) < tiny_c ) THEN
                DEALLOCATE( data%Index_C_fixed ) ; ELSE ; reallocate = .FALSE.
              END IF
            END IF
            IF ( reallocate ) THEN 
              ALLOCATE( data%Index_C_fixed( tiny_c ),                          &
                        STAT = inform%alloc_status )
              IF ( inform%alloc_status /= 0 ) THEN 
                inform%bad_alloc = 'lpb:data%Index_C_fixed' ; GO TO 900
              END IF
            END IF
          END IF

!  Fix the problem bounds as required

          IF ( tiny_x > 0 ) THEN
            tiny_x = 0
    
            DO i = data%dims%x_free + 1, data%dims%x_l_start - 1
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i )
              IF ( prob%X( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_u( i ) = zero
              END IF
            END DO
    
            DO i = data%dims%x_l_start, data%dims%x_u_start - 1
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i ) - prob%X_l( i )
              IF ( prob%X( i ) - prob%X_l( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_u( i ) =  prob%X_l( i )
              END IF
            END DO
    
            DO i = data%dims%x_u_start, data%dims%x_l_end
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i ) - prob%X_l( i )
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', prob%X_u( i ) - prob%X( i )
              IF ( prob%X( i ) - prob%X_l( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_u( i ) =  prob%X_l( i )
              ELSE IF ( prob%X( i ) - prob%X_u( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_l( i ) =  prob%X_u( i )
              END IF
            END DO
  
            DO i = data%dims%x_l_end + 1, data%dims%x_u_end
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', prob%X_u( i ) - prob%X( i )
              IF ( prob%X( i ) - prob%X_u( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_l( i ) =  prob%X_u( i )
              END IF
            END DO
    
            DO i = data%dims%x_u_end + 1, prob%n
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', - prob%X( i )
              IF ( prob%X( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_l( i ) = zero
              END IF
            END DO
          END IF
       
!  Do the same for the constraint bounds

          IF ( tiny_c > 0 ) THEN
            tiny_c = 0
    
            DO i = data%dims%c_l_start, data%dims%c_u_start - 1
              IF ( prob%C( i ) - prob%C_l( i ) < tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_u( i )
                data%Index_C_fixed( tiny_c ) = i
                prob%C_u( i ) =  prob%C_l( i )
              END IF
            END DO
    
            DO i = data%dims%c_u_start, data%dims%c_l_end
              IF ( prob%C( i ) - prob%C_l( i ) < tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_u( i )
                data%Index_C_fixed( tiny_c ) = i
                prob%C_u( i ) =  prob%C_l( i )
              ELSE IF ( prob%C( i ) - prob%C_u( i ) > - tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_l( i )
                data%Index_C_fixed( tiny_c ) = - i
                prob%C_l( i ) =  prob%C_u( i )
              END IF
            END DO
    
            DO i = data%dims%c_l_end + 1, data%dims%c_u_end
              IF ( prob%C( i ) - prob%C_u( i ) > - tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_l( i )
                data%Index_C_fixed( tiny_c ) = - i
                prob%C_l( i ) =  prob%C_u( i )
              END IF
            END DO
          END IF
  
          CALL QPP_initialize( data%QPP_map_fixed, data%QPP_control )
          data%QPP_control%infinity = control%infinity
          data%QPP_control%treat_zero_bounds_as_general =                     &
            control%treat_zero_bounds_as_general

!  Store the problem dimensions

          data%dims_save_fixed = data%dims
          a_ne = prob%A%ne 
  
          IF ( printi ) WRITE( control%out,                                    &
                 "( /, ' problem dimensions before preprocessing: ', /,        &
     &           ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                 &
                 prob%n, prob%m, a_ne
  
!  Perform the preprocessing

          CALL CPU_TIME( time ) 
          CALL QPP_reorder( data%QPP_map_fixed, data%QPP_control,              &
                             data%QPP_inform, data%dims, prob,                 &
                             .FALSE., .FALSE., .FALSE. )
          CALL CPU_TIME( dum ) ; dum = dum - time
          inform%time%preprocess = inform%time%preprocess + dum
    
          data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1 
          data%dims%x_s = 1 ; data%dims%x_e = prob%n
          data%dims%c_s = data%dims%x_e + 1 
          data%dims%c_e = data%dims%x_e + data%dims%nc  
          data%dims%c_b = data%dims%c_e - prob%m
          data%dims%y_s = data%dims%c_e + 1 
          data%dims%y_e = data%dims%c_e + prob%m
          data%dims%y_i = data%dims%c_s + prob%m 
          data%dims%v_e = data%dims%y_e

!  Test for satisfactory termination

          IF ( data%QPP_inform%status /= 0 ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( ' status ', I3, ' after QPP_reorder ')" ) &
               data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) inform%status 
            IF ( control%out > 0 .AND. control%print_level > 0 .AND.           &
                 inform%status == - 4 ) WRITE( control%error, 2240 ) 
            CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
            CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,          &
                                data%QPP_inform )
            CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,          &
                                data%QPP_inform )
            CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform)
            GO TO 800 
          END IF 

!  Record revised array lengths

          IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
            a_ne = prob%m * prob%n
          ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
            a_ne = prob%A%ptr( prob%m + 1 ) - 1
          ELSE
            a_ne = prob%A%ne 
          END IF

          IF ( printi ) WRITE( control%out,                                    &
                 "( /, ' problem dimensions after preprocessing: ', /,         &
     &           ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                 &
                 prob%n, prob%m, a_ne

!  ====================================================================
!  Check to see if the equality constraints remain linearly independent
!  ====================================================================

          IF ( prob%m > 0 .AND. control%remove_dependencies ) THEN
    
            CALL CPU_TIME( time ) 
            IF ( control%out > 0 .AND. control%print_level >= 1 )              &
              WRITE( control%out,                                              &
                "( /, 1X, I0, ' equalities from ', I0, ' constraints ' )" )    &
                data%dims%c_equality, prob%m

!  Find any dependent rows

            nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1 
            CALL FDC_find_dependent( prob%n, data%dims%c_equality,             &
                                     prob%A%val( : nzc ),                      &
                                     prob%A%col( : nzc ),                      &
                                     prob%A%ptr( : data%dims%c_equality + 1 ), &
                                     prob%C_l,                                 &
                                     n_more_depen, data%Index_C_more_freed,    &
                                     FDC_data, FDC_control, FDC_inform )
            inform%status = FDC_inform%status
            inform%non_negligible_pivot =                                      &
              MIN( FDC_inform%non_negligible_pivot, inform%non_negligible_pivot )

!  Record output parameters

            inform%alloc_status = FDC_inform%alloc_status
            inform%factorization_status = FDC_inform%factorization_status
            inform%factorization_integer = FDC_inform%factorization_integer
            inform%factorization_real = FDC_inform%factorization_real
            inform%bad_alloc = FDC_inform%bad_alloc
            inform%time%find_dependent =                                       &
              inform%time%find_dependent + FDC_inform%time%total
            inform%time%analyse = inform%time%analyse + FDC_inform%time%analyse
            inform%time%factorize =                                           &
              inform%time%factorize + FDC_inform%time%factorize
            inform%nfacts = 1

            CALL CPU_TIME( dum )
            IF ( control%cpu_time_limit >= zero .AND.                          &
                 dum - time_start > control%cpu_time_limit ) THEN
              inform%status = - 12
              IF ( control%error > 0 .AND. control%print_level > 0 )           &
                WRITE( control%error, 2010 ) inform%status 
              inform%time%total = dum - time_start ; GO TO 800 
            END IF 
            inform%time%preprocess = inform%time%preprocess + dum - time
            IF ( printi .AND. inform%non_negligible_pivot <                    &
                 thousand * control%zero_pivot ) WRITE( control%out, "(        &
           &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /,                   & 
           &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',&
           &  /, ' ***  perhaps increase control%zero_pivot from', ES11.4,     &
           &  '  ***', /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ) )" )         &
               inform%non_negligible_pivot, control%zero_pivot

            IF ( inform%status /= 0 ) THEN

!  On error exit, compute the current objective function value

              inform%obj = DOT_PRODUCT( prob%X( : prob%n ),                    &
                                        prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

              IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
                WRITE( control%out, "( ' ' )" )
                IF ( inform%status /= 0 )                                      &
                  WRITE( control%error, 2040 ) inform%status, 'LSQP_dependent'
              END IF
              GO TO 750
            END IF
    
            IF ( control%out > 0 .AND. control%print_level >= 2                &
                 .AND. n_more_depen > 0 )                                      &
              WRITE( control%out, "(/, ' The following ', I7,                  &
           &         ' constraints appear to be dependent', /, ( 8I8 ) )" )    &
                n_more_depen, data%Index_C_more_freed
    
            remap_more_freed = n_more_depen > 0
          ELSE
            remap_more_freed = .FALSE.
          END IF
    
          IF ( remap_more_freed ) THEN

!  Some of the current constraints will be removed by freeing them

            IF ( control%error > 0 .AND. control%print_level >= 1 )            &
              WRITE( control%out, "( /, ' -> ', i6, ' constraints are',        &
             & ' dependent and will be temporarily removed' )" ) n_more_depen

!  Allocate arrays to indicate which constraints have been freed

            reallocate = .TRUE.
            IF ( ALLOCATED( data%C_more_freed ) ) THEN
              IF ( SIZE( data%C_more_freed ) < n_more_depen ) THEN
                DEALLOCATE( data%C_more_freed ) ; ELSE ; reallocate = .FALSE.
              END IF
            END IF
            IF ( reallocate ) THEN 
              ALLOCATE( data%C_more_freed( n_more_depen ),                     &
                        STAT = inform%alloc_status )
              IF ( inform%alloc_status /= 0 ) THEN 
                inform%bad_alloc = 'lpb:data%C_more_freed' ; GO TO 900
              END IF
            END IF
        
!  Free the constraint bounds as required

            DO i = 1, n_more_depen
              j = data%Index_C_more_freed( i )
              data%C_more_freed( i ) = prob%C_l( j )
              prob%C_l( j ) = - control%infinity
              prob%C_u( j ) = control%infinity
              prob%Y( j ) = zero
            END DO
    
            CALL QPP_initialize( data%QPP_map_more_freed, data%QPP_control )
            data%QPP_control%infinity = control%infinity
            data%QPP_control%treat_zero_bounds_as_general =                    &
              control%treat_zero_bounds_as_general

!  Store the problem dimensions

            data%dims_save_more_freed = data%dims
            a_ne = prob%A%ne 
    
            IF ( printi ) WRITE( control%out,                                  &
              "( /, ' problem dimensions before removal of dependecies: ',     &
             &   /, ' n = ', I8, ' m = ', I8, ' a_ne = ',                      &
             &        I8 )" ) prob%n, prob%m, a_ne
    
!  Perform the preprocessing

            CALL CPU_TIME( time ) 
            CALL QPP_reorder( data%QPP_map_more_freed, data%QPP_control,       &
                              data%QPP_inform, data%dims, prob,                &
                              .FALSE., .FALSE., .FALSE. )
            CALL CPU_TIME( dum ) ; dum = dum - time
            inform%time%preprocess = inform%time%preprocess + dum
      
            data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1 
            data%dims%x_s = 1 ; data%dims%x_e = prob%n
            data%dims%c_s = data%dims%x_e + 1 
            data%dims%c_e = data%dims%x_e + data%dims%nc  
            data%dims%c_b = data%dims%c_e - prob%m
            data%dims%y_s = data%dims%c_e + 1 
            data%dims%y_e = data%dims%c_e + prob%m
            data%dims%y_i = data%dims%c_s + prob%m 
            data%dims%v_e = data%dims%y_e

!  Test for satisfactory termination

            IF ( data%QPP_inform%status /= 0 ) THEN
              inform%status = data%QPP_inform%status
              IF ( control%out > 0 .AND. control%print_level >= 5 )            &
                WRITE( control%out, "( ' status ', I3, ' after QPP_reorder ')")&
                 data%QPP_inform%status
              IF ( control%error > 0 .AND. control%print_level > 0 )           &
                WRITE( control%error, 2010 ) inform%status 
              IF ( control%out > 0 .AND. control%print_level > 0 .AND.         &
                   inform%status == - 4 ) WRITE( control%error, 2240 ) 
              CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
              CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,   &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,        &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,        &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map, data%QPP_control,              &
                                  data%QPP_inform )
              GO TO 800 
            END IF 

!  Record revised array lengths

            IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
              a_ne = prob%m * prob%n
            ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
              a_ne = prob%A%ptr( prob%m + 1 ) - 1
            ELSE
              a_ne = prob%A%ne 
            END IF

            IF ( printi ) WRITE( control%out,                                  &
              "( /, ' problem dimensions after removal of dependencies: ', /,  &
      &             ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )              &
                   prob%n, prob%m, a_ne
  
          END IF

!  Experiment!!

!         GO TO 10
        ELSE
          remap_more_freed = .FALSE.
        END IF
      END IF

!  Allocate additional real workspace

      reallocate = .TRUE.
      IF ( ALLOCATED( data%DZ_l ) ) THEN
        IF ( LBOUND( data%DZ_l, 1 ) /= data%dims%x_l_start .OR.               &
             UBOUND( data%DZ_l, 1 ) /= data%dims%x_l_end ) THEN 
          DEALLOCATE( data%DZ_l )
        ELSE ; reallocate = .FALSE. ; END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%DZ_l( data%dims%x_l_start : data%dims%x_l_end ),       &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          inform%bad_alloc = 'lpb:data%DZ_l' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%DZ_u ) ) THEN
        IF ( LBOUND( data%DZ_u, 1 ) /= data%dims%x_u_start .OR.               &
             UBOUND( data%DZ_u, 1 ) /= data%dims%x_u_end ) THEN 
          DEALLOCATE( data%DZ_u )
        ELSE ; reallocate = .FALSE. ; END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%DZ_u( data%dims%x_u_start : data%dims%x_u_end ),       &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          inform%bad_alloc = 'lpb:data%DZ_u' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%GRAD ) ) THEN
        IF ( SIZE( data%GRAD ) < prob%n ) THEN ; DEALLOCATE( data%GRAD )
        ELSE ; reallocate = .FALSE.
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%GRAD( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          inform%bad_alloc = 'lpb:data%GRAD' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%X_trial ) ) THEN
        IF ( SIZE( data%X_trial ) < prob%n ) THEN ; DEALLOCATE( data%X_trial )
        ELSE ; reallocate = .FALSE.
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%X_trial( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          inform%bad_alloc = 'lpb:data%X_trial' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%GRAD_X_phi ) ) THEN
        IF ( SIZE( data%GRAD_X_phi ) < prob%n ) THEN 
          DEALLOCATE( data%GRAD_X_phi )
        ELSE ; reallocate = .FALSE.
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%GRAD_X_phi( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          inform%bad_alloc = 'lpb:data%GRAD_X_phi' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%GRAD_C_phi ) ) THEN
        IF ( LBOUND( data%GRAD_C_phi, 1 ) /= data%dims%c_l_start .OR.          &
             UBOUND( data%GRAD_C_phi, 1 ) /= data%dims%c_u_end ) THEN 
          DEALLOCATE( data%GRAD_C_phi )
        ELSE ; reallocate = .FALSE.
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%GRAD_C_phi( data%dims%c_l_start : data%dims%c_u_end ),  &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          inform%bad_alloc = 'lpb:data%GRAD_C_phi' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%S ) ) THEN
        IF ( SIZE( data%S ) < data%dims%c_e ) THEN ; DEALLOCATE( data%S )
        ELSE ; reallocate = .FALSE.
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%S( data%dims%c_e ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN ; inform%bad_alloc = 'lpb:data%S' ; GO TO 900
        END IF
      END IF

      IF ( stat_required ) THEN

        reallocate = .TRUE.
        IF ( ALLOCATED( data%A_s ) ) THEN
          IF ( SIZE( data%A_s ) < prob%m ) THEN ; DEALLOCATE( data%A_s )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( data%A_s( prob%m ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb:data%A_s' ; GO TO 900
          END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( data%X_last ) ) THEN
          IF ( SIZE( data%X_last ) < prob%m ) THEN ; DEALLOCATE( data%X_last )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( data%X_last( prob%m ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb:data%X_last' ; GO TO 900
          END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( data%Y_last ) ) THEN
          IF ( SIZE( data%Y_last ) < prob%m ) THEN ; DEALLOCATE( data%Y_last )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( data%Y_last( prob%m ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb:data%Y_last' ; GO TO 900
          END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( data%Z_last ) ) THEN
          IF ( SIZE( data%Z_last ) < prob%n ) THEN ; DEALLOCATE( data%Z_last )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( data%Z_last( prob%n ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb:data%Z_last' ; GO TO 900
          END IF
        END IF
      END IF

!  =================
!  Solve the problem
!  =================

!  overlaps: DY_l => DIST_C_l_trial 
!            DY_u => DIST_C_u_trial
!            DELTA => VECTOR
!            RHS( : c_e ) => R 
!            DZ_l( x_l_start : x_l_end ) => DIST_X_l_trial
!            DZ_u( x_u_start : x_u_end ) => DIST_X_u_trial

      IF ( stat_required ) THEN
        CALL LPB_solve_main( data%dims, prob%n, prob%m,                       &
                             prob%G, prob%f, prob%A%val, prob%A%col,          &
                             prob%A%ptr, prob%C_l, prob%C_u, prob%X_l,        &
                             prob%X_u, prob%C, prob%X, prob%Y, prob%Z,        &
                             data%RES_x, data%X_trial, data%Y_trial,          &
                             data%RES_y, data%BEST_y, data%SOL, data%RES,     &
                             data%BEST, data%HX, data%GRAD_L, data%DIST_X_l,  &
                             data%DIST_X_u, data%Z_l, data%Z_u,               &
                             data%BARRIER_X, data%Y_l, data%DY_l,             &
                             data%DIST_C_l, data%Y_u, data%DY_u,              &
                             data%DIST_C_u, data%C, data%BARRIER_C,           &
                             data%SCALE_C, data%DELTA,                        &
                             data%RHS( : data%dims%c_e ),                     &
                             data%GRAD, data%GRAD_X_phi, data%GRAD_C_phi,     &
                             data%DZ_l( data%dims%x_l_start :                 &
                                        data%dims%x_l_end ),                  &
                             data%DZ_u( data%dims%x_u_start :                 &
                                        data%dims%x_u_end ), data%S,          &
                             data%Abycol_val, data%DIAG_X, data%DIAG_C,       &
                             data%IW, data%K_colptr, data%Abycol_ptr,         &
                             data%Abycol_row, data%K,        &
                             data%FACTORS, data%CNTL, control, inform,        &
                             C_last = data%A_s, X_last = data%X_last,         &
                             Y_last = data%Y_last, Z_last = data%Z_last,      &
                             C_stat = C_stat, B_Stat = B_Stat )
      ELSE
        CALL LPB_solve_main( data%dims, prob%n, prob%m,                       &
                             prob%G, prob%f, prob%A%val, prob%A%col,          &
                             prob%A%ptr, prob%C_l, prob%C_u, prob%X_l,        &
                             prob%X_u, prob%C, prob%X, prob%Y, prob%Z,        &
                             data%RES_x, data%X_trial, data%Y_trial,          &
                             data%RES_y, data%BEST_y, data%SOL, data%RES,     &
                             data%BEST, data%HX, data%GRAD_L, data%DIST_X_l,  &
                             data%DIST_X_u, data%Z_l, data%Z_u,               &
                             data%BARRIER_X, data%Y_l, data%DY_l,             &
                             data%DIST_C_l, data%Y_u, data%DY_u,              &
                             data%DIST_C_u, data%C, data%BARRIER_C,           &
                             data%SCALE_C, data%DELTA,                        &
                             data%RHS( : data%dims%c_e ),                     &
                             data%GRAD, data%GRAD_X_phi, data%GRAD_C_phi,     &
                             data%DZ_l( data%dims%x_l_start :                 &
                                        data%dims%x_l_end ),                  &
                             data%DZ_u( data%dims%x_u_start :                 &
                                        data%dims%x_u_end ), data%S,          &
                             data%Abycol_val, data%DIAG_X, data%DIAG_C,       &
                             data%IW, data%K_colptr, data%Abycol_ptr,         &
                             data%Abycol_row, data%K,                  &
                             data%FACTORS, data%CNTL, control, inform )
      END IF

  700 CONTINUE 

!  If some of the constraints were freed having first been fixed during 
!  the computation, refix them now

      IF ( remap_more_freed ) THEN

        CALL CPU_TIME( time )
        CALL QPP_restore( data%QPP_map_more_freed, data%QPP_inform,            &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,         &
!                           data%QPP_inform )
        CALL CPU_TIME( dum ) ; dum = dum - time
        inform%time%preprocess = inform%time%preprocess + dum
        data%dims = data%dims_save_more_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_more_depen
          j = data%Index_c_more_freed( i )
          prob%C_l( j ) = data%C_more_freed( i )
          prob%C_u( j ) = data%C_more_freed( i )
        END DO
      END IF

!  If some of the variables/constraints were fixed during the computation,
!  free them now

      IF ( remap_fixed ) THEN

        CALL CPU_TIME( time )
        CALL QPP_restore( data%QPP_map_fixed, data%QPP_inform,                 &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,              &
!                           data%QPP_inform )
        CALL CPU_TIME( dum ) ; dum = dum - time
        inform%time%preprocess = inform%time%preprocess + dum
        data%dims = data%dims_save_fixed

!  Release the temporarily fixed problem bounds

        DO i = 1, tiny_x
          j = data%Index_X_fixed( i )
          IF ( j > 0 ) THEN
            prob%X_u( j ) = data%X_fixed( i )
          ELSE
            prob%X_l( - j ) = data%X_fixed( i )
          END IF
        END DO
       
!  Do the same for the constraint bounds

        DO i = 1, tiny_c
          j = data%Index_C_fixed( i )
          IF ( j > 0 ) THEN
            prob%C_u( j ) = data%C_fixed( i )
          ELSE
            prob%C_l( - j ) = data%C_fixed( i )
          END IF
        END DO

      END IF

!  If some of the constraints were freed during the computation, refix them now

      IF ( remap_freed ) THEN

        CALL CPU_TIME( time )
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform,                 &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,              &
!                           data%QPP_inform )
        CALL CPU_TIME( dum ) ; dum = dum - time
        inform%time%preprocess = inform%time%preprocess + dum
        data%dims = data%dims_save_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          prob%C_l( j ) = data%C_freed( i )
          prob%C_u( j ) = data%C_freed( i )
        END DO

        CALL FDC_terminate( FDC_data, FDC_control, FDC_inform )

      END IF
      data%tried_to_remove_deps = .FALSE.

!  Retore the problem to its original form

  750 CONTINUE 
      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
        CALL CPU_TIME( time )

!  Full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_all = .TRUE. )

!  Restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  Solution recovery

        ELSE
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                             get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,   &
                             get_c = .TRUE. )
        END IF
!       CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
        CALL CPU_TIME( dum ) ; dum = dum - time
        inform%time%preprocess = inform%time%preprocess + dum
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  Compute total time

      CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
      IF ( printi ) WRITE( control%out, 2000 )                                 &
        inform%time%total, inform%time%preprocess, inform%time%phase1_total,   &
        inform%time%analyse, inform%time%factorize, inform%time%solve,         &
        inform%time%phase1_analyse, inform%time%phase1_factorize,              &
        inform%time%phase1_solve

  800 CONTINUE 
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving LPB_solve ' )" )

      RETURN  

  900 CONTINUE 
      inform%status = - 2
      IF ( printi ) WRITE( control%out, 2900 )                                 &
        inform%bad_alloc, inform%alloc_status

!  Compute total time

  920 CONTINUE 
      CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving LPB_solve ' )" )

      RETURN  

!  Non-executable statements

 2000 FORMAT( /, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',               &
              '-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=',                           &
              /, ' =', 28X,  'LPB timing statistics', 27X, '=',                &
              /, ' =                   total           preprocess ',           &
                 '        phase 1               =',                            &
              /, ' =', 12X, 0P, F12.2, 6X, F12.2, 6X, F12.2, 16X, '='          &
              /, ' =      analyse    factorize     solve',                     &
                 '      analyse    factorize     solve    =',                  &
              /, ' =', 6F12.2, 4x, '=',                                        &
              /, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',               &
                 '-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=' )
 2010 FORMAT( ' ', /, '   **  Error return ',I3,' from LPB ' ) 
 2040 FORMAT( '   **  Error return ', I6, ' from ', A15 ) 
 2050 FORMAT( /, ' ===================== end of feasible point phase ',        &
                 ' ==================== ' )
 2240 FORMAT( /, '  Warning - an entry from strict upper triangle of H given ' )
 2900 FORMAT( ' ** Message from -LPB_solve-', /,                               &
              ' Allocation error, for ', A, /, ' status = ', I6 ) 

!  End of LPB_solve

      END SUBROUTINE LPB_solve

!-*-*-*-*-   L P B _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LPB_solve_main( dims, n, m, G, f, A_val, A_col,               &
                                 A_ptr, C_l, C_u, X_l, X_u, C_RES, X, Y, Z,    &
                                 RES_x, X_trial, Y_trial, RES_y, BEST_y,       &
                                 SOL, RES, BEST, HX, GRAD_L,                   &
                                 DIST_X_l, DIST_X_u, Z_l, Z_u, BARRIER_X,      &
                                 Y_l, DIST_C_l_trial, DIST_C_l, Y_u,           &
                                 DIST_C_u_trial, DIST_C_u, C, BARRIER_C,       &
                                 SCALE_C, VECTOR, R,                           &
                                 GRAD, GRAD_X_phi, GRAD_C_phi,                 &
                                 DIST_X_l_trial, DIST_X_u_trial, S,            &
                                 Abycol_val, DIAG_X, DIAG_C, IW, K_colptr,     &
                                 Abycol_ptr, Abycol_row,                       &
                                 K, FACTORS, CNTL, control, inform,            &
                                 C_last, X_last, Y_last, Z_last,               &
                                 C_stat, B_Stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric matrix, 
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A and H are sparse
!
!  In order that many of the internal computations may be performed
!  efficiently, it is required that   
!  
!  * the variables are ordered so that their bounds appear in the order
!
!    free                      x
!    non-negativity      0  <= x
!    lower              x_l <= x
!    range              x_l <= x <= x_u   (x_l < x_u)
!    upper                     x <= x_u
!    non-positivity            x <=  0
!
!    Fixed variables are not permitted (ie, x_l < x_u for range variables). 
!
!  * the constraints are ordered so that their bounds appear in the order
!
!    equality           c_l  = A x
!    lower              c_l <= A x
!    range              c_l <= A x <= c_u
!    upper                     A x <= c_u
!
!    Free constraints are not permitted (ie, at least one of c_l and c_u
!    must be finite). Bounds with the value zero are not treated separately.
!
!  These transformations may be effected, in place, using the module
!  GALAHAD_QPP. The same module may subsequently used to recover the solution.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  dims is a structure of type LPB_data_type, whose components hold SCALAR
!   information about the problem on input. The components will be unaltered
!   on exit. The following components must be set:
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!                 
!   %x_free is an INTEGER variable, which must be set by the user to the
!    number of free variables. RESTRICTION: %x_free >= 0
!                 
!   %x_l_start is an INTEGER variable, which must be set by the user to the
!    index of the first variable with a nonzero lower (or lower range) bound.
!    RESTRICTION: %x_l_start >= %x_free + 1
!                 
!   %x_l_end is an INTEGER variable, which must be set by the user to the
!    index of the last variable with a nonzero lower (or lower range) bound.
!    RESTRICTION: %x_l_end >= %x_l_start
!                 
!   %x_u_start is an INTEGER variable, which must be set by the user to the
!    index of the first variable with a nonzero upper (or upper range) bound. 
!    RESTRICTION: %x_u_start >= %x_l_start
!                 
!   %x_u_end is an INTEGER variable, which must be set by the user to the
!    index of the last variable with a nonzero upper (or upper range) bound. 
!    RESTRICTION: %x_u_end >= %x_u_start
!                 
!   %c_equality is an INTEGER variable, which must be set by the user to the
!    number of equality constraints, m. RESTRICTION: %c_equality >= 0
!                 
!   %c_l_start is an INTEGER variable, which must be set by the user to the
!    index of the first inequality constraint with a lower (or lower range) 
!    bound. RESTRICTION: %c_l_start = %c_equality + 1 
!    (strictly, this information is redundant!)
!                 
!   %c_l_end is an INTEGER variable, which must be set by the user to the
!    index of the last inequality constraint with a lower (or lower range) 
!    bound. RESTRICTION: %c_l_end >= %c_l_start
!                 
!   %c_u_start is an INTEGER variable, which must be set by the user to the
!    index of the first inequality constraint with an upper (or upper range) 
!    bound. RESTRICTION: %c_u_start >= %c_l_start
!    (strictly, this information is redundant!)
!                 
!   %c_u_end is an INTEGER variable, which must be set by the user to the
!    index of the last inequality constraint with an upper (or upper range) 
!    bound. RESTRICTION: %c_u_end = %m
!    (strictly, this information is redundant!)
!
!   %nc is an INTEGER variable, which must be set by the user to the
!    value dims%c_u_end - dims%c_l_start + 1
!
!   %x_s is an INTEGER variable, which must be set by the user to the
!    value 1
!
!   %x_e is an INTEGER variable, which must be set by the user to the
!    value n
!
!   %c_s is an INTEGER variable, which must be set by the user to the
!    value dims%x_e + 1 
!
!   %c_e is an INTEGER variable, which must be set by the user to the
!    value dims%x_e + dims%nc
!
!   %c_b is an INTEGER variable, which must be set by the user to the
!    value dims%c_e - m
!
!   %y_s is an INTEGER variable, which must be set by the user to the
!    value dims%c_e + 1
!
!   %y_i is an INTEGER variable, which must be set by the user to the
!    value dims%c_s + m
!
!   %y_e is an INTEGER variable, which must be set by the user to the
!    value dims%c_e + m
!
!   %v_e is an INTEGER variable, which must be set by the user to the
!    value dims%y_e
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. 
!
!  G is a REAL array of length n, which must be set by
!   the user to the value of the gradient, g, of the linear term of the
!   quadratic objective function. The i-th component of G, i = 1, ....,
!   n, should contain the value of g_i.  
!  
!  A_* is used to hold the matrix A by rows. In particular:
!      A_col( : )   the column indices of the components of A
!      A_ptr( : )   pointers to the start of each row, and past the end of
!                   the last row. 
!      A_val( : )   the values of the components of A
!
!  C_l, C_u are REAL arrays of length m, which must be set by the user to 
!   the values of the arrays x_l and x_u of lower and upper bounds on x, ordered
!   as described above (strictly only C_l( dims%c_l_start : dims%c_l_end )
!   and C_u( dims%c_u_start : dims%c_u_end ) need be set, as the other
!   components are ignored!).
!  
!  X_l, X_u are REAL arrays of length n, which must be set by the user to 
!   the values of the arrays x_l and x_u of lower and upper bounds on x, ordered
!   as described above (strictly only X_l( dims%x_l_start : dims%x_l_end )
!   and X_u( dims%x_u_start : dims%x_u_end ) need be set, as the other
!   components are ignored!).
!  
!  C_RES is a REAL array of length m, which need not be set on entry. On exit,
!   the i-th component of C_RES will contain (A*x)_i, for i = 1, .... , m. 
!
!  X is a REAL array of length n, which must be set by
!   the user on entry to LPB_solve to give an initial estimate of the 
!   optimization parameters, x. The i-th component of X should contain 
!   the initial estimate of x_i, for i = 1, .... , n.  The estimate need 
!   not satisfy the simple bound constraints and may be perturbed by 
!   LPB_solve prior to the start of the minimization.  Any estimate which is 
!   closer to one of its bounds than control%prfeas may be reset to try to
!   ensure that it is at least control%prfeas from its bounds. On exit from 
!   LPB_solve, X will contain the best estimate of the optimization 
!   parameters found
!  
!  Y is a REAL array of length m, which must be set by the user
!   on entry to LPB_solve to give an initial estimates of the
!   optimal Lagrange multipiers, y. The i-th component of Y 
!   should contain the initial estimate of y_i, for i = 1, .... , m.  
!   Any estimate which is smaller than control%dufeas may be 
!   reset to control%dufeas. The dual variable for any variable with both
!   On exit from LPB_solve, Y will contain the best estimate of
!   the Lagrange multipliers found
!  
!  Z, is a REAL array of length n, which must be set by
!   on entry to LPB_solve to hold the values of the the dual variables 
!   associated with the simple bound constraints. 
!   Any estimate which is smaller than control%dufeas may be 
!   reset to control%dufeas. The dual variable for any variable with both
!   infinite lower and upper bounds need not be set. On exit from
!   LPB_solve, Z will contain the best estimates obtained
!  
!  control and inform are exactly as for LPB_solve
!
!  The remaining arguments are used as internal workspace, and need not be 
!  set on entry
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: B_stat
      REAL ( KIND = wp ), INTENT( IN ):: f
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, X_l, X_u, Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( n ) :: RES_x, X_trial, GRAD, GRAD_X_phi
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( m ) :: C_RES, Y_trial, BEST_y, RES_y
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: X_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: C_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( m ) :: Y_last
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( n ) :: Z_last
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%v_e ) :: SOL, RES, BEST
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%v_e ) :: HX
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%c_e ) :: S
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%c_e ) :: GRAD_L
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l, DIST_X_l_trial
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
        DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u, DIST_X_u_trial
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%x_free + 1 : dims%x_l_end ) ::  Z_l
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%x_u_start : n ) :: Z_u
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( dims%x_free + 1 : n ) :: BARRIER_X
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: Y_l, DIST_C_l,      &
                                                           DIST_C_l_trial 
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: Y_u, DIST_C_u,      &
                                                           DIST_C_u_trial
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, BARRIER_C,       &
                                                           SCALE_C, GRAD_C_phi
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%v_e ) :: VECTOR
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%c_e ) :: R

      INTEGER, ALLOCATABLE, DIMENSION( : ) ::                                  &
        IW, K_colptr, Abycol_ptr, Abycol_row
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) ::                       &
        DIAG_X, DIAG_C, Abycol_val

      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
      TYPE ( SILS_control ), INTENT( INOUT ) :: CNTL
      TYPE ( LPB_control_type ), INTENT( IN ) :: control
      TYPE ( LPB_inform_type ), INTENT( INOUT ) :: inform

!  GLTR derived types

      TYPE ( GLTR_data_type ) :: gltr_data
      TYPE ( GLTR_control_type ) :: gltr_control        
      TYPE ( GLTR_info_type ) :: gltr_info

!  Parameters

      INTEGER, PARAMETER :: hist = 5
      REAL, PARAMETER :: max_ratio = 2.0
      REAL ( KIND = wp ), PARAMETER :: eta_1 = tenm2
      REAL ( KIND = wp ), PARAMETER :: eta_2 = point9
      REAL ( KIND = wp ), PARAMETER :: sigma = point1
!     REAL ( KIND = wp ), PARAMETER :: sigma = 0.8_wp
!     REAL ( KIND = wp ), PARAMETER :: theta_df = ten
      REAL ( KIND = wp ), PARAMETER :: theta_df = one
!     REAL ( KIND = wp ), PARAMETER :: theta_df = 0.0001_wp
!     REAL ( KIND = wp ), PARAMETER :: theta_cs = ten
      REAL ( KIND = wp ), PARAMETER :: theta_cs = one
!     REAL ( KIND = wp ), PARAMETER :: theta_cs = 0.0001_wp
      REAL ( KIND = wp ), PARAMETER :: theta_min = ten ** 20
      REAL ( KIND = wp ), PARAMETER :: nu_1 = point01
      REAL ( KIND = wp ), PARAMETER :: eta = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: hmin = one
      REAL ( KIND = wp ), PARAMETER :: beta = 1.01_wp
      REAL ( KIND = wp ), PARAMETER :: radius_max = ten ** 20

!  Local variables

      INTEGER :: A_ne, i, j, l, lk, nnzk, nnzks, ldiag_c_l, ldiag_c_u
      INTEGER :: kplus, dplus, dzero, nbacts, factor, seq, liw, print_level
      INTEGER :: start_print, stop_print, cg_maxit, itref_max, ierr, ldiag_x
      INTEGER :: nbnds, out, error, zeig, precon, cg_iter, fact_hist
      INTEGER :: prev_factorization_integer, prev_factorization_real
      REAL :: dum, time, time_start, time_iter, time_last, time_mean, time_ratio
      REAL :: time_kkt, time_itsol, time_hist( 0 : hist - 1 ) = 0.0
      REAL ( KIND = wp ) :: mu, amax, teneps, ared, prered, old_mu
      REAL ( KIND = wp ) :: alpha, hmax, delta, zeta, c_feasmin
      REAL ( KIND = wp ) :: model, obj_trial, H_perturb, radius, old_radius
      REAL ( KIND = wp ) :: dufeas, ratio, norm_c, norm_d, norm_d_alt, sn
      REAL ( KIND = wp ) :: obj, phi, phi_trial, theta_c, theta_d
      REAL ( KIND = wp ) :: phi_model, phi_slope, obj_slope, obj_curv
      REAL ( KIND = wp ) :: p_min, p_max, d_min, d_max, step_max, pivot_tol
      REAL ( KIND = wp ) :: initial_radius, res_norm, small_x, perturb_min
      LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
      LOGICAL :: set_printm, printt, printi, printm, printw, printd, printe 
      LOGICAL :: one_fact, primal_hessian, got_time_kkt, alter_H
      LOGICAL :: first_iteration, got_ratio, start_major, new_prec, big_res
      LOGICAL :: auto, full_iteration, scaled_c, set_z, get_factors
      LOGICAL :: new_fact, refact, primal, stat_required, get_stat
      LOGICAL :: use_scale_c = .FALSE.

      CHARACTER ( LEN = 1 ) :: re, mo, bdry

!-----------------------------------------------
!   M A 2 7    V a r i a b l e s
!-----------------------------------------------

      TYPE ( SILS_finfo ) :: FINFO
      INTEGER :: sif = 52
!     LOGICAL :: generate_sif = .TRUE.
      LOGICAL :: generate_sif = .FALSE.

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering LPB_solve_main ' )" )

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed 

      IF ( generate_sif ) THEN
        WRITE( sif, "( 'NAME          LPBD_OUT', //, 'VARIABLES', / )" )
        DO i = 1, n
          WRITE( sif, "( '    X', I8 )" ) i
        END DO

        WRITE( sif, "( /, 'GROUPS', / )" )
        DO i = 1, n
          IF ( G( i ) /= zero )                                                &
            WRITE( sif, "( ' N  OBJ      ', ' X', I8, ' ', ES12.5 )" ) i, G( i )
        END DO
        DO i = 1, dims%c_l_start - 1
          DO j = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( j ), A_val( j )
          END DO
        END DO
        DO i = dims%c_l_start, dims%c_l_end
          DO j = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' G  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( j ), A_val( j )
          END DO
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          DO j = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' L  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( j ), A_val( j )
          END DO
        END DO

        WRITE( sif, "( /, 'CONSTANTS', / )" )
        DO i = 1, dims%c_l_end
          IF ( C_l( i ) /= zero )                                              &
          WRITE( sif, "( '    RHS      ', ' C', I8, ' ', ES12.5 )" ) i, C_l( i )
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          IF ( C_u( i ) /= zero )                                              &
          WRITE( sif, "( '    RHS      ', ' C', I8, ' ', ES12.5 )" ) i, C_u( i )
        END DO

        IF ( dims%c_u_start <= dims%c_l_end ) THEN
          WRITE( sif, "( /, 'RANGES', / )" )
          DO i = dims%c_u_start, dims%c_l_end
            WRITE( sif, "( '    RANGE    ', ' C', I8, ' ', ES12.5 )" )        &
              i, C_u( i ) - C_l( i )
          END DO
        END IF

        IF ( dims%x_free /= 0 .OR. dims%x_l_start <= n ) THEN
          WRITE( sif, "( /, 'BOUNDS', /, ' FR BND       ''DEFAULT''' )" )
          DO i = dims%x_free + 1, dims%x_l_start - 1
            WRITE( sif, "( ' LO BND       X', I8, ' ', ES12.5 )" ) i, zero
          END DO
          DO i = dims%x_l_start, dims%x_l_end
            WRITE( sif, "( ' LO BND       X', I8, ' ', ES12.5 )" ) i, X_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_u_end
            WRITE( sif, "( ' UP BND       X', I8, ' ', ES12.5 )" ) i, X_u( i )
          END DO
          DO i = dims%x_u_end + 1, n
            WRITE( sif, "( ' UP BND       X', I8, ' ', ES12.5 )" ) i, zero
          END DO
        END IF

        WRITE( sif, "( /, 'START POINT', / )" )
        DO i = 1, n
          IF ( X( i ) /= zero )                                                &
            WRITE( sif, "( ' V  START    ', ' X', I8, ' ', ES12.5 )" ) i, X( i )
        END DO

        WRITE( sif, "( /, 'ENDATA' )" )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

      CALL CPU_TIME( time_start ) ; time_last = time_start

!  ===========================
!  Control the output printing
!  ===========================

      print_level = 0
      IF ( control%start_print < 0 ) THEN
        start_print = - 1
      ELSE
        start_print = control%start_print
      END IF

      IF ( control%stop_print < 0 ) THEN
        stop_print = control%maxit
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

!  Start setting control parameters

      IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
        printe = set_printe ; printi = set_printi ; printt = set_printt
        printm = set_printm ; printw = set_printw ; printd = set_printd
        print_level = control%print_level
      ELSE
        printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
        printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
        print_level = 0
      END IF

      factor = control%factor
      IF ( factor < 0 .OR. factor > 2 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( ' factor = ', I6, ' out of range [0,2]. Reset to 0') ") factor
        factor = 0
      END IF

      precon = control%precon 
      auto = precon <= 0
      IF ( .NOT. auto ) THEN
        IF ( precon >= 3 ) THEN
          IF ( printi ) WRITE( out,                                            &
            "( ' precon = ', I6, ' out of range [0,2]. Reset to 2') ") precon
          precon = 2
        END IF
      END IF

!  If there are no variables, exit

      IF ( n == 0 ) THEN 
        i = COUNT( ABS( C_l( : dims%c_equality ) ) > control%stop_p ) +        &
            COUNT( C_l( dims%c_l_start : dims%c_l_end ) > control%stop_p ) +   &
            COUNT( C_u( dims%c_u_start : dims%c_u_end ) < - control%stop_p )
        IF ( i == 0 ) THEN
          inform%status = 0
        ELSE
          inform%status = - 6
        END IF
        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 1, m ; C_RES( i ) = zero ; Y( i ) = zero ; END DO
        ELSE
          C_RES = zero ; Y = zero
        END IF
        inform%obj = f
        GO TO 810
      END IF 

!  Record array sizes

      A_ne = A_ptr( m + 1 ) - 1

!  Set control parameters

      itref_max = control%itref_max
      cg_maxit = control%cg_maxit
      IF ( cg_maxit < 0 ) cg_maxit = dims%c_b + 2
      seq = 0

      dufeas = control%dufeas 
      pivot_tol = control%pivot_tol
      initial_radius = control%initial_radius
      scaled_c = .TRUE.

      primal = control%primal ; primal_hessian = primal
      start_major = .TRUE.
      small_x = epsmch
      teneps = 10.0 * epsmch
      perturb_min = epsmch ** 0.25
      c_feasmin = SQRT( epsmch )
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )
      IF ( stat_required ) THEN
        B_stat  = 0
        C_stat( : dims%c_equality ) = - 1
        C_stat( dims%c_equality + 1 : ) = 0
      END IF
      get_stat = .FALSE.

!  Initialize counts

      nbacts = 0

      CNTL%u = pivot_tol ; CNTL%pivoting = 1
      CNTL%lp = - 1 ; CNTL%mp = - 1 ; CNTL%wp = - 1
      
      IF ( print_level > 4 ) THEN
        IF ( print_level < 10 ) THEN
          CNTL%ldiag = 1
        ELSE
          CNTL%ldiag = 2
        END IF
      END IF
          
!  If required, write out the problem

      IF ( printd ) THEN
        WRITE( out, 2180 ) ' g ', ( G( i ), i = 1, n )
        WRITE( out, 2190 ) ' A ', ( ( i, A_col( l ), A_val( l ),               &
                           l = A_ptr( i ), A_ptr( i + 1 ) - 1 ), i = 1, m )
      END IF 

!  Record the initial point, move the starting point away from any bounds, 
!  and move that for dual variables away from zero

      nbnds = 0
      set_z = .FALSE.

!  The variable is free

      IF ( printd ) THEN
        WRITE( out,                                                            &
        "( /, 5X, 'i', 6x, 'x', 10X, 'x_l', 9X, 'x_u', 9X, 'z_l', 9X, 'z_u')" )
        DO i = 1, dims%x_free
          WRITE( out, "( I6, ES12.4, 4( '      -     ') )" ) i, X( i )
        END DO
      END IF

!  The variable is a non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        IF ( X( i ) <= small_x ) THEN 
!         write(6,"('i,X, small',I5, 2ES12.4)") i, X(i), small_x
          inform%status = - 6
          GO TO 700
        END IF
        nbnds = nbnds + 1
        Z_l( i ) = MAX( Z( i ), dufeas )
        IF ( printd ) WRITE( out, "( I6, 2ES12.4, '      -     ', ES12.4,      &
       &  '      -     ' )" ) i, X( i ), zero, Z_l( i )
      END DO

!  The variable has just a lower bound

      DO i = dims%x_l_start, dims%x_u_start - 1
        IF ( X( i ) - X_l( i ) <= small_x ) THEN 
          inform%status = - 6
          GO TO 700
        END IF
        nbnds = nbnds + 1
        Z_l( i ) = MAX( Z( i ), dufeas )
        DIST_X_l( i ) = X( i ) - X_l( i )
        IF ( printd ) WRITE( out, "( I6, 2ES12.4, '      -     ', ES12.4,      &
       &  '      -     ' )" ) i, X( i ), X_l( i ), Z_l( i )
      END DO

!  The variable has both lower and upper bounds

      DO i = dims%x_u_start, dims%x_l_end

!  Check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

        IF ( X_u( i ) - X_l( i ) <= epsmch ) THEN 
          inform%status = - 6
          GO TO 700
        END IF
        nbnds = nbnds + 2
        IF ( X( i ) - X_l( i ) <= small_x .OR.                                 &
             X_u( i ) - X( i ) <= small_x ) THEN 
          inform%status = - 6
          GO TO 700
        END IF
        Z_l( i ) = MAX(   ABS( Z( i ) ),   dufeas )  
        Z_u( i ) = MIN( - ABS( Z( i ) ), - dufeas )
        DIST_X_l( i ) = X( i ) - X_l( i ) ; DIST_X_u( i ) = X_u( i ) - X( i )
        IF ( printd ) WRITE( out, "( I6, 5ES12.4 )" )                          &
             i, X( i ), X_l( i ), X_u( i ), Z_l( i ), Z_u( i )
      END DO

!  The variable has just an upper bound

      DO i = dims%x_l_end + 1, dims%x_u_end
        nbnds = nbnds + 1
        IF ( X_u( i ) - X( i ) <= small_x ) THEN 
          inform%status = - 6
          GO TO 700
        END IF
        Z_u( i ) = MIN( Z( i ), - dufeas ) 
        DIST_X_u( i ) = X_u( i ) - X( i )
        IF ( printd ) WRITE( out, "( I6, ES12.4, '      -     ', ES12.4,       &
       &  '      -     ', ES12.4 )" ) i, X( i ), X_u( i ), Z_u( i )
      END DO

!  The variable is a non-positivity

      DO i = dims%x_u_end + 1, n
        nbnds = nbnds + 1
        IF ( - X( i ) <= small_x ) THEN 
          inform%status = - 6
          GO TO 700
        END IF
        Z_u( i ) = MIN( Z( i ), - dufeas ) 
        IF ( printd ) WRITE( out, "( I6, ES12.4, '      -     ', ES12.4,       &
       &  '      -     ',  ES12.4 )" ) i, X( i ), zero, Z_u( i )
      END DO

      DIST_X_l_trial = DIST_X_l ; DIST_X_u_trial = DIST_X_u
      set_z = .TRUE.

!  Compute the value of the constraint, and their residuals

      IF ( m > 0 ) THEN
        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i =  1, m ; R( i ) = zero ; END DO
        ELSE
          R( : m ) = zero
        END IF
        CALL LPB_AX( m, R( : m), m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )

        IF ( printd ) THEN
          WRITE( out,                                                          &
          "( /, 5X,'i', 6x, 'c', 10X, 'c_l', 9X, 'c_u', 9X, 'y_l', 9X, 'y_u' )")
          DO i = 1, dims%c_l_start - 1
            WRITE( out, "( I6, 3ES12.4 )" ) i, R( i ), C_l( i ), C_u( i )
          END DO
        END IF

!  The constraint has just a lower bound

        DO i = dims%c_l_start, dims%c_u_start - 1
          nbnds = nbnds + 1

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( R( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MAX( R( i ) / SCALE_C( i ),                                 &
                        C_l( i ) + c_feasmin * MAX( one, ABS( C_l( i ) ) ) )
          DIST_C_l( i ) = C( i ) - C_l( i )
          Y_l( i ) = MAX( Y( i ), dufeas )
          IF ( printd ) WRITE( out,  "( I6, 2ES12.4, '      -     ', ES12.4,   &
         &  '      -     ' )" ) i, C( i ), C_l( i ), Y_l( i )
        END DO

!  The constraint has both lower and upper bounds

        DO i = dims%c_u_start, dims%c_l_end
          nbnds = nbnds + 2

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( R( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_l( i ) = C_l( i ) / SCALE_C( i ) 
          C_u( i ) = C_u( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MIN( C_u( i ) - c_feasmin * MAX( one, ABS( C_u( i ) ) ),    &
                        MAX( R( i ) / SCALE_C( i ),                            &
                          C_l( i ) + c_feasmin * MAX( one, ABS( C_l( i ) ) ) ) )
          DIST_C_l( i ) = C( i ) - C_l( i ) 
          DIST_C_u( i ) = C_u( i ) - C( i )
          Y_l( i ) = MAX( Y( i ),   dufeas )
          Y_u( i ) = MIN( Y( i ), - dufeas )
          IF ( printd ) WRITE( out, "( I6, 5ES12.4 )" )                        &
            i, C( i ), C_l( i ), C_u( i ), Y_l( i ), Y_u( i )
        END DO

!  The constraint has just an upper bound

        DO i = dims%c_l_end + 1, dims%c_u_end
          nbnds = nbnds + 1

!  Compute an appropriate scale factor

          IF ( use_scale_c ) THEN
            SCALE_C( i ) = MAX( one, ABS( R( i ) ) )
          ELSE
            SCALE_C( i ) = one
          END IF

!  Scale the bounds

          C_u( i ) = C_u( i ) / SCALE_C( i )

!  Compute an appropriate initial value for the slack variable

          C( i ) = MIN( R( i ) / SCALE_C( i ),                                 &
                        C_u( i ) - c_feasmin * MAX( one, ABS( C_u( i ) ) ) )
          DIST_C_u( i ) = C_u( i ) - C( i )
          Y_u( i ) = MIN( Y( i ), - dufeas ) 
          IF ( printd ) WRITE( out, "( I6, ES12.4, '      -     ', ES12.4,     &
         &  '      -     ', ES12.4 )" ) i, C( i ), C_u( i ), Y_u( i )
        END DO
      END IF

      scaled_c = .TRUE.
      IF ( printi )                                                            &
        WRITE( out, "( /, ' >>>>> factorization package SILS used <<<<<', / )" )
      IF ( printi .AND. m > 0 .AND. dims%c_l_start <= dims%c_u_end )           &
        WRITE( out, "( ' largest/smallest scale factor ', 2ES12.4 )" )         &
          MAXVAL( SCALE_C ), MINVAL( SCALE_C )

!  Find the largest components of A and H

      IF ( A_ne > 0 ) THEN
        amax = MAXVAL( ABS( A_val( : A_ne ) ) )
      ELSE
        amax = zero
      END IF

      hmax = hmin

      IF ( printi ) WRITE( out, 2090 ) amax, hmax 

!  Now, calculate the value .... 

      obj = DOT_PRODUCT( X, G )
      inform%obj = obj + f

!  ... and gradient of the objective function

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, n ; GRAD( i ) = G( i ) ; END DO
      ELSE
        GRAD = G
      END IF

!  ===============
!  Outer iteration
!  ===============

!  Initialize penalty parameter, mu

      norm_c = DOT_PRODUCT( X( dims%x_free + 1 : dims%x_l_start - 1 ),         &
                            Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ) +    &
               DOT_PRODUCT( DIST_X_l( dims%x_l_start : dims%x_l_end ),         &
                            Z_l( dims%x_l_start : dims%x_l_end ) ) -           &
               DOT_PRODUCT( DIST_X_u( dims%x_u_start : dims%x_u_end ),         &
                            Z_u( dims%x_u_start : dims%x_u_end ) ) +           &
               DOT_PRODUCT( X( dims%x_u_end + 1 : n ),                         &
                            Z_u( dims%x_u_end + 1 : n ) ) +                    &
               DOT_PRODUCT( DIST_C_l( dims%c_l_start : dims%c_l_end ),         &
                            Y_l( dims%c_l_start : dims%c_l_end ) ) -           &
               DOT_PRODUCT( DIST_C_u( dims%c_u_start : dims%c_u_end ),         &
                            Y_u( dims%c_u_start : dims%c_u_end ) )

      IF ( nbnds > 0 ) THEN
        norm_c = ten ** ( 2 * ( NINT( LOG10( norm_c / nbnds ) ) / 2  ) )
      ELSE
        norm_c = zero
      END IF

      IF ( control%muzero < zero ) THEN
        mu = norm_c
      ELSE
        mu = control%muzero
      END IF

!  If the starting point is very close to one of its bounds, 
!  the feasible region likely has no interior. Be cautious

      old_mu = mu ; zeta = point01 * point01

!  Initialize convergence tolerances, theta_c, theta_d and theta_e

      theta_c = MIN( theta_min, theta_cs * mu )
      theta_d = MIN( theta_min, theta_df * mu )

!  Prepare for the major iteration

      inform%iter = 0 ; inform%nfacts = 0 ; ratio = - one
      got_time_kkt = .FALSE. ; time_kkt = 0.0

!  Intialize GLTR data

      CALL GLTR_initialize( gltr_data, gltr_control )
      gltr_control%stop_relative = control%inner_stop_relative
      gltr_control%stop_absolute = control%inner_stop_absolute
      gltr_control%fraction_opt = control%inner_fraction_opt
      gltr_control%unitm = .FALSE.
      gltr_control%steihaug_toint = .FALSE.
      gltr_control%error = control%error
      gltr_control%out = control%out
      gltr_control%print_level = print_level - 1
      gltr_control%itmax = control%cg_maxit
      gltr_control%Lanczos_itmax = 5

!  If the preconditioner is to be picked automatically, start with the identity.
!  Otherwise, use the Hessian of the barrier function

      one_fact = .FALSE.
      new_fact = .TRUE.
      IF ( printi ) WRITE( out, "( ' ' )" )
      IF ( auto ) THEN
        IF ( printi ) WRITE( out, 2400 )
        precon = 2

!  fact_hist indicates which factors are currently being used. Possible values:
!   1 barrier factors used
!   2 identity used
!   3 barrier factors used for the final time
!   4 identity used for the final time
!   5 barrier factors as a last resort

        fact_hist = 4
      END IF

!  Assign space for the preconditioner - the analyis phase
!  -------------------------------------------------------

!  The preconditioner will be of the form
!     ( M  A(trans) )
!     ( A      0    )
!  where M is a suitable, second-order sufficient symmetric matric

!  ==================================================
!  Analyse the sparsity pattern of the preconditioner
!  ==================================================

      CALL CPU_TIME( time ) 
      CALL LPB_analyse( dims, n, m, A_ne, A_val, A_col, A_ptr, SCALE_C,        &
                         factor, precon, nnzks, lk,                            &
                         liw, ldiag_x, ldiag_c_l, ldiag_c_u, IW, Abycol_val,   &
                         Abycol_row, Abycol_ptr, K_colptr, DIAG_X, DIAG_C,     &
                         K, FACTORS, CNTL, print_level, control, inform )
      CALL CPU_TIME( dum ) ; dum = dum - time

      IF ( printt ) WRITE( out, "( ' analysis time = ', F10.2 ) " ) dum
      inform%time%analyse = inform%time%analyse + dum

      refact = .TRUE. ; re = ' ' ; mo = ' '
      new_prec = .FALSE. ; full_iteration = .FALSE. ; got_ratio = .FALSE.

      cg_iter = 0 ; inform%cg_iter = cg_iter

!  Compute the value of the barrier function, phi

      phi = LPB_barrier_value( dims, n, obj, X, DIST_X_l, DIST_X_u,            &
                                DIST_C_l, DIST_C_u, mu )

      gltr_control%boundary = .FALSE.
      gltr_info%status = 1 ; gltr_info%negative_curvature = .TRUE.

      DO  ! outer iteration loop

!  The vectors x, z_l and z_u satisfy
!  a) A x = b
!  b) X_l < x < X_u, z_l > 0, z_u < 0

!  Set the initial trust-region radius radius
   
        IF ( initial_radius <= zero ) THEN
          radius = ( ten ** 3 ) * MAX( one, mu )
        ELSE
          radius = MIN( initial_radius, radius_max )
        END IF
        old_radius = radius

!  ===============
!  Inner iteration
!  ===============

 inner: DO  ! inner iteration loop

          CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
          time_iter = time - time_last ; time_last = time

!         WRITE( out, "( ' time_iter ', F10.2 )" ) time_iter

!  Estimate the time for an iteration with the full factorization

          IF ( auto ) THEN
            IF( fact_hist == 2 .AND.                                           &
               ( .NOT. got_time_kkt .AND. full_iteration ) ) THEN
              got_time_kkt = .TRUE.
              time_kkt = time_iter
            END IF

!  Check the effectiveness of the preconditioner over the previous iteration

            IF ( .NOT. start_major ) THEN 
              IF ( seq > 0 ) THEN
                 time_mean = SUM( time_hist( 0 : MIN( seq - 1, hist - 1 ) ) ) &
                             / MIN( seq, hist )
                 IF ( time_mean > 0.0 ) THEN
                   time_ratio = time_iter / time_mean
                 ELSE
                   time_ratio = 0.0
                 END IF
              ELSE
                 time_mean = 0.0 ; time_ratio = 0.0
              END IF
   
              time_hist( MOD( seq, hist ) ) = time_iter
              seq = seq + 1

!  The previous preconditioner appears to be ineffective. Try another

              SELECT CASE( fact_hist ) 

!  The time/iteration has significantly increased. See if a full factorization
!  is better

              CASE ( 1 )
                IF ( seq >= hist .AND. time_ratio > max_ratio ) THEN
                  IF ( printi )                                                &
                     WRITE( out, 2370 ) time_iter, time_mean, max_ratio
                  new_prec = .TRUE.
                  time_itsol = time_iter
                  precon = 1 ; fact_hist = 2 ; factor = control%factor
                END IF

!  The full factorization is more expensive than the barrier factorization.
!  Revert to the latter

              CASE ( 2 )
               IF ( got_time_kkt .AND. time_kkt > time_itsol ) THEN
                 IF ( printi ) WRITE( out, 2380 ) time_kkt, time_itsol
                 new_prec = .TRUE.
                 precon = 2 ; fact_hist = 3 ; factor = control%factor
               END IF

!  The barrier factorization is more expensive than the full factorization.
!  Revert to the latter

              CASE ( 3 )
                IF ( time_iter > time_kkt ) THEN
                  IF ( printi ) WRITE( out, 2390 ) time_iter, time_kkt
                  new_prec = .TRUE.
                  precon = 1 ; fact_hist = 4 ; factor = control%factor
                END IF
              END SELECT
            END IF
          END IF

!  Compute the derivatives of the barrier function, and the norm of the
!  complentarity, || (X-L) z_l - \mu e, (X-U) z_u - \mu e \|

!  Problem variables:

          p_min = infinity ; p_max = zero ; d_min = infinity ; d_max = zero 
          norm_c = zero

          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, dims%x_free ; GRAD_X_phi( i ) = GRAD( i ) ; END DO
          ELSE
            GRAD_X_phi( : dims%x_free ) = GRAD( : dims%x_free )
          END IF
          DO i = dims%x_free + 1, dims%x_l_start - 1
            GRAD_X_phi( i ) = GRAD( i ) - mu / X( i )
            p_min = MIN( p_min, X( i ) )
            p_max = MAX( p_max, X( i ) )
            d_min = MIN( d_min, Z_l( i ) )
            d_max = MAX( d_max, Z_l( i ) )
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_l( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS( X( i ) * Z_l( i ) - mu )
          END DO

          DO i = dims%x_l_start, dims%x_u_start - 1
            GRAD_X_phi( i ) = GRAD( i ) - mu / DIST_X_l( i )
            p_min = MIN( p_min, DIST_X_l( i ) )
            p_max = MAX( p_max, DIST_X_l( i ) )
            d_min = MIN( d_min, Z_l( i ) )
            d_max = MAX( d_max, Z_l( i ) )
            IF ( ABS( DIST_X_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS(   DIST_X_l( i ) * Z_l( i ) - mu )
          END DO

          DO i = dims%x_u_start, dims%x_l_end
            GRAD_X_phi( i ) = GRAD( i ) - mu / DIST_X_l( i )                   &
                                        + mu / DIST_X_u( i )
            p_min = MIN( p_min, DIST_X_l( i ), DIST_X_u( i ) )
            p_max = MAX( p_max, DIST_X_l( i ), DIST_X_u( i ) )
            d_min = MIN( d_min, Z_l( i ), - Z_u( i ) )
            d_max = MAX( d_max, Z_l( i ), - Z_u( i ) )
            IF ( MIN( ABS( DIST_X_l( i ) ), ABS( DIST_X_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) - mu ),    &
                                    ABS( - DIST_X_u( i ) * Z_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, MAX( ABS(   DIST_X_l( i ) * Z_l( i ) - mu ),      &
!                     ABS( - DIST_X_u( i ) * Z_u( i ) - mu ) )
          END DO

          DO i = dims%x_l_end + 1, dims%x_u_end
            GRAD_X_phi( i ) = GRAD( i ) + mu / DIST_X_u( i )
            p_min = MIN( p_min, DIST_X_u( i ) )
            p_max = MAX( p_max, DIST_X_u( i ) )
            d_min = MIN( d_min, - Z_u( i ) )
            d_max = MAX( d_max, - Z_u( i ) )
            IF ( ABS( DIST_X_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_X_u( i ) * Z_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS( - DIST_X_u( i ) * Z_u( i ) - mu )
          END DO

          DO i = dims%x_u_end + 1, n
            GRAD_X_phi( i ) = GRAD( i ) - mu / X( i )
            p_min = MIN( p_min, - X( i ) )
            p_max = MAX( p_max, - X( i ) )
            d_min = MIN( d_min, - Z_u( i ) )
            d_max = MAX( d_max, - Z_u( i ) )
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'X', i6, ES12.4 )" ) &
!             i, ABS( X( i ) * Z_l( i ) - mu )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            GRAD_C_phi( i ) = - mu / DIST_C_l( i )
            p_min = MIN( p_min, DIST_C_l( i ) )
            p_max = MAX( p_max, DIST_C_l( i ) )
            d_min = MIN( d_min, Y_l( i ) )
            d_max = MAX( d_max, Y_l( i ) )
            IF ( ABS( DIST_C_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( DIST_C_l( i ) * Y_l( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'C', i6, ES12.4 )" ) &
!             i, ABS( DIST_C_l( i ) * Y_l( i ) - mu )
          END DO

          DO i = dims%c_u_start, dims%c_l_end
            GRAD_C_phi( i ) = - mu / DIST_C_l( i ) + mu / DIST_C_u( i )
            p_min = MIN( p_min, DIST_C_l( i ), DIST_C_u( i ) )
            p_max = MAX( p_max, DIST_C_l( i ), DIST_C_u( i ) )
            d_min = MIN( d_min, Y_l( i ), - Y_u( i ) )
            d_max = MAX( d_max, Y_l( i ), - Y_u( i ) )
            IF ( MIN( ABS( DIST_C_l( i ) ), ABS( DIST_C_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_C_l( i ) * Y_l( i ) - mu ),    &
                                    ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'C', i6, ES12.4 )" ) &
!             i, MAX( ABS(   DIST_C_l( i ) * Y_l( i ) - mu ),      &
!                     ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) )
          END DO

          DO i = dims%c_l_end + 1, dims%c_u_end
            GRAD_C_phi( i ) = mu / DIST_C_u( i )
            p_min = MIN( p_min, DIST_C_u( i ) )
            p_max = MAX( p_max, DIST_C_u( i ) )
            d_min = MIN( d_min, - Y_u( i ) )
            d_max = MAX( d_max, - Y_u( i ) )
            IF ( ABS( DIST_C_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) )
!           IF ( inform%iter == 41 ) WRITE( 6, "( 'C', i6, ES12.4 )" ) &
!             i, ABS( - DIST_C_u( i ) * Y_u( i ) - mu ) 
          END DO

          IF ( printt ) WRITE( out, 2160 ) p_min, p_max, d_min, d_max

!  Build the model of the barrier function
!  ---------------------------------------

!  Construct the quadratic model
!  m(s) = phi + <s,grad phi> + 1/2 <s,(Hess f + (X-L)(-2) + (U-X)(-2))s>

!  Compute the Hessian matrix of the barrier terms

!  problem variables:

          IF ( primal_hessian ) THEN
            DO i = dims%x_free + 1, dims%x_l_start - 1
              BARRIER_X( i ) = MAX( bar_min, old_mu / X( i ) ** 2 )
            END DO
            DO i = dims%x_l_start, dims%x_u_start - 1
              BARRIER_X( i ) = MAX( bar_min,old_mu / DIST_X_l( i ) ** 2 )
            END DO
            DO i = dims%x_u_start, dims%x_l_end
              BARRIER_X( i ) = MAX( bar_min, old_mu / DIST_X_l( i ) ** 2 +     &
                                             old_mu / DIST_X_u( i ) ** 2 )
            END DO
            DO i = dims%x_l_end + 1, dims%x_u_end
              BARRIER_X( i ) = MAX( bar_min, old_mu / DIST_X_u( i ) ** 2 )
            END DO
            DO i = dims%x_u_end + 1, n
              BARRIER_X( i ) = MAX( bar_min, old_mu / X( i ) ** 2 )
            END DO
          ELSE
            DO i = dims%x_free + 1, dims%x_l_start - 1
              BARRIER_X( i ) = MAX( bar_min, Z_l( i ) / X( i ) )
            END DO
            DO i = dims%x_l_start, dims%x_u_start - 1
              BARRIER_X( i ) = MAX( bar_min, Z_l( i ) / DIST_X_l( i ) )
            END DO
            DO i = dims%x_u_start, dims%x_l_end
              BARRIER_X( i ) = MAX( bar_min, Z_l( i ) / DIST_X_l( i ) -        &
                                             Z_u( i ) / DIST_X_u( i ) )
            END DO
            DO i = dims%x_l_end + 1, dims%x_u_end
              BARRIER_X( i ) = MAX( bar_min, - Z_u( i ) / DIST_X_u( i ) )
            END DO
            DO i = dims%x_u_end + 1, n
              BARRIER_X( i ) = MAX( bar_min, Z_u( i ) / X( i ) )
            END DO
          END IF

!  slack variables:

!         IF ( control%array_syntax_worse_than_do_loop ) THEN
!           DO i = dims%c_l_start, dims%c_u_end ; BARRIER_C( i ) = zero ; END DO
!         ELSE
!           BARRIER_C( dims%c_l_start : dims%c_u_end ) = zero
!         END IF

          IF ( primal_hessian ) THEN
            DO i = dims%c_l_start, dims%c_u_start - 1
              BARRIER_C( i ) = MAX( bar_min, old_mu / DIST_C_l( i ) ** 2 )
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              BARRIER_C( i ) = MAX( bar_min, old_mu / DIST_C_l( i ) ** 2 +     &
                                             old_mu / DIST_C_u( i ) ** 2 )
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              BARRIER_C( i ) = MAX( bar_min, old_mu / DIST_C_u( i ) ** 2 )
            END DO
          ELSE
            DO i = dims%c_l_start, dims%c_u_start - 1
              BARRIER_C( i ) = MAX( bar_min, Y_l( i ) / DIST_C_l( i ) )
            END DO
            DO i = dims%c_u_start, dims%c_l_end
              BARRIER_C( i ) = MAX( bar_min, Y_l( i ) / DIST_C_l( i ) -        &
                                             Y_u( i ) / DIST_C_u( i ) )
            END DO
            DO i = dims%c_l_end + 1, dims%c_u_end
              BARRIER_C( i ) = MAX( bar_min, - Y_u( i ) / DIST_C_u( i ) )
            END DO
          END IF

!  If required, form and factorize the preconditioner
!  --------------------------------------------------

          mo = ' '
          IF ( refact ) THEN 

!  Only refactorize if M has changed

            IF ( one_fact .AND. .NOT. new_fact ) THEN
              re = ' ' 
            ELSE
              re = 'r'
              new_fact = .FALSE.

!  The previous preconditioner appears to be ineffective. Try another

              IF ( new_prec ) THEN
                seq = 0
                IF ( printi ) THEN
                  WRITE( out, 2200 )
                  WRITE( out, 2400 )
                END IF

                prev_factorization_integer = inform%factorization_integer
                prev_factorization_real = inform%factorization_real
                IF ( printt )                                                  &
                  WRITE( out, "( ' previous int, real space used ', 2I12 )" )  &
                    inform%factorization_integer, inform%factorization_real

                CALL CPU_TIME( time ) 
                CALL LPB_analyse(                                              &
                          dims, n, m, A_ne, A_val, A_col, A_ptr,               &
                          SCALE_C, factor, precon, nnzks, lk,   &
                          liw, ldiag_x, ldiag_c_l, ldiag_c_u,                  &
                          IW, Abycol_val, Abycol_row,                          &
                          Abycol_ptr, K_colptr, DIAG_X, DIAG_C,                &
                          K, FACTORS, CNTL, print_level, control, inform )
                CALL CPU_TIME( dum ) ; dum = dum - time

                IF ( printt )                                                  &
                  WRITE( out, "( ' int, real space requested ', 2I12, /,       &
                 &  ' analysis time = ', F10.2 ) " )                           &
                    inform%factorization_integer, inform%factorization_real, dum
                inform%time%analyse = inform%time%analyse + dum

                IF ( auto .AND. precon /= 2 ) THEN
                  IF ( inform%factorization_real >                             &
                       max_real_store_ratio * prev_factorization_real          &
                       .OR. inform%factorization_integer >                     &
                       max_integer_store_ratio * prev_factorization_integer )  &
                     THEN
                    IF ( printi .AND. inform%factorization_real >              &
                         max_real_store_ratio * prev_factorization_real )      &
                      WRITE( out, 2360 ) inform%factorization_real,            &
                         prev_factorization_real, max_real_store_ratio
                    IF (  printi .AND. inform%factorization_integer >          &
                       max_integer_store_ratio * prev_factorization_integer )  &
                      WRITE( out, 2350 ) inform%factorization_integer,         &
                         prev_factorization_integer, max_integer_store_ratio
                    refact = .TRUE. ; new_prec = .TRUE.
                    precon = 2 ; fact_hist = 5 ; factor = control%factor
!                   WRITE( out, "( ' change precon to ', I2 )" ) precon
                    CYCLE inner
                  END IF
                END IF

                new_prec = .FALSE.
              END IF

              CNTL%liw = MAX( 2 * inform%factorization_integer, control%indmin )
              CNTL%la = MAX( 2 * inform%factorization_real, control%valmin )

              H_perturb = zero ; alter_H = .FALSE.

              CALL CPU_TIME( time ) 
              IF ( printw ) WRITE( out,                                        &
                   "( ' ............... factorization ............... ' )" )

   50         CONTINUE

!  For the Schur complement matrix
!  ===============================

              get_factors = .TRUE.
              IF ( factor == 0 .OR. factor == 1 ) THEN

!  Set the diagonal values of M in DIAG_X

                 SELECT CASE( precon )

!  * M is a diagonal matrix

                 CASE DEFAULT
                   IF ( control%array_syntax_worse_than_do_loop ) THEN
                     DO i = 1, ldiag_x ; DIAG_X( i ) = k_diag ; END DO
                     DO i = ldiag_c_l, ldiag_c_u ; DIAG_C( i ) = k_diag ; END DO
                   ELSE
                     DIAG_X = k_diag
                     DIAG_C = k_diag
                   END IF

                 CASE ( 2 )
             
                   IF ( control%array_syntax_worse_than_do_loop ) THEN
                     DO i = 1, dims%x_free 
                       DIAG_X( i ) = k_diag + H_perturb ; END DO
                     DO i = dims%x_free + 1, ldiag_x
                       DIAG_X( i ) =  BARRIER_X( i ) + H_perturb ; END DO
                     DO i = dims%c_l_start, dims%c_u_end
                       DIAG_C( i ) =  BARRIER_C( i ) + H_perturb ; END DO
                   ELSE
                     DIAG_X( : dims%x_free ) = k_diag + H_perturb
                     DIAG_X( dims%x_free + 1 : ldiag_x ) = BARRIER_X + H_perturb
                     DIAG_C = BARRIER_C + H_perturb
                   END IF

!  If required, ensure the diagonals are significantly positive

                   IF ( alter_H ) THEN
                     IF ( control%array_syntax_worse_than_do_loop ) THEN
                       DO i = 1, ldiag_x
                         DIAG_X( i ) = MAX( DIAG_X( i ), perturb_min ) ; END DO
                       DO i = ldiag_c_l, ldiag_c_u 
                         DIAG_C( i ) = MAX( DIAG_C( i ), perturb_min ) ; END DO
                     ELSE
                       DIAG_X = MAX( DIAG_X, perturb_min )
                       DIAG_C = MAX( DIAG_C, perturb_min )
                     END IF
                   END IF

                 END SELECT

!  Compute the number of positive eigenvalues of D

                 dplus = COUNT( DIAG_X > zero )
                 dzero = COUNT( DIAG_X == zero )

!  Form the Schur complement matrix

                 IF ( m > 0 ) THEN

                    CALL LPB_form_Schur_complement(                            &
                      dims, n, m, A_ne, Abycol_val, Abycol_row, Abycol_ptr,    &
                      DIAG_X, SCALE_C, DIAG_C, lk, K%val, K%row,               &
                      K_colptr, nnzk, ierr, IW( : n ), IW( n + 1 :  ),         &
                      .TRUE., control%error, print_level )
!                   inform%nfacts = inform%nfacts + 1 
                  ELSE
                    get_factors = .FALSE.
                    nnzk = 0
                  END IF

!  For the KKT matrix
!  ==================

              ELSE

!  Set the remaining values of M in K

                 IF ( precon == 2 ) THEN

!  * M is just the barrier terms

                   IF ( control%array_syntax_worse_than_do_loop ) THEN
                     DO i = 1, dims%x_free
                       K%val( nnzks + i ) = k_diag + H_perturb ; END DO
                     DO i = dims%x_free + 1, dims%x_e
                       K%val( nnzks + i ) =  BARRIER_X( i ) + H_perturb ; END DO
                     DO i = 0, dims%nc - 1
                       K%val( nnzks + dims%c_s + i ) =                         &
                         BARRIER_C( dims%c_l_start + i ) + H_perturb ; END DO
                   ELSE
                     K%val( nnzks + 1 : nnzks + dims%x_free ) =                &
                       k_diag + H_perturb
                     K%val( nnzks + dims%x_free + 1 : nnzks + dims%x_e )       &
                       = BARRIER_X + H_perturb
                     K%val( nnzks + dims%c_s : nnzks + dims%c_e )              &
                       = BARRIER_C + H_perturb
                   END IF

!  If required, ensure the diagonals are significantly positive

                   IF ( control%array_syntax_worse_than_do_loop ) THEN
                     IF ( alter_H ) THEN
                       DO i = nnzks + dims%c_s, nnzks + dims%c_e
                         K%val( i ) = MAX( K%val( i ), perturb_min ) ; END DO
                     END IF
                   ELSE
                     IF ( alter_H ) K%val( nnzks + 1 : nnzks + dims%c_e ) =    &
                       MAX( K%val( nnzks + 1 : nnzks + dims%c_e ), perturb_min )
                   END IF
                 END IF
              END IF

! :::::::::::::::::::::::::::::
!  Factorize the preconditioner
! :::::::::::::::::::::::::::::

              IF ( printd ) WRITE( out, "( (2I6, ES12.4 ) )" ) ( K%row( l ),   &
                              K%col( l ), K%val( l ), l = 1, K%ne )

              IF ( K%n > 0 .AND. get_factors ) THEN

!               IF ( inform%iter == 25 ) THEN
!                 WRITE( 6, " ( ' ----------- dumping -------- ' )" )
!                 WRITE( 22, "( 2I6 )" ) K%n, K%ne
!                 WRITE( 22, "( ( 10I6 ) )" ) K%row( : K%ne )
!                 WRITE( 22, "( ( 10I6 ) )" ) K%col( : K%ne ) 
!                 WRITE( 22, "( ( 3ES24.16 ) )" ) K%val( : K%ne ) 
!                 WRITE( 22, "( ( 3ES24.16 ) )" ) SOL( : K%n )
!               END IF

                CALL SILS_factorize( K, FACTORS, CNTL, FINFO )

!  Record the storage required

                inform%nfacts = inform%nfacts + 1 
                inform%factorization_integer = FINFO%nirbdu 
                inform%factorization_real = FINFO%nrlbdu

!  Test that the factorization succeeded

                zeig = 0
                inform%factorization_status = FINFO%flag
                IF ( FINFO%flag < 0 ) THEN
                  IF ( printe ) WRITE( control%error, 2040 ) FINFO%flag,       &
                                                             'SILS_factorize'
                  IF ( FINFO%flag == - 5 .OR. FINFO%flag == - 6 ) THEN
!                   IF ( CNTL%u < half ) THEN
                    IF ( CNTL%u < half .AND. factor == 2 ) THEN
                      IF ( CNTL%u < point01 ) THEN
                        CNTL%u = point01
                      ELSE
                        IF ( CNTL%u < point1 ) THEN
                          CNTL%u = point1
                        ELSE
                          CNTL%u = half
                        END IF
                      END IF
                      refact = .TRUE. ; new_fact = .TRUE.
                      IF ( printi ) WRITE( out, "( ' increasing pivot',        &
                     &  ' tolerance to ', ES12.4, / )" ) CNTL%u
                      CYCLE inner
                    ELSE IF ( factor /= 2 ) THEN
                      CNTL%u = pivot_tol
                      factor = 2
                      refact = .TRUE. ; new_fact = .TRUE. ; new_prec = .TRUE.
                      IF ( printi ) WRITE( out,                                &
                        "( ' changing to augmented system ', / )" )
                      CYCLE inner
                    END IF
                  ELSE IF ( FINFO%flag == - 3 ) THEN
                    IF ( auto .AND. precon /= 2 ) THEN
                      refact = .TRUE. ; new_prec = .TRUE.
                      precon = 2 ; fact_hist = 5 ; factor = control%factor
!                     WRITE( out, "( ' change precon to ', I2 )" ) precon
                      CYCLE inner
                    END IF
                  END IF
                  inform%status = - 7
                  GO TO 700

!  Record warning conditions

                ELSE IF ( FINFO%flag > 0 ) THEN
                  IF ( printt ) WRITE( control%out, 2050 )                     &
                                       FINFO%flag, 'SILS_factorize'
                  IF ( FINFO%flag == 4 ) THEN 
                     zeig = K%n - FINFO%rank
                     IF ( printt ) WRITE( control%out, 2070 ) zeig
                  END IF 
                END IF 

!  Test that the problem is convex in the null-space of the constraints

                IF ( factor == 2 ) THEN
                  kplus = n + m - zeig - FINFO%neig
                ELSE
                  kplus = dplus + FINFO%neig
                END IF
                
                IF ( kplus < n ) THEN 
                  IF ( mo == ' ' ) inform%nmods = inform%nmods + 1
                  mo = 'm'
                  IF ( precon == 4 ) THEN
                    alter_H = .TRUE.
                    IF ( printt ) WRITE( out, 2080 ) n + m - kplus - zeig,     &
                                                     zeig
                  ELSE
                    H_perturb = H_perturb + hmax
                    IF ( printt ) WRITE( out, 2080 ) n + m - kplus - zeig,     &
                                                     zeig, H_perturb
                  END IF
                  GO TO 50
                END IF 
              ELSE
                IF ( dplus == n ) THEN
                  inform%factorization_integer = 0 
                  inform%factorization_real = 0
                ELSE
                  IF ( mo == ' ' ) inform%nmods = inform%nmods + 1
                  mo = 'm'
                  IF ( precon == 4 ) THEN
                    alter_H = .TRUE.
                    IF ( printt ) WRITE( out, 2080 ) n - dplus - dzero, dzero
                  ELSE
                    H_perturb = H_perturb + hmax
                    IF ( printt ) WRITE( out, 2080 ) n - dplus - dzero,        &
                                                     dzero, H_perturb
                  END IF
                  GO TO 50
                END IF
              END IF

              old_mu = mu

              CALL CPU_TIME( dum ) ; dum = dum - time
              inform%time%factorize = inform%time%factorize + dum
   
              IF ( printt ) WRITE( out, 2060 ) inform%factorization_integer,   &
                                               inform%factorization_real
              IF ( K%n > 0 .AND. printt ) &
                 WRITE( out, "( ' factorize time = ', F10.2, /, &
                         & ' real/integer space used for factors ', 2I10 )" )  &
                             dum, FINFO%nrlbdu, FINFO%nirbdu
            END IF

!  Ensure that the projection is only computed once

            IF ( precon == 1 ) one_fact = .TRUE.
            full_iteration = .TRUE.
          ELSE 
            re = ' ' 
            full_iteration = .FALSE.
          END IF 

!  Check for convergence of the inner iteration
!  --------------------------------------------

!  Check if x, z_l, z_u satisfy the convergence tests

!  a) A x = b
!  b) X_l < x < X_u, z_l > 0, z_u < 0
!  c) || (X-L) z_l - \mu e, (X-U) z_u - \mu e \|_2 <= theta_c
!  d) || GRAD - z_l - z_u ||_M <= theta_d
!  e) leftmost eigenvalue of
!         N(trans)( H + (X-L)(inv)Z_l + (X-U)(inv)Z_u)N >= - theta_e

!  where GRAD = grad phi + mu ( (X-L)(inv) - (U-X)(inv) ) e = grad f
!  and N is an orthononormal basis for the null-space of A

!  Compute GRAD - z_l - z_u

!  Problem variables

          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, dims%x_free ; VECTOR( i ) = GRAD( i ) ; END DO
          ELSE
            VECTOR( : dims%x_free ) = GRAD( : dims%x_free )
          END IF

          DO i = dims%x_free + 1, dims%x_u_start - 1
            VECTOR( i ) = GRAD( i ) - Z_l( i )
          END DO

          DO i = dims%x_u_start, dims%x_l_end
            VECTOR( i ) = GRAD( i ) - Z_l( i ) - Z_u( i )
          END DO

          DO i = dims%x_l_end + 1, n
            VECTOR( i ) = GRAD( i ) - Z_u( i )
          END DO

!  Slack variables

          DO i = dims%c_l_start, dims%c_u_start - 1
            VECTOR( dims%c_b + i ) = - Y_l( i )
          END DO

          DO i = dims%c_u_start, dims%c_l_end
            VECTOR( dims%c_b + i ) = - Y_l( i ) - Y_u( i )
          END DO

          DO i = dims%c_l_end + 1, dims%c_u_end
            VECTOR( dims%c_b + i ) = - Y_u( i )
          END DO

          CALL LPB_AX( n, VECTOR( : n ), m, A_ne, A_val,                        &
                        A_col, A_ptr, m, Y, '-T' )
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 0, dims%nc - 1
              VECTOR( dims%c_s + i ) = VECTOR( dims%c_s + i ) +                &
                SCALE_C( dims%c_l_start + i ) * Y( dims%c_l_start + i ) ; END DO
            DO i = dims%y_s, dims%y_e ; VECTOR( i ) = zero ; END DO
          ELSE
            VECTOR( dims%c_s : dims%c_e ) = VECTOR( dims%c_s : dims%c_e ) +    &
                                  SCALE_C * Y( dims%c_l_start : dims%c_u_end )
            VECTOR( dims%y_s : dims%y_e ) = zero
          END IF

!  Calculate || GRAD - A^T y - z_l - z_u ||_2

          norm_d_alt = SQRT( ABS( SUM( VECTOR( : dims%c_e ) ** 2 ) ) )

!  Calculate || GRAD - z_l - z_u ||_M

          IF ( printw ) WRITE( out,                                            &
               "( ' .............. get multipliers .............. ' )" )

          IF ( m > 0 ) THEN
            CALL LPB_iterative_refinement(                                     &
                          dims, n, m, A_ne, A_val, A_col, A_ptr,               &
                          VECTOR, dims%v_e, factor, DIAG_X, ldiag_x, SCALE_C,  &
                          DIAG_C, ldiag_c_l, ldiag_c_u,                        &
                          SOL, RES, BEST, Y_trial, BEST_y, RES_y, RES_x,       &
                          precon, nnzks, itref_max,                            &
                          norm_d, res_norm, big_res, K, FACTORS, CNTL,         &
                          print_level, control, inform )

!  Take appropriate action if the residual is too large

            IF ( big_res ) THEN
              IF ( printi ) CALL LPB_cond( K%n, FINFO%rank, FACTORS, out )
              IF ( printi ) WRITE( out, 2340 ) res_norm, res_large
!             IF ( CNTL%u < half ) THEN
              IF ( CNTL%u < half .AND. factor == 2 ) THEN
                IF ( CNTL%u < point01 ) THEN
                  CNTL%u = point01
                ELSE
                  IF ( CNTL%u < point1 ) THEN
                    CNTL%u = point1
                  ELSE
                    CNTL%u = half
                  END IF
                END IF
                refact = .TRUE. ; new_fact = .TRUE.
                IF ( printi ) WRITE( out,                                      &
                  "( ' increasing pivot tolerance to ', ES12.4, / )" ) CNTL%u
                CYCLE inner
              ELSE IF ( factor /= 2 ) THEN
                CNTL%u = pivot_tol
                factor = 2
                refact = .TRUE. ; new_fact = .TRUE. ; new_prec = .TRUE.

                IF ( printi ) WRITE( out,                                      &
                  "( ' changing to augmented system method ', / )" )
                CYCLE inner
              ELSE
                inform%status = - 8
                GO TO 700
              END IF
            END IF

            IF ( inform%status /= 0 ) GO TO 700   

            IF ( auto .AND. res_norm > control%stop_d ) THEN
              IF ( ( fact_hist == 1 .OR. fact_hist == 3 ) .AND.                &
                   ( factor == 0 .OR. factor == 1 ) ) THEN
                 factor = 2
                 refact = .TRUE. ; new_prec = .TRUE.
                 IF ( printi ) WRITE( out, 2340 ) res_norm, control%stop_d
                 CYCLE
              END IF
            END IF
            norm_d = MIN( norm_d, norm_d_alt )
          ELSE
            norm_d = norm_d_alt
          END IF

!  Print a summary of the iteration

          IF ( printi ) THEN
            bdry = ' '
            IF ( .NOT. start_major ) THEN 
              IF ( ABS( gltr_info%mnormx - old_radius ) < teneps) bdry = 'b'
              IF ( printt .OR. gltr_control%print_level > 0 .OR.               &
                ( printi .AND. inform%iter == start_print ) ) WRITE( out, 2000 )
              IF ( got_ratio ) THEN
                 WRITE( out, 2020 ) inform%iter, re, norm_d, norm_c,           &
                                    inform%obj, mo, ratio, old_radius, bdry,   &
                                    nbacts, cg_iter, inform%time%total
              ELSE
                 WRITE( out, 2030 ) inform%iter, re, norm_d, norm_c,           &
                                    inform%obj, mo, old_radius, bdry,          &
                                    nbacts, cg_iter, inform%time%total
              END IF
            ELSE
              IF ( printi ) WRITE( out, 2130 ) mu, theta_c, theta_d
              WRITE( out, 2000 ) 
              WRITE( out, 2010 ) inform%iter, re, norm_d, norm_c, inform%obj,  &
                                 mo, old_radius, inform%time%total
            END IF 
            IF ( printd ) THEN 
              WRITE( out, 2100 ) ' X ', X
              WRITE( out, 2100 ) ' Z_l ', Z_l
              WRITE( out, 2100 ) ' Z_u ', Z_u
            END IF 
          END IF 
          start_major = .FALSE.

!  Record the Lagrange multipliers

          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, m ; Y( i ) = Y( i ) + VECTOR( dims%c_e + i ) ; END DO
          ELSE
            Y = Y + VECTOR( dims%y_s : dims%y_e )
          END IF
!         WRITE( 6, "( ' norm Y = ', ES22.14 )") MAXVAL( ABS( Y ) )

!  Test for convergence

          IF ( norm_c <= theta_c .AND. norm_d <= theta_d .AND. .NOT.           &
               gltr_info%negative_curvature ) THEN
            inform%status = 0
            gltr_info%status = 1
            start_major = .TRUE.
            got_ratio = .FALSE.
            full_iteration = .FALSE.
            EXIT
          END IF

!  Test to see if more than maxit iterations have been performed

          inform%iter = inform%iter + 1 
          IF ( inform%iter > control%maxit ) THEN 
            inform%status = - 10 ; GO TO 600 
          END IF 

!  Check that the CPU time limit has not been reached

          IF ( control%cpu_time_limit >= zero .AND.                            &
               time > control%cpu_time_limit ) THEN
            inform%status = - 12 ; GO TO 600
          END IF 

          IF ( inform%iter == start_print ) THEN
            printe = set_printe ; printi = set_printi ; printt = set_printt
            printm = set_printm ; printw = set_printw ; printd = set_printd
            print_level = control%print_level
            gltr_control%print_level = print_level - 1
          END IF
 
          IF ( inform%iter == stop_print + 1 ) THEN
            printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
            printm = .FALSE. ; printw = .FALSE. ; printd = .FALSE.
            print_level = 0 ; gltr_control%print_level = 0
          END IF

!       WRITE( 6, "( ' start, stop print, iter ', 3I8 )" )                     &
!         start_print, stop_print, inform%iter

!  Test to see if the objective appears to be unbounded from below

          IF ( inform%obj < control%obj_unbounded ) THEN 
            inform%status = - 11 ; GO TO 600 
          END IF 

!  Compute the trial step
!  ----------------------

          IF ( printw ) WRITE( out,                                            &
               "( ' ............... compute step  ............... ' )" )

!  Compute a trial step, s, to ``sufficiently'' reduce the model within 
!  the region defined by the intersection of the affine constraints A s = 0
!  and the trust region || s ||_M <= radius

!  Compute the derivatives of the Lagrangian function

          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i =  dims%x_s, dims%x_e ; GRAD_L( i ) = GRAD_X_phi( i ) ; END DO
          ELSE
            GRAD_L( dims%x_s : dims%x_e ) = GRAD_X_phi
          END IF
          CALL LPB_AX( n, GRAD_L( dims%x_s : dims%x_e ), m, A_ne,                                 &
                        A_val, A_col, A_ptr, m, Y, '-T' )
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 0, dims%nc - 1
              GRAD_L( dims%c_s + i ) = GRAD_C_phi( dims%c_l_start + i ) +      &
                SCALE_C( dims%c_l_start + i ) * Y( dims%c_l_start + i ) ; END DO
          ELSE
            GRAD_L( dims%c_s : dims%c_e ) = GRAD_C_phi +                       &
                                  SCALE_C * Y( dims%c_l_start : dims%c_u_end )
          END IF

          IF ( printm ) WRITE( out, "( ' norm GRAD_L ', ES12.4 ) " )           &
                               MAXVAL( ABS( GRAD_L ) )
!  Set initial data

!         R( : dims%c_e ) = GRAD_L
          DO i =  1, dims%c_e ; R( i ) = GRAD_L( i ) ; END DO
          first_iteration = .TRUE.

          IF ( printm ) WRITE( out,                                            &
           "(/, '   |------------------------------------------------------|', &
         &   /, '   |        start to solve trust-region subproblem        |', &
         &   / )" )

          CALL CPU_TIME( time )
  100     CONTINUE
          CALL GLTR_solve( dims%c_e, radius, model, S, R, VECTOR( : dims%c_e ),        &
                           gltr_data, gltr_control, gltr_info )

!  Check for error returns

          SELECT CASE( gltr_info%status )

!  Successful return

          CASE ( 0 )

!  Warnings

          CASE ( - 2, - 1 )
            IF ( printt ) WRITE( out, "( /,                                    &
           &  ' Warning return from GLTR, status = ', I6 )" ) gltr_info%status
          
!  Allocation errors

           CASE ( - 6 )
             inform%status = - 2
             inform%alloc_status = gltr_info%alloc_status
             inform%bad_alloc = gltr_info%bad_alloc
             GO TO 920

!  Deallocation errors

           CASE ( - 7 )
             inform%status = - 3
             inform%alloc_status = gltr_info%alloc_status
             inform%bad_alloc = gltr_info%bad_alloc
             GO TO 920

!  Error return

          CASE DEFAULT
            IF ( printt ) WRITE( out, "( /,                                    &
           &  ' Error return from GLTR, status = ', I6 )" ) gltr_info%status

!  Find the preconditioned gradient

          CASE ( 2, 6 )
            IF ( printw ) WRITE( out,                                          &
               "( ' ............... precondition  ............... ' )" )

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              VECTOR( dims%y_s : dims%y_e ) = zero
            ELSE
              DO i = dims%y_s, dims%y_e ; VECTOR( i ) = zero ; END DO
            END IF

            CALL LPB_iterative_refinement(                                     &
                          dims, n, m, A_ne, A_val, A_col, A_ptr,               &
                          VECTOR, dims%v_e, factor,                            &
                          DIAG_X, ldiag_x, SCALE_C, DIAG_C, ldiag_c_l,         &
                          ldiag_c_u, SOL, RES, BEST, Y_trial, BEST_y, RES_y,   &
                          RES_x, precon, nnzks,                                &
                          itref_max, sn, res_norm, big_res,                    &
                          K, FACTORS, CNTL, print_level, control, inform )   

!  Take appropriate action if the residual is too large

            IF ( big_res ) THEN
              IF ( printi ) CALL LPB_cond( K%n, FINFO%rank, FACTORS, out )
              IF ( printi ) WRITE( out, 2340 ) res_norm, res_large
!             IF ( CNTL%u < half ) THEN
              IF ( CNTL%u < half .AND. factor == 2 ) THEN
                IF ( CNTL%u < point01 ) THEN
                  CNTL%u = point01
                ELSE
                  IF ( CNTL%u < point1 ) THEN
                    CNTL%u = point1
                  ELSE
                    CNTL%u = half
                  END IF
                END IF
                refact = .TRUE. ; new_fact = .TRUE.
                IF ( printi ) WRITE( out,                                      &
                  "( ' increasing pivot tolerance to ', ES12.4, / )" ) CNTL%u
                CYCLE inner
              ELSE IF ( factor /= 2 ) THEN
                CNTL%u = pivot_tol
                factor = 2
                refact = .TRUE. ; new_fact = .TRUE. ; new_prec = .TRUE.
                IF ( printi ) WRITE( out,                                      &
                  "( ' changing to augmented system ', / )" )
                CYCLE inner
              ELSE
                inform%status = - 8
                GO TO 700
              END IF
            END IF

            IF ( inform%status /= 0 ) GO TO 700   
            
            IF ( auto .AND. res_norm > control%stop_d ) THEN
              IF ( ( fact_hist == 1 .OR. fact_hist == 3 ) .AND.                &
                   ( factor == 0 .OR. factor == 1 ) ) THEN
                factor = 2
                refact = .TRUE. ; new_prec = .TRUE.
                IF ( printi ) WRITE( out, 2340 ) res_norm, control%stop_d
                CYCLE
              END IF
              IF ( first_iteration ) norm_d = sn
            END IF
            GO TO 100

!  Form the product of VECTOR with H

          CASE ( 3, 7 )

            IF ( printw ) WRITE( out,                                          &
                 "( ' ............ matrix-vector product .......... ' )" )

!  Compute the largest error in the residuals

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, dims%x_free ; HX( i ) = zero ; END DO
              DO i = dims%x_free + 1, n 
                HX( i ) = BARRIER_X( i ) * VECTOR( i ) ; END DO
            ELSE
              HX( : dims%x_free ) = zero
              HX( dims%x_free + 1 : n ) =                                      &
                BARRIER_X * VECTOR( dims%x_free + 1 : n )
            END IF
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 0, dims%nc - 1
                HX( dims%c_s + i ) =                                           &
                  BARRIER_C( dims%c_l_start + i ) * VECTOR( dims%c_s + i )
              END DO
            ELSE
              HX( dims%c_s : dims%c_e ) =                                      &
                BARRIER_C * VECTOR( dims%c_s : dims%c_e )
            END IF

!  Print the residuals if required 

            IF ( printm .AND. m > 0 ) THEN
              IF ( control%array_syntax_worse_than_do_loop ) THEN
                DO i = dims%y_s, dims%y_i - 1 ; HX( i ) = zero ; END DO
                DO i = 0, dims%nc - 1
                  HX( dims%y_i + i ) =                                         &
                   - SCALE_C( dims%c_l_start + i ) * VECTOR( dims%c_s + i )
                END DO
              ELSE
                HX( dims%y_s : dims%y_i - 1 ) = zero
                HX( dims%y_i : dims%y_e ) =                                    &
                   - SCALE_C * VECTOR( dims%c_s : dims%c_e )
              END IF
              CALL LPB_AX( m, HX( dims%y_s : dims%y_e ), m, A_ne, A_val, A_col,                 &
                            A_ptr, n, VECTOR( : n ), '+ ' )
              WRITE( out, "(' constraint residual ', ES12.4 )")                &
                MAXVAL( ABS( HX( dims%y_s : dims%y_e ) ) )
            END IF
          
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, dims%c_e ; VECTOR( i ) = HX( i ) ; END DO
            ELSE
              VECTOR( : dims%c_e ) = HX( : dims%c_e )
            END IF
          
            GO TO 100

!  Reform the initial residual

          CASE ( 5 )
          
            IF ( printw ) WRITE( out,                                          &
                 "( ' ................. restarting ................ ' )" )

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i =  1, dims%c_e ; R( i ) = GRAD_L( i ) ; END DO
            ELSE
              R( : dims%c_e ) = GRAD_L
            END IF
            GO TO 100

          END SELECT

          CALL CPU_TIME( dum ) ; dum = dum - time

          IF ( printm ) WRITE( out,                                            &
           "(/, '   |           trust-region subproblem solved             |', &
         &   /, '   |------------------------------------------------------|', &
         &     / )" )

          IF ( printw ) WRITE( out,                                            &
               "( ' ............... step computed ............... ' )" )

          IF ( printt ) WRITE( out, "( ' solve time = ', F10.2 ) " ) dum
          inform%time%solve = inform%time%solve + dum

          cg_iter = gltr_info%iter
          inform%cg_iter = inform%cg_iter + cg_iter

!  If the overall search direction is unlikely to make a significant
!  impact on the residual, exit

          IF ( gltr_info%mnormx <= teneps ) THEN
            inform%status = - 9
            gltr_info%status = 1
            got_ratio = .FALSE.
            full_iteration = .FALSE.

!  Update the Lagrange multipliers

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, m
                Y( i ) = Y( i ) + VECTOR( dims%c_e + i ) ; END DO
            ELSE
              Y = Y + VECTOR( dims%y_s : dims%y_e )
            END IF

!  Update the dual variables so that z_l > 0 and z_u > 0

!  Problem variables

            DO i = dims%x_free + 1, dims%x_l_start - 1
              Z_l( i ) = mu / X( i )
            END DO
            DO i = dims%x_l_start, dims%x_l_end
              Z_l( i ) = mu / DIST_X_l( i )
            END DO
            DO i = dims%x_u_start, dims%x_u_end
              Z_u( i ) = - mu / DIST_X_u( i )
            END DO       
            DO i = dims%x_u_end + 1, n
              Z_u( i ) = mu / X( i )
            END DO       

!  Slack variables

            DO i = dims%c_l_start, dims%c_l_end
              Y_l( i ) = mu / DIST_C_l( i )
            END DO
            DO i = dims%c_u_start, dims%c_u_end
              Y_u( i ) = - mu / DIST_C_u( i )
            END DO       

            EXIT
          END IF

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Find the largest feasible step for the primal variables
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

          IF ( printw ) WRITE( out,                                            &
               "( ' .............. get steplength  .............. ' )" )

          step_max = infinity

!  Problem variables:

          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( S( i ) < zero ) step_max = MIN( step_max, - X( i ) / S( i ) ) 
          END DO

          DO i = dims%x_l_start, dims%x_l_end
            IF ( S( i ) < zero )                                               &
              step_max = MIN( step_max, - DIST_X_l( i ) / S( i ) ) 
          END DO

          DO i = dims%x_u_start, dims%x_u_end
            IF ( S( i ) > zero )                                               &
              step_max = MIN( step_max, DIST_X_u( i ) / S( i ) ) 
          END DO 

          DO i = dims%x_u_end + 1, n
            IF ( S( i ) > zero ) step_max = MIN( step_max, - X( i ) / S( i ) ) 
          END DO 

!  Slack variables:

          DO i = dims%c_l_start, dims%c_l_end
            IF ( S( dims%c_b + i ) < zero )                                    &
              step_max = MIN( step_max, - DIST_C_l( i ) / S( dims%c_b + i ) ) 
          END DO

          DO i = dims%c_u_start, dims%c_u_end
            IF ( S( dims%c_b + i ) > zero )                                    &
              step_max = MIN( step_max, DIST_C_u( i ) / S( dims%c_b + i ) ) 
          END DO 

!  Test whether the new point is acceptable
!  ----------------------------------------   

!  If x + s - X_l >= zeta(x - l) and X_u - x - s >= zeta(u - x),
!  compute phi at x+s and define the ratio
!       (phi(x+s) - phi(x))/m(x+s) - m(x))

          got_ratio = .TRUE.

          DO i = dims%x_free + 1, dims%x_l_start - 1

!  Check that x + s >= zeta(x)

            IF ( X( i ) + S( i ) < zeta * X( i ) ) THEN
               got_ratio = .FALSE.
               EXIT
            END IF
          END DO

          IF ( got_ratio ) THEN
            DO i = dims%x_l_start, dims%x_l_end

!  Calculate x + s - x_l

              DIST_X_l_trial( i ) = DIST_X_l( i ) + S( i ) 

!  Check that x + s - x_l >= zeta(x - x_l)

              IF ( DIST_X_l_trial( i ) < zeta * DIST_X_l( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO
          END IF

          IF ( got_ratio ) THEN
            DO i = dims%x_u_start, dims%x_u_end

!  Calculate x_u - x - s

              DIST_X_u_trial( i ) = DIST_X_u( i ) - S( i ) 

!  Check that x_u - x - s >= zeta(x_u - x)

              IF ( DIST_X_u_trial( i ) < zeta * DIST_X_u( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO 
          END IF

          IF ( got_ratio ) THEN
            DO i = dims%x_u_end + 1, n

!  Check that x + s <= zeta(x)

              IF ( X( i ) + S( i ) > zeta * X( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO 
          END IF

!  Do the same for the slack variables

!  Calculate x + s - l

          IF ( got_ratio ) THEN
            DO i = dims%c_l_start, dims%c_l_end
              DIST_C_l_trial( i ) = DIST_C_l( i ) + S( dims%c_b + i ) 


!  Check that x + s - X_l >= zeta(x - l)

              IF ( DIST_C_l_trial( i ) < zeta * DIST_C_l( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO
          END IF

          IF ( got_ratio ) THEN
            DO i = dims%c_u_start, dims%c_u_end

!  Calculate X_u - x - s

              DIST_C_u_trial( i ) = DIST_C_u( i ) - S( dims%c_b + i ) 

!  Check that X_u - x - s >= zeta(u - x)

              IF ( DIST_C_u_trial( i ) < zeta * DIST_C_u( i ) ) THEN
                 got_ratio = .FALSE.
                 EXIT
              END IF
            END DO 
          END IF

!  The new point is feasible. Calculate the new value of the objective function

          IF ( got_ratio ) THEN
      
!  Calculate x + s

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = dims%x_s, dims%x_e ; X_trial( i ) = X( i ) + S( i ); END DO
            ELSE
              X_trial = X + S( dims%x_s : dims%x_e )
            END IF
            nbacts = 0
            gltr_control%boundary = .FALSE.
            gltr_info%status = 1

!  Now evaluate the objective function ...

            obj_trial = DOT_PRODUCT( X_trial, G )

!  ... and the barrier function

            phi_trial = LPB_barrier_value( dims, n, obj_trial, X_trial,        &
                                            DIST_X_l_trial, DIST_X_u_trial,    &
                                            DIST_C_l_trial, DIST_C_u_trial,    &
                                            mu )

!  Compute the ratio of actual to predicted reduction over the current iteration

            ared   = ( phi - phi_trial ) + MAX( one, ABS( phi ) ) * teneps
            prered = - model + MAX( one, ABS( phi ) ) * teneps
            IF ( ABS( ared ) < teneps .AND. ABS( phi ) > teneps )              &
               ared = prered
            ratio = ared / prered

            IF ( printt )                                                      &
                WRITE( out, " ( ' ared, pred ', 2ES12.4 ) " ) ared, prered

          ELSE
            nbacts = 1 ; gltr_control%boundary = .TRUE.
          END IF

!  If ratio >= eta_1, the iteration was successful

          IF ( got_ratio .AND. ratio >= eta_1 ) THEN

            IF ( printw ) WRITE( out,                                          &
                 "( ' ............... successful step ............... ' )" )

!  Compute new dual variables so that z_l > 0 and z_u > 0
!  ------------------------------------------------------

!  Problem variables:

!  For the lower bounds ...
 
            DO i = dims%x_free + 1, dims%x_l_start - 1
              Z_l( i ) = MAX( z_min, ( mu - Z_l( i ) * S( i ) ) / X( i ),      &
                                nu_1 * MIN( one, Z_l( i ), mu / X_trial( i ) ) )
            END DO

            DO i = dims%x_l_start, dims%x_l_end
              Z_l( i ) = MAX( z_min,                                           &
                              ( mu - Z_l( i ) * S( i ) ) / DIST_X_l( i ),      &
                           nu_1 * MIN( one,                                    &
                                       Z_l( i ), mu / DIST_X_l_trial( i ) ) )
            END DO
     
!  .... and now the upper bounds

            DO i = dims%x_u_start, dims%x_u_end
              Z_u( i ) = MIN( - z_min,                                         &
                              ( - mu + Z_u( i ) * S( i ) ) / DIST_X_u( i ),    &
                           nu_1 * MAX( - one,                                  &
                                       Z_u( i ), - mu / DIST_X_u_trial( i ) ) )
            END DO       

            DO i = dims%x_u_end + 1, n
              Z_u( i ) = MIN( - z_min, ( mu - Z_u( i ) * S( i ) ) / X( i ),    &
                           nu_1 * MAX( - one, Z_u( i ), mu / X_trial( i ) ) )
            END DO       

!  Now replace x by x + s

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, n 
                X( i ) = X_trial( i ) ; END DO
              DO i = dims%x_l_start, dims%x_l_end
                DIST_X_l( i ) = DIST_X_l_trial( i ) ; END DO
              DO i = dims%x_u_start, dims%x_u_end
                DIST_X_u( i ) = DIST_X_u_trial( i ) ; END DO
            ELSE
              X = X_trial
              DIST_X_l = DIST_X_l_trial
              DIST_X_u = DIST_X_u_trial
            END IF

!  Slack variables:

!  For the lower bounds ...
 
            DO i = dims%c_l_start, dims%c_l_end
              Y_l( i ) = MAX( ( mu - Y_l( i ) * S( dims%c_b + i ) ) /          &
                                DIST_C_l( i ), nu_1 * MIN( one,                &
                                       Y_l( i ), mu / DIST_C_l_trial( i ) ) )
            END DO
     
!  .... and now the upper bounds

            DO i = dims%c_u_start, dims%c_u_end
              Y_u( i ) = MIN( ( - mu + Y_u( i ) * S( dims%c_b + i ) ) /        &
                                DIST_C_u( i ), nu_1 * MAX( - one,              &
                                       Y_u( i ), - mu / DIST_C_u_trial( i ) ) )
            END DO       

!  Now replace c by c + s

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = dims%c_l_start, dims%c_u_end
                C( i ) = C( i ) + S( dims%c_b + i ) ; END DO
              DO i = dims%c_l_start, dims%c_l_end
                DIST_C_l( i ) = DIST_C_l_trial( i ) ; END DO
              DO i = dims%c_u_start, dims%c_u_end
                DIST_C_u( i ) = DIST_C_u_trial( i ) ; END DO
            ELSE
              C = C + S( dims%c_s : dims%c_e )
              DIST_C_l = DIST_C_l_trial
              DIST_C_u = DIST_C_u_trial
            END IF

!  Update the objective and barrier function values

            phi  = phi_trial
            obj  = obj_trial
            inform%obj = obj + f

!  Compute the derivatives of the objective function

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, n ; GRAD( i ) = G( i ) ; END DO
            ELSE
              GRAD = G
            END IF
            refact = .TRUE.
          ELSE

            IF ( printw ) WRITE( out,                                          &
                 "( ' .............. unsuccessful step .............. ' )" )

!  As we have not achieved a sufficient reduction, use the Nocedal-Yuan 
!  technique to achieve one. To do this, perform a line-search to find
!  a point x + alpha s which sufficiently reduces the barrier function

!  Find the largest feasible step for x

            alpha = one
   
!  Problem variables:

            DO i = dims%x_free + 1, dims%x_l_start - 1
              IF( S( i ) < zero ) alpha = MIN( alpha, - X( i ) / S( i ) )
            END DO
            DO i = dims%x_l_start, dims%x_l_end
              IF( S( i ) < zero ) alpha = MIN( alpha, - DIST_X_l( i ) / S( i ) )
            END DO

            DO i = dims%x_u_start, dims%x_u_end
              IF( S( i ) > zero ) alpha = MIN( alpha, DIST_X_u( i ) / S( i ) ) 
            END DO 
            DO i = dims%x_u_end + 1, n
              IF( S( i ) > zero ) alpha = MIN( alpha, - X( i ) / S( i ) ) 
            END DO 

!  Slack variables:

            DO i = dims%c_l_start, dims%c_l_end
              IF ( S( dims%c_b + i ) < zero )                                  &
                alpha = MIN( alpha, - DIST_C_l( i ) / S( dims%c_b + i ) )
            END DO

            DO i = dims%c_u_start, dims%c_u_end
              IF ( S( dims%c_b + i ) > zero )                                  &
                alpha = MIN( alpha, DIST_C_u( i ) / S( dims%c_b + i ) ) 
            END DO 

!  A step of no larger than zeta of the distance to the nearest 
!  bound will be attempted

            alpha = ( one - zeta ) * alpha
!           alpha = half * alpha

! ::::::::::::::::::::::::::::::::::::::::::::
!  Record the slope along the search direction
! ::::::::::::::::::::::::::::::::::::::::::::

!  Compute the slope and curvature of the objective function

            obj_slope = DOT_PRODUCT( GRAD( : n ), S( : n ) ) 
            obj_curv  = zero

!  Compute the slope of the barrier function

            phi_slope = DOT_PRODUCT( GRAD_L, S ) 

! ::::::::::::::::::::::::::::::::::::::::::::::::::::::
!  Use a backtracking line-search, starting from alpha_v
! ::::::::::::::::::::::::::::::::::::::::::::::::::::::

            IF ( printt ) WRITE( out, "( /, ' value = ', ES12.4, ' slope = ',  &
           &     ES12.4 )" ) phi, phi_slope
            IF ( printw ) WRITE( out,                                          &
                 "( ' ................ linesearch ................ ' )" )
            IF ( printt ) WRITE( out, 2140 ) 
            DO

!  The barrier value should  be smaller than a linear model

              phi_model = phi + alpha * eta * phi_slope

!  Calculate the distances of x + alpha s from the bounds

              IF ( control%array_syntax_worse_than_do_loop ) THEN
                DO i = 1, n
                  X_trial( i ) = X( i ) + alpha * S( i ) ; END DO
              ELSE
                X_trial = X + alpha * S( : n )
              END IF

!  Problem variables:

              DO i = dims%x_l_start, dims%x_l_end
                DIST_X_l_trial( i ) = DIST_X_l( i ) + alpha * S( i ) 
              END DO 
              DO i = dims%x_u_start, dims%x_u_end
                DIST_X_u_trial( i ) = DIST_X_u( i ) - alpha * S( i ) 
              END DO 
        
!  Slack variables:

              DO i = dims%c_l_start, dims%c_l_end
                DIST_C_l_trial( i ) = DIST_C_l( i ) + alpha * S( dims%c_b + i )
              END DO 
              DO i = dims%c_u_start, dims%c_u_end
                DIST_C_u_trial( i ) = DIST_C_u( i ) - alpha * S( dims%c_b + i )
              END DO 
        
              obj_trial = obj + alpha * ( obj_slope + alpha * obj_curv )
              phi_trial = LPB_barrier_value( dims, n, obj_trial, X_trial,      &
                                              DIST_X_l_trial, DIST_X_u_trial,  &
                                              DIST_C_l_trial, DIST_C_u_trial,  &
                                              mu )

              IF ( printt ) WRITE( out, 2150 ) alpha, phi_trial, phi_model 

!  Check to see if the Armijo criterion is satisfied. If not, halve the 
!  steplength

              IF ( phi_trial <= phi_model ) EXIT
              alpha = half * alpha ;  nbacts = nbacts + 1 
              IF ( alpha < epsmch ) THEN
                 inform%status = - 9
                 GO TO 500
              END IF
            END DO
            inform%nbacts = inform%nbacts + nbacts
            refact = .TRUE.

!  Update the objective and barrier function values

            phi  = phi_trial ; obj  = obj_trial ; inform%obj = obj + f

!  Update the primal variables and derivatives of the objective function

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, n
                X( i ) = X( i ) + alpha * S( i )
                END DO
              DO i = dims%c_l_start, dims%c_u_end
                C( i ) = C( i ) + alpha * S( dims%c_b + i ) ; END DO
            ELSE
              X = X + alpha * S( dims%x_s : dims%x_e )
              C = C + alpha * S( dims%c_s : dims%c_e )
            END IF

!  Update the distances to the bounds

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = dims%x_l_start, dims%x_l_end
                DIST_X_l( i ) = DIST_X_l_trial( i ) ; END DO 
              DO i = dims%x_u_start, dims%x_u_end
                DIST_X_u( i ) = DIST_X_u_trial( i ) ; END DO 
              DO i = dims%c_l_start, dims%c_l_end
                DIST_C_l( i ) = DIST_C_l_trial( i ) ; END DO 
              DO i = dims%c_u_start, dims%c_u_end
                DIST_C_u( i ) = DIST_C_u_trial( i ) ; END DO 
            ELSE
              DIST_X_l = DIST_X_l_trial ; DIST_X_u = DIST_X_u_trial
              DIST_C_l = DIST_C_l_trial ; DIST_C_u = DIST_C_u_trial
            END IF

          END IF
!         write(6,"( ' X, C ', 2ES12.4 )" ) X(2), C(79)

!  Update the trust-region radius
!  ------------------------------   

          old_radius = radius

!  If ratio >= eta_2, possibly increase radius

          IF ( got_ratio .AND. ratio >= eta_2 ) THEN
            radius = MIN( radius_max,                                          &
                          radius * MIN( two, half * ( step_max + one ) ),      &
                          MAX( radius, two * gltr_info%mnormx ) )

!  If eta_2 > ratio >= eta_1, replace radius by something 
!  in [gamma_2 radius, radius] 

          ELSE IF ( got_ratio .AND. ratio >= eta_1 ) THEN
            
!  If eta_1 > ratio, replace radius by something
!  in [gamma_1 radius, gamma_2 radius] 

          ELSE
            delta = one
  410       CONTINUE
            gltr_info%status = 1
            delta = half * delta
            IF ( alpha <= delta ) GO TO 410
            radius = delta * radius
          END IF

        END DO inner  ! end of inner iteration loop

!  ======================
!  End of Inner iteration
!  ======================

!  Compare the primal and primal-dual dual variables

        IF ( printd ) THEN
          DO i = dims%x_free + 1, dims%x_l_start - 1
            WRITE( out," ( ' z_l, dz_l = ', 2ES12.4 )" )                       &
              Z_l( i ),    mu / X( i ) - Z_l( i )
          END DO
          DO i = dims%x_l_start, dims%x_l_end
            WRITE( out," ( ' z_l, dz_l = ', 2ES12.4 )" )                       &
              Z_l( i ),    mu / DIST_X_l( i ) - Z_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_u_end
            WRITE( out," ( ' z_u, dz_u = ', 2ES12.4 )" )                       &
              Z_u( i ),  - mu / DIST_X_u( i ) - Z_u( i ) 
          END DO
          DO i = dims%x_u_end + 1, n
            WRITE( out," ( ' z_u, dz_u = ', 2ES12.4 )" )                       &
              Z_u( i ),  mu / X( i ) - Z_u( i ) 
          END DO
          DO i = dims%c_l_start, dims%c_l_end
            WRITE( out," ( ' y_l, dy_l, y = ', 3ES12.4 )" )                    &
              Y_l( i ),    mu / DIST_C_l( i ) - Y_l( i ), Y( i )
          END DO
          DO i = dims%c_u_start, dims%c_u_end
            WRITE( out," ( ' y_u, dy_u, y = ', 3ES12.4 )" )                    &
              Y_u( i ),  - mu / DIST_C_u( i ) - Y_u( i ), Y( i )
          END DO
        END IF

        IF ( get_stat ) THEN

!  Estimate the variable and constraint exit status

          CALL LSQP_indicators( dims, n, m, C_l, C_u, C_last, C,               &
                                DIST_C_l, DIST_C_u, X_l, X_u, X_last, X,       &
                                DIST_X_l, DIST_X_u, Y_l, Y_u, Z_l, Z_u,        &
                                Y_last, Z_last,                                &
                                control%LSQP_control, C_stat = C_stat,         &
                                B_stat = B_stat )

!  Count the number of active constraints/bounds

          IF ( printt )                                                        &
            WRITE( out, "( ' indicators: n_active/n, m_active/m ', 4I7 )" )    &
               COUNT( B_stat /= 0 ), n, COUNT( C_stat /= 0 ), m
        END IF

!  Check for termination of the outer iteration

        IF ( norm_c <= control%stop_c .AND.                                    &
             norm_d <= control%stop_d ) THEN

          norm_c = zero

          DO i = dims%x_free + 1, dims%x_l_start - 1
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_l( i ) ) )
          END DO

          DO i = dims%x_l_start, dims%x_u_start - 1
            IF ( ABS( DIST_X_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) ) )
          END DO

          DO i = dims%x_u_start, dims%x_l_end
            IF ( MIN( ABS( DIST_X_l( i ) ), ABS( DIST_X_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_X_l( i ) * Z_l( i ) ),         &
                                    ABS( - DIST_X_u( i ) * Z_u( i ) ) )
          END DO

          DO i = dims%x_l_end + 1, dims%x_u_end
            IF ( ABS( DIST_X_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_X_u( i ) * Z_u( i ) ) )
          END DO

          DO i = dims%x_u_end + 1, n
            IF ( ABS( X( i ) ) < remote )                                      &
              norm_c = MAX( norm_c, ABS( X( i ) * Z_u( i ) ) )
          END DO

!  Slack variables:

          DO i = dims%c_l_start, dims%c_u_start - 1
            IF ( ABS( DIST_C_l( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( DIST_C_l( i ) * Y_l( i ) ) )
          END DO

          DO i = dims%c_u_start, dims%c_l_end
            IF ( MIN( ABS( DIST_C_l( i ) ), ABS( DIST_C_u( i ) ) ) < remote )  &
              norm_c = MAX( norm_c, ABS(   DIST_C_l( i ) * Y_l( i ) ),         &
                                    ABS( - DIST_C_u( i ) * Y_u( i ) ) )
          END DO

          DO i = dims%c_l_end + 1, dims%c_u_end
            IF ( ABS( DIST_C_u( i ) ) < remote )                               &
              norm_c = MAX( norm_c, ABS( - DIST_C_u( i ) * Y_u( i ) ) )
          END DO

          IF ( printt ) WRITE( out, 2160 ) p_min, p_max, d_min, d_max

!         write(6,*) ' --------- normc, stop_c ', norm_c, control%stop_c

!         write(6,*)    MAXVAL( X( dims%x_free + 1 : dims%x_l_start - 1 ) *    &
!                               Z_l( dims%x_free + 1 : dims%x_l_start - 1 ) ), &
!                       MAXVAL( DIST_X_l( dims%x_l_start : dims%x_l_end ) *    &
!                               Z_l( dims%x_l_start : dims%x_l_end ) ), -      &
!                       MAXVAL( DIST_X_u( dims%x_u_start : dims%x_u_end ) *    &
!                               Z_u( dims%x_u_start : dims%x_u_end ) ),        &
!                       MAXVAL( X( dims%x_u_end + 1 : n ) *               &
!                               Z_u( dims%x_u_end + 1 : n ) ),            &
!                       MAXVAL( DIST_C_l( dims%c_l_start : dims%c_l_end ) *    &
!                               Y_l( dims%c_l_start : dims%c_l_end ) ),  -     &
!                       MAXVAL( DIST_C_u( dims%c_u_start : dims%c_u_end ) *    &
!                               Y_u( dims%c_u_start : dims%c_u_end ) )

          IF ( norm_c <= control%stop_c ) THEN
            inform%status = 0
            EXIT
          END IF
        END IF

  500   CONTINUE

        IF ( printi .AND. m > 0 ) THEN 
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i =  1, dims%c_equality ; R( i ) = - C_l( i ) ; END DO
            DO i =  dims%c_l_start, dims%c_u_end  
              R( i ) = - SCALE_C( i ) * C( i ) ; END DO
          ELSE
            R( : dims%c_equality ) = - C_l( : dims%c_equality )
            R( dims%c_l_start : dims%c_u_end ) = - SCALE_C * C
          END IF
          CALL LPB_AX( m, R( : m), m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )
          WRITE( out, "( /, '  Constraint residual ', ES14.6,                  &
         &                  '  objective value ', ES14.6  )" )                 &
                 MAXVAL( ABS( R( : m) ) ), inform%obj
        END IF

!  Update penalty parameter, mu

        old_mu = mu ; mu = sigma * mu
        IF (  old_mu > zero ) THEN
          zeta = sigma * mu / old_mu 
        ELSE
          zeta = zero
        END IF

!  Record X and C

        IF ( mu < one .AND. stat_required ) THEN
          get_stat = .TRUE. 
          C_last( dims%c_l_start : dims%c_u_end )                            &
            = C( dims%c_l_start : dims%c_u_end )
          X_last = X

          DO i = dims%c_l_start, dims%c_u_start - 1
            Y_last( i ) = Y_l( i )
          END DO
          DO i = dims%c_u_start, dims%c_l_end
            IF ( DIST_C_l( i ) <= DIST_C_u( i ) ) THEN
              Y_last( i ) = Y_l( i )
            ELSE
              Y_last( i ) = Y_u( i )
            END IF
          END DO
          DO i = dims%c_l_end + 1, dims%c_u_end
            Y_last( i ) = Y_u( i )
          END DO

          Z_last( : dims%x_free ) = zero
          DO i = dims%x_free + 1, dims%x_u_start - 1
            Z_last( i ) = Z_l( i )
          END DO
          DO i = dims%x_u_start, dims%x_l_end
            IF ( DIST_X_l( i ) <= DIST_X_u( i ) ) THEN
              Z_last( i ) = Z_l( i )
            ELSE
              Z_last( i ) = Z_u( i )
            END IF
          END DO
          DO i = dims%x_l_end + 1, n
            Z_last( i ) = Z_u( i )
          END DO
        END IF

!  Update convergence tolerances, theta_c, theta_d and theta_e

        theta_c = MIN( theta_min,                                              &
                       MAX( theta_cs * mu ** beta, point99 * control%stop_c ) )
        theta_d = MIN( theta_min,                                              &
                       MAX( theta_df * mu ** beta, point99 * control%stop_d ) )

!  Recompute the value of the barrier function, phi

        phi = LPB_barrier_value( dims, n, obj, X, DIST_X_l, DIST_X_u,          &
                                  DIST_C_l, DIST_C_u, mu )

        IF ( .NOT. primal_hessian ) refact = .FALSE.
        full_iteration = .FALSE.

        IF ( ( inform%status == - 8 .OR. inform%status == - 9 ) .AND.          &
               old_mu > point1 * control%stop_c ) THEN
          inform%status = 0
          start_major = .TRUE.
          got_ratio = .FALSE.
        END IF
        IF ( inform%status /= 0 ) EXIT

!  ======================
!  End of outer iteration
!  ======================

      END DO   ! end of outer iteration loop

  600 CONTINUE 

!  Deallocate GLTR internal arrays

      CALL GLTR_terminate( gltr_data, gltr_control, gltr_info )

!  Print details of the solution obtained

      IF ( printi ) WRITE( out, 2120 ) inform%obj, inform%iter, inform%cg_iter

!  If required, make the solution exactly complementary

!  Problem variables

      IF ( control%feasol ) THEN 
        DO i = dims%x_free + 1, dims%x_l_start - 1
          IF ( ABS( Z_l( i ) ) < ABS( X( i ) ) ) THEN 
            Z_l( i ) = zero 
          ELSE 
            X( i ) = X_l( i ) 
          END IF 
        END DO
        DO i = dims%x_l_start, dims%x_l_end
          IF ( ABS( Z_l( i ) ) < ABS( DIST_X_l( i ) ) ) THEN 
            Z_l( i ) = zero 
          ELSE 
            X( i ) = X_l( i ) 
          END IF 
        END DO
        DO i = dims%x_u_start, dims%x_u_end
          IF ( ABS( Z_u( i ) ) < ABS( DIST_X_u( i ) ) ) THEN 
            Z_u( i ) = zero 
          ELSE 
            X( i ) = X_u( i ) 
          END IF 
        END DO
        DO i = dims%x_u_end + 1, n
          IF ( ABS( Z_u( i ) ) < ABS( X( i ) ) ) THEN 
            Z_u( i ) = zero 
          ELSE 
            X( i ) = X_u( i ) 
          END IF 
        END DO

!  slack variables

        DO i = dims%c_l_start, dims%c_l_end
          IF ( ABS( Y_l( i ) ) < ABS( DIST_C_l( i ) ) ) THEN 
            Y_l( i ) = zero 
          ELSE 
            C( i ) = C_l( i ) 
          END IF 
        END DO
        DO i = dims%c_u_start, dims%c_u_end
          IF ( ABS( Y_u( i ) ) < ABS( DIST_C_u( i ) ) ) THEN 
            Y_u( i ) = zero 
          ELSE 
            C( i ) = C_u( i ) 
          END IF 
        END DO
      END IF 

  700 CONTINUE

!  Set the dual variables

      IF ( set_z ) THEN
        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 1,dims%x_free ; Z( i ) = zero ; END DO
        ELSE
          Z( : dims%x_free ) = zero
        END IF
  
        DO i = dims%x_free + 1, dims%x_u_start - 1
          Z( i ) = Z_l( i )
        END DO

        DO i = dims%x_u_start, dims%x_l_end
          IF ( ABS( Z_l( i ) ) <= ABS( Z_u( i ) ) ) THEN
            Z( i ) = Z_u( i )
          ELSE
            Z( i ) = Z_l( i )
          END IF
        END DO
  
        DO i = dims%x_l_end + 1, n
          Z( i ) = Z_u( i )
        END DO
      END IF

!  Unscale the constraint bounds

      IF ( scaled_c ) THEN
        DO i = dims%c_l_start, dims%c_l_end
          C_l( i ) = C_l( i ) * SCALE_C( i )
        END DO
  
        DO i = dims%c_u_start, dims%c_u_end
          C_u( i ) = C_u( i ) * SCALE_C( i )
        END DO
      END IF

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, m ; C_RES( i ) = zero ; END DO
      ELSE
        C_RES = zero
      END IF
      CALL LPB_AX( m, C_RES, m, A_ne, A_val, A_col, A_ptr, n, X, '+ ')
      IF ( printi .AND. m > 0 ) THEN 
        WRITE( out, "( '  Constraint residual ', ES12.4 )" )                   &
             MAX( zero, MAXVAL( ABS( C_l( : dims%c_equality) -                 &
                                     C_RES(: dims%c_equality ) ) ),            &
                        MAXVAL( C_l(  dims%c_l_start : dims%c_l_end ) -        &
                                C_RES(  dims%c_l_start : dims%c_l_end ) ),     &
                        MAXVAL( C_RES( dims%c_u_start : dims%c_u_end ) -       &
                                C_u( dims%c_u_start : dims%c_u_end ) ) )     
      END IF

!  If necessary, print warning messages

  810 CONTINUE
      IF ( printi ) then

        SELECT CASE( inform%status )
          CASE( - 1  ) ; WRITE( out, 2210 ) 
          CASE( - 3  ) ; WRITE( out, 2230 ) 
          CASE( - 5  ) ; WRITE( out, 2250 ) 
          CASE( - 6  ) ; WRITE( out, 2260 ) 
          CASE( - 7  ) ; WRITE( out, 2270 ) 
          CASE( - 8  ) ; WRITE( out, 2280 ) 
          CASE( - 9  ) ; WRITE( out, 2290 ) 
          CASE( - 10 ) ; WRITE( out, 2300 ) 
          CASE( - 11 ) ; WRITE( out, 2310 ) 
        END SELECT

        IF ( auto ) WRITE( out, 2400 )
        SELECT CASE( precon )
          CASE( 1 ) ; WRITE( out, 2410 )
          CASE( 2 ) ; WRITE( out, 2420 ) 
        END SELECT

        IF ( factor == 0 .OR. factor == 1 ) THEN
          WRITE( out, "( /, '  Schur-complement method used ' )" )
        ELSE
          WRITE( out, "( /, '  Augmented system method used ' )" )
        END IF

      END IF
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving LPB_solve_main ' )" )

      RETURN  

!  Compute total time

  920 CONTINUE 
      CALL CPU_TIME( time ) ; inform%time%total = time - time_start 

      RETURN  

!  Non-executable statements

 2000 FORMAT( /,' Iter   d-feas com-slk    obj     ratio   radius',            &
             ' nbacts cgits      time' ) 
 2010 FORMAT( I5, A1, 2ES8.1, ES9.1, A1, '    -   ', ES8.1, 1X,                &
            '     -     -', 0P, F10.2 ) 
 2020 FORMAT( I5, A1, 2ES8.1, ES9.1, A1, 2ES8.1, A1, 2I6, 0P, F10.2 ) 
 2030 FORMAT( I5, A1, 2ES8.1, ES9.1, A1, '    -   ', ES8.1, A1, 2I6, 0P, F10.2 )
 2040 FORMAT( '   **  Error return ', I6, ' from ', A15 ) 
 2050 FORMAT( '   **  Warning ', I6, ' from ', A15 ) 
 2060 FORMAT( I8, ' integer and ', I8, ' real words needed for factorization' )
 2070 FORMAT( ' ** Matrix has ', I7, ' zero eigenvalues ' )
 2080 FORMAT( /, ' Preconditioner is inappropriate as it has ', /, I6,         &
                 ' negative and ', I6, ' zero eigenvalues ', /                 &
                 ' Perturbing H', :, ' by ', ES12.4, ' and restarting ' )
 2090 FORMAT( '  maximum element of A = ', ES12.4,' maximum element of H = ',  &
              ES12.4 ) 
 2100 FORMAT( A10, 7ES10.2, /, ( 10X, 7ES10.2 ) ) 
 2120 FORMAT( ' ', /, '  Final objective function value ', ES22.14,            &
              /, '  Total number of iterations = ', I6,                        &
              /, '  Total number of c.g. its   = ', I6 )
 2130 FORMAT( /, ' ', 33( '=-' ), '=',                                         &
              /, '  mu = ', ES12.4, '  theta_c = ', ES12.4,                    &
                 '  theta_d = ', ES12.4,                                       &
              /, ' ', 33( '=-' ), '=' )
 2140 FORMAT( ' ',/,'       ***  Linesearch     step',                         &
              '      trial value           model value ' ) 
 2150 FORMAT( '                   ', ES12.4, 2ES22.14 ) 
 2160 FORMAT( /, ' min/max primal = ', 2ES12.4,                                &
              /, ' min/max dual   = ', 2ES12.4 )
 2180 FORMAT( A6, /, ( 8ES10.2 ) )
 2190 FORMAT( A6, /, ( 4( 2I5, ES10.2 ) ) )
 2200 FORMAT( /, ' *** changing preconditioner ... ', / )
 2210 FORMAT( /, '  Warning - input paramters incorrect ' ) 
 2230 FORMAT( /, '  Warning - input arrays not large enough ' ) 
 2250 FORMAT( /, '  Warning - the constraints are inconsistent ' ) 
 2260 FORMAT( /, '  Warning - the constraints appear to be inconsistent ' ) 
 2270 FORMAT( /, '  Warning - factorization failure ' ) 
 2280 FORMAT( /, '  Warning - no further progress possible ' ) 
 2290 FORMAT( /, '  Warning - step too small to make further progress ' ) 
 2300 FORMAT( /, '  Warning - iteration bound exceeded ' ) 
 2310 FORMAT( /, '  Warning - objective unbounded below ' ) 
 2350 FORMAT( /, '  Required integer storage for factors ', I12, ' exceeds', /,&
               '  existing storage',     I12, ' by more than a factor of', I6 )
 2360 FORMAT( /, '  Required real    storage for factors ', I12, ' exceeds', /,&
               '  existing storage',     I12, ' by more than a factor of', I6 )
 2370 FORMAT( /, '  Iteration time (', 0P, F9.2, ') exceeds', /,               &
            '    average time (',     F9.2, ') by more than a factor of', F9.2 )
 2380 FORMAT( /, '  Time (', 0P, F9.2, ') for full factorization exceeds ',    &
              /, '  time (',     F9.2, ') for barrier factorization' )
 2390 FORMAT( /, '  Time (', 0P, F9.2, ') for barrier factorization exceeds ', &
              /, '  time (',     F9.2, ') for full factorization' )
 2340 FORMAT( /, '  norm of residual', ES12.4, ' for barrier factorization',   &
            ' exceeds ', ES12.4 )
 2400 FORMAT( '  Automatic preconditioner ' )
 2410 FORMAT( '  Identity Hessian ' )
 2420 FORMAT( '  Barrier Hessian ' )

!     CONTAINS

!  Internal function that returns the largest element of X, unless
!  X has no elements, in which case zero is returned

!       FUNCTION LPB_max_vect( X )
!       REAL ( KIND = wp ) LPB_max_vect
!       REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: X

!       IF ( SIZE( X ) > 0 ) THEN
!          LPB_max_vect = MAXVAL( X )
!       ELSE
!          LPB_max_vect = zero
!       END IF

!       RETURN
!       END FUNCTION LPB_max_vect

!  End of LPB_solve_main

      END SUBROUTINE LPB_solve_main

!-*-*-*-*-*-*-   L P B _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE LPB_terminate( data, control, inform )

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
!   data    see Subroutine LPB_initialize
!   control see Subroutine LPB_initialize
!   inform  see Subroutine LPB_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPB_control_type ), INTENT( IN ) :: control        
      TYPE ( LPB_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      TYPE ( LSQP_inform_type ) :: LSQP_inform

!  Deallocate all arrays allocated by LSQP

      CALL LSQP_terminate( data, control%LSQP_control, LSQP_inform )
      IF ( LSQP_inform%status /= 0 )                                           &
        inform%status = LSQP_inform%status 

!  Deallocate QPP internal arrays

      CALL QPP_terminate( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform)
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,                &
                          data%QPP_inform)
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,           &
                          data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaining allocated arrays

      IF ( ALLOCATED( data%X_fixed ) ) THEN
        DEALLOCATE( data%X_fixed, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%X_fixed'
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
        END IF
      END IF

     IF ( ALLOCATED( data%C_fixed ) ) THEN
        DEALLOCATE( data%C_fixed, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%C_fixed'
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
        END IF
      END IF

     IF ( ALLOCATED( data%Index_X_fixed ) ) THEN
        DEALLOCATE( data%Index_X_fixed, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%Index_X_fixed'
          IF ( control%error > 0 ) WRITE( control%error, 2900 )                &
               inform%bad_alloc, inform%alloc_status
        END IF
      END IF

     IF ( ALLOCATED( data%Index_C_fixed ) ) THEN
        DEALLOCATE( data%Index_C_fixed, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%Index_C_fixed'
          IF ( control%error > 0 ) WRITE( control%error, 2900 )                &
             inform%bad_alloc, inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%GRAD ) ) THEN
        DEALLOCATE( data%GRAD, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%GRAD'
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%X_trial ) ) THEN
        DEALLOCATE( data%X_trial, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%X_trial'
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
        END IF
      END IF

     IF ( ALLOCATED( data%X0 ) ) THEN
        DEALLOCATE( data%X0, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%X0'
          IF ( control%error > 0 ) WRITE( control%error, 2900 )                &
               inform%bad_alloc, inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%GRAD_X_phi ) ) THEN
        DEALLOCATE( data%GRAD_X_phi, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%GRAD_X_phi'
          IF ( control%error > 0 )                                             &
             WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%GRAD_C_phi ) ) THEN
        DEALLOCATE( data%GRAD_C_phi, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%GRAD_C_phi'
          IF ( control%error > 0 )                                             &
             WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%S ) ) THEN
        DEALLOCATE( data%S, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 3
          inform%bad_alloc = 'lpb: data%S'
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) inform%bad_alloc, inform%alloc_status
        END IF
      END IF

      RETURN

!  Non-executable statement

 2900 FORMAT( ' ** Message from -LPB_terminate-', /,                          &
              ' Deallocation error, for ', A, /, ' status = ', I6 ) 

!  End of subroutine LPB_terminate

      END SUBROUTINE LPB_terminate

!-*-*-*-*-  L P B _ B A R R I E R _ V A L U E   S U B R O U T I N E   -*-*-*-*

      FUNCTION LPB_barrier_value( dims, n, objf, X, DIST_X_l, DIST_X_u,       &
                                   DIST_C_l, DIST_C_u, mu )
      REAL ( KIND = wp ) LPB_barrier_value

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the barrier function
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: mu 
      REAL ( KIND = wp ), INTENT( IN ) :: objf
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_l_start : dims%x_l_end ) :: DIST_X_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%x_u_start : dims%x_u_end ) :: DIST_X_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_l_end ) :: DIST_C_l
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_u_start : dims%c_u_end ) :: DIST_C_u

!  Local variables

      INTEGER :: i

! Compute the barrier terms

      LPB_barrier_value = zero

!  Problem variables: 

      DO i = dims%x_free + 1, dims%x_l_start - 1
        LPB_barrier_value = LPB_barrier_value + LOG( X( i ) ) 
      END DO 
      DO i = dims%x_l_start, dims%x_l_end
        LPB_barrier_value = LPB_barrier_value + LOG( DIST_X_l( i ) ) 
      END DO 
      DO i = dims%x_u_start, dims%x_u_end
        LPB_barrier_value = LPB_barrier_value + LOG( DIST_X_u( i ) ) 
      END DO 
      DO i = dims%x_u_end + 1, n
        LPB_barrier_value = LPB_barrier_value + LOG( - X( i ) ) 
      END DO 

!  Slack variables: 

      DO i = dims%c_l_start, dims%c_l_end
        LPB_barrier_value = LPB_barrier_value + LOG( DIST_C_l( i ) ) 
      END DO 
      DO i = dims%c_u_start, dims%c_u_end
        LPB_barrier_value = LPB_barrier_value + LOG( DIST_C_u( i ) ) 
      END DO 

!  Form the barrier function

      LPB_barrier_value = objf - mu * LPB_barrier_value

      RETURN  

!  End of LPB_barrier_value

      END FUNCTION LPB_barrier_value
 
!-*-  L P B _ I T E R A T I V E _ R E F I N E M E N T  S U B R O U T I N E -*-

      SUBROUTINE LPB_iterative_refinement(                                     &
                       dims, n, m, A_ne, A_val, A_col, A_ptr,                  &
                       RHS, l_rhs, factor, DIAG_X, ldiag_x, SCALE_C, DIAG_C,   &
                       ldiag_c_l, ldiag_c_u, SOL, RES, BEST, Y_trial, BEST_y,  &
                       RES_y, RES_x, precon, nnzks, itref_max,                 &
                       semi_norm, res_norm, big_res, K, FACTORS, CNTL,         &
                       print_level, control, inform ) 

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the block linear system

!     (  M_X             A(trans)  )
!     (           M_C   - SCALE_C  ) ( sol ) = ( rhs )
!     (   A   - SCALE_C            )

!  using iterative refinement, and returning sol in rhs

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( IN ) :: A_ne, print_level
      INTEGER, INTENT( IN ) :: itref_max, factor, precon, nnzks
      INTEGER, INTENT( IN ) :: l_rhs, ldiag_x, ldiag_c_l, ldiag_c_u
      REAL ( KIND = wp ), INTENT( OUT ) :: semi_norm, res_norm
      LOGICAL, INTENT( OUT ) :: big_res
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: SCALE_C
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ldiag_x ) :: DIAG_X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( ldiag_c_l : ldiag_c_u ) :: DIAG_C
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( l_rhs ) :: RHS 
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( dims%v_e ) :: SOL, RES
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( dims%v_e ) :: BEST
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( m ) :: Y_trial, BEST_y, RES_y
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RES_x
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
      TYPE ( SILS_control ), INTENT( INOUT ) :: CNTL
      TYPE ( LPB_control_type ), INTENT( IN ) :: control        
      TYPE ( LPB_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, it, itref_max_block
      REAL ( KIND = wp ) :: old_res
      TYPE ( SILS_sinfo ) :: SINFO

      big_res = .FALSE.
      IF ( control%out > 0 .AND. print_level >= 3 ) THEN
        IF ( dims%nc > 0 ) THEN
          WRITE( control%out, 2000 )                                           &
            MAXVAL( ABS( RHS( dims%x_s : dims%x_e ) ) ),                       &
            MAXVAL( ABS( RHS( dims%c_s : dims%c_e ) ) ),                       &
            MAXVAL( ABS( RHS( dims%y_s : dims%y_e ) ) )
        ELSE IF ( m > 0 ) THEN
          WRITE( control%out, 2010 )                                           &
            MAXVAL( ABS( RHS( dims%x_s : dims%x_e ) ) ),                       &
            MAXVAL( ABS( RHS( dims%y_s : dims%y_e ) ) )
        ELSE
          WRITE( control%out, 2020 ) MAXVAL( ABS( RHS( dims%x_s : dims%x_e ) ) )
        END IF
      END IF

!     write(6,*) ' l_rhs ', l_rhs, dims%v_e, lbound( RHS ), &
!        ubound( RHS ), size( RHS )
      old_res = NRM2( dims%v_e, RHS( : dims%v_e ), 1 )
!     old_res = MAXVAL( ABS( RHS ) )
      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, dims%v_e ; SOL( i ) = RHS( i ) ; END DO
      ELSE
        SOL = RHS
      END IF

      IF ( factor == 0 .OR. factor == 1 ) THEN
        itref_max_block = itref_max - 1
        CALL LPB_block_solve( dims, n, m, SOL( dims%x_s : dims%x_e ),          &
                               SOL( dims%c_s : dims%c_e ),                     &
                               SOL( dims%y_s : dims%y_e ),                     &
                               A_ne, A_val, A_col, A_ptr, DIAG_X, ldiag_x,     &
                               SCALE_C, DIAG_C, ldiag_c_l, ldiag_c_u,          &
                               Y_trial, BEST_y, RES_y, RES_x, itref_max_block, &
                               big_res, K, FACTORS, CNTL, print_level,         &
                               control, inform )
                               
      ELSE
        CALL SILS_solve( K, FACTORS, SOL, CNTL, SINFO )
        inform%factorization_status = SINFO%flag
        IF ( SINFO%flag /= 0 ) THEN
          IF ( control%error > 0 .AND. print_level >= 1 )                      &
              WRITE( control%error, 2100 ) SINFO%flag
          inform%status = - 7 ; RETURN
        END IF
      END IF

      IF ( control%out > 0 .AND. print_level >= 3 ) THEN
        IF ( dims%nc > 0 ) THEN
          WRITE( control%out, 2030 )                                           &
            MAXVAL( ABS( SOL( dims%x_s : dims%x_e ) ) ),                       &
            MAXVAL( ABS( SOL( dims%c_s : dims%c_e ) ) ),                       &
            MAXVAL( ABS( SOL( dims%y_s : dims%y_e ) ) )
        ELSE IF ( m > 0 ) THEN
          WRITE( control%out, 2040 )                                           &
            MAXVAL( ABS( SOL( dims%x_s : dims%x_e ) ) ),                       &
            MAXVAL( ABS( SOL( dims%y_s : dims%y_e ) ) )
        ELSE
          WRITE( control%out, 2050 ) MAXVAL( ABS( SOL( dims%x_s : dims%x_e ) ) )
        END IF
      END IF

!  Perform the iterative refinement

      DO it = 1, itref_max

!  Compute the residual

        IF ( factor == 0 .OR. factor == 1 ) THEN
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = dims%x_s, dims%x_e
              RES( i ) = RHS( i ) - DIAG_X( i ) * SOL( i ) ; END DO
            DO i = 0, dims%nc - 1
              RES( dims%c_s + i ) = RHS( dims%c_s + i ) -                      &
                DIAG_C( dims%c_l_start + i ) * SOL( dims%c_s + i ) ; END DO
          ELSE
            RES( dims%x_s : dims%x_e ) = RHS( dims%x_s : dims%x_e ) -          &
                                          DIAG_X * SOL( dims%x_s : dims%x_e )
            RES( dims%c_s : dims%c_e ) = RHS( dims%c_s : dims%c_e ) -          &
                                           DIAG_C * SOL( dims%c_s : dims%c_e )
          END IF
        ELSE
          IF ( precon > 1 ) THEN
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = dims%x_s, dims%c_e
                RES( i ) = RHS( i ) - K%val( nnzks + i ) * SOL( i ) ; END DO
            ELSE
              RES( dims%x_s : dims%c_e ) = RHS( dims%x_s : dims%c_e ) -        &
                K%val( nnzks + dims%x_s : nnzks + dims%c_e ) *                 &
                  SOL( dims%x_s : dims%c_e )
            END IF
          ELSE
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = dims%x_s, dims%c_e
                RES( i ) = RHS( i ) - k_diag * SOL( i ) ; END DO
            ELSE
              RES( dims%x_s : dims%c_e ) = RHS( dims%x_s : dims%c_e ) -        &
                k_diag * SOL( dims%x_s : dims%c_e )
            END IF
          END IF
 
        END IF

!  Include the contribution from the slack variables

        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 0, dims%nc - 1
            RES( dims%c_s + i ) = RES( dims%c_s + i ) +                        &
                SCALE_C( dims%c_l_start + i ) * SOL( dims%y_i + i )
            RES( dims%y_i + i ) = RHS( dims%y_i + i ) +                        &
                SCALE_C( dims%c_l_start + i ) * SOL( dims%c_s + i )  
          END DO
          DO i = dims%y_s, dims%y_i - 1 ; RES( i ) = RHS( i ) ; END DO
        ELSE
          RES( dims%c_s : dims%c_e ) =                                         &
            RES( dims%c_s : dims%c_e ) + SCALE_C * SOL( dims%y_i : dims%y_e )
          RES( dims%y_s : dims%y_i - 1 ) = RHS( dims%y_s : dims%y_i - 1 )
          RES( dims%y_i : dims%y_e ) =                                         &
            RHS( dims%y_i : dims%y_e ) + SCALE_C * SOL( dims%c_s : dims%c_e )
        END IF

!  Include the contribution from A

        CALL LPB_AX( n, RES( dims%x_s : dims%x_e ), m, A_ne, A_val, A_col,     &
                      A_ptr, m, SOL( dims%y_s : dims%y_e ), '-T' )
        CALL LPB_AX( m, RES( dims%y_s : dims%y_e ), m, A_ne, A_val, A_col,     &
                      A_ptr, n, SOL( dims%x_s : dims%x_e ), '- ' )

        IF ( control%out > 0 .AND. print_level >= 3 ) THEN
          IF ( dims%nc > 0 ) THEN
            WRITE( control%out, 2000 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%c_s : dims%c_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%y_s : dims%y_e ) ) )
          ELSE IF ( m > 0 ) THEN
            WRITE( control%out, 2010 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%y_s : dims%y_e ) ) )
          ELSE
            WRITE( control%out, 2020 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) )
          END IF
        END IF

!  Compute the norm of the residual

        res_norm = NRM2( dims%v_e, RES, 1 )
!       res_norm = MAXVAL( ABS( RES ) )
        IF ( res_norm > res_large ) THEN
          big_res = .TRUE.
          RETURN
        END IF

!  If the norm has increased, quit

        IF ( it > 1 ) THEN
          IF ( res_norm < old_res ) THEN
            old_res = res_norm
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, dims%v_e ; BEST( i ) = SOL( i ) ; END DO
            ELSE
              BEST = SOL
            END IF
          ELSE
            res_norm = old_res
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, dims%v_e ; SOL( i ) = BEST( i ) ; END DO
            ELSE
              SOL = BEST
            END IF
            GO TO 100
          END IF
        ELSE
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, dims%v_e ; BEST( i ) = SOL( i ) ; END DO
          ELSE
            BEST = SOL
          END IF
        END IF
   
!  Obtain a new correction

        IF ( factor == 0 .OR. factor == 1 ) THEN
          CALL LPB_block_solve( dims, n, m, RES( dims%x_s : dims%x_e ),        &
                                RES( dims%c_s : dims%c_e ),                    &
                                RES( dims%y_s : dims%y_e ),                    &
                                A_ne, A_val, A_col, A_ptr,                     &
                                DIAG_X, ldiag_x, SCALE_C, DIAG_C, ldiag_c_l,   &
                                ldiag_c_u, Y_trial, BEST_y, RES_y, RES_x,      &
                                itref_max_block, big_res, K, FACTORS, CNTL,    &
                                print_level, control, inform )
        ELSE
          CALL SILS_solve( K, FACTORS, RES, CNTL, SINFO )
          inform%factorization_status = SINFO%flag
          IF ( SINFO%flag /= 0 ) THEN
            IF ( control%error > 0 .AND. print_level >= 1 )                    &
              WRITE( control%error, 2100 ) SINFO%flag
            inform%status = - 7 ; RETURN
          END IF
        END IF

        IF ( control%out > 0 .AND. print_level >= 3 ) THEN
          IF ( dims%nc > 0 ) THEN
            WRITE( control%out, 2060 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%c_s : dims%c_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%y_s : dims%y_e ) ) )
          ELSE IF ( m > 0 ) THEN
            WRITE( control%out, 2070 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%y_s : dims%y_e ) ) )
          ELSE
            WRITE( control%out, 2080 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) )
          END IF
        END IF
        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 1, dims%v_e ; SOL( i ) = SOL( i ) + RES( i ) ; END DO
        ELSE
          SOL = SOL + RES
        END IF

      END DO

!  Obtain final residuals if required

      IF ( it >= itref_max .OR. itref_max == 0 ) THEN
      
!  Compute the residual

        IF ( factor == 0 .OR. factor == 1 ) THEN
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = dims%x_s, dims%x_e
              RES( i ) = RHS( i ) - DIAG_X( i ) * SOL( i ) ; END DO
            DO i = 0, dims%nc - 1
              RES( dims%c_s + i ) = RHS( dims%c_s + i ) -                      &
                DIAG_C( dims%c_l_start + i ) * SOL( dims%c_s + i ) ; END DO
          ELSE
            RES( dims%x_s : dims%x_e ) = RHS( dims%x_s : dims%x_e ) -          &
                                           DIAG_X * SOL( dims%x_s : dims%x_e )
            RES( dims%c_s : dims%c_e ) = RHS( dims%c_s : dims%c_e ) -          &
                                          DIAG_C * SOL( dims%c_s : dims%c_e )
          END IF
        ELSE
          IF ( precon > 1 ) THEN
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = dims%x_s, dims%c_e
                RES( i ) = RHS( i ) - K%val( nnzks + i ) * SOL( i ) ; END DO
            ELSE
              RES( dims%x_s : dims%c_e ) = RHS( dims%x_s : dims%c_e ) -        &
                K%val( nnzks + dims%x_s : nnzks + dims%c_e ) *                 &
                  SOL( dims%x_s : dims%c_e )
            END IF
          ELSE
            IF ( control%array_syntax_worse_than_do_loop ) THEN
               DO i = dims%x_s, dims%c_e
                RES( i ) = RHS( i ) - k_diag * SOL( i ) ; END DO
            ELSE
              RES( dims%x_s : dims%c_e ) = RHS( dims%x_s : dims%c_e ) -        &
                k_diag * SOL( dims%x_s : dims%c_e )
            END IF
          END IF
        END IF
 
!  Include the contribution from the slack variables

        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 0, dims%nc - 1
            RES( dims%c_s + i ) = RES( dims%c_s + i )                          &
              + SCALE_C( dims%c_l_start + i ) * SOL( dims%y_i + i )
            RES( dims%y_i + i ) = RHS( dims%y_i + i )                          &
              + SCALE_C( dims%c_l_start + i ) * SOL( dims%c_s + i )
          END DO
          DO i = dims%y_s, dims%y_i - 1 ; RES( i ) = RHS( i ) ; END DO
        ELSE
          RES( dims%c_s : dims%c_e ) =                                         &
            RES( dims%c_s : dims%c_e ) + SCALE_C * SOL( dims%y_i : dims%y_e )
          RES( dims%y_s : dims%y_i - 1 ) = RHS( dims%y_s : dims%y_i - 1 )
          RES( dims%y_i : dims%y_e ) =                                         &
            RHS( dims%y_i : dims%y_e ) + SCALE_C * SOL( dims%c_s : dims%c_e )
        END IF

!  Include the contribution from A

        CALL LPB_AX( n, RES( dims%x_s : dims%x_e ), m, A_ne, A_val, A_col,     &
                      A_ptr, m, SOL( dims%y_s : dims%y_e ), '-T' )
        CALL LPB_AX( m, RES( dims%y_s : dims%y_e ), m, A_ne, A_val, A_col,     &
                      A_ptr, n, SOL( dims%x_s : dims%x_e ), '- ' )

        IF ( control%out > 0 .AND. print_level >= 3 ) THEN
          IF ( dims%nc > 0 ) THEN
            WRITE( control%out, 2000 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%c_s : dims%c_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%y_s : dims%y_e ) ) )
          ELSE IF ( m > 0 ) THEN
            WRITE( control%out, 2010 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) ),                     &
              MAXVAL( ABS( RES( dims%y_s : dims%y_e ) ) )
          ELSE
            WRITE( control%out, 2020 )                                         &
              MAXVAL( ABS( RES( dims%x_s : dims%x_e ) ) )
          END IF
        END IF
        res_norm = NRM2( dims%v_e, RES, 1 )
!       res_norm = MAXVAL( ABS( RES ) )
        IF ( res_norm > res_large ) THEN
          big_res = .TRUE.
          RETURN
        END IF
      END IF

 100  CONTINUE
      IF ( control%out > 0 .AND. print_level >= 3 ) THEN
        IF ( dims%nc > 0 ) THEN
          WRITE( control%out, 2030 )                                           &
            MAXVAL( ABS( SOL( dims%x_s : dims%x_e ) ) ),                       &
            MAXVAL( ABS( SOL( dims%c_s : dims%c_e ) ) ),                       &
            MAXVAL( ABS( SOL( dims%y_s : dims%y_e ) ) )
        ELSE IF ( m > 0 ) THEN
          WRITE( control%out, 2040 )                                           &
            MAXVAL( ABS( SOL( dims%x_s : dims%x_e ) ) ),                       &
            MAXVAL( ABS( SOL( dims%y_s : dims%y_e ) ) )
        ELSE
          WRITE( control%out, 2050 ) MAXVAL( ABS( SOL( dims%x_s : dims%x_e ) ) )
        END IF
      END IF

      semi_norm = zero
      DO i = 1, dims%c_e 
        semi_norm = semi_norm + RHS( i ) * SOL( i )
        RHS( i ) = SOL( i )
      END DO
      semi_norm = SQRT( ABS( semi_norm ) )
      DO i = dims%c_e + 1, dims%v_e
        RHS( i ) = SOL( i )
      END DO

      inform%status = 0
      RETURN

!  Non-executable statements

 2000 FORMAT( '  res(dual) ', ES12.4, '  res(slack ) ', ES12.4,                &
              '  res(primal) ', ES12.4 )
 2010 FORMAT( '  res(dual) ', ES12.4, '  res(primal) ', ES12.4 )
 2020 FORMAT( '  res(dual) ', ES12.4 )
 2030 FORMAT( '  sol(x   ) ', ES12.4, '  sol(slack ) ', ES12.4,                &
              '  sol(y     ) ', ES12.4 )
 2040 FORMAT( '  sol(x   ) ', ES12.4, '  sol(y     ) ', ES12.4 )
 2050 FORMAT( '  sol(x   ) ', ES12.4 )
 2060 FORMAT( ' dsol(x   ) ', ES12.4, ' dsol(slack ) ', ES12.4,                &
              ' dsol(y     ) ', ES12.4 )
 2070 FORMAT( ' dsol(x   ) ', ES12.4, ' dsol(y     ) ', ES12.4 )
 2080 FORMAT( ' dsol(x   ) ', ES12.4 )
 2100 FORMAT( '   **  Error return ', I3, ' from SILS_solve ' ) 

!  End of LPB_iterative_refinement

      END SUBROUTINE LPB_iterative_refinement

!-*-*-*-*-*-   L P B _ B L O C K _ S O L V E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LPB_block_solve( dims, n, m, RHS_x, RHS_c, RHS_y, A_ne,       &
                                  A_val, A_col, A_ptr, DIAG_X, ldiag_x,        &
                                  SCALE_C, DIAG_C, ldiag_c_l, ldiag_c_u,       &
                                  Y_trial, BEST_y, RES_y, RES_x, itref_max,    &
                                  big_res, K, FACTORS, CNTL, print_level,      &
                                  control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the block system 

!     ( DIAG_X           A(trans)  ) (sol_x) = (rhs_x)
!     (         DIAG_C  - SCALE_C  ) (sol_c)   (rhs_c)
!     (   A   - SCALE_C            ) (sol_y)   (rhs_y)

!  returning ( sol_x , sol_c, sol_y ) in ( rhs_x , rhs_c, rhs_y )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!   Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( IN ) :: A_ne, ldiag_x, ldiag_c_l, ldiag_c_u
      INTEGER, INTENT( IN ) :: itref_max, print_level
      LOGICAL, INTENT( INOUT ) :: big_res
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_start : dims%c_u_end ) :: SCALE_C
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ldiag_x ) :: DIAG_X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( ldiag_c_l : ldiag_c_u ) :: DIAG_C
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: RHS_x
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: RHS_y
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( dims%c_l_start : m ) :: RHS_c
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( m ) :: Y_trial, BEST_y, RES_y
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RES_x
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
      TYPE ( SILS_control ), INTENT( INOUT ) :: CNTL
      TYPE ( LPB_control_type ), INTENT( IN ) :: control        
      TYPE ( LPB_inform_type ), INTENT( INOUT ) :: inform

      INTEGER :: i

!  Obtain DIAG_X(inv) * rhs_x ...

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, n ; RHS_x( i ) = RHS_x( i ) / DIAG_X( i ) ; END DO
      ELSE
        RHS_x = RHS_x / DIAG_X
      END IF

      IF (  m == 0 ) RETURN

!  ... and DIAG_C(inv) * rhs_c

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = dims%c_l_start, m
          RHS_c( i ) = RHS_c( i ) / DIAG_C( i ) ; END DO
      ELSE
        RHS_c = RHS_c / DIAG_C
      END IF

!  Now find A * DIAG_X(inv) * rhs_x - rhs_y - SCALE_C * DIAG_C(inv) * rhs_c

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, dims%c_equality ; RHS_y( i ) = - RHS_y( i ) ; END DO
        DO i = dims%c_l_start,  dims%c_u_end
          RHS_y( i ) = - RHS_y( i ) - SCALE_C( i ) * RHS_c( i ) ; END DO
      ELSE
        RHS_y( : dims%c_equality ) = - RHS_y( : dims%c_equality )
        RHS_y( dims%c_l_start : ) = - RHS_y( dims%c_l_start : ) - SCALE_C *RHS_c
      END IF
      CALL LPB_AX( m, RHS_y, m, A_ne, A_val, A_col, A_ptr, n, RHS_x, '+ ' )

!  Next solve K sol_y = A * DIAG_X(inv) * rhs_x - rhs_y - DIAG_X(inv) * rhs_c, 
!  where K = A * DIAG_X(inv) * A(trans) + SCALE_C * DIAG_C(inv) * SCALE_C

      CALL LPB_block_iterative_refinement(                                     &
                dims, n, m, K, FACTORS, CNTL, RHS_y, A_ne, A_val, A_col,       &
                A_ptr, DIAG_X, ldiag_x, SCALE_C, DIAG_C, ldiag_c_l, ldiag_c_u, &
                Y_trial, BEST_y, RES_y, RES_x, itref_max, big_res,             &
                print_level, control, inform )

!  Finally, recover rhs_x = DIAG_X(inv) ( rhs_x - A(trans) sol_y ) ...

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, n ; RHS_x( i ) = RHS_x( i ) * DIAG_X( i ) ; END DO
      ELSE
        RHS_x = RHS_x * DIAG_X
      END IF

      CALL LPB_AX( n, RHS_x, m, A_ne, A_val, A_col, A_ptr, m, RHS_y, '-T' )

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, n ; RHS_x( i ) = RHS_x( i ) / DIAG_X( i ) ; END DO
      ELSE
        RHS_x = RHS_x / DIAG_X
      END IF

!  ... and rhs_c = DIAG_C(inv) ( rhs_c + SCALE_C * sol_y )

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = dims%c_l_start,  dims%c_u_end
          RHS_c( i ) = RHS_c( i ) + ( SCALE_C( i ) / DIAG_C( i ) ) * RHS_y( i )
        END DO
      ELSE
        RHS_c = RHS_c + ( SCALE_C / DIAG_C ) * RHS_y( dims%c_l_start : )
      END IF

      RETURN

!  End of subroutine LPB_block_solve

      END SUBROUTINE LPB_block_solve

!-*-*-*-*-*-   L P B _ B L O C K _ I-R   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE LPB_block_iterative_refinement( dims, n, m, K, FACTORS,       &
                            CNTL, RHS_y, A_ne, A_val, A_col, A_ptr, DIAG_X,    &
                            ldiag_x, SCALE_C, DIAG_C, ldiag_c_l, ldiag_c_u,    &
                            Y_trial, BEST_y, RES_y, RES_x,                     &
                            itref_max, big_res, print_level, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the system 

!   (  A * DIAG_X(inv) * A(trans) + SCALE_C * DIAG_C(inv) * SCALE_C ) *
!     sol_y = rhs_y

!  using iterative refinement, and returning sol_y in rhs_y

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( IN ) :: A_ne, ldiag_x, ldiag_c_l, ldiag_c_u
      INTEGER, INTENT( IN ) :: itref_max, print_level
      LOGICAL, INTENT( INOUT ) :: big_res
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ldiag_x ) :: DIAG_X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_start : dims%c_u_end ) :: SCALE_C
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( ldiag_c_l : ldiag_c_u ) :: DIAG_C
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: RHS_y
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( m ) :: Y_trial, BEST_y, RES_y
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RES_x
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
      TYPE ( SILS_control ), INTENT( IN ) :: CNTL
      TYPE ( LPB_control_type ), INTENT( IN ) :: control        
      TYPE ( LPB_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, it
      REAL ( KIND = wp ) :: res_norm, old_res
      TYPE ( SILS_sinfo ) :: SINFO

      old_res = NRM2( m, RHS_y, 1 )
!     old_res = MAXVAL( ABS( RHS_y ) )
      IF ( control%out > 0 .AND. print_level >= 3 )                            &
        WRITE( control%out, 2000 ) old_res

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, m ; Y_trial( i ) = RHS_y( i ) ; END DO
      ELSE
        Y_trial = RHS_y
      END IF

      CALL SILS_solve( K, FACTORS, Y_trial, CNTL, SINFO )
      inform%factorization_status = SINFO%flag
      IF ( SINFO%flag /= 0 ) THEN
        IF ( control%error > 0 .AND. print_level >= 1 )                        &
          WRITE( control%error, 2040 ) SINFO%flag
        inform%status = - 7 ; RETURN
      END IF

      IF ( control%out > 0 .AND. print_level >= 3 )                            &
        WRITE( control%out, 2010 ) MAXVAL( ABS( Y_trial ) )

      IF ( itref_max > 0 ) THEN
        DO it = 1, itref_max

!  Form res_x = DIAG_X(inv) * A(trans) sol_y

          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, n ; RES_x( i ) = zero ; END DO
          ELSE
            RES_x = zero
          END IF
          CALL LPB_AX( n, RES_x, m, A_ne, A_val, A_col,  A_ptr, m, Y_trial,'+T')
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, n ; RES_x( i ) = RES_x( i ) / DIAG_X( i ) ; END DO
          ELSE
            RES_x = RES_x / DIAG_X
          END IF

!  Form res_y = rhs_y - A * res_x - SCALE_C * DIAG_C(inv) * SCALE_C * sol_y

          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, dims%c_equality ; RES_y( i ) = RHS_y( i ) ; END DO
            DO i = dims%c_l_start,  dims%c_u_end
              RES_y( i ) = RHS_y( i ) - ( SCALE_C( i ) / DIAG_C( i ) ) *       &
                             SCALE_C( i ) * Y_trial( i )
            END DO
          ELSE
            RES_y( : dims%c_equality ) = RHS_y( : dims%c_equality )
            RES_y( dims%c_l_start : ) = RHS_y( dims%c_l_start : )              &
                   - ( SCALE_C / DIAG_C ) * SCALE_C * Y_trial( dims%c_l_start :)
          END IF
          CALL LPB_AX( m, RES_y, m, A_ne, A_val, A_col, A_ptr, n, RES_x, '- ' )
  
!  Print residuals if required

          res_norm = NRM2( m, RES_y, 1 )
!         res_norm = MAXVAL( ABS( RES_y ) )

          IF ( control%out > 0 .AND. print_level >= 3 )                        &
            WRITE( control%out, 2000 ) res_norm
  
!  Check that the residuals are not too large

          IF ( res_norm > res_large ) THEN
            big_res = .TRUE.
            RETURN
          END IF

          IF ( it > 1 ) THEN
            IF ( res_norm < old_res ) THEN
              old_res = res_norm
              IF ( control%array_syntax_worse_than_do_loop ) THEN
                DO i = 1, m ; BEST_y( i ) = Y_trial( i ) ; END DO
              ELSE
                BEST_y = Y_trial
              END IF
            ELSE
              IF ( control%array_syntax_worse_than_do_loop ) THEN
                DO i = 1, m ; Y_trial( i ) = BEST_y( i ) ; END DO
              ELSE
                Y_trial = BEST_y
              END IF
              EXIT
            END IF
          ELSE
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, m ; BEST_y( i ) = Y_trial( i ) ; END DO
            ELSE
              BEST_y = Y_trial
            END IF
          END IF
     
          CALL SILS_solve( K, FACTORS, RES_y, CNTL, SINFO )
          inform%factorization_status = SINFO%flag
          IF ( SINFO%flag /= 0 ) THEN
            IF ( control%error > 0 .AND. print_level >= 1 )                    &
              WRITE( control%error, 2040 ) SINFO%flag
            inform%status = - 7 ; RETURN
          END IF
          IF ( control%out > 0 .AND. print_level >= 3 )                        &
            WRITE( control%out, 2020 ) MAXVAL( ABS( RES_y ) )
          IF ( control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, m ; Y_trial( i ) = Y_trial( i ) + RES_y( i ) ; END DO
          ELSE
            Y_trial = Y_trial + RES_y
          END IF

!  Print final residuals if required

          IF ( it == itref_max ) THEN
   
!  Form res_x = DIAG_X(inv) * A(trans) sol_y

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, n ; RES_x( i ) = zero ; END DO
            ELSE
              RES_x = zero
            END IF
            CALL LPB_AX( n, RES_x, m, A_ne, A_val, A_col, A_ptr, m,Y_trial,'+T')
            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, n ; RES_x( i ) = RES_x( i ) / DIAG_X( i ) ; END DO
            ELSE
              RES_x = RES_x / DIAG_X
            END IF

!  Form res_y = rhs_y - A * res_x - SCALE_C * DIAG_C(inv) * SCALE_C * sol_y

            IF ( control%array_syntax_worse_than_do_loop ) THEN
              DO i = 1, dims%c_equality ; RES_y( i ) = RHS_y( i ) ; END DO
              DO i = dims%c_l_start,  dims%c_u_end
                RES_y( i ) = RHS_y( i ) - ( SCALE_C( i ) / DIAG_C( i ) ) *     &
                               SCALE_C( i ) * Y_trial( i )
              END DO
            ELSE
              RES_y( : dims%c_equality ) = RHS_y( : dims%c_equality )
              RES_y( dims%c_l_start : ) = RHS_y( dims%c_l_start : )            &
                    - ( SCALE_C / DIAG_C ) * SCALE_C * Y_trial( dims%c_l_start:)
            END IF
            CALL LPB_AX( m, RES_y, m, A_ne, A_val, A_col, A_ptr, n, RES_x, '- ')
     
!  Print residuals if required

            res_norm = NRM2( m, RES_y, 1 )
!           res_norm = MAXVAL( ABS( RES_y ) )

            IF ( control%out > 0 .AND. print_level >= 3 )                      &
              WRITE( control%out, 2000 ) res_norm

!  Check that the residuals are not too large

            IF ( res_norm > res_large ) THEN
              big_res = .TRUE.
              RETURN
            END IF

          END IF
        END DO
      ELSE

!  Form res_x = DIAG_X(inv) * A(trans) sol_y

        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 1, n ; RES_x( i ) = zero ; END DO
        ELSE
          RES_x = zero
        END IF
        CALL LPB_AX( n, RES_x, m, A_ne, A_val, A_col, A_ptr, m, Y_trial, '+T' )
        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 1, n ; RES_x( i ) = RES_x( i ) / DIAG_X( i ) ; END DO
        ELSE
          RES_x = RES_x / DIAG_X
        END IF

!  Form res_y = rhs_y - A * res_x - DIAG_C(inv) sol_y

        IF ( control%array_syntax_worse_than_do_loop ) THEN
          DO i = 1, dims%c_equality ; RES_y( i ) = RHS_y( i ) ; END DO
          DO i = dims%c_l_start,  dims%c_u_end
            RES_y( i ) = RHS_y( i ) - ( SCALE_C( i ) / DIAG_C( i ) ) *         &
                           SCALE_C( i ) * Y_trial( i )
          END DO
        ELSE
          RES_y( : dims%c_equality ) = RHS_y( : dims%c_equality )
          RES_y( dims%c_l_start : ) = RHS_y( dims%c_l_start : )                &
                - ( SCALE_C / DIAG_C ) * SCALE_C * Y_trial( dims%c_l_start : )
        END IF
        CALL LPB_AX( m, RES_y, m, A_ne, A_val, A_col, A_ptr, n, RES_x, '- ' )
      
!  Print residuals if required
   
        res_norm = NRM2( m, RES_y, 1 )
!       res_norm = MAXVAL( ABS( RES_y ) )

        IF ( control%out > 0 .AND. print_level >= 3 )                          &
          WRITE( control%out, 2000 ) res_norm

!  Check that the residuals are not too large

        IF ( res_norm > res_large ) THEN
          big_res = .TRUE.
          RETURN
        END IF

      END IF

      IF ( control%out > 0 .AND. print_level >= 3 )                            &
        WRITE( control%out, 2010 ) MAXVAL( ABS( Y_trial ) )

      IF ( control%array_syntax_worse_than_do_loop ) THEN
        DO i = 1, m ; RHS_y( i ) = Y_trial( i ) ; END DO
      ELSE
        RHS_y = Y_trial
      END IF
      inform%status = 0
      RETURN

!  Non-executable statements

 2000 FORMAT( '  res(y) ', ES12.4 )
 2010 FORMAT( '  sol(y) ', ES12.4 )
 2020 FORMAT( ' dsol(y) ', ES12.4 )
 2040 FORMAT( '   **  Error return ', I3, ' from SILS_solve ' ) 

!  End of LPB_block_iterative_refinement

      END SUBROUTINE LPB_block_iterative_refinement

!-*-*-*-*-*-*-*-   L P B _ A N A L Y S E   S U B R O U T I N E  -*-*-*-*-*-*-

      SUBROUTINE LPB_analyse( dims, n, m, A_ne, A_val, A_col,                  &
                              A_ptr, SCALE_C, factor, precon,      &
                              nnzks, lk, liw, ldiag_x, ldiag_c_l,      &
                              ldiag_c_u, IW, Abycol_val, Abycol_row,           &
                              Abycol_ptr, K_colptr, DIAG_X, DIAG_C,            &
                              K, FACTORS, CNTL, print_level, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Analyse the sparsity pattern of the preconditioner

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, A_ne, print_level
      INTEGER, INTENT( IN ) :: precon
      INTEGER, INTENT( INOUT ) :: factor
      INTEGER, INTENT( OUT ) :: nnzks, lk, liw, ldiag_x, ldiag_c_l, ldiag_c_u
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: SCALE_C
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, K_colptr, Abycol_ptr, Abycol_row
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIAG_X, DIAG_C, Abycol_val
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      TYPE ( SILS_factors ), INTENT( INOUT ) :: FACTORS
      TYPE ( SILS_control ), INTENT( INOUT ) :: CNTL
      TYPE ( LPB_control_type ), INTENT( IN ) :: control        
      TYPE ( LPB_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ierr, l, max_col, max_len, nnzk, y_ii
      LOGICAL :: printi, printt, printe, reallocate, get_factors
      TYPE ( SILS_ainfo ) :: AINFO

!  Set useful values

      printe = control%error > 0 .AND. print_level >= 1
      printt = control%out > 0 .AND. print_level >= 2 
      printi = control%out > 0 .AND. print_level >= 1 
      max_col = control%max_col
      IF ( max_col < 0 ) max_col = n
      inform%alloc_status = 0

      IF ( factor < 0 .OR. factor > 2 ) factor = 0

!  Print a header indicating the method selected

   10 CONTINUE

      IF ( printi ) THEN
        SELECT CASE( precon )
          CASE( 1 ) ; WRITE( control%out, 2710 )
          CASE( 2 ) ; WRITE( control%out, 2720 ) 
        END SELECT

        IF ( factor == 0 .OR. factor == 1 ) THEN
          WRITE( control%out, "( '  Schur-complement method used ', / )" )
        ELSE
          WRITE( control%out, "( '  Augmented system method used ', / )" )
        END IF
      END IF

      get_factors = .TRUE.

!  For the Schur complement matrix
!  ===============================

      IF ( factor == 0 .OR. factor == 1 ) THEN

!  Allocate the arrays for the analysis phase

        liw = MAX( n + 1, n + m )
        reallocate = .TRUE.
        IF ( ALLOCATED( IW ) ) THEN
          IF ( SIZE( IW ) < liw ) THEN ; DEALLOCATE( IW )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( IW( liw ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%IW' ; GO TO 900 ; END IF
        END IF

!  Continue allocating the arrays for the analysis phase

        reallocate = .TRUE.
        IF ( ALLOCATED( Abycol_val ) ) THEN
          IF ( SIZE( Abycol_val ) /= A_ne ) THEN 
            DEALLOCATE( Abycol_val )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( Abycol_val( A_ne ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%Abycol_val' ; GO TO 900
          END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( Abycol_row ) ) THEN
          IF ( SIZE( Abycol_row ) /= A_ne ) THEN 
            DEALLOCATE( Abycol_row )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( Abycol_row( A_ne ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%Abycol_row' ; GO TO 900
          END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( Abycol_ptr ) ) THEN
          IF ( SIZE( Abycol_ptr ) /= n + 1 ) THEN 
            DEALLOCATE( Abycol_ptr )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( Abycol_ptr( n + 1 ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%Abycol_ptr' ; GO TO 900
          END IF
        END IF

        IF ( m > 0 ) THEN

!  Reorder A so that its entries are ordered by columns

          CALL LSQP_A_by_cols( n, m, A_ne, A_val, A_col, A_ptr,                &
                               Abycol_val, Abycol_row, Abycol_ptr )

!  Compute the length of the largest column as well as the average length

          max_len = MAXVAL( Abycol_ptr( 2 : ) - Abycol_ptr( : n ) )
          IF ( printi ) WRITE( control%out, 2000 ) max_len,                    &
            float( ( Abycol_ptr( n + 1 ) - 1 ) ) / float( n ),                 &
            max_col, COUNT( Abycol_ptr( 2 : ) - Abycol_ptr( : n )              &
             > max_col )
        ELSE
          max_len = 0
        END IF

!  Check that the largest column is not too long

        IF ( factor == 0 .AND. max_len > max_col .AND. m > max_sc ) THEN
          IF ( printi ) WRITE( control%out, 2030 ) max_col
          factor = 2
          DEALLOCATE( Abycol_val, Abycol_row, Abycol_ptr )
          GO TO 10
        END IF

!  Continue allocating the arrays for the analysis phase

        lk = max( A_ne + m, 2 * A_ne, control%valmin )

        reallocate = .TRUE.
        IF ( ALLOCATED( K%row ) ) THEN
          IF ( SIZE( K%row ) < lk ) THEN ; DEALLOCATE( K%row )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( K%row( lk ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%K%row' ; GO TO 900 ; END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( K%col ) ) THEN
          IF ( SIZE( K%col ) < lk ) THEN ; DEALLOCATE( K%col )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( K%col( lk ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%K%col' ; GO TO 900 ; END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( K%val ) ) THEN
          IF ( SIZE( K%val ) < lk ) THEN ; DEALLOCATE( K%val )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( K%val( lk ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%K%val' ; GO TO 900 ; END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( K_colptr ) ) THEN
          IF ( SIZE( K_colptr ) /= m + 1 ) THEN ; DEALLOCATE( K_colptr )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( K_colptr( m + 1 ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%K_colptr' ; GO TO 900
          END IF
        END IF

        ldiag_x = n
        reallocate = .TRUE.
        IF ( ALLOCATED( DIAG_X ) ) THEN
          IF ( SIZE( DIAG_X ) /= ldiag_x ) THEN ; DEALLOCATE( DIAG_X )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( DIAG_X( ldiag_x ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%DIAG' ; GO TO 900 ; END IF
        END IF

        ldiag_c_l = dims%c_l_start
        ldiag_c_u = dims%c_u_end
        reallocate = .TRUE.
        IF ( ALLOCATED( DIAG_C ) ) THEN
          IF ( LBOUND( DIAG_C, 1 ) /= ldiag_c_l .OR.                           &
               UBOUND( DIAG_C, 1 ) /= ldiag_c_u ) THEN ; DEALLOCATE( DIAG_C )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( DIAG_C( ldiag_c_l : ldiag_c_u ),                           &
                    STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%DIAG_C' ; GO TO 900
          END IF
        END IF

!  Permute A so that its entries appear by columns, with the entries in
!  each column sorted by increasing row number. Also form the data
!  structure required to hold K = A DIAG_X(inv) A(transpose) + DIAG_C(inv)

   20   CONTINUE

        IF ( m > 0 ) THEN

          CALL LPB_form_Schur_complement(                                      &
                  dims, n, m, A_ne, Abycol_val, Abycol_row, Abycol_ptr,        &
                  DIAG_X, SCALE_C, DIAG_C, lk, K%val, K%row,                   &
                  K_colptr, nnzk, ierr, IW( : n ), IW( n + 1 :  ), .FALSE.,    &
                  control%error, print_level )
!         inform%nfacts = inform%nfacts + 1 

!  Check for error returns

          IF ( ierr == 3 ) THEN

!  Insufficient space. Allocate more and retry

            DEALLOCATE( K%row, K%col, K%val )
            lk = 2 * lk
            ALLOCATE( K%row( lk ), K%col( lk ), K%val( lk ),    &
                      STAT = inform%alloc_status )
            IF ( inform%alloc_status /= 0 ) THEN 
              inform%bad_alloc = 'lpb: data%K' ; GO TO 900 ; END IF
  
            GO TO 20
          END IF
  
          IF ( ierr == 4 ) THEN

!  Insufficient space. Allocate more and retry

            DEALLOCATE( K%row, K%col, K%val )
            lk = nnzk
            ALLOCATE( K%row( lk ), K%col( lk ), K%val( lk ),    &
                      STAT = inform%alloc_status )
            IF ( inform%alloc_status /= 0 ) THEN 
              inform%bad_alloc = 'lpb: data%K' ; GO TO 900 ; END IF
            GO TO 20
          END IF

!  Record the column numbers of each entry in A A^T

          DO i = 1, m
            K%col( K_colptr( i ) : K_colptr( i + 1 ) - 1 ) = i
          END DO
        ELSE
          get_factors = .FALSE.
          nnzk = 0
        END IF

!  Now find a good ordering of the rows and columns of A A^T for
!  the sparse factorization. Set the dimensions of K

        K%n = m ; K%ne = nnzk

!  For the KKT matrix
!  ==================

      ELSE
      
!  Compute the space required

        lk = A_ne + n + 2 * dims%nc

!  Allocate the arrays for the analysis phase

        reallocate = .TRUE.
        IF ( ALLOCATED( K%row ) ) THEN
          IF ( SIZE( K%row ) < lk ) THEN ; DEALLOCATE( K%row )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( K%row( lk ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%K%row' ; GO TO 900 ; END IF
        END IF
   
        reallocate = .TRUE.
        IF ( ALLOCATED( K%col ) ) THEN
          IF ( SIZE( K%col ) < lk ) THEN ; DEALLOCATE( K%col )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( K%col( lk ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%K%col' ; GO TO 900 ; END IF
        END IF

        reallocate = .TRUE.
        IF ( ALLOCATED( K%val ) ) THEN
          IF ( SIZE( K%val ) < lk ) THEN ; DEALLOCATE( K%val )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( K%val( lk ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%K%val' ; GO TO 900 ; END IF
        END IF
        
        ldiag_x = 0
        reallocate = .TRUE.
        IF ( ALLOCATED( DIAG_X ) ) THEN
          IF ( SIZE( DIAG_X ) /= ldiag_x ) THEN ; DEALLOCATE( DIAG_X )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( DIAG_X( ldiag_x ), STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%DIAG' ; GO TO 900 ; END IF
        END IF

        ldiag_c_l = 0
        ldiag_c_u = 0
        reallocate = .TRUE.
        IF ( ALLOCATED( DIAG_C ) ) THEN
          IF ( LBOUND( DIAG_C, 1 ) /= ldiag_c_l .OR.                           &
               UBOUND( DIAG_C, 1 ) /= ldiag_c_u ) THEN ; DEALLOCATE( DIAG_C )
          ELSE ; reallocate = .FALSE.
          END IF
        END IF
        IF ( reallocate ) THEN 
          ALLOCATE( DIAG_C( ldiag_c_l : ldiag_c_u ),                           &
                    STAT = inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN 
            inform%bad_alloc = 'lpb: data%DIAG_C' ; GO TO 900
          END IF
        END IF

!  Set the coordinates and value of A in K

        DO i = 1, m
          y_ii = dims%y_s + i - 1
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            K%row( l ) = y_ii
          END DO
        END DO
        K%col( : A_ne ) = A_col( : A_ne ) 
        K%val( : A_ne ) = A_val( : A_ne )

!  Set the coodinates corresponding to the slack variables

        DO i = 1, dims%nc
          K%row( A_ne + i ) = dims%y_i + i - 1
          K%col( A_ne + i ) = n + i
          K%val( A_ne + i ) = - SCALE_C( dims%c_equality + i )
        END DO

!  Set the coordinates and values of M in K

        nnzk = A_ne + dims%nc
        SELECT CASE( precon )

!  * M is a diagonal matrix

        CASE DEFAULT
          DO i = dims%x_s, dims%c_e
            nnzk = nnzk + 1 ; K%row( nnzk ) = i ; K%col( nnzk ) = i
            K%val( nnzk ) = k_diag
          END DO
          nnzks = nnzk
          K%n = nnzks

!  * M is just the barrier terms

        CASE ( 2 )
          nnzks = nnzk
        END SELECT

!  Finally, include the co-ordinates of the barrier terms

        IF ( precon == 2 ) THEN
          DO i = dims%x_s, dims%c_e
            K%row( nnzks + i ) = i ; K%col( nnzks + i ) = i
          END DO
          K%ne = nnzks + dims%c_e
        ELSE
          K%ne = nnzks
        END IF

        K%n = dims%v_e

      END IF

      IF ( K%n > 0 .AND. get_factors ) THEN

!  Analyse the sparsity pattern of the preconditioner

        CALL SILS_analyse( K, FACTORS, CNTL, AINFO )

!  Record the storage requested

        inform%factorization_integer = AINFO%nirnec 
        inform%factorization_real = AINFO%nrlnec

!  Check for error returns

        inform%factorization_status = AINFO%flag
        IF ( AINFO%flag < 0 ) THEN
          IF ( printe ) WRITE( control%error, 2040 ) AINFO%flag 
          inform%status = - 7 ; RETURN
        ELSE IF ( AINFO%flag > 0 ) THEN 
          IF ( printt ) WRITE( control%out, 2050 ) AINFO%flag, 'SILS_analyse'
        END IF
   
        IF ( printt ) WRITE( control%out,                                      &
          "( ' real/integer space required for factors ', 2I10 )" )            &
            AINFO%nrladu, AINFO%niradu
      ELSE
        inform%factorization_integer = 0 ; inform%factorization_real = 0
      END IF

      RETURN

!  Unsuccessful returns

  900 CONTINUE
      inform%status = - 2
      WRITE( control%out, 2900 ) inform%bad_alloc, inform%alloc_status
      RETURN

!  Non-executable statements

 2000 FORMAT( '  maximum, average column lengths ', I7, 0P, F10.1, /,          &
              '  number of columns longer than   ', I7, ' is', I7 )
 2030 FORMAT( ' ** The maximum column length in A is larger than', /,          &
              '    max_col =', I7, ' - abandon the Schur-complement', /,       &
              '    factorization in favour of one of the augmented matrix', / )
 2040 FORMAT( '   **  Error return ', I6, ' from ', A15 ) 
 2050 FORMAT( '   **  Warning ', I6, ' from ', A15 ) 
 2710 FORMAT( '  Identity Hessian ' )
 2720 FORMAT( '  Barrier Hessian ' )
 2900 FORMAT( ' ** Message from -LPB_analyse-', /,                            &
              ' Allocation error, for ', A, /, ' status = ', I6 )

      END SUBROUTINE LPB_analyse

!-*-*-*-*-*-*-*-*-*-   L P B _ c o n d  S U B R O U T I N E  -*-*-*-*-*-*-*-

      SUBROUTINE LPB_cond( K_n, rank, FACTORS, out )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the smallest and largest eigenvalues of the block diagonal
!  part of the factors
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: K_n, rank, out
      TYPE ( SILS_factors ), INTENT( IN ) :: FACTORS

!  Local variables

      INTEGER :: i, nroots
      REAL ( KIND = wp ) :: root1, root2, dmax, dmin
      LOGICAL ::  twobytwo
      INTEGER :: P( K_n )
      REAL ( KIND = wp ) :: D( 2, K_n )

      IF ( k_n <= 0 ) RETURN
      CALL  SILS_enquire( FACTORS, PIVOTS = P, D = D )

      twobytwo = .FALSE.
      dmax = zero
      dmin = HUGE( one )
      DO i = 1, rank
        IF ( twobytwo ) THEN
          twobytwo = .FALSE.
          CYCLE
        END IF
        IF ( i <  rank ) THEN
          IF ( D( 2, i ) /= zero ) THEN
            twobytwo = .TRUE.
            CALL ROOTS_quadratic( D( 1, i ) * D( 1, i + 1 ) - D( 2, i ) ** 2,  &
              - D( 1, i ) - D( 1, i + 1 ), one, epsmch, nroots, root1, root2 ) 
            dmax = MAX( ABS( root1 ), ABS( root2 ), dmax )
            dmin = MIN( ABS( root1 ), ABS( root2 ), dmin )
          ELSE
            dmax = MAX( ABS( D( 1, i ) ), dmax )
            dmin = MIN( ABS( D( 1, i ) ), dmin )
          END IF
        ELSE
          dmax = MAX( ABS( D( 1, i ) ), dmax )
          dmin = MIN( ABS( D( 1, i ) ), dmin )
        END IF
      END DO

      IF ( K_n > rank ) dmax = HUGE( one )

      IF ( dmin == zero .OR. dmax == zero ) THEN
        WRITE( out, "( ' 1/ smallest,largest eigenvalues =',  2ES12.4 )" )     &
          dmin, dmax
      ELSE
        WRITE( out, "( ' smallest,largest eigenvalues =',  2ES12.4 )" )        &
          one / dmax, one / dmin
      END IF
      RETURN

!  End of subroutine LPB_cond

      END SUBROUTINE LPB_cond

!-*-*-*-   L P B _ o p t i m a l _ f o r _ S B Q P   S U B R O U T I N E  -*-*-*-

      SUBROUTINE LPB_optimal_for_SBLP( prob, control, inform, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute a global minimizer of a separable bound-constrained 
!  quadratic program
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LPB_control_type ), INTENT( IN ) :: control
      TYPE ( LPB_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: g, h, x_l, x_u
      LOGICAL :: stat_required
      REAL :: time, time_start

      CALL CPU_TIME( time_start )
      stat_required = PRESENT( B_stat )

!  Set information parameters

      inform%alloc_status = 0
      inform%iter = 0
      inform%factorization_status = 0
      inform%factorization_integer = 0
      inform%factorization_real = 0
      inform%nfacts = 0
      inform%nbacts = 0
      inform%nmods = 0
      inform%non_negligible_pivot = zero

!  Set initial timing breakdowns

      inform%time%total = 0.0     ; inform%time%phase1_total = 0.0    
      inform%time%analyse = 0.0   ; inform%time%phase1_analyse = 0.0  
      inform%time%factorize = 0.0 ; inform%time%phase1_factorize = 0.0
      inform%time%solve = 0.0     ; inform%time%phase1_solve = 0.0    
      inform%time%preprocess = 0.0 ; inform%time%find_dependent = 0.0

      inform%status = 0
      inform%obj = prob%f

!  Now consider the solution, one component at a time

      DO i = 1, prob%n
        x_l = prob%X_l( i ) ; x_u = prob%X_u( i )
        IF ( x_l > x_u ) THEN
          inform%feasible = .FALSE. ; inform%status = - 6 ; RETURN
        END IF
        g = prob%G( i ) ; h = prob%X( i )

        IF ( g > zero ) THEN
          prob%X( i ) = x_l

!  The objective is unbounded 

          IF ( x_l < - control%infinity ) THEN
            inform%status = - 11
            prob%Z( i ) = zero
            IF ( stat_required ) B_stat( i ) = 0

!  The minimizer occurs at the lower bound

          ELSE
            prob%Z( i ) = g + h * x_l
            IF ( stat_required ) B_stat( i ) = - 1
          END IF
        ELSE IF ( g < zero ) THEN
          prob%X( i ) = x_u

!  The objective is unbounded 

          IF ( x_u > control%infinity ) THEN
            inform%status = - 11
            prob%Z( i ) = zero
            IF ( stat_required ) B_stat( i ) = 0

!  The minimizer occurs at the upper bound

          ELSE
            prob%Z( i ) = g + h * x_u
            IF ( stat_required ) B_stat( i ) = 1
          END IF
        ELSE

!  The objective is constant along this component direction

          prob%Z( i ) = zero

!  Pick an arbitrary minimizer between the bounds

          IF ( stat_required ) B_stat( i ) = 0
          IF ( x_l >= - control%infinity .AND. x_u <=  control%infinity ) THEN
            prob%X( i ) = half * ( x_l + x_u )
          ELSE IF ( x_l >= - control%infinity ) THEN
            prob%X( i ) = x_l
          ELSE IF ( x_u <= control%infinity ) THEN
            prob%X( i ) = x_u
          ELSE
            prob%X( i ) = zero
          END IF
        END IF
        inform%obj = inform%obj + prob%X( i ) * g
      END DO
      IF ( inform%obj < control%obj_unbounded ) inform%status = - 11

      CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
      RETURN

!  End of subroutine LPB_optimal_for_SBLP

      END SUBROUTINE LPB_optimal_for_SBLP

!-*-*-*-*-*-*-*-   Q P B _ F O R M _ S _ C   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LPB_form_Schur_complement(                                    &
                          dims, n, m, Abycol_ne, Abycol_val, Abycol_row,       &
                          Abycol_ptr, DIAG_X, SCALE_C, DIAG_C, ls, S_val,      &
                          S_row, S_colptr, nes, ierr, col_count, row_count,    &
                          reals, control_error, control_print_level )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  This subroutine computes the Schur-complement matrix 
!    S = A DIAG_X(inv) A(trans) + SCALE_C * DIAG_C(inv) * SCALE_C
!  given the matrix A and vector of weights DIAG_X and DIAG_C

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!  m is the number of rows in matrix A.
!  n is the number of columns in matrix A, and the number of rows and
!    columns in matrix S.
!  Abycol_ne is the number of entries in matrix A. It is the length of 
!     arrays Abycol_val and Abycol_row.
!  Abycol_val holds the values of the entries in matrix A.
!  Abycol_row holds the row indices of the corresponding entries in A
!  Abycol_ptr points to the start of each column of A.
!  DIAG_X, DIAG_C are vectors of weights
!  ls is the length of arrays S and S_row.
!  S_val is set to the values of the entries in the lower triangle of S, 
!   ordered by columns
!  S_row is set to the row numbers of the entries in the lower triangle of S
!  S_colptr is set to point to the start of each column of S.
!  ierr is set by the routine to the code specifying the type of input
!       error:
!            = 0    no input errors;
!            = 1    n less than or equal to zero;
!            = 2    m less than or equal to zero;
!            = 3    ls less than Abycol_ne+m;
!            = 4    ls less than the number of entries in S (the
!                   number required is in nes);
!  nes is the actual number of entries in S
!  COL_count is the number of entries in each column of A
!  ROW_count is the number of entries in each row in A and, later,
!            the number of entries in each row of the lower triangle of S

!  Based on Iain Duff's routine MC35

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LSQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, ls, Abycol_ne
      INTEGER, INTENT( IN ) :: control_error, control_print_level
      INTEGER, INTENT( OUT ) :: nes, ierr
      LOGICAL, INTENT( IN ) :: reals
      INTEGER, INTENT( INOUT ), DIMENSION( Abycol_ne ) :: Abycol_row
      INTEGER, INTENT( INOUT ), DIMENSION( n + 1 ) :: Abycol_ptr
      INTEGER, INTENT( OUT ), DIMENSION( ls )  :: S_row
      INTEGER, INTENT( OUT ), DIMENSION( m + 1 )  :: S_colptr
      INTEGER, INTENT( OUT ), DIMENSION( n )  :: COL_count
      INTEGER, INTENT( OUT ), DIMENSION( m )  :: ROW_count
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n )  :: DIAG_X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_start : dims%c_u_end ) :: SCALE_C
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( dims%c_l_start : dims%c_u_end ) :: DIAG_C
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( Abycol_ne )  :: Abycol_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( ls )  :: S_val

!  Local variables

      INTEGER :: i, i1, i2, ii, ij, ipos, j, j1, j2, jj
      INTEGER :: icount, idist, k, k1, k2, kk
      REAL ( KIND = wp ) :: amult
      LOGICAL :: error

!  Check for input errors

      IF ( n <= 0 ) THEN
        ierr = 1
        IF ( control_error > 0 .AND. control_print_level >= 2 )                &
          WRITE( control_error, 2010 ) n
        GO TO 900
      END IF

      IF ( m <= 0 ) THEN
        ierr = 2
        IF ( control_error > 0 .AND. control_print_level >= 2 )                &
          WRITE( control_error, 2020 ) m
        GO TO 900
      END IF

      IF ( ls < Abycol_ne + m ) THEN
        ierr = 3
        IF ( control_error > 0 .AND. control_print_level >= 2 )                &
          WRITE( control_error, 2030 ) ls, Abycol_ne + m
        GO TO 900
      END IF

      ierr = 0

!  Set up column counts

!     COL_count( : n ) =                                                  &
!       Abycol_ptr( 2 : n + 1 ) - Abycol_ptr( : n )
      DO i = 1, n
        COL_count( i ) = Abycol_ptr( i + 1 ) - Abycol_ptr( i )
      END DO

!  Set up row counts

      ROW_count( : m ) = 0
      DO i = 1, Abycol_ne
        j = Abycol_row( i )
        ROW_count( j ) = ROW_count( j ) + 1
      END DO

!  Set up column pointer

      S_colptr( 1 ) = ls - Abycol_ne + 1
      DO i = 1, m
        S_colptr( i + 1 ) = S_colptr( i ) + ROW_count( i )
      END DO

!  Generate structure of A by columns in S/S_row

      j1 = 1
      DO i = 1, n
        j2 = j1 + COL_count( i ) - 1
        DO jj = j1, j2
          j = Abycol_row( jj )
          ipos = S_colptr( j )
          S_val( ipos ) = Abycol_val( jj )
          S_row( ipos ) = i
          S_colptr( j ) = ipos + 1
        END DO
        j1 = j2 + 1
      END DO

!  Regenerate structure of A by columns but in row order within each column.
!  Order the row indices in Abycol_row using S_val/S_row to create 
!  Abycol_val/Abycol_row

      i1 = ls - Abycol_ne + 1
      DO j = 1, m
        ij = m - j + 1

!  Reset S_colptr after the last loop altered it

        IF ( ij > 1 ) S_colptr( ij ) = S_colptr( ij - 1 )
        i2 = i1 + ROW_count( j ) - 1
        DO ii = i1, i2
          i = S_row( ii )
          ipos = Abycol_ptr( i )
          Abycol_row( ipos ) = j
          Abycol_val( ipos ) = S_val( ii )
          Abycol_ptr( i ) = ipos + 1
        END DO
        i1 = i2 + 1
      END DO
      S_colptr( 1 ) = 1 + ls - Abycol_ne

!  Reset Abycol_ptr since it has been altered

      DO j = n, 2, - 1
        Abycol_ptr( j ) = Abycol_ptr( j - 1 )
      END DO
      Abycol_ptr( 1 ) = 1

!  Now assemble S

      error = .FALSE.
      nes = 0
      icount = 0
      idist = 0
      k1 = S_colptr( 1 )

!  Form row i of S from a linear combination of rows of A

      DO i = 1, m

!  Leave room for possible diagonal if row i of A is empty

        IF ( ROW_count( i ) == 0 ) THEN
          IF ( i >= dims%c_l_start ) THEN
            IF ( error ) THEN
              icount = icount + 1
            ELSE
              ROW_count( i ) = 1
              nes = nes + 1
              IF ( nes <= ls ) THEN
                S_row( nes ) = i
                IF ( reals ) S_val( nes ) = zero
              ELSE
                error = .TRUE.
                icount = 1
              END IF
            END IF
          END IF
          CYCLE
        END IF

!  row i of A is non-empty

        k2 = k1 + ROW_count( i ) - 1
        ROW_count( i ) = 0

!  Scan row i of A(trans) to find which rows of A contribute to row i of S

        DO kk = k1, k2
          k = S_row( kk )
          IF ( reals ) amult = S_val( kk ) / DIAG_X( k )
          j1 = Abycol_ptr( k )
          j2 = j1 + COL_count( k ) - 1

!  The column indices for the entries in each row are in order

          Abycol_ptr( k ) = j1 + 1
          COL_count( k ) = j2 - j1

!  Scan row k of A and add multiple of it to row i of S

          DO jj = j1, j2
            j = Abycol_row( jj )
            IF ( error ) THEN

!  Only count the size of arrays needed because of the error

              IF ( S_colptr( j ) < 0 ) CYCLE
              ROW_count( i ) = ROW_count( i ) + 1
              ipos = ROW_count( i )
              S_colptr( j ) = - ROW_count( i )
              IF ( ipos + idist > kk ) THEN
               icount = icount + 1
               idist = idist - 1
              END IF
  
              S_row( ipos ) = j
              CYCLE

            END IF

            IF ( S_colptr( j ) >= 0 ) THEN

!  First contribution to row i of S

              ROW_count( i ) = ROW_count( i ) + 1

!  S_colptr is used to obtain offset for column positions in present row

              S_colptr( j ) = - ROW_count( i )
              ipos = nes + ROW_count( i )
              IF ( ipos > kk ) THEN

!  Insufficient room in S to continue calculation, so only calculate
!  the size required

                error = .TRUE.

!  Overwrite the beginning of S_row since it is not now needed.

                ij = ROW_count( i )
                S_row( 1 : ij - 1 ) = S_row( nes + 1 : nes + ij - 1 )
                nes = 0

!  idist holds the difference between the current position in S_row
!  and the position that it would be if S_row was long enough

                idist = kk - ij

!  icount is the amount by which S_row is too short

                icount = 1
                ipos = ij
                S_row( ipos ) = j
                CYCLE

              END IF

              S_row( ipos ) = j
              IF ( reals ) S_val( ipos ) = Abycol_val( jj ) * amult
              CYCLE

            END IF
            ipos = nes - S_colptr( j )
            IF ( reals ) S_val( ipos ) = S_val( ipos ) + Abycol_val( jj) * amult
          END DO
        END DO

!  Reset S_colptr

        IF ( error ) nes = 0
        j1 = nes + 1
        nes = nes + ROW_count( i )
        S_colptr( S_row( j1 : nes ) ) = 0

        IF ( error ) THEN
         idist = idist + ROW_count( i )
         nes = 0
        END IF

        k1 = k2 + 1
      END DO

!  Set pointer arrays for the start of each column of A and S

      IF ( .NOT. error ) THEN
        S_colptr( 1 ) = 1
        DO i = 1, m
          S_colptr( i + 1 ) = S_colptr( i ) + ROW_count( i )
        END DO
      END IF

      DO j = n, 2, - 1
        Abycol_ptr( j ) = Abycol_ptr( j - 1 )
      END DO
      Abycol_ptr( 1 ) = 1

      IF ( icount > 0 ) THEN
        nes = icount + ls
        ierr = 4
        IF ( control_error > 0 .AND. control_print_level >= 2 )                &
          WRITE( control_error, 2040 ) ls, nes
      END IF

!  Find the diagonal entries in columns dims%c_l_start to dims%c_u_end

      IF ( reals ) THEN
        DO j = dims%c_l_start, dims%c_u_end
          i = S_colptr( j )
          IF ( j == S_row( i ) ) THEN
            S_val( i ) = S_val( i ) +                                          &
                           ( SCALE_C( j ) / DIAG_C( j ) ) * SCALE_C( j )
          ELSE
            WRITE( 6, "( ' j = ', i6, ' i = ', /, ( 10i6 ) )" ) &
              j, S_row( S_colptr( j ) : S_colptr( j + 1 ) - 1 )
            WRITE( 6, "( ' This should be impossible: '/,                      &
           &             ' please report to n.gould@rl.ac.uk' )" )
            STOP
          END IF
        END DO
      END IF

  900 CONTINUE
      RETURN

!  Non-executable statements

 2010 FORMAT( ' ** Error return from LPB_form_Schur_complement', /,            &
              '    Value of n (number of rows of A) is set to ', I10, /,       &
              '    but n must be at least 1 ' )
 2020 FORMAT( ' ** Error return from LPB_form_Schur_complement', /,            &
              '    Value of m (number of columns of A) is set to ', I10, /,    &
              '    but m must be at least 1 ' )
 2030 FORMAT( ' ** Error return from LPB_form_Schur_complement', /,            &
              '    Value of ls = ', I10, ' too small. ', /,                    &
              '    ls must be at least', I10 )
 2040 FORMAT( ' ** Error return from LPB_form_Schur_complement', /,            &
              '    Arrays S_row and S are too small ', /, '    ls is ', I10, /,&
              '    but must be at least ', I10 )

!  End of LPB_form_Schur_complement

      END SUBROUTINE LPB_form_Schur_complement

!  End of module LPB

   END MODULE GALAHAD_LPB_double
