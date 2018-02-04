! THIS VERSION: GALAHAD 2.4 - 19/05/2010 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*-*-  G A L A H A D _ P Q P   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started September 14th 2004
!   originally released GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_PQP_double

!      ---------------------------------------------------------------------
!     |                                                                     |
!     | Given the solution to QP(theta) when theta = 0, solve the           |
!     | parametric quadratic program                                        |
!     |                                                                     |
!     | QP(theta):  minimize   1/2 x(T) H x + g(T) x + f + theta dg(T) x    |
!     |             subject to c_l + theta dc_l <= A x <= c_u + theta dc_u  |
!     |             and        x_l + theta dx_l <=  x  <= x_u + theta dx_u  |
!     |                                                                     |
!     | for all 0 <= theta <= theta_max                                     |
!     |                                                                     |
!      ---------------------------------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_SYMBOLS
      USE GALAHAD_QPT_double
      USE GALAHAD_RAND_double
      USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                      &
         SORT_heapsort_smallest, SORT_inplace_permute, SORT_inverse_permute
      USE GALAHAD_SLS_double
      USE GALAHAD_SCU_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPP_double, PQP_dims_type => QPP_dims_type
      USE GALAHAD_QPA_double, PQP_time_type => QPA_time_type,                  &
                              PQP_control_type => QPA_control_type,            &
                              PQP_inform_type => QPA_inform_type,              &
                              PQP_data_type => QPA_data_type
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: PQP_initialize, PQP_read_specfile, PQP_solve,                  &
                PQP_terminate, QPT_problem_type, SMT_type,                     &
                PQP_time_type, PQP_control_type,                               &
                PQP_inform_type, PQP_data_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------------
!   D e r i v e d   T y p e s
!----------------------------

      TYPE, PUBLIC :: PQP_interval_type
        REAL ( KIND = wp ) :: theta_l, theta_u, f, g, h
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) ::                     &
          X, Y, Z, DX, DY, DZ
      END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( one )

   CONTAINS

!-*-*-*-*-*-   Q P A _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE PQP_initialize( interval, data, control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for PQP. This routine should be called before
!  PQP_solve
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
!   max_sc. The maximum permitted size of the Schur complement before a 
!    refactorization is performed.
!
!   indmin. An initial guess as to the integer workspace required by SLS
!
!   valmin. An initial guess as to the real workspace required by SLS
! 
!   itref_max. The maximum number of iterative refinements allowed
!
!   infeas_check_interval. The infeasibility will be checked for improvement
!    every infeas_check_interval iterations (see infeas_g_improved_by_factor 
!    and infeas_b_improved_by_factor below)
!
!   cg_maxit. The maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the number of degrees of freedom in the
!     system + 1
!
!   precon. The preconditioner to be used for the CG is defined by precon. 
!    Possible values are
!
!     <0  full factorization as direct method
!      0  automatic 
!      1  full factorization
!      2  replace Hessian block by the identity
!      3  replace Hessian block by those entries lying within a band
!      4  replace Hessian block for free variables by the identity
!      5  replace Hessian block for free variables by those entries lying 
!         within a band
!
!   nsemib. The semi-bandwidth of a band preconditioner, if appropriate
!
!   full_max_fill. If the ratio of the number of nonzeros in the factors
!    of the reference matrix to the number of nonzeros in the matrix
!    itself exceeds full_max_fill, and the preconditioner is being selected
!    automatically (precon = 0), a banded approximation will be used instead
!
!   deletion_strategy. The constraint deletion strategy to be used.
!    Possible values are
!
!      0  most violated of all
!      1  LIFO (last in, first out)
!      k  LIFO(k) most violated of the last k in LIFO
!
!   restore_problem. Indicates whether and how much of the input problem
!    should be restored on output. Possible values are
!
!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters
!
!   monitor_residuals. The frequency at which residuals will be monitored
!
!   cold_start. Indicates whether a cold or warm start should be made.
!    Possible values are 
!
!     0 warm start - the values set in C_stat and B_stat indicate which 
!       constraints will be included in the initial working set. 
!     1 cold start from the value set in X; constraints active
!       at X will determine the initial working set.
!     2 cold start with no active constraints
!     3 cold start with only equality constraints active
!     4 cold start with as many active constraints as possible
!
!  REAL control parameters:
!
!   infinity. Any bound larger than infinity in modulus will be regarded as 
!    infinite 
!   
!   feas_tol. Any constraint violated by less than feas_tol will be considered
!    to be satisfied
!
!   obj_unbounded. If the objective function value is smaller than
!    obj_unbounded, it will be flagged as unbounded from below
!
!   increase_rho_g_factor. If the problem is currently infeasible and 
!    solve_qp (see below) is .TRUE., the current penalty parameter for the
!    general constraints will be increased by increase_rho_g_factor when needed
!
!   increase_rho_b_factor. If the problem is currently infeasible and 
!    solve_qp or solve_within_bounds (see below) is .TRUE., the current 
!    penalty parameter for the simple bound constraints will be increased by 
!    increase_rho_b_factor when needed
!
!   infeas_g_improved_by_factor. If the infeasibility of the general constraints
!    has not dropped by a factor of infeas_g_improved_by_factor over the 
!    previous infeas_check_interval iterations, the current corresponding 
!    penalty parameter will be increased
!
!   infeas_b_improved_by_factor. If the infeasibility of the simple bounds
!    has not dropped by a factor of infeas_b_improved_by_factor over the 
!    previous infeas_check_interval iterations, the current corresponding 
!    penalty parameter will be increased
!
!   pivot_tol. The threshold pivot used by the matrix factorization.
!    See the documentation for SLS for details
!
!   pivot_tol_for_dependencies. The threshold pivot used by the matrix 
!    factorization when attempting to detect linearly dependent constraints.
!    See the documentation for SLS for details
!
!   zero_pivot. Any pivots smaller than zero_pivot in absolute value will 
!    be regarded to be zero when attempting to detect linearly dependent 
!    constraints
!
!   multiplier_tol. Any dual variable or Lagrange multiplier which is less than 
!    multiplier_tol outside its optimal interval will be regarded
!    as being acceptable when checking for optimality.
!
!   inner_stop_relative and inner_stop_absolute. The search direction is
!    considered as an acceptable approximation to the minimizer of the
!    model if the gradient of the model in the preconditioning(inverse) 
!    norm is less than 
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )
!
!  LOGICAL control parameters:
!
!   treat_zero_bounds_as_general. If true, any problem bound with the value
!    zero will be treated as if it were a general value
!
!   solve_qp. If .TRUE., the value of prob%rho_g and prob%rho_b will be 
!    increased as many times as are needed to ensure that the output 
!    solution is feasible
!
!   solve_within_bounds. If .TRUE., the value of prob%rho_b will be 
!    increased as many times as are needed to ensure that the output 
!    solution is feasible with respect to the simple bounds
!
!   randomize. If .TRUE., the constraint bounds will be perturbed by
!    small random quantities during the first stage of the solution
!    process. Any randomization will ultimately be removed. Randomization
!    helps when solving degenerate problems
!
!   array_syntax_worse_than_do_loop. If array_syntax_worse_than_do_loop is
!    true, f77-style do loops will be used rather than 
!    f90-style array syntax for vector operations
!
!   each_interval. If each_interval is true, control will pass back to the
!    user after the solution has been computed between each pair of breakpoints
!    in the parametric interval (see argument action for pqp_solve. 
!    Otherwise, only the solution at omega_max will be computed.

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( PQP_interval_type ), INTENT( INOUT ) :: interval
      TYPE ( PQP_data_type ), INTENT( OUT ) :: data
      TYPE ( PQP_control_type ), INTENT( OUT ) :: control        

!  Initalize random number seed

      CALL RAND_initialize( data%seed )

!  Integer parameters

      control%error  = 6
      control%out  = 6
      control%print_level = 0
      control%maxit  = 1000
      control%start_print = - 1
      control%stop_print = - 1
      control%factor = 0
      control%max_col = 35
      control%max_sc = 75
      control%indmin = 1000
      control%valmin = 1000
      control%itref_max = 1
      control%infeas_check_interval = 100
      control%cg_maxit = - 1
      control%precon = 0
      control%nsemib = 5
      control%full_max_fill = 10
      control%deletion_strategy = 0
      control%restore_problem = 2
      control%monitor_residuals = 1
      control%cold_start = 3

!  Real parameters

      control%infinity = ten ** 19
      control%feas_tol = epsmch ** 0.75
      control%obj_unbounded = - epsmch ** ( - 2.0 )
!     control%obj_unbounded = - ten ** 30
      control%increase_rho_g_factor = two
      control%increase_rho_b_factor = two
!     control%increase_rho_g_factor = ten
!     control%increase_rho_b_factor = ten
      control%infeas_g_improved_by_factor = 0.75_wp
      control%infeas_b_improved_by_factor = 0.75_wp
      control%pivot_tol = point1 * epsmch ** 0.5
      control%pivot_tol_for_dependencies = half
      control%zero_pivot = epsmch ** 0.75
!     control%zero_pivot = epsmch ** 0.25
!     control%zero_pivot = epsmch ** 0.5
!     control%multiplier_tol = epsmch ** 0.25
      control%multiplier_tol = epsmch ** 0.5
      control%inner_stop_relative = zero
      control%inner_stop_absolute = SQRT( EPSILON( one ) )
 
!  Logical parameters

      control%treat_zero_bounds_as_general = .FALSE.
      control%solve_qp = .FALSE.
      control%solve_within_bounds = .FALSE.
      control%randomize = .TRUE.
      control%array_syntax_worse_than_do_loop = .FALSE.
      control%each_interval = .FALSE.

      RETURN  

!  End of PQP_initialize

      END SUBROUTINE PQP_initialize

!-*-*-*-*-   Q P A _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE PQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by PQP_initialize could (roughly) 
!  have been set as:

!  BEGIN PQP SPECIFICATIONS (DEFAULT)
!    error-printout-device                             6
!    printout-device                                   6
!    print-level                                       0
!    maximum-number-of-iterations                      1000
!    start-print                                       -1
!    stop-print                                        -1
!    factorization-used                                0
!    maximum-column-nonzeros-in-schur-complement       35
!    maximum-dimension-of-schur-complement             75
!    initial-integer-workspace                         1000
!    initial-real-workspace                            1000
!    maximum-refinements                               1
!    maximum-infeasible-iterations-before-rho-increase 100
!    maximum-number-of-cg-iterations                   -1
!    preconditioner-used                               0
!    semi-bandwidth-for-band-preconditioner            5
!    full-max-fill-ratio                               10
!    deletion-strategy                                 0
!    restore-problem-on-output                         2
!    residual-monitor-interval                         1
!    cold-start-strategy                               3
!    infinity-value                                    1.0D+19
!    feasibility-tolerance                             1.0D-12
!    minimum-objective-before-unbounded                -1.0D+32
!    increase-rho-g-factor                             2.0
!    increase-rho-b-factor                             2.0
!    infeasible-g-required-improvement-factor          0.75
!    infeasible-b-required-improvement-factor          0.75
!    pivot-tolerance-used                              1.0D-8
!    pivot-tolerance-used-for-dependencies             0.5
!    zero-pivot-tolerance                              1.0D-12
!    multiplier-tolerance                              1.0D-8
!    inner-iteration-relative-accuracy-required        0.0
!    inner-iteration-absolute-accuracy-required        1.0E-8
!    treat-zero-bounds-as-general                      F
!    solve-qp                                          F
!    solve-within-bounds                               F
!    temporarily-perturb-constraint-bounds             T
!    array-syntax-worse-than-do-loop                   F
!    find-solution-over-each-interval                  F
!  END PQP SPECIFICATIONS

!  Dummy arguments

      TYPE ( PQP_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 40
!     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'PQP        '
      CHARACTER( LEN = 16 ), PARAMETER :: specname = 'QPA             '
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
      spec(  9 )%keyword = 'maximum-dimension-of-schur-complement'
      spec( 10 )%keyword = 'initial-integer-workspace'
      spec( 11 )%keyword = 'initial-real-workspace'
      spec( 12 )%keyword = 'maximum-refinements'
      spec( 13 )%keyword = 'maximum-infeasible-iterations-before-rho-increase'
      spec( 14 )%keyword = 'maximum-number-of-cg-iterations'
      spec( 15 )%keyword = 'preconditioner-used'
      spec( 16 )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( 17 )%keyword = 'full-max-fill-ratio'
      spec( 18 )%keyword = 'deletion-strategy'
      spec( 19 )%keyword = 'restore-problem-on-output'
      spec( 20 )%keyword = 'residual-monitor-interval'
      spec( 21 )%keyword = 'cold-start-strategy'

!  Real key-words

      spec( 22 )%keyword = 'infinity-value'
      spec( 23 )%keyword = 'feasibility-tolerance'
      spec( 24 )%keyword = 'minimum-objective-before-unbounded'
      spec( 25 )%keyword = 'increase-rho-g-factor'
      spec( 26 )%keyword = 'increase-rho-b-factor'
      spec( 27 )%keyword = 'infeasible-g-required-improvement-factor'
      spec( 28 )%keyword = 'infeasible-b-required-improvement-factor'
      spec( 29 )%keyword = 'pivot-tolerance-used'
      spec( 30 )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( 31 )%keyword = 'zero-pivot-tolerance'
      spec( 32 )%keyword = 'inner-iteration-relative-accuracy-required'
      spec( 33 )%keyword = 'inner-iteration-absolute-accuracy-required'
      spec( 34 )%keyword = 'multiplier-tolerance'

!  Logical key-words

      spec( 35 )%keyword = 'treat-zero-bounds-as-general'
      spec( 36 )%keyword = 'solve-qp'
      spec( 37 )%keyword = 'solve-within-bounds'
      spec( 38 )%keyword = 'temporarily-perturb-constraint-bounds'
      spec( 39 )%keyword = 'array-syntax-worse-than-do-loop'
      spec( 40 )%keyword = 'find-solution-over-each-interval'

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
      CALL SPECFILE_assign_integer( spec( 9 ), control%max_sc,                 &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 10 ), control%indmin,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 11 ), control%valmin,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 12 ), control%itref_max,             &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 13 ), control%infeas_check_interval, &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 14 ), control%cg_maxit,              &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 15 ), control%precon,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 16 ), control%nsemib,                &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 17 ), control%full_max_fill,         &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 18 ), control%deletion_strategy,     &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 19 ), control%restore_problem,       &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 20 ), control%monitor_residuals,     &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 21 ), control%cold_start,            &
                                    control%error )

!  Set real values


      CALL SPECFILE_assign_real( spec( 22 ), control%infinity,                 &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 23 ), control%feas_tol,                 &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 24 ), control%obj_unbounded,            &
                                 control%error )


      CALL SPECFILE_assign_real( spec( 25 ), control%increase_rho_g_factor,    &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 26 ), control%increase_rho_g_factor,    &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 27 ),                                   &
                                 control%infeas_g_improved_by_factor,          &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 28 ),                                   &
                                 control%infeas_b_improved_by_factor,          &
                                 control%error )     


      CALL SPECFILE_assign_real( spec( 29 ), control%pivot_tol,                &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 30 ),                                   &
                                 control%pivot_tol_for_dependencies,           &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 31 ), control%zero_pivot,               &
                                 control%error )

      CALL SPECFILE_assign_real( spec( 32 ), control%inner_stop_relative,      &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 33 ), control%inner_stop_absolute,      &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 34 ), control%multiplier_tol,           &
                                 control%error )

!  Set logical values

      CALL SPECFILE_assign_logical( spec( 35 ),                                &
                                    control%treat_zero_bounds_as_general,      &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 36 ), control%solve_qp,              &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 37 ), control%solve_within_bounds,   &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 38 ), control%randomize,             &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 39 ),                                &
                                    control%array_syntax_worse_than_do_loop,   &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 40 ), control%each_interval,         &
                                    control%error )

      RETURN

      END SUBROUTINE PQP_read_specfile

!-*-*-*-*-*-*-**-*-   P Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE PQP_solve( action, prob, interval, C_stat, B_stat,           &
                            data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!

!  Solve the quadratic program
!
!      QP(theta):  minimize   1/2 x(T) H x + g(T) x + f + theta dg(T) x 
!                  subject to c_l + theta dc_l <= A x <= c_u + theta dc_u
!                  and        x_l + theta dx_l <=  x  <= x_u + theta dx_u
!                                                               
!      for all 0 <= theta <= theta_max
!
!  where x is a vector of n components ( x_1, .... , x_n ), f is a constant
!  g and dg are an n-vectors, H is a symmetric matrix, A is an m by n matrix, 
!  and any of the bounds c_l, c_u, x_l, x_u may be infinite, using an active
!  set method. The subroutine is particularly appropriate when A and H are 
!  sparse, and when we do not anticipate a large number of active set
!  changes prior to optimality
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  action is a CHARACTER variable of length 20, which should be set
!   to the string "start" on initial entry. On exit, if action contains
!   the string "re-enter", the subroutine should be re-entered with
!   all data preserved since the last exit. This variable allows the
!   user to examine the status of the solution in the current 
!   (sub-)interval, as indicated in the varible interval (see below).
!   If action contains the string "end", the final interval has been
!   processed, and no further action is necessary.
!
!  prob is a structure of type QPT_problem_type, whose components hold 
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to 
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to PQP_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!                 
!   %H_* is used to hold the LOWER TRIANGULAR part of H.
!   Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %H%val( : )  the values of the components of H
!       %H%row( : )  the row indices of the components of H
!       %H%col( : )  the column indices of the components of H
!       %H%ne        the number of nonzeros used to store 
!                    the LOWER TRIANGULAR part of H
!
!       In addition, the array
!
!       %H%ptr( : )   must be of length %n + 1 
!
!       but need not be set
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %H%val( : )  the values of the components of H, stored row by row
!       %H%col( : )  the column indices of the components of H
!       %H%ptr( : )  pointers to the start of each row, and past the end of
!                    the last row
!       %H%ne    = - 1
!
!       In addition, the array
!
!       %H%row( : )   must be of length >= 0
!
!       but need not be set
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       %H%val( : )  the values of the components of H, stored row by row,
!                    with each the entries in each row in order of 
!                    increasing column indicies.
!       %H%ne    = - 2
!
!       In addition, the arrays
!
!       %H%row( : )   must be of length >= 0
!       %H%col( : )   must be of length %n * ( %n + 1 ) / 2
!       %H%ptr( : )   must be of length %n + 1
!
!       but need not be set
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output %H%row will contain
!    the row numbers corresponding to the values in %H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
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
!   %rho_g is a REAL variable, which must be set by the user to the 
!    required value of the penalty parameter for the general constraints
!
!   %rho_b is a REAL variable, which must be set by the user to the 
!    required value of the penalty parameter for the simple bound constraints
!
!   %A_* is used to hold the matrix A. Three storage formats
!    are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       %A%val( : )   the values of the components of A
!       %A%row( : )   the row indices of the components of A
!       %A%col( : )   the column indices of the components of A
!       %A%ne         the number of nonzeros used to store A
!
!       In addition, the array
!
!       %A%ptr( : )   must be of length %m + 1 
!
!       but need not be set
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       %A%val( : )   the values of the components of A, stored row by row
!       %A%col( : )   the column indices of the components of A
!       %A%ptr( : )   pointers to the start of each row, and past the end of
!                     the last row
!       %A%ne    = -1
!
!       In addition, the array
!
!       %A%row( : )   must be of length >= 0
!
!       but need not be set
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       %A%val( : )   the values of the components of A, stored row by row,
!                     with each the entries in each row in order of 
!                     increasing column indicies.
!       %A%ne    = -2
!
!       In addition, the arrays
!
!       %A%row( : )   must be of length >= 0
!       %A%col( : )   must be of length %n * %m
!       %A%ptr( : )   must be of length %m + 1
!
!       but need not be set
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output %A%row will contain
!    the row numbers corresponding to the values in %A%val, and thus in this
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
!  interval is a structure of type PQP_interval_type, whose components hold 
!   information about the solution within the current sub-interval.
!   The following components will have been set:
!
!   %......
!
!  C_stat is a INTEGER array of length m, which may be set by the user
!   on entry to PQP_solve to indicate which of the constraints are to
!   be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; C_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   C_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   C_stat( i ) < 0, the i-th constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to PQP_solve to indicate which of the simple bound constraints 
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; B_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   B_stat( i ) < 0, the i-th bound constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  data is a structure of type PQP_data_type which holds private internal 
!   data
!
!  control is a structure of type PQP_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to PQP_initialize. 
!   See PQP_initialize for details
!
!  inform is a structure of type PQP_inform_type that provides 
!    information on exit from PQP_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!   - 1 one of the restrictions 
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%ne  >=  -2
!          prob%H%ne  >=  -2
!       has been violated.
!
!    -2 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 One of more of the components A/H_val,  A/H_row,  A/H_col is not
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
!    -8 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of 
!       a badly scaled problem.
!
!    -9 The objective function appears to be unbounded from below on the
!       feasible set.
!
!  On exit from PQP_solve, other components of inform give the 
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
!     nmods  = The total number of factorizations which were modified to 
!       ensure that the matrix was an appropriate preconditioner. 
!     factorization_status = the return status from the matrix factorization
!       package.   
!     obj = the value of the objective function at the best estimate of the 
!       solution determined by PQP_solve.
!     infeas_g = the 1-norm of the infeasibility of the general constraints
!     infeas_b = the 1-norm of the infeasibility of the simple bound constraints
!     merit = obj + rho_g * infeas_g + rho_b * infeas_b
!     time%total = the total time spent in the package.
!     time%preprocess = the time spent preprocessing the problem.
!     time%analyse = the time spent analysing the required matrices prior to
!       factorization.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      CHARACTER ( LEN = 20 ), INTENT( INOUT ) :: action
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( PQP_interval_type ), INTENT( INOUT ) :: interval
      INTEGER, INTENT( INOUT ), DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( prob%n ) :: B_stat
      TYPE ( PQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( PQP_control_type ), INTENT( INOUT ) :: control
!     TYPE ( PQP_control_type ), INTENT( IN ) :: control
      TYPE ( PQP_inform_type ), INTENT( OUT ) :: inform

!  Local variables

      INTEGER :: i, ii, l, a_ne, h_ne, lbd, m_link, k_n_max, lbreak, n_pcg
      INTEGER :: hd_start, hd_end, hnd_start, hnd_end, type
      INTEGER :: n_depen
!     INTEGER, DIMENSION( 1 ) :: initial_seed
      REAL ( KIND = wp ) :: a_x, a_norms
      REAL :: time, time_start, time_inner_start, dum
      LOGICAL :: reallocate, printe, printi, printt
      CHARACTER ( LEN = 30 ) :: bad_alloc

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering PQP_solve ' )" )

!  Branch to the appropriate point on initial and re-entry

      IF ( action( 1: 5 ) == "start" ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 5 )                  &
          WRITE( control%out, "( ' initial entry ' )" )
      ELSE IF ( action( 1: 5 ) == "re-enter" ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 5 )                  &
          WRITE( control%out, "( ' re-entry ' )" )
        GO TO 200
      ELSE
        RETURN
      END IF

!  Initialize time

      CALL CPU_TIME( time_start )

!  Set initial timing breakdowns

      inform%time%total = 0.0
      inform%time%analyse = 0.0
      inform%time%factorize = 0.0
      inform%time%solve = 0.0
      inform%time%preprocess = 0.0

!  Initialize counts

      inform%major_iter = 0
      inform%iter = 0 ; inform%cg_iter = 0 ; inform%nfacts = 0
      inform%nmods = 0 ; inform%alloc_status = 0
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%factorization_status = 0
      inform%obj = prob%f 

!  Reduce print level for QP phase

      control%print_level = control%print_level - 1

!  Basic single line of output per iteration

      printe = control%error > 0 .AND. control%print_level >= 1 
      printi = control%out > 0 .AND. control%print_level >= 1 
      printt = control%out > 0 .AND. control%print_level >= 2

!  Ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           prob%A%ne < - 2 .OR. prob%H%ne < - 2 ) THEN
        inform%status = - 1
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) inform%status 
        CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
        GO TO 800
      END IF 

!  ===========================
!  Preprocess the problem data
!  ===========================

      prob%Y = zero ; prob%Z = zero
      data%new_problem_structure = prob%new_problem_structure
      IF ( prob%new_problem_structure ) THEN
        CALL QPP_initialize( data%QPP_map, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general
        IF ( control%randomize )                                               &
          data%QPP_control%treat_zero_bounds_as_general = .TRUE.

!  Store the problem dimensions

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          a_ne = prob%A%ne 
        END IF

        IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          h_ne = prob%H%ne 
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions before preprocessing: ', /,          &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prob%n, prob%m, a_ne, h_ne

!  Perform the preprocessing

        CALL CPU_TIME( time ) 
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                      &
                           data%QPP_inform, data%dims, prob,                   &
                           .FALSE., .FALSE., .FALSE., parametric = .TRUE. )
        CALL CPU_TIME( dum ) ; dum = dum - time
        inform%time%preprocess = inform%time%preprocess + dum
  
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

        IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          h_ne = prob%H%ne 
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions after preprocessing: ', /,           &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prob%n, prob%m, a_ne, h_ne

        prob%new_problem_structure = .FALSE.

!  Recover the problem dimensions after preprocessing

      ELSE
        CALL CPU_TIME( time ) 
        CALL QPP_apply( data%QPP_map, data%QPP_inform, prob,                   &
                        get_all_parametric = .TRUE. )
        CALL CPU_TIME( dum ) ; dum = dum - time  
        inform%time%preprocess = inform%time%preprocess + dum  
   
!  Test for satisfactory termination  
  
        IF ( data%QPP_inform%status /= 0 ) THEN  
          inform%status = data%QPP_inform%status  
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( ' status ', I3, ' after QPP_apply ')" )     &
             data%QPP_inform%status  
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status   
          CALL CPU_TIME( time ) ; inform%time%total = time - time_start   
          GO TO 800   
        END IF   
      END IF

!  Permute initial working sets if provided

      IF ( control%cold_start == 0 ) THEN
        CALL SORT_inplace_permute( data%QPP_map%m, data%QPP_map%c_map,         &
                                   IX = C_stat( : data%QPP_map%m ) )
        CALL SORT_inplace_permute( data%QPP_map%n, data%QPP_map%x_map,         &
                                   IX = B_stat( : data%QPP_map%n ) )
      END IF

!  ===========================================================================
!  Check to see if the constraints in the working set are linearly independent
!  ===========================================================================

!  Allocate workspace arrays

      lbreak = prob%m + data%dims%c_l_end - data%dims%c_u_start +              &
                prob%n - data%dims%x_free + data%dims%x_l_end -                &
                data%dims%x_u_start + 2

      reallocate = .TRUE.
      IF ( ALLOCATED( data%IBREAK ) ) THEN
        IF ( SIZE( data%IBREAK ) < lbreak ) THEN
          DEALLOCATE( data%IBREAK ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%IBREAK( lbreak ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%IBREAK' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( data%RES_l ) ) THEN
        IF ( LBOUND( data%RES_l, 1 ) /= 1 .OR.                                 &
             UBOUND( data%RES_l, 1 ) /= data%dims%c_l_end ) THEN 
          DEALLOCATE( data%RES_l )
        ELSE ; reallocate = .FALSE. ; END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%RES_l( 1 : data%dims%c_l_end ),                         &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%RES_l' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%RES_u ) ) THEN
        IF ( LBOUND( data%RES_u, 1 ) /= data%dims%c_u_start .OR.               &
             UBOUND( data%RES_u, 1 ) /= prob%m ) THEN 
          DEALLOCATE( data%RES_u )
        ELSE ; reallocate = .FALSE. ; END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%RES_u( data%dims%c_u_start : prob%m ),             &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%RES_u' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%A_norms ) ) THEN
        IF ( SIZE( data%A_norms ) < prob%m ) THEN
          DEALLOCATE( data%A_norms ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%A_norms( prob%m ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%A_norms' ; GO TO 900
        END IF
      END IF
      
!  Compute the initial residuals

      DO i = 1, data%dims%c_u_start - 1
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'l', i, a_x, prob%C_l( i )
        data%RES_l( i ) = a_x - prob%C_l( i )
        data%A_norms( i ) = SQRT( a_norms )
      END DO

      DO i = data%dims%c_u_start, data%dims%c_l_end
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO 
!       write(6,*) 'l', i, a_x, prob%C_l( i )
!       write(6,*) 'u', i, a_x, prob%C_u( i )
        data%RES_l( i ) = a_x - prob%C_l( i )
        data%RES_u( i ) = prob%C_u( i ) - a_x
        data%A_norms( i ) = SQRT( a_norms )
      END DO

      DO i = data%dims%c_l_end + 1, prob%m
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'u', i, a_x, prob%C_u( i )
        data%RES_u( i ) = prob%C_u( i ) - a_x
        data%A_norms( i ) = SQRT( a_norms )
      END DO

!  If necessary, determine which constraints occur in the reference set

!  cold start from the value set in X; constraints active
!  at X will determine the initial working set.

      IF ( control%cold_start == 1 ) THEN

!  constraints with lower bounds

        DO i = 1, data%dims%c_u_start - 1
          IF ( ABS( data%RES_l( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_l( i )
            C_stat( i ) = - 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  constraints with both lower and upper bounds

        DO i = data%dims%c_u_start, data%dims%c_l_end
          IF ( ABS( data%RES_l( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_l( i )
            C_stat( i ) = - 1
          ELSE IF ( ABS( data%RES_u( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_u( i )
            C_stat( i ) = 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  constraints with upper bounds

        DO i = data%dims%c_l_end + 1, prob%m
          IF ( ABS( data%RES_u( i ) ) <= teneps ) THEN
!           write(6,*) i, data%RES_u( i )
            C_stat( i ) = 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  free variables

        B_stat( : data%dims%x_free ) = 0

!  simple non-negativity

        DO i = data%dims%x_free + 1, data%dims%x_l_start - 1
          IF ( ABS( prob%X( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i )
            B_stat( i ) = - 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple bound from below

        DO i = data%dims%x_l_start, data%dims%x_u_start - 1
          IF ( ABS( prob%X( i ) - prob%X_l( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i ) - prob%X_l( i )
            B_stat( i ) = - 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple bound from below and above

        DO i = data%dims%x_u_start, data%dims%x_l_end
          IF ( ABS( prob%X( i ) - prob%X_l( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i ) - prob%X_l( i )
            B_stat( i ) = - 1
          ELSE IF ( ABS( prob%X( i ) - prob%X_u( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i ) - prob%X_u( i )
            B_stat( i ) = 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple bound from above

        DO i = data%dims%x_l_end + 1, data%dims%x_u_end
          IF ( ABS( prob%X( i ) - prob%X_u( i ) ) <= teneps ) THEN
!           write(6,*) dims%m + i, prob%X( i ) - prob%X_u( i )
            B_stat( i ) = 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  simple non-positivity

        DO i = data%dims%x_u_end + 1, prob%n
          IF ( ABS( prob%X( i ) ) <= teneps ) THEN
!           write(6,*) prob%m + i, prob%X( i )
            B_stat( i ) = 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  cold start with only equality constraints active

      ELSE IF ( control%cold_start == 3 ) THEN
        B_stat = 0 
        C_stat( : MIN( data%dims%c_equality, prob%n ) ) = 1
        C_stat( MIN( data%dims%c_equality, prob%n ) + 1 : ) = 0

!  cold start with as many active constraints as possible

      ELSE IF ( control%cold_start == 4 ) THEN
        B_stat = 0 ; C_stat = 0
        l = 0 

!  equality constraints

        DO i = 1,  data%dims%c_equality
          IF ( l > prob%n ) EXIT
          C_stat( i ) = - 1
          l = l + 1
        END DO

!  simple bound from below

        DO i = data%dims%x_free + 1, data%dims%x_l_end
          IF ( l > prob%n ) EXIT
          B_stat( i ) = - 1
          l = l + 1
        END DO

!  simple bound from above

        DO i = data%dims%x_l_end + 1, data%dims%x_u_end
          IF ( l > prob%n ) EXIT
          B_stat( i ) = 1
          l = l + 1
        END DO

!  constraints with lower bounds

        DO i = data%dims%c_equality + 1, data%dims%c_l_end
          IF ( l > prob%n ) EXIT
          C_stat( i ) = - 1
          l = l + 1
        END DO

!  constraints with upper bounds

        DO i = data%dims%c_l_end + 1, prob%m
          IF ( l > prob%n ) EXIT
          C_stat( i ) = 1
          l = l + 1
        END DO

!  cold start with no active constraints

      ELSE IF ( control%cold_start /= 0 ) THEN
        B_stat = 0 ; C_stat = 0
      END IF

!     WRITE( out, "( ' b_stat ', /, ( 10I5 ) )" )                              &
!       B_stat( dims%x_free + 1 : prob%n )

!  Remove any dependent working constraints
!  ========================================

      CALL CPU_TIME( time ) 
      CALL QPA_remove_dependent( prob%n, prob%m, prob%A%val, prob%A%col,       &
                                 prob%A%ptr, data%K, data%SLS_data,            &
                                 data%SLS_control, C_stat, B_stat,             &
                                 data%IBREAK, data%P, data%SOL, data%D,        &
                                 prefix, control, inform, n_depen )
      CALL CPU_TIME( dum ) ; dum = dum - time
      inform%time%preprocess = inform%time%preprocess + dum
      
!  Allocate more real workspace arrays

      reallocate = .TRUE.
      IF ( ALLOCATED( data%H_s ) ) THEN
        IF ( SIZE( data%H_s ) < prob%n ) THEN ; DEALLOCATE( data%H_s ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%H_s( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%H_s' ; GO TO 900
        END IF
      END IF
      
!  Check for error exits

      IF ( inform%status /= 0 ) THEN

!  On error exit, compute the current objective function value

        data%H_s( : prob%n ) = zero
        CALL QPA_HX( data%dims,  prob%n, data%H_s( : prob%n ),                &
                      prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,               &
                      prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
        inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),              &
                                         data%H_s( : prob%n ) )           &
                     + DOT_PRODUCT( prob%X( : prob%n ),                   &
                                    prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

        IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
          WRITE( control%out, "( ' ' )" )
          IF ( inform%status /= 0 ) WRITE( control%error, 2040 )               &
            inform%status, 'QPA_remove_dependent'
        END IF
        GO TO 750

!       IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
!         WRITE( control%out, "(/, ' The following ',I7,' constraints appear', &
!      &         ' to be dependent', /, ( 8I8 ) )" ) n_depen, data%Index_C_freed

      END IF

!  Continue allocating workspace arrays

!     m_link = MIN( prob%m + prob%n - data%dims%x_free, prob%n )
      m_link = prob%m + prob%n - data%dims%x_free
      k_n_max = prob%n + m_link

!  Allocate real workspace

      reallocate = .TRUE.
      IF ( ALLOCATED( data%BREAKP ) ) THEN
        IF ( SIZE( data%BREAKP ) < lbreak ) THEN
          DEALLOCATE( data%BREAKP )
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%BREAKP( lbreak ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%BREAKP' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%A_s ) ) THEN
        IF ( SIZE( data%A_s ) < prob%m ) THEN ; DEALLOCATE( data%A_s ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%A_s( prob%m ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%A_s' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( data%PERT ) ) THEN
        IF ( SIZE( data%PERT ) < prob%m + prob%n ) THEN
          DEALLOCATE( data%PERT ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%PERT( prob%m + prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%PERT' ; GO TO 900
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
          bad_alloc = 'data%GRAD' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%VECTOR ) ) THEN
        IF ( SIZE( data%VECTOR ) < k_n_max ) THEN ; DEALLOCATE( data%VECTOR ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%VECTOR( k_n_max ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%VECTOR' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( data%RHS ) ) THEN
        IF ( SIZE( data%RHS ) < k_n_max + control%max_sc ) THEN
          DEALLOCATE( data%RHS ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%RHS( k_n_max + control%max_sc ),                        &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%RHS' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%S ) ) THEN
        IF ( SIZE( data%S ) < k_n_max + control%max_sc ) THEN 
          DEALLOCATE( data%S ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%S( k_n_max + control%max_sc ),                          &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%S' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%B ) ) THEN
        IF ( SIZE( data%B ) < k_n_max + control%max_sc ) THEN 
          DEALLOCATE( data%B ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%B( k_n_max + control%max_sc ),                          &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%B' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%RES ) ) THEN
        IF ( SIZE( data%RES ) < k_n_max + control%max_sc ) THEN 
          DEALLOCATE( data%RES ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%RES( k_n_max + control%max_sc ),                        &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%RES' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%S_perm ) ) THEN
        IF ( SIZE( data%S_perm ) < k_n_max + control%max_sc ) THEN
          DEALLOCATE( data%S_perm ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%S_perm( k_n_max + control%max_sc ),                     &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%S_perm' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%DX ) ) THEN
        IF ( SIZE( data%DX ) < k_n_max + control%max_sc ) THEN 
          DEALLOCATE( data%DX ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%DX( k_n_max + control%max_sc ),                         &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%DX' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%RES_print ) ) THEN
        IF ( SIZE( data%RES_print ) < k_n_max + control%max_sc ) THEN 
          DEALLOCATE( data%RES_print ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%RES_print( k_n_max + control%max_sc ),                  &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%RES_print' ; GO TO 900
        END IF
      END IF

      IF ( control%precon >= 0 ) THEN
        n_pcg = prob%n
      ELSE
        n_pcg = 0
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%R_pcg ) ) THEN
        IF ( SIZE( data%R_pcg ) < n_pcg ) THEN ; DEALLOCATE( data%R_pcg ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%R_pcg( n_pcg ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%R_pcg' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( data%X_pcg ) ) THEN
        IF ( SIZE( data%X_pcg ) < n_pcg ) THEN ; DEALLOCATE( data%X_pcg ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%X_pcg( n_pcg ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%X_pcg' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( data%P_pcg ) ) THEN
        IF ( SIZE( data%P_pcg ) < n_pcg ) THEN ; DEALLOCATE( data%P_pcg ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%P_pcg( n_pcg ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%P_pcg' ; GO TO 900
        END IF
      END IF

!  Allocate integer workspace arrays

      reallocate = .TRUE.
      IF ( ALLOCATED( data%SC ) ) THEN
        IF ( SIZE( data%SC ) < control%max_sc + 1 ) THEN 
          DEALLOCATE( data%SC ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%SC( control%max_sc + 1 ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%SC' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( data%REF ) ) THEN
        IF ( SIZE( data%REF ) < m_link ) THEN ; DEALLOCATE( data%REF ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
       ALLOCATE( data%REF( m_link ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%REF' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( data%C_up_or_low ) ) THEN
        IF ( LBOUND( data%C_up_or_low, 1 ) /= data%dims%c_u_start .OR.         &
             UBOUND( data%C_up_or_low, 1 ) /= data%dims%c_l_end ) THEN 
          DEALLOCATE( data%C_up_or_low )
        ELSE ; reallocate = .FALSE. ; END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%C_up_or_low( data%dims%c_u_start : data%dims%c_l_end ), &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%C_up_or_low' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%X_up_or_low ) ) THEN
        IF ( LBOUND( data%X_up_or_low, 1 ) /= data%dims%x_u_start .OR.         &
             UBOUND( data%X_up_or_low, 1 ) /= data%dims%x_l_end ) THEN 
          DEALLOCATE( data%X_up_or_low )
        ELSE ; reallocate = .FALSE. ; END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%X_up_or_low( data%dims%x_u_start : data%dims%x_l_end ), &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%X_up_or_low' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%PERM ) ) THEN
        IF ( SIZE( data%PERM ) < k_n_max + control%max_sc ) THEN
          DEALLOCATE( data%PERM ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%PERM( k_n_max + control%max_sc ),                       &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%PERM' ; GO TO 900
        END IF
      END IF

!  Find the total length of the control%max_sc largest rows

      DO i = 1, prob%m
        data%IBREAK( i ) = prob%A%ptr( i ) - prob%A%ptr( i + 1 )
      END DO
      CALL SORT_heapsort_build( prob%m, data%IBREAK( : prob%m ),               &
                                inform%status )
      lbd = 0
      DO i = 1, MIN( control%max_sc, prob%m )
        ii = prob%m - i + 1
        CALL SORT_heapsort_smallest( ii, data%IBREAK( : ii ), inform%status )
        lbd = lbd - data%IBREAK( ii )
      END DO
      IF ( control%max_sc > prob%m ) lbd = lbd + control%max_sc - prob%m

!  Allocate arrays

      reallocate = .TRUE.
      IF ( ALLOCATED( data%SCU_mat%BD_col_start ) ) THEN
        IF ( SIZE( data%SCU_mat%BD_col_start ) < control%max_sc + 1 ) THEN
           DEALLOCATE( data%SCU_mat%BD_col_start ) 
         ELSE ; reallocate = .FALSE. 
         END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%SCU_mat%BD_col_start( control%max_sc + 1 ),             &
                  STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%SCU_mat%BD_col_start' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%SCU_mat%BD_val ) ) THEN
        IF ( SIZE( data%SCU_mat%BD_val ) < lbd ) THEN
           DEALLOCATE( data%SCU_mat%BD_val ) 
         ELSE ; reallocate = .FALSE. 
         END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%SCU_mat%BD_val( lbd ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%SCU_mat%BD_val' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%SCU_mat%BD_row ) ) THEN
        IF ( SIZE( data%SCU_mat%BD_row ) < lbd ) THEN
           DEALLOCATE( data%SCU_mat%BD_row ) 
         ELSE ; reallocate = .FALSE. 
         END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%SCU_mat%BD_row( lbd ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%SCU_mat%BD_row' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( data%DIAG ) ) THEN
        IF ( SIZE( data%DIAG, 1 ) /= 2 .OR.                                    &
             SIZE( data%DIAG, 2 ) /= K_n_max ) THEN
           DEALLOCATE( data%DIAG ) 
         ELSE ; reallocate = .FALSE. 
         END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( data%DIAG( 2, K_n_max ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'data%DIAG' ; GO TO 900
        END IF
      END IF

!  decide on appropriate initial preconditioners and factorizations

      data%auto_prec = control%precon == 0
      data%auto_fact = control%factor == 0

!  If the Hessian has semi-bandwidth smaller than nsemib and the preconditioner 
!  is to be picked automatically, use the full Hessian. Otherwise, use the
!  Hessian of the specified semi-bandwidth.

      IF ( data%auto_prec ) THEN
        data%prec_hist = 2

!  prec_hist indicates which factors are currently being used. Possible values:
!   1 full factors used
!   2 band factors used
!   3 diagonal factors used (as a last resort)

!  Check to see if the Hessian is banded

 dod :  DO type = 1, 6
        
          SELECT CASE( type )
          CASE ( 1 )
        
            hd_start  = 1
            hd_end    = data%dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_free
        
          CASE ( 2 )
        
            hd_start  = data%dims%x_free + 1
            hd_end    = data%dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_l_start - 1
        
          CASE ( 3 )
        
            hd_start  = data%dims%x_l_start
            hd_end    = data%dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_u_start - 1
        
          CASE ( 4 )
        
            hd_start  = data%dims%x_u_start
            hd_end    = data%dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_l_end
        
          CASE ( 5 )
        
            hd_start  = data%dims%x_l_end + 1
            hd_end    = data%dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_u_end
        
          CASE ( 6 )
        
            hd_start  = data%dims%x_u_end + 1
            hd_end    = data%dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = prob%n
        
          END SELECT
    
!  rows with a diagonal entry
    
          hd_end = MIN( hd_end, prob%n )
          DO i = hd_start, hd_end
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 2
              IF ( ABS( i - prob%H%col( l ) ) > control%nsemib ) THEN
                data%prec_hist = 1
                EXIT dod
              END IF  
            END DO
          END DO
          IF ( hd_end == prob%n ) EXIT
    
!  rows without a diagonal entry
    
          hnd_end = MIN( hnd_end, prob%n )
          DO i = hnd_start, hnd_end
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              IF ( ABS( i - prob%H%col( l ) ) > control%nsemib ) THEN
                data%prec_hist = 1
                EXIT dod
              END IF  
            END DO
          END DO
          IF ( hd_end == prob%n ) EXIT
    
        END DO dod

      END IF

!  =========================
!  Solve the initial problem
!  =========================

      IF ( printi )                                                           &
        WRITE( control%out, "( /, ' Check for optimality when theta = 0.0' )" )

      CALL CPU_TIME( time_inner_start )
      CALL QPA_solve_qp( data%dims, prob%n, prob%m,                            &
                         prob%H%val, prob%H%col, prob%H%ptr,                   &
                         prob%G, prob%f, prob%rho_g, prob%rho_b, prob%A%val,   &
                         prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, prob%X_l, &
                         prob%X_u, prob%X, prob%Y, prob%Z, C_stat, B_stat,     &
                         m_link, K_n_max, lbreak,   data%RES_l, data%RES_u,    &
                         data%A_norms, data%H_s, data%BREAKP, data%A_s,        &
                         data%PERT, data%GRAD, data%VECTOR, data%RHS, data%S,  &
                         data%B, data%RES, data%S_perm, data%DX, n_pcg,        &
                         data%R_pcg, data%X_pcg, data%P_pcg, data%Abycol_val,  &
                         data%Abycol_row, data%Abycol_ptr, data%S_val,         &
                         data%S_row, data%S_col, data%S_colptr, data%IBREAK,   &
                         data%SC, data%REF, data%RES_print, data%DIAG,         &
                         data%C_up_or_low, data%X_up_or_low, data%PERM,        &
                         data%P, data%SOL, data%D,                             &
                         data%SLS_data, data%SLS_control,                      &
                         data%SCU_mat, data%SCU_info, data%SCU_data, data%K,   &
                         data%seed, time_inner_start,                          &
                         data%start_print, data%stop_print,                    &
                         data%prec_hist, data%auto_prec, data%auto_fact,       &
                         printi, prefix, control, inform )

      control%print_level = control%print_level + 1
      printi = control%out > 0 .AND. control%print_level >= 1 

      IF ( printi )                                                            &
        WRITE( control%out, "( /, ' Optimality established when theta = 0.0' )")

!  ================================
!  Now solve the parametric problem
!  ================================

!  Allocate arrays to hold interval information

      reallocate = .TRUE.
      IF ( ALLOCATED( interval%X ) ) THEN
        IF ( SIZE( interval%X ) < prob%n ) THEN
          DEALLOCATE( interval%X ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( interval%X( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'interval%X' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( interval%Y ) ) THEN
        IF ( SIZE( interval%Y ) < prob%m ) THEN
          DEALLOCATE( interval%Y ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( interval%Y( prob%m ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'interval%Y' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( interval%Z ) ) THEN
        IF ( SIZE( interval%Z ) < prob%n ) THEN
          DEALLOCATE( interval%Z ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( interval%Z( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'interval%Z' ; GO TO 900
        END IF
      END IF

      reallocate = .TRUE.
      IF ( ALLOCATED( interval%DX ) ) THEN
        IF ( SIZE( interval%DX ) < prob%n ) THEN
          DEALLOCATE( interval%DX ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( interval%DX( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'interval%DX' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( interval%DY ) ) THEN
        IF ( SIZE( interval%DY ) < prob%m ) THEN
          DEALLOCATE( interval%DY ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( interval%DY( prob%m ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'interval%DY' ; GO TO 900
        END IF
      END IF
      
      reallocate = .TRUE.
      IF ( ALLOCATED( interval%DZ ) ) THEN
        IF ( SIZE( interval%DZ ) < prob%n ) THEN
          DEALLOCATE( interval%DZ ) 
        ELSE ; reallocate = .FALSE. 
        END IF
      END IF
      IF ( reallocate ) THEN 
        ALLOCATE( interval%DZ( prob%n ), STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN 
          bad_alloc = 'interval%DZ' ; GO TO 900
        END IF
      END IF
      printi = .TRUE.
      
!  Loop over the intervals      

!  Compute the solution over the next parametric interval

  200 CONTINUE
      CALL PQP_solve_main( data%dims, prob%n, prob%m,                          &
                       prob%H%val, prob%H%col, prob%H%ptr,                     &
                       prob%G, prob%f, prob%theta, prob%theta_max, prob%A%val, &
                       prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, prob%X_l,   &
                       prob%X_u, prob%DG, prob%DC_l, prob%DC_u, prob%DX_l,     &
                       prob%DX_u, prob%X, prob%Y, prob%Z, prob%c, C_stat,      &
                       B_stat, m_link, K_n_max, lbreak,                        &
                       data%H_s, data%A_s,                                     &
                       data%PERT, data%GRAD, data%VECTOR, data%RHS, data%S,    &
                       data%B, data%RES, data%S_perm, data%DX, n_pcg,          &
                       data%R_pcg, data%X_pcg, data%P_pcg, data%Abycol_val,    &
                       data%Abycol_row, data%Abycol_ptr, data%S_val,           &
                       data%S_row, data%S_col, data%S_colptr, data%IBREAK,     &
                       data%SC, data%REF, data%RES_print, data%DIAG,           &
                       data%C_up_or_low, data%X_up_or_low, data%PERM,          &
                       data%SLS_data, data%SLS_control,                        &
                       data%SCU_mat, data%SCU_info, data%SCU_data, data%K,     &
                       interval%theta_l, interval%theta_u, interval%f,         &
                       interval%g, interval%h, interval%X, interval%Y,         &
                       interval%Z, interval%DX, interval%DY, interval%DZ,      &
                       time_inner_start,                                       &
                       data%prec_hist, data%auto_prec, data%auto_fact,         &
                       action, control%out, printi, printt, printe,            &
                       prefix, control, inform )

!  Decide on next action

!     IF ( action( 1: 3 ) == "end" .OR. action( 1: 8 ) == "re-enter" ) RETURN
      IF ( action( 1: 8 ) == "re-enter" ) RETURN

!  Restore the problem to its original form

  750 CONTINUE 
      CALL CPU_TIME( time )

!  Full restore

!     IF ( control%cold_start == 0 ) THExN
        CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,        &
                                   IX = C_stat( : data%QPP_map%m ) )
        CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,        &
                                   IX = B_stat( : data%QPP_map%n ) )
!     END IF
        
      IF ( control%restore_problem >= 2 ) THEN  
        CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,                &
                          get_all_parametric = .TRUE. )  
  
!  Restore vectors and scalars  
  
      ELSE IF ( control%restore_problem == 1 ) THEN  
        CALL QPP_restore( data%QPP_map, data%QPP_inform,                      &
                           prob, get_g = .TRUE.,                              &
                           get_x = .TRUE., get_x_bounds = .TRUE.,             &
                           get_y = .TRUE., get_z = .TRUE.,                    &
                           get_c = .TRUE., get_c_bounds = .TRUE.,             &
                           get_dg = .TRUE., get_dx_bounds = .TRUE.,           &
                           get_dc_bounds = .TRUE. )
  
!  Solution recovery  
  
      ELSE  
        CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,                &
                          get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,     &
                          get_c = .TRUE. )
      END IF
      CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
      CALL CPU_TIME( dum ) ; dum = dum - time
      inform%time%preprocess = inform%time%preprocess + dum
      prob%new_problem_structure = data%new_problem_structure

!  Compute total time

      CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
      IF ( printi ) WRITE( control%out, 2000 )                                 &
        inform%time%total, inform%time%preprocess,                             &
        inform%time%analyse, inform%time%factorize, inform%time%solve

  800 CONTINUE 
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving PQP_solve ' )" )

      RETURN  

!  Allocation error

  900 CONTINUE 
      inform%status = - 2
      CALL CPU_TIME( time ) ; inform%time%total = time - time_start 
      IF ( printi ) WRITE( control%out, 2900 ) bad_alloc, inform%alloc_status
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving PQP_solve ' )" )

      RETURN  

!  Non-executable statements

 2000 FORMAT( /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',            &
              /, 14X, ' =', 7X,  'PQP timing statistics', 6X, '=',             &
              /, 14X, ' =         total         preprocess      =',            &
              /, 14X, ' =', 2X, 0P, F12.2, 4X, F12.2, 9X, '=',                 &
              /, 14X, ' =     analyse   factorize    solve      =',            &
              /, 14X, ' =', 3F11.2, 6x, '=',                                   &
              /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=' )
 2010 FORMAT( ' ', /, '   **  Error return ',I3,' from PQP ' ) 
 2040 FORMAT( '   **  Error return ', I6, ' from ', A15 ) 
 2240 FORMAT( /, '  Warning - an entry from strict upper triangle of H given ' )
 2900 FORMAT( ' ** Message from -PQP_solve-', /,                          &
              ' Allocation error, for ', A30, /, ' status = ', I6 ) 

!  End of PQP_solve

      END SUBROUTINE PQP_solve

!-*-*-*-*-*-*-   Q P A _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE PQP_terminate( data, control, inform )

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine PQP_initialize
!   control see Subroutine PQP_initialize
!   inform  see Subroutine PQP_solve

!  Dummy arguments

      TYPE ( PQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( PQP_control_type ), INTENT( IN ) :: control        
      TYPE ( PQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: alloc_status, status

      inform%status = 0

!  Deallocate all allocated integer arrays

      IF ( ALLOCATED( data%SC ) ) THEN
        DEALLOCATE( data%SC, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%SC', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%REF ) ) THEN
        DEALLOCATE( data%REF, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%REF', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%IBREAK ) ) THEN
        DEALLOCATE( data%IBREAK, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%IBREAK', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%C_up_or_low ) ) THEN
        DEALLOCATE( data%C_up_or_low, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%C_up_or_low', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%X_up_or_low ) ) THEN
        DEALLOCATE( data%X_up_or_low, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%X_up_or_low', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%PERM ) ) THEN
        DEALLOCATE( data%PERM, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%PERM', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%Abycol_row ) ) THEN
        DEALLOCATE( data%Abycol_row, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%Abycol_row', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%Abycol_ptr ) ) THEN
        DEALLOCATE( data%Abycol_ptr, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%Abycol_ptr', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%S_row ) ) THEN
        DEALLOCATE( data%S_row, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%S_row', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%S_col ) ) THEN
        DEALLOCATE( data%S_col, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%S_col', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%S_colptr ) ) THEN
        DEALLOCATE( data%S_colptr, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%S_colptr', inform%alloc_status
        END IF
      END IF

!  Deallocate all allocated real arrays

      IF ( ALLOCATED( data%RES_l ) ) THEN
        DEALLOCATE( data%RES_l, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%RES_l', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%A_s ) ) THEN
        DEALLOCATE( data%A_s, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%A_s', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%A_norms ) ) THEN
        DEALLOCATE( data%A_norms, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%A_norms', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%PERT ) ) THEN
        DEALLOCATE( data%PERT, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%PERT', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%H_s ) ) THEN
        DEALLOCATE( data%H_s, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%H_s', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%GRAD ) ) THEN
        DEALLOCATE( data%GRAD, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%GRAD', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%VECTOR ) ) THEN
        DEALLOCATE( data%VECTOR, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%VECTOR', inform%alloc_status
        END IF
      END IF

     IF ( ALLOCATED( data%RHS ) ) THEN
        DEALLOCATE( data%RHS, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%RHS', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%S ) ) THEN
        DEALLOCATE( data%S, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%S', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%B ) ) THEN
        DEALLOCATE( data%B, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%B', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%B ) ) THEN
        DEALLOCATE( data%B, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
              WRITE( control%error, 2900 ) 'data%B', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%S_perm ) ) THEN
        DEALLOCATE( data%S_perm, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%S_perm', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%R_pcg ) ) THEN
        DEALLOCATE( data%R_pcg, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%R_pcg', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%X_pcg ) ) THEN
        DEALLOCATE( data%X_pcg, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%X_pcg', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%P_pcg ) ) THEN
        DEALLOCATE( data%P_pcg, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%P_pcg', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%Abycol_val ) ) THEN
        DEALLOCATE( data%Abycol_val, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%Abycol_val', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%S_val ) ) THEN
        DEALLOCATE( data%S_val, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 )                                             &
            WRITE( control%error, 2900 ) 'data%S_val', inform%alloc_status
        END IF
      END IF

!  Deallocate all arrays allocated within SCU

      IF ( ALLOCATED( data%SCU_mat%BD_col_start ) ) THEN
        DEALLOCATE( data%SCU_mat%BD_col_start,                                &
                    STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 ) WRITE( control%error, 2900 )                &
               'data%SCU_mat%BD_col_start', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%SCU_mat%BD_val ) ) THEN
        DEALLOCATE( data%SCU_mat%BD_val, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 ) WRITE( control%error, 2900 )                &
               'data%SCU_mat%BD_val', inform%alloc_status
        END IF
      END IF

      IF ( ALLOCATED( data%SCU_mat%BD_row ) ) THEN
        DEALLOCATE( data%SCU_mat%BD_row, STAT = inform%alloc_status )
        IF ( inform%alloc_status /= 0 ) THEN
          inform%status = - 2
          IF ( control%error > 0 ) WRITE( control%error, 2900 )                &
               'data%SCU_mat%BD_row', inform%alloc_status
        END IF
      END IF

      CALL SCU_terminate( data%SCU_data, status, data%SCU_info )

      IF ( status /= 0 .AND. control%error > 0 .AND. &
           control%print_level >= 1 ) WRITE ( control%out, &
           "( ' on exit from SCU_terminate,     status = ', I3 )" ) status

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF
      IF ( inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'qpa: data%SLS_data'
        IF ( control%error > 0 )                                               &
             WRITE( control%error, 2900 ) 'data%SLS_data', alloc_status
      END IF

      RETURN

!  Non-executable statement

 2900 FORMAT( ' ** Message from -PQP_terminate-', /,                      &
                 ' Allocation error, for ', A20, ', status = ', I6 ) 

!  End of subroutine PQP_terminate

      END SUBROUTINE PQP_terminate

! -*-*-*-*-*-   P Q P _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE PQP_solve_main(  dims, n, m,                                  &
                                  H_val, H_col, H_ptr, G, f, theta,            &
                                  theta_max, A_val, A_col, A_ptr, C_l, C_u,    &
                                  X_l, X_u, DG, DC_l, DC_u, DX_l, DX_u,        &
                                  X, Y, Z, C, C_stat, B_stat, m_link, K_n_max, &
                                  lbreak, H_dx,                                &
                                  A_dx, PERT, GRAD, VECTOR, RHS, S, B, RES,    &
                                  S_perm, DX, n_pcg, R_pcg, X_pcg, P_pcg,      &
                                  Abycol_val, Abycol_row, Abycol_ptr, S_val,   &
                                  S_row, S_col, S_colptr, IBREAK, SC, REF,     &
                                  RES_print, DIAG, C_up_or_low, X_up_or_low,   &
                                  PERM, SLS_data, SLS_control,                 &
                                  SCU_mat, SCU_info, SCU_data, K,              &
                                  theta_l, theta_u, f_int, g_int, h_int,       &
                                  X_int, Y_int, Z_int, DX_int,                 &
                                  DY_int,  DZ_int, time_start,                 &
                                  prec_hist, auto_prec, auto_fact,             &
                                  action, out, printi, printt, printe,         &
                                  prefix, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize 1/2 x(T) H x + g(T) x + f
!               + rho_g min( A x - c_l , 0 ) + rho_g max( A x - c_u , 0 )
!               + rho_b min(  x - x_l , 0  ) + rho_b max(  x - x_u , 0  )
!
!  where x is a vector of n components ( x_1, .... , x_n ), f, rho_g/rho_b are
!  constant, g is an n-vector, H is a symmetric matrix, A is an m by n matrix, 
!  and any of the bounds c_l, c_u, x_l, x_u may be infinite, using an active
!  set method. The subroutine is particularly appropriate when A and H are 
!  sparse, and when we do not anticipate a large number of active set
!  changes prior to optimality
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  dims is a structure of type QPA_data_type, whose components hold SCALAR
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
!   %h_diag_end_free is an INTEGER variable, which must be set by the user to 
!    the index of the last free variable whose for which the Hessian has a 
!    diagonal entry
!
!   %h_diag_end_nonneg is an INTEGER variable, which must be set by the user to
!    the index of the last nonnegative variable whose for which the Hessian has
!    a diagonal entry
!
!   %h_diag_end_lower is an INTEGER variable, which must be set by the user to 
!    the index of the last flower-bounded variable whose for which the Hessian 
!    has a diagonal entry
!
!   %h_diag_end_range is an INTEGER variable, which must be set by the user to 
!    the index of the last range variable whose for which the Hessian has a 
!    diagonal entry
!
!   %h_diag_end_upper is an INTEGER variable, which must be set by the user to 
!    the index of the last upper-bounded variable whose for which the Hessian 
!    has a diagonal entry
!
!   %h_diag_end_nonpos is an INTEGER variable, which must be set by the user to
!    the index of the last  nonpositive variable whose for which the Hessian 
!    has a diagonal entry
!
!   %np1 is an INTEGER variable, which must be set by the user to the
!    value n + 1
!
!   %npm is an INTEGER variable, which must be set by the user to the
!    value n + m
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
!    value dims%x_e + nc
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
!  f is a REAL variable, which must be set by the user to the value of
!   the constant term f in the objective function. 
!   This argument is not altered by the subroutine
!
!  rho_g and rho_b are REAL variables, which must be set by the
!   user to the values of the penalty parameters, rho_g and rho_b.
!  
!  G is a REAL array of length n, which must be set by
!   the user to the value of the gradient, g, of the linear term of the
!   quadratic objective function. The i-th component of G, i = 1, ....,
!   n, should contain the value of g_i.  The contents of this argument 
!   are not altered by the subroutine
!  
!  A_* is used to hold the matrix A by rows. In particular:
!      A_col( : )   the column indices of the components of A
!      A_ptr( : )   pointers to the start of each row, and past the end of
!                   the last row. 
!      A_val( : )   the values of the components of A
!
!  H_* is used to hold the LOWER TRIANGLULAR PART of H by rows. In particular:
!      H_col( : )   the column indices of the components of H
!      H_ptr( : )   pointers to the start of each row, and past the end of
!                   the last row. 
!      H_val( : )   the values of the components of H
!
!   NB. Each off-diagonal pair of nonzeros should be represented
!   by a single component of H. 
!  
!  X_l, X_u are REAL arrays of length n, which must be set by the user to the 
!   values of the arrays x_l and x_u of lower and upper bounds on x. Any
!   bound X_l( i ) or X_u( i ) larger than or equal to biginf in absolute value
!   will be regarded as being infinite (see the entry control%biginf).
!   Thus, an infinite lower bound may be specified by setting the appropriate 
!   component of X_l to a value smaller than -biginf, while an infinite 
!   upper bound can be specified by setting the appropriate element of X_u
!   to a value larger than biginf. If X_u( i ) < X_l( i ), X_u( i ) will be
!   reset to X_l( i ). Otherwise, the contents of these arguments are not 
!   altered by the subroutine
!  
!  C_l, C_u are  REAL array of length m, which must be set by the user to the 
!  values of the arrays bl and bu of lower and upper bounds on A x.
!   Any bound bl_i or bu_i larger than or equal to biginf in absolute value
!   will be regarded as being infinite (see the entry control%biginf).
!   Thus, an infinite lower bound may be specified by setting the appropriate 
!   component of C_u to a value smaller than -biginf, while an infinite 
!   upper bound can be specified by setting the appropriate element of BU
!   to a value larger than biginf. If C_u( i ) < C_l( i ), C_u( i ) will be
!   reset to C_u( i ). Otherwise, the contents of these arguments are not 
!   altered by the subroutine
!  
!  X is a REAL array of length n, which must be set by
!   the user on entry to QPA_solve to give an initial estimate of the 
!   optimization parameters, x. The i-th component of X should contain 
!   the initial estimate of x_i, for i = 1, .... , n.  The estimate need 
!   not satisfy the simple bound constraints and may be perturbed by 
!   QPA_solve prior to the start of the minimization.  On exit from 
!   QPA_solve, X will contain the best estimate of the optimization 
!   parameters found
!  
!  Y is a REAL array of length m, which need not be set on entry.
!   On exit, the i-th component of Y contains the best estimate of the
!   the Lagrange multiplier connected to constraint i.
!  
!  Z is a REAL array of length n, which need not be set on entry.
!   On exit, the i-th component of Z contains the best estimate of the
!   the Dual variable connected to simple bound constraint i.
!  
!  C is a REAL array of length m, which need not be set on entry.
!   On exit, the i-th component of C contains the product Ax
!
!  C_stat is a INTEGER array of length m, which may be set by the user
!   on entry to QPA_solve to indicate which of the constraints are to
!   be included in the initial working set. If this facility is required,
!   the component control%warm_start must be set .TRUE. on entry; C_stat
!   need not be set if control%warm_start is .FALSE. . On exit,
!   C_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   C_stat( i ) < 0, the i-th constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to QPA_solve to indicate which of the simple bound constraints 
!   are to be included in the initial working set. If this facility is required,
!   the component control%warm_start must be set .TRUE. on entry; B_stat
!   need not be set if control%warm_start is .FALSE. . On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   B_stat( i ) < 0, the i-th bound constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  control is a structure of type QPA_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to QPA_initialize. See QPA_initialize 
!   for details
!
!  inform is a structure of type QPA_inform_type that provides 
!    information on exit from QPA_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!     1 The objective function is unbounded below along the line
!       starting at X and pointing in the direction ??
!
!     2 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of 
!       a badly scaled problem.
!
!     3 one of the restrictions 
!          n     >=  1
!          m     >=  0
!       has been violated.
!
!     4 The step is too small to make further impact.
!
!     5 The Newton residuals are larger than the current measure of
!       optimality so further progress is unlikely.
!
!     6 The Hessian matrix is non-convex in the manifold defined by
!       the linear constraints.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      CHARACTER ( LEN = 20 ), INTENT( INOUT ) :: action
      TYPE ( PQP_control_type ), INTENT( INOUT ) :: control        
!     TYPE ( PQP__control_type ), INTENT( IN ) :: control        
! *** nb change control back to "in" here and before
      TYPE ( PQP_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( PQP_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, m_link, k_n_max, lbreak, n_pcg
      INTEGER, INTENT( IN ) :: out
      INTEGER, INTENT( INOUT ) :: prec_hist
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( INOUT ) :: theta, theta_max
      REAL, INTENT( IN ) :: time_start
      LOGICAL, INTENT( INOUT ) :: auto_prec, auto_fact
      LOGICAL, INTENT( IN ) :: printi, printt, printe
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: B_stat
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G, DG
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: DC_l, DC_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: DX_l, DX_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y, C
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      INTEGER, INTENT( OUT ), DIMENSION( lbreak ) :: IBREAK
      INTEGER, INTENT( OUT ), DIMENSION( control%max_sc + 1 ) :: SC
      INTEGER, INTENT( OUT ), DIMENSION( m_link ) :: REF
      INTEGER, INTENT( OUT ),                                                  &
               DIMENSION( dims%c_u_start : dims%c_l_end ) :: C_up_or_low
      INTEGER,                                                                 &
               DIMENSION( dims%x_u_start : dims%x_l_end ) :: X_up_or_low
      INTEGER, INTENT( OUT ), DIMENSION( k_n_max + control%max_sc ) :: PERM
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: Abycol_row, Abycol_ptr
      INTEGER, ALLOCATABLE, DIMENSION( : )  :: S_row, S_col, S_colptr
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: A_dx
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: H_dx
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: GRAD
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( k_n_max ) :: VECTOR
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m + n ) :: PERT
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( k_n_max + control%max_sc ) :: RHS, S, B,  &
                                     RES, S_perm, DX, RES_print
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, k_n_max ) :: DIAG
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
                          DIMENSION( n_pcg ) :: R_pcg, X_pcg, P_pcg
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Abycol_val, S_val
      TYPE ( SLS_control_type ), INTENT( INOUT ) :: SLS_control
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_mat
      TYPE ( SCU_info_type ), INTENT( INOUT ) :: SCU_info
      TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
      TYPE ( SMT_type ), INTENT( INOUT ) :: K
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix

      REAL ( KIND = wp ) :: theta_l, theta_u, f_int, g_int, h_int
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) ::                     &
                          X_int, Z_int, DX_int, DZ_int
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) ::                     &
                          Y_int, DY_int

!  Parameters

! ===========================================================================
!
!  Constraints: 1, .., m are general constraints
!               m+1, .., m+n might be simple bounds on variables
!
!  Reference constraints: those constraints which are active on refactorization
!  Other constraints: those constraints which are not reference constraints
!
!                                  Constraint:
!  C_stat( i ) = j   j > 0         active other pointing to SC(j)
!                    j = 0         inactive other
!                    j < 0         reference pointing to REF
!
!  B_stat( i ) = j   j > 0         active other pointing to SC(j)
!                    j = 0         inactive other
!                    j < 0         reference pointing to REF
!
!  SC( j ) = i   i in (0,m]        active other pointing to C_stat(i)
!                i in (m,m+n]      active other pointing to B_stat(i-m)
!                i in [-m,0)       inactive other pointing to C_stat(-i)
!                i in [-m-n,-m)    inactive other pointing to B_stat(-i-m)
!                i = 0             artificial row (dead inactive reference)
!
!  REF( j ) = i  i in (0,m]        active reference pointing to C_stat(i)
!                i in (m,m+n]      active reference pointing to B_stat(i-m)
!                i < 0             inactive reference pointing to SC
!
! In other words:
!
!  Active other:       C_stat +            => SC +  => C_stat
!                      B_stat +            => SC ++ => B_stat
!  Active reference:   C_stat -  => REF +           => C_stat
!                      B_stat -  => REF ++          => B_stat
!  Inactive reference: C_stat -  => REF -  => SC -  => C_stat
!                      B_stat -  => REF -  => SC -- => B_stat
!
! ===========================================================================

!  Local variables

      INTEGER :: i, ii, iii, j, jj, l, pcount, nsemib, max_col, print_level
      INTEGER :: precon, factor, dof, jumpto_factorize_reference, j_add, j_del
      INTEGER :: itref_max, cg_maxit, pcg_status, imin, pcg_iter, m_active
      INTEGER :: QPA_addel_constraint_status, scu_status, s_minus, s_plus
      INTEGER :: interval
      REAL ( KIND = wp ) :: delta, theta_c, lh, rh, G_perturb, mult, dmult, hmax
      REAL ( KIND = wp ) :: inner_stop_absolute, inner_stop_relative, dtheta
      REAL ( KIND = wp ) :: theta_end
      LOGICAL :: new_reference, printm, printd, check_dependent, warmer_start
      LOGICAL :: negative_curvature
      LOGICAL :: x_r0, y0, x_f0, z0, g_eq_h
      CHARACTER (LEN = 1 ) :: uplow, mo
      CHARACTER ( LEN = 10 ) :: addel
      CHARACTER ( LEN = 12 ) :: sc_data, outd, outf, outg, outh

      TYPE ( QPA_partition_type ) :: K_part

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering PQP_solve_main ' )" )
      printm = out > 0 .AND. print_level >= 3
      printd = out > 0 .AND. print_level >= 4

!  Branch to the appropriate point on initial and re-entry

      IF ( action( 1: 5 ) == "start" ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 5 )                  &
          WRITE( control%out, "( ' initial entry ' )" )
      ELSE IF ( action( 1: 8 ) == "re-enter" ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 5 )                  &
          WRITE( control%out, "( ' re-entry ' )" )
        GO TO 200
      ELSE
        RETURN
      END IF

!  Initial entry

      IF ( theta_max == zero ) THEN
        action = "end"
        RETURN
      END IF

!  Check extent of parametric interval

      theta_end = theta_max

!  constraints with both lower and upper bounds

      DO i = dims%c_u_start, dims%c_l_end
        delta = DC_u( i ) - DC_l( i )
        IF ( delta < zero )                                                    &
          theta = MIN( theta, - ( C_u( i ) - C_l( i ) ) / delta )
      END DO

!  simple bound from below and above

      DO i = dims%x_u_start, dims%x_l_end
        delta = DX_u( i ) - DX_l( i )
        IF ( delta < zero )                                                    &
          theta = MIN( theta, - ( X_u( i ) - X_l( i ) ) / delta )
      END DO

      IF ( theta_end < theta_max ) THEN
        IF ( printi .AND. theta < theta_max ) WRITE( out,                      &
            "( /, ' ** Warning: parametric solution will be infeasible',       &
         &     ' for all theta >', ES11.4, /, '    Reducing upper limit from', &
         &     ' requested upper limit' ,  ES11.4 )" ) theta, theta_max
      END IF 

      IF ( printd ) WRITE( control%out, "( ' entering QPA_solve_main ' )" )

      pcount = 0 ; print_level = control%print_level

!  Ensure that precon has a reasonable value

      precon = control%precon
      IF ( precon >= 6 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( ' precon = ', I6, ' out of range [0,5]. Reset to 3') ") precon
        precon = 3
      END IF

!  Do the same for factor

      factor = control%factor
      IF ( factor < 0 .OR. factor > 2 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( ' factor = ', I6, ' out of range [0,2]. Reset to 2') ") precon
        factor = 2
      END IF

      interval = 0
      nsemib = control%nsemib
      max_col = control%max_col
      IF ( max_col < 0 ) max_col = n
      inner_stop_absolute = control%inner_stop_absolute
      inner_stop_relative = control%inner_stop_relative
      new_reference = .TRUE.
      itref_max = control%itref_max
      check_dependent = .FALSE. ; warmer_start = .FALSE.

!  Compute the constraint value

     C = zero
     DO i = 1, m
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          C( i ) = C( i ) + A_val( l ) * x( A_col( l ) )
        END DO
      END DO

!  Compute the objective gradient at the initial point

      GRAD = G
      CALL QPA_HX( dims, n, GRAD, H_ptr( n + 1 ) - 1,                          &
                   H_val, H_col, H_ptr, X, '+' )

!  =========
!  Main loop
!  =========

      theta = zero ; action = "re-enter"

  200 CONTINUE

      theta_c = theta
      
!  If necessary, form the new reference matrix

      IF ( .NOT. new_reference ) GO TO 20
        new_reference = .FALSE.
        G_perturb = zero
        addel = '          '

!  Determine which constraints occur in the reference set
    
        K_part%m_ref = 0

!  general equalities

        DO i = 1, dims%c_equality
          IF ( C_stat( i ) /= 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            C_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = i
          END IF
        END DO

!  general inequalities bounded from below

        DO i = dims%c_equality + 1, dims%c_u_start - 1
          IF ( C_stat( i ) < 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            C_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = i
          ELSE
            C_stat( i ) = 0
          END IF
        END DO

!  general inequalities bounded both from below and above

        DO i = dims%c_u_start, dims%c_l_end
          C_up_or_low( i ) = C_stat( i )
          IF ( C_stat( i ) /= 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            C_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = i
          END IF
        END DO

!  general inequalities bounded from above

        DO i = dims%c_l_end + 1, m
          IF ( C_stat( i ) > 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            C_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = i
          ELSE
            C_stat( i ) = 0
          END IF
        END DO
        K_part%c_ref = K_part%m_ref

!  simple non-negativity

        DO i = dims%x_free + 1, dims%x_l_start - 1
          ii = m + i
          IF ( B_stat( i ) < 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            B_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = ii
          END IF
        END DO

!  simple bound from below

        DO i = dims%x_l_start, dims%x_u_start - 1
          ii = m + i
          IF ( B_stat( i ) < 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            B_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = ii
          END IF
        END DO

!  simple bound from below and above

        DO i = dims%x_u_start, dims%x_l_end
          ii = m + i
          X_up_or_low( i ) = B_stat( i )
          IF ( B_stat( i ) /= 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            B_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = ii
          END IF
        END DO

!  simple bound from above

        DO i = dims%x_l_end + 1, dims%x_u_end
          ii = m + i
          IF ( B_stat( i ) > 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            B_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = ii
          END IF
        END DO

!  simple non-positivity

        DO i = dims%x_u_end + 1, n
          ii = m + i
          IF ( B_stat( i ) > 0 ) THEN
            K_part%m_ref = K_part%m_ref + 1
            B_stat( i ) = - K_part%m_ref
            REF( K_part%m_ref ) = ii
          END IF
        END DO

!       WRITE( out, "( ' c_stat ', /, ( 10I5 ) )" ) C_stat( : m )
!       WRITE( out, "( ' b_stat ', /, ( 10I5 ) )" ) B_stat( : n )
!       WRITE( out, "( ' ref ', /, ( 10I5 ) )" ) REF( : K_part%m_ref )
!       WRITE( 24, "( ' ref ', /, ( 10I5 ) )" ) REF( : K_part%m_ref )
        IF ( out > 0 .AND. print_level >= 4 ) THEN
          IF ( K_part%c_ref > 0 )                                              &
            WRITE( out, "( ' REF(c) ', /, ( 10I5 ) )" ) REF( : K_part%c_ref )
          IF ( K_part%m_ref >= K_part%c_ref )                                  &
            WRITE( out, "( ' REF(b) ', /, ( 10I5 ) )" )                        &
              REF( K_part%c_ref + 1 : K_part%m_ref ) - m
        END IF

! Set up the initial reference matrix

        K_part%k_ref = n + K_part%m_ref ; SCU_mat%n = K_part%k_ref

!  Set automatic choices for the factorizations and preconditioner

   5    CONTINUE
        dof = n - K_part%m_ref
        IF ( auto_fact ) factor = 0
        IF ( auto_prec ) THEN
          IF ( dof <= 1 ) THEN
            precon = 3
            nsemib = 0
          ELSE IF ( prec_hist == 1 ) THEN
            precon = 1
            nsemib = control%nsemib
          ELSE IF ( prec_hist == 2 ) THEN
            precon = 3
            nsemib = control%nsemib
          ELSE
            precon = 3
            nsemib = 0
          END IF
        END IF

!  Form and factorize the reference matrix

        jumpto_factorize_reference = 0
! 10    CONTINUE

!       WRITE(6,*) ' factorize reference matrix '
        CALL QPA_factorize_reference(                                          &
                    dims, n, m, jumpto_factorize_reference,                    &
                    k_n_max, print_level, m_link, max_col, factor, precon,     &
                    nsemib, hmax, G_perturb, out, prec_hist, printi, printt,   &
                    printe, G_eq_H, auto_prec, auto_fact,                      &
                    check_dependent, mo, PERM, REF, C_stat, B_stat,            &
                    A_ptr, A_col, H_ptr, H_col, A_val, H_val, K, K_part, S,    &
                    Abycol_row, Abycol_ptr, S_row, S_col, S_colptr,            &
                    Abycol_val, S_val, DIAG, SLS_data, SLS_control,            &
                    prefix, control, inform )

        IF ( jumpto_factorize_reference == 1 ) THEN
          RETURN
        ELSE IF ( jumpto_factorize_reference == 2 ) THEN
          CALL QPA_new_reference_set( control, inform, prefix, n, m, K_part,   &
                                      SCU_mat, out, m_link, pcount, printd,    &
                                      warmer_start, check_dependent, C_stat,   &
                                      B_stat, SC, REF, IBREAK, A_ptr,          &
                                      A_col, A_val, K, SLS_data, SLS_control )
          GO TO 5
        END IF
!       WRITE(6,*) ' factors formed '
        IF ( printi ) WRITE( out, 2000 )

! Initialize the Schur complement

!       nbd = 0
        SCU_mat%m = 0 ; SCU_mat%class = 2 ; SCU_mat%m_max = control%max_sc
        SCU_mat%BD_col_start( 1 ) = 1
        scu_status = 1
        CALL SCU_factorize( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),          &
                            scu_status, SCU_info )
        m_active = K_part%m_ref
        s_minus = 0 ; s_plus = 0
        j_del = 0 ; j_add = 0
  20  CONTINUE

!     IF ( SCU_mat%m == control%max_sc ) THEN
!       CALL QPA_new_reference_set( control, inform, dims, K_part,             &
!                                   SCU_mat, out, m_link, pcount, printd,      &
!                                   warmer_start, check_dependent, C_stat,     &
!                                   B_stat, SC, REF, IBREAK, A_ptr,            &
!                                   A_col, A_val, K, CNTL, FACTORS )
!       GO TO 5
!     END IF

!   ------------------------------------------
!                DETERMINE DIRECTION
!   ------------------------------------------

!  ----------------------------------------------------------

!  Consider the interval starting frtom theta_c for which the
!  solution is (x_c,y_c,z_c). i.e., 

!    (  H   A_W  I_W^T ) ( x_c )   (  - (g  + theta_c  dg  ) )
!    ( A_W             ) ( y_c ) = (    c_W + theta_c dc_W   )
!    ( I_W             ) ( z_c )   (    b_W + theta_c db_W   )

!  So long as the aboove coefficient matrix has the correct
!  inertia (i.e., |A_w| + |I_W| -ve eigenvalues), compute

!    (  H   A_W  I_W^T ) ( dx )   ( - dg )
!    ( A_W             ) ( dy ) = ( dc_W )
!    ( I_W             ) ( dz )   ( db_W )

!  and examine the piecewise linear solution 

!    ( x_c )                     ( dx )
!    ( y_c ) + (theta - theta_c) ( dy ) 
!    ( z_c )                     ( dz )

!  for theta > theta_c

!  N.B. The parametric value of the quadratic objective is

!  q(x(theta)) = f + (g + theta dg)^T x(theta) + 1/2x(theta)^T H x(theta)
!              = f_c + g_c (theta-theta_c) + h_c (theta-theta_c)^2

!  where f_c = f + (g + theta_c dg)^T x_c + 1/2 x_c^T H x_c
!        g_c = x_c^T dg + dx^T (g + theta_c dg + H x_c)
!        h_c = 1/2 dx^T H dx + dg^T dx

!  ----------------------------------------------------------

!  First solve

!    (  P   A_W  I_W^T ) ( dx_f )     (  dg  )
!    ( A_W             ) ( dy_f ) = - ( dc_W )
!    ( I_W             ) ( dz_f )     ( db_W )

!  to find a point that satisfies the constraints

!  Set up the right-hand side

      S( : n ) = - DG
      DO i = 1, K_part%m_ref
        j =  REF( i )
        IF ( j > 0 ) THEN
          IF ( j <= m ) THEN
            IF ( j < dims%c_u_start ) THEN
              S( n + i ) = DC_l( j )
            ELSE IF ( j > dims%c_l_end ) THEN
              S( n + i ) = DC_u( j )
            ELSE IF ( j >= dims%c_u_start ) THEN
              IF ( C_up_or_low( j ) == 1 ) THEN
                S( n + i ) = DC_u( j )
              ELSE
                S( n + i ) = DC_l( j )
              END IF
            END IF
          ELSE
            j = j - m
            IF ( j < dims%x_u_start ) THEN
              S( n + i ) = DX_l( j )
            ELSE IF ( j > dims%x_l_end ) THEN
              S( n + i ) = DX_u( j )
            ELSE IF ( j >= dims%x_u_start ) THEN
              IF ( X_up_or_low( j ) == 1 ) THEN
                S( n + i ) = DX_u( j )
              ELSE
                S( n + i ) = DX_l( j )
              END IF
            END IF
          END IF
        ELSE
          S( n + i ) = zero
        END IF
      END DO
!     write( 6, "( I6, ES12.4 )" ) ( i, S( i ), i = 1, K_part%k_ref )
          
!  Solve the system to find a feasible point; store dx in S

      S_perm( PERM( : K_part%k_ref ) ) = S( : K_part%k_ref )

!     write( 6, "( I6, ES12.4 )" ) ( i, S_perm( i ), i =1,K_part%k_ref )
!     write(6,*) ' =========== warm start '

      x_r0 = .FALSE. ; y0 = .FALSE. ; x_f0 = .FALSE. ; z0 = .FALSE.

!     WRITE(6,*) ' find feasible point '

      CALL QPA_ir( K, SLS_data, K_part, S_perm( : K_part%k_ref ),              &
                   B( : K_part%k_ref ), RES( : K_part%k_ref ),                 &
                   x_r0, y0, x_f0, z0, SLS_control, itref_max + 1,             &
                   out, printm, RES_print, inform )

!     WRITE(6,*) ' feasible point found '

      inform%factorization_status = inform%status
      IF ( printm ) WRITE( out, "( ' ' )" )
  
!     write(6,"( 's,rhs ', /, ( 2ES12.4 ) )" ) ( S_perm( PERM( i ) ),          &
!          S( i ), i = 1, n_all )
!     write( 6, "( I6, ES12.4 )" ) ( i, S( i ), i = 1, K_part%k_ref )
  
      S( : K_part%k_ref ) = S_perm( PERM( : K_part%k_ref ) ) 

!     WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" ) S( : n )


!  Now solve

!    (  H   A_W  I_W^T ) ( dx^f + dx^s )     ( dg )
!    ( A_W             ) (     dy      ) = - (  0 )
!    ( I_W             ) (     dz      )     (  0 )

!  by attempting to minimize the quadratic on the current working set

      IF ( .NOT. G_eq_H ) THEN
        inner_stop_relative = zero
        inner_stop_absolute = SQRT( EPSILON( one ) )
        cg_maxit = control%cg_maxit
        X_pcg = S( : n )
        RHS( : n ) = - DG
        iii = itref_max + 1

!       WRITE(6,*) ' find EQP solution '
!       iterr = inform%iter
        CALL QPA_pcg( dims, n, S, RHS, R_pcg, X_pcg, P_pcg, S_perm,            &
                      RES, B, DX, VECTOR, PERM, SCU_mat, SCU_data,             &
                      K, K_part, H_val, H_col, H_ptr, prefix, control,         &
                      SLS_control, SLS_data, print_level - 1,                  &
                      cg_maxit, dof, .TRUE., inner_stop_absolute,              &
                      inner_stop_relative, iii, inform,                        &
                      pcg_iter, negative_curvature, pcg_status,                &
                      RES_print( : K_part%n_free + K_part%c_ref ) )
        itref_max = iii - 1
!       WRITE(6,*) ' EQP solution found '
  
        IF ( pcg_status < 0 ) THEN
          IF ( printt ) WRITE( out, "( /,                                      &
         &  ' Warning return from QPA_pcg, status = ', I6 )") pcg_status
        END IF          
!       IF ( pcg_status == 0 .OR. pcg_status == 1 ) X = S( : n )
        inner_stop_absolute = control%inner_stop_absolute
        inner_stop_relative = control%inner_stop_relative
      END IF          

!     WRITE( out, "( ' s ', /, ( 5ES12.4 ) )" ) S( : n )
      write( 6, "( I6, ES12.4 )" ) ( i, S( i ), i = 1, K_part%k_ref )

!     write(6,"( ' x_l(1), x(1), x_u(1)   ', 3ES12.4 )") x_l(1), x(1), x_u(1)
!     write(6,"( ' x_l(n), x(n), x_u(n)   ', 3ES12.4 )") &
!                  x_l(n), x(n), x_u(n)

!  Compute A dx

     A_dx = zero
     DO i = 1, m
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          A_dx( i ) = A_dx( i ) + A_val( l ) * S( A_col( l ) )
        END DO
      END DO

!  Record the piecewise solution 
!    (x_c,y_c,z_c)+(theta-theta_c) (dx,dy,dz)

      X_int = X
      DX_int = S( : n )
      Y_int = Y
      DY_int = zero
      Z_int = Z
      DZ_int = zero

!   ------------------------------------------
!                DETERMINE INTERVAL 
!   ------------------------------------------

!  Now determine the upper extent of the current interval:
!  find the largrest theta for which

!    c_l + theta dc_l <= A x_c + ( theta - theta_c ) A dx <= c_u + theta dc_u
!    x_l + theta dx_l <=  x_c  + ( theta - theta_c )  dx  <= x_u + theta dx_u
!    y_c + ( theta - theta_c ) dy_l >= 0
!    z_c + ( theta - theta_c ) dz_l >= 0

      theta = theta_end
      imin = 0
      IF ( printm ) WRITE( out, "( /, ' initial    ', ES12.4 )" ) theta

! Check

! a)  c_l + theta dc_l <= A x_c + ( theta - theta_c ) A dx <= c_u + theta dc_u
!     (for inactive constrints) 

!  equality constraints

      DO i = 1, dims%c_equality
        l = C_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

!  Nothing to do ... should be active

      END DO

!  constraints with lower bounds

      DO i = dims%c_equality + 1, dims%c_l_end
        l = C_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

!  require that lh <= rh * theta (true for theta = theta_c .... )

        lh = C_l( i ) - C( i ) + theta_c * A_dx( i )
        rh = A_dx( i ) - DC_l( i )

        IF ( ( lh < zero .AND. rh < zero ) .OR.                                &
             ( lh > zero .AND. rh > zero ) ) THEN
!         WRITE( out, 2010 ) 'cl+', i, lh / rh
          IF ( lh / rh < theta ) THEN
            theta = lh / rh
            imin = i
            IF ( printm ) WRITE( out, 2010 ) 'cl+', i, theta
          END IF
        END IF
      END DO

!  constraints with upper bounds

      DO i = dims%c_u_start, m
        l = C_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

!  require that lh >= rh * theta (true for theta = theta_c .... )

        lh = C_u( i ) - C( i ) + theta_c * A_dx( i )
        rh = A_dx( i ) - DC_u( i )

        IF ( ( lh < zero .AND. rh < zero ) .OR.                                &
             ( lh > zero .AND. rh > zero ) ) THEN
!         WRITE( out, 2010 ) 'cu+', i, lh / rh
          IF ( lh / rh < theta ) THEN
            theta = lh / rh
            imin = i
            IF ( printm ) WRITE( out, 2010 ) 'cu+', i, theta
          END IF
        END IF
      END DO

! Now check

! b)  x_l + theta dx_l <= x_c + ( theta - theta_c ) dx <= x_u + theta dx_u
!     (for inactive bounds) 

!  simple non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

!  require that lh <= rh * theta (true for theta = theta_c .... )

        lh = - X( i ) + theta_c * S( i )
        rh = S( i ) - DX_l( i )

        IF ( ( lh < zero .AND. rh < zero ) .OR.                                &
             ( lh > zero .AND. rh > zero ) ) THEN
!         WRITE( out, 2010 ) 'bl+', i, lh / rh
          IF ( lh / rh < theta ) THEN
            theta = lh / rh
            imin = m + i
            IF ( printm ) WRITE( out, 2010 ) 'bl+', i, theta
          END IF
        END IF
      END DO

!  simple bound from below

      DO i = dims%x_l_start, dims%x_l_end
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

!  require that lh <= rh * theta (true for theta = theta_c .... )

        lh = X_l( i ) - X( i ) + theta_c * S( i )
        rh = S( i ) - DX_l( i )

        IF ( ( lh < zero .AND. rh < zero ) .OR.                                &
             ( lh > zero .AND. rh > zero ) ) THEN
!         WRITE( out, 2010 ) 'bl+', i, lh / rh
          IF ( lh / rh < theta ) THEN
            theta = lh / rh
            imin = m + i
            IF ( printm ) WRITE( out, 2010 ) 'bl+', i, theta
          END IF
        END IF
      END DO

!  simple bound from above

      DO i = dims%x_u_start, dims%x_u_end
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        lh = X_u( i ) - X( i ) + theta_c * S( i )
        rh = S( i ) - DX_u( i )

!  require that lh >== rh * theta (true for theta = theta_c .... )

        IF ( ( lh < zero .AND. rh < zero ) .OR.                                &
             ( lh > zero .AND. rh > zero ) ) THEN
!         WRITE( out, 2010 ) 'bu+', i, lh / rh
          IF ( lh / rh < theta ) THEN
            theta = lh / rh
            imin = m + i
            IF ( printm ) WRITE( out, 2010 ) 'bu+', i, theta
          END IF
        END IF
      END DO

!  simple non-positivity

      DO i = dims%x_u_end + 1, n
        l = B_stat( i )
        IF ( l > 0 ) CYCLE
        IF ( l < 0 ) THEN
          IF ( REF( - l ) > 0 ) CYCLE
        END IF

        lh = - X( i ) + theta_c * S( i )
        rh = S( i ) - DX_u( i )

!  require that lh >== rh * theta (true for theta = theta_c .... )

        IF ( ( lh < zero .AND. rh < zero ) .OR.                                &
             ( lh > zero .AND. rh > zero ) ) THEN
!         WRITE( out, 2010 ) 'bu+', i, lh / rh
          IF ( lh / rh < theta ) THEN
            theta = lh / rh
            imin = m + i
            IF ( printm ) WRITE( out, 2010 ) 'bu+', i, theta
          END IF
        END IF
      END DO

!  Finally check

! c)  y_c + ( theta - theta_c ) dy_l >= 0
!     (for active constrints) and 
! d)  z_c + ( theta - theta_c ) dz_l >= 0
!     (for active bounds)

!  Constraints in the reference set

      DO i = 1, K_part%m_ref
        j = REF( i )
        IF ( j > 0 ) THEN
          dmult = - S( n + i )
          uplow = 'l'
          IF ( j <= m ) THEN
            mult = Y( j )
            IF ( j <= dims%c_equality ) THEN
              DY_int( j ) = dmult
              CYCLE
            ELSE IF ( j > dims%c_l_end ) THEN
              mult = - mult
              dmult = - dmult
              uplow = 'u'
            ELSE IF ( j >= dims%c_u_start ) THEN
              IF ( C_up_or_low( j ) == 1 ) THEN
                mult = - mult
                dmult = - dmult
                uplow = 'u'
              END IF
            END IF
            DY_int( j ) = dmult
          ELSE
            jj = j - m
            mult = Z( jj )
            IF ( jj > dims%x_l_end ) THEN
              mult = - mult
              dmult = - dmult
              uplow = 'u'
            ELSE IF ( jj >= dims%x_u_start ) THEN
              IF ( X_up_or_low( jj ) == 1 ) THEN
                mult = - mult
                dmult = - dmult
                uplow = 'u'
              END IF
            END IF
            DZ_int( jj ) = dmult
          END IF

!  Ensure that mult + (theta-theta_c) dmult >= 0

          lh = mult - theta_c * dmult
          rh = - dmult

!  require that lh >== rh * theta (true for theta = theta_c .... )

          IF ( ( lh < zero .AND. rh < zero ) .OR.                              &
               ( lh > zero .AND. rh > zero ) ) THEN
!           IF ( j <= m ) THEN
!             WRITE( out, 2010 ) 'c' // uplow // '-', j, lh / rh
!           ELSE
!             WRITE( out, 2010 ) 'b' // uplow // '-', jj, lh / rh
!           END IF
            IF ( lh / rh < theta ) THEN
              theta = lh / rh
              imin = - j
              IF ( printm ) THEN
                IF ( j <= m ) THEN
                  WRITE( out, 2010 ) 'c' // uplow // '-', j, theta
                ELSE
                  WRITE( out, 2010 ) 'b' // uplow // '-', jj, theta
                END IF
              END IF
            END IF
          END IF
        END IF
      END DO

!  Constraints added since

      DO i = 1, SCU_mat%m
        j = SC( i )
        IF ( j > 0 ) THEN
          dmult = - S( K_part%k_ref + i )
          uplow = 'l'
          IF ( j <= m ) THEN
            mult = Y( j )
            IF ( j <= dims%c_equality ) THEN
              DY_int( j ) = dmult
              CYCLE
            ELSE IF ( j > dims%c_l_end ) THEN
              mult = - mult
              dmult = - dmult
              uplow = 'u'
            ELSE IF ( j >= dims%c_u_start ) THEN
              IF ( C_up_or_low( j ) == 1 ) THEN
                mult = - mult
                dmult = - dmult
                uplow = 'u'
              END IF
            END IF
            DY_int( j ) = dmult
          ELSE
            jj = j - m
            mult = Z( jj )
            IF ( jj > dims%x_l_end ) THEN
              mult = - mult
              dmult = - dmult
              uplow = 'u'
            ELSE IF ( jj >= dims%x_u_start ) THEN
              IF ( X_up_or_low( jj ) == 1 ) THEN
                mult = - mult
                dmult = - dmult
                uplow = 'u'
              END IF
            END IF
            DZ_int( jj ) = dmult
          END IF

!  Ensure that mult + (theta-theta_c) dmult >= 0

          lh = mult - theta_c * dmult
          rh = - dmult

!  require that lh >== rh * theta (true for theta = theta_c .... )

          IF ( ( lh < zero .AND. rh < zero ) .OR.                              &
               ( lh > zero .AND. rh > zero ) ) THEN
!           IF ( j <= m ) THEN
!             WRITE( out, 2010 ) 'c' // uplow // '-', j, lh / rh
!           ELSE
!             WRITE( out, 2010 ) 'b' // uplow // '-', jj, lh / rh
!           END IF
            IF ( lh / rh < theta ) THEN
              theta = lh / rh
              imin = - j
              IF ( printm ) THEN
                IF ( j <= m ) THEN
                  WRITE( out, 2010 ) 'c' // uplow // '-', j, theta
                ELSE
                  WRITE( out, 2010 ) 'b' // uplow // '-', jj, theta
                END IF
              END IF
            END IF
          END IF
        END IF
      END DO

      interval = interval + 1

!   ----------------------------------------------------------------
!                REVISE WORKING SET AND PRECONDITIONER
!   ----------------------------------------------------------------

!  Add a constraint to the working set

     IF ( imin > 0 ) THEN

       CALL QPA_add_constraint( QPA_addel_constraint_status, control,          &
               inform, dims, n, m, K_part, imin, out, k_n_max, m_link,         &
               itref_max, j_add, j_del, scu_status, m_active, s_plus,          &
               s_minus, printt, printm, printd, addel, sc_data, C_stat,        &
               B_stat, SC, REF, PERM, C_up_or_low, X_up_or_low, B, RES,        &
               RES_print, VECTOR, PERT, A_ptr, A_col, A_val, SCU_mat,          &
               SCU_info, SCU_data, K, SLS_control, SLS_data )

!  Delete a constraint from the working set

     ELSE IF ( imin < 0 ) THEN

       CALL QPA_delete_constraint( QPA_addel_constraint_status, control,       &
                inform, dims, n, m, K_part, - imin, out, k_n_max, m_link,      &
                itref_max, j_add, j_del, scu_status, m_active, s_plus,         &
                s_minus, printt, printm, printd, printe, addel, sc_data,       &
                C_stat, B_stat, SC, REF, PERM, C_up_or_low, X_up_or_low,       &
                B, RES, RES_print, VECTOR, PERT, SCU_mat,                      &
                SCU_info, SCU_data, K, SLS_control, SLS_data )                  

     ELSE
       addel = '      end'
     END IF

     IF ( printi ) THEN
       IF ( printt ) WRITE( out, 2000 )
       WRITE( out, "( I9, 2ES12.4, 1X, A10 )" ) interval, theta_c, theta, addel
     END IF

!  Check to see if a refedrence refactorization is required

!    WRITE(6,"( ' imin, status ', 2I7 )" ) imin, QPA_addel_constraint_status

     IF ( imin /= 0 .AND. QPA_addel_constraint_status > 0 ) THEN

       SELECT CASE( QPA_addel_constraint_status )
       CASE ( 1 )
!        CALL CPU_TIME( time ) ; time = time - time_start 
!        IF ( printt .OR. ( printi .AND. ( imin < 0 .AND. pcount == 0 ) ) )    &
!          WRITE( out, 2040 ) precon
!        IF ( printi ) WRITE( out, 2050 ) inform%iter, inform%merit,           &
!          t_opt, inform%infeas_g + inform%infeas_b,                           &
!          inform%num_g_infeas + inform%num_b_infeas,                          &
!          pcg_iter, sc_data, addel, time
       CASE ( 2 )
         IF ( printe ) WRITE ( out, 2062 ) inform%iter, addel
         IF ( printe ) WRITE ( out, 2020 ) scu_status
       CASE ( 3 )
         IF ( printe ) WRITE ( out, 2062 ) inform%iter, addel
         IF ( printe ) WRITE ( out, 2030 ) scu_status
       CASE ( 4 )
         WRITE( out, 2140 ) s_minus, s_plus, SCU_info%inertia( 1 : 2 )
         IF ( printe ) WRITE ( out, 2062 ) inform%iter, addel
       END SELECT

!  Ccompute the new reference set

  55   CONTINUE
       CALL QPA_new_reference_set( control, inform, prefix, n, m, K_part,      &
                                   SCU_mat, out, m_link, pcount, printd,       &
                                   warmer_start, check_dependent, C_stat,      &
                                   B_stat, SC, REF, IBREAK, A_ptr,             &
                                   A_col, A_val, K, SLS_data, SLS_control )

!  Set automatic choices for the factorizations and preconditioner

        dof = n - K_part%m_ref
        IF ( auto_fact ) factor = 0
        IF ( auto_prec ) THEN
          IF ( dof <= 1 ) THEN
            precon = 3
            nsemib = 0
          ELSE IF ( prec_hist == 1 ) THEN
            precon = 1
            nsemib = control%nsemib
          ELSE IF ( prec_hist == 2 ) THEN
            precon = 3
            nsemib = control%nsemib
          ELSE
            precon = 3
            nsemib = 0
          END IF
        END IF

!  Form and factorize the reference matrix

        jumpto_factorize_reference = 0

        WRITE(6,*) ' factorize reference matrix '
        CALL QPA_factorize_reference(                                          &
                    dims, n, m, jumpto_factorize_reference,                    &
                    k_n_max, print_level, m_link, max_col, factor, precon,     &
                    nsemib, hmax, G_perturb, out, prec_hist, printi, printt,   &
                    printe, G_eq_H, auto_prec, auto_fact,                      &
                    check_dependent, mo, PERM, REF, C_stat, B_stat,            &
                    A_ptr, A_col, H_ptr, H_col, A_val, H_val, K, K_part, S,    &
                    Abycol_row, Abycol_ptr, S_row, S_col, S_colptr,            &
                    Abycol_val, S_val, DIAG, SLS_data, SLS_control,            &
                    prefix, control, inform )

        IF ( jumpto_factorize_reference == 1 ) THEN
          RETURN
        ELSE IF ( jumpto_factorize_reference == 2 ) THEN
          GO TO 55
        END IF
        WRITE(6,*) ' factorization found '

! Initialize the Schur complement

!       nbd = 0
        SCU_mat%m = 0 ; SCU_mat%class = 2 ; SCU_mat%m_max = control%max_sc
        SCU_mat%BD_col_start( 1 ) = 1
        scu_status = 1
        CALL SCU_factorize( SCU_mat, SCU_data, VECTOR( : SCU_mat%n ),          &
                            scu_status, SCU_info )
        m_active = K_part%m_ref
        s_minus = 0 ; s_plus = 0
        j_del = 0 ; j_add = 0
        IF ( printi ) WRITE( out, 2000 )
     END IF

!  Record solution information in the current interval

      theta_l = theta_c
      theta_u = theta
      dtheta = theta_u - theta_l

! Compute the product H dx and update the interval function value

      H_dx = zero
      CALL QPA_HX( dims, n, H_dx, H_ptr( n + 1 ) - 1,                          &
                   H_val, H_col, H_ptr, S( : n ), '+' )

      f_int = inform%obj
      g_int = DOT_PRODUCT( GRAD, S( : n ) ) + DOT_PRODUCT( DG, X ) 
      h_int = half * DOT_PRODUCT( H_dx, S( : n ) ) +                           &
                     DOT_PRODUCT( DG, S( : n ) ) 

!  Record the value of the function and gradient at the end of the interval

      inform%obj = f_int + g_int * dtheta + h_int * dtheta ** 2
      GRAD = GRAD + dtheta * ( DG + H_dx )

!  Record the values of the primal and dual variables and multipliers 
!  at the end of the interval

      X = X_int + dtheta * DX_int
      Y = Y_int + dtheta * DY_int
      Z = Z_int + dtheta * DZ_int

!  Print details of the interval if required

!     IF ( printm ) THEN

        WRITE( out, "( /, ' Over interval [', ES10.4, ', ', ES10.4, ']' )" )   &
          theta_l, theta_u
        outf = PQP_pmvalue( f_int, .FALSE. )
        outg = PQP_pmvalue( g_int, .TRUE. )
        outh = PQP_pmvalue( h_int, .TRUE. )
        IF ( g_int /= zero ) THEN
          IF ( h_int /= zero ) THEN
            WRITE( out, "( /, ' f = ', A12, ' ', A12, ' * (theta - ', ES10.4,  &
            &      ')', /, 18X, A12, ' * (theta - ', ES10.4, ') ** 2' )" )     &
              outf, outg, theta_l, outh, theta_l
          ELSE
            WRITE( out, "( /, ' f = ', A12, ' ', A12, ' * (theta - ', ES10.4,  &
            &      ')' )" ) outf, outg, theta_l
          END IF
        ELSE
          IF ( h_int /= zero ) THEN
            WRITE( out, "( /, ' f = ', A12, ' ', A12, ' * (theta - ', ES10.4,  &
            &      ') ** 2 ' )" ) outf, outh, theta_l
          ELSE
            WRITE( out, "( /, ' f = ', A12 )" ) outf
          END IF
        END IF

        WRITE( out, "( /, ' X:      i   value' )" )
        DO i = 1, n
          IF ( DX_int( i ) /= zero ) THEN
            outd = PQP_pmvalue( DX_int( i ), .TRUE. )
            WRITE( out, "( I10, ES12.4, ' ', A12, ' * (theta - ',              &
           &       ES10.4, ') ' )" ) i, X_int( i ), outd, theta_l
          ELSE
            WRITE( out, "( I10, ES12.4 )" ) i, X_int( i )
          END IF
        END DO

        WRITE( out, "( /, ' Y:      i   value' )" )
        DO i = 1, m
          IF ( DY_int( i ) /= zero ) THEN
            outd = PQP_pmvalue( DY_int( i ), .TRUE. )
            WRITE( out, "( I10, ES12.4, ' ', A12, ' * (theta - ',              &
           &       ES10.4, ') ' )" ) i, Y_int( i ), outd, theta_l
          ELSE
            WRITE( out, "( I10, ES12.4 )" ) i, Y_int( i )
          END IF
        END DO

        WRITE( out, "( /, ' Z:      i   value' )" )
        DO i = 1, n
          IF ( DZ_int( i ) /= zero ) THEN
            outd = PQP_pmvalue( DZ_int( i ), .TRUE. )
            WRITE( out, "( I10, ES12.4, ' ', A12, ' * (theta - ',              &
           &       ES10.4, ') ' )" ) i, Z_int( i ), outd, theta_l
          ELSE
            WRITE( out, "( I10, ES12.4 )" ) i, Z_int( i )
          END IF
        END DO
!     END IF

!     theta = theta_max

!  Decide on next action

      IF ( theta < theta_end ) THEN
        IF ( control%each_interval ) THEN
          RETURN
        ELSE
          GO TO 200
        END IF
      ELSE
        IF ( printi .AND. theta_end < theta_max )                              &
          WRITE( out, "( I9, 2ES12.4, 1X, A11 )" )                             &
            interval + 1, theta_end, theta_max, 'no solution'

        action = "end"
      END IF

!  Re-entry

      RETURN  

!  Non-executable statements

 2000 FORMAT( /, ' interval   theta_l     theta_u      action ' )
 2010 FORMAT( 1X, A3, I8, ES12.4 )
 2020 FORMAT( /, '  on exit from SCU_append,   status = ', I3 )
 2030 FORMAT( /, '  on exit from SCU_delete,   status = ', I3 )
 2062 FORMAT( I7, 54X, A9 )
 2140 FORMAT( /, ' =+=> Inertia should be (', I3, ',', I3, ',  0)',            &
                              ' but is    (', I3, ',', I3, ',  0)' )
!  End of PQP_solve_main

      END SUBROUTINE PQP_solve_main

      CHARACTER ( LEN = 12 ) FUNCTION PQP_pmvalue( value, leading )
      REAL ( KIND = wp ), INTENT( IN ) :: value
      LOGICAL, INTENT( IN ) :: leading
      CHARACTER ( LEN = 12 ) :: pmvalue
      IF ( value >= zero ) THEN
        IF ( leading ) THEN
          WRITE( pmvalue, "( A2, ES10.4 )" ) '+ ', value
        ELSE
          WRITE( pmvalue, "( A2, ES10.4 )" ) '  ', value
        END IF
      ELSE
        WRITE( pmvalue, "( A2, ES10.4 )" ) '- ', - value
      END IF
      PQP_pmvalue = pmvalue
      END FUNCTION PQP_pmvalue

!  End of module PQP_double

   END MODULE GALAHAD_PQP_double



