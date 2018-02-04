! THIS VERSION: GALAHAD 2.4 - 3/05/2010 AT 12:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ L C F    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.0.July 20th 2006

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LCF_double

!     ----------------------------------------------
!     |                                            |
!     | Find a feasible point within the polytope  |
!     |                                            |
!     |          c_l <= A x <= c_u                 |
!     |          x_l <=  x <= x_u                  |
!     |                                            |
!     | using either the Simulateneous or the      |
!     ! Successive Orthogonal Projection methods   |
!     |                                            |
!     ----------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPP_double, LCF_dims_type => QPP_dims_type
      USE GALAHAD_QPD_double, LCF_data_type => QPD_data_type,                  &
                              LCF_AX => QPD_AX
      USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                      &
         SORT_heapsort_smallest, SORT_inverse_permute
      USE GALAHAD_ROOTS_double
!     USE GALAHAD_LSQP_double, ONLY: 
      USE GALAHAD_FDC_double
      USE GALAHAD_STRING_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LCF_initialize, LCF_read_specfile, LCF_solve,                  &
                LCF_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get,   &
                LCF_Ax, LCF_data_type, LCF_dims_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: tenm1 = ten ** ( - 1 )
      REAL ( KIND = wp ), PARAMETER :: tenm2 = ten ** ( - 2 )
      REAL ( KIND = wp ), PARAMETER :: tenm3 = ten ** ( - 3 )
      REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: tenm5 = ten ** ( - 5 )
      REAL ( KIND = wp ), PARAMETER :: tenm7 = ten ** ( - 7 )
      REAL ( KIND = wp ), PARAMETER :: tenm8 = ten ** ( - 8 )
      REAL ( KIND = wp ), PARAMETER :: tenm10 = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: ten4 = ten ** 4
      REAL ( KIND = wp ), PARAMETER :: ten5 = ten ** 5
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: gzero = ten ** ( - 14 )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: eps_nz = 0.01_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: LCF_control_type
        INTEGER :: error, out, print_level, start_print, stop_print, print_gap
        INTEGER :: maxit, initial_point
        INTEGER :: factor, max_col, indmin, valmin, itref_max, infeas_max
        INTEGER :: restore_problem, step_strategy
        REAL ( KIND = wp ) :: infinity, stop_p, stop_d, stop_c, prfeas, dufeas
        REAL ( KIND = wp ) :: reduce_infeas, weight_bound_projection, step
        REAL ( KIND = wp ) :: pivot_tol, pivot_tol_for_dependencies, zero_pivot
        REAL ( KIND = wp ) :: identical_bounds_tol

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

        LOGICAL :: remove_dependencies, treat_zero_bounds_as_general
        LOGICAL :: just_feasible, feasol, balance_initial_complentarity
        LOGICAL :: use_simultaneous_projection, project_onto_bounds_first
        LOGICAL :: space_critical, deallocate_error_fatal
        CHARACTER ( LEN = 30 ) :: prefix
        TYPE ( SBLS_control_type ) :: SBLS_control
      END TYPE

      TYPE, PUBLIC :: LCF_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = wp ) :: find_dependent = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ) :: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = wp ) :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

        REAL ( KIND = wp ) :: clock_preprocess = 0.0

!  the clock time spent detecting linear dependencies

        REAL ( KIND = wp ) :: clock_find_dependent = 0.0

!  the clock time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

        REAL ( KIND = wp ) :: clock_solve = 0.0
      END TYPE

      TYPE, PUBLIC :: LCF_inform_type
        INTEGER :: status, alloc_status, iter, factorization_status, num_infeas
        INTEGER ( KIND = long ) :: factorization_integer, factorization_real
        INTEGER :: iterp1, iterp01, iterp001, iterp0001, nfacts
        REAL ( KIND = wp ) :: obj, size_b, size_l
        REAL ( KIND = wp ) :: non_negligible_pivot
        LOGICAL :: feasible
        CHARACTER ( LEN = 80 ) :: bad_alloc
        TYPE ( LCF_time_type ) :: time
        TYPE ( SBLS_inform_type ) :: SBLS_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   L C F _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LCF_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LCF. This routine should be called before
!  LCF_primal_dual
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
!   print_gap. Printing will only occur every print_gap iterations
!
!   initial_point. How to choose the initial point.
!     Possible values are
!
!      0  the values input in X, shifted to be at least prfeas from
!         their nearest bound, will be used
!      1  the nearest point to the "bound average" 0.5(X_l+X_u) that satisfies
!          the linear constraints will be used
!
!   factor. The factorization to be used.
!     Possible values are
!
!      0  automatic 
!      1  Schur-complement factorization
!      2  augmented-system factorization
!
!   max_col. The maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization
!
!   indmin. An initial guess as to the integer workspace required by SBLS
!
!   valmin. An initial guess as to the real workspace required by SBLS
! 
!   itref_max. The maximum number of iterative refinements allowed
!
!   infeas_max. The number of iterations for which the overall infeasibility
!     of the problem is not reduced by at least a factor control%reduce_infeas
!     before the problem is flagged as infeasible (see reduce_infeas)
!
!   restore_problem. Indicates whether and how much of the input problem
!    should be restored on output. Possible values are
!
!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters
!
!   step strategy. Indicates the stepsize selection strategy used.
!     Possible values are
!
!      0 step of 1
!      1 reduce infeasibility by 1
!      2 minimize infeasibility
!      4 Bauschke-Combettes-Krug extrapolation

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
!   reduce_infeas. If the overall infeasibility of the problem is not reduced 
!    by at least a factor reduce_infeas over control%infeas_max iterations,
!    the problem is flagged as infeasible (see infeas_max)
!
!   pivot_tol. The threshold pivot used by the matrix factorization.
!    See the documentation for SBLS for details
!
!   pivot_tol_for_dependencies. The threshold pivot used by the matrix 
!    factorization when attempting to detect linearly dependent constraints.
!    See the documentation for SBLS for details
!
!   zero_pivot. Any pivots smaller than zero_pivot in absolute value will 
!    be regarded to be zero when attempting to detect linearly dependent 
!    constraints
!
!   identical_bounds_tol. Any pair of constraint bounds (c_l,c_u) or (x_l,x_u)
!    that are closer than identical_bounds_tol will be reset to the average
!    of their values
!
!   weight_bound_projection, The weight assigned to the projection for the
!     bound constraints in the Block Iterative Projection method
!
!   step. The step size for the Block Iterative Projection method
!      
!  LOGICAL control parameters:
!
!   use_simultaneous_projection. If true, use the simulatneous orthogonal
!    projection method, otherwise use successive orthogonal projection
!
!   project_onto_bounds_first. If true, the first projection will be
!    onto the bound constraints, not onto the equality manifold

!   remove_dependencies. If true, the equality constraints will be preprocessed
!    to remove any linear dependencies
!
!   treat_zero_bounds_as_general. If true, any problem bound with the value
!    zero will be treated as if it were a general value
!
!   just_feasible. If just_feasible is .TRUE., the algorithm will stop as
!    soon as a feasible interior point is found. Otherwise, a well-centered
!    interior point will be sought
!
!   feasol. If feasol is true, the final solution obtained will be perturbed 
!    so that variables close to their bounds are moved onto these bounds
!
!   balance_initial_complentarity is .true. if the initial complemetarity
!    is required to be balanced
!
!   space_critical. If true, every effort will be made to use as little
!     space as possible. This may result in longer computation times
!
!   deallocate_error_fatal. If true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue
!
!  CHARACTER control parameters:
!
!  prefix (len=30). All output lines will be prefixed by 
!    %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( LCF_data_type ), INTENT( INOUT ) :: data
      TYPE ( LCF_control_type ), INTENT( OUT ) :: control
      TYPE ( LCF_inform_type ), INTENT( OUT ) :: inform     

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      control%SBLS_control%preconditioner = 2

      inform%status = GALAHAD_ok

!  Set control parameters

!  Integer parameters

      control%error = 6
      control%out = 6
      control%print_level = 0
      control%maxit = 1000
      control%start_print = - 1
      control%stop_print = - 1
      control%print_gap = 1
      control%initial_point = 1
      control%factor = 0
      control%max_col = 35
      control%indmin = 1000
      control%valmin = 1000
      control%itref_max = 1
      control%infeas_max = 200
      control%restore_problem = 2
      control%step_strategy = 1

!  Real parameters

      control%infinity = ten ** 19
      control%stop_p = epsmch ** 0.33
      control%stop_c = epsmch ** 0.33
      control%stop_d = epsmch ** 0.33
      control%prfeas = ten ** 3
      control%dufeas = ten ** 3
      control%pivot_tol = epsmch ** 0.75
      control%pivot_tol_for_dependencies = half
      control%zero_pivot = epsmch ** 0.75
      control%identical_bounds_tol = epsmch
      control%reduce_infeas = one - point01
      control%weight_bound_projection = half
      control%step = one
      control%cpu_time_limit = - one
      control%clock_time_limit = - one

!  Logical parameters

      control%use_simultaneous_projection = .TRUE.
      control%remove_dependencies = .TRUE.
      control%treat_zero_bounds_as_general = .FALSE.
      control%just_feasible = .FALSE.
!     control%feasol = .TRUE.
      control%feasol = .FALSE.
      control%balance_initial_complentarity = .FALSE.
      control%space_critical = .FALSE.
      control%deallocate_error_fatal  = .FALSE.
      control%project_onto_bounds_first = .TRUE.

!  Character parameters

      control%prefix = '""                            '

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN  

!  End of LCF_initialize

      END SUBROUTINE LCF_initialize

!-*-*-*-*-   L C F _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LCF_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LCF_initialize could (roughly) 
!  have been set as:

! BEGIN LCF SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-number-of-iterations                      1000
!  start-print                                       -1
!  stop-print                                        -1
!  iterations-between-printing                       1
!  initial-point-used                                1
!  factorization-used                                0
!  maximum-column-nonzeros-in-schur-complement       35
!  initial-integer-workspace                         1000
!  initial-real-workspace                            1000
!  maximum-refinements                               1
!  maximum-poor-iterations-before-infeasible         200
!  restore-problem-on-output                         0
!  step-strategy-used                                2
!  infinity-value                                    1.0D+19
!  primal-accuracy-required                          1.0D-5
!  dual-accuracy-required                            1.0D-5
!  complementary-slackness-accuracy-required         1.0D-5
!  mininum-initial-primal-feasibility                1000.0
!  mininum-initial-dual-feasibility                  1000.0
!  poor-iteration-tolerance                          0.98
!  pivot-tolerance-used                              1.0D-12
!  pivot-tolerance-used-for-dependencies             0.5
!  zero-pivot-tolerance                              1.0D-12
!  identical-bounds-tolerance                        1.0D-15
!  weight-for-bound-projection                       0.5
!  stepsize                                          1.0
!  maximum-cpu-time-limit                            -1.0
!  use-simultaneous-orthogonal-projection            T
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  just-find-feasible-point                          F
!  balance-initial-complentarity                     F
!  move-final-solution-onto-bound                    F
!  space-critical                                    F
!  record-x-status                                   T
!  record-c-status                                   T
!  deallocate-error-fatal                            F
! END LCF SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( LCF_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 47
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'LCF'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec( : )%keyword = ''

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level' 
      spec(  4 )%keyword = 'maximum-number-of-iterations'
      spec(  5 )%keyword = 'start-print'
      spec(  6 )%keyword = 'stop-print'
      spec( 22 )%keyword = 'iterations-between-printing'
      spec( 32 )%keyword = 'initial-point-used'
      spec(  7 )%keyword = 'factorization-used'
      spec(  8 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec(  9 )%keyword = 'initial-integer-workspace'
      spec( 10 )%keyword = 'initial-real-workspace'
      spec( 11 )%keyword = 'maximum-refinements'
      spec( 12 )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( 13 )%keyword = 'restore-problem-on-output'
      spec( 20 )%keyword = 'step-strategy-used'

!  Real key-words

      spec( 14 )%keyword = 'infinity-value'
      spec( 15 )%keyword = 'primal-accuracy-required'
      spec( 16 )%keyword = 'dual-accuracy-required'
      spec( 17 )%keyword = 'complementary-slackness-accuracy-required'
      spec( 18 )%keyword = 'mininum-initial-primal-feasibility'
      spec( 19 )%keyword = 'mininum-initial-dual-feasibility'
      spec( 21 )%keyword = 'poor-iteration-tolerance'
      spec( 23 )%keyword = 'pivot-tolerance-used'
      spec( 24 )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( 25 )%keyword = 'zero-pivot-tolerance'
      spec( 26 )%keyword = 'identical-bounds-tolerance'
      spec( 34 )%keyword = 'weight-for-bound-projection'
      spec( 35 )%keyword = 'stepsize'
      spec( 39 )%keyword = 'maximum-cpu-time-limit'
      spec( 30 )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( 38 )%keyword = 'use-simultaneous-orthogonal-projection'
      spec( 27 )%keyword = 'remove-linear-dependencies'
      spec( 28 )%keyword = 'treat-zero-bounds-as-general'
      spec( 29 )%keyword = 'just-find-feasible-point'
      spec( 31 )%keyword = 'move-final-solution-onto-bound'
      spec( 33 )%keyword = 'balance-initial-complentarity'
      spec( 36 )%keyword = 'space-critical'
      spec( 37 )%keyword = 'deallocate-error-fatal'
      spec( 44 )%keyword = 'record-x-status'
      spec( 45 )%keyword = 'record-c-status'
      spec( 42 )%keyword = 'project-onto-bounds-first'
      spec( 47 )%keyword = ''

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
      CALL SPECFILE_assign_integer( spec( 22 ), control%print_gap,             &
                                    control%error )                
      CALL SPECFILE_assign_integer( spec( 32 ), control%initial_point,         &
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
      CALL SPECFILE_assign_integer( spec( 13 ), control%restore_problem,       &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 20 ), control%step_strategy,         &
                                    control%error )

!  Set real values


      CALL SPECFILE_assign_real( spec( 14 ), control%infinity,                 &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 15 ), control%stop_p,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 16 ), control%stop_d,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 17 ), control%stop_c,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 18 ), control%prfeas,                   &
                                 control%error )     
      CALL SPECFILE_assign_real( spec( 19 ), control%dufeas,                   &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 21 ), control%reduce_infeas,            &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 23 ), control%pivot_tol,                &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 24 ),                                   &
                                 control%pivot_tol_for_dependencies,           &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 25 ), control%zero_pivot,               &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 26 ), control%identical_bounds_tol,     &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 34 ), control%weight_bound_projection,  &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 35 ), control%step,                     &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 39 ), control%cpu_time_limit,           &
                                 control%error )
      CALL SPECFILE_assign_real( spec( 30 ), control%clock_time_limit,         &
                                 control%error )

!  Set logical values


      CALL SPECFILE_assign_logical( spec( 38 ),                                &
                                    control%use_simultaneous_projection,       &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 27 ), control%remove_dependencies,   &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 28 ),                                &
                                    control%treat_zero_bounds_as_general,      &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 29 ), control%just_feasible,         &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 31 ), control%feasol,                &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 33 ),                                &
                                    control%balance_initial_complentarity,     &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 36 ), control%space_critical,        &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 37 ),                                &
                                    control%deallocate_error_fatal,            &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( 42 ),                                &
                                    control%project_onto_bounds_first,         &
                                    control%error )

!  Read the controls for the factorization

!  Read the specfile for SBLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-SBLS')
      ELSE
        CALL SBLS_read_specfile( control%SBLS_control, device )
      END IF

      RETURN

      END SUBROUTINE LCF_read_specfile

!-*-*-*-*-*-*-*-*-*-   L C F _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE LCF_solve( prob, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Finds a feasible point for the system
!
!               (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!    and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A is sparse.
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
!    to be solved since the last call to LCF_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
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
!    to estimaes of the solution, x. On successful exit, it will contain 
!    the required solution, x.
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
!  data is a structure of type LCF_data_type which holds private internal data
!
!  control is a structure of type LCF_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LCF_initialize. See LCF_initialize 
!   for details
!
!  inform is a structure of type LCF_inform_type that provides 
!    information on exit from LCF_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!   - 1 one of the restrictions 
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -2 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -4 The analytic center appears to be unbounded.
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
!   -11 Too much CPU time has passed. This may happen if control%cpu_time_limit 
!       is too small, but may also be symptomatic of a badly scaled problem.
!
!  On exit from LCF_solve, other components of inform give the 
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation 
!     iter   = The total number of iterations required.
!     factorization_integer = The total integer workspace required by the 
!              factorization.
!     factorization_real = The total real workspace required by the 
!              factorization.
!     nfacts = The total number of factorizations performed.
!     factorization_status = the return status from the matrix factorization
!              package.   
!     obj = the value of the objective function ||W*(x-x^0)||_2.
!     non_negligible_pivot = the smallest pivot which was not judged to be
!       zero when detecting linearly dependent constraints
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package.
!     time%preprocess = the time spent preprocessing the problem.
!     time%find_dependent = the time spent detecting linear dependencies
!     time%analyse = the time spent analysing the required matrices prior to
!                  factorization.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LCF_data_type ), INTENT( INOUT ) :: data
      TYPE ( LCF_control_type ), INTENT( INOUT ) :: control
      TYPE ( LCF_inform_type ), INTENT( OUT ) :: inform

!  Local variables

      INTEGER :: i, j, a_ne, n_depen, nzc
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: av_bnd
      LOGICAL :: printi, remap_freed, reset_bnd
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( FDC_data_type ) :: FDC_data
      TYPE ( FDC_control_type ) :: FDC_control        
      TYPE ( FDC_inform_type ) :: FDC_inform

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering LCF_solve ' )" )

!  Initialize time

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

      inform%size_l = biginf
      inform%size_b = biginf
      inform%num_infeas = - 1

!  Initialize counts

      inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%iter = - 1 ; inform%nfacts = - 1
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1 
      inform%obj = - one ; inform%non_negligible_pivot = zero
      inform%feasible = .FALSE. ; inform%factorization_status = 0

!  Basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1 

!  Ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) inform%status 
        GO TO 800 
      END IF 
      prob%Hessian_kind = 0

!  If required, write out problem 

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', 2I8 )" ) prob%n, prob%m
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
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne)
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
          GO TO 800 
        ELSE IF ( prob%X_u( i ) == prob%X_l( i )  ) THEN
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
        data%QPP_control%treat_zero_bounds_as_general =                        &
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

!  Perform the preprocessing. 

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
  
!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= 0 ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( ' status ', I3, ' after QPP_reorder ')" )   &
             data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status 
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
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map, data%QPP_inform,                       &
                          prob, get_f = .TRUE., get_g = .TRUE.,                &
                          get_x_bounds = .TRUE., get_c_bounds = .TRUE.,        &
                          get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,      &
                          get_c = .TRUE., get_A = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record

!  Test for satisfactory termination

          IF ( data%QPP_inform%status /= 0 ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( ' status ', I3, ' after QPP_apply ')" )   &
               data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) inform%status 
            GO TO 800 
          END IF 
        END IF 
        data%trans = data%trans + 1
      END IF

!  =================================================================
!  Check to see if the equality constraints are linearly independent
!  =================================================================

      IF ( .NOT. data%tried_to_remove_deps .AND. control%remove_dependencies ) &
        THEN

        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out,                                                  &
            "( /, 1X, I0, ' equalities from ', I0, ' constraints ' )" )        &
            data%dims%c_equality, prob%m

!  Set control parameters

        CALL FDC_initialize( FDC_data, FDC_control, FDC_inform )
        FDC_control%error = 6
        FDC_control%out = 6
        FDC_control%zero_pivot = control%zero_pivot
        FDC_control%pivot_tol = control%pivot_tol_for_dependencies
        FDC_control%max_infeas = control%stop_p

!  Find any dependent rows

        nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1 
        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record ) 
        CALL FDC_find_dependent( prob%n, data%dims%c_equality,                 &
                                 prob%A%val( : nzc ),                          &
                                 prob%A%col( : nzc ),                          &
                                 prob%A%ptr( : data%dims%c_equality + 1 ),     &
                                 prob%C_l, n_depen, data%Index_C_freed,        &
                                 FDC_data, FDC_control, FDC_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%find_dependent =                                           &
          inform%time%find_dependent + time_now - time_record
        inform%time%clock_find_dependent =                                     &
          inform%time%clock_find_dependent + clock_now - clock_record

!  Record output parameters

        inform%status = FDC_inform%status
        inform%non_negligible_pivot = FDC_inform%non_negligible_pivot
        inform%alloc_status = FDC_inform%alloc_status
        inform%factorization_status = FDC_inform%factorization_status
        inform%factorization_integer = FDC_inform%factorization_integer
        inform%factorization_real = FDC_inform%factorization_real
        inform%bad_alloc = FDC_inform%bad_alloc
        inform%nfacts = 1

        CALL FDC_terminate( FDC_data, FDC_control, FDC_inform )

        CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now ) 
        IF ( ( control%cpu_time_limit >= zero .AND.                            &
               time_now - time_start > control%cpu_time_limit ) .OR.           &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - clock_start > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status 
          GO TO 800 
        END IF 

        IF ( printi .AND. inform%non_negligible_pivot <                        &
             thousand * control%zero_pivot ) WRITE( control%out, "(            &
       &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /,                       &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, ' ***  perhaps increase control%zero_pivot from', ES11.4,'  ***', &
       &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ) )" )                      &
           inform%non_negligible_pivot, control%zero_pivot

!  Check for error exits

        IF ( inform%status /= 0 ) THEN

!  Print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= 0 )                                          &
              WRITE( control%error, 2020 ) inform%status, 'LCF_dependent'
          END IF
          GO TO 700
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
          WRITE( control%out, "(/, ' The following ',I0,' constraints appear', &
       &         ' to be dependent', /, ( 8I8 ) )" ) n_depen, data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  Special case: no free variables

        IF ( prob%n == 0 ) THEN
          prob%Y( : prob%m ) = zero
          prob%Z( : prob%n ) = zero
          prob%C( : prob%m ) = zero
          CALL LCF_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
                       prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,               &
                       prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          GO TO 700
        END IF
        data%tried_to_remove_deps = .TRUE.
      ELSE
        remap_freed = .FALSE.
      END IF

      IF ( remap_freed ) THEN

!  Some of the current constraints will be removed by freeing them

        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE( control%out, "( /, ' -> ', I0, ' constraints are',            &
         & ' dependent and will be temporarily removed' )" ) n_depen

!  Allocate arrays to indicate which constraints have been freed

        array_name = 'lcf: data%C_freed'
        CALL SPACE_resize_array( n_depen, data%C_freed, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 900
        
!  Free the constraint bounds as required

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          data%C_freed( i ) = prob%C_l( j )
          prob%C_l( j ) = - control%infinity
          prob%C_u( j ) = control%infinity
          prob%Y( j ) = zero
        END DO

        CALL QPP_initialize( data%QPP_map_freed, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

!  Store the problem dimensions

        data%dims_save_freed = data%dims
        a_ne = prob%A%ne 

        IF ( printi ) WRITE( control%out,                                      &
               "( /, ' problem dimensions before removal of dependecies: ', /, &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )                   &
               prob%n, prob%m, a_ne

!  Perform the preprocessing

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
  
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

!  Compute the dimension of the KKT system 

      data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1 

!  Arrays containing data relating to the composite vector ( x  c  y )
!  are partitioned as follows:

!   <---------- n --------->  <---- nc ------>  <-------- m --------->
!                             <-------- m --------->
!                        <-------- m --------->
!   -------------------------------------------------------------------
!   |                   |    |                 |    |                 |
!   |         x              |       c         |          y           |
!   |                   |    |                 |    |                 |
!   -------------------------------------------------------------------
!    ^                 ^    ^ ^               ^ ^    ^               ^
!    |                 |    | |               | |    |               |
!   x_s                |    |c_s              |y_s  y_i             y_e = v_e
!                      |    |                 |                     
!                     c_b  x_e               c_e

      data%dims%x_s = 1 ; data%dims%x_e = prob%n
      data%dims%c_s = data%dims%x_e + 1 
      data%dims%c_e = data%dims%x_e + data%dims%nc  
      data%dims%c_b = data%dims%c_e - prob%m
      data%dims%y_s = data%dims%c_e + 1 
      data%dims%y_e = data%dims%c_e + prob%m
      data%dims%y_i = data%dims%c_s + prob%m 
      data%dims%v_e = data%dims%y_e

!  Allocate real workspace

      array_name = 'lcf: data%SOL'
      CALL SPACE_resize_array( data%dims%v_e, data%SOL, inform%status,         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

      array_name = 'lcf: data%C'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%C, inform%status,                                            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 900

!  =================
!  Solve the problem
!  =================

      IF ( printi ) WRITE( control%out,                                        &
           "( /, ' <------ variable bounds ------>',                           &
        &        ' <----- constraint bounds ----->',                           &
        &     /, '    free   below    both   above',                           &
        &        '   equal   below    both   above',                           &
        &     /,  8I8 )" )                                                     &
            data%dims%x_free, data%dims%x_u_start - data%dims%x_free - 1,      &
            data%dims%x_l_end - data%dims%x_u_start + 1,                       &
            prob%n - data%dims%x_l_end, data%dims%c_equality,                  &
            data%dims%c_u_start - data%dims%c_equality - 1,                    &
            data%dims%c_l_end - data%dims%c_u_start + 1,                       &
            prob%m - data%dims%c_l_end

      CALL LCF_solve_main( data%dims, prob%n, prob%m,                          &
                           prob%A%val, prob%A%col, prob%A%ptr,                 &
                           prob%C_l, prob%C_u, prob%X_l, prob%X_u,             &
                           prob%C, prob%X, data%SOL,                           &
                           data%C, data%H_sbls, data%A_sbls, data%C_sbls,      &
                           data%SBLS_data, time_start, clock_start,            &
                           control, inform )

!  If some of the constraints were freed during the computation, refix them now

      IF ( remap_freed ) THEN

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform, prob,           &
                          get_all = .TRUE. )
        CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,              &
                            data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          prob%C_l( j ) = data%C_freed( i )
          prob%C_u( j ) = data%C_freed( i )
        END DO
      END IF
      data%tried_to_remove_deps = .FALSE.

!  Retore the problem to its original form

  700 CONTINUE 
      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )

!  Full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform,                     &
                            prob, get_f = .TRUE., get_g = .TRUE.,              &
                            get_x_bounds = .TRUE., get_c_bounds = .TRUE.,      &
                            get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,    &
                            get_c = .TRUE., get_A = .TRUE., get_H = .TRUE. )

!  Restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  Recover solution

        ELSE
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_x = .TRUE., get_y = .TRUE.,                    &
                            get_z = .TRUE., get_c = .TRUE. )
        END IF
        CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now ) 
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  Compute total time

  800 CONTINUE 
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start 
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start 

      IF ( printi ) WRITE( control%out, 2000 ) inform%time%total,              &
        inform%time%preprocess, inform%time%analyse, inform%time%factorize,    &
        inform%time%solve

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving LCF_solve ' )" )

      RETURN  

!  Allocation error

  900 CONTINUE 
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start 
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start 
!     IF ( printi ) WRITE( control%out, 2900 ) bad_alloc, inform%alloc_status
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving LCF_solve ' )" )

      RETURN  

!  Non-executable statements

 2000 FORMAT( /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',&
              /, 14X, ' =                  LCF total time                   =',&
              /, 14X, ' =', 16X, 0P, F12.2, 23x, '='                           &
              /, 14X, ' =    preprocess    analyse    factorize     solve   =',&
              /, 14X, ' =', 4F12.2, 3x, '=',                                   &
              /, 14X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=' )
 2010 FORMAT( ' ', /, '   **  Error return ',I3,' from LCF ' ) 
 2020 FORMAT( '   **  Error return ', I6, ' from ', A15 ) 

!  End of LCF_solve

      END SUBROUTINE LCF_solve

!-*-*-*-*-*-   L C F _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LCF_solve_main( dims, n, m, A_val, A_col, A_ptr,              &
                                 C_l, C_u, X_l, X_u, C_RES, X,                 &
                                 SOL, C, H_sbls, A_sbls, C_sbls,               &
                                 SBLS_data, time_start, clock_start,           &
                                 control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Finds a feasible point within the polytope
!
!               (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!    and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A is sparse.
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
!  dims is a structure of type LCF_data_type, whose components hold SCALAR
!   information about the problem on input. The components will be unaltered
!   on exit. The following components must be set:
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
!  n, m, ..., X exactly as for prob% in LCF_solve
!  
!  The remaining arguments are used as internal workspace, and need not be 
!  set on entry
!  
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LCF_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      REAL, INTENT( IN ) :: time_start
      REAL ( KIND = wp ), INTENT( IN ) :: clock_start
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: C_RES
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( dims%v_e ) :: SOL
      REAL ( KIND = wp ), INTENT( OUT ),                                       &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C

!  Allocatable arrays and structures

      TYPE ( SMT_type ), INTENT( INOUT ) :: H_sbls, A_sbls, C_sbls
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: SBLS_data
      TYPE ( LCF_control_type ), INTENT( INOUT ) :: control        
      TYPE ( LCF_inform_type ), INTENT( INOUT ) :: inform

!  Parameters

      REAL ( KIND = wp ), PARAMETER :: eta = tenm4
      REAL ( KIND = wp ), PARAMETER :: sigma_max = point01
      REAL ( KIND = wp ), PARAMETER :: degen_tol = tenm5

!  Local variables

      INTEGER :: A_ne, i, j, l, start_print, stop_print
      INTEGER :: out, error, factor, ns, n_slack, lbreak, print_gap
      REAL :: time_record, time_now, time_solve
      REAL ( KIND = wp ) :: clock_record, clock_now, clock_solve
      REAL ( KIND = wp ) :: amax, size_r, q_0, q_1, q_2, one_minus_alpha
      REAL ( KIND = wp ) :: lambda_b, lambda_l, alpha, one_minus_eps_nz
      REAL ( KIND = wp ) :: size_b2, size_l2, size_p
!     REAL ( KIND = wp ) :: pl_x_sqr, pl_c_sqr, gmax
      LOGICAL :: set_printt, set_printi, set_printw, set_printd, set_printe
      LOGICAL :: printt, printi, printe, printd, printw
      LOGICAL :: set_eq
      INTEGER :: sif = 50
!     LOGICAL :: generate_sif = .TRUE.
      LOGICAL :: generate_sif = .FALSE.
      CHARACTER ( LEN = 80 ) :: array_name

!  Automatic arrays

      INTEGER, DIMENSION( n + dims%nc + dims%c_l_end - dims%c_u_start +        &
                          dims%x_l_end - dims%x_u_start + 2 ) :: IBREAK
      REAL ( KIND = wp ), DIMENSION( n + dims%nc + dims%c_l_end -              &
        dims%c_u_start + dims%x_l_end - dims%x_u_start + 2 ) :: BREAKP
      REAL ( KIND = wp ),                                                      &
        DIMENSION( dims%c_l_start : dims%c_u_end ) :: Pb_c, Pl_c
      REAL ( KIND = wp ),                                                      &
        DIMENSION( dims%c_l_start : dims%c_u_end ) :: C_eq, Z_c, PZ_c
      REAL ( KIND = wp ), DIMENSION( n ) :: Pb_x, Pl_x, X_eq, Z_x, PZ_x

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' entering LCF_solve_main ' )" )

      lbreak = n + dims%nc + dims%c_l_end - dims%c_u_start +                   &
         dims%x_l_end - dims%x_u_start + 2

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed 

      IF ( generate_sif ) THEN
        WRITE( sif, "( 'NAME          LCF_OUT', //, 'VARIABLES', / )" )
        DO i = 1, n
          WRITE( sif, "( '    X', I8 )" ) i
        END DO

        WRITE( sif, "( /, 'GROUPS', / )" )
        DO i = 1, dims%c_l_start - 1
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( l ), A_val( l )
          END DO
        END DO
        DO i = dims%c_l_start, dims%c_l_end
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' G  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( l ), A_val( l )
          END DO
        END DO
        DO i = dims%c_l_end + 1, dims%c_u_end
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            WRITE( sif, "( ' L  C', I8, ' X', I8, ' ', ES12.5 )" )             &
              i, A_col( l ), A_val( l )
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
            WRITE( sif, "( '    RANGE    ', ' C', I8, ' ', ES12.5 )" )         &
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

!  ===========================
!  Control the output printing
!  ===========================

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

      error = control%error ; out = control%out 

      set_printe = error > 0 .AND. control%print_level >= 1

!  Basic single line of output per iteration

      set_printi = out > 0 .AND. control%print_level >= 1 

!  As per printi, but with additional timings for various operations

      set_printt = out > 0 .AND. control%print_level >= 2 

!  As per printt but also with details of innner iterations

      set_printw = out > 0 .AND. control%print_level >= 4

!  Full debugging printing with significant arrays printed

      set_printd = out > 0 .AND. control%print_level >= 5

!  Start setting control parameters

      IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
        printe = set_printe ; printi = set_printi ; printt = set_printt
        printw = set_printw ; printd = set_printd
      ELSE
        printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
        printw = .FALSE. ; printd = .FALSE.
      END IF

      IF ( control%print_gap < 2 ) THEN
        print_gap = 1
      ELSE
        print_gap = control%print_gap
      END IF

      factor = control%factor
      IF ( factor < 0 .OR. factor > 2 ) THEN
        IF ( printi ) WRITE( out,                                             &
          "( ' factor = ', I6, ' out of range [0,2]. Reset to 0' )" ) factor
        factor = 0
      END IF

!  ==================
!  Input error checks
!  ==================

!  If there are no variables, exit

      IF ( n == 0 ) THEN 
        i = COUNT( ABS( C_l( : dims%c_equality ) ) > control%stop_p ) +        &
            COUNT( C_l( dims%c_l_start : dims%c_l_end ) > control%stop_p ) +   &
            COUNT( C_u( dims%c_u_start : dims%c_u_end ) < - control%stop_p )
        IF ( i == 0 ) THEN
          inform%status = GALAHAD_ok
        ELSE
          inform%status = GALAHAD_error_primal_infeasible
        END IF
        C_RES = zero
        inform%obj = zero
        GO TO 600
      END IF 

!  Check that range constraints are not simply fixed variables,
!  and that the upper bounds are larger than the corresponing lower bounds

      DO i = dims%x_u_start, dims%x_l_end
        IF ( X_u( i ) - X_l( i ) <= epsmch ) THEN 
          inform%status = GALAHAD_error_bad_bounds ; GO TO 600 ; END IF
      END DO

      DO i = dims%c_u_start, dims%c_l_end
        IF ( C_u( i ) - C_l( i ) <= epsmch ) THEN 
          inform%status = GALAHAD_error_bad_bounds ; GO TO 600 ; END IF
      END DO

!  Record array size

      A_ne = A_ptr( m + 1 ) - 1

!  Compute the initial values of C

      C_RES = zero
      CALL LCF_AX( m, C_RES( : m ), m, A_ne, A_val, A_col, A_ptr, n, X, '+ ' )
      C = C_RES( dims%c_l_start : dims%c_u_end )

!  If required, write out the data matrix for the problem

      IF ( printd ) WRITE( out, 2150 ) ' a ', ( ( i, A_col( l ), A_val( l ),   &
                           l = A_ptr( i ), A_ptr( i + 1 ) - 1 ), i = 1, m )

!  Find the largest components of A

      IF ( A_ne > 0 ) THEN
        amax = MAXVAL( ABS( A_val( : A_ne ) ) )
      ELSE
        amax = zero
      END IF

      IF ( printi ) WRITE( out, "( '  maximum element of A = ', ES12.4 )" ) amax
!     IF ( printi ) WRITE( out, "( '  maximum element of g = ', ES12.4 )" ) gmax

!  ............................................................................
!                            FACTORIZE THE PROJECTOR
!  ............................................................................


!  Provide storage for the 1,2 block, A augmented by its slack variables

      n_slack = dims%c_u_end - dims%c_l_start + 1
      ns = n + n_slack

      A_sbls%m = m
      A_sbls%n = ns
      A_sbls%ne = A_ne + n_slack
      CALL SMT_put( A_sbls%type, 'COORDINATE', inform%alloc_status )

      array_name = 'wcp: data%A_sbls%row'
      CALL SPACE_resize_array( A_sbls%ne, A_sbls%row, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'wcp: data%A_sbls%col'
      CALL SPACE_resize_array( A_sbls%ne, A_sbls%col, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'wcp: data%A_sbls%val'
      CALL SPACE_resize_array( A_sbls%ne, A_sbls%val, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

!  Insert A into A_sbls

      DO i = 1, m
        A_sbls%row( A_ptr( i ) : A_ptr( i + 1 ) - 1 ) = i
      END DO
      A_sbls%col( : A_ne ) = A_col( : A_ne ) 
      A_sbls%val( : A_ne ) = A_val( : A_ne )
      j = 0
      DO i = dims%c_l_start, dims%c_u_end
        j = j + 1
        A_sbls%row( A_ne + j ) = i
        A_sbls%col( A_ne + j ) = n + j
        A_sbls%val( A_ne + j ) = - one
      END DO

!  provide storage for the identity 1,1 block

      H_sbls%n = ns
      array_name = 'wcp: data%H_sbls%val'
      CALL SPACE_resize_array( H_sbls%n, H_sbls%val, inform%status,            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      CALL SMT_put( H_sbls%type, 'DIAGONAL', inform%alloc_status )
      H_sbls%val = one

!  provide storage for the zero 2,2 block

      CALL SMT_put( C_sbls%type, 'ZERO', inform%alloc_status )

!  Factorize the "projector"
!
!      ( H  A^T ) = ( I  A^T )
!      ( A  -C  )   ( A   0  )

      CALL CLOCK_time( clock_record )
      CALL SBLS_form_and_factorize( ns, m, H_sbls, A_sbls, C_sbls,             &
        SBLS_data, control%SBLS_control, inform%SBLS_inform )
      inform%time%analyse = inform%time%analyse +                              &
        inform%SBLS_inform%SLS_inform%time%analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%SBLS_inform%SLS_inform%time%clock_analyse
      inform%time%factorize = inform%time%factorize +                          &
        inform%SBLS_inform%SLS_inform%time%factorize
      inform%time%clock_factorize = inform%time%clock_factorize +              &
        inform%SBLS_inform%SLS_inform%time%clock_factorize
      time_solve = 0.0 ; clock_solve = 0.0

      inform%nfacts = 1
      IF ( inform%SBLS_inform%status < 0 ) THEN
        IF( printe ) WRITE( error,                                             &
          "( ' Error status ', I0, ' from SBLS_form_and_factorize ' )" )       &
          inform%SBLS_inform%status
        inform%status = GALAHAD_error_factorization ; RETURN
      END IF

!  provide storage for the solution to the projection system

!     array_name = 'wcp: data%H_sbls%val'
!     CALL SPACE_resize_array( ns + m, SOL, inform%status,                     &
!            inform%alloc_status, array_name = array_name,                     &
!            deallocate_error_fatal = control%deallocate_error_fatal,          &
!            exact_size = control%space_critical,                              &
!            bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( inform%status /= 0 ) RETURN

! ==============================================================================
!                              MAIN LOOP
! ==============================================================================

      one_minus_eps_nz = one - eps_nz
      inform%iter = 0
      inform%iterp1 = - 1
      inform%iterp01 = - 1
      inform%iterp001 = - 1
      inform%iterp0001 = - 1
      set_eq = .FALSE.
      alpha = zero
!     pl_x_sqr = zero ; pl_c_sqr = zero

      IF ( printi ) THEN
        IF ( control%use_simultaneous_projection ) THEN
          WRITE( out, "(/, '  Simultaneous Orthogonal Projection method used')")
        ELSE
          WRITE( out, "(/, '  Successive Orthogonal Projection method used')" )
        END IF
        WRITE( out, "( '  Stepsize strategy = ', I0 )" ) control%step_strategy
      END IF

! ==============================================================================
!                 SIMULATNEOUS ORTHOGONAL PROJECTION METHOD
! ==============================================================================

      IF ( control%use_simultaneous_projection ) THEN

        DO

!  Compute the orthogonal projection Pb of (x,c) into their bounds

          Pb_x = MIN( X_u, MAX( X_l, X ) )
          Pb_c = MIN( C_u( dims%c_l_start : dims%c_u_end ),                    &
                      MAX( C_l( dims%c_l_start : dims%c_u_end ), C ) )

          IF ( printd ) WRITE( out, "(' Pb_x:', /, ( 5ES12.4) ) " ) Pb_x
          IF ( printd .AND. n_slack > 0 )                                      &
            WRITE( out, "(' Pb_c:', /, ( 5ES12.4 ) )" ) Pb_c

          IF ( control%step_strategy == 0 .OR. control%step_strategy == 4 .OR. &
              inform%iter == 0 ) THEN

!  Compute the orthogonal projection Pb of (x,c) onto the linear constraints

!  compute the rhs for the projection

            SOL( : n ) = X
            SOL( n + 1: ns ) = C
            SOL( ns + 1 : ns + dims%c_equality ) = C_l( : dims%c_equality )
            SOL( ns + dims%c_l_start : ) = zero

!  project onto linear constraints

            CALL SBLS_solve( ns, m, A_sbls, C_sbls, SBLS_data,                 &
                             control%SBLS_control, inform%SBLS_inform, SOL )

            Pl_x = SOL( : n )
            Pl_c = SOL( n + 1: ns )

!           WRITE(6,"( ' error Pl_x, Pc_x ', 2ES22.14 )" )                     &
!             SUM( Pl_x ** 2 ) - pl_x_sqr, SUM( Pl_c ** 2 ) - pl_c_sqr
          END IF

          IF ( printd ) WRITE( out, "(' Pl_x:', /, ( 5ES12.4) ) " ) Pl_x
          IF ( printd .AND. n_slack > 0 )                                      &
            WRITE( out, "(' Pl_c:', /, ( 5ES12.4 ) )" ) Pl_c

!  record the sizes of the projections

          inform%size_b =                                                      &
            SQRT( SUM( ( X - Pb_x ) ** 2 ) + SUM( ( C - Pb_c ) ** 2 ) )
          inform%size_l =                                                      &
            SQRT( SUM( ( X - Pl_x ) ** 2 ) + SUM( ( C - Pl_c ) ** 2 ) )

!  Compute the number of violated bounds

          inform%num_infeas =                                                  &
                  COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < zero )    &
                + COUNT( X_l( dims%x_l_start : dims%x_l_end ) >                &
                         X( dims%x_l_start : dims%x_l_end ) )                  &
                + COUNT( X( dims%x_u_start : dims%x_u_end ) >                  &
                         X_u( dims%x_u_start : dims%x_u_end ) )                &
                + COUNT( X( dims%x_u_end + 1: n ) > zero )                     &
                + COUNT( C_l( dims%c_l_start : dims%c_l_end ) >                &
                         C_RES( dims%c_l_start : dims%c_l_end ) )              &
                + COUNT( C_RES( dims%c_u_start : dims%c_u_end ) >              &
                         C_u( dims%c_u_start : dims%c_u_end ) )

!  Compute the weights for the simultaneous projection

          IF ( control%step_strategy>= 3 ) THEN
            lambda_b = half
          ELSE
            lambda_b = control%weight_bound_projection
            DO
              IF ( lambda_b >= eps_nz .AND. lambda_b <= one_minus_eps_nz ) EXIT
              CALL RANDOM_NUMBER( lambda_b )
            END DO
          END IF
          lambda_l = one - lambda_b

!  Compute the merit function

          inform%obj = lambda_b * inform%size_b ** 2 +                         &
                       lambda_l * inform%size_l ** 2

!  Print a summary of the current iteration

          inform%iter = inform%iter + 1
          CALL CLOCK_TIME( clock_now ) ; clock_now = clock_now - clock_start

          IF ( MOD( inform%iter - MAX( 0, start_print ), print_gap ) == 0 ) THEN
            IF ( printt ) THEN
              WRITE( out, "( /, '    iter       obj      size_b      size_l',  &
             &    '   #infeas  l_b  alp      time')")
            ELSE IF ( printi ) THEN
              IF ( MOD( inform%iter - MAX( 1, start_print ), 50 * print_gap )  &
                   == 0 ) WRITE( out, "(/, '    iter       obj      size_b',   &
           &     '      size_l   #infeas  l_b  alp      time' )" )
            END IF
            IF ( printi ) WRITE( out, "( I8, 3ES12.4, I8, 2F5.2, F10.2 )" )    &
              inform%iter, inform%obj, inform%size_b, inform%size_l,           &
              inform%num_infeas, lambda_b, alpha, clock_now
          END IF

!  Check for termination

          size_r = MAX( inform%size_b, inform%size_l )
          IF ( size_r <= tenm4 .AND. inform%iterp0001 == - 1 )                 &
            inform%iterp0001 = inform%iter
          IF ( size_r <= tenm3 .AND. inform%iterp001 == - 1 )                  &
            inform%iterp001 = inform%iter
          IF ( size_r <= tenm2 .AND. inform%iterp01 == - 1 )                   &
            inform%iterp01 = inform%iter
          IF ( size_r <= tenm1 .AND. inform%iterp1 == - 1 )                    &
            inform%iterp1 = inform%iter
          IF ( size_r <= control%stop_p ) EXIT

!  Check that the iteration limit has not been reached

          IF ( inform%iter > control%maxit ) THEN
            inform%status = GALAHAD_error_max_iterations ; GO TO 600
          END IF

!  Check that the CPU time limit has not been reached

          CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now ) 
          IF ( ( control%cpu_time_limit >= zero .AND.                          &
                 time_now - time_start > control%cpu_time_limit ) .OR.         &
               ( control%clock_time_limit >= zero .AND.                        &
                 clock_now - clock_start > control%clock_time_limit ) ) THEN
            inform%status = GALAHAD_error_cpu_limit ; GO TO 600
          END IF 

          IF ( inform%iter >= start_print .AND. inform%iter <= stop_print ) THEN
            printe = set_printe ; printi = set_printi ; printt = set_printt
            printw = set_printw ; printd = set_printd
          ELSE
            printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
            printw = .FALSE. ; printd = .FALSE.
          END IF

!  Compute the new point using Combettes' Extrapolated Parallel Projection 

          IF ( control%step_strategy == 4 ) THEN
            size_p = SUM( ( two * X - Pb_x - Pl_x ) ** 2 ) +                   &
                     SUM( ( two * C - Pb_c - Pl_c ) ** 2 )
            IF ( size_p /= zero ) THEN
              alpha = 3.8_wp *                                                 &
                ( inform%size_b ** 2 + inform%size_l ** 2 ) / size_p
            ELSE
              alpha = 1.9_wp
            END IF

            X = X + alpha * ( half * ( Pl_x + Pb_x ) - X )
            C = C + alpha * ( half * ( Pl_c + Pb_c ) - C )

!  Compute the new point by minimizing along the search direction

          ELSE IF ( control%step_strategy > 0 ) THEN

!  Let (z_x,z_c) = lambda_b * (Pb_x,Pb_c) + lambda_l * (Pl_x,Pl_c). 

            Z_x = lambda_b * Pb_x + lambda_l * Pl_x
            Z_c = lambda_b * Pb_c + lambda_l * Pl_c

            IF ( printd ) WRITE( out, "(' Z_x:', /, ( 5ES12.4) ) " ) Z_x
            IF ( printd .AND. n_slack > 0 )                                    &
              WRITE( out, "(' Z_c:', /, ( 5ES12.4 ) )" ) Z_c

!  Compute the orthogonal projection Pz of (z_x,z_c) onto the linear constraints

!  compute the rhs for the projection

            SOL( : n ) = Z_x
            SOL( n + 1: ns ) = Z_c
            SOL( ns + 1 : ns + dims%c_equality ) = C_l( : dims%c_equality )
            SOL( ns + dims%c_l_start : ) = zero

!  project onto linear constraints

!           control%SBLS_control%out = 6
!           control%SBLS_control%print_level = 2
            CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SBLS_solve( ns, m, A_sbls, C_sbls, SBLS_data,                 &
                             control%SBLS_control, inform%SBLS_inform, SOL )
            CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
            time_solve = time_solve + time_now - time_record
            clock_solve = clock_solve + clock_now - clock_record

            PZ_x = SOL( : n )
            PZ_c = SOL( n + 1: ns )

!  Find the "best" point on the line (x,z) + alpha(z_x-x,z_c-c)

            SOL( : n ) = Pl_x - X
            SOL( n + 1: ns ) = Pl_c - C
            Pb_x = PZ_x - Z_x - SOL( : n )
            Pb_c = PZ_c - Z_c - SOL( n + 1: ns )
            q_0 = DOT_PRODUCT( SOL( : ns ), SOL( : ns ) )
            q_1 = two * ( DOT_PRODUCT( SOL( : n ), Pb_x ) +                    &
                          DOT_PRODUCT( SOL( n + 1 : ns ), Pb_c ) )
            q_2 = DOT_PRODUCT( Pb_x, Pb_x ) + DOT_PRODUCT( Pb_c, Pb_c )
            SOL( : n ) = Z_x - X
            SOL( n + 1: ns ) = Z_c - C

            CALL LCF_linesearch( dims, n, m,                                   &
                                 q_0, q_1, q_2, lambda_l, lambda_b,            &
                                 X, C, X_l, X_u, C_l, C_u,                     &
                                 SOL( : n ), SOL( n + 1 : ns ),                &
                                 IBREAK, BREAKP, lbreak,                       &
                                 out, printt, printw, printd, alpha,           &
                                 epsmch ** 0.5, i )
!           write(6,"(' alpha, inform ', ES12.4, 1X, I0 )" ) alpha, i
            IF ( printt .AND. i < 0 )                                          &
              WRITE( out, "( ' exit status from linesearch ', I0 )" ) i

!  compute (1-alpha) (x,c) + alpha * ( laabda_b * Pb(x,c) + lambda_l * Pl(x,c) )

            IF ( alpha == zero ) alpha = one           
            one_minus_alpha = one - alpha
            X = one_minus_alpha * X + alpha * Z_x
            C = one_minus_alpha * C + alpha * Z_c
            
            Pl_x = one_minus_alpha * Pl_x + alpha * PZ_x
            Pl_c = one_minus_alpha * Pl_c + alpha * PZ_c

!           WRITE(6,"( ' Pl_x, Pc_x ', 2ES22.14 )" )                           &
!             SUM( Pl_x ** 2 ), SUM( Pl_c ** 2 )
!           pl_x_sqr = SUM( Pl_x ** 2 )
!           pl_c_sqr = SUM( Pl_c ** 2 )

          ELSE

!  Compute the stepsize for the simultaneous projection

            alpha = control%step / two
            DO
              IF ( alpha >= eps_nz .AND. alpha <= one_minus_eps_nz ) EXIT
              CALL RANDOM_NUMBER( alpha )
            END DO
            alpha = two * alpha

!  compute (1-alpha) (x,c) + alpha * ( laabda_b * Pb(x,c) + lambda_l * Pl(x,c) )

            one_minus_alpha = one - alpha
            X = one_minus_alpha * X                                            &
                + alpha * ( lambda_b * Pb_x + lambda_l * Pl_x )
            C = one_minus_alpha * C                                            &
                + alpha * ( lambda_b * Pb_c + lambda_l * Pl_c )

          END IF

          IF ( printd ) WRITE( out, "(' X:', /, ( 5ES12.4 ) ) " ) X
          IF ( printd .AND. n_slack > 0 )                                      &
            WRITE( out, "(' C:', /, ( 5ES12.4 ) )" ) C

        END DO

! ==============================================================================
!                 SUCCESSIVE ORTHOGONAL PROJECTION METHOD
! ==============================================================================

      ELSE

        DO

          IF ( printi .AND. MOD( inform%iter - MAX( 1, start_print ),          &
               50 * print_gap ) == 0 )                                         &
            WRITE( out, "(/, '    iter       obj      size_b      size_l   ',  &
           &    '  #infeas alpha       time')")

!  Project onto the feasible region for the bounds

          IF ( control%project_onto_bounds_first .OR.                          &
               control%step_strategy == 4 .OR. inform%iter > 1 ) THEN
            Pb_x = MIN( X_u, MAX( X_l, X ) )
            Pb_c = MIN( C_u( dims%c_l_start : dims%c_u_end ),                  &
                     MAX( C_l( dims%c_l_start : dims%c_u_end ), C ) )

            IF ( control%step_strategy >= 3 ) THEN
              X = Pb_x
              C = Pb_c
            ELSE

!  Compute the stepsize for the sequential projection

              alpha = control%step / two
              DO
                IF ( alpha >= eps_nz .AND. alpha <= one_minus_eps_nz ) EXIT
                CALL RANDOM_NUMBER( alpha )
              END DO
              alpha = two * alpha

!  Step along (x,c) towards ( Pb_x -x, Pb_c - c )

              X = X + alpha * ( Pb_x - X )
              C = C + alpha * ( Pb_c - C )
            END IF
            IF ( printd ) WRITE( out, "(' X:', /, ( 5ES12.4) ) " ) X
            IF ( printd .AND. n_slack > 0 )                                    &
              WRITE( out, "(' C:', /, ( 5ES12.4 ) )" ) C

!  Compute the residual for the equality constraints

            C_RES = zero
            CALL LCF_AX( m, C_RES( : m ), m, A_ne, A_val, A_col, A_ptr, n,     &
                         X, '+ ' )
            inform%size_l =                                                    &
              MAX( MAXVAL( ABS( C_RES( : dims%c_equality ) -                   &
                                C_l( : dims%c_equality ) ) ),                  &
                   MAXVAL( ABS( C_RES( dims%c_l_start : dims%c_u_end ) -       &
                                     C( dims%c_l_start : dims%c_u_end ) ) ) )
          END IF

!  Compute the rhs for the projection

          SOL( : n ) = X
          SOL( n + 1: ns ) = C
          SOL( ns + 1 : ns + dims%c_equality ) = C_l( : dims%c_equality )
          SOL( ns + dims%c_l_start : ) = zero

!  Project onto linear constraints

          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SBLS_solve( ns, m, A_sbls, C_sbls, SBLS_data,                   &
                           control%SBLS_control, inform%SBLS_inform, SOL )
          CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
          time_solve = time_solve + time_now - time_record
          clock_solve = clock_solve + clock_now - clock_record

!  Recover the solution

          IF ( control%step_strategy > 0 .AND. set_eq ) THEN
            SOL( : n ) = SOL( : n ) - X_eq
            SOL( n + 1 : ns ) = SOL( n + 1 : ns ) - C_eq
            IF ( control%step_strategy == 2 ) THEN
              alpha = control%infinity
              DO i = dims%x_free + 1, dims%x_l_end
                IF ( X_eq( i ) > X_l( i ) ) THEN
                  IF ( SOL( i ) > zero )                                       &
                    alpha = MIN( alpha, ( X_eq( i ) - X_l( i ) ) / SOL( i ) )
                ELSE IF ( X_eq( i ) < X_l( i ) ) THEN
                  IF ( SOL( i ) < zero )                                       &
                    alpha = MIN( alpha, ( X_eq( i ) - X_l( i ) ) / SOL( i ) )
                END IF
              END DO
              DO i = dims%x_u_start, n
                IF ( X_eq( i ) < X_u( i ) ) THEN
                  IF ( SOL( i ) < zero )                                       &
                    alpha = MIN( alpha, ( X_eq( i ) - X_u( i ) ) / SOL( i ) )
                ELSE IF ( X_eq( i ) > X_u( i ) ) THEN
                  IF ( SOL( i ) > zero )                                       &
                    alpha = MIN( alpha, ( X_eq( i ) - X_u( i ) ) / SOL( i ) )
                END IF
              END DO

              DO i = dims%c_l_start, dims%c_l_end
                IF ( C_eq( i ) > C_l( i ) ) THEN
                  IF ( SOL( dims%c_b + i ) > zero )  alpha = MIN( alpha,       &
                     ( C_eq( i ) - C_l( i ) ) / SOL( dims%c_b + i ) )
                ELSE IF ( C_eq( i ) < C_l( i ) ) THEN
                  IF ( SOL( dims%c_b + i ) < zero ) alpha = MIN( alpha,        &
                     ( C_eq( i ) - C_l( i ) ) / SOL( dims%c_b + i ) )
                END IF
              END DO
              DO i = dims%c_u_start, dims%c_u_end
                IF ( C_eq( i ) < C_u( i ) ) THEN
                  IF ( SOL( dims%c_b + i ) < zero ) alpha = MIN( alpha,        &
                     ( C_eq( i ) - C_u( i ) ) / SOL( dims%c_b + i ) )
                ELSE IF ( C_eq( i ) > C_u( i ) ) THEN
                  IF ( SOL( dims%c_b + i ) > zero ) alpha = MIN( alpha,        &
                     ( C_eq( i ) - C_u( i ) ) / SOL( dims%c_b + i ) )
                END IF
              END DO
              alpha = MAX( one, alpha )
            ELSE IF ( control%step_strategy == 3 ) THEN
              CALL LCF_linesearch( dims, n, m,                                 &
                                   zero, zero, zero, zero, one,                &
                                   X_eq, C_eq, X_l, X_u, C_l, C_u, SOL(: n),   &
                                   SOL( n + 1 : ns ),                          &
                                   IBREAK, BREAKP, lbreak,                     &
                                   out, printt, printw, printd, alpha,         &
                                   epsmch ** 0.5, i )
              IF ( printt .AND. i < 0 )                                        &
                WRITE( out, "( ' exit status from linesearch ', I0 )" ) i
  !           alpha = MAX( one, alpha )
              IF ( alpha == zero ) alpha = one           
            ELSE IF ( control%step_strategy == 4 ) THEN

!  record the sizes of the projections

              size_b2 = SUM( ( X_eq - X ) ** 2 ) +                             &
                        SUM( ( C_eq - C ) ** 2 )
              size_l2 = SUM( SOL( : n ) ** 2 ) +                               &
                        SUM( SOL( n + 1 : ns ) ** 2 )
              IF ( size_l2 /= zero ) THEN
                alpha = 1.9_wp * ( size_b2 / size_l2 )
              ELSE
                alpha = 1.9_wp
              END IF
            ELSE
              alpha = 1.1_wp
            END IF
            X = X_eq + alpha * SOL( : n )
            C = C_eq + alpha * SOL( n + 1 : ns )
          ELSE
            alpha = one
            X = SOL( : n )
            C = SOL( n + 1 : ns )
          END IF

          X_eq = X
          C_eq = C
          set_eq = .TRUE.

          IF ( printd ) WRITE( out, "(' X:', /, ( 5ES12.4) ) " ) X
          IF ( printd ) WRITE( out, "(' C:', /, ( 5ES12.4 ) )" ) C

!  Compute the residual for the equality constraints (this should be tiny)

          C_RES = zero
          CALL LCF_AX( m, C_RES( : m ), m, A_ne, A_val, A_col, A_ptr, n,       &
                       X, '+ ' )

          inform%size_l =                                                      &
            MAX( MAXVAL( ABS( C_RES( : dims%c_equality ) -                     &
                              C_l( : dims%c_equality ) ) ),                    &
                 MAXVAL( ABS( C_RES( dims%c_l_start : dims%c_u_end ) -         &
                              C( dims%c_l_start : dims%c_u_end ) ) ) )

!  If the residual has grown unacceptably, re-project

          IF ( inform%size_l > ten ** (-10) ) THEN

!  Compute the rhs for the projection

            SOL( : n ) = X
            SOL( n + 1: ns ) = C
            SOL( ns + 1 : ns + dims%c_equality ) = C_l( : dims%c_equality )
            SOL( ns + dims%c_l_start : ) = zero

!  Project onto linear constraints

            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SBLS_solve( ns, m, A_sbls, C_sbls, SBLS_data,                 &
                             control%SBLS_control, inform%SBLS_inform, SOL )
            CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
            time_solve = time_solve + time_now - time_record
            clock_solve = clock_solve + clock_now - clock_record

            X = SOL( : n )
            C = SOL( n + 1 : ns )

            C_RES = zero
            CALL LCF_AX( m, C_RES( : m ), m, A_ne, A_val, A_col, A_ptr, n,     &
                         X, '+ ' )

            inform%size_l =                                                    &
              MAX( MAXVAL( ABS( C_RES( : dims%c_equality ) -                   &
                                C_l( : dims%c_equality ) ) ),                  &
                   MAXVAL( ABS( C_RES( dims%c_l_start : dims%c_u_end ) -       &
                                C( dims%c_l_start : dims%c_u_end ) ) ) )
          END IF

!write(6,*) ' res ', &
!           MAXVAL( ABS( C_RES( : dims%c_equality ) -                     &
!                             C_l( : dims%c_equality ) ) ),                    &
!           MAXVAL( ABS( C_RES( dims%c_l_start : dims%c_u_end ) -         &
!                             C( dims%c_l_start : dims%c_u_end ) ) )

!  Compute the violation of the bounds

          inform%num_infeas =                                                  &
                  COUNT( X( dims%x_free + 1 : dims%x_l_start - 1 ) < zero )    &
                + COUNT( X_l( dims%x_l_start : dims%x_l_end ) >                &
                         X( dims%x_l_start : dims%x_l_end ) )                  &
                + COUNT( X( dims%x_u_start : dims%x_u_end ) >                  &
                         X_u( dims%x_u_start : dims%x_u_end ) )                &
                + COUNT( X( dims%x_u_end + 1: n ) > zero )                     &
                + COUNT( C_l( dims%c_l_start : dims%c_l_end ) >                &
                         C_RES( dims%c_l_start : dims%c_l_end ) )              &
                + COUNT( C_RES( dims%c_u_start : dims%c_u_end ) >              &
                         C_u( dims%c_u_start : dims%c_u_end ) )

          inform%size_b = SQRT(                                                &
             SUM( MAX( zero,                                                   &
                       - X( dims%x_free + 1 : dims%x_l_start - 1 ) ) ** 2 ) +  &
             SUM( MAX( zero, X_l( dims%x_l_start : dims%x_l_end ) -            &
                             X( dims%x_l_start : dims%x_l_end ) ) ** 2 ) +     &
             SUM( MAX( zero, X( dims%x_u_start : dims%x_u_end ) -              &
                             X_u( dims%x_u_start : dims%x_u_end ) ) ** 2 ) +   &
             SUM( MAX( zero, X( dims%x_u_end + 1 : n ) ) ** 2 ) +              &
             SUM( MAX( zero, C_l( dims%c_l_start : dims%c_l_end ) -            &
                             C( dims%c_l_start : dims%c_l_end ) ) ** 2 ) +     &
             SUM( MAX( zero, C( dims%c_u_start : dims%c_u_end ) -              &
                             C_u( dims%c_u_start : dims%c_u_end ) ) ** 2 ) )

         inform%obj = inform%size_b ** 2
!        inform%obj = half * inform%size_b ** 2 +                             &
!                      half * inform%size_l ** 2

          inform%iter = inform%iter + 1
          CALL CLOCK_TIME( clock_now ) ; clock_now = clock_now - clock_start

          IF ( MOD( inform%iter - MAX( 0, start_print ), print_gap ) == 0 ) THEN
            IF ( printt )                                                      &
              WRITE( out, "(/, '    iter       obj      size_b      size_l ',  &
             &    '  #infeas alpha       time')")
            IF ( printi ) THEN
              WRITE( out, "( I8, 3ES12.4, I8, ES9.2, F10.2 )" ) inform%iter,   &
                inform%obj, inform%size_b, inform%size_l, inform%num_infeas,   &
                alpha, clock_now
            END IF
          END IF

!  Check for termination

          size_r = MAX( inform%size_b, inform%size_l )
          IF ( size_r <= tenm4 .AND. inform%iterp0001 == - 1 )                 &
            inform%iterp0001 = inform%iter
          IF ( size_r <= tenm3 .AND. inform%iterp001 == - 1 )                  &
            inform%iterp001 = inform%iter
          IF ( size_r <= tenm2 .AND. inform%iterp01 == - 1 )                   &
            inform%iterp01 = inform%iter
          IF ( size_r <= tenm1 .AND. inform%iterp1 == - 1 )                    &
            inform%iterp1 = inform%iter
          IF ( size_r <= control%stop_p ) EXIT

!  Check that the iteration limit has not been reached

          IF ( inform%iter > control%maxit ) THEN
            inform%status = GALAHAD_error_max_iterations ; GO TO 600
          END IF

!  Check that the CPU time limit has not been reached

          CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now ) 
          IF ( ( control%cpu_time_limit >= zero .AND.                          &
                 time_now - time_start > control%cpu_time_limit ) .OR.         &
               ( control%clock_time_limit >= zero .AND.                        &
                 clock_now - clock_start > control%clock_time_limit ) ) THEN
            inform%status = GALAHAD_error_cpu_limit ; GO TO 600
          END IF 

          IF ( inform%iter >= start_print .AND. inform%iter < stop_print ) THEN
            printe = set_printe ; printi = set_printi ; printt = set_printt
            printw = set_printw ; printd = set_printd
          ELSE
            printe = .FALSE. ; printi = .FALSE. ; printt = .FALSE.
            printw = .FALSE. ; printd = .FALSE.
          END IF
        END DO
      END IF

! ==============================================================================
!                              END OF MAIN
! ==============================================================================

      IF ( printi ) THEN

        IF ( control%use_simultaneous_projection ) THEN
          WRITE( out, "(/, '  Simultaneous Orthogonal Projection method used')")
        ELSE
          WRITE( out, "(/, '  Successive Orthogonal Projection method used' )" )
        END IF
        WRITE( out, "( '  Stepsize strategy = ', I0 )" ) control%step_strategy
        WRITE( out, 2010 ) inform%obj, inform%iter
        WRITE( out, 2110 ) inform%size_l, inform%size_b

        IF ( factor == 0 .OR. factor == 1 ) THEN
          WRITE( out, "( /, '  Schur-complement method used ' )" )
        ELSE
          WRITE( out, "( /, '  Augmented system method used ' )" )
        END IF
      END IF

!  If necessary, print warning messages

  600 CONTINUE
      IF ( printi ) then

        SELECT CASE( inform%status )
          CASE( - 1  ) ; WRITE( out, 2210 ) 
          CASE( - 5  ) ; WRITE( out, 2250 ) 
          CASE( - 6  ) ; WRITE( out, 2260 ) 
          CASE( - 7  ) ; WRITE( out, 2270 ) 
          CASE( - 8  ) ; WRITE( out, 2280 ) 
          CASE( - 9  ) ; WRITE( out, 2290 ) 
          CASE( - 10 ) ; WRITE( out, 2300 ) 
        END SELECT

      END IF
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving LCF_solve_main ' )" )

      RETURN  

!  Non-executable statements

 2010 FORMAT( //, '  Final objective function value ', ES22.14,                &
              /,  '  Total number of iterations = ', I0 )
 2110 FORMAT( /,'  Norm of equality constraints is ', ES12.4, /,               &
                '  Norm of bound infeasibility is  ', ES12.4 ) 
 2150 FORMAT( A6, /, ( 4( 2I5, ES10.2 ) ) )
 2210 FORMAT( /, '  Warning - input paramters incorrect ' ) 
 2250 FORMAT( /, '  Warning - the constraints are inconsistent ' ) 
 2260 FORMAT( /, '  Warning - the constraints appear to be inconsistent ' ) 
 2270 FORMAT( /, '  Warning - factorization failure ' ) 
 2280 FORMAT( /, '  Warning - no further progress possible ' ) 
 2290 FORMAT( /, '  Warning - step too small to make further progress ' ) 
 2300 FORMAT( /, '  Warning - iteration bound exceeded ' ) 

!  End of LCF_solve_main

      END SUBROUTINE LCF_solve_main

!-*-*-*-*-*-*-   L C F _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LCF_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine LCF_initialize
!   control see Subroutine LCF_initialize
!   inform  see Subroutine LCF_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LCF_data_type ), INTENT( INOUT ) :: data
      TYPE ( LCF_control_type ), INTENT( IN ) :: control        
      TYPE ( LCF_inform_type ), INTENT( INOUT ) :: inform
 
!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated within SBLS

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
      IF ( inform%SBLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = ''
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated for the preprocessing stage

      CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
      IF ( data%QPP_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = ''
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'lcf: data%C_freed'
      CALL SPACE_dealloc_array( data%C_freed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lcf: data%SOL'
      CALL SPACE_dealloc_array( data%SOL,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lcf: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lcf: data%A_sbls%row'
      CALL SPACE_dealloc_array( data%A_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lcf: data%A_sbls%col'
      CALL SPACE_dealloc_array( data%A_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lcf: data%A_sbls%val'
      CALL SPACE_dealloc_array( data%A_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lcf: data%H_sbls%val'
      CALL SPACE_dealloc_array( data%H_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lcf: data%Index_C_freed'
      CALL SPACE_dealloc_array( data%Index_C_freed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine LCF_terminate

      END SUBROUTINE LCF_terminate 

!-*-*-*-*-*-    L C F _ L I N E S E A R C H    S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LCF_linesearch( dims, n, m,                                   &
                                 q_0, q_1, q_2, lambda_q, lambda_b,            &
                                 X, C, X_l, X_u, C_l, C_u, DX, DC,             &
                                 IBREAK, BREAKP, lbreak,                       &
                                 out, print_1line, print_detail,               &
                                 print_debug, t_opt, too_small, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the global minimizer of the function
!
!        lambda_q ( q0 + q1 t + q2 t ** 2 ) 
!        + lambda_b ( min( c - c_l , 0 ) ** 2 + max( c - c_u , 0 ) ** 2
!                   + min( x - x_l , 0 ) ** 2 + max( x - x_u , 0 ) ** 2 )
!
!  along the arc (x(t),c(t)) = (x,t) + t (dx,dc)  (t >= 0)
!
!  where x is a vector of n components ( x_1, .... , x_n ), 
!  and c is a vector of m components ( c_1, .... , c_m ), 
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  dims is a structure of type LCF_data_type, whose components hold SCALAR
!   information about the problem on input. The components will be unaltered
!   on exit. See LCF_solve_main for details.
!                 
!  q_0, q_1, q_2 are REAL variables that give the constant, linear and quadratic
!   coefficients for the quadratic part of the objective
!
!  lambda_q, lambda_b are REAL variables that give the weights assigned to
!   the quadratic and other parts of the objective
!
!  X is a REAL array of length n, which must be set by the user to the 
!   value of x
!
!  C is a REAL array of length m, which must be set by the user to the 
!   value of c
!
!  X_l is a REAL array of length n, that must be set by 
!   the user to the value of x_l for all components which have lower bounds
!
!  X_u is a REAL array of  length n, that must be set
!   by the user to the value of x_u for all components which have upper bounds
!
!  C_l is a REAL array of length m that must be set by the user to the 
!   value of c_l for all components which have lower bounds
!
!  C_u is a REAL array of length m that must be set by the user to the 
!   value of c_u for all components which have upper bounds
!
!  DX is a REAL array of length n, which must be set by the user to the 
!   value of dx
!
!  DC is a REAL array of length m, which must be set by the user to the 
!   value of dc
!
!  IBREAK is an INTEGER workspace array of length lbreak
!
!  BREAKP is a REAL workspace array of length lbreak
!
!  lbreak is an INTEGER that must be at least 
!   m + dims%x_l_end - dims%x_u_start + 1
!
!  t_opt  is a REAL variable, which gives the required value of t on exit
!
!  inform is an INTEGER variable, which gives the exit status. Possible
!   values are:
!
!    0     the minimizer given in t_opt occurs between breakpoints after first
!    1     the minimizer given in t_opt occurs at the breakpoint indicated by
!          the variable active
!   -1     the minimizer given in t_opt occurs before the first breakpoint
!   -2     the function is unbounded from below. Ignore the value in t_opt
!   -3     the value m is negative. Ignore the value in t_opt
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LCF_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m, lbreak, out
      INTEGER, INTENT( OUT ) :: inform
      REAL ( KIND = wp ), INTENT( IN ) ::q_0, q_1, q_2, lambda_q, lambda_b
      REAL ( KIND = wp ), INTENT( IN ) ::too_small
      REAL ( KIND = wp ), INTENT( OUT ) :: t_opt
      LOGICAL, INTENT( IN ) :: print_1line, print_detail, print_debug
      INTEGER, INTENT( OUT ), DIMENSION( lbreak ) :: IBREAK
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, DX
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ),                                        &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, DC
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lbreak ) :: BREAKP

!  Local variables

      INTEGER :: i, j, nbreak, inheap, iter, ibreakp, nbreak_total
      INTEGER :: cluster_start, cluster_end, nx, nc, enx, enc
      REAL ( KIND = wp ) :: d, res, val, slope, slope_old, eval, eslope
      REAL ( KIND = wp ) :: t_break, t_star, feasep, epsqrt, infeas_c, infeas_x
      REAL ( KIND = wp ) :: fun, t_pert, t_old, gradient
      REAL ( KIND = wp ) :: pert_val, pert_eps, curv, ecurv
      REAL ( KIND = wp ) :: breakp_max, slope_infeas_c, slope_infeas_x
      REAL ( KIND = wp ) :: curv_infeas_c, curv_infeas_x
      LOGICAL :: beyond_first_breakpoint
      CHARACTER ( LEN = 14 ) :: cluster

      t_opt = zero
      IF ( m < 0 ) THEN
         inform = - 3
         RETURN
      END IF

      iter = 0 ; nbreak = 0 ; t_break = zero
      epsqrt = SQRT( epsmch )

!  Find the distance to each constraint boundary, and the slope of this function
!  =============================================================================

      t_pert = zero
      infeas_c = zero ; slope_infeas_c = zero ; curv_infeas_c = zero
      infeas_x = zero ; slope_infeas_x = zero ; curv_infeas_x = zero
      breakp_max = zero
      nx = 0 ; nc = 0

      IF ( print_debug ) THEN
        WRITE( out, "( '      i,     xl           x          xu          dx' )" )
        DO i = dims%x_free + 1, n
          WRITE( out, "( I7, 4ES12.4 )" ) i, X_l( i ), X( i ), X_u( i ), DX( i )
        END DO
        IF ( dims%c_l_start <= dims%c_u_end )                                  &
          WRITE( out, "( '      i,     cl           c          cu          dc')")
        DO i = dims%c_l_start, dims%c_u_end
          WRITE( out, "( I7, 4ES12.4 )" ) i, C_l( i ), C( i ), C_u( i ), DC( i )
        END DO
      END IF 

!  simple non-negativity
!  ---------------------

      DO i = dims%x_free + 1, dims%x_l_start - 1
        res = X( i )
        IF ( res < zero ) infeas_x = infeas_x + res ** 2

        d = DX( i )
        IF ( ABS( d ) < too_small ) CYCLE
        IF ( res + t_pert * d < zero ) THEN
          nx = nx + 1
          slope_infeas_x = slope_infeas_x + two * d * res
          curv_infeas_x = curv_infeas_x + d * d
        END IF

!  Find if the step will change the status of the constraint

        IF ( ( d > zero .AND. res >= zero ) .OR.                               &
             ( d < zero .AND. res < zero ) ) CYCLE
        nbreak = nbreak + 1
        IBREAK( nbreak ) = m + i
        BREAKP( nbreak ) = - res / d
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  simple bound from below
!  -----------------------

      DO i = dims%x_l_start, dims%x_l_end
        res = X( i ) - X_l( i )
        IF ( res < zero ) infeas_x = infeas_x + res ** 2

        d = DX( i )
        IF ( ABS( d ) < too_small ) CYCLE

        IF ( res + t_pert * d < zero ) THEN
          nx = nx + 1
          slope_infeas_x = slope_infeas_x + two * d * res
          curv_infeas_x = curv_infeas_x + d * d
        END IF

!  Find if the step will change the status of the constraint

        IF ( ( d > zero .AND. res >= zero ) .OR.                              &
             ( d < zero .AND. res < zero ) ) CYCLE
        nbreak = nbreak + 1
        IBREAK( nbreak ) = m + i
        BREAKP( nbreak ) = - res / d
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  simple bound from above
!  -----------------------

      DO i = dims%x_u_start, dims%x_u_end
        res = X_u( i ) - X( i )
        IF ( res < zero ) infeas_x = infeas_x + res ** 2

        d = - DX( i )
        IF ( ABS( d ) < too_small ) CYCLE

        IF ( res + t_pert * d < zero ) THEN
          nx = nx + 1
          slope_infeas_x = slope_infeas_x + two * d * res
          curv_infeas_x = curv_infeas_x + d * d
        END IF

!  Find if the step will change the status of the constraint

        IF ( ( d > zero .AND. res >= zero ) .OR.                              &
             ( d < zero .AND. res < zero ) ) CYCLE
        nbreak = nbreak + 1
        IBREAK( nbreak ) = - m - i
        BREAKP( nbreak ) = - res / d
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  simple non-positivity
!  ---------------------

      DO i = dims%x_u_end + 1, n
        res = - X( i )
        IF ( res < zero ) infeas_x = infeas_x + res ** 2

        d = - DX( i )
        IF ( ABS( d ) < too_small ) CYCLE

        IF ( res + t_pert * d < zero ) THEN
          nx = nx + 1
          slope_infeas_x = slope_infeas_x + two * d * res
          curv_infeas_x = curv_infeas_x + d * d
        END IF

!  Find if the step will change the status of the constraint

        IF ( ( d > zero .AND. res >= zero ) .OR.                              &
             ( d < zero .AND. res < zero ) ) CYCLE
        nbreak = nbreak + 1
        IBREAK( nbreak ) = - m - i
        BREAKP( nbreak ) = - res / d
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  constraints with lower bounds
!  -----------------------------

      DO i = dims%c_l_start, dims%c_l_end
        res = C( i ) - C_l( i )
        IF ( res < zero ) infeas_c = infeas_c + res ** 2

        d = DC( i )
        IF ( ABS( d ) < too_small ) CYCLE

        IF ( res + t_pert * d < zero ) THEN
          nc = nc + 1
          slope_infeas_c = slope_infeas_c + two * d * res
          curv_infeas_c = curv_infeas_c + d * d
        END IF

!  Find if the step will change the status of the constraint

        IF ( ( d > zero .AND. res >= zero ) .OR.                              &
             ( d < zero .AND. res < zero ) ) CYCLE
        nbreak = nbreak + 1
        IBREAK( nbreak ) = i
        BREAKP( nbreak ) = - res / d
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

!  constraints with upper bounds
!  -----------------------------

      DO i = dims%c_u_start, dims%c_u_end
        res = C_u( i ) - C( i )
        IF ( res < zero ) infeas_c = infeas_c + res ** 2

        d = - DC( i )
        IF ( ABS( d ) < too_small ) CYCLE

        IF ( res + t_pert * d < zero ) THEN
          nc = nc + 1
          slope_infeas_c = slope_infeas_c + two * d * res
          curv_infeas_c = curv_infeas_c + d * d
        END IF

!  Find if the step will change the status of the constraint

        IF ( ( d > zero .AND. res >= zero ) .OR.                              &
             ( d < zero .AND. res < zero ) ) CYCLE
        nbreak = nbreak + 1
        IBREAK( nbreak ) = - i
        BREAKP( nbreak ) = - res / d
        breakp_max = MAX( breakp_max, BREAKP( nbreak ) )
      END DO

      nbreak_total = nbreak

!  Record the initial function and slope

      val = lambda_q * q_0 + lambda_b * ( infeas_c + infeas_x )
      slope = lambda_q * q_1 + lambda_b * ( slope_infeas_c + slope_infeas_x ) 
      curv = lambda_q * q_2 + lambda_b * ( curv_infeas_c + curv_infeas_x )
      gradient = slope

      IF ( print_detail ) THEN
        CALL LCF_n_val_and_derivs( dims, n, m, X, C, X_l, X_u, C_l, C_u,       &
                                   DX, DC, 0.0_wp, too_small,                  &
                                   eval, eslope, ecurv, enc, enx )
        eval = lambda_q * q_0 + lambda_b * eval
        eslope = lambda_q * q_1 + lambda_b * eslope
        ecurv = lambda_q * q_2 + lambda_b * ecurv
        WRITE( out, "( /, ' t = ', ES10.4 )" ) 0.0_wp
        WRITE( out, "( ' nc, enc = ', I0, 1X, I0, ', nx, enx = ',              &
       &    I0, 1X, I0 )" ) nc, enc, nx, enx
        WRITE( out, 2010 ) '  val', val, eval
        WRITE( out, 2010 ) 'slope', slope, eslope
        WRITE( out, 2010 ) ' curv', curv, ecurv
      END IF

!  Record the function value at (just on the other side of) the initial point

      fun = val

!  Order the breakpoints in increasing size using a heapsort. Build the heap

      CALL SORT_heapsort_build( nbreak, BREAKP, inheap, INDA = IBREAK )
      cluster_start = 1
      cluster_end = 0
      cluster = '      0      0'

!  =======================================================================
!  Start the main loop to find the first local minimizer of the piecewise
!  quadratic function. Consider the problem over successive pieces
!  =======================================================================

      beyond_first_breakpoint = .FALSE.
      inform = - 1
      DO

!  ---------------------------------------------------------------
!  The piecewise quadratic function within the current interval is
!    val + slope * t
!  ---------------------------------------------------------------

!  Print details of the piecewise quadratic in the next interval

        iter = iter + 1
        IF ( ( print_1line .AND. cluster_end == 0 ) .OR. print_detail )        &
          WRITE( out, 2000 )
        IF ( print_1line ) WRITE( out, "( 3X, I7, ES12.4, A14, 3ES12.4 )" )    &
           iter, t_break, cluster, fun, gradient, curv

!  If the slope of the unvariate function is insufficiently negative, exit

        IF ( slope >= - gzero .AND. curv >= zero ) THEN
          IF ( beyond_first_breakpoint ) t_opt = t_break
          IF ( inform == 0 ) t_opt = t_break
          fun = val + t_opt * ( slope + t_opt * curv )
          EXIT
        END IF

!  Find the next breakpoint

        t_old = t_break
        IF ( nbreak > 0 ) THEN
          t_break = BREAKP( 1 ) * ( one + epsmch )
          CALL SORT_heapsort_smallest( nbreak, BREAKP, inheap, INDA = IBREAK )
          cluster_end = cluster_end + 1
          cluster_start = cluster_end
        ELSE
          t_break = biginf
        END IF

!  If the slope of the univariate function is nonzero and its curvature is
!  positive, compute the line minimum

        IF ( curv > zero ) THEN
!         IF ( print_detail ) WRITE( out, "( ' slope, curv ', 2ES12.4 )" )     &
!            slope,  curv
          t_star = - half * slope / curv

!  If the line minimum occurs before the breakpoint, the line minimum gives
!  the required minimizer. Exit

          IF ( nbreak == 0 .OR. t_star < t_break ) THEN
            t_opt = t_star
            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star

!  Calculate the function value for the piecewise quadratic

            fun = val + t_opt * ( slope + t_opt * curv )
            gradient = slope + two * t_opt * curv
!           val = val + half * t_opt * slope

            IF ( print_detail ) WRITE( out, 2000 )
            IF ( print_1line ) WRITE( out, &
              "( 3X, I7, ES12.4, A14, 3ES12.4 )" )                             &
                 iter, t_opt, '      -      -', fun, gradient, curv
            IF ( print_debug ) THEN
              eval = LCF_n_val( dims, n, m, X, C, X_l, X_u,                    &
                                C_l, C_u, DX, DC, t_opt, too_small )
              eval = lambda_q * q_0 + lambda_b * eval
              write( out, 2010 ) '  val', fun, eval
            END IF
            IF ( beyond_first_breakpoint ) inform = 0
            EXIT
          ELSE
            IF ( print_detail ) WRITE( out, 2050 ) t_old, t_break, t_star
          END IF
        ELSE
          IF ( print_detail ) WRITE( out, 2040 ) t_old, t_break
        END IF

!  Exit if the function is unbounded from below

        IF ( nbreak == 0 ) THEN
          t_opt = biginf
          IF ( print_detail ) WRITE( out, 2000 )
          IF ( print_1line ) WRITE( out, &
                                    "( 3X, I7, 5X, 'inf', 22X, '-inf')" ) iter
          inform = - 2
          EXIT
        END IF

!  Update the univariate function and slope values

        slope_old = slope

!  Record the new breakpoint and the amount by which other breakpoints
!  are allowed to vary from this one and still be considered to be
!  within the same cluster

        pert_val = t_break * ( one + teneps ) + teneps
        pert_eps = epsmch

        feasep = pert_val
!       t_break = feasep

        IF ( t_break <= breakp_max ) THEN

          DO
            ibreakp = IBREAK( nbreak )
            i = ABS( ibreakp )

!  Update the slope and curvature

            IF ( i <= m ) THEN    ! c passes bound
              IF ( print_detail ) WRITE( out, 2020 )                           &
                'C', i, BREAKP( nbreak )
              d = DC( i )
              IF ( ibreakp > 0 ) THEN    ! lower bound on c
                res = C( i ) - C_l( i )
                IF ( d > zero ) THEN     ! c becomes feasible
                  nc = nc - 1
                  val = val - lambda_b * res ** 2
                  slope = slope - two * lambda_b * d * res
                  curv = curv - lambda_b * d * d
                ELSE                     ! c becomes infeasible
                  nc = nc + 1
                  val = val + lambda_b * res ** 2
                  slope = slope + two * lambda_b * d * res
                  curv = curv + lambda_b * d * d
                END IF
              ELSE                       ! upper bound on c
                res = C_u( i ) - C( i )
                IF ( d < zero ) THEN     ! c becomes feasible
                  nc = nc - 1
                  val = val - lambda_b * res ** 2
                  slope = slope + two * lambda_b * d * res
                  curv = curv - lambda_b * d * d
                ELSE                     ! c becomes infeasible
                  nc = nc + 1
                  val = val + lambda_b * res ** 2
                  slope = slope - two * lambda_b * d * res
                  curv = curv + lambda_b * d * d
                END IF
              END IF
            ELSE                        ! x passes bound
              j = i - m
              IF ( print_detail ) WRITE( out, 2020 )                           &
                'B', j, BREAKP( nbreak )
              d = DX( j )
              IF ( ibreakp > 0 ) THEN    ! lower bound on x
                IF ( j >= dims%x_l_start ) THEN
                  res = X( j ) - X_l( j )
                ELSE
                  res = X( j )
                END IF
                IF ( d > zero ) THEN     ! x becomes feasible
                  nx = nx - 1
                  val = val - lambda_b * res ** 2
                  slope = slope - two * lambda_b * d * res
                  curv = curv - lambda_b * d * d
                ELSE                     ! x becomes infeasible
                  nx = nx + 1
                  val = val + lambda_b * res ** 2
                  slope = slope + two * lambda_b * d * res
                  curv = curv + lambda_b * d * d
                END IF
              ELSE                       ! upper bound on x
                IF ( j <= dims%x_u_end ) THEN
                  res = X_u( j ) - X( j )
                ELSE
                  res = - X( j )
                END IF
                IF ( d < zero ) THEN     ! x becomes feasible
                  nx = nx - 1
                  val = val - lambda_b * res ** 2
                  slope = slope + two * lambda_b * d * res
                  curv = curv - lambda_b * d * d
                ELSE                     ! x becomes infeasible
                  nx = nx + 1
                  val = val + lambda_b * res ** 2
                  slope = slope - two * lambda_b * d * res
                  curv = curv + lambda_b * d * d
                END IF
              END IF
            END IF

!  If the last breakpoint has been passed, exit

            nbreak = nbreak - 1
            IF( nbreak == 0 ) EXIT

!  Determine if other terms become active at the breakpoint

            IF ( BREAKP( 1 ) >= feasep ) EXIT
            t_break = BREAKP( 1 ) * ( one + pert_eps )
            CALL SORT_heapsort_smallest( nbreak, BREAKP( : nbreak ), inheap,   &
                                         INDA = IBREAK )
            cluster_end = cluster_end + 1
          END DO

          pert_val = t_break * ( one + pert_eps ) + pert_eps
          IF ( print_detail ) THEN
            CALL LCF_n_val_and_derivs( dims, n, m, X, C, X_l, X_u, C_l, C_u,   &
                                       DX, DC, pert_val, too_small,            &
                                       eval, eslope, ecurv, enc, enx )
            eval = lambda_q * q_0 + lambda_b * eval
            eslope = lambda_q * q_1 + lambda_b * eslope
            ecurv = lambda_q * q_2 + lambda_b * ecurv
            WRITE( out, "( /, ' t = ', ES10.4 )" ) pert_val
            WRITE( out, "( ' nc, enc = ', 2I6, ' nx, enx = ', 2I6 )" )        &
              nc, enc, nx, enx
            WRITE( out, 2010 ) '  val', val, eval
            WRITE( out, 2010 ) 'slope', slope, eslope
            WRITE( out, 2010 ) ' curv', curv, ecurv
          END IF

!  Check that the size of the line gradient has not shrunk significantly in
!  the current segment of the piecewise arc. If it has, there may be a loss
!  of accuracy, so the line derivative should be recomputed.

          IF ( ABS( slope ) < - epsqrt * slope_old ) THEN
            IF ( print_debug )                                                 &
              WRITE( out, "( ' recompute line derivative ... ' )" )
            CALL LCF_n_val_and_derivs( dims, n, m, X, C, X_l, X_u, C_l, C_u,   &
                                       DX, DC, pert_val, too_small,            &
                                       val, slope, curv, nc, nx )
            val = lambda_q * q_0 + lambda_b * val
            slope = lambda_q * q_1 + lambda_b * slope
            curv = lambda_q * q_2 + lambda_b * curv
            IF ( print_debug )                                                 &
              WRITE( out, "( ' val, slope curv ', 3ES22.14 )" ) val, slope, curv
          ENDIF

        ELSE

!  Special case: all the remaining breakpoints are reached

          IF ( print_detail )                                                  &
            WRITE( out, "( ' all remaining breakpoints reached' )" )

          CALL LCF_n_val_and_derivs( dims, n, m, X, C, X_l, X_u, C_l, C_u,     &
                                     DX, DC, t_break, too_small, val, slope,   &
                                     curv, nc, nx )
          val = lambda_q * q_0 + lambda_b * val
          slope = lambda_q * q_1 + lambda_b * slope
          curv = lambda_q * q_2 + lambda_b * curv
          cluster_end = nbreak_total
          nbreak = 0

        END IF

!  Compute the function value at (just on the other side of) the breakpoint

        WRITE( cluster, "( 2I7 )" ) cluster_start, cluster_end
        fun = val + t_break * ( slope + t_break * curv )
        gradient = slope + two * t_break * curv

        beyond_first_breakpoint = .TRUE.
        t_opt = BREAKP( cluster_end  )
        inform = 1

!  ================
!  End of main loop
!  ================

      END DO

      RETURN

!  Non-executable statements

 2000 FORMAT( /, '  **  iter break point      cluster      ',                  &
                 ' val       slope     curvature ', / )
 2010 FORMAT( 1X, A5, '(est,true) = ', 2ES22.14 )
 2020 FORMAT( ' breakpoint for ', A1, '-term ', I7, ' reached, step = ', ES12.4)
 2040 FORMAT( /, ' Interval = [', ES12.4, ',', ES12.4, ']' )
 2050 FORMAT( /, ' Interval = [', ES12.4, ',', ES12.4, &
              '], stationary point = ', ES12.4 )

!  End of subroutine LCF_linesearch

      END SUBROUTINE LCF_linesearch
      
!-*-*-*-*-*-*-*-*-*-   L C F _ P _ V A L   F U N C T I O N   -*-*-*-*-*-*-*-*-*-

      FUNCTION LCF_n_val( dims, n, m, X, C, X_l, X_u, C_l, C_u, DX, DC,        &
                          t, too_small )
      REAL ( KIND = wp ) LCF_n_val

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value of the l_2 norm squared of the projection of
!     (x,c) + t(dx,dc) into x_l <= x <= x_u and c_l <= c <= c_u
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LCF_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( IN ) :: t, too_small
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, DX
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ),                                       &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, DC

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: d, infeas_x, infeas_c

      infeas_x = zero ; infeas_c = zero

!  simple non-negativity

      DO i = dims%x_free + 1, dims%x_l_start - 1
        d = DX( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( X( i ), zero ) ) ** 2
        ELSE
          infeas_x = infeas_x + ( MIN( X( i ) + t * d, zero ) ) ** 2
        END IF
      END DO

!  simple bound from below

      DO i = dims%x_l_start, dims%x_l_end
        d = DX( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( X( i ) - X_l( i ), zero ) ) ** 2
        ELSE
          infeas_x = infeas_x + ( MIN( X( i ) - X_l( i ) + t * d, zero ) ) ** 2
        END IF
      END DO

!  simple bound from above

      DO i = dims%x_u_start, dims%x_u_end
        d = DX( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( X_u( i ) - X( i ), zero ) ) ** 2
        ELSE
          infeas_x = infeas_x + ( MIN( X_u( i ) - X( i ) - t * d, zero ) ) ** 2
        END IF
      END DO

!  simple non-positivity

      DO i = dims%x_u_end + 1, n
        d = DX( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( - X( i ), zero ) ) ** 2
        ELSE
          infeas_x = infeas_x + ( MIN( - X( i ) - t * d, zero ) ) ** 2
        END IF
      END DO

!  simple bound on c from below

      DO i = dims%c_l_start, dims%c_l_end
        d = DC( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_c = infeas_c + ( MIN( C( i ) - C_l( i ), zero ) ) ** 2
        ELSE
          infeas_c = infeas_c + ( MIN( C( i ) - C_l( i ) + t * d, zero ) ) ** 2
        END IF
      END DO

!  simple bound on c from above

      DO i = dims%c_u_start, dims%c_u_end
        d = DC( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_c = infeas_c + ( MIN( C_u( i ) - C( i ), zero ) ) ** 2
        ELSE
          infeas_c = infeas_c + ( MIN( C_u( i ) - C( i ) - t * d, zero ) ) ** 2
        END IF
      END DO

      LCF_n_val = infeas_x + infeas_c

      RETURN

!  End of function LCF_n_val

      END FUNCTION LCF_n_val

!-*-*-*-   L C F_ P _ V A L _ A N D _ S L O P E   S U B R O U T I N E -*-*-*-

      SUBROUTINE LCF_n_val_and_derivs( dims, n, m, X, C, X_l, X_u, C_l, C_u,   &
                                       DX, DC, t, too_small, val, slope, curv, &
                                       nc, nx )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the value, slope and curvature (in the direction S) 
!  of the l_2 norm squared of the projection of
!  (x,c) + t(dx,dc) into x_l <= x <= x_u and c_l <= c <= c_u
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LCF_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( OUT ) :: nc, nx
      REAL ( KIND = wp ), INTENT( IN ) :: t, too_small
      REAL ( KIND = wp ), INTENT( OUT ) :: val, slope, curv
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, DX
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ),                                       &
             DIMENSION( dims%c_l_start : dims%c_u_end ) :: C, DC

!  Local variables

      INTEGER :: i
      REAL ( KIND = wp ) :: d, res, infeas_x, infeas_c, slope_x, slope_c
      REAL ( KIND = wp ) :: curv_x, curv_c

      infeas_x = zero ; slope_x = zero ; curv_x = zero
      infeas_c = zero ; slope_c = zero ; curv_c = zero
      nc = 0 ; nx = 0

!  non-negativity bound on x

      DO i = dims%x_free + 1, dims%x_l_start - 1
        d = DX( i )
        res = X( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( res, zero ) ) ** 2
        ELSE
          IF ( res + t * d < zero ) THEN
            nx = nx + 1
            infeas_x = infeas_x + res ** 2
            slope_x = slope_x + two * d * res
            curv_x = curv_x + d * d
          END IF
        END IF
      END DO

!  simple bound on x from below

      DO i = dims%x_l_start, dims%x_l_end
        d = DX( i )
        res = X( i ) - X_l( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( res, zero ) ) ** 2
        ELSE
          IF ( res + t * d < zero ) THEN
            nx = nx + 1
            infeas_x = infeas_x + res ** 2
            slope_x = slope_x + two * d * res
            curv_x = curv_x + d * d
          END IF
        END IF
      END DO

!  simple bound on x from above

      DO i = dims%x_u_start, dims%x_u_end
        d = DX( i )
        res = X_u( i ) - X( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( res, zero ) ) ** 2
        ELSE
          IF ( res - t * d < zero ) THEN
            nx = nx + 1
            infeas_x = infeas_x + res ** 2
            slope_x = slope_x - two * d * res
            curv_x = curv_x + d * d
          END IF
        END IF
      END DO

!  non-positivity bound on x

      DO i = dims%x_u_end + 1, n
        d = DX( i )
        res = - X( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_x = infeas_x + ( MIN( res, zero ) ) ** 2
        ELSE
          IF ( res - t * d < zero ) THEN
            nx = nx + 1
            infeas_x = infeas_x + res ** 2
            slope_x = slope_x - two * d * res
            curv_x = curv_x + d * d
          END IF
        END IF
      END DO

!  simple bound on c from below

      DO i = dims%c_l_start, dims%c_l_end
        d = DC( i )
        res = C( i ) - C_l( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_c = infeas_c + ( MIN( res, zero ) ) ** 2
        ELSE
          IF ( res + t * d < zero ) THEN
            nc = nc + 1
            infeas_c = infeas_c + res ** 2
            slope_c = slope_c + two * d * res
            curv_c = curv_c + d * d
          END IF
        END IF
      END DO

!  simple bound on c from above

      DO i = dims%c_u_start, dims%c_u_end
        d = DC( i )
        res = C_u( i ) - C( i )
        IF ( ABS( d ) < too_small ) THEN
          infeas_c = infeas_c + ( MIN( res, zero ) ) ** 2
        ELSE
          IF ( res - t * d < zero ) THEN
            nc = nc + 1
            infeas_c = infeas_c + res ** 2
            slope_c = slope_c - two * d * res
            curv_c = curv_c + d * d
          END IF
        END IF
      END DO

      val = infeas_x + infeas_c
      slope = slope_x + slope_c ; curv = curv_x + curv_c

      RETURN

!  End of subroutine LCF_n_val_and_derivs

      END SUBROUTINE LCF_n_val_and_derivs

!  End of module LCF

   END MODULE GALAHAD_LCF_double



