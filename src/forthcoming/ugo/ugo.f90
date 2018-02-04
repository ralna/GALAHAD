! THIS VERSION: GALAHAD 2.8 - 20/05/2016 AT 15:15 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ U G O   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.8. May 20th 2016

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_UGO_double

!     -------------------------------------------------------
!    |                                                       |
!    | UGO, an algorithm for univariate global optimization  |
!    |                                                       |
!    |   Aim: find the global minimizer of the univariate    |
!    |        objective f(x) within the interval [x_l, x_u]  |
!    |                                                       |
!     -------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_STRING_double, ONLY: STRING_integer_6
     USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: UGO_initialize, UGO_read_specfile, UGO_solve, UGO_terminate

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: quarter = 0.25_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

     INTEGER, PARAMETER :: header_interval = 50
     REAL ( KIND = wp ), PARAMETER :: midxdg_min = ten * epsmch

!  Lipschitz constant estimate used (step 2)

     INTEGER, PARAMETER  :: global_lipschitz_available = 1
     INTEGER, PARAMETER  :: global_lipschitz_estimated = 2
     INTEGER, PARAMETER  :: local_lipschitz_estimated = 3

!  next interval selection method (step 4)

     INTEGER, PARAMETER  :: interval_traditional = 1
     INTEGER, PARAMETER  :: interval_local_improvement = 2

!  other algorithm constants

     INTEGER :: storage_increment_min = 1000

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: UGO_control_type

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

!   the maximum number of iterations allowed

       INTEGER :: maxit = 100000

!   the number of initial (uniformly-spaced) evaluation points (<2 reset to 2)

       INTEGER :: initial_points = 2

!   incremenets of storage allocated (less that 1000 will be reset to 1000)

       INTEGER :: storage_increment = 1000

!  unit for any out-of-core writing when expanding arrays

      INTEGER :: buffer = 70

!   what sort of Lipschitz constant estimate will be used:
!     1 = global contant provided, 2 = global contant estimated,
!     3 = local costants estimated

       INTEGER :: lipschitz_estimate_used = 3

!  how is the next interval for examination chosen:
!     1 = traditional, 2 = local_improvement

       INTEGER ::  next_interval_selection = 1

!   try refine_with_newton Newton steps from the vacinity of the global
!    minimizer to try to improve the estimate

       INTEGER :: refine_with_newton = 5

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   overall convergence tolerances. The iteration will terminate when
!     the step is less than %stop_length

       REAL ( KIND = wp ) :: stop_length = ten ** ( - 5 )

!  if the absolute value of the gradient is smaller than small_g_for_newton, the
!   next evaluation point may be at a Newton estimate of a local minimizer

       REAL ( KIND = wp ) :: small_g_for_newton = ten ** ( - 3 )

!  if the absolute value of the gradient at the end of the interval search is
!   smaller than small_g, no Newton serach is necessary

       REAL ( KIND = wp ) :: small_g = ten ** ( - 6 )

!   stop if the objective function is smaller than a specified value

       REAL ( KIND = wp ) :: obj_sufficient = - epsmch ** ( - 2 )

!  the global Lipschitz constant for the gradient (-ve => unknown)

       REAL ( KIND = wp ) :: global_lipschitz_constant = - one

!  the reliability parameter that is used to boost insufficiently large
!  estimates of the Lipschitz constant

       REAL ( KIND = wp ) :: reliability_parameter = 1.2_wp

!  a lower bound on the Lipscitz constant for the gradient (not zero unless
!  the function is constant)

       REAL ( KIND = wp ) :: lipschitz_lower_bound = ten ** ( - 8 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

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

     END TYPE UGO_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: UGO_time_type

!  the total CPU time spent in the package

       REAL :: total = 0.0

!  the total clock time spent in the package

       REAL ( KIND = wp ) :: clock_total = 0.0
     END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: UGO_inform_type

!  return status. See UGO_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER :: iter = 0

!  the total number of evaluations of the objection function

       INTEGER :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER :: g_eval = 0

!  the total number of evaluations of the Hessian of the objection function

       INTEGER :: h_eval = 0

!  timings (see above)

       TYPE ( UGO_time_type ) :: time
     END TYPE UGO_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: UGO_data_type
       INTEGER :: branch = 1
       INTEGER :: eval_status, out, start_print, stop_print, advanced_start_iter
       INTEGER :: print_level, print_level_gltr, print_level_trs, ref( 1 )
       INTEGER :: storage_increment, storage, lipschitz_estimate_used
       INTEGER :: print_gap, iters_printed, newton, intervals, initial_points
       REAL :: time_start, time_record, time_now
       REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
       REAL ( KIND = wp ) :: x_best, f_best, g_best, h_best, x_l, x_u, dx
       LOGICAL :: printi, printt, printd, printm
       LOGICAL :: print_iteration_header
       LOGICAL :: set_printi, set_printt, set_printd, set_printm
       LOGICAL :: f_is_nan, reverse_f, reverse_g, reverse_h
       LOGICAL :: fgh_available, newton_refinement, x_extra_used
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NEXT
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PREVIOUS
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: UNFATHOMED
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: F
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_lips
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V

!  copy of controls

       TYPE ( UGO_control_type ) :: control
     END TYPE UGO_data_type

   CONTAINS

!-*-*-  G A L A H A D -  U G O _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE UGO_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for UGO controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( UGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( UGO_control_type ), INTENT( OUT ) :: control
     TYPE ( UGO_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initial private data. Set branch for initial entry

     data%branch = 10

     RETURN

!  End of subroutine UGO_initialize

     END SUBROUTINE UGO_initialize

!-*-*-*-*-   U G O _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE UGO_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by UGO_initialize could (roughly)
!  have been set as:

! BEGIN UGO SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  start-print                                     -1
!  stop-print                                      -1
!  maximum-number-of-iterations                    1000
!  number-of-initial-points                        2
!  block-of-storage-allocated                      1000
!  iterations-between-printing                     1
!  out-of-core-buffer                              70
!  lipschitz-estimate-used                         3
!  next-interval-selection-method                  2
!  refine-with-newton-iterations                   5
!  global-lipschitz-constant                       -1.0
!  reliability-parameter                           1.2
!  lipschitz-lower-bound                           1.0D-8
!  max-interval-length-required                    1.0D-5
!  try-newton-tolerence                            1.0D-3
!  sufficient-objective-value                      -1.0D+32
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END UGO SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( UGO_control_type ), INTENT( INOUT ) :: control
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
     INTEGER, PARAMETER :: initial_points = maxit + 1
     INTEGER, PARAMETER :: storage_increment = initial_points + 1
     INTEGER, PARAMETER :: buffer = storage_increment + 1
     INTEGER, PARAMETER :: lipschitz_estimate_used = buffer + 1
     INTEGER, PARAMETER :: next_interval_selection = lipschitz_estimate_used + 1
     INTEGER, PARAMETER :: refine_with_newton = next_interval_selection + 1
     INTEGER, PARAMETER :: alive_unit = refine_with_newton + 1
     INTEGER, PARAMETER :: global_lipschitz_constant = alive_unit + 1
     INTEGER, PARAMETER :: reliability_parameter = global_lipschitz_constant + 1
     INTEGER, PARAMETER :: lipschitz_lower_bound = reliability_parameter + 1
     INTEGER, PARAMETER :: stop_length = lipschitz_lower_bound + 1
     INTEGER, PARAMETER :: small_g_for_newton = stop_length + 1
     INTEGER, PARAMETER :: small_g = small_g_for_newton + 1
     INTEGER, PARAMETER :: obj_sufficient = small_g + 1
     INTEGER, PARAMETER :: cpu_time_limit = obj_sufficient + 1
     INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER, PARAMETER :: space_critical = clock_time_limit + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'UGO '
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
     spec( initial_points )%keyword = 'number-of-initial-points'
     spec( storage_increment )%keyword = 'block-of-storage-allocated'
     spec( buffer )%keyword = 'out-of-core-buffer'
     spec( lipschitz_estimate_used )%keyword = 'lipschitz-estimate-used'
     spec( next_interval_selection )%keyword = 'next-interval-selection-method'
     spec( refine_with_newton )%keyword = 'refine-with-newton-iterations'
     spec( alive_unit )%keyword = 'alive-device'

!  Real key-words

     spec( global_lipschitz_constant )%keyword = 'global-lipschitz-constant'
     spec( reliability_parameter )%keyword = 'reliability-parameter'
     spec( lipschitz_lower_bound )%keyword = 'lipschitz-lower-bound'
     spec( stop_length )%keyword = 'max-interval-length-required'
     spec( small_g_for_newton )%keyword = 'try-newton-tolerence'
     spec( small_g )%keyword = 'sufficient-gradient-tolerence'
     spec( obj_sufficient )%keyword = 'sufficient-objective-value'

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
     CALL SPECFILE_assign_value( spec( initial_points ),                       &
                                 control%initial_points,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( storage_increment ),                    &
                                 control%storage_increment,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( buffer ),                               &
                                 control%buffer,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lipschitz_estimate_used ),              &
                                 control%lipschitz_estimate_used,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( next_interval_selection ),              &
                                 control%next_interval_selection,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( refine_with_newton ),                   &
                                 control%refine_with_newton,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )
!  Set real values


     CALL SPECFILE_assign_value( spec( global_lipschitz_constant ),            &
                                 control%global_lipschitz_constant,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( reliability_parameter ),                &
                                 control%reliability_parameter,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lipschitz_lower_bound ),                &
                                 control%lipschitz_lower_bound,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_length ),                          &
                                 control%stop_length,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( small_g_for_newton ),                   &
                                 control%small_g_for_newton,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( small_g ),                              &
                                 control%small_g,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( obj_sufficient ),                       &
                                 control%obj_sufficient,                       &
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

     END SUBROUTINE UGO_read_specfile

!-*-*-*-  G A L A H A D -  U G O _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE UGO_solve( x_l, x_u, x, f, g, h, control, inform, data,        &
                           userdata, eval_FGH, X_extra )
!                          userdata, eval_F, eval_FG, eval_FGH )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  UGO_solve, a method for finding the global minimizer of a univariate
!    continuous function with a Lipschitz gradient in an interval

!  Many ingredients in the algorithm are based on the paper

!   Daniela Lera and Yaroslav D. Sergeyev,
!   "Acceleration of univariate global optimization algorithms working with
!    Lipschitz functions and Lipschitz first derivatives"
!   SIAM J. Optimization Vol. 23, No. 1, pp. 508–529 (2013)

!  but adapted to use 2nd derivatives

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_UGO.
!
!   x_l and x_u are scalars of type default real, that hold the bound on the
!    search interval [x_l,x_u]
!
!   x is a scalar of type default real, that holds the next value of x at
!    which the user is required to evaluate the objective (and its derivatives)
!    when inform%status > 0, or the value of the approximate global minimizer
!    when inform%status = 0
!
!   f is a scalar of type default real, that must be set by the user to hold
!    the value of f(x) if required by inform%status > 0 (see below), and will
!    return the value of the approximate global minimum when inform%status = 0
!
!   g is a scalar of type default real, that must be set by the user to hold
!    the value of f'(x) if required by inform%status > 0 (see below), and will
!    return the value of the first derivative of f at the approximate global
!    minimizer when inform%status = 0
!
!   h is a scalar of type default real, that must be set by the user to hold
!    the value of f''(x) if required by inform%status > 0 (see below), and will
!    return the value of the second derivative of f at the approximate global
!    minimizer when inform%status = 0
!
! control is a scalar variable of type UGO_control_type. See UGO_initialize
!  for details
!
! inform is a scalar variable of type UGO_inform_type. See the preamble
!  for further details. On initial entry, inform%status should be set to 1.
!  The user must check the value of inform%status on exit, and if required
!  provide further information and re-enter the subroutine. Possible values
!  are:
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
!    -7. The objective function appears to be unbounded from below
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
!        x and then re-enter the subroutine. The required value should be set
!        in f, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) - for instance, if the function
!        is undefined at x - the user need not set f, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the objective function value f(x) and its
!        derivative f'(x) at the point x and then re-enter the subroutine.
!        The required values should be set in f and g respectively,
!        and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) or f'(x) - for instance, if the
!        function or its derivative is undefined at x - the user need not
!        set f or g, but should then set data%eval_status to a non-zero value.
!     4. The user should compute the objective function value f(x) and its
!        first two derivatives f'(x) and f''(x) at the point x and then
!        re-enter the subroutine. The required values should be set in f, g
!        and h respectively, and data%eval_status should be set to 0. If the
!         user is unable to evaluate f(x), f'(x) or f''(x) - for instance, if
!        the function or its derivatives are undefined at x - the user need not
!        set f, g or h, but should then set data%eval_status to a non-zero value
!
!  data is a scalar variable of type UGO_data_type used for internal data.
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
!  eval_F is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=x must be returned in f, and the status
!   variable set to 0. If the evaluation is impossible at x, status should
!   be set to a nonzero value. If eval_F is not present, UGO_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_FG is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) and its first derivative f'(x) evaluated at x=x must be
!   returned in f and g respectively, and the status variable set to 0.
!   If the evaluation is impossible at x, status should be set to a nonzero
!   value. If eval_FG is not present, UGO_solve will return to the user with
!   inform%status = 3 each time an evaluation is required.
!
!  eval_FGH is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) and its first two derivative f'(x) and f''(x) evaluated at
!   x=x must be returned in f, g and h respectively, and the status variable
!   set to 0. If the evaluation is impossible at x, status should be set to a
!   nonzero value. If eval_FGH is not present, UGO_solve will return to the
!   user with inform%status = 4 each time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = wp ), INTENT( IN ) :: x_l, x_u
     REAL ( KIND = wp ), INTENT( INOUT ) :: x, f, g, h
     TYPE ( UGO_control_type ), INTENT( IN ) :: control
     TYPE ( UGO_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( UGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_FGH
!    OPTIONAL :: eval_F, eval_FG, eval_FGH
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL :: x_extra

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

!     INTERFACE
!       SUBROUTINE eval_F( status, x, userdata, f )
!       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!       INTEGER, INTENT( OUT ) :: status
!       REAL ( KIND = wp ), INTENT( OUT ) :: f
!       REAL ( KIND = wp ), INTENT( IN ) :: x
!       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
!       END SUBROUTINE eval_F
!     END INTERFACE

!    INTERFACE
!      SUBROUTINE eval_FG( status, x, userdata, f, g )
!      USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!       INTEGER, INTENT( OUT ) :: status
!       REAL ( KIND = wp ), INTENT( OUT ) :: f, g
!       REAL ( KIND = wp ), INTENT( IN ) :: x
!       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
!       END SUBROUTINE eval_FG
!     END INTERFACE

     INTERFACE
       SUBROUTINE eval_FGH( status, x, userdata, f, g, h )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), INTENT( OUT ) :: f, g, h
       REAL ( KIND = wp ), INTENT( IN ) :: x
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_FGH
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, im, ip, l, status, t
     INTEGER :: used_length, new_length, min_length
     REAL ( KIND = wp ) :: xi, xip, dg, di, dx, fi, fip, gi, gip, bi, ci
     REAL ( KIND = wp ) :: mi, term, wi, yi, zi, vox, v_max, x_max, x_new
     REAL ( KIND = wp ) :: argmin, argmin_psi, min_psi
     REAL ( KIND = wp ) :: gpiwi, gpiyi, argmin_pi, min_pi, dx_best, midxdg
     REAL ( KIND = wp ) :: psi_best, phizi, argmin_fiip, min_fiip
     REAL ( KIND = wp ) :: vi, vim, vip, hi, hip, x_newton
     LOGICAL :: alive, new_point
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
     CASE ( 110 ) ! functions evaluated at x
       GO TO 110
     CASE ( 210 ) ! functions evaluated at x
       GO TO 210
     CASE ( 320 ) ! functions evaluated at x
       GO TO 320
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  check that the bounds are consistent

     IF ( x_l > x_u ) THEN
       inform%status = GALAHAD_error_bad_bounds ; GO TO 990
     END IF

!  control the output printing

     IF ( control%start_print < 0 ) THEN
       data%start_print = - 1
     ELSE
       data%start_print = control%start_print
     END IF

     IF ( control%stop_print < 0 ) THEN
       data%stop_print = control%maxit + 1
     ELSE
       data%stop_print = control%stop_print
     END IF

     IF ( control%print_gap < 2 ) THEN
       data%print_gap = 1
     ELSE
       data%print_gap = control%print_gap
     END IF
     data%iters_printed = 0

!  basic single line of output per iteration

     data%set_printi = control%out > 0 .AND. control%print_level >= 1

!  as per printi, but with additional details

     data%set_printd = data%out > 0 .AND. data%control%print_level >= 2

     inform%iter = 0
     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print .AND.                                  &
          MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printd = data%set_printd
       data%print_level = control%print_level
     ELSE
       data%printi = .FALSE. ; data%printd = .FALSE.
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

!  is eval_fgh available?

     data%fgh_available = PRESENT( eval_FGH )

!  set the number of initial points at which the objective is to be evaluated

     data%initial_points = MAX( control%initial_points, 2 )

!  is there a reasonable global Lipscitz constant available?

     data%lipschitz_estimate_used = control%lipschitz_estimate_used

     IF ( data%lipschitz_estimate_used == global_lipschitz_available .AND.     &
          control%global_lipschitz_constant <= zero )                          &
       data%lipschitz_estimate_used = local_lipschitz_estimated

!  allocate initial storage for the trial points, the function and derivative
!  values at these points, and arrays NEXT and PREVIOUS that point to the
!  trial point after and before the one in position i, i.e., X(NEXT(i))
!  and X(PREVIOUS(i)) are the trial points to either side of X(i).
!  The global minimizer in the ith interval has been found if UNFATHOMED(i)
!  is false

     data%storage_increment = MAX( control%storage_increment,                  &
                                   storage_increment_min, data%initial_points )
     data%storage = data%storage_increment

     array_name = 'ugo: data%NEXT'
     CALL SPACE_resize_array( data%storage, data%NEXT,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     array_name = 'ugo: data%PREVIOUS'
     CALL SPACE_resize_array( data%storage, data%PREVIOUS,                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     array_name = 'ugo: data%UNFATHOMED'
     CALL SPACE_resize_array( data%storage, data%UNFATHOMED,                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     array_name = 'ugo: data%X'
     CALL SPACE_resize_array( data%storage, data%X,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     array_name = 'ugo: data%F'
     CALL SPACE_resize_array( data%storage, data%F,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     array_name = 'ugo: data%G'
     CALL SPACE_resize_array( data%storage, data%G,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     array_name = 'ugo: data%H'
     CALL SPACE_resize_array( data%storage, data%H,                            &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     array_name = 'ugo: data%G_lips'
     CALL SPACE_resize_array( data%storage, data%G_lips,                       &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 900

     IF ( data%lipschitz_estimate_used == local_lipschitz_estimated ) THEN
       array_name = 'ugo: data%V'
       CALL SPACE_resize_array( data%storage, data%V,                          &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
     END IF

!   ----------------------------------------------------------------------------
!                            INITIAL POINT LOOP
!   ----------------------------------------------------------------------------

!  The initial trials are performed at the points
!     x_k = x_l + (i-1)/(p-1) (x_u-x_l), k = 1,...,p

     data%x_l = MIN( x_l, x_u ) ; data%x_u = MAX( x_l, x_u )
     data%dx = ( data%x_u - data%x_l )                                        &
                 / REAL( data%initial_points - 1, KIND = wp )

     data%x_extra_used = .NOT. PRESENT( x_extra )
     l = 0
     DO i = 1, data%initial_points
       x = data%x_l + REAL( i - 1, KIND = wp ) * data%dx
       IF ( .NOT. data%x_extra_used ) THEN
         IF ( x > x_extra ) THEN
           l = l + 1
           data%X( l ) = x_extra
           data%x_extra_used = .TRUE.
         ELSE IF ( x == x_extra ) THEN
           data%x_extra_used = .TRUE.
         END IF
       END IF
       l = l + 1
       data%X( l ) = x
     END DO
     data%initial_points = l

     inform%iter = 0
     inform%f_eval = 0 ; inform%g_eval = 0 ; inform%h_eval = 0
 100 CONTINUE
       inform%iter = inform%iter + 1

!  evaluate the function and its derivatives at the next trial point

       x = data%X( inform%iter )
       IF ( data%fgh_available ) THEN
         CALL eval_FGH( status, x, userdata, f, g, h )
       ELSE
         data%branch = 110 ; inform%status = 4 ; RETURN
       END IF

!  return from external evaluation

 110   CONTINUE
       inform%f_eval = inform%f_eval + 1 ; inform%g_eval = inform%g_eval + 1
       inform%h_eval = inform%h_eval + 1
       data%X( inform%iter ) = x ; data%F( inform%iter ) = f
       data%G( inform%iter ) = g ; data%H( inform%iter ) = h

!  if the point is an improvement, record it

!write(6,*) ' x, f, iter ', x, f, inform%iter, data%f_best
       IF ( inform%iter > 1 ) THEN
         IF ( f < data%f_best ) THEN
           data%x_best = x ; data%f_best = f ; data%g_best = g ; data%h_best = h
         END IF
       ELSE
         data%x_best = x ; data%f_best = f ; data%g_best = g ; data%h_best = h
         IF ( x_l + control%stop_length > x_u ) GO TO 300
       END IF

!  record details of the new point

       IF ( data%printi ) THEN
         data%iters_printed = data%iters_printed + 1
         IF ( MOD( data%iters_printed, header_interval ) == 1 .OR.             &
              data%printd ) WRITE( control%out, 2000 )
         WRITE( control%out, 2010 )                                            &
           inform%iter, x, f, g, data%x_best, data%f_best
       END IF

!  check to see if the new objective value suffices

       IF ( f < control%obj_sufficient ) THEN
         inform%status = GALAHAD_error_unbounded ; GO TO 330
       END IF

!  check to see if the iteration limit has been achieved

       IF ( inform%iter >= control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 990
       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = GALAHAD_error_alive ; GO TO 990
         END IF
       END IF

!   ----------------------------------------------------------------------------
!                       END OF INITIAL POINT LOOP
!   ----------------------------------------------------------------------------

       IF ( inform%iter < data%initial_points ) GO TO 100

!  record the order of the initial points

     DO i = 1, data%initial_points - 1
       data%NEXT( i ) = i + 1
       data%PREVIOUS( i + 1 ) = i
       data%UNFATHOMED( i ) = .TRUE.
     END DO
     data%NEXT( data%initial_points ) = 0 ; data%PREVIOUS( 1 ) = 0

!   ----------------------------------------------------------------------------
!                            MAIN ITERATION LOOP
!   ----------------------------------------------------------------------------

!  The point x_k+1, k ≥ 2, of the current (k+1)th iteration is chosen as follows

     IF ( data%printi ) WRITE( control%out,                                    &
        "( 10X, 12( '-' ), ' main iteration ', 12( '-' ) )" )
 200 CONTINUE

!  check whether to continue printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print .AND.                                &
            MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0     &
           ) THEN
         data%printi = data%set_printi ; data%printd = data%set_printd
         data%print_level = control%print_level
       ELSE
         data%printi = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
       END IF

!  check to see if there is sufficient room, and if not extend the relevant
!  arrays preserving the current data

       IF ( MOD( inform%iter + 1, data%storage_increment ) == 0 ) THEN
         used_length = inform%iter
         new_length = data%storage + data%storage_increment
         min_length = new_length
         CALL SPACE_extend_array( data%NEXT, data%storage,                     &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%NEXT' ; GO TO 900
         END IF

         CALL SPACE_extend_array( data%PREVIOUS, data%storage,                 &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%PREVIOUS' ; GO TO 900
         END IF

         CALL SPACE_extend_array( data%UNFATHOMED, data%storage,               &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%FATHOMED' ; GO TO 900
         END IF

         CALL SPACE_extend_array( data%X, data%storage,                        &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%X' ; GO TO 900
         END IF

         CALL SPACE_extend_array( data%F, data%storage,                        &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%F' ; GO TO 900
         END IF

         CALL SPACE_extend_array( data%G, data%storage,                        &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%G' ; GO TO 900
         END IF

         CALL SPACE_extend_array( data%H, data%storage,                        &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%H' ; GO TO 900
         END IF

         IF ( data%lipschitz_estimate_used == local_lipschitz_estimated ) THEN
           CALL SPACE_extend_array( data%V, data%storage,                      &
                  used_length, new_length, min_length,                         &
                  control%buffer, inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) THEN
             inform%bad_alloc = 'ugo: data%V' ; GO TO 900
           END IF
         END IF

         used_length = inform%iter - 1
         CALL SPACE_extend_array( data%G_lips, data%storage,                   &
                used_length, new_length, min_length,                           &
                control%buffer, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'ugo: data%G_lips' ; GO TO 900
         END IF
         data%storage = data%storage + data%storage_increment
       END IF

!  ============================================================================
!              Improve estimates of the Lipschitz constants
!  ============================================================================

!  compute estimates m_i of the Lipschitz constants for f'(x) over each
!  interval [x_i, x_i+1], i = 1, ..., k-1

       SELECT CASE( data%lipschitz_estimate_used )

!  Option 1. Set m_i = M, the given global Lipschitz constant

       CASE ( global_lipschitz_available )
         mi = control%global_lipschitz_constant

         i = 1
         DO l = 1, inform%iter - 1
           data%G_lips( i ) = mi
           i = data%NEXT( i )
         END DO

!  Option 2. Set m_i = r max{ ξ, v_max }, i = 1, ..., k-1, where ξ > 0 reflects
!  the supposition that f(x) is not constant over the interval [a, b], the
!  reliability parameter r > 1 provides a safeguard for an underestimated
!  Lipschitz constant, and
!
!    v_max = max{v_i: i = 1, ..., k-1},
!
!  where
!          |2(f_i − f_i+1) + (g_i + g_i+1)(x_i+1 − x_i)| + d_i,
!    v_i = -----------------------------------------------------
!                       (x_i+1 − x_i)^2
!
!    d_i = sqrt( |2(f_i − f_i+1) + (g_i + g_i+1)(x_i+1 − x_i) |^2
!                 + (f_i+1 − f_i)^2 (x_i+1 − x_i)2 )
!
!  and f_i = f(x_i) and g_i = f'(x_i)

       CASE ( global_lipschitz_estimated )
         mi = control%lipschitz_lower_bound
!write(6,*) ' mi ', mi
         i = 1
         xi = data%X( i ) ; fi = data%F( i ) ; gi = data%G( i )
         hi = ABS( data%H( i ) )

         DO l = 1, inform%iter - 1 ! loop through intervals in increasing order
           ip = data%NEXT( i ) !  consider the i-th interval
           xip = data%X( ip ) ; fip = data%F( ip ) ; gip = data%G( ip )
           hip = ABS( data%H( ip ) )

!  compute d_i and v_i, and upgrade the estimate of m_i

           IF ( data%UNFATHOMED( i ) ) THEN
             dx = xip - xi
             term = ABS( two * ( fi - fip ) + ( gi + gip ) * dx )
             di = SQRT( term ** 2 + ( ( gip - gi ) * dx ) ** 2 )
             vi = ( term + di ) / dx ** 2
             mi = MAX( mi, vi, hi, hip )
!write(6,*) ' mi ', mi
           END IF

           IF ( l < inform%iter - 1 ) THEN !  prepare for the next interval
             i = ip ; xi = xip ; fi = fip ; gi = gip ; hi = hip
           END IF
         END DO

!  multiply the estimate by a reliability parameter, and set for each interval

         mi = control%reliability_parameter * mi
!write(6,*) ' mi ', mi

         i = 1
         DO l = 1, inform%iter - 1
           data%G_lips( i ) = mi
           i = data%NEXT( i )
         END DO

!  Option 3. Set m_i = r max{ λ_i, γ_i, ξ}, where r > 1 and ξ > 0 are as above,
!
!    λ_i = max{v_i−1 , v_i , v_i+1 }, i = 2, . . . , k − 2,
!
!  where the values v_i are as above (when i = 1 and i = k - 1, v_1 and v_k
!  are omitted), and
!
!    γ_i = v_max (x_i+1 − x_i ) / x_max
!
!  where v_max is as above and x_max = max{(x_i+1 − x_i ), 1 = 1, . . . , k-1}.

       CASE ( local_lipschitz_estimated )
         mi = control%lipschitz_lower_bound
         x_max = zero
         i = 1
         xi = data%X( i ) ; fi = data%F( i ) ;  gi = data%G( i )

!  compute v_i and update x_max

         DO l = 1, inform%iter - 1  !loop through intervals in increasing order
           ip = data%NEXT( i ) !  consider the i-th interval
           xip = data%X( ip ) ; fip = data%F( ip ) ; gip = data%G( ip )
           dx = xip - xi
           term = ABS( two * ( fi - fip ) + ( gi + gip ) * dx )
           di = SQRT( term ** 2 + ( ( gip - gi ) * dx ) ** 2 )
           data%V( i ) = ( term + di ) / dx ** 2
           x_max = MAX( x_max, dx )
           IF ( l < inform%iter - 1 ) THEN !  prepare for the next interval
             i = ip ; xi = xip ; fi = fip ; gi = gip
           END IF
         END DO

!  compute v_max

         v_max = MAXVAL( data%V( 1 : inform%iter - 1 ) )
         vox = v_max / x_max

!  compute the local estimates of the gradient Lipschitz constants

         i = 1 ; xi = data%X( i )
         vi = data%V( i ) ; hi = ABS( data%H( i ) )
         ip = data%NEXT( i ) ; xip = data%X( ip ) ;
         vip = data%V( ip ) ; hip = ABS( data%H( ip ) )

         IF ( inform%iter >= 4 ) THEN
           data%G_lips( i ) = control%reliability_parameter *                  &
             MAX( control%lipschitz_lower_bound, vox * ( xip - xi ),           &
                  vi, vip, hi, hip )
           im = i ; i = ip ; xi = xip ; hi = hip
           DO l = 2, inform%iter - 2
             ip = data%NEXT( i ) ; xip = data%X( ip )
             vip = data%V( ip ) ; hip = ABS( data%H( ip ) )
             data%G_lips( i ) = control%reliability_parameter *                &
               MAX( control%lipschitz_lower_bound, vox * ( xip - xi ),         &
                    vim, vi, vip, hi, hip )
             im = i ; i = ip ; xi = xip ; vim = vi ; vi = vip ; hi = hip
           END DO
           ip = data%NEXT( i ) ; xip = data%X( ip ) ; hip = ABS( data%H( ip ) )
           data%G_lips( i ) = control%reliability_parameter *                  &
             MAX( control%lipschitz_lower_bound, vox * ( xip - xi ),           &
                  vim, vi, hi, hip )
         ELSE IF ( inform%iter == 3 ) THEN
           data%G_lips( i ) = control%reliability_parameter *                  &
             MAX( control%lipschitz_lower_bound, vox * ( xip - xi ),           &
                  vi, vip, hi, hip )
           data%G_lips( ip ) = data%G_lips( i )
         ELSE IF ( inform%iter == 2 ) THEN
           data%G_lips( i ) = control%reliability_parameter *                  &
             MAX( control%lipschitz_lower_bound, vox * ( xip - xi ),           &
             vi, hi, hip )
         END IF
       END SELECT

!  ============================================================================
!      Compute the minimizer of the support function within each interval
!  ============================================================================

!  Since f has a Lipschitz gradient (with constant m_i) on each interval
!  [x_i,x_i+1], it is bounded from below by the functions
!  phi_i(x) = f_i + f'_i (x - x_i) - 1/2 m_i (x - x_i)^2 and
!  phi_i+1(x) = f_i+1 + f'_i+1 (x - x_i+1) - 1/2 m_i (x - x_i+1)^2.
!  There are also values x_i < w_i < y_i < x_i+1 for which a convex
!  quadratic pi_i(x) = 1/2 m_i x^2 + b_i x + c_i satisfies
!  pi_i(w_i) = phi_i(w_i), pi_i(y_i) = phi_i+1(y_i), pi'_i(w_i) = phi'_i(w_i)
!  and pi'_i(y_i) = phi'_i+1(y_i), and thus f(x) >= psi_i(x) for all
!  x in [x_i,x_i+1], where the support function
!             phi_i(x)   x in [x_i,w_i]
!  psi_i(x) = pi_i(x)    x in [w_i,y_i]   .
!             phi_i+1(x) x in [w_i,x_i+1]
!  Formulae for w_i, y_i and the minimizer r_i of psi_i(x) are easily
!  determined, see below

!  Initiate the index sets I = ∅, Y' = ∅, and Y = ∅. Set the index of the
!  current interval l = 1

       i = 1
       xi = data%X( i ) ; fi = data%F( i ) ;  gi = data%G( i )

!  t indicates the interval with the smallest value of psi, psi_best;
!  initialize with hopeless values

       t = 0 ; psi_best = fi + one ; new_point = .FALSE.

       DO l = 1, inform%iter - 1 !loop through the intervals in increasing order
         mi = data%G_lips( i )
         ip = data%NEXT( i ) !  consider the i-th interval
         xip = data%X( ip ) ; fip = data%F( ip ) ; gip = data%G( ip )

         IF ( data%UNFATHOMED( i ) ) THEN
           dx = xip - xi ; dg = gip - gi
           midxdg = mi * dx + dg

!  check for the exceptional possibility that m_i is too small
!  so that w_i and y_i do not exist, and boost m_i in that case

           IF ( midxdg < midxdg_min ) THEN
             mi = control%reliability_parameter * mi
             data%G_lips( i ) = mi
             midxdg = mi * dx + dg
           END IF

!  find the point of intersection z_i of phi_i(x) and phi_i+1(x) and the
!  value of phi(zi)

           zi = ( fi - fip + gip * xip - gi * xi +                             &
                  half * mi * ( xip ** 2 - xi ** 2 ) ) / midxdg
           phizi = fi + gi * ( zi - xi ) - half * mi * ( zi - xi ) ** 2

!  find the smaller of f_i and fi+1 along with its corresponding x

           IF ( fi <= fip ) THEN
             min_fiip = fi
             argmin_fiip = xi
             argmin = - 1
           ELSE
             min_fiip = fip
             argmin_fiip = xip
             argmin = 1
           END IF

!  if phi(zi) > min (f_i, fi+1), the minimum of f in [x_i,x_i+1]
!  is min (f_i, fi+1)

           IF ( phizi >= min_fiip ) THEN
             min_psi = min_fiip
             argmin_psi = argmin_fiip
             data%UNFATHOMED( i ) = .FALSE.

!  otherwise compute w_i, y_i, b_i and c_i

           ELSE
             new_point = .TRUE.
             term = quarter * midxdg / mi
             wi = - term + zi
             yi = term + zi
             bi = gip - two * mi * yi + mi * xip
             ci = fip - gip * xip - half * mi * xip ** 2 + mi * yi ** 2

!  compute the derivative of psi at w_i and y_i

             gpiwi = mi * wi + bi
             gpiyi = mi * yi + bi

!  find r_i depending on the signs of these derivatives

            IF ( gpiwi < zero ) THEN

!  the minimizer of psi_i is that of phi_i

               IF ( gpiyi > zero ) THEN
                 argmin_pi = - bi / mi
                 min_pi = ci - half * bi * bi / mi
                 IF ( min_fiip > min_pi ) THEN
                   min_psi = min_pi
                   argmin_psi = argmin_pi
                   argmin = 0
                   IF ( min_psi >= data%f_best ) data%UNFATHOMED( i ) = .FALSE.

!  the minimizer of psi_i occurs at the end of the interval

                 ELSE
                   min_psi = min_fiip
                   argmin_psi = argmin_fiip
                   data%UNFATHOMED( i ) = .FALSE.
                 END IF
               ELSE
                 min_psi = min_fiip
                 argmin_psi = argmin_fiip
                 data%UNFATHOMED( i ) = .FALSE.
               END IF
             ELSE
               IF ( gpiyi >= zero ) THEN
                 min_psi = min_fiip
                 argmin_psi = argmin_fiip
                 data%UNFATHOMED( i ) = .FALSE.
               ELSE ! i.e. gpiyi > 0 & gpiwi >= 0
                 write(6,*) ' not possible ?'
                 write(6,"( ' xi, xip ', 2ES12.4 )" ) xi, xip
                 write(6,"( ' fi, fip ', 2ES12.4 )" ) fi, fip
                 write(6,"( ' gi, gip ', 2ES12.4, ' mi ', ES12.4 )" ) gi, gip,mi
               END IF
             END IF
           END IF

!  check to see if the estimate of psi_best has improved

           IF ( data%UNFATHOMED( i ) ) THEN
             IF ( min_psi < psi_best ) THEN
               t = i ; psi_best = min_psi ; dx_best = dx

               IF ( argmin == 0 ) THEN
                 x_new = argmin_psi
               ELSE IF ( argmin == - 1 ) THEN
                 x_new = wi
               ELSE
                 x_new = yi
               END IF

!  if the slope is small at either end of the interval, compute the
!  Newton step, and check that this also lies in the interval

               IF ( ABS( gi ) <= control%small_g_for_newton ) THEN
                 IF ( ABS( gip ) <= ABS( gi ) ) THEN
                   x_newton = xip - gip / data%H( ip )
                   IF ( x_newton > xi .AND. x_newton < xip ) x_new = x_newton
                 ELSE
                   x_newton = xi - gi / data%H( i )
                   IF ( x_newton > xi .AND. x_newton < xip ) x_new = x_newton
                 END IF
               ELSE IF ( ABS( gip ) <= control%small_g_for_newton ) THEN
                 x_newton = xip - gip / data%H( ip )
                 IF ( x_newton > xi .AND. x_newton < xip ) x_new = x_newton
               END IF
             END IF
           END IF
         END IF

 !  prepare for the next interval

         IF ( l < inform%iter - 1 ) THEN
           i = ip ; xi = xip ; fi = fip ; gi = gip
         END IF
       END DO

!  ============================================================================
!               Determine the interval for the next evaluation
!  ============================================================================

!  Find the interval (x_t−1, x_t) for the next possible trial

       SELECT CASE( control%next_interval_selection )

!  Option 1. select the interval (x_t, x_t+1) such that
!
!     R_t = min{ R_i : 1 ≤ i ≤ k-1 }

       CASE ( interval_traditional )

!  Option 2. (the local improvement technique).
!  flag is a parameter initially equal to zero.
!  imin is the index corresponding to the current estimate of the minimal value
!  of the function, that is, zimin = f (ximin ) ≤ f (xi ), i = 1, . . . , k.
!  z k is the result of the last trial corresponding to a point xj
!  in the line (2.1), i.e., xk = xj .
!
!  IF (flag=1) THEN
!    IF z k < zimin THEN imin = j.

!   Local improvement: Alternate the choice of the interval (xt−1 , xt ) among
!   t = imin + 1 and t = imin, if imin = 2, . . . , k − 1
!   (if imin = 1 or imin = k, take t = 2 or t = k, respectively)
!   in such a way that for δ > 0 it follows

!   (2.13)|xt − xt−1 | > δ.

!  ELSE (flag=0)
!    t = argmin{Ri : 2 ≤ i ≤ k}
!  ENDIF

!  flag=NOTFLAG(flag)


       CASE ( interval_local_improvement )

       END SELECT

!  ============================================================================
!                    Compute the new evaluation point
!  ============================================================================

!  checck to see if we have fathomed every interval

       IF ( .NOT. new_point ) THEN
         x = data%x_best ; f = data%f_best ; g = data%g_best ; h = data%h_best
         GO TO 300
       END IF

       x = x_new

!  if the best value of psi is in a tiny interval, or if every interval has
!  been fathomed, exit with an estimate of the global minimizer

       IF ( dx_best <= control%stop_length .OR. t == 0) THEN
         x = data%x_best ; f = data%f_best ; g = data%g_best ; h = data%h_best
         GO TO 300

!  record the next evaluation point

       ELSE
         inform%iter = inform%iter + 1
         ip = data%NEXT( t )
         data%NEXT( t ) = inform%iter ; data%NEXT( inform%iter ) = ip
         data%PREVIOUS( ip ) = inform%iter ; data%PREVIOUS( t ) = i
         data%UNFATHOMED( inform%iter ) = .TRUE.

!  evaluate the function and its derivatives at the new point

         IF ( data%fgh_available ) THEN
           CALL eval_FGH( status, x, userdata, f, g, h )
         ELSE
           data%branch = 210 ; inform%status = 4 ; RETURN
         END IF
       END IF

!  return from external evaluation

 210   CONTINUE
       data%X( inform%iter ) = x ; data%F( inform%iter ) = f
       data%G( inform%iter ) = g ; data%H( inform%iter ) = h
       inform%f_eval = inform%f_eval + 1 ; inform%g_eval = inform%g_eval + 1
       inform%h_eval = inform%h_eval + 1

!  if the point is an improvement, record it

       IF ( f < data%f_best ) THEN
         data%x_best = x ; data%f_best = f ; data%g_best = g ; data%h_best = h
       END IF

!  record details of the new point

       IF ( data%printi ) THEN
         data%iters_printed = data%iters_printed + 1
         IF ( MOD( data%iters_printed, header_interval ) == 1 .OR.             &
              data%printd ) WRITE( control%out, 2000 )
         WRITE( control%out, 2010 )                                            &
           inform%iter, x, f, g, data%x_best, data%f_best
       END IF

!  check to see if the new objective value suffices

       IF ( f < control%obj_sufficient ) THEN
         inform%status = GALAHAD_error_unbounded ; GO TO 330
       END IF

!  check to see if the iteration limit has been achieved

       IF ( inform%iter >= control%maxit ) THEN
         x = data%x_best ; f = data%f_best ; g = data%g_best ; h = data%h_best
         inform%status = GALAHAD_error_max_iterations ; GO TO 990
       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           x = data%x_best ; f = data%f_best ; g = data%g_best ; h = data%h_best
           inform%status = GALAHAD_error_alive ; GO TO 990
         END IF
       END IF
       GO TO 200

!   ----------------------------------------------------------------------------
!                       END OF MAIN ITERATION LOOP
!   ----------------------------------------------------------------------------

 300 CONTINUE
     data%intervals = inform%iter

!  check to see if a Newton refinement loop is required

     IF ( control%refine_with_newton <= 0 .OR. ABS( g ) <= control%small_g     &
          .OR. x == data%x_l .OR. x == data%x_u ) GO TO 330
     data%newton_refinement = .FALSE.

!  Newton refinement loop

     data%newton = 0
 310 CONTINUE

!  exit if the maximum number of permitted Newton steps has been exceeded, if
!  the gradient is sufficiently small or if the step is negligible

       IF ( data%newton > control%refine_with_newton .OR.                      &
            ABS( g ) <= control%small_g .OR. x == x - g / h ) GO TO 330
       data%x_best = x ; data%f_best = f ; data%g_best = g ; data%h_best = h

       IF ( .NOT. data%newton_refinement ) THEN
         IF ( data%printi ) WRITE( control%out,                                &
            "( 10X, 11( '-' ), ' Newton refinement ', 10( '-' ) )" )
         data%newton_refinement = .TRUE.
       END IF

!  compute the Newton improvement and evaluate the function at this point

       x = x - g / h
       data%newton = data%newton + 1
       inform%iter = inform%iter + 1
       IF ( data%fgh_available ) THEN
         CALL eval_FGH( status, x, userdata, f, g, h )
       ELSE
         data%branch = 320 ; inform%status = 4 ; RETURN
       END IF

!  return from external evaluation

 320   CONTINUE
       inform%f_eval = inform%f_eval + 1
       inform%g_eval = inform%g_eval + 1
       inform%h_eval = inform%h_eval + 1

!  if the objective or gradient deteriorate, exit with the previous
!  (better) values

       IF ( data%f_best < f .OR. ABS( data%g_best ) < ABS( g ) .OR.            &
            x < data%x_l .OR. x > data%x_u ) THEN
         x = data%x_best ; f = data%f_best ; g = data%g_best ; h = data%h_best
         GO TO 330
       END IF

       IF ( data%printi ) THEN
         data%iters_printed = data%iters_printed + 1
         IF ( MOD( data%iters_printed, header_interval ) == 1 .OR.             &
              data%printd ) THEN
           WRITE( control%out, 2000 )
         END IF
         WRITE( control%out, 2010 ) inform%iter, x, f, g, x, f
       END IF
     GO TO 310

!  end of Newton refinement loop

 330 CONTINUE

!  if desired, illustrate the intervals

     IF ( data%printi ) THEN
       WRITE(6,"( /, '   int      x           f           g           h  ',    &
      &              '   fathomed      L' )" )
       i = 1
       DO l = 1, data%intervals - 1
         WRITE(6,"( I6, 4ES12.4, 1X, L1, I6, ES12.4 )" ) l, data%X( i ),       &
           data%F( i ), data%G( i ), data%H( i ),                              &
           .NOT. data%UNFATHOMED( i ), i, data%G_lips( i )
         i = DATA%NEXT( i )
       END DO
       WRITE(6,"( I6, 4ES12.4, 1X, '-', I6, '      -' )" ) l, data%X( i ),     &
         data%F( i ), data%G( i ), data%H( i ), i
     END IF

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     inform%status = GALAHAD_ok
     RETURN

!  -------------
!  Error returns
!  -------------

!  allocation error

 900 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) WRITE( control%out,                                    &
       "( A, ' ** Message from -UGO_solve-', /,  A,                            &
      &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )        &
       prefix, prefix, inform%bad_alloc, inform%alloc_status
     RETURN

 990 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'UGO_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

2000 FORMAT( '    iter        x             f             g     ',             &
             ' |     x_best        f_best' )
2010 FORMAT( I8, 3ES14.6, ' |', 2ES14.6 )

!  End of subroutine UGO_solve


     END SUBROUTINE UGO_solve

!-*-*-  G A L A H A D -  U G O _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE UGO_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( UGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( UGO_control_type ), INTENT( IN ) :: control
     TYPE ( UGO_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'ugo: data%NEXT'
     CALL SPACE_dealloc_array( data%NEXT,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%PREVIOUS'
     CALL SPACE_dealloc_array( data%PREVIOUS,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%UNFATHOMED'
     CALL SPACE_dealloc_array( data%UNFATHOMED,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%X'
     CALL SPACE_dealloc_array( data%X,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%F'
     CALL SPACE_dealloc_array( data%F,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%G'
     CALL SPACE_dealloc_array( data%G,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%H'
     CALL SPACE_dealloc_array( data%H,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%G_lips'
     CALL SPACE_dealloc_array( data%G_lips,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ugo: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
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

!  End of subroutine UGO_terminate

     END SUBROUTINE UGO_terminate

!  End of module GALAHAD_UGO

   END MODULE GALAHAD_UGO_double

