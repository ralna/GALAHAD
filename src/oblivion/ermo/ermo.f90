! THIS VERSION: GALAHAD 2.4 - 03/04/2009 AT 16:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ E R M O   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.4. March 17th 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_ERMO_double

!     -------------------------------------------------------
!    |                                                       |
!    | ERMO, an enriched-recursive multilevel optimization   |
!    |  algorithm                                            |
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
     USE GALAHAD_MOP_double
     USE HSL_MI20_double
     USE GALAHAD_GLTR_double
     USE GALAHAD_TRU_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: ERMO_initialize, ERMO_read_specfile, ERMO_solve,                &
               ERMO_terminate, NLPT_problem_type, NLPT_userdata_type,          &
               SMT_type, SMT_put

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!--------------------------
!  Derived type definitions
!--------------------------

     TYPE, PUBLIC :: ERMO_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit
       INTEGER :: start_print, stop_print, print_gap, model
       INTEGER :: preconditioner, cycle, max_level_coarsest, max_coarse_dim
       INTEGER :: max_gltr_its_for_newton
       REAL ( KIND = wp ) :: stop_g, stop_relative_subspace_g
       REAL ( KIND = wp ) :: obj_unbounded
       REAL ( KIND = wp ) :: cpu_time_limit
       LOGICAL :: include_gradient_in_subspace, include_newton_like_in_subspace
       LOGICAL :: hessian_available
       LOGICAL :: space_critical, deallocate_error_fatal
       CHARACTER ( LEN = 30 ) :: alive_file
       CHARACTER ( LEN = 30 ) :: prefix
       TYPE ( GLTR_control_type ) :: GLTR_control
       TYPE ( TRU_control_type ) :: TRU_control
     END TYPE ERMO_control_type

     TYPE, PUBLIC :: ERMO_time_type
       REAL :: total, preprocess, analyse, factorize, solve
     END TYPE

     TYPE, PUBLIC :: ERMO_inform_type
       INTEGER :: status, alloc_status, iter, cg_iter
       INTEGER :: f_eval, g_eval, h_eval, h_prod
       REAL ( KIND = wp ) :: obj, norm_g
       CHARACTER ( LEN = 80 ) :: bad_alloc
       TYPE ( ERMO_time_type ) :: time
       TYPE ( GLTR_info_type ) :: GLTR_inform
       TYPE ( TRU_inform_type ) :: TRU_inform
     END TYPE ERMO_inform_type

     TYPE, PUBLIC :: ERMO_cascade_type
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
     END TYPE ERMO_cascade_type

     TYPE, PUBLIC :: ERMO_data_type
       INTEGER :: eval_status, out, start_print, stop_print
       INTEGER :: print_level, branch, succ, ref( 1 )
       INTEGER :: nprec, n_prolong, level_coarsest, level_coarsest_n
       INTEGER :: level, level_n, ig, in, print_level_gltr, print_level_tru
       REAL :: time, time_new, time_total
       REAL ( KIND = wp ) :: radius, model
       LOGICAL :: printi, printt, printd, printm
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printd, set_printm
       LOGICAL :: new_h, got_f, got_g, got_h
       LOGICAL :: reverse_f, reverse_g, reverse_h, reverse_hprod, reverse_prec
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: row
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: P_ind
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: P_ptr
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_current
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_newton
       TYPE ( NLPT_problem_type ) :: TRU_nlp
       TYPE ( SMT_type ) :: A
       TYPE ( ZD11_type ) :: H_sym, P_trans, C_times_P
       TYPE ( ZD11_type ), ALLOCATABLE, DIMENSION( : ) :: P
       TYPE ( ZD11_type ), ALLOCATABLE, DIMENSION( : ) :: C
       TYPE ( ERMO_cascade_type ), ALLOCATABLE, DIMENSION( : ) :: CASCADE
       TYPE ( mi20_data ), ALLOCATABLE, DIMENSION( : ) :: mi20_coarse_data
       TYPE ( mi20_keep ) :: mi20_keep
       TYPE ( mi20_control ) :: mi20_control
       TYPE ( mi20_info ) :: mi20_inform
       TYPE ( ERMO_control_type ) :: control
       TYPE ( GLTR_data_type ) :: GLTR_data
       TYPE ( TRU_data_type ) :: TRU_data
       TYPE ( NLPT_userdata_type ) :: TRU_userdata
     END TYPE ERMO_data_type

   CONTAINS

!-*-*-  G A L A H A D -  E R M O _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE ERMO_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for ERMO controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ERMO_data_type ), INTENT( INOUT ) :: data
     TYPE ( ERMO_control_type ), INTENT( OUT ) :: control
     TYPE ( ERMO_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!    INTEGER :: status, alloc_status

     inform%status = GALAHAD_ok

!  Initalize GLTR components

     CALL GLTR_initialize( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )
     control%GLTR_control%prefix = '" - GLTR:"                    '

!  Initalize TRU components

     CALL TRU_initialize( data%TRU_data, control%TRU_control,                  &
                          inform%TRU_inform )
     control%TRU_control%prefix = '" - TRU:"                     '
!    control%TRU_control%hessian_available = .TRUE.
     control%TRU_control%hessian_available = .FALSE.
!    control%TRU_control%preconditioner = - 1
     control%TRU_control%subproblem_direct = .FALSE.
!    control%TRU_control%subproblem_direct = .TRUE.

!  Error and ordinary output unit numbers

     control%error = 6
     control%out = 6
     control%GLTR_control%error = control%error
     control%GLTR_control%out = control%out
     control%TRU_control%error = control%error
     control%TRU_control%out = control%out

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

     control%alive_unit = 40
     control%alive_file = 'ALIVE.d'

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

     control%print_level = 0

!  Maximum number of iterations

     control%maxit = 100

!   Any printing will start on this iteration

     control%start_print = - 1

!   Any printing will stop on this iteration

     control%stop_print = - 1

!   Printing will only occur every print_gap iterations

     control%print_gap = 1

!  is the Hessian matrix of second derivatives available or is access only
!  via matrix-vector products?

     control%hessian_available = .TRUE.
!    control%hessian_available = .FALSE.

!  Specify the model used. Possible values are
!
!      0  dynamic (*not yet implemented*)
!      1  first-order (no Hessian)
!      2  second-order (exact Hessian)
!      3  barely second-order (identity Hessian)

     control%model = 2

!  Specify the preconditioner used. Possible values are
!
!     -3  users own preconditioner
!     -1  identity (= no preconditioner)
!      0  automatic (*not yet implemented*)
!      1  diagonal, P = diag( max( Hessian, %min_diagonal ) )

     control%preconditioner = 1

!  Specify the multilevel cycle stategy. Possible values are
!
!      0  automatic (*not yet implemented*)
!      1  single V-cycle
!      2  multiple V-cycles

     control%cycle = 2

!  The largest number of levels allowed

     control%max_level_coarsest = 100

!  The maximum dimension of the coarsest subproblem

     control%max_coarse_dim = 100

!  Should we include the gradient in the search subspace?

     control%include_gradient_in_subspace = .TRUE.

!  Should we include something closer to the Newton direction in the search
!  subspace? If so, at most how many GLTR (CG) iterations should be used?

     control%include_newton_like_in_subspace = .TRUE.
     control%max_gltr_its_for_newton = 20

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of the gradient of the objective function is smaller than %stop_g

     control%stop_g = tenm5

!  Convergence of the subspace minimization occurs as sson as the subspace
!  gradient relative to the gradient is smaller than %stop_relative_subspace_g

     control%stop_relative_subspace_g = SQRT( epsmch )

!  The maximum CPU time allowed

      control%cpu_time_limit = - one

!   The smallest value the onjective function may take before the problem
!     is marked as unbounded

      control%obj_unbounded = - epsmch ** ( - 2 )

!  If space_critical is true, every effort will be made to use as little
!  space as possible. This may result in longer computation times

     control%space_critical = .FALSE.

!   If deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

     control%deallocate_error_fatal  = .FALSE.

!  Each line of output from ERMO will be prefixed by %prefix

     control%prefix = '""     '

     data%branch = 1

     RETURN

!  End of subroutine ERMO_initialize

     END SUBROUTINE ERMO_initialize

!-*-*-*-*-   E R M O _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE ERMO_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by ERMO_initialize could (roughly)
!  have been set as:

! BEGIN ERMO SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  alive-device                                    40
!  print-level                                     0
!  maximum-number-of-iterations                    100
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  maximum-levels                                  100
!  maximum-dimension-of-coarsest-subproblem        100
!  maximum_gltr_iterations_for_newton_direction    20
!  model-used                                      2
!  preconditioner-used                             1
!  multilevel-cycle-stategy-used                   2
!  gradient-accuracy-required                      1.0D-5
!  relative-subspace-gradient-accuracy-required    1.0D-8
!  minimum-objective-before-unbounded              -1.0D+32
!  maximum-cpu-time-limit                          -1.0
!  include-gradient-in-subspace                    yes
!  include-newton-like-direction-in-subspace       yes
!  hessian-available                               yes
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END ERMO SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ERMO_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: lspec = 62
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'ERMO'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec(  1 )%keyword = 'error-printout-device'
     spec(  2 )%keyword = 'printout-device'
     spec(  4 )%keyword = 'print-level'
     spec(  5 )%keyword = 'maximum-number-of-iterations'
     spec(  6 )%keyword = 'start-print'
     spec(  7 )%keyword = 'stop-print'
     spec(  8 )%keyword = 'iterations-between-printing'
     spec(  3 )%keyword = 'alive-device'
     spec(  9 )%keyword = 'preconditioner-used'
     spec( 10 )%keyword = 'multilevel-cycle-stategy-used'
     spec( 12 )%keyword = 'maximum-levels'
     spec( 13 )%keyword = 'maximum-dimension-of-coarsest-subproblem'
     spec( 14 )%keyword = 'model-used'
     spec( 15 )%keyword = 'maximum_gltr_iterations_for_newton_direction'

!  Real key-words

     spec( 18 )%keyword = 'gradient-accuracy-required'
     spec( 20 )%keyword = 'relative-subspace-gradient-accuracy-required'
     spec( 28 )%keyword = 'minimum-objective-before-unbounded'
     spec( 44 )%keyword = 'maximum-cpu-time-limit'

!  Logical key-words

     spec( 30 )%keyword = 'include-gradient-in-subspace'
     spec( 31 )%keyword = 'include-newton-like-direction-in-subspace'
     spec( 34 )%keyword = 'hessian-available'
     spec( 32 )%keyword = 'space-critical'
     spec( 35 )%keyword = 'deallocate-error-fatal'

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

     CALL SPECFILE_assign_value( spec( 1 ), control%error,                     &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 2 ), control%out,                       &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 3 ), control%out,                       &
                                  control%alive_unit )
     CALL SPECFILE_assign_value( spec( 4 ), control%print_level,               &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 5 ), control%maxit,                     &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 6 ), control%start_print,               &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 7 ), control%stop_print,                &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 8 ), control%print_gap,                 &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 9 ), control%preconditioner,            &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 10 ), control%cycle,                    &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 12 ), control%max_level_coarsest,       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 13 ), control%max_coarse_dim,           &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 14 ), control%model,                    &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 15 ), control%max_gltr_its_for_newton,  &
                                  control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( 18 ), control%stop_g,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 20 ),                                   &
                                 control%stop_relative_subspace_g,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 28 ), control%obj_unbounded,            &
                                  control%error )
     CALL SPECFILE_assign_value( spec( 44 ), control%cpu_time_limit,           &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( 30 ),                                   &
                                 control%include_gradient_in_subspace,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 31 ),                                   &
                                 control%include_newton_like_in_subspace,      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 34 ), control%hessian_available,        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 32 ), control%space_critical,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( 35 ),                                   &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( lspec ), control%alive_file,            &
                                 control%error )

!  Read the controls for the preconditioner and iterative solver

     IF ( PRESENT( alt_specname ) ) THEN
       CALL GLTR_read_specfile( control%GLTR_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-GLTR' )
       CALL TRU_read_specfile( control%TRU_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-TRU' )
     ELSE
       CALL GLTR_read_specfile( control%GLTR_control, device )
       CALL TRU_read_specfile( control%TRU_control, device )
     END IF

     RETURN

     END SUBROUTINE ERMO_read_specfile

!-*-*-*-  G A L A H A D -  E R M O _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE ERMO_solve( nlp, control, inform, data, userdata,              &
                            eval_F, eval_G, eval_H, eval_HPROD, eval_PREC )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  ERMO_solve, an iterated-subspace minimization method for finding a local
!    unconstrained minimizer of a given function

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_ERMO.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_ERMO_double
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
!    entries in the lower triangular part of H in the sparse co-ordinate storage
!    scheme. It need not be set for any of the other three schemes.
!
!   H%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the  lower triangular part of the Hessian
!    matrix H in any of the available storage schemes.
!
!   H%row is a rank-one allocatable array of type default integer, that holds the
!    row indices of the  lower triangular part of H in the sparse co-ordinate
!    storage scheme. It need not be allocated for any of the other three schemes
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
!   that holds the values x of the optimization variables. The j-th component
!   of X, j = 1, ... , n, contains x_j.
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
! control is a scalar variable of type ERMO_control_type. See ERMO_initialize
!  for details
!
! inform is a scalar variable of type ERMO_inform_type. On initial entry,
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
!        value should be set in nlp%f, and data%eval_status should be set to 0.
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
!        subroutine. The value l-th component of the Hessian stored according to
!        the scheme input in the remainder of nlp%H should be set in
!        nlp%H%val(l), for l = 1, ..., nlp%H%ne and data%eval_status should
!        be set to 0. If the user is unable to evaluate a component of
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
!        resulting vector u = P(x)v should be set in data%U and data%eval_status
!        should be set to 0. If the user is unable to evaluate the product -
!        for instance, if a component of the preconditioner is undefined at x -
!        the user need not set data%U, but should then set data%eval_status to
!        a non-zero value.
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
!  f_eval is a scalar variable of type default integer, that gives the
!   total number of objective function evaluations performed.
!
!  g_eval is a scalar variable of type default integer, that gives the
!   total number of objective function gradient evaluations performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of objective function Hessian evaluations performed.
!
!  h_prod is a scalar variable of type default integer, that gives the
!   total number of Hessian-vector priducts performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_g is a scalar variable of type default real, that holds the
!   value of the norm of the objective function gradient at the best estimate
!   of the solution found.
!
!  time is a scalar variable of type ERMO_time_type whose components are used
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
!  data is a scalar variable of type ERMO_data_type used for internal data.
!
!  userdata is a scalar variable of type NLPT_userdata_type which may be used
!   to pass user data to and from the eval_* subroutines (see below).
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
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_F is not present, ERMO_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in
!   G, and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_G is not present,
!   ERMO_solve will return to the user with inform%status = 3 each time an
!   evaluation is required.
!
!  eval_H is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) of the objective function evaluated at x=X must be returned in
!   H in the same order as presented in nlp%H, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_H is not present, ERMO_solve will return to the user with
!   inform%status = 4 each time an evaluation is required.
!
!  eval_HPROD is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The sum u + nabla_xx f(x) v of the
!   product of the Hessian nabla_xx f(x) of the objective function evaluated
!   at x=X with the vector v=V and the vector u=U must be returned in U, and the
!   status variable set to 0. If the evaluation is impossible at X, status
!   should be set to a nonzero value. If eval_HPROD is not present, ERMO_solve
!   will return to the user with inform%status = 5 each time an evaluation is
!   required.
!
!  eval_PREC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = P(x) v of the
!   user's preconditioner P(x) evaluated at x=X with the vector v=V, the result
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_PREC
!   is not present, ERMO_solve will return to the user with inform%status = 6
!   each time an evaluation is required.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( ERMO_control_type ), INTENT( IN ) :: control
     TYPE ( ERMO_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( ERMO_data_type ), INTENT( INOUT ) :: data
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

     INTEGER :: i, j, ic, ir, k, l, level, nnz, size_p
     LOGICAL :: alive, allocate_p, deallocate_p
     CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
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
     CASE ( 5 )  ! Hessian-vetor product
       GO TO 500
     CASE ( 6 )  ! objective or gradient evaluation or Hessian-vetor product
       GO TO 600
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
     inform%h_prod = 0
     inform%obj = HUGE( one ) ; inform%norm_g = HUGE( one )
     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''

     data%n_prolong = ( nlp%n + 1 ) / 2

!  allocate sufficient space for the problem

     array_name = 'ermo: data%X_current'
     CALL SPACE_resize_array( nlp%n, data%X_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ermo: data%G_current'
     CALL SPACE_resize_array( nlp%n, data%G_current, inform%status,            &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

!     array_name = 'ermo: data%P_ind'
!     CALL SPACE_resize_array( nlp%n, data%P_ind,                               &
!            inform%status, inform%alloc_status, array_name = array_name,       &
!            deallocate_error_fatal = control%deallocate_error_fatal,           &
!            exact_size = control%space_critical,                               &
!            bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ermo: data%P_ptr'
     CALL SPACE_resize_array( data%n_prolong + 1, data%P_ptr,                  &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ermo: data%U'
     CALL SPACE_resize_array( nlp%n, data%U, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     array_name = 'ermo: data%V'
     CALL SPACE_resize_array( nlp%n, data%V, inform%status,                    &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( inform%status /= 0 ) GO TO 980

     IF ( control%include_newton_like_in_subspace ) THEN
       array_name = 'ermo: data%S'
       CALL SPACE_resize_array( nlp%n, data%S_newton, inform%status,           &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  ensure that the data is consistent

     data%control = control

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
     END IF
     data%reverse_prec = .NOT. PRESENT( eval_PREC )

     data%nprec = data%control%preconditioner
     data%control%GLTR_control%unitm = .TRUE.

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
     data%print_level_tru = data%control%TRU_control%print_level
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

!    data%new_h = data%control%hessian_available
     data%new_h = .TRUE.

     IF ( data%printi ) WRITE( data%out, "( A, ' Problem: ', A,                &
    &   '  (n = ', I0, ')', / )" ) prefix, TRIM( nlp%pname ), nlp%n

!  assign variables (arbitrarily, pairwise) to the prolongation matrix

!     i = 0
!     data%P_ptr( 1 ) = 1
!     DO l = 1, data%n_prolong
!       i = i + 1
!       data%P_ind( i ) = i
!       IF ( i <= nlp%n ) THEN
!         i = i + 1
!         data%P_ind( i ) = i
!       END IF
!       data%P_ptr( l + 1 ) = i + 1
!     END DO

!  =======================
!  Start of main iteration
!  =======================

 310 CONTINUE

       data%got_f = .TRUE. ; data%got_g = .TRUE. ; data%got_h = .TRUE.

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = - 40
           RETURN
         END IF
       END IF

!  check to see if the iteration limit has not been exceeded

       inform%iter = inform%iter + 1
       IF ( inform%iter > data%control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print ) THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printd = data%set_printd
         data%print_level = data%control%print_level
         data%control%GLTR_control%print_level = data%print_level_gltr
         data%control%TRU_control%print_level = data%print_level_tru
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
         data%control%GLTR_control%print_level = 0
         data%control%TRU_control%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1 .OR.                 &
         data%control%GLTR_control%print_level > 0 .OR.                        &
         data%control%TRU_control%print_level > 0

!  =======================
!  1. Test for convergence
!  =======================

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header )         &
            WRITE( data%out, 2100 ) prefix
         data%print_1st_header = .FALSE.
         CALL CPU_TIME( data%time_total )
         data%time_total = data%time_total - data%time
         IF ( inform%iter > 1 ) THEN
           WRITE( data%out, 2120 ) prefix, inform%iter, inform%obj,            &
              inform%norm_g, inform%tru_inform%f_eval,                         &
              inform%tru_inform%g_eval, inform%h_prod, data%time_total
         ELSE
           WRITE( data%out, 2140 ) prefix,                                     &
                inform%iter, inform%obj, inform%norm_g
         END IF
       END IF
       IF ( inform%norm_g <= data%control%stop_g ) THEN
         inform%status = 0 ; GO TO 910
       END IF

!  debug printing

       IF ( data%out > 0 .AND. data%print_level > 4 ) THEN
         WRITE ( data%out, 2040 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2000 ) prefix, inform%f_eval, prefix, inform%g_eval,&
           prefix, inform%h_prod, prefix, inform%iter, prefix, inform%cg_iter, &
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

!  ================================
!  2. Calculate the search space, S
!  ================================

!  recompute the Hessian if it has changed

       IF ( data%new_h .AND. data%control%hessian_available ) THEN

!  form the Hessian or a preconditioner based on the Hessian

         IF ( data%reverse_h ) THEN
           data%branch = 4 ; inform%status = 4 ; RETURN
         ELSE
           CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                    &
                        userdata, nlp%H%val( : nlp%H%ne ) )
         END IF
       END IF

!  return from reverse communication to obtain the Hessian

 400   CONTINUE
       IF ( data%new_h .AND. data%control%hessian_available ) THEN
         inform%h_eval = inform%h_eval + 1

!  print the Hessian if desired

         IF ( data%printd ) THEN
           WRITE( data%out, "( A, ' Hessian ' )" ) prefix
           DO l = 1, nlp%H%ne
             WRITE( data%out, "( A, 2I7, ES24.16 )" ) prefix,                  &
               nlp%H%row( l ), nlp%H%col( l ), nlp%H%val( l )
           END DO
         END IF

         IF ( inform%h_eval == 1 ) THEN

!  transform H from co-ordinate form to compressed row format (both lower
!  and upper triangles must be present as must diagonals)

           array_name = 'ermo: data%row'
           CALL SPACE_resize_array( nlp%n, data%row, inform%status,            &
                  inform%alloc_status, array_name = array_name,                &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 980

!  find how many nonzeros there are in each row

           data%row( : nlp%n ) = 1
           DO l = 1, nlp%H%ne
             i = nlp%H%row( l ) ; j = nlp%H%col( l )
             IF ( i /= j ) THEN
               data%row( i ) = data%row( i ) + 1
               data%row( j ) = data%row( j ) + 1
             END IF
           END DO
           nnz = SUM( data%row( : nlp%n ) )
           CALL SMT_put( data%H_sym%type, 'general', inform%status )

!  allocate space to hold the matrix

           array_name = 'ermo: data%H_sym%val'
           CALL SPACE_resize_array( nnz, data%H_sym%val, inform%status,        &
                  inform%alloc_status, array_name = array_name,                &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 980

           array_name = 'ermo: data%H_sym%col'
           CALL SPACE_resize_array( nnz, data%H_sym%col, inform%status,        &
                  inform%alloc_status, array_name = array_name,                &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 980

           array_name = 'ermo: data%H_sym%ptr'
           CALL SPACE_resize_array( nlp%n + 1, data%H_sym%ptr, inform%status,  &
                  inform%alloc_status, array_name = array_name,                &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 980

!  fill the matrix; first find the starting addresses for each row

           data%H_sym%m = nlp%n
           data%H_sym%ptr( 1 ) = 1
           DO i = 1, nlp%n
             data%H_sym%ptr( i + 1 ) = data%H_sym%ptr( i ) + data%row( i )
             data%row( i ) = data%H_sym%ptr( i ) + 1
             data%H_sym%val( data%H_sym%ptr( i ) ) = one
           END DO

!  now insert the column indices and values

           DO l = 1, nlp%H%ne
             i = nlp%H%row( l ) ; j = nlp%H%col( l )
             IF ( i /= j ) THEN
               data%H_sym%val( data%row( i ) ) = nlp%H%val( l )
               data%H_sym%col( data%row( i ) ) = j
               data%row( i ) = data%row( i ) + 1
               data%H_sym%val( data%row( j ) ) = nlp%H%val( l )
               data%H_sym%col( data%row( j ) ) = i
               data%row( j ) = data%row( j ) + 1
             ELSE
!              data%H_sym%val( data%H_sym%ptr( i ) ) = nlp%H%val( l )
               data%H_sym%val( data%H_sym%ptr( i ) )                           &
                 = MAX( nlp%H%val( l ), ten ** ( - 8 ) )
               data%H_sym%col( data%H_sym%ptr( i ) ) = i
             END IF
           END DO

!  deallocate array row

           array_name = 'ermo: data%row'
           CALL SPACE_dealloc_array( data%row,                                 &
              inform%status, inform%alloc_status, array_name = array_name,     &
              bad_alloc = inform%bad_alloc, out = control%error )
           IF ( control%deallocate_error_fatal .AND. inform%status /= 0 )      &
             GO TO 980

!  Find the prolongation matrices

           data%mi20_control%max_levels = control%max_level_coarsest
           data%mi20_control%max_points = control%max_coarse_dim

           CALL mi20_setup( data%H_sym, data%mi20_coarse_data, data%mi20_keep, &
                            data%mi20_control, data%mi20_inform )

!  deallocate any leftover previous cascades of prolongation matrices

           deallocate_p = .FALSE. ; allocate_p = .TRUE.
           IF ( ALLOCATED( data%P ) ) THEN
             size_p = SIZE( data%P )
             IF ( control%space_critical ) THEN
               IF ( size_p /= control%max_level_coarsest ) THEN
                 deallocate_p = .TRUE.
               ELSE
                 allocate_p = .FALSE.
               END IF
             ELSE
               IF ( size_p < control%max_level_coarsest ) THEN
                 deallocate_p = .TRUE.
               ELSE
                 allocate_p = .FALSE.
               END IF
             END IF
           END IF

           IF ( deallocate_p ) THEN
             DO level = 1, size_p
               array_name = 'ermo: data%P'
               CALL SPACE_dealloc_smt_type( data%P( level ), inform%status,    &
                      inform%alloc_status, array_name = array_name,            &
                      bad_alloc = inform%bad_alloc, out = control%error )
               IF ( inform%status /= 0 ) GO TO 980
             END DO

             DEALLOCATE( data%P, STAT = inform%alloc_status )
             IF ( inform%alloc_status /= 0 ) THEN
               inform%status = GALAHAD_error_deallocate
               inform%bad_alloc = 'ermo: data%P'
               IF ( control%error > 0 ) WRITE( control%error,                  &
               "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")&
                 inform%bad_alloc, inform%alloc_status
               IF ( control%deallocate_error_fatal ) GO TO 980
             END IF
           END IF

!  allocate space to hold the new cascade of prolongation matrices ...

           data%level_coarsest = data%mi20_inform%clevels
           IF ( allocate_p ) ALLOCATE( data%P( data%level_coarsest ),          &
                                       STAT = inform%alloc_status )
           IF ( inform%alloc_status /= 0 ) THEN
             inform%status = GALAHAD_error_allocate
             inform%bad_alloc = 'ermo: data%P'
             IF ( control%error > 0 ) WRITE( control%error,                    &
               "( ' ** Allocation error for ', A, /, '     status = ', I6 )")  &
               inform%bad_alloc, inform%alloc_status
             IF ( control%deallocate_error_fatal ) GO TO 980
           END IF

!  ... and the matrices themselves. Copy the prolongation matrices from
!  data%mi20_coarse_data to data%P

           DO level = 1, data%level_coarsest
             data%P( level )%m = data%mi20_coarse_data( level )%I_mat%m
             data%P( level )%n = data%mi20_coarse_data( level )%I_mat%n
             CALL SMT_put( data%P( level )%type, 'general', inform%status )
write(6,*) ' level ', level,  ' prolongation dimensions m = ',                 &
        data%P( level )%m, ' n = ',  data%P( level )%n

!  values

             k = SIZE( data%mi20_coarse_data( level )%I_mat%val )
             array_name = 'ermo: data%P%val'
             CALL SPACE_resize_array( k, data%P( level )%val, inform%status,   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 980
             data%P( level )%val( : k ) =                                      &
               data%mi20_coarse_data( level )%I_mat%val( : k )

!  column indices

             k = SIZE( data%mi20_coarse_data( level )%I_mat%col )
             array_name = 'ermo: data%P%col'
             CALL SPACE_resize_array( k, data%P( level )%col, inform%status,   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 980
             data%P( level )%col( : k ) =                                      &
               data%mi20_coarse_data( level )%I_mat%col( : k )

!  row pointers

             k = SIZE( data%mi20_coarse_data( level )%I_mat%ptr )
             array_name = 'ermo: data%P%ptr'
             CALL SPACE_resize_array( k, data%P( level )%ptr, inform%status,   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 980
             data%P( level )%ptr( : k ) =                                      &
               data%mi20_coarse_data( level )%I_mat%ptr( : k )
           END DO

!  deallocate any leftover previous cascades of coarsened matrices

           deallocate_p = .FALSE. ; allocate_p = .TRUE.
           IF ( ALLOCATED( data%C ) ) THEN
             size_p = SIZE( data%C )
             IF ( control%space_critical ) THEN
               IF ( size_p /= control%max_level_coarsest ) THEN
                 deallocate_p = .TRUE.
               ELSE
                 allocate_p = .FALSE.
               END IF
             ELSE
               IF ( size_p < control%max_level_coarsest ) THEN
                 deallocate_p = .TRUE.
               ELSE
                 allocate_p = .FALSE.
               END IF
             END IF
           END IF

           IF ( deallocate_p ) THEN
             DO level = 1, size_p
               array_name = 'ermo: data%C'
               CALL SPACE_dealloc_smt_type( data%C( level ), inform%status,    &
                      inform%alloc_status, array_name = array_name,            &
                      bad_alloc = inform%bad_alloc, out = control%error )
               IF ( inform%status /= 0 ) GO TO 980
             END DO

             DEALLOCATE( data%C, STAT = inform%alloc_status )
             IF ( inform%alloc_status /= 0 ) THEN
               inform%status = GALAHAD_error_deallocate
               inform%bad_alloc = 'ermo: data%C'
               IF ( control%error > 0 ) WRITE( control%error,                  &
               "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")&
                 inform%bad_alloc, inform%alloc_status
               IF ( control%deallocate_error_fatal ) GO TO 980
             END IF
           END IF

!  allocate space to hold the new cascade of coarsened matrices ...

           data%level_coarsest = data%mi20_inform%clevels
           IF ( allocate_p ) ALLOCATE( data%C( data%level_coarsest ),          &
                                       STAT = inform%alloc_status )
           IF ( inform%alloc_status /= 0 ) THEN
             inform%status = GALAHAD_error_allocate
             inform%bad_alloc = 'ermo: data%C'
             IF ( control%error > 0 ) WRITE( control%error,                    &
               "( ' ** Allocation error for ', A, /, '     status = ', I6 )")  &
               inform%bad_alloc, inform%alloc_status
             IF ( control%deallocate_error_fatal ) GO TO 980
           END IF

!  ... and the matrices themselves.

           DO level = 1, data%level_coarsest
             data%C( level )%m = data%mi20_coarse_data( level )%A_mat%m
             data%C( level )%n = data%mi20_coarse_data( level )%A_mat%n
             CALL SMT_put( data%C( level )%type, 'general', inform%status )

!  values

             k = SIZE( data%mi20_coarse_data( level )%A_mat%val )
             array_name = 'ermo: data%C%val'
             CALL SPACE_resize_array( k, data%C( level )%val, inform%status,   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 980

!  column indices

             k = SIZE( data%mi20_coarse_data( level )%A_mat%col )
             array_name = 'ermo: data%C%col'
             CALL SPACE_resize_array( k, data%C( level )%col, inform%status,   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 980

!  row pointers

             k = SIZE( data%mi20_coarse_data( level )%A_mat%ptr )
             array_name = 'ermo: data%C%ptr'
             CALL SPACE_resize_array( k, data%C( level )%ptr, inform%status,   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 980
           END DO

!  deallocate data from the calculation

           CALL mi20_finalize( data%mi20_coarse_data, data%mi20_keep,          &
                               data%mi20_control, data%mi20_inform )

!  remove previous cascades of vectors

           deallocate_p = .FALSE. ; allocate_p = .TRUE.
           IF ( ALLOCATED( data%CASCADE ) ) THEN
             size_p = SIZE( data%CASCADE )
             IF ( control%space_critical ) THEN
               IF ( size_p /= control%max_level_coarsest ) THEN
                 deallocate_p = .TRUE.
               ELSE
                 allocate_p = .FALSE.
               END IF
             ELSE
               IF ( size_p < control%max_level_coarsest ) THEN
                 deallocate_p = .TRUE.
               ELSE
                 allocate_p = .FALSE.
               END IF
             END IF
           END IF

           IF ( deallocate_p ) THEN
             DO level = 1, size_p
               array_name = 'ermo: data%CASCADE%V'
               CALL SPACE_dealloc_array( data%CASCADE( level )%V,              &
                      inform%status,                                           &
                      inform%alloc_status, array_name = array_name,            &
                      bad_alloc = inform%bad_alloc, out = control%error )
               IF ( inform%status /= 0 ) GO TO 980
             END DO

             DEALLOCATE( data%CASCADE, STAT = inform%alloc_status )
             IF ( inform%alloc_status /= 0 ) THEN
               inform%status = GALAHAD_error_deallocate
               inform%bad_alloc = 'ermo: data%CASCADE'
               IF ( control%error > 0 ) WRITE( control%error,                  &
               "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")&
                 inform%bad_alloc, inform%alloc_status
               IF ( control%deallocate_error_fatal ) GO TO 980
             END IF
           END IF

!  allocate space to hold the new cascade of vetors ...

           IF ( allocate_p ) ALLOCATE( data%CASCADE( data%level_coarsest ),    &
                                       STAT = inform%alloc_status )
           IF ( inform%alloc_status /= 0 ) THEN
             inform%status = GALAHAD_error_allocate
             inform%bad_alloc = 'ermo: data%CASCADE'
             IF ( control%error > 0 ) WRITE( control%error,                    &
               "( ' ** Allocation error for ', A, /, '     status = ', I6 )")  &
               inform%bad_alloc, inform%alloc_status
             IF ( control%deallocate_error_fatal ) GO TO 980
           END IF

!  ... and the vectors themselves

           DO level = 1, data%level_coarsest
             array_name = 'ermo: data%P%val'
             CALL SPACE_resize_array( data%P( level )%n,                       &
               data%CASCADE( level )%V,                                        &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
             IF ( inform%status /= 0 ) GO TO 980
           END DO

!  now set dimensions for the coarse minimization

           data%level = data%level_coarsest
           data%level_coarsest_n = data%P( data%level_coarsest )%n
           CALL SMT_put( data%tru_nlp%H%type, 'DENSE', inform%status )

           array_name = 'ermo: data%tru_nlp%G'
!          CALL SPACE_resize_array( data%tru_nlp%n, data%tru_nlp%G,            &
           CALL SPACE_resize_array( nlp%n, data%tru_nlp%G,                     &
                  inform%status, inform%alloc_status, array_name = array_name, &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 980

           array_name = 'ermo: data%tru_nlp%X'
!          CALL SPACE_resize_array( data%tru_nlp%n, data%tru_nlp%X,            &
           CALL SPACE_resize_array( nlp%n, data%tru_nlp%X,                     &
                  inform%status, inform%alloc_status, array_name = array_name, &
                  deallocate_error_fatal = control%deallocate_error_fatal,     &
                  exact_size = control%space_critical,                         &
                  bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= 0 ) GO TO 980
         END IF
       END IF

!  ====================================
!  3. perform the subspace minimization
!  ====================================

!  set dimensions for the minimization on the current level

       data%level_n = data%P( data%level )%n
       data%tru_nlp%n = data%level_n
       IF ( control%include_gradient_in_subspace ) THEN
         data%tru_nlp%n = data%tru_nlp%n + 1 ; data%ig = data%tru_nlp%n
       END IF
       IF ( control%include_newton_like_in_subspace ) THEN
         data%tru_nlp%n = data%tru_nlp%n + 1 ; data%in = data%tru_nlp%n
       END IF

!  generate the nest of coarsened matrices

       CALL ERMO_generate( data%level, data%H_sym, data%P, data%C,             &
                           data%P_trans, data%C_times_P, data%control, inform )
       IF ( inform%status /= 0 ) GO TO 990

!  if a Newton-like direction is required, compute it using GLTR with
!  a reasonable radius

       IF ( .NOT. control%include_newton_like_in_subspace ) GO TO 510
       data%radius = ten ** 5
       data%G_current( : nlp%n )  = nlp%G( : nlp%n )
       data%model = zero ; data%S_newton( : nlp%n ) = zero
       inform%GLTR_inform%status = 1
       data%control%GLTR_control%itmax = data%control%max_gltr_its_for_newton

!  loop to compute the Newton-like direction
!  -----------------------------------------

  410  CONTINUE

!  perform a generalized Lanczos iteration

         CALL GLTR_solve( nlp%n, data%radius, data%model,                      &
                          data%S_newton( : nlp%n ),                            &
                          data%G_current( : nlp%n ), data%V( : nlp%n ),        &
                          data%GLTR_data, data%control%GLTR_control,           &
                          inform%GLTR_inform )
         SELECT CASE( inform%GLTR_inform%status )

!  form the preconditioned gradient

         CASE ( 2, 6 )

!  form the Hessian-vector product

         CASE ( 3, 7 )
           inform%h_prod = inform%h_prod + 1

!  if the Hessian is unavailable, obtain a matrix-free product

           data%U( : nlp%n ) = zero
           IF ( data%reverse_hprod ) THEN
             data%branch = 5 ; inform%status = 5 ; RETURN
           ELSE
             CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ), userdata,    &
                              data%U( : nlp%n ), data%V( : nlp%n ) )
           END IF

!  restore the gradient

         CASE ( 5 )
           data%G_current( : nlp%n )  = nlp%G( : nlp%n )

!  successful return

         CASE ( GALAHAD_ok, GALAHAD_warning_on_boundary,                       &
                GALAHAD_error_max_iterations )
write(6,"( 1X, I0, ' GLTR iterations to find Newton-like step' )" )  &
 inform%GLTR_inform%iter
           GO TO 510

!  error returns

         CASE DEFAULT
           IF ( data%printt ) WRITE( data%out, "( /,                           &
           &  A, ' Error return from GLTR, status = ', I0 )" ) prefix,         &
             inform%GLTR_inform%status
           inform%status = inform%GLTR_inform%status
           GO TO 990
         END SELECT

!  return from reverse communication to obtain the Hessian-vector product
!  or preconditioned vetor

  500    CONTINUE
!        IF ( .NOT. data%control%hessian_available ) THEN
           IF ( inform%GLTR_inform%status == 3 .OR.                            &
                inform%GLTR_inform%status == 7 ) THEN
             inform%h_eval = inform%h_eval + 1 ; data%got_h = .TRUE.
             data%V( : nlp%n ) = data%U( : nlp%n )
           END IF
!        END IF
         IF ( ( inform%GLTR_inform%status == 2 .OR.                            &
                inform%GLTR_inform%status == 6 ) .AND.                         &
                data%nprec == - 3 .AND. data%reverse_prec ) THEN
           data%V( : nlp%n ) = data%U( : nlp%n )
         END IF
         GO TO 410

!  end of GLTR loop

 510   CONTINUE

!  save current values

       data%X_current( : nlp%n ) = nlp%X( : nlp%n )
       data%G_current( : nlp%n ) = nlp%G( : nlp%n ) / inform%norm_g
       data%S_newton( : nlp%n ) = data%S_newton( : nlp%n ) /                   &
                                    inform%GLTR_inform%mnormx

!write(6,*) data%G_current(nlp%n/2), data%S_newton(nlp%n/2)

!  loop to perform subpsapce minimization
!  --------------------------------------

write(6,*) ' level ', data%level
       data%tru_nlp%X( : data%tru_nlp%n ) = zero
       data%control%TRU_control%stop_g_absolute =                              &
         MIN( control%TRU_control%stop_g_absolute, point1 * inform%norm_g )
       inform%tru_inform%status = 1 ; data%succ = 0
 520   CONTINUE
         CALL TRU_solve( data%tru_nlp, data%control%TRU_control,               &
                         inform%tru_inform, data%tru_data, data%tru_userdata )

!  test for reasonable termination

         IF ( inform%tru_inform%status == 0 .OR.                               &
              inform%tru_inform%status == GALAHAD_error_unbounded .OR.         &
              inform%tru_inform%status == GALAHAD_error_ill_conditioned .OR.   &
              inform%tru_inform%status == GALAHAD_error_tiny_step .OR.         &
              inform%tru_inform%status == GALAHAD_error_max_iterations ) THEN
           inform%obj = data%tru_nlp%f
           GO TO 550
         END IF

!  test for failure

         IF ( inform%tru_inform%status < 0 ) THEN
           WRITE( 6, "(  ' status = ', I0, ' on return from tru_solve'  )" )   &
             inform%tru_inform%status
           STOP
         END IF

!  obtain further information

         SELECT CASE ( inform%tru_inform%status )

         CASE ( 2 )

!  compute the new point x + S x_s if necessary; here
!  S = ( P_1 ... P_L : g_c : s_n ), where g_c is the current gradient and s_n
!  is the Newton-like direction

           IF ( .NOT. ( data%got_f .OR. data%got_g .OR. data%got_h ) ) THEN
             nlp%X( : nlp%n ) = data%X_current( : nlp%n )
             CALL ERMO_prolongate( data%level, data%P( : data%level ),         &
                                   data%CASCADE( : data%level ),               &
                                   data%tru_nlp%X( : data%level_n ),           &
                                   nlp%X( : nlp%n ),  V = nlp%X( : nlp%n ) )
             IF ( control%include_gradient_in_subspace )                       &
               nlp%X( : nlp%n ) =  nlp%X( : nlp%n ) +                          &
                  data%G_current( : nlp%n ) * data%tru_nlp%X( data%ig )
             IF ( control%include_newton_like_in_subspace )                    &
               nlp%X( : nlp%n ) =  nlp%X( : nlp%n ) +                          &
                  data%S_newton( : nlp%n ) * data%tru_nlp%X( data%in )
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

!  obtain the gradient

         CASE ( 3 )
           IF ( data%got_g ) THEN
             data%got_g = .FALSE.
           ELSE
             inform%g_eval = inform%g_eval + 1
             data%succ = data%succ + 1

!  exit if more than ? successful iterations have occured

             IF ( data%level < data%level_coarsest .AND. data%succ > 2 ) THEN
               inform%obj = data%tru_nlp%f
               GO TO 550
             END IF
             IF ( data%reverse_g ) THEN
               data%branch = 6 ; inform%status = 3 ; RETURN
             ELSE
               CALL eval_G( data%eval_status, nlp%X( : nlp%n ),                &
                            userdata, nlp%G( : nlp%n ) )
             END IF
           END IF

!  obtain the Hessian

         CASE ( 4 )
           IF ( data%got_h ) THEN
             data%got_h = .FALSE.
           ELSE
             inform%h_eval = inform%h_eval + 1
             IF ( data%reverse_h ) THEN
               data%branch = 6 ; inform%status = 4 ; RETURN
             ELSE
               CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                &
                            userdata, nlp%H%val( : nlp%H%ne ) )
             END IF
           END IF

!  obtain the Hessian-vector product

         CASE ( 5 )
           data%got_h = .FALSE.
           inform%h_prod = inform%h_prod + 1

!  form v = S * v_s

           IF ( control%include_gradient_in_subspace ) THEN
             IF ( control%include_newton_like_in_subspace ) THEN
               CALL ERMO_prolongate( data%level, data%P( : data%level ),       &
                                     data%CASCADE( : data%level ),             &
                                     data%tru_data%V( : data%level_n ),        &
                                     data%V( : nlp%n ),                        &
                                     V = data%G_current( : nlp%n ) *           &
                                         data%tru_data%V( data%ig )            &
                                       + data%S_newton( : nlp%n ) *            &
                                         data%tru_data%V( data%in ) )
             ELSE
               CALL ERMO_prolongate( data%level, data%P( : data%level ),       &
                                     data%CASCADE( : data%level ),             &
                                     data%tru_data%V( : data%level_n ),        &
                                     data%V( : nlp%n ),                        &
                                     V = data%G_current( : nlp%n ) *           &
                                         data%tru_data%V( data%ig ) )
             END IF
           ELSE
             IF ( control%include_newton_like_in_subspace ) THEN
               CALL ERMO_prolongate( data%level, data%P( : data%level ),       &
                                     data%CASCADE( : data%level ),             &
                                     data%tru_data%V( : data%level_n ),        &
                                     data%V( : nlp%n ),                        &
                                     V = data%S_newton( : nlp%n ) *            &
                                         data%tru_data%V( data%in ) )
             ELSE
               CALL ERMO_prolongate( data%level, data%P( : data%level ),       &
                                     data%CASCADE( : data%level ),             &
                                     data%tru_data%V( : data%level_n ),        &
                                     data%V( : nlp%n ) )
             END IF
           END IF

!  form H * S * v_s

           data%U( : nlp%n ) = zero
           IF ( data%reverse_h ) THEN
             data%branch = 6 ; inform%status = 5 ; RETURN
           ELSE
             CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ), userdata,    &
                              data%U( : nlp%n ), data%V( : nlp%n ) )
           END IF
         END SELECT

!  obtain further information

  600    CONTINUE
         SELECT CASE ( inform%tru_inform%status )

!  obtain the objective function

         CASE ( 2 )
           data%tru_nlp%f = nlp%f
           data%tru_data%eval_status = 0

!  obtain the gradient, g_s = S^T g; here S^T = ( P_L ... P_1 )
!                                               (     g_c^T   )
!                                               (     s_n^T   )

         CASE ( 3 )
           CALL ERMO_restrict( data%level, data%P( : data%level ),             &
                               data%CASCADE( : data%level ), nlp%G(: nlp%n ),  &
                               data%tru_nlp%G( : data%level_n ) )
           IF ( control%include_gradient_in_subspace )                         &
             data%tru_nlp%G( data%ig ) =                                       &
               DOT_PRODUCT( data%G_current( : nlp%n ) , nlp%G( : nlp%n ) )
           IF ( control%include_newton_like_in_subspace )                      &
             data%tru_nlp%G( data%in ) =                                       &
               DOT_PRODUCT( data%S_newton( : nlp%n ) , nlp%G( : nlp%n ) )

!  see if the relative gradient is small

           IF ( TWO_NORM( data%tru_nlp%G( : data%tru_nlp%n ) ) <               &
                data%control%stop_relative_subspace_g *                        &
                  TWO_NORM( nlp%G( : nlp%n ) ) ) THEN
             inform%obj = data%tru_nlp%f
             GO TO 550
           END IF
           data%tru_data%eval_status = 0

!  obtain the Hessian H_s = S^T * H * S

         CASE ( 4 )
           write(6,*) ' should not be computing H'
           STOP

!  form H * S

!  form S^T * H * S

!  obtain the Hessian-vector product

         CASE ( 5 )

!  form u_s = u_s + S^T * H * S v_s

           CALL ERMO_restrict( data%level, data%P( : data%level ),             &
                               data%CASCADE( : data%level ), data%U(: nlp%n ), &
                               data%tru_data%U( : data%level_n ),              &
                               U = data%tru_data%U( : data%level_n ) )
           IF ( control%include_gradient_in_subspace )                         &
             data%tru_data%U( data%ig ) =                                      &
               data%tru_data%U( data%ig ) +                                    &
               DOT_PRODUCT( data%G_current( : nlp%n ), data%U( : nlp%n ) )
           IF ( control%include_newton_like_in_subspace )                      &
             data%tru_data%U( data%in ) =                                      &
               data%tru_data%U( data%in ) +                                    &
               DOT_PRODUCT( data%S_newton( : nlp%n ), data%U( : nlp%n ) )
           data%tru_data%eval_status = 0
         END SELECT

       GO TO 520

!  end of subspace iteration loop

  550  CONTINUE

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

!  choose the next level in the cycle

       SELECT CASE ( data%control%cycle )

!  a V-cycle

       CASE ( 1 )
         data%level = MAX( data%level - 1, 1 )

!  a half V-cycle

       CASE ( 3 )

         IF ( data%level > 2 * data%level_coarsest / 3 ) THEN
           data%level = data%level - 1
         ELSE
           data%level = data%level_coarsest
         END IF

!  a multiple V-cycle

       CASE DEFAULT
         IF ( data%level > 1 ) THEN
           data%level = data%level - 1
         ELSE
           data%level = data%level_coarsest
         END IF
       END SELECT

     GO TO 310

!  =========================
!  End of the main iteration
!  =========================

 900 CONTINUE
     IF ( data%printi ) WRITE( data%out, "( A, ' Inform = ', I0, ' Stopping')")&
       prefix, inform%status

 910 CONTINUE

!  print details of solution

     inform%norm_g = TWO_NORM( nlp%G( : nlp%n ) )

     CALL CPU_TIME( data%time_new )
     inform%time%total = data%time_new - data%time

     IF ( data%printi ) THEN
       WRITE( data%out, "( /, A, '  Problem: ', A,                             &
      &     '  (n = ', I0, ')' )") prefix, TRIM( nlp%pname ), nlp%n
       IF ( control%include_gradient_in_subspace ) WRITE( data%out,            &
         "( A, '  gradient included in search subspace')") prefix
       IF ( control%include_newton_like_in_subspace ) WRITE( data%out,         &
         "( A, '  Newton-like direction included in search subspace')") prefix
!      WRITE( data%out,                                                        &
!        "( A, '  Direct solution of trust-region sub-problem')") prefix
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
 2100 FORMAT( A, '  It        f          grad       # f # grad',               &
             '    # h_pr        time ' )
 2120 FORMAT( A, I6, 2ES12.4, 2I7,I10,  F12.2 )
 2140 FORMAT( A, I6, 2ES12.4 )

 !  End of subroutine ERMO_solve

     END SUBROUTINE ERMO_solve

!-*-*-  G A L A H A D -  E R M O _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE ERMO_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ERMO_data_type ), INTENT( INOUT ) :: data
     TYPE ( ERMO_control_type ), INTENT( IN ) :: control
     TYPE ( ERMO_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: level, mc65_status
     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

!     array_name = 'ermo: data%P_ind'
!     CALL SPACE_dealloc_array( data%P_ind,                                     &
!        inform%status, inform%alloc_status, array_name = array_name,           &
!        bad_alloc = inform%bad_alloc, out = control%error )
!     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ermo: data%P_ptr'
     CALL SPACE_dealloc_array( data%P_ptr,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ermo: data%X_current'
     CALL SPACE_dealloc_array( data%X_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ermo: data%G_current'
     CALL SPACE_dealloc_array( data%G_current,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ermo: data%U'
     CALL SPACE_dealloc_array( data%U,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ermo: data%V'
     CALL SPACE_dealloc_array( data%V,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'ermo: data%S_newton'
     CALL SPACE_dealloc_array( data%S_newton,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     IF ( ALLOCATED( data%C ) ) THEN
       DO level = 1, SIZE( data%C )
         array_name = 'ermo: data%C'
         CALL SPACE_dealloc_smt_type( data%C( level ), inform%status,          &
                inform%alloc_status, array_name = array_name,                  &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
       END DO
       DEALLOCATE( data%C, STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_deallocate
         inform%bad_alloc = 'ermo: data%C'
         IF ( control%error > 0 ) WRITE( control%error,                        &
         "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")      &
           inform%bad_alloc, inform%alloc_status
         IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
       END IF
     END IF

     IF ( ALLOCATED( data%P ) ) THEN
       DO level = 1, SIZE( data%P )
         array_name = 'ermo: data%P'
         CALL SPACE_dealloc_smt_type( data%P( level ), inform%status,          &
                inform%alloc_status, array_name = array_name,                  &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
       END DO
       DEALLOCATE( data%P, STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_deallocate
         inform%bad_alloc = 'ermo: data%P'
         IF ( control%error > 0 ) WRITE( control%error,                        &
         "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")      &
           inform%bad_alloc, inform%alloc_status
         IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
       END IF
     END IF

     IF ( ALLOCATED( data%CASCADE ) ) THEN
       DO level = 1, SIZE( data%CASCADE )
         array_name = 'ermo: data%CASCADE%V'
         CALL SPACE_dealloc_array( data%CASCADE( level )%V, inform%status,     &
                inform%alloc_status, array_name = array_name,                  &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) RETURN
       END DO

       DEALLOCATE( data%CASCADE, STAT = inform%alloc_status )
       IF ( inform%alloc_status /= 0 ) THEN
         inform%status = GALAHAD_error_deallocate
         inform%bad_alloc = 'ermo: data%CASCADE'
         IF ( control%error > 0 ) WRITE( control%error,                        &
         "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")      &
           inform%bad_alloc, inform%alloc_status
         IF ( control%deallocate_error_fatal ) RETURN
       END IF
     END IF

! Deallocate temporary matices

     CALL MC65_matrix_destruct( data%C_times_P, mc65_status,                   &
                                stat = inform%alloc_status )
     IF ( mc65_status == MC65_ERR_MEMORY_DEALLOC ) THEN
       inform%status = GALAHAD_error_deallocate
       inform%bad_alloc = 'ermo: data%C_times_P'
       IF ( control%error > 0 ) WRITE( control%error,                          &
       "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")        &
         inform%bad_alloc, inform%alloc_status
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

     CALL MC65_matrix_destruct( data%P_trans, mc65_status,                     &
                                stat = inform%alloc_status )
     IF ( mc65_status == MC65_ERR_MEMORY_DEALLOC ) THEN
       inform%status = GALAHAD_error_deallocate
       inform%bad_alloc = 'ermo: data%P_trans'
       IF ( control%error > 0 ) WRITE( control%error,                          &
       "( ' ** Deallocation error for ', A, /, '     status = ', I6 )")        &
         inform%bad_alloc, inform%alloc_status
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

!  End of subroutine ERMO_terminate

     END SUBROUTINE ERMO_terminate

!-*-*  G A L A H A D -  E R M O _ p r o l o n g a t e  S U B R O U T I N E *-*-

     SUBROUTINE ERMO_prolongate( level_coarse, P, CASCADE, U, V_new, V )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  compute v_new = v + S u where S = P_1 ... P_L is the product of prelongation
!  matrices. If v is absent, v = 0 is assumed

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: level_coarse
     TYPE ( ZD11_type ), INTENT( IN ), DIMENSION( level_coarse ) :: P
     TYPE ( ERMO_cascade_type ), INTENT( INOUT ),                              &
                                 DIMENSION( level_coarse ) :: CASCADE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: U
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: V_new
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: V

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, level, rs, re

!  When there is more than one level ...

     IF ( level_coarse > 1 ) THEN

!  coarse level

       DO i = 1, P( level_coarse )%m
         rs = P( level_coarse )%ptr( i )
         re = P( level_coarse )%ptr( i + 1 ) - 1
         CASCADE( level_coarse - 1 )%V( i ) =                                  &
           DOT_PRODUCT( P( level_coarse )%val( rs : re ),                      &
                        U( P( level_coarse )%col( rs : re ) ) )
       END DO

!  intermediate levels

       DO level = level_coarse - 1, 2, - 1
         DO i = 1, P( level )%m
           rs = P( level )%ptr( i )
           re = P( level )%ptr( i + 1 ) - 1
           CASCADE( level - 1 )%V( i ) =                                       &
             DOT_PRODUCT(  P( level )%val( rs : re ),                          &
                           CASCADE( level )%V( P( level )%col( rs : re ) ) )
         END DO
       END DO

!  fine level

       IF ( PRESENT ( V ) ) THEN
         DO i = 1, P( 1 )%m
           rs = P( 1 )%ptr( i )
           re = P( 1 )%ptr( i + 1 ) - 1
           V_new( i ) = V( i ) + DOT_PRODUCT(                                  &
              P( 1 )%val( rs : re ), CASCADE( 1 )%V( P( 1 )%col( rs : re ) ) )
         END DO
       ELSE
         DO i = 1, P( 1 )%m
           rs = P( 1 )%ptr( i )
           re = P( 1 )%ptr( i + 1 ) - 1
           V_new( i ) = DOT_PRODUCT(                                           &
              P( 1 )%val( rs : re ), CASCADE( 1 )%V( P( 1 )%col( rs : re ) ) )
         END DO
       END IF

!  ... and when there is only a single level ...

     ELSE IF ( level_coarse == 1 ) THEN
       IF ( PRESENT ( V ) ) THEN
         DO i = 1, P( 1 )%m
           rs = P( 1 )%ptr( i )
           re = P( 1 )%ptr( i + 1 ) - 1
           V_new( i ) = V( i ) +                                               &
             DOT_PRODUCT( P( 1 )%val( rs : re ), U( P( 1 )%col( rs : re ) ) )
         END DO
       ELSE
         DO i = 1, P( 1 )%m
           rs = P( 1 )%ptr( i )
           re = P( 1 )%ptr( i + 1 ) - 1
           V_new( i ) =                                                        &
             DOT_PRODUCT( P( 1 )%val( rs : re ), U( P( 1 )%col( rs : re ) ) )
         END DO
       END IF

!  .. or no levels at all

     ELSE
       IF ( PRESENT ( V ) ) THEN
         V_new = V + U
       ELSE
         V_new = U
       END IF
     END IF

     RETURN

!  End of subroutine ERMO_prolongate

     END SUBROUTINE ERMO_prolongate

!-*-*-*-  G A L A H A D -  E R M O _ r e s t r i c t  S U B R O U T I N E  -*-*-
     SUBROUTINE ERMO_restrict( level_coarse, P, CASCADE, V, U_new, U )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  compute u_new = u + S^T v where S = P_1 ... P_L is the product of
!  prelongation matrices. If u is absent, u = 0 is assumed

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: level_coarse
     TYPE ( ZD11_type ), INTENT( IN ), DIMENSION( level_coarse ) :: P
     TYPE ( ERMO_cascade_type ), INTENT( INOUT ),                              &
                                 DIMENSION( level_coarse ) :: CASCADE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: V
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: U_new
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: U

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, level, rs, re

!  When there is more than one level ...

     IF ( level_coarse > 1 ) THEN

!  fine level

       CASCADE( 1 )%V( :  P( 1 )%n ) = zero
       DO i = 1, P( 1 )%m
         rs = P( 1 )%ptr( i )
         re = P( 1 )%ptr( i + 1 ) - 1
         CASCADE( 1 )%V( P( 1 )%col( rs : re ) ) =                             &
           CASCADE( 1 )%V( P( 1 )%col( rs : re ) ) +                           &
             P( 1 )%val( rs : re ) * V( i )
       END DO

!  intermediate levels

       DO level = 2, level_coarse - 1
         CASCADE( level )%V( :  P( level )%n ) = zero
         DO i = 1, P( level )%m
           rs = P( level )%ptr( i )
           re = P( level )%ptr( i + 1 ) - 1
           CASCADE( level )%V( P( level )%col( rs : re ) ) =                   &
             CASCADE( level )%V( P( level )%col( rs : re ) ) +                 &
               P( level )%val( rs : re ) * CASCADE( level - 1 )%V( i )
         END DO
       END DO

! coarse level

       IF ( PRESENT ( U ) ) THEN
         U_new( : P( level_coarse )%n ) = U( : P( level_coarse )%n )
       ELSE
         U_new( : P( level_coarse )%n ) = zero
       END IF
       DO i = 1, P( level_coarse )%m
         rs = P( level_coarse )%ptr( i )
         re = P( level_coarse )%ptr( i + 1 ) - 1
         U_new( P( level_coarse )%col( rs : re ) ) =                           &
           U_new( P( level_coarse )%col( rs : re ) ) +                         &
             P( level_coarse )%val( rs : re ) * CASCADE( level_coarse - 1 )%V(i )
       END DO

!  ... and when there is only a single level ...

     ELSE IF ( level_coarse == 1 ) THEN
       IF ( PRESENT ( U ) ) THEN
         U_new( : P( 1 )%n ) = U( : P( 1 )%n )
       ELSE
         U_new( : P( 1 )%n ) = zero
       END IF
       DO i = 1, P( 1 )%m
         rs = P( 1 )%ptr( i )
         re = P( 1 )%ptr( i + 1 ) - 1
         U_new( P( 1 )%col( rs : re ) ) =                                      &
           U_new( P( 1 )%col( rs : re ) ) + P( 1 )%val( rs : re ) * V( i )
       END DO

!  .. or no levels at all

     ELSE
       IF ( PRESENT ( U ) ) THEN
         U_new = U + V
       ELSE
         U_new = V
       END IF
     END IF

     RETURN

!  End of subroutine ERMO_restrict

     END SUBROUTINE ERMO_restrict

!-*-*-*  G A L A H A D -  E R M O _ g e n e r a t e  S U B R O U T I N E -*-*-*-

     SUBROUTINE ERMO_generate( level_coarse, C_0, P, C, P_trans, C_times_P,    &
                               control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Form the sequence of coarsened matrices C_i = P_i^T C_i-1 P_i
!  where P_i is the ith Prolongation matrix, i = 1,...,level_coarse
!  and C_0 is a given symmetric C. This routine is based on gen_coarse_matrix
!  from hsl_mi20; all matrices are stored in CSR format

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: level_coarse
     TYPE ( ZD11_type ), INTENT( IN ) :: C_0
     TYPE ( ZD11_type ), INTENT( IN ), DIMENSION( level_coarse ) :: P
     TYPE ( ZD11_type ), INTENT( INOUT ), DIMENSION( level_coarse ) :: C
     TYPE ( ZD11_type ), INTENT( INOUT ) :: P_trans, C_times_P
     TYPE ( ERMO_control_type ), INTENT( IN ) :: control
     TYPE ( ERMO_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: level, mc65_status

     inform%status = 0

!  loop over the sequence of coarsenings

     DO level = 1, level_coarse

! compute C_times_P = C_{i-1} * P

       IF ( level > 1 ) THEN
         CALL MC65_matrix_multiply( C( level - 1 ), P( level ), C_times_P,     &
                                    mc65_status, stat = inform%alloc_status )
       ELSE
         CALL MC65_matrix_multiply( C_0, P( 1 ), C_times_P,                    &
                                    mc65_status, stat = inform%alloc_status )
       END IF
       IF ( mc65_status == MC65_ERR_MEMORY_ALLOC ) THEN
         inform%status = GALAHAD_error_allocate
         inform%bad_alloc = 'ermo: data%C_times_P'
         RETURN
       ELSE IF ( mc65_status == MC65_ERR_MEMORY_DEALLOC .AND.                  &
                 control%deallocate_error_fatal ) THEN
         inform%status = GALAHAD_error_deallocate
         inform%bad_alloc = 'ermo: data%C_times_P'
         RETURN
       END IF

!  compute P_trans = P_i^T

       call MC65_matrix_transpose( P( level ), P_trans, mc65_status,           &
                                   stat = inform%alloc_status )
       IF ( mc65_status == MC65_ERR_MEMORY_ALLOC ) THEN
         inform%status = GALAHAD_error_allocate
         inform%bad_alloc = 'ermo: data%P_trans'
         RETURN
       ELSE IF ( mc65_status == MC65_ERR_MEMORY_DEALLOC .AND.                  &
                 control%deallocate_error_fatal ) THEN
         inform%status = GALAHAD_error_deallocate
         inform%bad_alloc = 'ermo: data%P_trans'
         RETURN
       END IF

!  find C_i = P_i^T * C_i-1 * P_i = P_trans * C_times_P

       CALL MC65_matrix_multiply( P_trans, C_times_P, C( level ),              &
                                  mc65_status, stat = inform%alloc_status )
       IF ( mc65_status == MC65_ERR_MEMORY_ALLOC ) THEN
         inform%status = GALAHAD_error_allocate
         inform%bad_alloc = 'ermo: data%C'
         RETURN
       ELSE IF ( mc65_status == MC65_ERR_MEMORY_DEALLOC .AND.                  &
                 control%deallocate_error_fatal ) THEN
         inform%status = GALAHAD_error_deallocate
         inform%bad_alloc = 'ermo: data%C'
         RETURN
       END IF
     END DO

     RETURN

!  End of subroutine ERMO_generate

     END SUBROUTINE ERMO_generate

!  End of module GALAHAD_ERMO

   END MODULE GALAHAD_ERMO_double

