! THIS VERSION: GALAHAD 5.6 - 2026-11-02 AT 15:20 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ H I G H S   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 5.6. October 26th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_HIGHS_precision

!     ------------------------------------------------
!     |                                              |
!     | Minimize the quadratic objective function    |
!     |                                              |
!     |         1/2 x^T H x + g^T x + f              |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |          c_l <= A x <= c_u                   |
!     |          x_l <=  x <= x_u                    |
!     |                                              |
!     | for some positive definite Hessian using     |
!     | the HIGHS package (which must be provided    |
!     | externally). See README.highs for details    |
!     | of how this is possible                      |
!     |                                              |
!     ------------------------------------------------

      USE GALAHAD_KINDS_precision, ONLY: i4_, ip_, rp_, ipc_, rpc_
      USE GALAHAD_CLOCK, ONLY: CLOCK_time
      USE GALAHAD_SYMBOLS, ONLY: GALAHAD_ok,                                   &
                                 GALAHAD_error_restrictions,                   &
                                 GALAHAD_error_primal_infeasible,              &
                                 GALAHAD_error_unbounded,                      &
                                 GALAHAD_no_progress,                          &
                                 GALAHAD_error_max_iterations,                 &
                                 GALAHAD_error_time_limit,                     &
                                 GALAHAD_error_inertia,                        &
                                 GALAHAD_unavailable_solver,                   &
                                 GALAHAD_error_presolve,                       &
                                 GALAHAD_error_highs
      USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
      USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read, &
                                            SPECFILE_assign_value
      USE GALAHAD_SMT_precision, ONLY: SMT_type, SMT_get, SMT_put
      USE GALAHAD_QPT_precision, ONLY: QPT_problem_type, QPT_keyword_A,        &
                                       QPT_keyword_H, QPT_summarize_problem
      USE GALAHAD_QPD_precision, ONLY: QPD_data_type, QPD_SIF
      USE GALAHAD_CONVERT_precision, ONLY: CONVERT_control_type,               &
                                       CONVERT_inform_type,                    &
                                       CONVERT_to_sparse_column_format,        &
                                       CONVERT_to_symmetric_sparse_column_format
      USE HIGHS_INTERFACE_precision, ONLY: c_ptr, c_char, c_bool,              &
                                           c_null_char, c_associated,          &
                                           Highs_create,                       &
                                           Highs_setIntOptionValue,            &
                                           Highs_setDoubleOptionValue,         &
                                           Highs_setBoolOptionValue,           &
                                           Highs_setStringOptionValue,         &
                                           Highs_passLp,                       &
                                           Highs_passHessian,                  &
                                           Highs_run,                          &
                                           Highs_getModelStatus,               &
                                           Highs_getDoubleInfoValue,           &
                                           Highs_getStringOptionValue,         &
                                           Highs_getIntInfoValue,              &
                                           Highs_getDoubleInfoValue,           &
                                           Highs_getSolution,                  &
                                           Highs_getBasis,                     &
                                           Highs_destroy

      IMPLICIT NONE ( TYPE, EXTERNAL )

      PRIVATE
      PUBLIC :: HIGHS_initialize, HIGHS_read_specfile, HIGHS_solve,            &
                HIGHS_terminate, QPT_problem_type, SMT_type, SMT_get, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

      INTERFACE HIGHS_initialize
        MODULE PROCEDURE HIGHS_initialize, HIGHS_full_initialize
      END INTERFACE HIGHS_initialize

      INTERFACE HIGHS_terminate
        MODULE PROCEDURE HIGHS_terminate, HIGHS_full_terminate
      END INTERFACE HIGHS_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER ( KIND = ip_ ), PARAMETER :: inf = HUGE( 1_ip_ )
      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
      LOGICAL ( KIND = c_bool ), PARAMETER :: logical_false = .FALSE.
      LOGICAL ( KIND = c_bool ), PARAMETER :: logical_true = .TRUE.

!---------------------------------
!   H i G H S  p a r a m e t e r s
!---------------------------------

!  the problem could not be passed to HiGHS due to an error (e.g., invalid
!  data or structural issues)

      INTEGER ( KIND = ip_ ), PARAMETER :: kHighsStatusError = - 1

!  The LP/QP data was parsed and passed successfully

      INTEGER ( KIND = ip_ ), PARAMETER :: kHighsStatusOk = 0

!  the function completed, but encountered minor inconsistencies or
!  non-fatal issues

      INTEGER ( KIND = ip_ ), PARAMETER :: kHighsStatusWarning = 1

!  the model status has not yet been set (e.g., before the solver has run)

      INTEGER ( KIND = ip_ ), PARAMETER :: kNotset = 0

!  a model could not be read from a data file (not used)

      INTEGER ( KIND = ip_ ), PARAMETER :: kLoadError = 1

!  the problem data was faulty

      INTEGER ( KIND = ip_ ), PARAMETER :: kModelError = 2

!  HiGHS's internal presolve routine crashed or encountered an unexpected
!  structural failure

      INTEGER ( KIND = ip_ ), PARAMETER :: kPresolveError = 3

!  an error occurred while attempting to solve the problem

      INTEGER ( KIND = ip_ ), PARAMETER :: kSolveError = 4

!  HiGHS failed to accurately reconstruct that solution back into the
!  original problem structure

      INTEGER ( KIND = ip_ ), PARAMETER :: kPostsolveError = 5

!  the problem contains no variables or constraints

      INTEGER ( KIND = ip_ ), PARAMETER :: kModelEmpty = 6

!  an optimal solution has been foun

      INTEGER ( KIND = ip_ ), PARAMETER :: kOptimal = 7

!  the problem is infeasible

      INTEGER ( KIND = ip_ ), PARAMETER :: kInfeasible = 8

!  the problem is primal infeasible or unbounded

      INTEGER ( KIND = ip_ ), PARAMETER :: kUnboundedOrInfeasible = 9

!  the problem is unbounded

      INTEGER ( KIND = ip_ ), PARAMETER :: kUnbounded = 10

!  the bound on the objective function has been reached

      INTEGER ( KIND = ip_ ), PARAMETER :: kObjectiveBound = 11

!  the target for objective function has been reached

      INTEGER ( KIND = ip_ ), PARAMETER :: kObjectiveTarget = 12

!  the time limit has been reached

      INTEGER ( KIND = ip_ ), PARAMETER :: kTimeLimit = 13

!  the iteration limit has been reached

      INTEGER ( KIND = ip_ ), PARAMETER :: kIterationLimit = 14

!  the final status of the problem cannot be determined

      INTEGER ( KIND = ip_ ), PARAMETER :: kUnknown = 15

!  the solution limit has been reached (not used, only for MILP)

      INTEGER ( KIND = ip_ ), PARAMETER :: kSolutionLimit = 16

!  the run has been interrupted by user

      INTEGER ( KIND = ip_ ), PARAMETER :: kInterrupt = 17

!  the memory limit has been reached

      INTEGER ( KIND = ip_ ), PARAMETER :: kMemoryLimit = 18

!  the run has been interrupted by HiGHS

      INTEGER ( KIND = ip_ ), PARAMETER :: kHighsInterrupt = 19

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!   extended control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: HIGHS_control_type

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!  presolve: 'off', 'choose' or 'on'

        CHARACTER ( LEN = 6 ) :: presolve = 'choose'

!  LP/QP solver: 'choose', 'simplex', 'ipm', 'ipx', 'hipo' or 'qpasm'

        CHARACTER ( LEN = 7 ) :: solver = 'choose '

!  parallel: 'off', 'choose' or 'on'

        CHARACTER ( LEN = 6 ) :: parallel = 'choose'

!  maximum number of threads used by HiGHS (0: automatic).
!  Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: threads = 0

!  run IPM crossover: 'off', 'choose' or 'on'

        CHARACTER ( LEN = 6 ) :: run_crossover = 'on    '

!  time limit (seconds). Range: [0, inf]

        REAL ( KIND = rp_ ) :: time_limit = inf

!  compute cost, bound, RHS and basic solution ranging: 'off' or 'on'

        CHARACTER ( LEN = 3 ) :: ranging = 'off'

!  limit on |cost coefficient|: values greater than or equal to this will
!  be treated as infinite. Range: [1e+15, inf]

        REAL ( KIND = rp_ ) :: infinite_cost = ten ** 20

!  limit on |variable/constraint bound|: values greater than or equal to
!  this will be treated as infinite. Range: [1e+15, inf]

        REAL ( KIND = rp_ ) :: infinite_bound = ten ** 20

!  lower limit on |matrix entries|: values less than or equal to this
!  will be treated as zero. Range: [1e-12, inf]

        REAL ( KIND = rp_ ) :: small_matrix_value = ten ** ( - 9 )

!  upper limit on |matrix entries|: values greater than or equal to this
!  will be treated as infinite. Range: [1, inf]

        REAL ( KIND = rp_ ) :: large_matrix_value = ten ** 15

!  if changed from its default value, this tolerance is used for all
!  feasibility and optimality (KKT) measures. Range: [1e-10, inf]

        REAL ( KIND = rp_ ) :: kkt_tolerance = ten ** ( - 7 )

!  primal feasibility tolerance. Range: [1e-10, inf]

        REAL ( KIND = rp_ ) :: primal_feasibility_tolerance = ten ** ( - 7 )

!  dual feasibility tolerance. Range: [1e-10, inf]

        REAL ( KIND = rp_ ) :: dual_feasibility_tolerance = ten ** ( - 7 )

!  primal residual tolerance. Range: [1e-10, inf]

        REAL ( KIND = rp_ ) :: primal_residual_tolerance = ten ** ( - 7 )

!  dual residual tolerance. Range: [1e-10, inf]

        REAL ( KIND = rp_ ) :: dual_residual_tolerance = ten ** ( - 7 )

!  optimality tolerance. Range: [1e-10, inf]

        REAL ( KIND = rp_ ) :: optimality_tolerance = ten ** ( - 7 )

!  objective bound for termination of the dual simplex solver.
!  Range: [-inf, inf]

        REAL ( KIND = rp_ ) :: objective_bound = inf

!  objective target for termination of the MIP solver. Range: [-inf, inf]

        REAL ( KIND = rp_ ) :: objective_target = -inf

!  random seed used in HiGHS. Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: random_seed = 0

!  exponent of power-of-two objective scaling for model.
!  Range: {-inf, inf}

        INTEGER ( KIND = ip_ ) :: user_objective_scale = 0

!  exponent of power-of-two bound scaling for model.
!  Range: {-inf, inf}

        INTEGER ( KIND = ip_ ) :: user_bound_scale = 0

!  strategy for simplex solver 0 => Choose; 1 => Dual (serial);
!  2 => Dual (SIP); 3 => Dual (PAMI); 4 => Primal. Range: {0, 4}

        INTEGER ( KIND = ip_ ) :: simplex_strategy = 1

!  simplex scaling strategy: off / choose / equilibration (default) /
!  forced equilibration / max value (0/1/2/3/4). Range: {0, 4}

        INTEGER ( KIND = ip_ ) :: simplex_scale_strategy = 2

!  strategy for simplex dual edge weights: Choose / Dantzig / Devex /
!  Steepest Edge (-1/0/1/2). Range: {-1, 2}

        INTEGER ( KIND = ip_ ) :: simplex_dual_edge_weight_strategy = - 1

!  strategy for simplex primal edge weights: Choose / Dantzig / Devex /
!  Steepest Edge (-1/0/1/2). Range: {-1, 2}

        INTEGER ( KIND = ip_ ) :: simplex_primal_edge_weight_strategy = - 1

!  iteration limit for simplex solver when solving LPs.
!  Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: simplex_iteration_limit = inf

!  limit on the number of simplex UPDATE operations.
!  Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: simplex_update_limit = 5000

!  maximum level of concurrency in parallel simplex. Range: {1, 8}

        INTEGER ( KIND = ip_ ) :: simplex_max_concurrency = 8

!  enables or disables solver output

        LOGICAL ( KIND = c_bool ) :: output_flag = logical_false

!  enables or disables console logging

        LOGICAL ( KIND = c_bool ) :: log_to_console = logical_true

!  log file

        CHARACTER ( LEN = 80 ) :: log_file = REPEAT( ' ', 80 )

!  write the model to a file

        LOGICAL ( KIND = c_bool ) :: write_model_to_file = logical_false

!  write the presolved model to a file

        LOGICAL ( KIND = c_bool ) :: write_presolved_model_to_file             &
                                       = logical_false

!  write the primal and dual solution to a file

        LOGICAL ( KIND = c_bool ) :: write_solution_to_file = logical_false

!  style of solution file (raw = computer-readable, pretty = human-readable):
!  -1 => HiGHS old raw (deprecated); 0 => HiGHS raw; 1 => HiGHS pretty;
!  2 => Glpsol raw; 3 => Glpsol pretty; 4 => HiGHS sparse raw.
!  Range: {-1, 4}

        INTEGER ( KIND = ip_ ) :: write_solution_style = 0

!  read solution file (do not read if string is empty)

        CHARACTER ( LEN = 80 ) :: read_solution_file = REPEAT( ' ', 80 )

!  read basis file (do not read if string is empty)

        CHARACTER ( LEN = 80 ) :: read_basis_file = REPEAT( ' ', 80 )

!  write model file (do not read if string is empty)

        CHARACTER ( LEN = 80 ) :: write_model_file = REPEAT( ' ', 80 )

!  write solution file (do not write if string is empty)

        CHARACTER ( LEN = 80 ) :: solution_file = REPEAT( ' ', 80 )

!  write basis file (do not write if string is empty)

        CHARACTER ( LEN = 80 ) :: write_basis_file = REPEAT( ' ', 80 )

!  write presolved model file (do not write if string is empty)

        CHARACTER ( LEN = 80 ) :: write_presolved_model_file = REPEAT( ' ', 80 )

!  write IIS model file (do not write if string is empty)

        CHARACTER ( LEN = 80 ) :: write_iis_model_file = REPEAT( ' ', 80 )

!  IPM optimality tolerance. Range: [1e-12, inf]

        REAL ( KIND = rp_ ) :: ipm_optimality_tolerance = ten ** ( - 8 )

!  iteration limit for IPM solver. Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: ipm_iteration_limit = inf

!  HiPO Newton system: 'choose', 'augmented' or 'normaleq'

        CHARACTER ( LEN = 10 ) :: hipo_system = 'choose    '

!  HiPO parallelism: 'tree', 'node', 'both' or 'choose'

        CHARACTER ( LEN = 6 ) :: hipo_parallel_type = 'choose'

!  bit map to force the use of parallel techniques in HiPO. Range: {0, 1023}

        INTEGER ( KIND = ip_ ) :: hipo_parallel_force = 0

!  bit map to forbid the use of parallel techniques in HiPO. Range: {0, 1023}

        INTEGER ( KIND = ip_ ) :: hipo_parallel_forbid = 0

!  HiPO matrix reordering: 'choose', 'metis', 'amd' or 'rcm'

        CHARACTER ( LEN = 6 ) :: hipo_ordering = 'choose'

!  HiPO matrix factorisation: 'choose', 'multifrontal', or 'uplooking'

        CHARACTER ( LEN = 6 ) :: hipo_fatcor = 'choose'

!  block size for dense linear algebra within HiPO. Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: hipo_block_size = 128

!  force OpenBLAS to run in serial when used by HiPO. This setting is
!  process-wide, so affects any other use of OpenBLAS

        LOGICAL ( KIND = c_bool ) :: hipo_serial_blas = logical_true

!  allow the active set QP solver to hot start

        LOGICAL ( KIND = c_bool ) :: qp_allow_hot_start = logical_false

!  iteration limit for the active set QP solver. Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: qp_iteration_limit = inf

!  nullspace limit for the active set QP solver.  Range: {0, inf}

        INTEGER ( KIND = ip_ ) :: qp_nullspace_limit  = 4000

!  regularization value added to the Hessian in the active set QP solver
!  Range: [0, inf]

        REAL ( KIND = rp_ ) :: qp_regularization_value = ten ** ( - 7 )

!  strategy for IIS (Irreducible Infeasible System) calculation:
!  0 => Light test; 1 => Try dual ray; 2 => Try elastic LP;
!  4 => Find true IIS; 8 => Prioritise columns;
!  16 => Find relaxation IIS for MIP. Range: {0, 31}

        INTEGER ( KIND = ip_ ) :: iis_strategy = 0

!  time limit for computing IIS (seconds). Range: [0, inf]

        REAL ( KIND = rp_ ) :: iis_time_limit = inf

!   if %use_hipo is true, the hipo inter-point QP solver will be used in
!   place of the default qpasm active-set solver

        LOGICAL :: use_hipo = .FALSE.

!   if %space_critical is true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE HIGHS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   extended inform derived type with defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: HIGHS_inform_type

!  return status

        INTEGER ( KIND = ip_ ) :: status = 0

!  return run status from highs

        INTEGER ( KIND = ip_ ) :: highs_status = 0

!  return status when passing the problem data to HiGHS

        INTEGER ( KIND = ip_ ) :: highs_pass_status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by HIGHS_solve

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the number of primal infeasibilities

        INTEGER ( KIND = ip_ ) :: primal_infeasibilities = - 1

!  the number of dual infeasibilities

        INTEGER ( KIND = ip_ ) :: dual_infeasibilities = - 1

!  the value of the primal infeasibility

        REAL ( KIND = rp_ ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = rp_ ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

        REAL ( KIND = rp_ ) :: complementary_slackness = HUGE( one )

!  the internal solver used (qpasm or hipo)

        CHARACTER ( KIND = c_char, LEN = 10 ) :: solver_used = REPEAT( ' ', 10 )

      END TYPE HIGHS_inform_type

!  - - - - - - - - - - - - - -
!   data derived type for HIGHS
!  - - - - - - - - - - - - - -

      TYPE, PUBLIC :: HIGHS_data_type
        INTEGER ( KIND = ip_ ) :: n, m, a_m, a_n, h_n, h_ne, a_ne
        INTEGER ( KIND = ip_ ) :: modelstatus, runstatus
        INTEGER ( KIND = ipc_ ) :: num_primal_infeasibilities
        INTEGER ( KIND = ipc_ ) :: num_dual_infeasibilities
        REAL ( KIND = rpc_ ) :: max_primal_infeasibility
        REAL ( KIND = rpc_ ) :: max_dual_infeasibility
        INTEGER ( KIND = ipc_ ) :: numcol, numrow, numnz, hessian_numnz
        INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr
        INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: A_row
        INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: H_ptr
        INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: H_row
        REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: A_val
        REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: H_val
        LOGICAL :: new_structure = .TRUE.
        TYPE ( c_ptr ) :: highs
      END TYPE HIGHS_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: HIGHS_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( HIGHS_data_type ) :: HIGHS_data
        TYPE ( HIGHS_control_type ) :: HIGHS_control
        TYPE ( HIGHS_inform_type ) :: HIGHS_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE HIGHS_full_data_type

   CONTAINS

!-*-*-*-*-   H I G H S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE HIGHS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for HIGHS. This routine should be called before
!  HIGHS_solve
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( HIGHS_data_type ), INTENT( INOUT ) :: data
      TYPE ( HIGHS_control_type ), INTENT( OUT ) :: control
      TYPE ( HIGHS_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN

!  End of HIGHS_initialize

      END SUBROUTINE HIGHS_initialize

! G A L A H A D -  H I G H S _ F U L L _ I N I T I A L I Z E  S U B R O U T INE

     SUBROUTINE HIGHS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for HIGHS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( HIGHS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( HIGHS_control_type ), INTENT( OUT ) :: control
     TYPE ( HIGHS_inform_type ), INTENT( OUT ) :: inform

     CALL HIGHS_initialize( data%HIGHS_data, control, inform )

     RETURN

!  End of subroutine HIGHS_full_initialize

     END SUBROUTINE HIGHS_full_initialize

!-*-*-*-   H I G H S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE HIGHS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by HIGHS_initialize could (roughly)
!  have been set as:

! BEGIN HIGHS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  presolve                                          choose
!  solver                                            choose
!  parallel                                          choose
!  threads                                           0
!  run-crossover                                     on
!  time-limit                                        1.79E+308
!  ranging                                           off
!  infinite-cost                                     1.0E+20
!  infinite-bound                                    1.0E+20
!  small-matrix-value                                1.0E-9
!  large-matrix-value                                1.0E+15
!  kkt-tolerance                                     1.0E-7
!  primal-feasibility-tolerance                      1.0E-7
!  dual-feasibility-tolerance                        1.0E-7
!  primal-residual-tolerance                         1.0E-7
!  dual-residual-tolerance                           1.0E-7
!  optimality-tolerance                              1.0E-7
!  objective-bound                                   1.79E+308
!  objective-target                                  -1.79E+308
!  random-seed                                       0
!  user-objective-scale                              0
!  user-bound-scale                                  0
!  simplex-strategy                                  1
!  simplex-scale-strategy                            2
!  simplex-dual-edge-weight-strategy                 -1
!  simplex-primal-edge-weight-strategy               -1
!  simplex-iteration-limit                           2147483647
!  simplex-update-limit                              5000
!  simplex-max-concurrency                           8
!  output-flag                                       T
!  log-to-console                                    T
!  log-file                                          ""
!  write-model-to-file                               F
!  write-presolved-model-to-file                     F
!  write-solution-to-file                            F
!  write-solution-style                              0
!  read-solution-file                                ""
!  read-basis-file                                   ""
!  write-model-file                                  ""
!  solution-file                                     ""
!  write-basis-file                                  ""
!  write-presolved-model-file                        ""
!  write-iis-model-file                              ""
!  ipm-optimality-tolerance                          1.0E-8
!  ipm-iteration-limit                               2147483647
!  hipo-system                                       choose
!  hipo-parallel-type                                choose
!  hipo-parallel-force                               0
!  hipo-parallel-forbid                              0
!  hipo-ordering                                     choose
!  hipo-fatcor                                       choose
!  hipo-block-size                                   128
!  hipo-serial-blas                                  T
!  qp-allow-hot-start                                F
!  qp-iteration-limit                                2147483647
!  qp-nullspace-limit                                4000
!  qp-regularization-value                           1.0E-7
!  iis-strategy                                      0
!  iis-time-limit                                    1.79E+308
!  use-hipo-not-qpasm                                F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END HIGHS SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( HIGHS_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: threads = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: random_seed = threads + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: user_objective_scale &
                                             = random_seed + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: user_bound_scale                    &
                                             = user_objective_scale + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: simplex_strategy                    &
                                             = user_bound_scale + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: simplex_scale_strategy              &
                                             = simplex_strategy + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: simplex_dual_edge_weight_strategy   &
                                             = simplex_scale_strategy + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: simplex_primal_edge_weight_strategy &
                                     = simplex_dual_edge_weight_strategy + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: simplex_iteration_limit             &
                                     = simplex_primal_edge_weight_strategy + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: simplex_update_limit                &
                                             = simplex_iteration_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: simplex_max_concurrency             &
                                             = simplex_update_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_solution_style                &
                                             = simplex_max_concurrency + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: ipm_iteration_limit                 &
                                             = write_solution_style + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_parallel_force                 &
                                             = ipm_iteration_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_parallel_forbid                &
                                             = hipo_parallel_force + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_block_size                     &
                                             = hipo_parallel_forbid + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qp_iteration_limit                  &
                                             = hipo_block_size + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qp_nullspace_limit                  &
                                             = qp_iteration_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: iis_strategy                        &
                                             = qp_nullspace_limit  + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: time_limit = iis_strategy + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinite_cost = time_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinite_bound = infinite_cost + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: small_matrix_value                  &
                                             = infinite_bound + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: large_matrix_value                  &
                                             = small_matrix_value + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: kkt_tolerance                       &
                                             = large_matrix_value + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: primal_feasibility_tolerance        &
                                             = kkt_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dual_feasibility_tolerance          &
                                             = primal_feasibility_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: primal_residual_tolerance           &
                                             = dual_feasibility_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dual_residual_tolerance             &
                                             = primal_residual_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: optimality_tolerance                &
                                             = dual_residual_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: objective_bound                     &
                                             = optimality_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: objective_target                    &
                                             = objective_bound + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: ipm_optimality_tolerance            &
                                             = objective_target + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qp_regularization_value             &
                                             = ipm_optimality_tolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: iis_time_limit                      &
                                             = qp_regularization_value + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: output_flag = iis_time_limit + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: log_to_console = output_flag + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_model_to_file                 &
                                             = log_to_console + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_presolved_model_to_file       &
                                             = write_model_to_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_solution_to_file              &
                                             = write_presolved_model_to_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_serial_blas                    &
                                             = write_solution_to_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: qp_allow_hot_start                  &
                                             = hipo_serial_blas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: presolve = qp_allow_hot_start + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: solver = presolve + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: parallel = solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: run_crossover = parallel + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: ranging = run_crossover + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: log_file = ranging + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: read_solution_file = log_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: read_basis_file                     &
                                             = read_solution_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_model_file                    &
                                             = read_basis_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: solution_file = write_model_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_basis_file = solution_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_presolved_model_file          &
                                             = write_basis_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: write_iis_model_file                &
                                             = write_presolved_model_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_system                         &
                                             = write_iis_model_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_parallel_type = hipo_system + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_ordering                       &
                                             = hipo_parallel_type + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hipo_fatcor = hipo_ordering + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: use_hipo  = hipo_fatcor + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = use_hipo + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 5 ), PARAMETER :: specname = 'HIGHS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( threads )%keyword = 'threads'
      spec( random_seed )%keyword = 'random-seed'
      spec( user_objective_scale )%keyword = 'user-objective-scale'
      spec( user_bound_scale )%keyword = 'user-bound-scale'
      spec( simplex_strategy )%keyword = 'simplex-strategy'
      spec( simplex_scale_strategy )%keyword = 'simplex-scale-strategy'
      spec( simplex_dual_edge_weight_strategy )%keyword                        &
        = 'simplex-dual-edge-weight-strategy'
      spec( simplex_primal_edge_weight_strategy )%keyword                      &
        = 'simplex-primal-edge-weight-strategy'
      spec( simplex_iteration_limit )%keyword = 'simplex-iteration-limit'
      spec( simplex_update_limit )%keyword = 'simplex-update-limit'
      spec( simplex_max_concurrency )%keyword = 'simplex-max-concurrency'
      spec( write_solution_style )%keyword = 'write-solution-style'
      spec( ipm_iteration_limit )%keyword = 'ipm-iteration-limit'
      spec( hipo_parallel_force )%keyword = 'hipo-parallel-force'
      spec( hipo_parallel_forbid )%keyword = 'hipo-parallel-forbid'
      spec( hipo_block_size )%keyword = 'hipo-block-size'
      spec( qp_iteration_limit )%keyword = 'qp-iteration-limit'
      spec( qp_nullspace_limit )%keyword = 'qp-nullspace-limit'
      spec( iis_strategy )%keyword = 'iis-strategy'

!  Real key-words

      spec( time_limit )%keyword = 'time-limit'
      spec( infinite_cost )%keyword = 'infinite-cost'
      spec( infinite_bound )%keyword = 'infinite-bound'
      spec( small_matrix_value )%keyword = 'small-matrix-value'
      spec( large_matrix_value )%keyword = 'large-matrix-value'
      spec( kkt_tolerance )%keyword = 'kkt-tolerance'
      spec( primal_feasibility_tolerance )%keyword                             &
        = 'primal-feasibility-tolerance'
      spec( dual_feasibility_tolerance )%keyword = 'dual-feasibility-tolerance'
      spec( primal_residual_tolerance )%keyword = 'primal-residual-tolerance'
      spec( dual_residual_tolerance )%keyword = 'dual-residual-tolerance'
      spec( optimality_tolerance )%keyword = 'optimality-tolerance'
      spec( objective_bound )%keyword = 'objective-bound'
      spec( objective_target )%keyword = 'objective-target'
      spec( ipm_optimality_tolerance )%keyword = 'ipm-optimality-tolerance'
      spec( qp_regularization_value )%keyword = 'qp-regularization-value'
      spec( iis_time_limit )%keyword = 'iis-time-limit'

!  Logical key-words
      spec( output_flag )%keyword = 'output-flag'
      spec( log_to_console )%keyword = 'log-to-console'
      spec( write_model_to_file )%keyword = 'write-model-to-file'
      spec( write_presolved_model_to_file )%keyword                            &
        = 'write-presolved-model-to-file'
      spec( write_solution_to_file )%keyword = 'write-solution-to-file'
      spec( hipo_serial_blas )%keyword = 'hipo-serial-blas'
      spec( qp_allow_hot_start )%keyword = 'qp-allow-hot-start'
      spec( use_hipo )%keyword = 'use-hipo-not-qpasm'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( presolve )%keyword = 'presolve'
      spec( solver )%keyword = 'solver'
      spec( parallel )%keyword = 'parallel'
      spec( run_crossover )%keyword = 'run-crossover'
      spec( ranging )%keyword = 'ranging'
      spec( log_file )%keyword = 'log-file'
      spec( read_solution_file )%keyword = 'read-solution-file'
      spec( read_basis_file )%keyword = 'read-basis-file'
      spec( write_model_file )%keyword = 'write-model-file'
      spec( solution_file )%keyword = 'solution-file'
      spec( write_basis_file )%keyword = 'write-basis-file'
      spec( write_presolved_model_file )%keyword = 'write-presolved-model-file'
      spec( write_iis_model_file )%keyword = 'write-iis-model-file'
      spec( hipo_system )%keyword = 'hipo-system'
      spec( hipo_parallel_type )%keyword = 'hipo-parallel-type'
      spec( hipo_ordering )%keyword = 'hipo-ordering'
      spec( hipo_fatcor )%keyword = 'hipo-fatcor'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' highs: ', alt_specname

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( threads ),                             &
                                  control%threads,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( random_seed ),                         &
                                  control%random_seed,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( user_objective_scale ),                &
                                  control%user_objective_scale,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( user_bound_scale ),                    &
                                  control%user_bound_scale,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( simplex_strategy ),                    &
                                  control%simplex_strategy,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( simplex_scale_strategy ),              &
                                  control%simplex_scale_strategy,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( simplex_dual_edge_weight_strategy ),   &
                                  control%simplex_dual_edge_weight_strategy,   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( simplex_primal_edge_weight_strategy),  &
                                  control%simplex_primal_edge_weight_strategy, &
                                  control%error )
      CALL SPECFILE_assign_value( spec( simplex_iteration_limit ),             &
                                  control%simplex_iteration_limit,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( simplex_update_limit ),                &
                                  control%simplex_update_limit,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( simplex_max_concurrency ),             &
                                  control%simplex_max_concurrency,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_solution_style ),                &
                                  control%write_solution_style,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( ipm_iteration_limit ),                 &
                                  control%ipm_iteration_limit,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_parallel_force ),                 &
                                  control%hipo_parallel_force,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_parallel_forbid ),                &
                                  control%hipo_parallel_forbid,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_block_size ),                     &
                                  control%hipo_block_size,                     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( qp_iteration_limit ),                  &
                                  control%qp_iteration_limit,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( qp_nullspace_limit ),                  &
                                  control%qp_nullspace_limit,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( iis_strategy ),                        &
                                  control%iis_strategy,                        &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( time_limit ),                          &
                                  control%time_limit,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( infinite_cost ),                       &
                                  control%infinite_cost,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( infinite_bound ),                      &
                                  control%infinite_bound,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( small_matrix_value ),                  &
                                  control%small_matrix_value,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( large_matrix_value ),                  &
                                  control%large_matrix_value,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( kkt_tolerance ),                       &
                                  control%kkt_tolerance,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( primal_feasibility_tolerance ),        &
                                  control%primal_feasibility_tolerance,        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( dual_feasibility_tolerance ),          &
                                  control%dual_feasibility_tolerance,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( primal_residual_tolerance ),           &
                                  control%primal_residual_tolerance,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( dual_residual_tolerance ),             &
                                  control%dual_residual_tolerance,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( optimality_tolerance ),                &
                                  control%optimality_tolerance,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( objective_bound ),                     &
                                  control%objective_bound,                     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( objective_target ),                    &
                                  control%objective_target,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( ipm_optimality_tolerance ),            &
                                  control%ipm_optimality_tolerance,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( qp_regularization_value ),             &
                                  control%qp_regularization_value,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( iis_time_limit ),                      &
                                  control%iis_time_limit,                      &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( output_flag ),                         &
                                  control%output_flag,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( log_to_console ),                      &
                                  control%log_to_console,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_model_to_file ),                 &
                                  control%write_model_to_file,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_presolved_model_to_file ),       &
                                  control%write_presolved_model_to_file,       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_solution_to_file ),              &
                                  control%write_solution_to_file,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_serial_blas ),                    &
                                  control%hipo_serial_blas,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( qp_allow_hot_start ),                  &
                                  control%qp_allow_hot_start,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( use_hipo  ),                           &
                                  control%use_hipo,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set character values

      CALL SPECFILE_assign_value( spec( presolve ),                            &
                                  control%presolve,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( solver ),                              &
                                  control%solver,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( parallel ),                            &
                                  control%parallel,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( run_crossover ),                       &
                                  control%run_crossover,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( ranging ),                             &
                                  control%ranging,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( log_file ),                            &
                                  control%log_file,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( read_solution_file ),                  &
                                  control%read_solution_file,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( read_basis_file ),                     &
                                  control%read_basis_file,                     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_model_file ),                    &
                                  control%write_model_file,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( solution_file ),                       &
                                  control%solution_file,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_basis_file ),                    &
                                  control%write_basis_file,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_presolved_model_file ),          &
                                  control%write_presolved_model_file,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( write_iis_model_file ),                &
                                  control%write_iis_model_file,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_system ),                         &
                                  control%hipo_system,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_parallel_type ),                  &
                                  control%hipo_parallel_type,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_ordering ),                       &
                                  control%hipo_ordering,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( hipo_fatcor ),                         &
                                  control%hipo_fatcor,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )
      RETURN

      END SUBROUTINE HIGHS_read_specfile

!-*-*-*-*-*-*-*-*-   H I G H S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE HIGHS_solve( prob, data, control, inform )

!  solve the quadratic program using the HIGHS package

!  A - by columns
!  H - upper triangle by columns (=> lower triangle by rows)

!  dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( HIGHS_data_type ), INTENT( INOUT ) :: data
      TYPE ( HIGHS_control_type ), INTENT( IN ) :: control
      TYPE ( HIGHS_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: m, n, status, simplex_iteration_count
      INTEGER ( KIND = ip_ ) :: ipm_iteration_count, qp_iteration_count
      TYPE ( CONVERT_control_type ) :: control_convert
      TYPE ( CONVERT_inform_type ) :: inform_convert
      LOGICAL :: qp, hipo, highs_logging
      CHARACTER ( LEN = 80 ) :: array_name

!  Highs parameters

      INTEGER ( KIND = ipc_ ), PARAMETER :: sense = 1
      INTEGER ( KIND = ipc_ ), PARAMETER :: a_colwise = 1
      INTEGER ( KIND = ipc_ ), PARAMETER :: h_triangular = 1
      INTEGER ( KIND = ipc_ ), PARAMETER :: modelstatus_optimal = 7
      INTEGER ( KIND = ipc_ ), PARAMETER :: runstatus_ok = 0
      INTEGER ( KIND = ipc_ ), PARAMETER :: runstatus_error = - 1
      INTEGER ( KIND = ipc_ ), PARAMETER :: runstatus_warning = - 1

!  transfer the data into HIGHS's QP format

!  record the problem dimensions

      n = prob%n ; m = prob%m

!  convert the input A into by sparse-column format

      prob%A%m = m ; prob%A%n = n
      CALL CONVERT_to_sparse_column_format( prob%A, data%a_m, data%a_n,        &
                                            data%a_ne, data%A_val,             &
                                            data%A_row, data%A_ptr,            &
                                            control_convert, inform_convert )

!  convert one-based indices to zero-based

      data%A_ptr( : n ) = data%A_ptr( : n ) - 1
      data%A_row( : data%a_ne ) = data%A_row( : data%a_ne ) - 1
!     write(6,*) ' A_ptr ', data%A_ptr( : n )
!     write(6,*) ' A_row ', data%A_row( : data%a_ne )

!  check to see in the problem is a QP

      SELECT CASE ( SMT_get( prob%H%type ) )
      CASE ( 'IDENTITY', 'SCALED_IDENTITY', 'DIAGONAL' )
        data%h_ne = n
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        data%h_ne = ( n * ( n + 1 ) ) / 2
      CASE ( 'SPARSE_BY_ROWS', 'SPARSE_BY_COLUMNS' )
        data%h_ne = prob%H%ptr( n + 1 ) - 1
      CASE ( 'COORDINATE' )
        data%h_ne = prob%H%ne
      CASE DEFAULT
        data%h_ne = 0
      END SELECT
      qp = data%h_ne > 0

!  convert the input H into by sparse-column format

      IF ( qp ) THEN
        prob%H%m = n ; prob%H%n = n
        CALL CONVERT_to_symmetric_sparse_column_format( prob%H, data%h_n,      &
                                                        data%h_ne, data%H_val, &
                                                        data%H_row, data%H_ptr,&
                                                        control_convert,       &
                                                        inform_convert )

!  convert one-based indices to zero-based

       data%H_ptr( : n ) = data%H_ptr( : n ) - 1
       data%H_row( : data%h_ne ) = data%H_row( : data%h_ne ) - 1
!      write(6,*) ' H_ptr ', data%H_ptr( : n )
!      write(6,*) ' H_row ', data%H_row( : data%h_ne )
      END IF

!  allocate other arrays needed by HiGHS

      IF ( data%new_structure ) THEN
        array_name = 'highs: prob%X'
        CALL SPACE_resize_array( n, prob%X, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'highs: prob%Y'
        CALL SPACE_resize_array( m, prob%Y, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'highs: prob%Z'
        CALL SPACE_resize_array( n, prob%Z, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'highs: prob%C'
        CALL SPACE_resize_array( m, prob%C, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'highs: prob%X_status'
        CALL SPACE_resize_array( n, prob%X_status, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'highs: prob%C_status'
        CALL SPACE_resize_array( m, prob%C_status, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

!  create the HiGHS environment

      CALL Highs_create( data%highs )

!  make sure that HiGHS is available

      IF ( .NOT. C_ASSOCIATED( data%highs ) ) THEN
        inform%status = GALAHAD_unavailable_solver
        IF ( control%error > 0 ) WRITE( control%error,                         &
          "( ' call to HiGHS failed, substitute dummy package called' )" )
        RETURN
      END IF

!  detect whether the hipo solver is preferred

      SELECT CASE ( TRIM( control%solver ) )
      CASE ( 'hipo', 'ipx' )
        hipo = .TRUE.
      CASE DEFAULT
        hipo = .FALSE.
      END SELECT

!  see if we want HiGHS to report output

      highs_logging = control%output_flag
!     highs_logging = control%output_flag .equiv logical_true

      IF ( highs_logging ) THEN
        CALL Highs_setBoolOptionValue( data%highs,                             &
                                       "output_flag" // C_NULL_CHAR,           &
                                       logical_true, status )
        CALL Highs_setBoolOptionValue( data%highs,                             &
                                       "log_to_console" // C_NULL_CHAR,        &
                                       logical_true, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
                                      "log_dev_level" // C_NULL_CHAR,          &
                                      1_ipc_, status )
      ELSE
        CALL Highs_setBoolOptionValue( data%highs,                             &
                                       "output_flag" // C_NULL_CHAR,           &
                                       logical_false, status )
      END IF

!  record general integer options

      CALL Highs_setIntOptionValue( data%highs,                                &
             "threads" // C_NULL_CHAR,                                         &
             control%threads, status )
      CALL Highs_setIntOptionValue( data%highs,                                &
             "random_seed" // C_NULL_CHAR,                                     &
             control%random_seed, status )
      CALL Highs_setIntOptionValue( data%highs,                                &
             "user_objective_scale" // C_NULL_CHAR,                            &
             control%user_objective_scale, status )
      CALL Highs_setIntOptionValue( data%highs,                                &
             "user_bound_scale" // C_NULL_CHAR,                                &
             control%user_bound_scale, status )
      CALL Highs_setIntOptionValue( data%highs,                                &
             "write_solution_style" // C_NULL_CHAR,                            &
             control%write_solution_style, status )
      CALL Highs_setIntOptionValue( data%highs,                                &
             "iis_strategy" // C_NULL_CHAR,                                    &
             control%iis_strategy, status )


!  record interior-point method specific integer options

      IF ( hipo ) THEN
        CALL Highs_setIntOptionValue( data%highs,                              &
               "hipo_parallel_force" // C_NULL_CHAR,                           &
               control%hipo_parallel_force, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "hipo_parallel_forbid" // C_NULL_CHAR,                          &
               control%hipo_parallel_forbid, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "hipo_block_size" // C_NULL_CHAR,                               &
               control%hipo_block_size, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "ipm_iteration_limit" // C_NULL_CHAR,                           &
               control%ipm_iteration_limit, status )

!  record active-set method specific integer options

      ELSE
        CALL Highs_setIntOptionValue( data%highs,                              &
               "qp_iteration_limit" // C_NULL_CHAR,                            &
               control%qp_iteration_limit, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "qp_nullspace_limit" // C_NULL_CHAR,                            &
               control%qp_nullspace_limit, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "simplex_strategy" // C_NULL_CHAR,                              &
               control%simplex_strategy, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "simplex_scale_strategy" // C_NULL_CHAR,                        &
               control%simplex_scale_strategy, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "simplex_dual_edge_weight_strategy" // C_NULL_CHAR,             &
               control%simplex_dual_edge_weight_strategy, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "simplex_primal_edge_weight_strategy" // C_NULL_CHAR,           &
               control%simplex_primal_edge_weight_strategy, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "simplex_iteration_limit" // C_NULL_CHAR,                       &
               control%simplex_iteration_limit, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "simplex_update_limit" // C_NULL_CHAR,                          &
               control%simplex_update_limit, status )
        CALL Highs_setIntOptionValue( data%highs,                              &
               "simplex_max_concurrency" // C_NULL_CHAR,                       &
               control%simplex_max_concurrency, status )
      END IF

!  record real options

      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "time_limit" // C_NULL_CHAR,                                      &
             control%time_limit, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "infinite_cost" // C_NULL_CHAR,                                   &
             control%infinite_cost, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "infinite_bound" // C_NULL_CHAR,                                  &
             control%infinite_bound, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "small_matrix_value" // C_NULL_CHAR,                              &
             control%small_matrix_value, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "large_matrix_value" // C_NULL_CHAR,                              &
             control%large_matrix_value, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "kkt_tolerance" // C_NULL_CHAR,                                   &
             control%kkt_tolerance, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "primal_feasibility_tolerance" // C_NULL_CHAR,                    &
             control%primal_feasibility_tolerance, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "dual_feasibility_tolerance" // C_NULL_CHAR,                      &
             control%dual_feasibility_tolerance, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "primal_residual_tolerance" // C_NULL_CHAR,                       &
             control%primal_residual_tolerance, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "dual_residual_tolerance" // C_NULL_CHAR,                         &
             control%dual_residual_tolerance, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "optimality_tolerance" // C_NULL_CHAR,                            &
             control%optimality_tolerance, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "objective_bound" // C_NULL_CHAR,                                 &
             control%objective_bound, status )
      CALL Highs_setDoubleOptionValue( data%highs,                             &
             "objective_target" // C_NULL_CHAR,                                &
             control%objective_target, status )

!  record interior-point method specific real options

      IF ( hipo ) THEN
        CALL Highs_setDoubleOptionValue( data%highs,                           &
               "ipm_optimality_tolerance" // C_NULL_CHAR,                      &
               control%ipm_optimality_tolerance, status )

!  record active-set method specific real options

      ELSE
        CALL Highs_setDoubleOptionValue( data%highs,                           &
               "qp_regularization_value" // C_NULL_CHAR,                       &
               control%qp_regularization_value, status )
      END IF

!  record linear-programming specific real options

      IF ( .NOT. qp ) THEN
        CALL Highs_setDoubleOptionValue( data%highs,                           &
               "iis_time_limit" // C_NULL_CHAR,                                &
               control%iis_time_limit, status )
      END IF

!  record logical options

      CALL Highs_setBoolOptionValue( data%highs,                               &
             "output_flag" // C_NULL_CHAR,                                     &
             control%output_flag, status )

!  record loging logical options

      IF ( highs_logging ) THEN
        CALL Highs_setBoolOptionValue( data%highs,                             &
               "log_to_console" // C_NULL_CHAR,                                &
               control%log_to_console, status )
        CALL Highs_setBoolOptionValue( data%highs,                             &
               "write_model_to_file" // C_NULL_CHAR,                           &
               control%write_model_to_file, status )
        CALL Highs_setBoolOptionValue( data%highs,                             &
               "write_presolved_model_to_file" // C_NULL_CHAR,                 &
               control%write_presolved_model_to_file, status )
        CALL Highs_setBoolOptionValue( data%highs,                             &
               "write_solution_to_file" // C_NULL_CHAR,                        &
               control%write_solution_to_file, status )
      END IF

!  record interior-point method specific logical options

      IF ( hipo ) THEN
        CALL Highs_setBoolOptionValue( data%highs,                             &
               "hipo_serial_blas" // C_NULL_CHAR,                              &
               control%hipo_serial_blas, status )

!  record active-set method specific logical options

      ELSE
        CALL Highs_setBoolOptionValue( data%highs,                             &
               "qp_allow_hot_start" // C_NULL_CHAR,                            &
               control%qp_allow_hot_start, status )
      END IF

!  record character options

      CALL Highs_setStringOptionValue( data%highs,                             &
             "solver" // C_NULL_CHAR,                                          &
             TRIM( control%solver ) // C_NULL_CHAR, status )
      CALL Highs_setStringOptionValue( data%highs,                             &
             "presolve" // C_NULL_CHAR,                                        &
             TRIM( control%presolve ) // C_NULL_CHAR, status )
      CALL Highs_setStringOptionValue( data%highs,                             &
             "parallel" // C_NULL_CHAR,                                        &
             TRIM( control%parallel ) // C_NULL_CHAR, status )
      CALL Highs_setStringOptionValue( data%highs,                             &
             "read_solution_file" // C_NULL_CHAR,                              &
             TRIM( control%read_solution_file ) // C_NULL_CHAR, status )
      CALL Highs_setStringOptionValue( data%highs,                             &
             "read_basis_file" // C_NULL_CHAR,                                 &
             TRIM( control%read_basis_file ) // C_NULL_CHAR, status )
      CALL Highs_setStringOptionValue( data%highs,                             &
             "write_model_file" // C_NULL_CHAR,                                &
             TRIM( control%write_model_file ) // C_NULL_CHAR, status )
      CALL Highs_setStringOptionValue( data%highs,                             &
             "solution_file" // C_NULL_CHAR,                                   &
             TRIM( control%solution_file ) // C_NULL_CHAR, status )
      CALL Highs_setStringOptionValue( data%highs,                             &
             "write_presolved_model_file" // C_NULL_CHAR,                      &
             TRIM( control%write_presolved_model_file ) // C_NULL_CHAR,        &
             status )
      IF ( highs_logging ) THEN
        CALL Highs_setStringOptionValue( data%highs,                           &
               "log_file" // C_NULL_CHAR,                                      &
               TRIM( control%log_file ) // C_NULL_CHAR, status )
      END IF

!  record interior-point method specific character options

      IF ( hipo ) THEN
        CALL Highs_setStringOptionValue( data%highs,                           &
               "ranging" // C_NULL_CHAR,                                       &
               TRIM( control%ranging ) // C_NULL_CHAR, status )
        CALL Highs_setStringOptionValue( data%highs,                           &
               "write_basis_file" // C_NULL_CHAR,                              &
               TRIM( control%write_basis_file ) // C_NULL_CHAR, status )
        CALL Highs_setStringOptionValue( data%highs,                           &
               "run_crossover" // C_NULL_CHAR,                                 &
               TRIM( control%run_crossover ) // C_NULL_CHAR, status )
        CALL Highs_setStringOptionValue( data%highs,                           &
               "hipo_system" // C_NULL_CHAR,                                   &
               TRIM( control%hipo_system ) // C_NULL_CHAR, status )
        CALL Highs_setStringOptionValue( data%highs,                           &
               "hipo_parallel_type" // C_NULL_CHAR,                            &
               TRIM( control%hipo_parallel_type ) // C_NULL_CHAR, status )
        CALL Highs_setStringOptionValue( data%highs,                           &
               "hipo_ordering" // C_NULL_CHAR,                                 &
               TRIM( control%hipo_ordering ) // C_NULL_CHAR, status )
        CALL Highs_setStringOptionValue( data%highs,                           &
               "hipo_fatcor" // C_NULL_CHAR,                                   &
               TRIM( control%hipo_fatcor ) // C_NULL_CHAR, status )

!  record active-set method specific character options

      ELSE
        CALL Highs_setStringOptionValue( data%highs,                           &
               "write_iis_model_file" // C_NULL_CHAR,                          &
               TRIM( control%write_iis_model_file ) // C_NULL_CHAR, status )
      END IF

!  set the solver

      CALL Highs_setStringOptionValue( data%highs, "solver" // C_NULL_CHAR,    &
                                       TRIM( control%solver ) // C_NULL_CHAR,  &
                                       status )

!  pass in the linear part of the problem data

      CALL Highs_passLp( data%highs, data%a_n, data%a_m, data%a_ne, a_colwise, &
                         sense, prob%f, prob%G( : n ), prob%X_l( : n ),        &
                         prob%X_u( : n ), prob%C_l( : m ), prob%C_u( : m ),    &
                         data%A_ptr( : n ), data%A_row( : data%a_ne ),         &
                         data%A_val( : data%a_ne ), inform%highs_pass_status )

!  record the model status

      SELECT CASE ( inform%highs_pass_status )
      CASE ( kHighsStatusOk, kHighsStatusWarning )
        inform%status = GALAHAD_ok
      CASE ( kHighsStatusError )
        inform%status = GALAHAD_error_restrictions
      CASE DEFAULT
        inform%status = GALAHAD_error_highs
      END SELECT
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  now pass in the quadratic part of the problem data (if any)

      IF ( qp ) THEN
        CALL Highs_passHessian( data%highs, data%a_n, data%h_ne, h_triangular, &
                                data%H_ptr( : n ), data%H_row( : data%h_ne ),  &
                                data%H_val( : data%a_ne ),                     &
                                inform%highs_pass_status )

!  record the model status

        SELECT CASE ( inform%highs_pass_status )
        CASE ( kHighsStatusOk, kHighsStatusWarning )
          inform%status = GALAHAD_ok
        CASE ( kHighsStatusError )
          inform%status = GALAHAD_error_restrictions
        CASE DEFAULT
          inform%status = GALAHAD_error_highs
        END SELECT
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

!  solve the problem

      CALL Highs_run( data%highs, inform%highs_status )

!  record the solve status

      SELECT CASE ( inform%highs_status )
      CASE ( kHighsStatusOk, kHighsStatusWarning )
        inform%status = GALAHAD_ok
      CASE ( kHighsStatusError )
        inform%status = GALAHAD_error_highs
      CASE DEFAULT
        inform%status = GALAHAD_error_highs
      END SELECT
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  record the solver exit status

      CALL Highs_getModelStatus( data%highs, inform%highs_status )

      SELECT CASE ( inform%highs_status )
      CASE (  kOptimal, kObjectiveBound, kObjectiveTarget )
        inform%status = GALAHAD_ok
      CASE ( kModelError, kModelEmpty  )
        inform%status = GALAHAD_error_restrictions
      CASE ( kPresolveError, kPostsolveError )
        inform%status = GALAHAD_error_presolve
      CASE ( kInfeasible, kUnboundedOrInfeasible )
        inform%status = GALAHAD_error_primal_infeasible
      CASE ( kUnbounded )
        inform%status = GALAHAD_error_unbounded
      CASE ( kIterationLimit, kSolutionLimit )
        inform%status = GALAHAD_error_max_iterations
      CASE ( kTimeLimit )
        inform%status = GALAHAD_error_time_limit
      CASE ( kNotset, kLoadError, kSolveError, kUnknown, kInterrupt,           &
             kHighsInterrupt )
        inform%status = GALAHAD_error_highs
      CASE DEFAULT
        inform%status = GALAHAD_error_highs
      END SELECT

!  record which HiGHS solver was actually used

      CALL Highs_getIntInfoValue( data%highs, "simplex_iteration_count",       &
                                  simplex_iteration_count, status )
      CALL Highs_getIntInfoValue( data%highs, "ipm_iteration_count",           &
                                  ipm_iteration_count, status )
      CALL Highs_getIntInfoValue( data%highs, "qp_iteration_count",            &
                                  qp_iteration_count, status )

      IF ( qp_iteration_count > 0) THEN
        inform%solver_used = 'qpasm'
      ELSE IF ( ipm_iteration_count > 0 ) THEN
        inform%solver_used = 'hipo'
      ELSE IF ( simplex_iteration_count > 0) THEN
        inform%solver_used = 'simplex'
      ELSE
        inform%solver_used = 'presolve'
      END IF

!  recover the objective function value and iteration count

      CALL Highs_getDoubleInfoValue( data%highs,                               &
                                    "objective_function_value" // C_NULL_CHAR, &
                                     inform%obj, status)
      IF ( qp ) THEN
        IF ( TRIM( inform%solver_used ) == 'hipo' ) THEN
          CALL Highs_getIntInfoValue( data%highs,                              &
            "ipm_iteration_count" // C_NULL_CHAR, inform%iter, status )
        ELSE
          CALL Highs_getIntInfoValue( data%highs,                              &
            "qp_iteration_count" // C_NULL_CHAR, inform%iter, status )
        END IF
      ELSE
        CALL Highs_getIntInfoValue( data%highs,                                &
          "simplex_iteration_count" // C_NULL_CHAR, inform%iter, status )
      END IF
!  WRITE( 6, "( 1X, I0, ' iterations required' )" ) inform%iter

!  get the primal and dual solution ...

      CALL Highs_getSolution( data%highs, prob%X( : n ), prob%Z( : n ),        &
                              prob%C( : m ), prob%Y( : m ), status )

!  ... and the basis

      CALL Highs_getBasis( data%highs, prob%X_status, prob%C_status, status )

!    IF ( status /= runstatus_ok ) THEN
!      WRITE( 6, "( ' Highs_lpCall run status is not ', I0, ' but ', I0 )" )   &
!        runstatus_ok, inform%highs_status
!    ELSE
!      WRITE( 6, "( ' Run status = ', I0, ', model status = ', I0 )" )         &
!         inform%highs_status, inform%highs_pass_status
!    END IF

!  compute the number and maximum values of the primal and dual infeasibilities

      CALL Highs_getIntInfoValue( data%highs,                                  &
                                  "num_primal_infeasibilities" // C_NULL_CHAR, &
                                  inform%primal_infeasibilities, status )
      CALL Highs_getDoubleInfoValue( data%highs,                               &
                                     "max_primal_infeasibility" // C_NULL_CHAR,&
                                     inform%primal_infeasibility, status )
      CALL Highs_getIntInfoValue( data%highs,                                  &
                                  "num_dual_infeasibilities" // C_NULL_CHAR,   &
                                  inform%dual_infeasibilities, status )
      CALL Highs_getDoubleInfoValue( data%highs,                               &
                                     "max_dual_infeasibility" // C_NULL_CHAR,  &
                                     inform%dual_infeasibility, status )

!  WRITE( 6, "( ' number of & maximum primal infeasibilities: ', I0, ES12.4 )")&
!    inform%primal_infeasibilities, inform%primal_infeasibility
!  WRITE( 6, "( ' number of & maximum dual infeasibilities: ', I0, ES12.4 )" ) &
!    inform%dual_infeasibilities, inform%dual_infeasibility

!  ensure that the existing structure remains for subsequent calls until
!  HIGHS_terminate removes it

      data%new_structure = .FALSE.

      RETURN

!  End of HIGHS_solve

      END SUBROUTINE HIGHS_solve

!-*-*-*-*-*-   H I G H S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE HIGHS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine HIGHS_initialize
!   control see Subroutine HIGHS_initialize
!   inform  see Subroutine HIGHS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( HIGHS_data_type ), INTENT( INOUT ) :: data
     TYPE ( HIGHS_control_type ), INTENT( IN ) :: control
     TYPE ( HIGHS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate arrays

     array_name = 'highs: data%A_val'
     CALL SPACE_dealloc_array( data%A_val, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%A_row'
     CALL SPACE_dealloc_array( data%A_row, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%A_ptr'
     CALL SPACE_dealloc_array( data%A_ptr, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%H_ptr'
     CALL SPACE_dealloc_array( data%H_ptr, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%H_row'
     CALL SPACE_dealloc_array( data%H_row, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%H_val'
     CALL SPACE_dealloc_array( data%H_val, inform%status,                      &
            inform%alloc_status, array_name = array_name,                      &
            bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  destroy the HiGHS environment

     CALL Highs_destroy( data%highs )

!  ensure that the structure will be re-initialised on any subsequent call

     data%new_structure = .TRUE.
     RETURN

!  End of subroutine HIGHS_terminate

     END SUBROUTINE HIGHS_terminate

!  G A L A H A D -  H I G H S _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE HIGHS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( HIGHS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( HIGHS_control_type ), INTENT( IN ) :: control
     TYPE ( HIGHS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL HIGHS_terminate( data%highs_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'highs: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%G'
     CALL SPACE_dealloc_array( data%prob%G,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%C'
     CALL SPACE_dealloc_array( data%prob%C,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%H_type'
     CALL SPACE_dealloc_array( data%prob%H%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%H%ptr'
     CALL SPACE_dealloc_array( data%prob%H%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%H%row'
     CALL SPACE_dealloc_array( data%prob%H%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%H%col'
     CALL SPACE_dealloc_array( data%prob%H%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%H%val'
     CALL SPACE_dealloc_array( data%prob%H%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%A%type'
     CALL SPACE_dealloc_array( data%prob%A%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%A%ptr'
     CALL SPACE_dealloc_array( data%prob%A%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%A%row'
     CALL SPACE_dealloc_array( data%prob%A%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%A%col'
     CALL SPACE_dealloc_array( data%prob%A%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'highs: data%prob%A%val'
     CALL SPACE_dealloc_array( data%prob%A%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine HIGHS_full_terminate

     END SUBROUTINE HIGHS_full_terminate

!  End of module HIGHS

   END MODULE GALAHAD_HIGHS_precision
