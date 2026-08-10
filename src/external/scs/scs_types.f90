! THIS VERSION: GALAHAD 5.6 - 2026-08-10 AT 13:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D _ S C S _ T Y P E S    M O D U L E  -*-*-*-*-*-

!  module to provide fortran types for SCS C++ structures

!  Copyright reserved, GALAHAD productions
!  Principal authors: Alexis Montoison, Nick Gould and their pet AIs

!  History -
!   originally released in GALAHAD Version 5.6. August 10th 2026

  MODULE SCS_TYPES_precision
    USE GALAHAD_KINDS_precision, ONLY: ipc_, rpc_
    USE, INTRINSIC :: iso_c_binding
    IMPLICIT NONE ( TYPE, EXTERNAL )

    PRIVATE

!----------------------
!   P a r a m e t e r s
!----------------------

    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_NULL = 0
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_INFEASIBLE_INACCURATE = -7
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_UNBOUNDED_INACCURATE = -6
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_SIGINT = -5
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_FAILED = -4
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_INDETERMINATE = -3
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_INFEASIBLE = -2
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_UNBOUNDED = -1
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_UNFINISHED = 0
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_SOLVED = 1
    INTEGER ( ipc_ ), PUBLIC, PARAMETER :: SCS_SOLVED_INACCURATE = 2

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  derived type to hold Anderson Acceleration (AA) stats

    TYPE,  PUBLIC, BIND( C ) :: AaStats

!  Internal AA iteration counter.

      INTEGER ( ipc_ ) :: iter

!  Number of AA updates accepted by aa_apply before safeguarding.

      INTEGER ( ipc_ ) :: n_accept

!  Number of AA rejections due to LAPACK errors.

      INTEGER ( ipc_ ) :: n_reject_lapack

!  Number of AA rejections due to rank-zero reduced systems.

      INTEGER ( ipc_ ) :: n_reject_rank0

!  Number of AA rejections due to non-finite weights.

      INTEGER ( ipc_ ) :: n_reject_nonfinite

!  Number of AA rejections due to the weight-norm cap.

      INTEGER ( ipc_ ) :: n_reject_weight_cap

!  Number of AA steps rejected by safeguarding.

      INTEGER ( ipc_ ) :: n_safeguard_reject

!  Rank of the most recent AA solve.

      INTEGER ( ipc_ ) :: last_rank

!  Weight norm from the most recent AA solve. NaN if no solve was attempted.

      REAL ( rpc_ ) :: last_aa_norm

!  Regularization used in the most recent AA solve.

      REAL ( rpc_ ) :: last_regularization

    END TYPE AaStats

!  derived type to hold matrix structure

    TYPE,  PUBLIC, BIND( C ) :: ScsMatrix

!  Matrix values, size: number of non-zeros.

      TYPE ( c_ptr ) :: x

!  Matrix row indices, size: number of non-zeros.

      TYPE ( c_ptr ) :: i

!  Matrix column pointers, size: n+1.

      TYPE ( c_ptr ) :: p

!  Number of rows.

      INTEGER ( ipc_ ) :: m

!  Number of columns.

      INTEGER ( ipc_ ) :: n

    END TYPE ScsMatrix

!  derived type to hold SCS solver settings

    TYPE,  PUBLIC, BIND( C ) :: ScsSettings

!  Whether to heuristically rescale the data before solve.

      INTEGER ( ipc_ ) :: normalize

!  Initial dual scaling factor (may be updated if adaptive_scale is on).

      REAL ( rpc_ ) :: scale

!  Whether to adaptively update scale.

      INTEGER ( ipc_ ) :: adaptive_scale

!  Primal constraint scaling factor.

      REAL ( rpc_ ) :: rho_x

!  Maximum iterations to take.

      INTEGER ( ipc_ ) :: max_iters

!  Absolute convergence tolerance.

      REAL ( rpc_ ) :: eps_abs

!  Relative convergence tolerance.

      REAL ( rpc_ ) :: eps_rel

!  Infeasible convergence tolerance.

      REAL ( rpc_ ) :: eps_infeas

!  Douglas-Rachford relaxation parameter.

      REAL ( rpc_ ) :: alpha

!  Time limit in secs (can be fractional).

      REAL ( rpc_ ) :: time_limit_secs

!  Whether to log progress to stdout.

      INTEGER ( ipc_ ) :: verbose

!  Whether to use warm start (put initial guess in ScsSolution struct).

      INTEGER ( ipc_ ) :: warm_start

!  Memory for acceleration. Set to 0 to disable AA. Must be nonnegative.

      INTEGER ( ipc_ ) :: acceleration_lookback

!  Interval to apply acceleration.

      INTEGER ( ipc_ ) :: acceleration_interval

!  Whether AA uses type-I (1) or type-II (0).

      INTEGER ( ipc_ ) :: acceleration_type_1

!  Tikhonov regularization for the AA least-squares solve. See aa_init in...

      REAL ( rpc_ ) :: acceleration_regularization

!  AA relaxation factor in [0, 2]. 1.0 recovers vanilla AA.

      REAL ( rpc_ ) :: acceleration_relaxation

!  String, if set will dump raw prob data to this file.

      TYPE ( c_ptr ) :: write_data_filename

!  String, if set will log data to this csv file (makes SCS very slow).

      TYPE ( c_ptr ) :: log_csv_filename

    END TYPE ScsSettings

!  derived type to hold  SCS solver data

!  the problem is to minimize 1/2 x' P x + c'c
!  subject to A x + s = b, where s lies in an appropriate cone

    TYPE,  PUBLIC, BIND( C ) :: ScsData

!  A has m rows.

      INTEGER ( ipc_ ) :: m

!  A has n cols, P has n cols and n rows.

      INTEGER ( ipc_ ) :: n

!  A is supplied in CSC format (size m x n).

      TYPE ( c_ptr ) :: A

!  P is supplied in CSC format (size n x n), must be symmetric 
!  positive semidefinite. Only pass in the lower triangular part 
!  (including the diagonal)

      TYPE ( c_ptr ) :: P

!  Dense array for b (size m).

      TYPE ( c_ptr ) :: b

!  Dense array for c (size n).

      TYPE ( c_ptr ) :: c

    END TYPE ScsData

!  derived type to hold  SCS solver cone data

    TYPE,  PUBLIC, BIND( C ) :: ScsCone

!  Number of linear equality constraints (primal zero, dual free).

      INTEGER ( ipc_ ) :: z

!  Number of positive orthant cones.

      INTEGER ( ipc_ ) :: l

!  Upper box values, len(bu) = len(bl) = max(bsize-1, 0).

      TYPE ( c_ptr ) :: bu

!  Lower box values, len(bu) = len(bl) = max(bsize-1, 0).

      TYPE ( c_ptr ) :: bl

!  Total length of box cone (includes scale t).

      INTEGER ( ipc_ ) :: bsize

!  Array of second-order cone constraints, len(q) = qsize.

      TYPE ( c_ptr ) :: q

!  Length of second-order cone array q.

      INTEGER ( ipc_ ) :: qsize

!  Array of semidefinite cone constraints, len(s) = ssize.

      TYPE ( c_ptr ) :: s

!  Length of semidefinite constraints array s.

      INTEGER ( ipc_ ) :: ssize

!  Array of complex semidefinite cone constraints, len(cs) = cssize.

      TYPE ( c_ptr ) :: cs

!  Length of complex semidefinite constraints array cs.

      INTEGER ( ipc_ ) :: cssize

!  Number of primal exponential cone triples.

      INTEGER ( ipc_ ) :: ep

!  Number of dual exponential cone triples.

      INTEGER ( ipc_ ) :: ed

!  Array of power cone params, must be in [-1, 1], negative values are 
!  interpreted as specifying the dual power cone

      TYPE ( c_ptr ) :: p

!  Number of (primal and dual) power cone triples.

      INTEGER ( ipc_ ) :: psize

    END TYPE ScsCone

!  derived type to hold  SCS solution stats

    TYPE, BIND( C ) :: ScsSolution

!  Primal variable.

      TYPE ( c_ptr ) :: x

!  Dual variable.

      TYPE ( c_ptr ) :: y

!  Slack variable.

      TYPE ( c_ptr ) :: s

    END TYPE ScsSolution

!  derived type to hold SCS solver stats

    TYPE, BIND( c ) :: ScsInfo

!  Number of iterations taken.

      INTEGER ( ipc_ ) :: iter

!  Status string, e.g. 'solved'.

      CHARACTER ( c_char ) :: status( 128 )

!  Linear system solver used.

      CHARACTER ( c_char ) :: lin_sys_solver( 128 )

!  Status as scs_int, defined in glbopts.h.

      INTEGER ( ipc_ ) :: status_val

!  Number of updates to scale.

      INTEGER ( ipc_ ) :: scale_updates

!  Primal objective.

      REAL ( rpc_ ) :: pobj

!  Dual objective.

      REAL ( rpc_ ) :: dobj

!  Primal equality residual.

      REAL ( rpc_ ) :: res_pri

!  Dual equality residual.

      REAL ( rpc_ ) :: res_dual

!  Duality gap.

      REAL ( rpc_ ) :: gap

!  Infeasibility cert residual.

      REAL ( rpc_ ) :: res_infeas

!  Unbounded cert residual.

      REAL ( rpc_ ) :: res_unbdd_a

!  Unbounded cert residual.

      REAL ( rpc_ ) :: res_unbdd_p

!  Time taken for setup phase (milliseconds).

      REAL ( rpc_ ) :: setup_time

!  Time taken for solve phase (milliseconds).

      REAL ( rpc_ ) :: solve_time

!  Final scale parameter.

      REAL ( rpc_ ) :: scale

!  Complementary slackness.

      REAL ( rpc_ ) :: comp_slack

!  Number of rejected AA steps.

      INTEGER ( ipc_ ) :: rejected_accel_steps

!  Number of accepted AA steps.

      INTEGER ( ipc_ ) :: accepted_accel_steps

!  Detailed Anderson acceleration diagnostics.

      TYPE ( AaSta ts) :: aa_stats

!  Total time (milliseconds) spent in the linear system solver.

      REAL ( rpc_ ) :: lin_sys_time

!  Total time (milliseconds) spent in the cone projection.

      REAL ( rpc_ ) :: cone_time

!  Total time (milliseconds) spent in the acceleration routine.

      REAL ( rpc_ ) :: accel_time

    END TYPE ScsInfo

  END MODULE SCS_TYPES_precision
