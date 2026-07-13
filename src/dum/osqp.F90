! THIS VERSION: GALAHAD 5.6 - 2026-07-11 AT 14:20 GMT.

!-*-*-*-*-  G A L A H A D  -  D U M M Y   O S Q P     M O D U L E  -*-*-*-*-

! This is the dummy Fortran interface for the OSQP solver.

MODULE OSQP

  USE iso_c_binding
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: OSQP_settings, OSQP_solve, OSQP_resolve, OSQP_cleanup

!  integer and real precisions as defined during osqp installation

#ifdef INTEGER_64
    INTEGER, PARAMETER :: ip = C_INT64_T
#else
    INTEGER, PARAMETER :: ip = C_INT32_T
#endif

#ifdef REAL_32
    INTEGER, PARAMETER :: wp = C_FLOAT
#elif REAL_128
    INTEGER, PARAMETER :: wp = C_FLOAT128
#else
    INTEGER, PARAMETER :: wp = C_DOUBLE
#endif

!  parameters

    REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
    REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( 1.0_wp )

!  --------------------------
!  OSQP_settings derived type
!  --------------------------

  TYPE, BIND( C ), PUBLIC :: OSQP_settings_type

!  linear algebra settings
!  -----------------------

!  device identifier; currently used for CUDA devices

    INTEGER ( KIND = ip ) :: device = 0               

!  linear system solver to use (1 = direct, 2 = iterative)

    INTEGER ( KIND = ip ) :: linsys_solver = 1

!  control settings
!  ----------------

!  allocate solution in OSQPSolver during osqp_setup (0 = no, 1 = yes)?

    INTEGER ( KIND = ip ) :: allocate_solution = 1               

!  write out progres (0 = no, 1 = yes)?

    INTEGER ( KIND = ip ) :: verbose = 1

!  level of detail for profiler annotations

    INTEGER ( KIND = ip ) :: profiler_level = 0                 

!  warm start (0 = no, 1 = yes)

    INTEGER ( KIND = ip ) :: warm_starting = 1

!  heuristic data scaling iterations; if 0, scaling disabled

    INTEGER ( KIND = ip ) :: scaling = 10

!  polish ADMM solution (0 = no, 1 = yes)?

    INTEGER ( KIND = ip ) :: polishing = 0

!  ADMM parameters
!  ---------------

!  penalty parameter

    REAL ( KIND = wp ) :: rho = ten ** ( - 1 )

!  is rho a scalar or a vector (0 = no, 1 = yes)?

#ifdef OSQP_ALGEBRA_CUDA
    INTEGER ( KIND = ip ) :: rho_is_vec = 0
#else
    INTEGER ( KIND = ip ) :: rho_is_vec = 1
#endif

!  step parameter

    REAL ( KIND = wp ) :: sigma = ten ** ( - 6 )

!  relaxation parameter

    REAL ( KIND = wp ) :: alpha = 1.6_wp

!  CG settings
!  -----------

!  maximum number of CG iterations per solve

    INTEGER ( KIND = ip ) :: cg_max_iter = 20

!  number of consecutive zero CG iterations before tolerance gets halved

    INTEGER ( KIND = ip ) :: cg_tol_reduction = 10

!  CG tolerance (fraction of ADMM residuals)

    REAL ( KIND = wp ) :: cg_tol_fraction = 0.15_wp

!  preconditioner to use in the CG method

    INTEGER ( KIND = ip ) :: cg_precond = 1

!  adaptive rho logic
!  ------------------

!  is rho step size adaptive (0 = no, 1 = yes)?

    INTEGER ( KIND = ip ) :: adaptive_rho = 1

!  number of iterations between rho adaptations rho. If 0, it is automatic

    INTEGER ( KIND = ip ) :: adaptive_rho_interval = 0

!  interval for adapting rho (fraction of the setup time)

    REAL ( KIND = wp ) :: adaptive_rho_fraction = 0.4_wp

!  tolerance X for adapting rho. The new rho has to be X times larger or 1/X
!  times smaller than the current one to trigger a new factorization.

#ifdef OSQP_ALGEBRA_CUDA
    REAL ( KIND = wp ) :: adaptive_rho_tolerance = 2.0_wp
#else
    REAL ( KIND = wp ) :: adaptive_rho_tolerance = 5.0_wp
#endif

!  termination parameters
!  ----------------------  

!  maximum iterations

    INTEGER ( KIND = ip ) :: max_iter = 4000

!  absolute convergence tolerance

    REAL ( KIND = wp ) :: eps_abs = ten ** ( - 3 )

!  relative convergence tolerance

    REAL ( KIND = wp ) :: eps_rel = ten ** ( - 3 )

!  primal infeasibility tolerance

    REAL ( KIND = wp ) :: eps_prim_inf = ten ** ( - 4 )

!  dual infeasibility tolerance

    REAL ( KIND = wp ) :: eps_dual_inf = ten ** ( - 4 )

!  use scaled termination criteria (0 = no, 1 = yes)?

    INTEGER ( KIND = ip ) :: scaled_termination = 0

!  check termination interval; if 0, termination checking is disabled

#ifdef OSQP_ALGEBRA_CUDA
    INTEGER ( KIND = ip ) :: check_termination = 5
#else
    INTEGER ( KIND = ip ) :: check_termination = 25
#endif

!  use duality gap termination criteria (0 = no, 1 = yes)?

#ifdef OSQP_USE_FLOAT
    INTEGER ( KIND = ip ) :: check_dualgap = 0
#else
    INTEGER ( KIND = ip ) :: check_dualgap = 1
#endif

!  maximum time to solve the problem (seconds) > 0

    REAL ( KIND = wp ) :: time_limit = ten ** 10

!  regularization parameter for polish

    REAL ( KIND = wp ) :: delta = ten ** ( - 6 )

!  iterative refinement steps in polish

    INTEGER ( KIND = ip ) :: polish_refine_iter = 3

  END TYPE OSQP_settings_type

!  ------------------------
!  OSQP_inform derived type
!  ------------------------

  TYPE, BIND( C ), PUBLIC :: OSQP_info_type

!  solver status
!  -------------

!  status string, e.g. 'solved'

!   CHARACTER ( KIND = c_char, LEN = 31 ) :: status = REPEAT( ' ', 31 )
    CHARACTER ( KIND = c_char ) :: status( 31 ) = ' '

!  status as defined in osqp_api_constants.h

    INTEGER ( KIND = ip ) :: status_val = - 10

!  polish status: successful (1), unperformed (0), (-1) unsuccessful

    INTEGER ( KIND = ip ) :: status_polish = 0

!  solution quality
!  ----------------

!  primal objective

    REAL ( KIND = wp ) :: obj_val = biginf

!  dual objective value

    REAL ( KIND = wp ) :: dual_obj_val = biginf

!  norm of primal residual

    REAL ( KIND = wp ) :: prim_res = biginf

!  norm of dual residual

    REAL ( KIND = wp ) :: dual_res = biginf

!  duality gap (primal objective - dual objective)

    REAL ( KIND = wp ) :: duality_gap = biginf

!  algorithm information
!  ---------------------

!  number of iterations taken

    INTEGER ( KIND = ip ) :: iter = - 1

!  number of rho updates

    INTEGER ( KIND = ip ) :: rho_updates = 0

!  best rho estimate so far from residuals

    REAL ( KIND = wp ) :: rho_estimate = biginf

!  timing information
!  ------------------

!  time taken for setup phase (seconds)

    REAL ( KIND = wp ) :: setup_time = 0.0_wp

!  time taken for solve phase (seconds)

    REAL ( KIND = wp ) :: solve_time = 0.0_wp

!  time taken for update phase (seconds)

    REAL ( KIND = wp ) :: update_time = 0.0_wp

!  time taken for polish phase (seconds)

    REAL ( KIND = wp ) :: polish_time = 0.0_wp

!  total time  (seconds)

    REAL ( KIND = wp ) :: run_time = 0.0_wp

!  convergence information
!  -----------------------

!  integral of duality gap over time (Primal-dual integral), requires profiling

    REAL ( KIND = wp ) :: primdual_int = biginf

!  relative KKT error

    REAL ( KIND = wp ) :: rel_kkt_error = biginf

  END TYPE OSQP_info_type

!  ----------------------
!  OSQP_data derived type
!  ----------------------

  TYPE, BIND( C ), PUBLIC :: OSQP_data_type

!  internal structures

    TYPE ( c_ptr ) :: c_settings
    TYPE ( c_ptr ) :: c_solver

  END TYPE OSQP_data_type

CONTAINS

!  copy settings into solver data

  SUBROUTINE OSQP_settings( settings, data, status )
  TYPE( OSQP_settings_type ), INTENT( IN ) :: settings
  TYPE( OSQP_data_type ), INTENT( INOUT ) :: data
  INTEGER ( ip ), INTENT( OUT ) :: status

  status = - 199  ! error code

  RETURN
  END SUBROUTINE OSQP_settings

!  solve the given problem

  SUBROUTINE OSQP_solve( n, m, P_ptr, P_row, P_val, q, A_ptr, A_row, A_val,    &
                         l, u, x, y, info, data, status )
  INTEGER ( KIND = ip ), INTENT( IN ) :: n, m
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( n + 1 ) :: P_ptr
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( P_ptr( n + 1 ) - 1  ) :: P_row
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( P_ptr( n + 1 ) - 1  ) :: P_val
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: q
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( n + 1 ) :: A_ptr
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( A_ptr( n + 1 ) - 1  ) :: A_row
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ptr( n + 1 ) - 1  ) :: A_val
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: l
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: u
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: y
  TYPE( OSQP_info_type ), INTENT( INOUT ) :: info
  TYPE( OSQP_data_type ), INTENT( INOUT ) :: data
  INTEGER ( ip ), INTENT( OUT ) :: status

  status = - 199 ! error code
  info%status_val = status

  RETURN
  END SUBROUTINE OSQP_solve

!  resolve the given problem

  SUBROUTINE OSQP_resolve( n, m, x, y, info, data, status,                     &
                           q_new, l_new, u_new, x_new, y_new )
  INTEGER ( KIND = ip ), INTENT( IN ) :: n, m
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: y
  TYPE( OSQP_info_type ), INTENT( INOUT ) :: info
  TYPE( OSQP_data_type ), INTENT( INOUT ) :: data
  INTEGER ( ip ), INTENT( OUT ) :: status
  REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: q_new
  REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: l_new
  REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: u_new
  REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: x_new
  REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: y_new

  status = - 199 ! error code
  info%status_val = status

  RETURN
  END SUBROUTINE OSQP_resolve

!  clean up after solution

  SUBROUTINE OSQP_cleanup( data, status )
  TYPE( OSQP_data_type ), INTENT( INOUT ) :: data
  INTEGER ( ip ), INTENT( OUT ) :: status

  status = - 199 ! error code

  RETURN

  END SUBROUTINE OSQP_cleanup

END MODULE OSQP
