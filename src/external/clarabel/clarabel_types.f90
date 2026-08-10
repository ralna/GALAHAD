! THIS VERSION: GALAHAD 5.6 - 2026-08-10 AT 13:40 GMT.

#include "galahad_modules.h"

!-*-*-*-*-  G A L A H A D _ C L A R A B E L  _ T Y P E S    M O D U L E  -*-*-*-

!  module to provide fortran types for Clarabel C++ structures

!  Copyright reserved, GALAHAD productions
!  Principal authors: Alexis Montoison, Nick Gould and their pet AIs

!  History -
!   originally released in GALAHAD Version 5.6. August 10th 2026

  MODULE CLARABEL_TYPES_precision
    USE, INTRINSIC :: iso_c_binding, ONLY: c_ptr, c_intptr_t, c_bool,          &
                                           c_float, c_double
    IMPLICIT NONE ( TYPE, EXTERNAL )

!  ClarabelDirectSolveMethods

    ENUM, BIND( C )
      ENUMERATOR :: AUTO = 0
      ENUMERATOR :: QDLDL = 1
    END ENUM

!  ClarabelSolverStatus

    ENUM, BIND( C )
      ENUMERATOR :: ClarabelUnsolved = 0
      ENUMERATOR :: ClarabelSolved = 1
      ENUMERATOR :: ClarabelPrimalInfeasible = 2
      ENUMERATOR :: ClarabelDualInfeasible = 3
      ENUMERATOR :: ClarabelAlmostSolved = 4
      ENUMERATOR :: ClarabelAlmostPrimalInfeasible = 5
      ENUMERATOR :: ClarabelAlmostDualInfeasible = 6
      ENUMERATOR :: ClarabelMaxIterations = 7
      ENUMERATOR :: ClarabelMaxTime = 8
      ENUMERATOR :: ClarabelNumericalError = 9
      ENUMERATOR :: ClarabelInsufficientProgress = 10
      ENUMERATOR :: ClarabelCallbackTerminated = 11
    END ENUM

!  ClarabelSupportedConeT_Tag

    ENUM, BIND( C )
      ENUMERATOR :: ClarabelZeroConeT_Tag = 0
      ENUMERATOR :: ClarabelNonnegativeConeT_Tag = 1
      ENUMERATOR :: ClarabelSecondOrderConeT_Tag = 2
      ENUMERATOR :: ClarabelExponentialConeT_Tag = 3
      ENUMERATOR :: ClarabelPowerConeT_Tag = 4
      ENUMERATOR :: ClarabelGenPowerConeT_Tag = 5
    END ENUM

    TYPE, BIND( C ) :: ClarabelCscMatrix

!  number of rows

      INTEGER ( c_intptr_t ) :: m

!  number of columns

      INTEGER ( c_intptr_t ) :: n 

!  CSC format 0-based column pointers. This field should have length n+1. 
!  The last entry corresponds to the number of nonzero entries

      TYPE ( c_ptr ) :: colptr

!  Vector of 0-based row indices. If this is a zero matrix, use `NULL` 
!  for this field.

      TYPE ( c_ptr ) :: rowval

!  Vector of non-zero matrix elements If this is a zero matrix, use `NULL` 
!  for this field.

      TYPE ( c_ptr ) :: nzval

    END TYPE ClarabelCscMatrix

    TYPE, BIND( C ) :: ClarabelCscMatrix_f32
      INTEGER ( c_intptr_t ) :: m
      INTEGER ( c_intptr_t ) :: n
      TYPE ( c_ptr ) :: colptr
      TYPE ( c_ptr ) :: rowval
      TYPE ( c_ptr ) :: nzval
    END TYPE ClarabelCscMatrix_f32

    TYPE, BIND( C ) :: ClarabelDefaultSettings
      INTEGER ( c_int32_t ) :: max_iter
      REAL ( c_double ) :: time_limit
      LOGICAL ( c_bool ) :: verbose
      REAL ( c_double ) :: max_step_fraction
      REAL ( c_double ) :: tol_gap_abs
      REAL ( c_double ) :: tol_gap_rel
      REAL ( c_double ) :: tol_feas
      REAL ( c_double ) :: tol_infeas_abs
      REAL ( c_double ) :: tol_infeas_rel
      REAL ( c_double ) :: tol_ktratio
      REAL ( c_double ) :: reduced_tol_gap_abs
      REAL ( c_double ) :: reduced_tol_gap_rel
      REAL ( c_double ) :: reduced_tol_feas
      REAL ( c_double ) :: reduced_tol_infeas_abs
      REAL ( c_double ) :: reduced_tol_infeas_rel
      REAL ( c_double ) :: reduced_tol_ktratio
      LOGICAL ( c_bool ) :: equilibrate_enable
      INTEGER ( c_int32_t ) :: equilibrate_max_iter
      REAL ( c_double ) :: equilibrate_min_scaling
      REAL ( c_double ) :: equilibrate_max_scaling
      REAL ( c_double ) :: linesearch_backtrack_step
      REAL ( c_double ) :: min_switch_step_length
      REAL ( c_double ) :: min_terminate_step_length
      INTEGER ( c_int32_t ) :: max_threads
      LOGICAL ( c_bool ) :: direct_kkt_solver
      INTEGER ( c_int ) :: direct_solve_method
      LOGICAL ( c_bool ) :: static_regularization_enable
      REAL ( c_double ) :: static_regularization_constant
      REAL ( c_double ) :: static_regularization_proportional
      LOGICAL ( c_bool ) :: dynamic_regularization_enable
      REAL ( c_double ) :: dynamic_regularization_eps
      REAL ( c_double ) :: dynamic_regularization_delta
      LOGICAL ( c_bool ) :: iterative_refinement_enable
      REAL ( c_double ) :: iterative_refinement_reltol
      REAL ( c_double ) :: iterative_refinement_abstol
      INTEGER ( c_int32_t ) :: iterative_refinement_max_iter
      REAL ( c_double ) :: iterative_refinement_stop_ratio
      LOGICAL ( c_bool ) :: presolve_enable
    END TYPE ClarabelDefaultSettings

    TYPE, BIND( c) :: ClarabelDefaultSettings_f32
      INTEGER ( c_int32_t ) :: max_iter
      REAL ( c_double ) :: time_limit
      LOGICAL ( c_bool ) :: verbose
      REAL ( c_float ) :: max_step_fraction
      REAL ( c_float ) :: tol_gap_abs
      REAL ( c_float ) :: tol_gap_rel
      REAL ( c_float ) :: tol_feas
      REAL ( c_float ) :: tol_infeas_abs
      REAL ( c_float ) :: tol_infeas_rel
      REAL ( c_float ) :: tol_ktratio
      REAL ( c_float ) :: reduced_tol_gap_abs
      REAL ( c_float ) :: reduced_tol_gap_rel
      REAL ( c_float ) :: reduced_tol_feas
      REAL ( c_float ) :: reduced_tol_infeas_abs
      REAL ( c_float ) :: reduced_tol_infeas_rel
      REAL ( c_float ) :: reduced_tol_ktratio
      LOGICAL ( c_bool ) :: equilibrate_enable
      INTEGER ( c_int32_t ) :: equilibrate_max_iter
      REAL ( c_float ) :: equilibrate_min_scaling
      REAL ( c_float ) :: equilibrate_max_scaling
      REAL ( c_float ) :: linesearch_backtrack_step
      REAL ( c_float ) :: min_switch_step_length
      REAL ( c_float ) :: min_terminate_step_length
      INTEGER ( c_int32_t ) :: max_threads
      LOGICAL ( c_bool ) :: direct_kkt_solver
      INTEGER ( c_int ) :: direct_solve_method
      LOGICAL ( c_bool ) :: static_regularization_enable
      REAL ( c_float ) :: static_regularization_constant
      REAL ( c_float ) :: static_regularization_proportional
      LOGICAL ( c_bool ) :: dynamic_regularization_enable
      REAL ( c_float ) :: dynamic_regularization_eps
      REAL ( c_float ) :: dynamic_regularization_delta
      LOGICAL ( c_bool ) :: iterative_refinement_enable
      REAL ( c_float ) :: iterative_refinement_reltol
      REAL ( c_float ) :: iterative_refinement_abstol
      INTEGER ( c_int32_t ) :: iterative_refinement_max_iter
      REAL ( c_float ) :: iterative_refinement_stop_ratio
      LOGICAL ( c_bool ) :: presolve_enable
    END TYPE ClarabelDefaultSettings_f32

    TYPE, BIND( C ) :: ClarabelDefaultSolution
      TYPE ( c_ptr ) :: x
      INTEGER ( c_intptr_t ) :: x_length
      TYPE ( c_ptr ) :: z
      INTEGER ( c_intptr_t ) :: z_length
      TYPE ( c_ptr ) :: s
      INTEGER ( c_intptr_t ) :: s_length
      INTEGER ( c_int ) :: status
      REAL ( c_double ) :: obj_val
      REAL ( c_double ) :: obj_val_dual
      REAL ( c_double ) :: solve_time
      INTEGER ( c_int32_t ) :: iterations
      REAL ( c_double ) :: r_prim
      REAL ( c_double ) :: r_dual
    END TYPE ClarabelDefaultSolution

    TYPE, BIND( C ) :: ClarabelDefaultSolution_f32
      TYPE ( c_ptr ) :: x
      INTEGER ( c_intptr_t ) :: x_length
      TYPE ( c_ptr ) :: z
      INTEGER ( c_intptr_t ) :: z_length
      TYPE ( c_ptr ) :: s
      INTEGER ( c_intptr_t ) :: s_length
      INTEGER ( c_int ) :: status
      REAL ( c_float ) :: obj_val
      REAL ( c_float ) :: obj_val_dual
      REAL ( c_double ) :: solve_time
      INTEGER ( c_int32_t ) :: iterations
      REAL ( c_float ) :: r_prim
      REAL ( c_float ) :: r_dual
    END TYPE ClarabelDefaultSolution_f32

    TYPE, BIND( C ) :: ClarabelLinearSolverInfo
      INTEGER ( c_int ) :: name
      INTEGER ( c_int32_t ) :: threads
      LOGICAL ( c_bool ) :: direct
      INTEGER ( c_int32_t ) :: nnzA
      INTEGER ( c_int32_t ) :: nnzL
      INTEGER ( c_int ) :: status
    END TYPE ClarabelLinearSolverInfo

    TYPE, BIND( C ) :: ClarabelDefaultInfo
      REAL ( c_double ) :: mu
      REAL ( c_double ) :: sigma
      REAL ( c_double ) :: step_length
      INTEGER ( c_int32_t ) :: iterations
      REAL ( c_double ) :: cost_primal
      REAL ( c_double ) :: cost_dual
      REAL ( c_double ) :: res_primal
      REAL ( c_double ) :: res_dual
      REAL ( c_double ) :: res_primal_inf
      REAL ( c_double ) :: res_dual_inf
      REAL ( c_double ) :: gap_abs
      REAL ( c_double ) :: gap_rel
      REAL ( c_double ) :: ktratio
      REAL ( c_double ) :: solve_time
      INTEGER ( c_int ) :: status
      TYPE ( ClarabelLinearSolverInfo ) :: linsolver
    END TYPE ClarabelDefaultInfo

    TYPE, BIND( C ) :: ClarabelDefaultInfo_f32
      REAL ( c_float ) :: mu
      REAL ( c_float ) :: sigma
      REAL ( c_float ) :: step_length
      INTEGER ( c_int32_t ) :: iterations
      REAL ( c_float ) :: cost_primal
      REAL ( c_float ) :: cost_dual
      REAL ( c_float ) :: res_primal
      REAL ( c_float ) :: res_dual
      REAL ( c_float ) :: res_primal_inf
      REAL ( c_float ) :: res_dual_inf
      REAL ( c_float ) :: gap_abs
      REAL ( c_float ) :: gap_rel
      REAL ( c_float ) :: ktratio
      REAL ( c_double ) :: solve_time
      INTEGER ( c_int ) :: status
      TYPE ( ClarabelLinearSolverInfo ) :: linsolver
    END TYPE ClarabelDefaultInfo_f32

    TYPE, BIND( C) :: ClarabelSupportedConeT
      INTEGER ( c_int ) :: tag
    END TYPE ClarabelSupportedConeT

    TYPE, BIND( C ) :: ClarabelSupportedConeT_f32
      INTEGER ( c_int ) :: tag
    END TYPE ClarabelSupportedConeT_f32

  END MODULE CLARABEL_TYPES_precision
