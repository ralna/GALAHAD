! THIS VERSION: GALAHAD 5.6 - 2026-08-10 AT 13:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*- G A L A H A D  -  C L A R A B E L    M O D U L E -*-*-*-*-*-*-

  MODULE GALAHAD_CLARABEL_precision

    USE GALAHAD_KINDS_precision, ONLY: ipc_, rpc_, c_loc
    USE, INTRINSIC :: iso_c_binding, ONLY: c_ptr, c_intptr_t, c_funptr
    USE CLARABEL_TYPES_precision, ONLY: ClarabelCscMatrix_f32,                 &
                                        ClarabelDefaultSettings_f32,           &
                                        ClarabelSupportedConeT_f32,            &
                                        ClarabelDefaultSolution_f32,           &
                                        ClarabelDefaultInfo_f32

    IMPLICIT NONE ( TYPE, EXTERNAL )

!----------------------
!   I n t e r f a c e s
!----------------------

!  interface blocks for C functions

    INTERFACE

!---------------------------------------------
! clarabel_CscMatrix_f32_init
!---------------------------------------------

      SUBROUTINE clarabel_CscMatrix_f32_init(ptr, m, n, colptr, rowval, nzval) &
         BIND( C, name = "clarabel_CscMatrix_f32_init" )
         IMPORT :: ClarabelCscMatrix_f32, c_intptr_t, c_ptr
         TYPE ( ClarabelCscMatrix_f32 ) :: ptr
         INTEGER ( c_intptr_t ), value :: m
         INTEGER ( c_intptr_t ), value :: n
         TYPE ( c_ptr ), value :: colptr
         TYPE ( c_ptr ), value :: rowval
         TYPE ( c_ptr ), value :: nzval
      END SUBROUTINE clarabel_CscMatrix_f32_init

!---------------------------------------------
! clarabel_CscMatrix_f64_init
!---------------------------------------------

      SUBROUTINE clarabel_CscMatrix_f64_init(ptr, m, n, colptr, rowval, nzval) &
         BIND( C, name = "clarabel_CscMatrix_f64_init" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: ptr
         INTEGER ( c_intptr_t ), value :: m
         INTEGER ( c_intptr_t ), value :: n
         TYPE ( c_ptr ), value :: colptr
         TYPE ( c_ptr ), value :: rowval
         TYPE ( c_ptr ), value :: nzval
      END SUBROUTINE clarabel_CscMatrix_f64_init

!---------------------------------------------
! clarabel_DefaultSettings_f64_default
!---------------------------------------------

      FUNCTION clarabel_DefaultSettings_f64_default() &
         RESULT( DefaultSettings_f64_default ) &
         BIND( C, name = "clarabel_DefaultSettings_f64_default" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ) :: DefaultSettings_f64_default
      END FUNCTION clarabel_DefaultSettings_f64_default

!---------------------------------------------
! clarabel_DefaultSettings_f32_default
!---------------------------------------------

      FUNCTION clarabel_DefaultSettings_f32_default() &
         RESULT( DefaultSettings_f32_default ) &
         BIND( C, name = "clarabel_DefaultSettings_f32_default" )
         IMPORT :: ClarabelDefaultSettings_f32
         TYPE ( ClarabelDefaultSettings_f32 ) :: DefaultSettings_f32_default
      END FUNCTION clarabel_DefaultSettings_f32_default

!---------------------------------------------
! clarabel_DefaultSolver_f64_new
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f64_new(P, q, A, b, n_cones, cones, settings) &
         RESULT( DefaultSolver_f64_new ) &
         BIND( C, name = "clarabel_DefaultSolver_f64_new" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: P
         TYPE ( c_ptr ), value :: q
         TYPE ( c_ptr ), value :: A
         TYPE ( c_ptr ), value :: b
         INTEGER ( c_intptr_t ), value :: n_cones
         TYPE ( c_ptr ), value :: cones
         TYPE ( c_ptr ), value :: settings
         TYPE ( c_ptr ) :: DefaultSolver_f64_new
      END FUNCTION clarabel_DefaultSolver_f64_new

!---------------------------------------------
! clarabel_DefaultSolver_f32_new
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f32_new(P, q, A, b, n_cones, cones, settings) &
         RESULT( DefaultSolver_f32_new ) &
         BIND( C, name = "clarabel_DefaultSolver_f32_new" )
         IMPORT :: ClarabelCscMatrix_f32, c_ptr, c_intptr_t, ClarabelSupportedConeT_f32, &
                   ClarabelDefaultSettings_f32
         TYPE ( ClarabelCscMatrix_f32 ) :: P
         TYPE ( c_ptr ), value :: q
         TYPE ( ClarabelCscMatrix_f32 ) :: A
         TYPE ( c_ptr ), value :: b
         INTEGER ( c_intptr_t ), value :: n_cones
         TYPE ( ClarabelSupportedConeT_f32 ) :: cones
         TYPE ( ClarabelDefaultSettings_f32 ) :: settings
         TYPE ( c_ptr ) :: DefaultSolver_f32_new
      END FUNCTION clarabel_DefaultSolver_f32_new

!---------------------------------------------
! clarabel_DefaultSolver_f64_print_to_stdout
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_print_to_stdout(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f64_print_to_stdout" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f64_print_to_stdout

!---------------------------------------------
! clarabel_DefaultSolver_f32_print_to_stdout
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_print_to_stdout(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f32_print_to_stdout" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f32_print_to_stdout

!---------------------------------------------
! clarabel_DefaultSolver_f64_print_to_file
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_print_to_file(solver, filename) &
         BIND( C, name = "clarabel_DefaultSolver_f64_print_to_file" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: filename
      END SUBROUTINE clarabel_DefaultSolver_f64_print_to_file

!---------------------------------------------
! clarabel_DefaultSolver_f32_print_to_file
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_print_to_file(solver, filename) &
         BIND( C, name = "clarabel_DefaultSolver_f32_print_to_file" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: filename
      END SUBROUTINE clarabel_DefaultSolver_f32_print_to_file

!---------------------------------------------
! clarabel_DefaultSolver_f64_print_to_buffer
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_print_to_buffer(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f64_print_to_buffer" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f64_print_to_buffer

!---------------------------------------------
! clarabel_DefaultSolver_f32_print_to_buffer
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_print_to_buffer(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f32_print_to_buffer" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f32_print_to_buffer

!---------------------------------------------
! clarabel_DefaultSolver_f64_get_print_buffer
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f64_get_print_buffer(solver) &
         RESULT( DefaultSolver_f64_get_print_buffer ) &
         BIND( C, name = "clarabel_DefaultSolver_f64_get_print_buffer" )
         IMPORT :: c_ptr, c_char
         TYPE ( c_ptr ), value :: solver
         CHARACTER ( c_char ) :: DefaultSolver_f64_get_print_buffer
      END FUNCTION clarabel_DefaultSolver_f64_get_print_buffer

!---------------------------------------------
! clarabel_DefaultSolver_f32_get_print_buffer
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f32_get_print_buffer(solver) &
         RESULT( DefaultSolver_f32_get_print_buffer ) &
         BIND( C, name = "clarabel_DefaultSolver_f32_get_print_buffer" )
         IMPORT :: c_ptr, c_char
         TYPE ( c_ptr ), value :: solver
         CHARACTER ( c_char ) :: DefaultSolver_f32_get_print_buffer
      END FUNCTION clarabel_DefaultSolver_f32_get_print_buffer

!---------------------------------------------
! clarabel_free_print_buffer
!---------------------------------------------

      SUBROUTINE clarabel_free_print_buffer(buffer) &
         BIND( C, name = "clarabel_free_print_buffer" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: buffer
      END SUBROUTINE clarabel_free_print_buffer

!---------------------------------------------
! clarabel_DefaultSolver_f64_solve
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_solve(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f64_solve" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f64_solve

!---------------------------------------------
! clarabel_DefaultSolver_f32_solve
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_solve(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f32_solve" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f32_solve

!---------------------------------------------
! clarabel_DefaultSolver_f64_free
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_free(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f64_free" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f64_free

!---------------------------------------------
! clarabel_DefaultSolver_f32_free
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_free(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f32_free" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f32_free

!---------------------------------------------
! clarabel_DefaultSolver_f64_solution
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f64_solution(solver) &
         RESULT( DefaultSolver_f64_solution ) &
         BIND( C, name = "clarabel_DefaultSolver_f64_solution" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ) :: DefaultSolver_f64_solution
      END FUNCTION clarabel_DefaultSolver_f64_solution

!---------------------------------------------
! clarabel_DefaultSolver_f32_solution
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f32_solution(solver) &
         RESULT( DefaultSolver_f32_solution ) &
         BIND( C, name = "clarabel_DefaultSolver_f32_solution" )
         IMPORT :: c_ptr, ClarabelDefaultSolution_f32
         TYPE ( c_ptr ), value :: solver
         TYPE ( ClarabelDefaultSolution_f32 ) :: DefaultSolver_f32_solution
      END FUNCTION clarabel_DefaultSolver_f32_solution

!---------------------------------------------
! clarabel_DefaultSolver_f64_info
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f64_info(solver) &
         RESULT( DefaultSolver_f64_info ) &
         BIND( C, name = "clarabel_DefaultSolver_f64_info" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ) :: DefaultSolver_f64_info
      END FUNCTION clarabel_DefaultSolver_f64_info

!---------------------------------------------
! clarabel_DefaultSolver_f32_info
!---------------------------------------------

      FUNCTION clarabel_DefaultSolver_f32_info(solver) &
         RESULT( DefaultSolver_f32_info ) &
         BIND( C, name = "clarabel_DefaultSolver_f32_info" )
         IMPORT :: c_ptr, ClarabelDefaultInfo_f32
         TYPE ( c_ptr ), value :: solver
         TYPE ( ClarabelDefaultInfo_f32 ) :: DefaultSolver_f32_info
      END FUNCTION clarabel_DefaultSolver_f32_info

!---------------------------------------------
! clarabel_DefaultSolver_f64_set_termination_callback
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_set_termination_callback(solver, callback, userdata) &
         BIND( C, name = "clarabel_DefaultSolver_f64_set_termination_callback" )
         IMPORT :: c_ptr, c_funptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_funptr ), value :: callback
         TYPE ( c_ptr ), value :: userdata
      END SUBROUTINE clarabel_DefaultSolver_f64_set_termination_callback

!---------------------------------------------
! clarabel_DefaultSolver_f32_set_termination_callback
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_set_termination_callback(solver, callback, userdata) &
         BIND( C, name = "clarabel_DefaultSolver_f32_set_termination_callback" )
         IMPORT :: c_ptr, c_funptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_funptr ), value :: callback
         TYPE ( c_ptr ), value :: userdata
      END SUBROUTINE clarabel_DefaultSolver_f32_set_termination_callback

!---------------------------------------------
! clarabel_DefaultSolver_f64_unset_termination_callback
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_unset_termination_callback(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f64_unset_termination_callback" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f64_unset_termination_callback

!---------------------------------------------
! clarabel_DefaultSolver_f32_unset_termination_callback
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_unset_termination_callback(solver) &
         BIND( C, name = "clarabel_DefaultSolver_f32_unset_termination_callback" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
      END SUBROUTINE clarabel_DefaultSolver_f32_unset_termination_callback

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_P
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_P(solver, Pnzval, nnzP) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_P" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: Pnzval
         INTEGER ( c_intptr_t ), value :: nnzP
      END SUBROUTINE clarabel_DefaultSolver_f64_update_P

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_P
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_P(solver, Pnzval, nnzP) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_P" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: Pnzval
         INTEGER ( c_intptr_t ), value :: nnzP
      END SUBROUTINE clarabel_DefaultSolver_f32_update_P

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_P_partial
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_P_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_P_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f64_update_P_partial

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_P_partial
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_P_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_P_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f32_update_P_partial

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_P_csc
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_P_csc(solver, P) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_P_csc" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: P
      END SUBROUTINE clarabel_DefaultSolver_f64_update_P_csc

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_P_csc
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_P_csc(solver, P) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_P_csc" )
         IMPORT :: c_ptr, ClarabelCscMatrix_f32
         TYPE ( c_ptr ), value :: solver
         TYPE ( ClarabelCscMatrix_f32 ) :: P
      END SUBROUTINE clarabel_DefaultSolver_f32_update_P_csc

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_A
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_A(solver, Anzval, nnzA) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_A" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: Anzval
         INTEGER ( c_intptr_t ), value :: nnzA
      END SUBROUTINE clarabel_DefaultSolver_f64_update_A

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_A
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_A(solver, Anzval, nnzA) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_A" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: Anzval
         INTEGER ( c_intptr_t ), value :: nnzA
      END SUBROUTINE clarabel_DefaultSolver_f32_update_A

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_A_partial
!---------------------------------------------
      SUBROUTINE clarabel_DefaultSolver_f64_update_A_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_A_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f64_update_A_partial

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_A_partial
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_A_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_A_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f32_update_A_partial

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_A_csc
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_A_csc(solver, A) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_A_csc" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: A
      END SUBROUTINE clarabel_DefaultSolver_f64_update_A_csc

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_A_csc
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_A_csc(solver, A) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_A_csc" )
         IMPORT :: c_ptr, ClarabelCscMatrix_f32
         TYPE ( c_ptr ), value :: solver
         TYPE ( ClarabelCscMatrix_f32 ) :: A
      END SUBROUTINE clarabel_DefaultSolver_f32_update_A_csc

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_q
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_q(solver, values, n) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_q" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: n
      END SUBROUTINE clarabel_DefaultSolver_f64_update_q

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_q
!---------------------------------------------
      SUBROUTINE clarabel_DefaultSolver_f32_update_q(solver, values, n) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_q" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: n
      END SUBROUTINE clarabel_DefaultSolver_f32_update_q

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_q_partial
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_q_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_q_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f64_update_q_partial

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_q_partial
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_q_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_q_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f32_update_q_partial

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_b
!---------------------------------------------
!> ////// b data updating
      SUBROUTINE clarabel_DefaultSolver_f64_update_b(solver, values, n) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_b" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: n
      END SUBROUTINE clarabel_DefaultSolver_f64_update_b

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_b
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_b(solver, values, n) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_b" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: n
      END SUBROUTINE clarabel_DefaultSolver_f32_update_b

!---------------------------------------------
! clarabel_DefaultSolver_f64_update_b_partial
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f64_update_b_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f64_update_b_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f64_update_b_partial

!---------------------------------------------
! clarabel_DefaultSolver_f32_update_b_partial
!---------------------------------------------

      SUBROUTINE clarabel_DefaultSolver_f32_update_b_partial(solver, index, values, nvals) &
         BIND( C, name = "clarabel_DefaultSolver_f32_update_b_partial" )
         IMPORT :: c_ptr, c_intptr_t
         TYPE ( c_ptr ), value :: solver
         TYPE ( c_ptr ), value :: index
         TYPE ( c_ptr ), value :: values
         INTEGER ( c_intptr_t ), value :: nvals
      END SUBROUTINE clarabel_DefaultSolver_f32_update_b_partial

  END INTERFACE

 END MODULE GALAHAD_CLARABEL_precision
