! THIS VERSION: GALAHAD 5.6 - 2026-07-22 AT 13:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*- G A L A H A D  -  Q P O A S E S    M O D U L E -*-*-*-*-*-*-*-

  MODULE GALAHAD_QPOASES_precision

    USE GALAHAD_KINDS_precision, ONLY: ipc_, rpc_, c_ptr, c_loc
    USE QPOASES_INTERFACE_precision, ONLY: qpOASES_options_type

    IMPLICIT NONE

    PRIVATE
    PUBLIC :: qpOASES_options_type, qpOASES_solve

!----------------------
!   I n t e r f a c e s
!----------------------

!  interface blocks for C functions

    INTERFACE
      FUNCTION c_sparse_solve_all( nV, nC, H_row, H_ptr, H_val, g,             &
                                   A_row, A_ptr, A_val, lb, ub, lbA, ubA,      &
                                   userOptions, nWSR, cputime, x, y, objVal )  &
#ifdef INTEGER_64
#ifdef REAL_32
             BIND( C, NAME = "qpOASES_c_solve_sgl_64" )
#elif REAL_128
             BIND( C, NAME = "qpOASES_c_solve_qul_64" )
#else
             BIND( C, NAME = "qpOASES_c_solve_dbl_64" )
#endif
#else
#ifdef REAL_32
             BIND( C, NAME = "qpOASES_c_solve_sgl" )
#elif REAL_128
             BIND( C, NAME = "qpOASES_c_solve_qul" )
#else
             BIND( C, NAME = "qpOASES_c_solve_dbl" )
#endif
#endif

          IMPORT :: c_ptr, ipc_, rpc_
          INTEGER ( ipc_ ) :: c_sparse_solve_all
          INTEGER ( ipc_ ), VALUE :: nV, nC
          TYPE ( c_ptr ), VALUE :: H_row, H_ptr, H_val, g
          TYPE ( c_ptr ), VALUE :: A_row, A_ptr, A_val
          TYPE ( c_ptr ), VALUE :: lb, ub, lbA, ubA
          TYPE ( c_ptr ), VALUE :: userOptions
          INTEGER ( ipc_ ), INTENT( INOUT ) :: nWSR
          REAL ( rpc_ ), INTENT( INOUT ) :: cputime
          TYPE ( c_ptr ), VALUE :: x, y, objVal
      END FUNCTION c_sparse_solve_all
    END INTERFACE

  CONTAINS

!  fortran subroutines corresponding to C functions

    SUBROUTINE qpOASES_solve( n, m, H_row, H_ptr, H_val, g, A_row, A_ptr,      &
                              A_val, x_l, x_u, c_l, c_u, options,              &
                              iter, cputime, x_sol, y_sol, f_sol, status )

!  input problem with 1-based indices

    INTEGER ( ipc_ ), INTENT( IN ), VALUE :: n, m
    INTEGER ( ipc_ ), INTENT( IN ) :: H_ptr( n + 1 )
    INTEGER ( ipc_ ), INTENT( IN ) :: H_row( H_ptr( n + 1 ) - 1 )
    REAL ( rpc_ ), INTENT( IN ) :: H_val( H_ptr( n + 1 ) - 1 )
    REAL ( rpc_ ), INTENT( IN ) :: G( n )
    INTEGER ( ipc_ ), INTENT( IN ) :: A_ptr( n + 1 )
    INTEGER ( ipc_ ), INTENT( IN ) :: A_row( A_ptr( n + 1 ) - 1 )
    REAL ( rpc_ ), INTENT( IN ) :: A_val( A_ptr( n + 1 ) - 1 )
    REAL ( rpc_ ), INTENT( IN ) :: X_l( n ), X_u( n )
    REAL ( rpc_ ), INTENT( IN ) :: C_l( m ), C_u( m )
    INTEGER ( ipc_ ), INTENT( INOUT ) :: iter
    REAL ( rpc_ ), INTENT( INOUT ) :: cpuTime
    TYPE ( qpOASES_options_type ) :: options
    REAL ( rpc_ ), INTENT( OUT ) :: X_sol( n )
    REAL ( rpc_ ), INTENT( OUT ) :: Y_sol( m )
    REAL ( rpc_ ), INTENT( OUT ) :: f_sol
    INTEGER ( ipc_ ), INTENT( OUT ) :: status

!  set call to use 0-based indices

    CALL qpOASES_solve_c( n, m, H_row - 1, H_ptr - 1, H_val, g, A_row - 1,     &
                          A_ptr - 1, A_val, x_l, x_u, c_l, c_u, options,       &
                          iter, cputime, x_sol, y_sol, f_sol, status )

    END SUBROUTINE qpOASES_solve

    SUBROUTINE qpOASES_solve_c( n, m, H_row, H_ptr, H_val, g, A_row, A_ptr,    &
                              A_val, x_l, x_u, c_l, c_u, options,              &
                              iter, cputime, x_sol, y_sol, f_sol, status )

!  input problem with 0-based indices

    INTEGER ( ipc_ ), INTENT( IN ), VALUE :: n, m
    INTEGER ( ipc_ ), TARGET, INTENT( IN ) :: H_ptr( n + 1 )
    INTEGER ( ipc_ ), TARGET, INTENT( IN ) :: H_row( H_ptr( n + 1 ) )
    REAL ( rpc_ ), TARGET, INTENT( IN ) :: H_val( H_ptr( n + 1 ) )
    REAL ( rpc_ ), TARGET, INTENT( IN ) :: G( n )
    INTEGER ( ipc_ ), TARGET, INTENT( IN ) :: A_ptr( n + 1 )
    INTEGER ( ipc_ ), TARGET, INTENT( IN ) :: A_row( A_ptr( n + 1 ) )
    REAL ( rpc_ ), TARGET, INTENT( IN ) :: A_val( A_ptr( n + 1 ) )
    REAL ( rpc_ ), TARGET, INTENT( IN ) :: X_l( n ), X_u( n )
    REAL ( rpc_ ), TARGET, INTENT( IN ) :: C_l( m ), C_u( m )
    INTEGER ( ipc_ ), INTENT( INOUT ) :: iter
    REAL ( rpc_ ), INTENT( INOUT ) :: cpuTime
    TYPE ( qpOASES_options_type ), TARGET :: options
    REAL ( rpc_ ), TARGET, INTENT( OUT ) :: X_sol( n )
    REAL ( rpc_ ), TARGET, INTENT( OUT ) :: Y_sol( m )
    REAL ( rpc_ ), TARGET, INTENT( OUT ) :: f_sol
    INTEGER ( ipc_ ), INTENT( OUT ) :: status

!  call the C interface to qpOASES

    status = c_sparse_solve_all( n, m, C_LOC( H_row ), C_LOC( H_ptr ),         &
                                 C_LOC( H_val ), C_LOC( g ), C_LOC( A_row ),   &
                                 C_LOC( A_ptr ), C_LOC( A_val ), C_LOC( x_l ), &
                                 C_LOC( x_u ), C_LOC( c_l ), C_LOC( c_u ),     &
                                 C_LOC( options ), iter, cputime,              &
                                 C_LOC( x_sol ), C_LOC( y_sol ),               &
                                 C_LOC( f_sol ) )
    END SUBROUTINE qpOASES_solve_c

  END MODULE GALAHAD_QPOASES_precision
