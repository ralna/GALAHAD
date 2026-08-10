! THIS VERSION: GALAHAD 5.6 - 2026-07-20 AT 15:10 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*- G A L A H A D  -  Q P O A S E S    M O D U L E -*-*-*-*-*-*-*-

  MODULE GALAHAD_QPOASES_precision

    USE GALAHAD_KINDS_precision, ONLY: ipc_, rpc_, c_ptr, c_loc
    USE QPOASES_TYPES_precision, ONLY: qpOASES_options_type

    IMPLICIT NONE

    PRIVATE
    PUBLIC :: qpOASES_solve

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

    status =  - 199  ! error code

    END SUBROUTINE qpOASES_solve

  END MODULE GALAHAD_QPOASES_precision
