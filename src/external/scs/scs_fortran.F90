! THIS VERSION: GALAHAD 5.6 - 2026-08-10 AT 13:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D  -  S C S    M O D U L E -*-*-*-*-*-*-*-*-

  MODULE GALAHAD_SCS_precision

    USE GALAHAD_KINDS_precision, ONLY: ipc_, rpc_
    USE, INTRINSIC :: iso_c_binding, ONLY: c_ptr, c_int, c_char, c_loc
    USE SCS_TYPES_precision, ONLY: ScsData, ScsCone, ScsSettings,              &
                                   ScsSolution, ScsInfo

    IMPLICIT NONE ( TYPE, EXTERNAL )

!----------------------
!   I n t e r f a c e s
!----------------------

!  interface blocks for C functions

    INTERFACE

!---------------------------------------------
! scs_init
!---------------------------------------------
!  Initialize SCS and allocate memory.
!
!  All the inputs must be already allocated in memory before calling. After
!  this function returns then the memory associated with `d`, `k`, and `stgs`
!  can be freed as SCS maintains deep copies of these internally.
!
!  It performs:
!  - data and settings validation
!  - problem data scaling
!  - automatic parameters tuning (if enabled)
!  - setup linear system solver:
!       - direct solver: KKT matrix factorization is performed here
!       - indirect solver: KKT matrix preconditioning is performed here.
!
!
!  @param d - Problem data.
!  @param k - Cone data.
!  @param stgs - SCS solve settings.
!  @return        Solver workspace.

      FUNCTION scs_init(d, k, stgs) &
         RESULT( init ) &
         BIND( C, name = "scs_init" )
         IMPORT :: ScsData, ScsCone, ScsSettings, c_ptr
         TYPE ( ScsData ) :: d
         TYPE ( ScsCone ) :: k
         TYPE ( ScsSettings ) :: stgs
         TYPE ( c_ptr ) :: init
      END FUNCTION scs_init

!---------------------------------------------
! scs_update
!---------------------------------------------
!  Update the `b` vector, `c` vector, or both, before another solve call.
!
!  After a solve we can reuse the SCS workspace in another solve if the only
!  problem data that has changed are the `b` and `c` vectors.
!
!  @param w - SCS workspace from scs_init (modified in-place).
!  @param b - New `b` vector (can be `SCS_NULL` if unchanged).
!  @param c - New `c` vector (can be `SCS_NULL` if unchanged).
!
!  @return              0 if update successful.

      FUNCTION scs_update(w, b, c) &
         RESULT( update ) &
         BIND( C, name = "scs_update" )
         IMPORT :: c_ptr, c_int
         TYPE ( c_ptr ), value :: w
         TYPE ( c_ptr ), value :: b
         TYPE ( c_ptr ), value :: c
         INTEGER ( c_int ) :: update
      END FUNCTION scs_update

!---------------------------------------------
! scs_solve
!---------------------------------------------
!  Solve quadratic cone program initialized by scs_init.
!
!  @param w - Workspace allocated by scs_init.
!  @param sol - Solution will be stored here. If members `x`, `y`, `s`
!                       are NULL then SCS will allocate memory for them which
!                       must be freed by the caller.
!  @param myInfo - Information about the solve will be stored here.
!  @param warm_start - Whether to use the entries of `sol` as warm-start for
!                       the solve.
!
!  @return       Flag containing problem status (see \a glbopts.h).

      FUNCTION scs_solve(w, sol, myInfo, warm_start) &
         RESULT( solve ) &
         BIND( C, name = "scs_solve" )
         IMPORT :: c_ptr, ScsSolution, ScsInfo, c_int
         TYPE ( c_ptr ), value :: w
         TYPE ( ScsSolution ) :: sol
         TYPE ( ScsInfo ) :: myInfo
         INTEGER ( c_int ), value :: warm_start
         INTEGER ( c_int ) :: solve
      END FUNCTION scs_solve

!---------------------------------------------
! scs_finish
!---------------------------------------------
!  Clean up allocated SCS workspace.
!
!  @param w - Workspace allocated by init, will be deallocated.

      SUBROUTINE scs_finish(w) &
         BIND( C, name = "scs_finish" )
         IMPORT :: c_ptr
         TYPE ( c_ptr ), value :: w
      END SUBROUTINE scs_finish

!---------------------------------------------
! scs
!---------------------------------------------
!  Solve quadratic cone program defined by data in d and cone k.
!
!  All the inputs must already be allocated in memory before calling.
!
!  @param d - Problem data.
!  @param k - Cone data.
!  @param stgs - SCS solver settings.
!  @param sol - Solution will be stored here. If members `x`, `y`, `s` are
!                 NULL then SCS will allocate memory for them.
!  @param myInfo - Information about the solve will be stored here.
!  @return        Flag containing problem status (see \a glbopts.h).

      FUNCTION scs(d, k, stgs, sol, myInfo) &
         RESULT( scs_ret ) &
         BIND( C, name = "scs" )
         IMPORT :: ScsData, ScsCone, ScsSettings, ScsSolution, ScsInfo, c_int
         TYPE ( ScsData ) :: d
         TYPE ( ScsCone ) :: k
         TYPE ( ScsSettings ) :: stgs
         TYPE ( ScsSolution ) :: sol
         TYPE ( ScsInfo ) :: myInfo
         INTEGER ( c_int ) :: scs_ret
      END FUNCTION scs

!---------------------------------------------
! scs_set_default_settings
!---------------------------------------------
!  Helper function to set all settings to default values (see \a glbopts.h).
!
!  @param stgs - Settings struct that will be populated.

      SUBROUTINE scs_set_default_settings(stgs) &
         BIND( C, name = "scs_set_default_settings" )
         IMPORT :: ScsSettings
         TYPE ( ScsSettings ) :: stgs
      END SUBROUTINE scs_set_default_settings

!---------------------------------------------
! scs_version
!---------------------------------------------
!  Helper function simply returns the current version of SCS as a string.
!
!  @return       SCS version as a string.

      FUNCTION scs_version() &
         RESULT( version ) &
         BIND( C, name = "scs_version" )
         IMPORT :: c_char
         CHARACTER ( c_char ) :: version
      END FUNCTION scs_version

  END INTERFACE

END MODULE GALAHAD_SCS_precision
