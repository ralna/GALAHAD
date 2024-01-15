! THIS VERSION: GALAHAD 4.1 - 2022-12-28 AT 11:00 GMT.
#include "galahad_modules.h"
!! Implements an F90 wrapper around the solvers implemented in the
!!
!!            G  A  L  A  H  A  D
!!
!! optimization library, to make functions within Fortran 90 modules
!! visible to the linker when linking with a C object file.
!!
!! Grants access to
!!
!!   QPC
!!     Crossover Primal-Dual Interior Point/Working-Set
!!     Method for Quadratic Programs,
!!
!! D. Orban@ECE                                   Chicago, March 2003
!!
!! Thanks to the f2py project      http://cens.ioc.ee/projects/f2py2e
!!===================================================================

!! ===============================
!! Main wrapper around USE_QPC( )
!! ===============================

Subroutine Wrap_Use_Qpc( setup_use_qpc )

  Use GALAHAD_USEQPC_precision      ! Main QPC driver module
  External setup_use_qpc         ! Declared in galahad.c

  ! Pass pointer to subroutine USE_Qpc to calling function
  Call setup_use_qpc( USE_QPC )

  Return

End Subroutine Wrap_Use_Qpc

!!=================================
!! End of wrapper
!! ================================
