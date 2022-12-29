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
!!   CQP
!!     interior-point Method for Convex Quadratic Programs,
!!
!! D. Orban@ECE                                   Chicago, December 2009
!!
!! Thanks to the f2py project      http://cens.ioc.ee/projects/f2py2e
!!===================================================================

!! ===============================
!! Main wrapper around USE_CQP( )
!! ===============================

Subroutine Wrap_Use_Cqp( setup_use_cqp )

  Use GALAHAD_USECQP_precision      ! Main CQP driver module
  External setup_use_cqp         ! Declared in galahad.c

  ! Pass pointer to subroutine USE_Cqp to calling function
  Call setup_use_cqp( USE_CQP )

  Return

End Subroutine Wrap_Use_Cqp

!!=================================
!! End of wrapper
!! ================================
