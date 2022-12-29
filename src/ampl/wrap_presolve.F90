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
!!   PRESOLVE
!!     Quadratic Program Preprocessor,
!!
!! D. Orban@ECE                                   Chicago, March 2003
!!
!! Thanks to the f2py project      http://cens.ioc.ee/projects/f2py2e
!!===================================================================

!! ===================================
!! Main wrapper around USE_PRESOLVE( )
!! ===================================

Subroutine Wrap_Use_Presolve( setup_use_presolve )

  Use GALAHAD_USEPRESOLVE_precision ! Main PRESOLVE driver module
  External setup_use_presolve    ! Declared in galahad.c

  ! Pass pointer to subroutine USE_PRESOLVE to calling function
  Call setup_use_presolve( USE_PRESOLVE )

  Return

End Subroutine Wrap_Use_Presolve

!!=================================
!! End of wrapper
!! ================================
