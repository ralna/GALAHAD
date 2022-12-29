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
!!    FILTRANE
!!     Smooth Feasibility Problem solver
!!
!! D. Orban@ECE                                    Chicago, June 2003
!!
!! Thanks to the f2py project      http://cens.ioc.ee/projects/f2py2e
!!===================================================================

!! ===================================
!! Main wrapper around USE_FILTRANE( )
!! ===================================

Subroutine Wrap_Use_Filtrane( setup_use_filtrane )

  Use GALAHAD_USEFILTRANE_precision   ! Main FILTRANE driver module
  External setup_use_filtrane      ! Declared in galahad.c

  ! Pass pointer to subroutine USE_FILTRANE to calling function
  Call setup_use_filtrane( USE_FILTRANE )

  Return

End Subroutine Wrap_Use_Filtrane

!!=================================
!! End of wrapper
!! ================================
