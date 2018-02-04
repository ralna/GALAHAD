! THIS VERSION: GALAHAD 2.4 - 07/12/2009 AT 10:30 GMT.
!! Implements an F90 wrapper around the solvers implemented in the
!!
!!            G  A  L  A  H  A  D
!!
!! optimization library, to make functions within Fortran 90 modules
!! visible to the linker when linking with a C object file.
!!
!! Grants access to
!!
!!   BQP
!!     Preconditiond Projected Conjugate-Gradient Method
!!     for Convex Bound-Constrained Quadratic Programs,
!!
!! D. Orban@ECE                                   Chicago, December 2009
!!
!! Thanks to the f2py project      http://cens.ioc.ee/projects/f2py2e
!!===================================================================

!! ===============================
!! Main wrapper around USE_BQP( )
!! ===============================

Subroutine Wrap_Use_Bqp( setup_use_bqp )

  Use GALAHAD_USEBQP_double      ! Main BQP driver module
  External setup_use_bqp         ! Declared in galahad.c

  ! Pass pointer to subroutine USE_Bqp to calling function
  Call setup_use_bqp( USE_BQP )

  Return

End Subroutine Wrap_Use_Bqp

!!=================================
!! End of wrapper
!! ================================
