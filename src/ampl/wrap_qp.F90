! THIS VERSION: GALAHAD 2.4 - 08/02/2011 AT 08:30 GMT.
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
!!     Generic interface for a variety of Methods for Quadratic Programs,
!!
!! D. Orban@ECE                                   Chicago, February 2011
!!
!! Thanks to the f2py project      http://cens.ioc.ee/projects/f2py2e
!!===================================================================

!! ===============================
!! Main wrapper around USE_QP( )
!! ===============================

Subroutine Wrap_Use_QP( setup_use_QP )

  Use GALAHAD_USEQP_double      ! Main QP driver module
  External setup_use_QP         ! Declared in galahad.c

  ! Pass pointer to subroutine USE_QP to calling function
  Call setup_use_QP( USE_QP )

  Return

End Subroutine Wrap_Use_QP

!!=================================
!! End of wrapper
!! ================================
