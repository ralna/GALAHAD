! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 15:50 GMT

#include "ssids_procedures.h"

!  copyright 2016 The Science and Technology Facilities Council (STFC)
!  licence: BSD licence, see LICENCE file for details
!  author: Jonathan Hogg
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

MODULE GALAHAD_SSIDS_subtree_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SSIDS_contrib_precision, ONLY: contrib_type
   USE GALAHAD_SSIDS_types_precision, ONLY: SSIDS_control_type,                &
                                            SSIDS_inform_type
   IMPLICIT NONE

   PRIVATE
   PUBLIC :: symbolic_subtree_base, numeric_subtree_base

!  abstract base class for Symbolic subtrees.
! 
!  The symbolic subtrees encode the information from the analyse phase
!  necessary to generate a numeric factorization in the subsequent
!  factorization phases, which call the factor() entry.
! 
!  A subtree may have child subtrees that hang off it. At factorization
!  time, multifrontal contribution blocks from all children will be supplied.
!  Each subtree must in turn generate a contribution block if it is not a
!  root subtree.
! 
!  see also numeric_subtree_base

   TYPE, ABSTRACT :: symbolic_subtree_base

   CONTAINS

!  perform numeric factorization, returning a subclass of numeric_subtree_base
!  representing this.

      PROCEDURE( factor_iface ), DEFERRED :: factor

!  free associated memory/resources

      PROCEDURE( symbolic_cleanup_iface ), DEFERRED :: cleanup
   END TYPE symbolic_subtree_base

   TYPE, ABSTRACT :: numeric_subtree_base

!  abstract base class for Numeric subtrees. The numeric subtree represents 
!  the numeric factorization of a subtree and is returned from the 
!  corresponding factor() call of a Symbolic subtree
!  
!  see also symbolic_subtree_base

   CONTAINS

!  return contribution block from this subtree to parent. Behaviour is 
!  undefined if called on a root subtree. Routine will spinlock with 
!  taskyield if factorization is still ongoing.

      PROCEDURE( get_contrib_iface ), DEFERRED :: get_contrib

!  perform forward solve.

      PROCEDURE( solve_proc_iface ), DEFERRED :: solve_fwd

!  perform diagonal solve.

      PROCEDURE( solve_proc_iface ), DEFERRED :: solve_diag

!  perform combined diagonal and backward solve.

      PROCEDURE( solve_proc_iface ), DEFERRED :: solve_diag_bwd

!  perform backward solve.

      PROCEDURE( solve_proc_iface ), DEFERRED :: solve_bwd

!  free associated memory/resources

      PROCEDURE( numeric_cleanup_iface ), DEFERRED :: cleanup
   END TYPE numeric_subtree_base

   ABSTRACT interface

      FUNCTION factor_iface( this, posdef, aval, child_contrib, control,       &
                             inform, scaling)

!  perform numeric factorization, returning a subclass of numeric_subtree_base 
!  representing this. Arguments:
!   this Instance pointer.
!   posdef Perform Cholesky-like unpivoted factorization if true.
!   aval Value component of CSC datatype for original matrix A.
!   child_contrib Array of contribution blocks from children.
!   control User-supplied options.
!   inform Information/statistics to be returned to user.
!   scaling Scaling to be applied (if present).

        IMPORT symbolic_subtree_base, numeric_subtree_base, rp_
        IMPORT SSIDS_inform_type, SSIDS_control_type
        IMPORT contrib_type
        IMPLICIT none
        CLASS(numeric_subtree_base ), POINTER :: factor_iface
        CLASS(symbolic_subtree_base ), TARGET, INTENT( INOUT ) :: this
        LOGICAL, INTENT( IN ) :: posdef
        REAL( rp_ ), dimension(*), target, intent(in) :: aval
        TYPE(contrib_type ), DIMENSION( : ), TARGET,                           &
           INTENT( INOUT ) :: child_contrib
        TYPE( SSIDS_control_type ), INTENT( IN ) :: control
        TYPE( SSIDS_inform_type) , INTENT( INOUT ) :: inform
        REAL( rp_), DIMENSION( * ), TARGET, OPTIONAL, INTENT( IN ) :: scaling
      END FUNCTION factor_iface

      SUBROUTINE symbolic_cleanup_iface( this )

!  free associated memory/resources
!   this Instance pointer

        IMPORT symbolic_subtree_base
        IMPLICIT none
        CLASS( symbolic_subtree_base ), INTENT( INOUT ) :: this
      END SUBROUTINE symbolic_cleanup_iface

      FUNCTION get_contrib_iface( this )

!  return contribution block from this subtree to parent.
!  Behaviour is undefined if called on a root subtree.
!  Routine will spinlock with taskyield if factorization is still ongoing.
!  Arguments:
!   this Instance pointer.

        IMPORT contrib_type, numeric_subtree_base
        IMPLICIT none
        TYPE( contrib_type ) :: get_contrib_iface
        CLASS( numeric_subtree_base ), INTENT( IN ) :: this
      END FUNCTION get_contrib_iface

      SUBROUTINE solve_proc_iface( this, nrhs, x, ldx, inform )

!  performs an in-place solve with x. Arguments:
!   this Instance pointer.
!   nrhs Number of right-hand sides.
!   x Right-hand side on entry, solution on return.
!   ldx Leading dimension of x.
!   inform Information/statistics to be returned to user.

        import numeric_subtree_base, ssids_inform_type, ip_, rp_
        implicit none
        CLASS( numeric_subtree_base ), INTENT( INOUT ) :: this
        INTEGER( ip_ ), INTENT( IN ) :: nrhs
        REAL( rp_ ), DIMENSION( * ), INTENT( INOUT ) :: x
        INTEGER( ip_), intent(in) :: ldx
        TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: inform
      END SUBROUTINE solve_proc_iface

      SUBROUTINE numeric_cleanup_iface( this )

!  free associated memory/resources. Arguments:
!   this Instance pointer.

        IMPORT numeric_subtree_base
        IMPLICIT none
        CLASS( numeric_subtree_base ), INTENT( INOUT ) :: this
      END SUBROUTINE numeric_cleanup_iface
   END interface
END MODULE GALAHAD_SSIDS_subtree_precision
