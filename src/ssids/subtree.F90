!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_subtree
   use spral_ssids_contrib, only : contrib_type
   use spral_ssids_datatypes, only : long, wp, ssids_options
   use spral_ssids_inform, only : ssids_inform
   implicit none

   private
   public :: symbolic_subtree_base, numeric_subtree_base

   !> @brief Abstract base class for Symbolic subtrees.
   !>
   !> The symbolic subtrees encode the information from the analyse phase
   !> necessary to generate a numeric factorization in the subsequent
   !> factorization phases, which call the factor() entry.
   !>
   !> A subtree may have child subtrees that hang off it. At factorization
   !> time, multifrontal contribution blocks from all children will be supplied.
   !> Each subtree must in turn generate a contribution block if it is not a
   !> root subtree.
   !>
   !> @sa numeric_subtree_base
   type, abstract :: symbolic_subtree_base
   contains
      !> @brief Perform numeric factorization, returning a subclass of
      !>        numeric_subtree_base representing this.
      procedure(factor_iface), deferred :: factor
      !> @brief Free associated memory/resources
      procedure(symbolic_cleanup_iface), deferred :: cleanup
   end type symbolic_subtree_base

   !> @brief Abstract base class for Numeric subtrees.
   !>
   !> The numeric subtree represents the numeric factorization of a subtree
   !> and is returned from the corresponding factor() call of a Symbolic
   !> subtree.
   !>
   !> @sa symbolic_subtree_base
   type, abstract :: numeric_subtree_base
   contains
      !> @brief Return contribution block from this subtree to parent.
      !>        Behaviour is undefined if called on a root subtree.
      !>        Routine will spinlock with taskyield if factorization is still
      !>        ongoing.
      procedure(get_contrib_iface), deferred :: get_contrib
      !> @brief Perform forward solve.
      procedure(solve_proc_iface), deferred :: solve_fwd
      !> @brief Perform diagonal solve.
      procedure(solve_proc_iface), deferred :: solve_diag
      !> @brief Perform combined diagonal and backward solve.
      procedure(solve_proc_iface), deferred :: solve_diag_bwd
      !> @brief Perform backward solve.
      procedure(solve_proc_iface), deferred :: solve_bwd
      !> @brief Free associated memory/resources
      procedure(numeric_cleanup_iface), deferred :: cleanup
   end type numeric_subtree_base

   abstract interface
      !> @brief Perform numeric factorization, returning a subclass of
      !>        numeric_subtree_base representing this.
      !> @param this Instance pointer.
      !> @param posdef Perform Cholesky-like unpivoted factorization if true.
      !> @param aval Value component of CSC datatype for original matrix A.
      !> @param child_contrib Array of contribution blocks from children.
      !> @param options User-supplied options.
      !> @param inform Information/statistics to be returned to user.
      !> @param scaling Scaling to be applied (if present).
      function factor_iface(this, posdef, aval, child_contrib, options, &
            inform, scaling)
         import symbolic_subtree_base, numeric_subtree_base, wp
         import ssids_inform, ssids_options
         import contrib_type
         implicit none
         class(numeric_subtree_base), pointer :: factor_iface
         class(symbolic_subtree_base), target, intent(inout) :: this
         logical, intent(in) :: posdef
         real(wp), dimension(*), target, intent(in) :: aval
         type(contrib_type), dimension(:), target, intent(inout) :: child_contrib
         type(ssids_options), intent(in) :: options
         type(ssids_inform), intent(inout) :: inform
         real(wp), dimension(*), target, optional, intent(in) :: scaling
      end function factor_iface
      !> @brief Free associated memory/resources
      !> @param this Instance pointer.
      subroutine symbolic_cleanup_iface(this)
         import symbolic_subtree_base
         implicit none
         class(symbolic_subtree_base), intent(inout) :: this
      end subroutine symbolic_cleanup_iface
      !> @brief Return contribution block from this subtree to parent.
      !>
      !> Behaviour is undefined if called on a root subtree.
      !> Routine will spinlock with taskyield if factorization is still ongoing.
      !>
      !> @param this Instance pointer.
      function get_contrib_iface(this)
         import contrib_type, numeric_subtree_base
         implicit none
         type(contrib_type) :: get_contrib_iface
         class(numeric_subtree_base), intent(in) :: this
      end function get_contrib_iface
      !> @brief Performs an in-place solve with x.
      !> @param this Instance pointer.
      !> @param nrhs Number of right-hand sides.
      !> @param x Right-hand side on entry, solution on return.
      !> @param ldx Leading dimension of x.
      !> @param inform Information/statistics to be returned to user.
      subroutine solve_proc_iface(this, nrhs, x, ldx, inform)
         import numeric_subtree_base, ssids_inform, wp
         implicit none
         class(numeric_subtree_base), intent(inout) :: this
         integer, intent(in) :: nrhs
         real(wp), dimension(*), intent(inout) :: x
         integer, intent(in) :: ldx
         type(ssids_inform), intent(inout) :: inform
      end subroutine solve_proc_iface
      !> @brief Free associated memory/resources
      !> @param this Instance pointer.
      subroutine numeric_cleanup_iface(this)
         import numeric_subtree_base
         implicit none
         class(numeric_subtree_base), intent(inout) :: this
      end subroutine numeric_cleanup_iface
   end interface
end module spral_ssids_subtree
