!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_akeep
   use spral_ssids_datatypes, only : long, wp, SSIDS_ERROR_CUDA_UNKNOWN, &
                                     ssids_options
   use spral_hw_topology, only : numa_region
   use spral_ssids_inform, only : ssids_inform
   use spral_ssids_subtree, only : symbolic_subtree_base
   use, intrinsic :: iso_c_binding
   implicit none

   private
   public :: ssids_akeep

   type symbolic_subtree_ptr
      integer :: exec_loc
      class(symbolic_subtree_base), pointer :: ptr => null()
   end type symbolic_subtree_ptr

   !
   ! Data type for information generated in analyse phase
   !
   type ssids_akeep
      logical :: check ! copy of check as input to analyse phase
      integer :: n ! Dimension of matrix
      integer :: nnodes = -1 ! Number of nodes in assembly tree

      ! Subtree partition
      integer :: nparts
      integer, dimension(:), allocatable :: part
      type(symbolic_subtree_ptr), dimension(:), allocatable :: subtree
      integer, dimension(:), allocatable :: contrib_ptr
      integer, dimension(:), allocatable :: contrib_idx

      integer(C_INT), dimension(:), allocatable :: invp ! inverse of pivot order
         ! that is passed to factorize phase
      integer(long), dimension(:,:), allocatable :: nlist ! map from A to
         ! factors. For nodes i, the entries nlist(1:2, nptr(i):nptr(i+1)-1)
         ! define a relationship:
         ! nodes(node)%lcol( nlist(2,j) ) = val( nlist(1,j) )
     integer(long), dimension(:), allocatable :: nptr ! Entries into nlist for
         ! nodes of the assembly tree. Has length nnodes+1
      integer, dimension(:), allocatable :: rlist ! rlist(rptr(i):rptr(i+1)-1)
         ! contains the row indices for node i of the assembly tree.
         ! At each node, the list
         ! is in elimination order. Allocated within mc78_analyse.
      integer(long), dimension(:), allocatable :: rptr ! Pointers into rlist
         ! for nodes of assembly tree. Has length nnodes+1.
         ! Allocated within mc78_analyse.
      integer, dimension(:), allocatable :: sparent ! sparent(i) is parent
         ! of node i in assembly tree. sparent(i)=nnodes+1 if i is a root.
         ! The parent is always numbered higher than each of its children.
         ! Allocated within mc78_analyse.
      integer, dimension(:), allocatable :: sptr ! (super)node pointers.
         ! Supernode i consists of sptr(i) through sptr(i+1)-1.
         ! Allocated within mc78_analyse.

      ! Following components are for cleaned up matrix data.
      ! LOWER triangle only. We have to retain these for factorize phase
      ! as used if the user wants to do scaling.
      ! These components are NOT used if check is set to .false.
      ! on call to ssids_analyse.
      integer(long), allocatable :: ptr(:) ! column pointers
      integer, allocatable :: row(:) ! row indices
      integer(long) :: lmap ! length of map
      integer(long), allocatable :: map(:) ! map from old A to cleaned A

      ! Scaling from matching-based ordering
      real(wp), dimension(:), allocatable :: scaling

      ! Machine topology
      type(numa_region), dimension(:), allocatable :: topology

      ! Inform at end of analyse phase
      type(ssids_inform) :: inform
   contains
      procedure, pass(akeep) :: free => free_akeep
   end type ssids_akeep

contains

subroutine free_akeep(akeep, flag)
   class(ssids_akeep), intent(inout) :: akeep
   integer, intent(out) :: flag

   integer :: i
   integer :: st

   flag = 0

   deallocate(akeep%part, stat=st)
   if (allocated(akeep%subtree)) then
      do i = 1, size(akeep%subtree)
         if (associated(akeep%subtree(i)%ptr)) then
            call akeep%subtree(i)%ptr%cleanup()
            deallocate(akeep%subtree(i)%ptr)
            nullify(akeep%subtree(i)%ptr)
         endif
      end do
      deallocate(akeep%subtree, stat=st)
   endif
   deallocate(akeep%contrib_ptr, stat=st)
   deallocate(akeep%contrib_idx, stat=st)
   deallocate(akeep%invp, stat=st)
   deallocate(akeep%nlist, stat=st)
   deallocate(akeep%nptr, stat=st)
   deallocate(akeep%rlist, stat=st)
   deallocate(akeep%rptr, stat=st)
   deallocate(akeep%sparent, stat=st)
   deallocate(akeep%sptr, stat=st)
   deallocate(akeep%ptr, stat=st)
   deallocate(akeep%row, stat=st)
   deallocate(akeep%map, stat=st)
   deallocate(akeep%scaling, stat=st)
   deallocate(akeep%topology, stat=st)
end subroutine free_akeep

end module spral_ssids_akeep
