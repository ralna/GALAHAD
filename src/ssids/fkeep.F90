! THIS VERSION: GALAHAD 5.1 - 2024-11-26 AT 13:10 GMT.

#include "spral_procedures.h"

!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
!> \author    Florent Lopez
!
!> \brief Define ssids_fkeep type and associated procedures (CPU version)
module spral_ssids_fkeep_precision
   use spral_kinds_precision
   use :: omp_lib
   use spral_ssids_akeep_precision, only : ssids_akeep
   use spral_ssids_contrib_precision, only : contrib_type
   use spral_ssids_types_precision
   use spral_ssids_inform_precision, only : ssids_inform
   use spral_ssids_subtree_precision, only : numeric_subtree_base
   use spral_ssids_cpu_subtree_precision, only : cpu_numeric_subtree
#ifdef PROFILE
   use spral_ssids_profile_precision, only : profile_begin, profile_end, &
                                             profile_add_event
#endif
   implicit none

   private
   public :: ssids_fkeep

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   type numeric_subtree_ptr
      class(numeric_subtree_base), pointer :: ptr
   end type numeric_subtree_ptr

   !
   ! Data type for data generated in factorise phase
   !
   type ssids_fkeep
      real(rp_), dimension(:), allocatable :: scaling ! Stores scaling for
         ! each entry (in original matrix order)
      logical :: pos_def ! set to true if user indicates matrix pos. definite

      ! Factored subtrees
      type(numeric_subtree_ptr), dimension(:), allocatable :: subtree

      ! Copy of inform on exit from factorize
      type(ssids_inform) :: inform

   contains
      procedure, pass(fkeep) :: inner_factor => inner_factor_cpu 
        ! Do actual factorization
      procedure, pass(fkeep) :: inner_solve => inner_solve_cpu 
        ! Do actual solve
      procedure, pass(fkeep) :: enquire_posdef => enquire_posdef_cpu
      procedure, pass(fkeep) :: enquire_indef => enquire_indef_cpu
      procedure, pass(fkeep) :: alter => alter_cpu ! Alter D values
      procedure, pass(fkeep) :: free => free_fkeep ! Frees memory
   end type ssids_fkeep

contains

subroutine inner_factor_cpu(fkeep, akeep, val, options, inform)
  implicit none
  type(ssids_akeep), intent(in) :: akeep
  class(ssids_fkeep), target, intent(inout) :: fkeep
  real(rp_), dimension(*), target, intent(in) :: val
  type(ssids_options), intent(in) :: options
  type(ssids_inform), intent(inout) :: inform

  integer(ip_) :: i, numa_region, exec_loc, my_loc
  integer(ip_) :: total_threads, max_gpus, to_launch, thread_num
  integer(ip_) :: nth ! Number of threads within a region
  integer(ip_) :: ngpus ! Number of GPUs in a given NUMA region
  logical :: abort, all_region
  type(contrib_type), dimension(:), allocatable :: child_contrib
  type(ssids_inform), dimension(:), allocatable :: thread_inform
#ifdef PROFILE
  ! Begin profile trace (noop if not enabled)
  call profile_begin(akeep%topology)
#endif

  ! Allocate space for subtrees
  allocate(fkeep%subtree(akeep%nparts), stat=inform%stat)
  if(inform%stat.ne.0) goto 200

  ! Determine resources
  total_threads = 0
  max_gpus = 0
  do i = 1, size(akeep%topology)
     total_threads = total_threads + akeep%topology(i)%nproc
     max_gpus = max(max_gpus, size(akeep%topology(i)%gpus))
  end do

  ! Call subtree factor routines
  allocate(child_contrib(akeep%nparts), stat=inform%stat)
  if(inform%stat.ne.0) goto 200
  ! Split into numa regions; parallelism within a region is responsibility
  ! of subtrees.
  to_launch = size(akeep%topology)*(1+max_gpus)
  allocate(thread_inform(to_launch), stat=inform%stat)
  if(inform%stat.ne.0) goto 200
  all_region = .false.

  !$omp parallel proc_bind(spread) num_threads(to_launch) &
  !$omp    default(none) &
  !$omp    private(abort, i, exec_loc, numa_region, my_loc, thread_num) &
  !$omp    private(nth, ngpus) &
  !$omp    shared(akeep, fkeep, val, options, thread_inform, child_contrib, &
  !$omp           all_region) &
  !$omp    if(to_launch.gt.1)

  thread_num = 0
  !$ thread_num = omp_get_thread_num()
  numa_region = mod(thread_num, size(akeep%topology)) + 1
  my_loc = thread_num + 1
  if (thread_num .lt. size(akeep%topology)) then
     ngpus = size(akeep%topology(numa_region)%gpus,1)
     ! CPU, control number of inner threads (not needed for gpu)
     nth = akeep%topology(numa_region)%nproc
     ! nth = nth - ngpus
  else
     nth = 1
  endif

  !$ call omp_set_num_threads(nth)
  ! Split into threads for this NUMA region (unless we're running a GPU)
  exec_loc = -1 ! avoid compiler warning re uninitialized
  abort = .false.

  !$omp parallel proc_bind(close) default(shared) &
  !$omp    num_threads(nth) &
  !$omp    if(my_loc.le.size(akeep%topology))

  !$omp single
  !$omp taskgroup

  do i = 1, akeep%nparts
     exec_loc = akeep%subtree(i)%exec_loc
     if(numa_region.eq.1 .and. exec_loc.eq.-1) all_region = .true.
     if(exec_loc.ne.my_loc) cycle

     !$omp task untied default(shared) firstprivate(i, exec_loc) &
     !$omp    if(my_loc.le.size(akeep%topology))
     if (abort) goto 10
     if (allocated(fkeep%scaling)) then
        fkeep%subtree(i)%ptr => akeep%subtree(i)%ptr%factor( &
             fkeep%pos_def, val, &
             child_contrib(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
             options, thread_inform(my_loc), scaling=fkeep%scaling &
             )
     else
        fkeep%subtree(i)%ptr => akeep%subtree(i)%ptr%factor( &
             fkeep%pos_def, val, &
             child_contrib(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
             options, thread_inform(my_loc) &
             )
     endif
     if (thread_inform(my_loc)%flag .lt. 0) then
        abort = .true.
        goto 10
        ! !$omp    cancel taskgroup
     endif
     if (akeep%contrib_idx(i) .le. akeep%nparts) then
        ! There is a parent subtree to contribute to
        child_contrib(akeep%contrib_idx(i)) = &
             fkeep%subtree(i)%ptr%get_contrib()
        !$omp    flush
        child_contrib(akeep%contrib_idx(i))%ready = .true.
     endif
10   continue ! jump target for abort
     !$omp end task

  end do

  !$omp end taskgroup
  !$omp end single

  !$omp end parallel
  !$omp end parallel
  do i = 1, size(thread_inform)
     call inform%reduce(thread_inform(i))
  end do
  if (inform%flag.lt.0) goto 100 ! cleanup and exit

  if (all_region) then

#ifdef PROFILE
     ! At least some all region subtrees exist
     call profile_add_event("EV_ALL_REGIONS", &
                            "Starting processing root subtree", 0)
#endif

     !$omp parallel num_threads(total_threads) default(shared)
     !$omp single
     do i = 1, akeep%nparts
        exec_loc = akeep%subtree(i)%exec_loc
        if (exec_loc.ne.-1) cycle
        if (allocated(fkeep%scaling)) then
           fkeep%subtree(i)%ptr => akeep%subtree(i)%ptr%factor( &
                fkeep%pos_def, val, &
                child_contrib(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
                options, inform, scaling=fkeep%scaling &
                )
        else
           fkeep%subtree(i)%ptr => akeep%subtree(i)%ptr%factor( &
                fkeep%pos_def, val, &
                child_contrib(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
                options, inform &
                )
        endif
        if (akeep%contrib_idx(i) .gt. akeep%nparts) cycle ! part is a root
        child_contrib(akeep%contrib_idx(i)) = &
             fkeep%subtree(i)%ptr%get_contrib()
        !$omp    flush
        child_contrib(akeep%contrib_idx(i))%ready = .true.
     end do
     !$omp end single
     !$omp end parallel
  end if

100 continue ! cleanup and exit

#ifdef PROFILE
  ! End profile trace (noop if not enabled)
  call profile_end()
#endif

  return

200 continue
  inform%flag = SSIDS_ERROR_ALLOCATION
  goto 100 ! cleanup and exit
end subroutine inner_factor_cpu

subroutine inner_solve_cpu(local_job, nrhs, x, ldx, akeep, fkeep, inform)
   type(ssids_akeep), intent(in) :: akeep
   class(ssids_fkeep), intent(inout) :: fkeep
   integer(ip_), intent(inout) :: local_job
   integer(ip_), intent(in) :: nrhs
   integer(ip_), intent(in) :: ldx
   real(rp_), dimension(ldx,nrhs), target, intent(inout) :: x
   type(ssids_inform), intent(inout) :: inform

   integer(ip_) :: i, r, part
   integer(ip_) :: n
   real(rp_), dimension(:,:), allocatable :: x2

   n = akeep%n

   allocate(x2(n, nrhs), stat=inform%stat)
   if(inform%stat.ne.0) goto 100

   ! Permute/scale
   if (allocated(fkeep%scaling) .and. (local_job == SSIDS_SOLVE_JOB_ALL .or. &
            local_job == SSIDS_SOLVE_JOB_FWD)) then
      ! Copy and scale
      do r = 1, nrhs
         do i = 1, n
            x2(i,r) = x(akeep%invp(i),r) * fkeep%scaling(i)
         end do
      end do
   else
      ! Just copy
      do r = 1, nrhs
         x2(1:n, r) = x(akeep%invp(1:n), r)
      end do
   end if

   ! Perform relevant solves
   if (local_job.eq.SSIDS_SOLVE_JOB_FWD .or. &
         local_job.eq.SSIDS_SOLVE_JOB_ALL) then
      do part = 1, akeep%nparts
         call fkeep%subtree(part)%ptr%solve_fwd(nrhs, x2, n, inform)
         if (inform%stat .ne. 0) goto 100
      end do
   endif

   if (local_job.eq.SSIDS_SOLVE_JOB_DIAG) then
      do part = 1, akeep%nparts
         call fkeep%subtree(part)%ptr%solve_diag(nrhs, x2, n, inform)
         if (inform%stat .ne. 0) goto 100
      end do
   endif

   if (local_job.eq.SSIDS_SOLVE_JOB_BWD) then
      do part = akeep%nparts, 1, -1
         call fkeep%subtree(part)%ptr%solve_bwd(nrhs, x2, n, inform)
         if (inform%stat .ne. 0) goto 100
      end do
   endif

   if (local_job.eq.SSIDS_SOLVE_JOB_DIAG_BWD .or. &
         local_job.eq.SSIDS_SOLVE_JOB_ALL) then
      do part = akeep%nparts, 1, -1
         call fkeep%subtree(part)%ptr%solve_diag_bwd(nrhs, x2, n, inform)
         if (inform%stat .ne. 0) goto 100
      end do
   endif

   ! Unscale/unpermute
   if (allocated(fkeep%scaling) .and. ( &
            local_job == SSIDS_SOLVE_JOB_ALL .or. &
            local_job == SSIDS_SOLVE_JOB_BWD .or. &
            local_job == SSIDS_SOLVE_JOB_DIAG_BWD)) then
      ! Copy and scale
      do r = 1, nrhs
         do i = 1, n
            x(akeep%invp(i),r) = x2(i,r) * fkeep%scaling(i)
         end do
      end do
   else
      ! Just copy
      do r = 1, nrhs
         x(akeep%invp(1:n), r) = x2(1:n, r)
      end do
   end if

   return

   100 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   return

end subroutine inner_solve_cpu

!****************************************************************************

subroutine enquire_posdef_cpu(akeep, fkeep, d)
   type(ssids_akeep), intent(in) :: akeep
   class(ssids_fkeep), target, intent(in) :: fkeep
   real(rp_), dimension(*), intent(out) :: d

   integer(ip_) :: n
   integer(ip_) :: part, sa, en

   n = akeep%n
   ! ensure d is not returned undefined
   d(1:n) = 0.0 ! ensure do not returned with this undefined

   do part = 1, akeep%nparts
      sa = akeep%part(part)
      en = akeep%part(part+1)-1
      associate(subtree => fkeep%subtree(part)%ptr)
         select type(subtree)
         type is (cpu_numeric_subtree)
            call subtree%enquire_posdef(d(sa:en))
         end select
      end associate
   end do

end subroutine enquire_posdef_cpu

!****************************************************************************

subroutine enquire_indef_cpu(akeep, fkeep, inform, piv_order, d)
   type(ssids_akeep), intent(in) :: akeep
   class(ssids_fkeep), target, intent(in) :: fkeep
   type(ssids_inform), intent(inout) :: inform
   integer(ip_), dimension(akeep%n), optional, intent(out) :: piv_order
      ! If i is used to index a variable, its position in the pivot sequence
      ! will be placed in piv_order(i), with its sign negative if it is
      ! part of a 2 x 2 pivot; otherwise, piv_order(i) will be set to zero.
   real(rp_), dimension(2,akeep%n), optional, intent(out) :: d ! The diagonal
      ! entries of D^{-1} will be placed in d(1,:i) and the off-diagonal
      ! entries will be placed in d(2,:). The entries are held in pivot order.

   integer(ip_) :: part, sa
   integer(ip_) :: i, n
   integer(ip_), dimension(:), allocatable :: po

   n = akeep%n
   if(present(d)) then
      ! ensure d is not returned undefined
      d(1:2,1:n) = 0.0
   end if

   ! We need to apply the invp externally to piv_order
   if(present(piv_order)) then
      allocate(po(akeep%n), stat=inform%stat)
      if(inform%stat.ne.0) then
         inform%flag = SSIDS_ERROR_ALLOCATION
         return
      endif
   endif

   ! FIXME: should probably return nelim from each part, due to delays passing
   ! between them
   do part = 1, akeep%nparts
      sa = akeep%part(part)
      associate(subtree => fkeep%subtree(1)%ptr)
         select type(subtree)
         type is (cpu_numeric_subtree)
            if(present(d)) then
               if(present(piv_order)) then
                  call subtree%enquire_indef(piv_order=po(sa:n), d=d(1:2,sa:n))
               else
                  call subtree%enquire_indef(d=d(1:2,sa:))
               endif
            else
               if(present(piv_order)) then
                  call subtree%enquire_indef(piv_order=po(sa:akeep%n))
               else
                  ! No-op
                  ! FIXME: should we report an error here? (or done higher up?)
               endif
            endif
         end select
      end associate
   end do

   ! Apply invp to piv_order
   if(present(piv_order)) then
      do i = 1, akeep%n
         piv_order( akeep%invp(i) ) = po(i)
      end do
   endif

end subroutine enquire_indef_cpu

! Alter D values
subroutine alter_cpu(d, akeep, fkeep)
   real(rp_), dimension(2,*), intent(in) :: d  ! The required diagonal entries
     ! of D^{-1} must be placed in d(1,i) (i = 1,...n)
     ! and the off-diagonal entries must be placed in d(2,i) (i = 1,...n-1).
   type(ssids_akeep), intent(in) :: akeep
   class(ssids_fkeep), target, intent(inout) :: fkeep

   integer(ip_) :: part

   do part = 1, akeep%nparts
      associate(subtree => fkeep%subtree(1)%ptr)
         select type(subtree)
         type is (cpu_numeric_subtree)
            call subtree%alter(d(1:2,akeep%part(part):akeep%part(part+1)-1))
         end select
      end associate
   end do
end subroutine alter_cpu

!****************************************************************************

subroutine free_fkeep(fkeep, flag)
   class(ssids_fkeep), intent(inout) :: fkeep
   integer(ip_), intent(out) :: flag ! not used for cpu version, set to 0

   integer(ip_) :: i
   integer(ip_) :: st

   flag = 0 ! Not used for basic SSIDS, just set to zero

   deallocate(fkeep%scaling, stat=st)
   if(allocated(fkeep%subtree)) then
      do i = 1, size(fkeep%subtree)
         if(associated(fkeep%subtree(i)%ptr)) then
            call fkeep%subtree(i)%ptr%cleanup()
            deallocate(fkeep%subtree(i)%ptr)
            nullify(fkeep%subtree(i)%ptr)
         endif
      end do
      deallocate(fkeep%subtree)
   endif
end subroutine free_fkeep

end module spral_ssids_fkeep_precision
