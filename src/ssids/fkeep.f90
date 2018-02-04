!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
!
!> \brief Define ssids_fkeep type and associated procedures (CPU version)
module spral_ssids_fkeep
  use, intrinsic :: iso_c_binding
!$ use :: omp_lib
  use spral_ssids_akeep, only : ssids_akeep
  use spral_ssids_contrib, only : contrib_type
  use spral_ssids_datatypes
  use spral_ssids_inform, only : ssids_inform
  use spral_ssids_subtree, only : numeric_subtree_base
  use spral_ssids_cpu_subtree, only : cpu_numeric_subtree
  use spral_ssids_profile, only : profile_begin, profile_end
  implicit none

  private
  public :: ssids_fkeep

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  type numeric_subtree_ptr
     class(numeric_subtree_base), pointer :: ptr => null()
  end type numeric_subtree_ptr

  !
  ! Data type for data generated in factorise phase
  !
  type ssids_fkeep
     real(wp), dimension(:), allocatable :: scaling ! Stores scaling for
     ! each entry (in original matrix order)
     logical :: pos_def ! set to true if user indicates matrix pos. definite

     ! Factored subtrees
     type(numeric_subtree_ptr), dimension(:), allocatable :: subtree

     ! Copy of inform on exit from factorize
     type(ssids_inform) :: inform

   contains
     procedure, pass(fkeep) :: inner_factor => inner_factor_cpu ! Do actual factorization
     procedure, pass(fkeep) :: inner_solve => inner_solve_cpu ! Do actual solve
     procedure, pass(fkeep) :: enquire_posdef => enquire_posdef_cpu
     procedure, pass(fkeep) :: enquire_indef => enquire_indef_cpu
     procedure, pass(fkeep) :: alter => alter_cpu ! Alter D values
     procedure, pass(fkeep) :: free => free_fkeep ! Frees memory
  end type ssids_fkeep

contains

  subroutine inner_factor_numa(fkeep, akeep, val, options, inform, &
       child_contrib, all_region)
    implicit none
    type(ssids_akeep), intent(in) :: akeep
    class(ssids_fkeep), intent(inout) :: fkeep
    real(wp), dimension(*), target, intent(in) :: val
    type(ssids_options), intent(in) :: options
    type(ssids_inform), dimension(*), intent(inout) :: inform
    type(contrib_type), dimension(*), intent(inout) :: child_contrib
    logical, intent(inout) :: all_region

    integer :: i, to_launch, numa_region, exec_loc, my_loc
    logical :: abort, my_abort

    numa_region = 0
!$  numa_region = omp_get_thread_num()
    numa_region = numa_region + 1

    to_launch = akeep%topology(numa_region)%nproc
    if (to_launch .le. 0) return
!$  call omp_set_num_threads(to_launch)

!$omp atomic write
    abort = .false.
!$omp end atomic

    ! Split into threads for this NUMA region
!$omp parallel proc_bind(close)                 &
!$omp    default(shared)                        &
!$omp    private(i, exec_loc, my_abort, my_loc) &
!$omp    num_threads(to_launch)
!$omp single
!$omp taskgroup
    do i = 1, akeep%nparts
       exec_loc = akeep%subtree(i)%exec_loc
       if (exec_loc .eq. -1) then
!$omp atomic write
          all_region = .true.
!$omp end atomic
          cycle
       end if
       if ((mod((exec_loc-1), size(akeep%topology))+1) .ne. numa_region) cycle
       my_abort = .false.
       my_loc = 0
!$omp task untied                                    &
!$omp    default(shared)                             &
!$omp    firstprivate(i, exec_loc, my_abort, my_loc)
!$omp atomic read
       my_abort = abort
!$omp end atomic
       if (my_abort) goto 10
!$     my_loc = omp_get_thread_num()
       my_loc = my_loc + 1
       if (allocated(fkeep%scaling)) then
          fkeep%subtree(i)%ptr => akeep%subtree(i)%ptr%factor(               &
               fkeep%pos_def, val,                                           &
               child_contrib(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
               options, inform(my_loc), scaling=fkeep%scaling                &
               )
       else
          fkeep%subtree(i)%ptr => akeep%subtree(i)%ptr%factor(               &
               fkeep%pos_def, val,                                           &
               child_contrib(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
               options, inform(my_loc)                                       &
               )
       end if
       if (inform(my_loc)%flag .lt. 0) then
!$omp atomic write
          abort = .true.
!$omp end atomic
          goto 10
!$omp     cancel taskgroup
       end if
       if (akeep%contrib_idx(i) .le. akeep%nparts) then
          ! There is a parent subtree to contribute to
          child_contrib(akeep%contrib_idx(i)) = &
               fkeep%subtree(i)%ptr%get_contrib()
          child_contrib(akeep%contrib_idx(i))%ready = .true.
       end if
10     continue ! jump target for abort
!$omp end task
    end do
!$omp end taskgroup
!$omp end single
!$omp end parallel
  end subroutine inner_factor_numa

  subroutine inner_factor_cpu(fkeep, akeep, val, options, inform)
    implicit none
    type(ssids_akeep), intent(in) :: akeep
    class(ssids_fkeep), intent(inout) :: fkeep ! target
    real(wp), dimension(*), target, intent(in) :: val
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    integer :: i, numa_region, exec_loc
    integer :: numa_regions, region_threads !, device
    integer :: max_cpus, max_cpus_per_numa, total_threads
    logical :: all_region, my_all_region
    type(contrib_type), dimension(:), allocatable :: child_contrib
    type(ssids_inform), dimension(:), allocatable :: thread_inform

    ! Begin profile trace (noop if not enabled)
    call profile_begin()

    ! Allocate space for subtrees
    allocate(fkeep%subtree(akeep%nparts), stat=inform%stat)
    if (inform%stat .ne. 0) goto 200

    numa_regions = size(akeep%topology)
    if (numa_regions .eq. 0) numa_regions = 1

    ! do i = 1, akeep%nparts
    !    exec_loc = akeep%subtree(i)%exec_loc
    !    if (exec_loc .eq. -1) then
    !       numa_region = 0
    !       device = 0
    !       all_region = .true.
    !    else
    !       numa_region = mod((exec_loc-1), numa_regions) + 1
    !       device = ((exec_loc-1) / numa_regions) + 1
    !    end if
    ! end do

    max_cpus_per_numa = 0
    total_threads = 0
    do numa_region = 1, numa_regions
       region_threads = akeep%topology(numa_region)%nproc
       max_cpus_per_numa = max(max_cpus_per_numa, region_threads)
       total_threads = total_threads + region_threads
    end do
    max_cpus = max_cpus_per_numa * numa_regions

    allocate(child_contrib(akeep%nparts), stat=inform%stat)
    if (inform%stat .ne. 0) goto 200
    allocate(thread_inform(max_cpus), stat=inform%stat)
    if (inform%stat .ne. 0) goto 200

    ! Call subtree factor routines
    ! Split into numa regions; parallelism within a region is responsibility
    ! of subtrees.

!$omp atomic write
    all_region = .false.
!$omp end atomic

!$omp parallel proc_bind(spread) num_threads(numa_regions) &
!$omp    default(none)                                     &
!$omp    shared(akeep, fkeep, val, options, thread_inform, &
!$omp       child_contrib, all_region, max_cpus_per_numa)  &
!$omp    private(i,numa_region)
    numa_region = 0
!$  numa_region = omp_get_thread_num()
    numa_region = numa_region + 1
    i = (numa_region-1)*max_cpus_per_numa + 1
    call inner_factor_numa(fkeep, akeep, val, options, &
         thread_inform(i), child_contrib, all_region)
!$omp end parallel
    do i = 1, max_cpus
       call inform%reduce(thread_inform(i))
    end do
    if (inform%flag .lt. 0) goto 100 ! cleanup and exit

!$omp atomic read
    my_all_region = all_region
!$omp end atomic
    if (my_all_region) then
       ! At least some all region subtrees exist
!$omp parallel num_threads(total_threads) default(shared) private(i,exec_loc)
!$omp single
       do i = 1, akeep%nparts
          exec_loc = akeep%subtree(i)%exec_loc
          if (exec_loc .ne. -1) cycle
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
          end if
          if (akeep%contrib_idx(i) .gt. akeep%nparts) cycle ! part is a root
          child_contrib(akeep%contrib_idx(i)) = &
               fkeep%subtree(i)%ptr%get_contrib()
          child_contrib(akeep%contrib_idx(i))%ready = .true.
       end do
!$omp end single
!$omp end parallel
    end if

100 continue ! cleanup and exit

    ! End profile trace (noop if not enabled)
    call profile_end()

    return

200 continue
    inform%flag = SSIDS_ERROR_ALLOCATION
    goto 100 ! cleanup and exit
  end subroutine inner_factor_cpu

  subroutine inner_solve_cpu(local_job, nrhs, x, ldx, akeep, fkeep, inform)
    implicit none
    type(ssids_akeep), intent(in) :: akeep
    class(ssids_fkeep), intent(inout) :: fkeep
    integer, intent(inout) :: local_job
    integer, intent(in) :: nrhs
    integer, intent(in) :: ldx
    real(wp), dimension(ldx,nrhs), target, intent(inout) :: x
    type(ssids_inform), intent(inout) :: inform

    integer :: i, r, part
    integer :: n
    real(wp), dimension(:,:), allocatable :: x2

    n = akeep%n

    allocate(x2(n, nrhs), stat=inform%stat)
    if (inform%stat .ne. 0) goto 100

    ! Permute/scale
    if (allocated(fkeep%scaling) .and. ( &
         (local_job .eq. SSIDS_SOLVE_JOB_ALL) .or. &
         (local_job .eq. SSIDS_SOLVE_JOB_FWD))) then
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
    if ((local_job .eq. SSIDS_SOLVE_JOB_FWD) .or. &
         (local_job .eq. SSIDS_SOLVE_JOB_ALL)) then
       do part = 1, akeep%nparts
          call fkeep%subtree(part)%ptr%solve_fwd(nrhs, x2, n, inform)
          if (inform%stat .ne. 0) goto 100
       end do
    end if

    if (local_job .eq. SSIDS_SOLVE_JOB_DIAG) then
       do part = 1, akeep%nparts
          call fkeep%subtree(part)%ptr%solve_diag(nrhs, x2, n, inform)
          if (inform%stat .ne. 0) goto 100
       end do
    end if

    if (local_job .eq. SSIDS_SOLVE_JOB_BWD) then
       do part = akeep%nparts, 1, -1
          call fkeep%subtree(part)%ptr%solve_bwd(nrhs, x2, n, inform)
          if (inform%stat .ne. 0) goto 100
       end do
    end if

    if ((local_job .eq. SSIDS_SOLVE_JOB_DIAG_BWD) .or. &
         (local_job .eq. SSIDS_SOLVE_JOB_ALL)) then
       do part = akeep%nparts, 1, -1
          call fkeep%subtree(part)%ptr%solve_diag_bwd(nrhs, x2, n, inform)
          if (inform%stat .ne. 0) goto 100
       end do
    end if

    ! Unscale/unpermute
    if (allocated(fkeep%scaling) .and. ( &
         (local_job .eq. SSIDS_SOLVE_JOB_ALL) .or. &
         (local_job .eq. SSIDS_SOLVE_JOB_BWD) .or. &
         (local_job .eq. SSIDS_SOLVE_JOB_DIAG_BWD))) then
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
    implicit none
    type(ssids_akeep), intent(in) :: akeep
    class(ssids_fkeep), target, intent(in) :: fkeep
    real(wp), dimension(*), intent(out) :: d

    integer :: n
    integer :: part, sa, en

    n = akeep%n
    ! ensure d is not returned undefined
    d(1:n) = 0.0 ! ensure do not returned with this undefined

    do part = 1, akeep%nparts
       sa = akeep%part(part)
       en = akeep%part(part+1)-1
       associate(subtree => fkeep%subtree(part)%ptr)
         select type (subtree)
         type is (cpu_numeric_subtree)
            call subtree%enquire_posdef(d(sa:en))
         end select
       end associate
    end do
  end subroutine enquire_posdef_cpu

!****************************************************************************

  subroutine enquire_indef_cpu(akeep, fkeep, inform, piv_order, d)
    implicit none
    type(ssids_akeep), intent(in) :: akeep
    class(ssids_fkeep), target, intent(in) :: fkeep
    type(ssids_inform), intent(inout) :: inform
    integer, dimension(akeep%n), optional, intent(out) :: piv_order
      ! If i is used to index a variable, its position in the pivot sequence
      ! will be placed in piv_order(i), with its sign negative if it is
      ! part of a 2 x 2 pivot; otherwise, piv_order(i) will be set to zero.
    real(wp), dimension(2,akeep%n), optional, intent(out) :: d ! The diagonal
      ! entries of D^{-1} will be placed in d(1,:i) and the off-diagonal
      ! entries will be placed in d(2,:). The entries are held in pivot order.

    integer :: part, sa
    integer :: i, n
    integer, dimension(:), allocatable :: po

    n = akeep%n
    if (present(d)) then
       ! ensure d is not returned undefined
       d(1:2,1:n) = 0.0
    end if

    ! We need to apply the invp externally to piv_order
    if (present(piv_order)) then
       allocate(po(akeep%n), stat=inform%stat)
       if (inform%stat .ne. 0) then
          inform%flag = SSIDS_ERROR_ALLOCATION
          return
       end if
    end if

    ! FIXME: should probably return nelim from each part, due to delays passing
    ! between them
    do part = 1, akeep%nparts
       sa = akeep%part(part)
       associate(subtree => fkeep%subtree(1)%ptr)
         select type (subtree)
         type is (cpu_numeric_subtree)
            if (present(d)) then
               if (present(piv_order)) then
                  call subtree%enquire_indef(piv_order=po(sa:n), d=d(1:2,sa:n))
               else
                  call subtree%enquire_indef(d=d(1:2,sa:))
               end if
            else
               if (present(piv_order)) then
                  call subtree%enquire_indef(piv_order=po(sa:akeep%n))
               else
                  ! FIXME: should we report an error here? (or done higher up?)
                  continue ! No-op
               end if
            end if
         end select
       end associate
    end do

    ! Apply invp to piv_order
    if (present(piv_order)) then
       do i = 1, akeep%n
          piv_order( akeep%invp(i) ) = po(i)
       end do
    end if
  end subroutine enquire_indef_cpu

  ! Alter D values
  subroutine alter_cpu(d, akeep, fkeep)
    implicit none
    real(wp), dimension(2,*), intent(in) :: d  ! The required diagonal entries
      ! of D^{-1} must be placed in d(1,i) (i = 1,...n)
      ! and the off-diagonal entries must be placed in d(2,i) (i = 1,...n-1).
    type(ssids_akeep), intent(in) :: akeep
    class(ssids_fkeep), target, intent(inout) :: fkeep

    integer :: part

    do part = 1, akeep%nparts
       associate(subtree => fkeep%subtree(1)%ptr)
         select type (subtree)
         type is (cpu_numeric_subtree)
            call subtree%alter(d(1:2,akeep%part(part):akeep%part(part+1)-1))
         end select
       end associate
    end do
  end subroutine alter_cpu

!****************************************************************************

  subroutine free_fkeep(fkeep, flag)
    implicit none
    class(ssids_fkeep), intent(inout) :: fkeep
    integer, intent(out) :: flag ! not actually used for cpu version, set to 0

    integer :: i
    integer :: st

    flag = 0 ! Not used for basic SSIDS, just zet to zero

    deallocate(fkeep%scaling, stat=st)
    if (allocated(fkeep%subtree)) then
       do i = 1, size(fkeep%subtree)
          if (associated(fkeep%subtree(i)%ptr)) then
             call fkeep%subtree(i)%ptr%cleanup()
             deallocate(fkeep%subtree(i)%ptr)
             nullify(fkeep%subtree(i)%ptr)
          end if
       end do
       deallocate(fkeep%subtree)
    end if
  end subroutine free_fkeep
end module spral_ssids_fkeep
