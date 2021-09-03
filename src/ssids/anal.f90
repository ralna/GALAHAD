!> \copyright 2010-2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
!> \note      Originally based on HSL_MA97 v2.2.0
module spral_ssids_anal
  use, intrinsic :: iso_c_binding
!$ use :: omp_lib
  use spral_core_analyse, only : basic_analyse
  use spral_cuda, only : detect_gpu
  use spral_hw_topology, only : guess_topology, numa_region
  use spral_pgm, only : writePPM
  use spral_ssids_akeep, only : ssids_akeep
  use spral_ssids_cpu_subtree, only : construct_cpu_symbolic_subtree
  use spral_ssids_gpu_subtree, only : construct_gpu_symbolic_subtree
  use spral_ssids_datatypes
  use spral_ssids_inform, only : ssids_inform
  implicit none

  private
  public :: analyse_phase,   & ! Calls core analyse and builds data strucutres
            check_order,     & ! Check order is a valid permutation
            expand_pattern,  & ! Specialised half->full matrix conversion
            expand_matrix      ! Specialised half->full matrix conversion

contains

!****************************************************************************

!
! Given lower triangular part of A held in row and ptr, expand to
! upper and lower triangular parts (pattern only). No checks.
!
! Note: we do not use half_to_full here to expand A since, if we did, we would
! need an extra copy of the lower triangle into the full structure before
! calling half_to_full
!
  subroutine expand_pattern(n,nz,ptr,row,aptr,arow)
    implicit none
    integer, intent(in) :: n ! order of system
    integer(long), intent(in) :: nz
    integer(long), intent(in) :: ptr(n+1)
    integer, intent(in) :: row(nz)
    integer(long), intent(out) :: aptr(n+1)
    integer, intent(out) :: arow(2*nz)

    integer :: i,j
    integer(long) :: kk

    ! Set aptr(j) to hold no. nonzeros in column j
    aptr(:) = 0
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          aptr(i) = aptr(i) + 1
          if (j .eq. i) cycle
          aptr(j) = aptr(j) + 1
       end do
    end do

    ! Set aptr(j) to point to where row indices will end in arow
    do j = 2, n
       aptr(j) = aptr(j-1) + aptr(j)
    end do
    aptr(n+1) = aptr(n) + 1

    ! Fill arow and aptr
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          arow(aptr(i)) = j
          aptr(i) = aptr(i) - 1
          if (j .eq. i) cycle
          arow(aptr(j)) = i
          aptr(j) = aptr(j) - 1
       end do
    end do
    do j = 1,n
       aptr(j) = aptr(j) + 1
    end do
  end subroutine expand_pattern

!****************************************************************************
!
! Given lower triangular part of A held in row, val and ptr, expand to
! upper and lower triangular parts.

  subroutine expand_matrix(n,nz,ptr,row,val,aptr,arow,aval)
    implicit none
    integer, intent(in)   :: n ! order of system
    integer(long), intent(in)   :: nz
    integer(long), intent(in)   :: ptr(n+1)
    integer, intent(in)   :: row(nz)
    real(wp), intent(in)  :: val(nz)
    integer(long), intent(out)  :: aptr(n+1)
    integer, intent(out)  :: arow(2*nz)
    real(wp), intent(out) :: aval(2*nz)

    integer :: i,j
    integer(long) :: kk, ipos, jpos
    real(wp) :: atemp

    ! Set aptr(j) to hold no. nonzeros in column j
    aptr(:) = 0
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          aptr(i) = aptr(i) + 1
          if (j .eq. i) cycle
          aptr(j) = aptr(j) + 1
       end do
    end do

    ! Set aptr(j) to point to where row indices will end in arow
    do j = 2, n
       aptr(j) = aptr(j-1) + aptr(j)
    end do
    aptr(n+1) = aptr(n) + 1

    ! Fill arow, aval and aptr
    do j = 1, n
       do kk = ptr(j), ptr(j+1) - 1
          i = row(kk)
          atemp = val(kk)
          ipos = aptr(i)
          arow(ipos) = j
          aval(ipos) = atemp
          aptr(i) = ipos - 1
          if (j .eq. i) cycle
          jpos = aptr(j)
          arow(jpos) = i
          aval(jpos) = atemp
          aptr(j) = jpos - 1
       end do
    end do
    do j = 1,n
       aptr(j) = aptr(j) + 1
    end do
  end subroutine expand_matrix

!****************************************************************************
!
! This routine requires the LOWER triangular part of A
! to be held in CSC format.
! The user has supplied a pivot order and this routine checks it is OK
! and returns an error if not. Also sets perm, invp.
!
  subroutine check_order(n, order, invp, options, inform)
    implicit none
    integer, intent(in) :: n ! order of system
    integer, intent(inout) :: order(:)
      ! If i is used to index a variable, |order(i)| must
      ! hold its position in the pivot sequence. If 1x1 pivot i required,
      ! the user must set order(i)>0. If a 2x2 pivot involving variables
      ! i and j is required, the user must set
      ! order(i)<0, order(j)<0 and |order(j)| = |order(i)|+1.
      ! If i is not used to index a variable, order(i) must be set to zero.
      ! !!!! In this version, signs are reset to positive value
    integer, intent(out) :: invp(n)
      ! Used to check order and then holds inverse of perm.
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    character(50)  :: context ! Procedure name (used when printing).

    integer :: i, j
    integer :: nout  ! stream for error messages

    context = 'ssids_analyse'
    nout = options%unit_error
    if (options%print_level .lt. 0) nout = -1

    if (size(order) .lt. n) then
       ! Order is too short
       inform%flag = SSIDS_ERROR_ORDER
       return
    end if

    ! initialise
    invp(:) = 0

    do i = 1, n
       order(i) = abs(order(i))
    end do
     
    ! Check user-supplied order and copy the absolute values to invp.
    ! Also add up number of variables that are not used (null rows)
    do i = 1, n
       j = order(i)
       if ((j .le. 0) .or. (j .gt. n)) exit ! Out of range entry
       if (invp(j) .ne. 0) exit ! Duplicate found
       invp(j) = i
    end do
    if ((i-1) .ne. n) then
       inform%flag = SSIDS_ERROR_ORDER
       return
    end if
  end subroutine check_order

!****************************************************************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Compute flops for processing a node
  !> @param akeep Information generated in analysis phase by SSIDS  
  !> @param node Node
  function compute_flops(nnodes, sptr, rptr, node)
    implicit none

    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer(long), dimension(nnodes+1), intent(in) :: rptr
    integer, intent(in) :: node ! node index
    integer(long) :: compute_flops ! return value

    integer :: n, m ! node sizes
    integer(long) :: jj

    compute_flops = 0

    m = int(rptr(node+1)-rptr(node))
    n = sptr(node+1)-sptr(node)
    do jj = m-n+1, m
       compute_flops = compute_flops + jj**2
    end do

  end function compute_flops

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Partition an elimination tree for execution on different NUMA regions
!>        and GPUs.
!>
!> Start with a single tree, and proceed top down splitting the largest subtree
!> (in terms of total flops)  until we have a sufficient number of independent
!> subtrees. A sufficient number is such that subtrees can be assigned to NUMA
!> regions and GPUs with a load balance no worse than max_load_inbalance.
!> Load balance is calculated as the maximum value over all regions/GPUs of:
!> \f[ \frac{ n x_i / \alpha_i } { \sum_j (x_j/\alpha_j) } \f]
!> Where \f$ \alpha_i \f$ is the performance coefficient of region/GPU i,
!> \f$ x_i \f$ is the number of flops assigned to region/GPU i and \f$ n \f$ is
!> the total number of regions. \f$ \alpha_i \f$ should be proportional to the
!> speed of the region/GPU (i.e. if GPU is twice as fast as CPU, set alpha for
!> CPU to 1.0 and alpha for GPU to 2.0).
!>
!> If the original number of flops is greater than min_gpu_work and the
!> performance coefficient of a GPU is greater than the combined coefficients
!> of the CPU, then subtrees will not be split to become smaller than
!> min_gpu_work until all GPUs are filled.
!>
!> If the balance criterion cannot be satisfied after we have split into
!> 2 * (total regions/GPUs), we just use the best obtained value.
!>
!> GPUs may only handle leaf subtrees, so the top nodes are assigned to the
!> full set of CPUs.
!>
!> Parts are returned as contigous ranges of nodes. Part i consists of nodes
!> part(i):part(i+1)-1
!>
!> @param nnodes Total number of nodes
!> @param sptr Supernode pointers. Supernode i consists of nodes
!>        sptr(i):sptr(i+1)-1.
!> @param sparent Supernode parent array. Supernode i has parent sparent(i).
!> @param rptr Row pointers. Supernode i has rows rlist(rptr(i):rptr(i+1)-1).
!> @param topology Machine topology to partition for.
!> @param min_gpu_work Minimum flops for a GPU execution to be worthwhile.
!> @param max_load_inbalance Number greater than 1.0 representing maximum
!>        permissible load inbalance.
!> @param gpu_perf_coeff The value of \f$ \alpha_i \f$ used for all GPUs,
!>        assuming that used for all NUMA region CPUs is 1.0.
!> @param nparts Number of parts found.
!> @param parts List of part ranges. Part i consists of supernodes
!>        part(i):part(i+1)-1.
!> @param exec_loc Execution location. Part i should be run on partition
!>        mod((exec_loc(i) - 1), size(topology)) + 1.
!>        It should be run on the CPUs if
!>        exec_loc(i) <= size(topology),
!>        otherwise it should be run on GPU number
!>        (exec_loc(i) - 1)/size(topology).
!> @param contrib_ptr Contribution pointer. Part i has contribution from
!>        subtrees contrib_idx(contrib_ptr(i):contrib_ptr(i+1)-1).
!> @param contrib_idx List of contributing subtrees, see contrib_ptr.
!> @param contrib_dest Node to which each subtree listed in contrib_idx(:)
!>        contributes.
!> @param st Allocation status parameter. If non-zero an allocation error
!>        occurred.
  subroutine find_subtree_partition(nnodes, sptr, sparent, rptr, options, &
       topology, nparts, part, exec_loc, contrib_ptr, contrib_idx, &
       contrib_dest, inform, st)
    implicit none
    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer, dimension(nnodes), intent(in) :: sparent
    integer(long), dimension(nnodes+1), intent(in) :: rptr
    type(ssids_options), intent(in) :: options
    type(numa_region), dimension(:), intent(in) :: topology
    integer, intent(out) :: nparts
    integer, dimension(:), allocatable, intent(inout) :: part
    integer, dimension(:), allocatable, intent(out) :: exec_loc
    integer, dimension(:), allocatable, intent(inout) :: contrib_ptr
    integer, dimension(:), allocatable, intent(inout) :: contrib_idx
    integer, dimension(:), allocatable, intent(out) :: contrib_dest
    type(ssids_inform), intent(inout) :: inform
    integer, intent(out) :: st

    integer :: i, j, k
    integer(long) :: jj
    integer :: m, n, node
    integer(long), dimension(:), allocatable :: flops
    integer, dimension(:), allocatable :: size_order
    logical, dimension(:), allocatable :: is_child
    real :: load_balance, best_load_balance
    integer :: nregion, ngpu
    logical :: has_parent

    ! Count flops below each node
    allocate(flops(nnodes+1), stat=st)
    if (st .ne. 0) return
    flops(:) = 0
    do node = 1, nnodes
       flops(node) = flops(node) + compute_flops(nnodes, sptr, rptr, node)
       j = sparent(node)
       flops(j) = flops(j) + flops(node)
       ! !print *, "Node ", node, "parent", j, " flops ", flops(node)
    end do
    !print *, "Total flops ", flops(nnodes+1)

    ! Initialize partition to be all children of virtual root
    allocate(part(nnodes+1), size_order(nnodes), exec_loc(nnodes), &
         is_child(nnodes), stat=st)
    if (st .ne. 0) return
    nparts = 0
    part(1) = 1
    do i = 1, nnodes
       if (sparent(i) .gt. nnodes) then
          nparts = nparts + 1
          part(nparts+1) = i+1
          is_child(nparts) = .true. ! All subtrees are intially child subtrees
       end if
    end do
    call create_size_order(nparts, part, flops, size_order)
    !print *, "Initial partition has ", nparts, " parts"
    !print *, "part = ", part(1:nparts+1)
    !print *, "size_order = ", size_order(1:nparts)

    ! Calculate number of regions/gpus
    nregion = size(topology)
    ngpu = 0
    do i = 1, size(topology)
       ngpu = ngpu + size(topology(i)%gpus)
    end do
    ! print *, "running on ", nregion, " regions and ", ngpu, " gpus"

    ! Keep splitting until we meet balance criterion
    best_load_balance = huge(best_load_balance)
    do i = 1, 2*(nregion+ngpu)
       ! Check load balance criterion
       load_balance = calc_exec_alloc(nparts, part, size_order, is_child,  &
            flops, topology, options%min_gpu_work, options%gpu_perf_coeff, &
            exec_loc, st)
       if (st .ne. 0) return
       best_load_balance = min(load_balance, best_load_balance)
       if (load_balance .lt. options%max_load_inbalance) exit ! allocation is good
       ! Split tree further
       call split_tree(nparts, part, size_order, is_child, sparent, flops, &
            ngpu, options%min_gpu_work, st)
       if (st .ne. 0) return
    end do

    if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
       write (options%unit_diagnostics,*) &
            "[find_subtree_partition] load_balance = ", best_load_balance
    end if
    ! Consolidate adjacent non-children nodes into same part and regen exec_alloc
    !print *
    !print *, "pre merge", part(1:nparts+1)
    !print *, "exec_loc ", exec_loc(1:nparts)
    j = 1
    do i = 2, nparts
       part(j+1) = part(i)
       if (is_child(i) .or. is_child(j)) then
          ! We can't merge j and i
          j = j + 1
          is_child(j) = is_child(i)
       end if
    end do
    part(j+1) = part(nparts+1)
    nparts = j
    !print *, "post merge", part(1:nparts+1)
    call create_size_order(nparts, part, flops, size_order)
    load_balance = calc_exec_alloc(nparts, part, size_order, is_child,  &
         flops, topology, options%min_gpu_work, options%gpu_perf_coeff, &
         exec_loc, st)
    if (st .ne. 0) return
    !print *, "exec_loc ", exec_loc(1:nparts)

    ! Merge adjacent subtrees that are executing on the same node so long as
    ! there is no more than one contribution to a parent subtree
    j = 1
    k = sparent(part(j+1)-1)
    has_parent = (k .le. nnodes)
    do i = 2, nparts
       part(j+1) = part(i)
       exec_loc(j+1) = exec_loc(i)
       k = sparent(part(i+1)-1)
       if ((exec_loc(i) .ne. exec_loc(j)) .or. (has_parent .and. (k .le. nnodes))) then
          ! We can't merge j and i
          j = j + 1
          has_parent = .false. 
       end if
       has_parent = has_parent.or.(k.le.nnodes)
    end do
    part(j+1) = part(nparts+1)
    nparts = j

    ! Figure out contribution blocks that are input to each part
    allocate(contrib_ptr(nparts+3), contrib_idx(nparts), contrib_dest(nparts), &
         stat=st)
    if (st .ne. 0) return
    ! Count contributions at offset +2
    contrib_ptr(3:nparts+3) = 0
    do i = 1, nparts-1 ! by defn, last part has no parent
       j = sparent(part(i+1)-1) ! node index of parent
       if (j .gt. nnodes) cycle ! part is a root
       k = i+1 ! part index of j
       do while(j .ge. part(k+1))
          k = k + 1
       end do
       contrib_ptr(k+2) = contrib_ptr(k+2) + 1
    end do
    ! Figure out contrib_ptr starts at offset +1
    contrib_ptr(1:2) = 1
    do i = 1, nparts
       contrib_ptr(i+2) = contrib_ptr(i+1) + contrib_ptr(i+2)
    end do
    ! Drop sources into list
    do i = 1, nparts-1 ! by defn, last part has no parent
       j = sparent(part(i+1)-1) ! node index of parent
       if (j .gt. nnodes) then
          ! part is a root
          contrib_idx(i) = nparts+1
          cycle
       end if
       k = i+1 ! part index of j
       do while (j .ge. part(k+1))
          k = k + 1
       end do
       contrib_idx(i) = contrib_ptr(k+1)
       contrib_dest(contrib_idx(i)) = j
       contrib_ptr(k+1) = contrib_ptr(k+1) + 1
    end do
    contrib_idx(nparts) = nparts+1 ! last part must be a root

    ! Fill out inform
    inform%nparts = nparts
    inform%gpu_flops = 0
    do i = 1, nparts
       if (exec_loc(i) .gt. size(topology)) &
            inform%gpu_flops = inform%gpu_flops + flops(part(i+1)-1)
    end do
    inform%cpu_flops = flops(nnodes+1) - inform%gpu_flops
  end subroutine find_subtree_partition

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Allocate execution of subtrees to resources and calculate load balance
!>
!> Given the partition supplied, uses a greedy algorithm to assign subtrees to
!> resources specified by topology and then returns the resulting load balance
!> as
!> \f[ \frac{\max_i( n x_i / \alpha_i )} { \sum_j (x_j/\alpha_j) } \f]
!> Where \f$ \alpha_i \f$ is the performance coefficient of region/GPU i,
!> \f$ x_i \f$ is the number of flops assigned to region/GPU i and \f$ n \f$ is
!> the total number of regions. \f$ \alpha_i \f$ should be proportional to the
!> speed of the region/GPU (i.e. if GPU is twice as fast as CPU, set alpha for
!> CPU to 1.0 and alpha for GPU to 2.0).
!>
!> Work is only assigned to GPUs if the subtree has at least min_gpu_work flops.
!>
!> None-child subtrees are ignored (they will be executed using all available
!> resources). They are recorded with exec_loc -1.
!>
!> @param nparts Number of parts.
!> @param parts List of part ranges. Part i consists of supernodes
!>        part(i):part(i+1)-1.
!> @param size_order Lists parts in decreasing order of flops.
!>        i.e. size_order(1) is the largest part.
!> @param is_child True if subtree is a child subtree (has no contributions
!>        from other subtrees).
!> @param flops Number of floating points in subtree rooted at each node.
!> @param topology Machine topology to allocate execution for.
!> @param min_gpu_work Minimum work before allocation to GPU is useful.
!> @param gpu_perf_coeff The value of \f$ \alpha_i \f$ used for all GPUs,
!>        assuming that used for all NUMA region CPUs is 1.0.
!> @param exec_loc Execution location. Part i should be run on partition
!>        mod((exec_loc(i) - 1), size(topology)) + 1.
!>        It should be run on the CPUs if
!>        exec_loc(i) <= size(topology),
!>        otherwise it should be run on GPU number
!>        (exec_loc(i) - 1)/size(topology).
!> @param st Allocation status parameter. If non-zero an allocation error
!>        occurred.
!> @returns Load balance value as detailed in subroutine description.
!> @sa find_subtree_partition()
! FIXME: Consider case when gpu_perf_coeff > 2.0 ???
!        (Round robin may not be correct thing)
  real function calc_exec_alloc(nparts, part, size_order, is_child, flops, &
       topology, min_gpu_work, gpu_perf_coeff, exec_loc, st)
    implicit none
    integer, intent(in) :: nparts
    integer, dimension(nparts+1), intent(in) :: part
    integer, dimension(nparts), intent(in) :: size_order
    logical, dimension(nparts), intent(in) :: is_child
    integer(long), dimension(*), intent(in) :: flops
    type(numa_region), dimension(:), intent(in) :: topology
    integer(long), intent(in) :: min_gpu_work
    real, intent(in) :: gpu_perf_coeff
    integer, dimension(nparts), intent(out) :: exec_loc
    integer, intent(out) :: st

    integer :: i, p, nregion, ngpu, max_gpu, next
    integer(long) :: pflops
    integer, dimension(:), allocatable :: map ! List resources in order of
      ! decreasing power
    real, dimension(:), allocatable :: load_balance
    real :: total_balance

    ! Initialise in case of an error return
    calc_exec_alloc = huge(calc_exec_alloc)

    !
    ! Create resource map
    !
    nregion = size(topology)
    ngpu = 0
    max_gpu = 0
    do i = 1, size(topology)
       ngpu = ngpu + size(topology(i)%gpus)
       max_gpu = max(max_gpu, size(topology(i)%gpus))
    end do
    allocate(map(nregion+ngpu), stat=st)
    if (st .ne. 0) return

    if (gpu_perf_coeff .gt. 1.0) then
       ! GPUs are more powerful than CPUs
       next = 1
       do i = 1, size(topology)
          do p = 1, size(topology(i)%gpus)
             map(next) = p*nregion + i
             next = next + 1
          end do
       end do
       do i = 1, size(topology)
          map(next) = i
          next = next + 1
       end do
    else
       ! CPUs are more powerful than GPUs
       next = 1
       do i = 1, size(topology)
          map(next) = i
          next = next + 1
       end do
       do i = 1, size(topology)
          do p = 1, size(topology(i)%gpus)
             map(next) = p*nregion + i
             next = next + 1
          end do
       end do
    end if

    !
    ! Simple round robin allocation in decreasing size order.
    !
    next = 1
    do i = 1, nparts
       p = size_order(i)
       if (.not. is_child(p)) then
          ! Not a child subtree
          exec_loc(p) = -1
          cycle
       end if
       pflops = flops(part(p+1)-1)
       if (pflops .lt. min_gpu_work) then
          ! Avoid GPUs
          do while (map(next) .gt. nregion)
             next = next + 1
             if (next .gt. size(map)) next = 1
          end do
       end if
       exec_loc(p) = map(next)
       next = next + 1
       if (next .gt. size(map)) next = 1
    end do

    !
    ! Calculate load inbalance
    !
    allocate(load_balance(nregion*(1+max_gpu)), stat=st)
    if (st .ne. 0) return
    load_balance(:) = 0.0
    total_balance = 0.0
    ! Sum total 
    do p = 1, nparts
       if (exec_loc(p) .eq. -1) cycle ! not a child subtree
       pflops = flops(part(p+1)-1)
       if (exec_loc(p) .gt. nregion) then
          ! GPU
          load_balance(exec_loc(p)) = load_balance(exec_loc(p)) + &
               real(pflops) / gpu_perf_coeff
          total_balance = total_balance + real(pflops) / gpu_perf_coeff
       else
          ! CPU
          load_balance(exec_loc(p)) = load_balance(exec_loc(p)) + real(pflops)
          total_balance = total_balance + real(pflops)
       end if
    end do
    ! Calculate n * max(x_i/a_i) / sum(x_j/a_j)
    calc_exec_alloc = (nregion+ngpu) * maxval(load_balance(:)) / total_balance
  end function calc_exec_alloc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Split tree into an additional part as required by
!>        find_subtree_partition().
!>
!> Split largest partition into two parts, unless doing so would reduce the
!> number of subtrees with at least min_gpu_work below ngpu.
!>
!> Note: We require all input parts to have a single root.
!>
!> @param nparts Number of parts: normally increased by one on return.
!> @param part Part i consists of nodes part(i):part(i+1).
!> @param size_order Lists parts in decreasing order of flops.
!>        i.e. size_order(1) is the largest part.
!> @param is_child True if subtree is a child subtree (has no contributions
!>        from other subtrees).
!> @param sparent Supernode parent array. Supernode i has parent sparent(i).
!> @param flops Number of floating points in subtree rooted at each node.
!> @param ngpu Number of gpus.
!> @param min_gpu_work Minimum worthwhile work to give to GPU.
!> @param st Allocation status parameter. If non-zero an allocation error
!>        occurred.
!> @sa find_subtree_partition()
  subroutine split_tree(nparts, part, size_order, is_child, sparent, flops, &
       ngpu, min_gpu_work, st)
    implicit none
    integer, intent(inout) :: nparts
    integer, dimension(*), intent(inout) :: part
    integer, dimension(*), intent(inout) :: size_order
    logical, dimension(*), intent(inout) :: is_child
    integer, dimension(*), intent(in) :: sparent
    integer(long), dimension(*), intent(in) :: flops
    integer, intent(in) :: ngpu
    integer(long), intent(in) :: min_gpu_work
    integer, intent(out) :: st

    integer :: i, p, nchild, nbig, root, to_split, old_nparts
    integer, dimension(:), allocatable :: children, temp

    ! Look for all children of root in biggest child part
    nchild = 0
    allocate(children(10), stat=st) ! we will resize if necessary
    if (st.ne.0) return
    ! Find biggest child subtree
    to_split = 1
    do while(.not. is_child(size_order(to_split)))
       to_split = to_split + 1
    end do
    to_split = size_order(to_split)
    ! Find all children of root
    root = part(to_split+1)-1
    do i = part(to_split), root-1
       if (sparent(i) .eq. root) then
          nchild = nchild+1
          if (nchild .gt. size(children)) then
             ! Increase size of children(:)
             allocate(temp(2*size(children)), stat=st)
             if (st .ne. 0) return
             temp(1:size(children)) = children(:)
             deallocate(children)
             call move_alloc(temp, children)
          end if
          children(nchild) = i
       end if
    end do

    ! Check we can split safely
    if (nchild .eq. 0) return ! singleton node, can't split
    nbig = 0 ! number of new parts > min_gpu_work
    do i = to_split+1, nparts
       p = size_order(i)
       if (.not. is_child(p)) cycle ! non-children can't go on GPUs
       root = part(p+1)-1
       if (flops(root) .lt. min_gpu_work) exit
       nbig = nbig + 1
    end do
    if ((nbig+1) .ge. ngpu) then
       ! Original partition met min_gpu_work criterion
       do i = 1, nchild
          if (flops(children(i)) .ge. min_gpu_work) nbig = nbig + 1
       end do
       if (nbig .lt. ngpu) return ! new partition fails min_gpu_work criterion
    end if

    ! Can safely split, so do so. As part to_split was contigous, when
    ! split the new parts fall into the same region. Thus, we first push any
    ! later regions back to make room, then add the new parts.
    part(to_split+nchild+1:nparts+nchild+1) = part(to_split+1:nparts+1)
    is_child(to_split+nchild+1:nparts+nchild) = is_child(to_split+1:nparts)
    do i = 1, nchild
       ! New part corresponding to child i *ends* at part(to_split+i)-1
       part(to_split+i) = children(i)+1
    end do
    is_child(to_split:to_split+nchild-1) = .true.
    is_child(to_split+nchild) = .false. ! Newly created non-parent subtree
    old_nparts = nparts
    nparts = old_nparts + nchild

    ! Finally, recreate size_order array
    call create_size_order(nparts, part, flops, size_order)
  end subroutine split_tree

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Determine order of subtrees based on size
!>
!> @note Sorting algorithm could be improved if this becomes a bottleneck.
!>
!> @param nparts Number of parts: normally increased by one on return.
!> @param part Part i consists of nodes part(i):part(i+1).
!> @param flops Number of floating points in subtree rooted at each node.
!> @param size_order Lists parts in decreasing order of flops.
!>        i.e. size_order(1) is the largest part.
  subroutine create_size_order(nparts, part, flops, size_order)
    implicit none
    integer, intent(in) :: nparts
    integer, dimension(nparts+1), intent(in) :: part
    integer(long), dimension(*), intent(in) :: flops
    integer, dimension(nparts), intent(out) :: size_order

    integer :: i, j
    integer(long) :: iflops

    do i = 1, nparts
       ! We assume parts 1:i-1 are in order and aim to insert part i
       iflops = flops(part(i+1)-1)
       do j = 1, i-1
          if (iflops .gt. flops(part(j+1)-1)) exit ! node i belongs in posn j
       end do
       size_order(j+1:i) = size_order(j:i-1)
       size_order(j) = i
    end do
  end subroutine create_size_order

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Prints assembly tree
  subroutine print_atree(nnodes, sptr, sparent, rptr)
    implicit none
    
    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer, dimension(nnodes), intent(in) :: sparent
    integer(long), dimension(nnodes+1), intent(in) :: rptr

    integer :: node
    integer :: n, m ! node sizes
    integer(long), dimension(:), allocatable :: flops
    integer :: j
    real :: tot_weight, weight

    ! Count flops below each node
    allocate(flops(nnodes+1))
    flops(:) = 0
    do node = 1, nnodes
       flops(node) = flops(node) + compute_flops(nnodes, sptr, rptr, node)
       j = sparent(node)
       if(j .gt. 0) flops(j) = flops(j) + flops(node)
       !print *, "Node ", node, "parent", j, " flops ", flops(node)
    end do
    tot_weight = real(flops(nnodes))

    open(2, file="atree.dot")

    write(2, '("graph atree {")')
    write(2, '("node [")')
    write(2, '("style=filled")')
    write(2, '("]")')

    do node = 1, nnodes

       weight = real(flops(node)) / tot_weight 

       if (weight .lt. 0.01) cycle ! Prune smallest nodes

       n = sptr(node+1) - sptr(node) 
       m = int(rptr(node+1) - rptr(node))
       
       ! node idx
       write(2, '(i10)', advance="no") node
       write(2, '(" ")', advance="no")
       write(2, '("[")', advance="no")

       ! Node label 
       write(2, '("label=""")', advance="no")
       write(2, '("node:", i5,"\n")', advance="no")node
       write(2, '("m:", i5,"\n")', advance="no")m
       write(2, '("n:", i5,"\n")', advance="no")n
       write(2, '("w:", f6.2,"\n")', advance="no")100*weight
       write(2, '("""")', advance="no")

       ! Node color
       write(2, '(" fillcolor=white")', advance="no")

       write(2, '("]")', advance="no")
       write(2, '(" ")')

       ! parent node
       if(sparent(node) .ne. -1) write(2, '(i10, "--", i10)')sparent(node), node

    end do

    write(2, '("}")')

    close(2)

  end subroutine print_atree

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !> @brief Prints assembly tree with partitions
  subroutine print_atree_part(nnodes, sptr, sparent, rptr, topology, nparts, & 
       part, exec_loc)
    implicit none

    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer, dimension(nnodes), intent(in) :: sparent
    integer(long), dimension(nnodes+1), intent(in) :: rptr
    type(numa_region), dimension(:), intent(in) :: topology
    integer, intent(in) :: nparts
    integer, dimension(:), allocatable, intent(in) :: part
    integer, dimension(:), allocatable, intent(in) :: exec_loc

    integer :: node
    integer :: n, m ! Node dimensions
    integer :: region ! Where to execute node
    integer(long), dimension(:), allocatable :: flops
    real :: tot_weight, weight
    integer :: i, j
    character(len=5) :: part_str 
    real :: small

    small = 0.001

    ! Count flops below each node
    allocate(flops(nnodes+1))
    flops(:) = 0
    do node = 1, nnodes
       flops(node) = flops(node) + compute_flops(nnodes, sptr, rptr, node)
       j = sparent(node)
       if(j .gt. 0) flops(j) = flops(j) + flops(node)
       !print *, "Node ", node, "parent", j, " flops ", flops(node)
    end do
    tot_weight = real(flops(nnodes))

    open(2, file="atree_part.dot")

    write(2, '("graph atree {")')
    write(2, '("node [")')
    write(2, '("style=filled")')
    write(2, '("]")')

    do i = 1, nparts

       region = mod((exec_loc(i)-1), size(topology))+1
       ! print *, "part = ", i, ", exec_loc = ", exec_loc(i), ", region = ", region 
       
       write(part_str, '(i5)')part(i)
       write(2, *)"subgraph cluster"// adjustl(trim(part_str)) // " {"
       if ( exec_loc(i) .gt. size(topology)) then ! GPU subtree
          write(2, *)"color=red"
       else
          write(2, *)"color=black"
       end if
       write(2, '("label=""")', advance="no")
       write(2, '("part:", i5,"\n")', advance="no")i
       write(2, '("region:", i5,"\n")', advance="no")region
       write(2, '("exec_loc:", i5,"\n")', advance="no")exec_loc(i)
       write(2, '("""")', advance="no")

       do node = part(i), part(i+1)-1

          weight = real(flops(node)) / tot_weight 
          if (weight .lt. small) cycle ! Prune smallest nodes

          n = sptr(node+1) - sptr(node) 
          m = int(rptr(node+1) - rptr(node))

          ! node idx
          write(2, '(i10)', advance="no") node
          write(2, '(" ")', advance="no")
          write(2, '("[")', advance="no")

          ! Node label 
          write(2, '("label=""")', advance="no")
          write(2, '("node:", i5,"\n")', advance="no")node
          write(2, '("m:", i5,"\n")', advance="no")m
          write(2, '("n:", i5,"\n")', advance="no")n
          write(2, '("w:", f6.2,"\n")', advance="no")100*weight
          write(2, '("""")', advance="no")

          ! Node color
          write(2, '(" fillcolor=white")', advance="no")

          write(2, '("]")', advance="no")
          write(2, '(" ")')

       end do

       write(2, '("}")') ! Subgraph

       do node = part(i), part(i+1)-1
          weight = real(flops(node)) / tot_weight
          if (weight .lt. small) cycle ! Prune smallest nodes
          if(sparent(node) .ne. -1) write(2, '(i10, "--", i10)')sparent(node), node
       end do
          
    end do

    write(2, '("}")') ! Graph

    close(2)

  end subroutine print_atree_part

!****************************************************************************

!
! This routine requires the LOWER and UPPER triangular parts of A
! to be held in CSC format using ptr2 and row2
! AND lower triangular part held using ptr and row.
!
! On exit from this routine, order is set to order
! input to factorization.
!
  subroutine analyse_phase(n, ptr, row, ptr2, row2, order, invp, &
       akeep, options, inform)
    implicit none
    integer, intent(in) :: n ! order of system
    integer(long), intent(in) :: ptr(n+1) ! col pointers (lower triangle) 
    integer, intent(in) :: row(ptr(n+1)-1) ! row indices (lower triangle)
    integer(long), intent(in) :: ptr2(n+1) ! col pointers (whole matrix)
    integer, intent(in) :: row2(ptr2(n+1)-1) ! row indices (whole matrix)
    integer, dimension(n), intent(inout) :: order
      !  On exit, holds the pivot order to be used by factorization.
    integer, dimension(n), intent(out) :: invp 
      ! Work array. Used to hold inverse of order but
      ! is NOT set to inverse for the final order that is returned.
    type(ssids_akeep), intent(inout) :: akeep
    type(ssids_options), intent(in) :: options
    type(ssids_inform), intent(inout) :: inform

    character(50)  :: context ! Procedure name (used when printing).
    integer, dimension(:), allocatable :: contrib_dest, exec_loc, level

    integer :: to_launch
    integer :: numa_region, device, thread_num
    integer :: nemin, flag
    integer :: blkm, blkn
    integer :: i, j
    integer :: nout, nout1 ! streams for errors and warnings
    integer(long) :: nz ! ptr(n+1)-1
    integer :: st

    context = 'ssids_analyse'
    nout = options%unit_error
    if (options%print_level .lt. 0) nout = -1
    nout1 = options%unit_warning
    if (options%print_level .lt. 0) nout1 = -1
    st = 0

    ! Check nemin and set to default if out of range.
    nemin = options%nemin
    if (nemin .lt. 1) nemin = nemin_default

    ! Perform basic analysis so we can figure out subtrees we want to construct
    call basic_analyse(n, ptr2, row2, order, akeep%nnodes, akeep%sptr, &
         akeep%sparent, akeep%rptr,akeep%rlist,                        &
         nemin, flag, inform%stat, inform%num_factor, inform%num_flops)
    select case(flag)
    case(0)
       ! Do nothing
    case(-1)
       ! Allocation error
       inform%flag = SSIDS_ERROR_ALLOCATION
       return
    case(1)
       ! Zero row/column.
       inform%flag = SSIDS_WARNING_ANAL_SINGULAR
    case default
       ! Should never reach here
       inform%flag = SSIDS_ERROR_UNKNOWN
    end select

    ! set invp to hold inverse of order
    do i = 1,n
       invp(order(i)) = i
    end do
    ! any unused variables are at the end and so can set order for them
    do j = akeep%sptr(akeep%nnodes+1), n
       i = invp(j)
       order(i) = 0
    end do

    ! Build map from A to L in nptr, nlist
    nz = ptr(n+1) - 1
    allocate(akeep%nptr(n+1), akeep%nlist(2,nz), stat=st)
    if (st .ne. 0) go to 100
    call build_map(n, ptr, row, order, invp, akeep%nnodes, akeep%sptr, &
         akeep%rptr, akeep%rlist, akeep%nptr, akeep%nlist, st)
    if (st .ne. 0) go to 100

    ! Sort out subtrees
    if ((options%print_level .ge. 1) .and. (options%unit_diagnostics .ge. 0)) then
       write (options%unit_diagnostics,*) "Input topology"
       do i = 1, size(akeep%topology)
         write (options%unit_diagnostics,*) &
              "Region ", i, " with ", akeep%topology(i)%nproc, " cores"
         if(size(akeep%topology(i)%gpus).gt.0) &
            write (options%unit_diagnostics,*) &
                 "---> gpus ", akeep%topology(i)%gpus
       end do
    end if
    call find_subtree_partition(akeep%nnodes, akeep%sptr, akeep%sparent,           &
         akeep%rptr, options, akeep%topology, akeep%nparts, akeep%part,            &
         exec_loc, akeep%contrib_ptr, akeep%contrib_idx, contrib_dest, inform, st)
    if (st .ne. 0) go to 100
    !print *, "invp = ", akeep%invp
    !print *, "sptr = ", akeep%sptr(1:akeep%nnodes+1)
    !print *, "sparent = ", akeep%sparent
    !print *, "Partition suggests ", akeep%nparts, " parts"
    !print *, "akeep%part = ", akeep%part(1:akeep%nparts+1)
    !print *, "exec_loc   = ", exec_loc(1:akeep%nparts)
    !print *, "parents = ", akeep%sparent(akeep%part(2:akeep%nparts+1)-1)
    !print *, "contrib_ptr = ", akeep%contrib_ptr(1:akeep%nparts+1)
    !print *, "contrib_idx = ", akeep%contrib_idx(1:akeep%nparts)
    !print *, "contrib_dest = ", &
    !   contrib_dest(1:akeep%contrib_ptr(akeep%nparts+1)-1)

    ! Generate dot file for assembly tree
    ! call print_atree(akeep%nnodes, akeep%sptr, akeep%sparent, akeep%rptr)
    call print_atree_part(akeep%nnodes, akeep%sptr, akeep%sparent, akeep%rptr, &
         akeep%topology, akeep%nparts, akeep%part, exec_loc)

    ! Construct symbolic subtrees
    allocate(akeep%subtree(akeep%nparts))

    ! Split into NUMA regions for setup (assume mem is first touch)
    to_launch = size(akeep%topology)
!$omp parallel proc_bind(spread) num_threads(to_launch) default(shared) &
!$omp    private(i, numa_region, device, thread_num)
    thread_num = 0
!$  thread_num = omp_get_thread_num()
    numa_region = thread_num + 1
    do i = 1, akeep%nparts
       ! only initialize subtree if this is the correct region: note that
       ! an "all region" subtree with location -1 is initialised by region 0
       if (exec_loc(i) .eq. -1) then
          if (numa_region .ne. 1) cycle
          device = 0
       else if ((mod((exec_loc(i)-1), size(akeep%topology))+1) .ne. numa_region) then
          cycle
       else
          device = (exec_loc(i)-1) / size(akeep%topology)
       end if
       akeep%subtree(i)%exec_loc = exec_loc(i)
       if (device .eq. 0) then
          ! CPU
          !print *, numa_region, "init cpu subtree ", i, akeep%part(i), &
          !   akeep%part(i+1)-1
          akeep%subtree(i)%ptr => construct_cpu_symbolic_subtree(akeep%n,   &
               akeep%part(i), akeep%part(i+1), akeep%sptr, akeep%sparent,   &
               akeep%rptr, akeep%rlist, akeep%nptr, akeep%nlist,            &
               contrib_dest(akeep%contrib_ptr(i):akeep%contrib_ptr(i+1)-1), &
               options)
       else
          ! GPU
          device = akeep%topology(numa_region)%gpus(device)
          !print *, numa_region, "init gpu subtree ", i, akeep%part(i), &
          !   akeep%part(i+1)-1, "device", device
          akeep%subtree(i)%ptr => construct_gpu_symbolic_subtree(device,        &
               akeep%n, akeep%part(i), akeep%part(i+1), akeep%sptr,             &
               akeep%sparent, akeep%rptr, akeep%rlist, akeep%nptr, akeep%nlist, &
               options)
       end if
    end do
!$omp end parallel

    ! Info
    allocate(level(akeep%nnodes+1), stat=st)
    if (st .ne. 0) go to 100
    level(akeep%nnodes+1) = 0
    inform%maxfront = 0
    inform%maxdepth = 0
    do i = akeep%nnodes, 1, -1
       blkn = akeep%sptr(i+1) - akeep%sptr(i) 
       blkm = int(akeep%rptr(i+1) - akeep%rptr(i))
       level(i) = level(akeep%sparent(i)) + 1
       inform%maxfront = max(inform%maxfront, blkn)
       inform%maxdepth = max(inform%maxdepth, level(i))
    end do
    deallocate(level, stat=st)
    inform%matrix_rank = akeep%sptr(akeep%nnodes+1)-1
    inform%num_sup = akeep%nnodes

    ! Store copy of inform data in akeep
    akeep%inform = inform

    return

100 continue
    inform%stat = st
    if (inform%stat .ne. 0) then
       inform%flag = SSIDS_ERROR_ALLOCATION
    end if
    return
  end subroutine analyse_phase

!****************************************************************************
!
! Build a map from A to nodes
! lcol( nlist(2,i) ) = val( nlist(1,i) )
! nptr defines start of each node in nlist
!
  subroutine build_map(n, ptr, row, perm, invp, nnodes, sptr, rptr, rlist, &
       nptr, nlist, st)
    implicit none
    ! Original matrix A
    integer, intent(in) :: n
    integer(long), dimension(n+1), intent(in) :: ptr
    integer, dimension(ptr(n+1)-1), intent(in) :: row
    ! Permutation and its inverse (some entries of perm may be negative to
    ! act as flags for 2x2 pivots, so need to use abs(perm))
    integer, dimension(n), intent(in) :: perm
    integer, dimension(n), intent(in) :: invp
    ! Supernode partition of L
    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    ! Row indices of L
    integer(long), dimension(nnodes+1), intent(in) :: rptr
    integer, dimension(rptr(nnodes+1)-1), intent(in) :: rlist
    ! Output mapping
    integer(long), dimension(nnodes+1), intent(out) :: nptr
    integer(long), dimension(2, ptr(n+1)-1), intent(out) :: nlist
    ! Error check paramter
    integer, intent(out) :: st

    integer :: i, j, k
    integer(long) :: ii, jj, pp
    integer :: blkm
    integer :: col
    integer :: node
    integer, dimension(:), allocatable :: ptr2, row2
    integer(long), dimension(:), allocatable :: origin
    integer, dimension(:), allocatable :: map

    allocate(map(n), ptr2(n+3), row2(ptr(n+1)-1), origin(ptr(n+1)-1), stat=st)
    if (st .ne. 0) return

    !
    ! Build transpose of A in ptr2, row2. Store original posn of entries in
    ! origin array.
    !
    ! Count number of entries in row i in ptr2(i+2). Don't include diagonals.
    ptr2(:) = 0
    do i = 1, n
       do jj = ptr(i), ptr(i+1)-1
          k = row(jj)
          if (k .eq. i) cycle
          ptr2(k+2) = ptr2(k+2) + 1
       end do
    end do
    ! Work out row starts such that row i starts in posn ptr2(i+1)
    ptr2(1:2) = 1
    do i = 1, n
       ptr2(i+2) = ptr2(i+2) + ptr2(i+1)
    end do
    ! Drop entries into place
    do i = 1, n
       do jj = ptr(i), ptr(i+1)-1
          k = row(jj)
          if (k .eq. i) cycle
          row2(ptr2(k+1)) = i
          origin(ptr2(k+1)) = jj
          ptr2(k+1) = ptr2(k+1) + 1
       end do
    end do

    !
    ! Build nptr, nlist map
    !
    pp = 1
    do node = 1, nnodes
       blkm = int(rptr(node+1) - rptr(node))
       nptr(node) = pp

       ! Build map for node indices
       do jj = rptr(node), rptr(node+1)-1
          map(rlist(jj)) = int(jj-rptr(node)+1)
       end do

       ! Build nlist from A-lower transposed
       do j = sptr(node), sptr(node+1)-1
          col = invp(j)
          do i = ptr2(col), ptr2(col+1)-1
             k = abs(perm(row2(i))) ! row of L
             if (k .lt. j) cycle
             nlist(2,pp) = (j-sptr(node))*blkm + map(k)
             nlist(1,pp) = origin(i)
             pp = pp + 1
          end do
       end do

       ! Build nlist from A-lower
       do j = sptr(node), sptr(node+1)-1
          col = invp(j)
          do ii = ptr(col), ptr(col+1)-1
             k = abs(perm(row(ii))) ! row of L
             if (k .lt. j) cycle
             nlist(2,pp) = (j-sptr(node))*blkm + map(k)
             nlist(1,pp) = ii
             pp = pp + 1
          end do
       end do
    end do
    nptr(nnodes+1) = pp
  end subroutine build_map
end module spral_ssids_anal
