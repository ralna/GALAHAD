module spral_ssids_gpu_subtree
  use, intrinsic :: iso_c_binding
  use spral_cuda
  use spral_ssids_gpu_smalloc, only : smalloc, smfreeall, smalloc_setup
  use spral_ssids_contrib, only : contrib_type
  use spral_ssids_datatypes
  use spral_ssids_inform, only : ssids_inform
  use spral_ssids_subtree, only : symbolic_subtree_base, numeric_subtree_base
  use spral_ssids_gpu_datatypes, only : gpu_type, free_gpu_type
  use spral_ssids_gpu_factor, only : parfactor
  use spral_ssids_gpu_solve, only : bwd_solve_gpu, fwd_solve_gpu, d_solve_gpu
  implicit none

  private
  public :: gpu_symbolic_subtree, construct_gpu_symbolic_subtree
  public :: gpu_numeric_subtree, gpu_free_contrib

  type, extends(symbolic_subtree_base) :: gpu_symbolic_subtree
     integer :: device
     integer :: n
     integer :: sa
     integer :: en
     integer :: nnodes
     integer(long) :: nfactor
     integer(long), dimension(:), allocatable :: nptr
     integer, dimension(:), allocatable :: child_ptr
     integer, dimension(:), allocatable :: child_list
     integer, dimension(:), pointer :: sptr
     integer, dimension(:), allocatable :: sparent
     integer(long), dimension(:), allocatable :: rptr
     integer, dimension(:), pointer :: rlist
     integer, dimension(:), allocatable :: rlist_direct
     ! GPU pointers (copies of the CPU arrays of same name)
     type(C_PTR) :: gpu_nlist = C_NULL_PTR
     type(C_PTR) :: gpu_rlist = C_NULL_PTR
     type(C_PTR) :: gpu_rlist_direct = C_NULL_PTR
     integer(long) :: max_a_idx
   contains
     procedure :: factor
     procedure :: cleanup => symbolic_cleanup
  end type gpu_symbolic_subtree

  type, extends(numeric_subtree_base) :: gpu_numeric_subtree
     logical :: posdef
     integer :: device ! Have own copy as symbolic may be freed first
     type(gpu_symbolic_subtree), pointer :: symbolic
     type(C_PTR) :: stream_handle = C_NULL_PTR
     type(gpu_type) :: stream_data
     type(C_PTR) :: gpu_rlist_with_delays = C_NULL_PTR
     type(C_PTR) :: gpu_rlist_direct_with_delays = C_NULL_PTR
     type(C_PTR) :: gpu_clists = C_NULL_PTR
     type(C_PTR) :: gpu_clists_direct = C_NULL_PTR
     type(C_PTR) :: gpu_clen = C_NULL_PTR
     logical :: host_factors = .false.
     type(smalloc_type), pointer :: alloc => null()
     type(node_type), dimension(:), allocatable :: nodes ! Stores pointers
       ! to information about nodes
     type(contrib_type) :: contrib
     type(C_PTR) :: contrib_wait = C_NULL_PTR
   contains
     procedure :: get_contrib
     procedure :: solve_fwd
     procedure :: solve_diag
     procedure :: solve_diag_bwd
     procedure :: solve_bwd
     procedure :: enquire_posdef
     procedure :: enquire_indef
     procedure :: alter
     procedure :: cleanup => numeric_cleanup
  end type gpu_numeric_subtree

contains

  function construct_gpu_symbolic_subtree(device, n, sa, en, sptr, sparent, &
       rptr, rlist, nptr, nlist, options) result(this)
    implicit none
    class(gpu_symbolic_subtree), pointer :: this
    integer, intent(in) :: device
    integer, intent(in) :: n
    integer, intent(in) :: sa
    integer, intent(in) :: en
    integer, dimension(*), target, intent(in) :: sptr
    integer, dimension(*), target, intent(in) :: sparent
    integer(long), dimension(*), target, intent(in) :: rptr
    integer, dimension(*), target, intent(in) :: rlist
    integer(long), dimension(*), target, intent(in) :: nptr
    integer(long), dimension(2,*), target, intent(in) :: nlist
    class(ssids_options), intent(in) :: options

    integer :: node
    integer :: cuda_error, st

    nullify(this)

    ! Specify which device we're using
    cuda_error = cudaSetDevice(device)
    if (cuda_error .ne. 0) then
       print *, "CUDA error on copy"
       return
    end if

   ! Allocate output
   allocate(this, stat=st)
   if (st .ne. 0) return

   ! Initialise members
   !FIXME: stat
   this%device = device
   this%n = n
   this%sa = sa
   this%en = en-1
   this%nnodes = en-sa
   allocate(this%nptr(this%nnodes+1))
   this%nptr(1:this%nnodes+1) = nptr(sa:en) - nptr(sa) + 1
   this%sptr => sptr(sa:en)
   allocate(this%sparent(this%nnodes))
   this%sparent = sparent(sa:en-1) - sa + 1
   allocate(this%rptr(this%nnodes+1))
   this%rptr(1:this%nnodes+1) = rptr(sa:en) - rptr(sa) + 1
   this%rlist => rlist(rptr(sa):rptr(en)-1)

   ! Calculate derived quantities
   this%max_a_idx = maxval(nlist(1,nptr(sa):nptr(en)-1)) ! needed for aval copy
   this%nfactor = 0
   do node = 1, this%nnodes
      this%nfactor = this%nfactor + &
           (this%sptr(node+1)-this%sptr(node)) * &
           (this%rptr(node+1)-this%rptr(node))
   end do

   ! Build rlist direct
   allocate(this%rlist_direct(this%rptr(this%nnodes+1)-1))
   call build_rlist_direct(n, this%nnodes, this%sparent, this%rptr, &
        this%rlist, this%rlist_direct, st) ! FIXME: st

   ! Build child pointers
   call build_child_pointers(this%nnodes, this%sparent, this%child_ptr, &
        this%child_list, st)

   ! Copy data to device
   cuda_error=0
   call copy_analyse_data_to_device(2*(this%nptr(this%nnodes+1)-1), &
        nlist(:,nptr(sa):nptr(en)-1), rptr(en)-rptr(sa), &
        rlist(rptr(sa):rptr(en)-1), this%rlist_direct, &
        this%gpu_nlist, this%gpu_rlist, this%gpu_rlist_direct, cuda_error)
   if (cuda_error .ne. 0) then
      print *, "CUDA error on copy"
      ! FIXME: error handling
      !select type(inform)
      !type is (ssids_inform_gpu)
      !   inform%cuda_error = cuda_error
      !end select
      !inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
      deallocate(this)
      nullify(this)
      return
   end if
 end function construct_gpu_symbolic_subtree

 subroutine build_child_pointers(nnodes, sparent, child_ptr, child_list, st)
   implicit none
   integer, intent(in) :: nnodes
   integer, dimension(nnodes), intent(in) :: sparent
   integer, dimension(:), allocatable :: child_ptr
   integer, dimension(:), allocatable :: child_list
   integer, intent(out) :: st

   integer :: i, j
   integer, dimension(:), allocatable :: child_next, child_head ! linked
     ! list for children

   ! Setup child_ptr, child_next and calculate work per subtree
   allocate(child_next(nnodes+1), child_head(nnodes+1), child_ptr(nnodes+2), &
        child_list(nnodes), stat=st)
   if (st .ne. 0) return
   child_head(:) = -1
   do i = nnodes, 1, -1 ! backwards so child list is in order
      ! Add to parent's child linked list
      j = min(sparent(i), nnodes+1)
      child_next(i) = child_head(j)
      child_head(j) = i
   end do
   ! Add work up tree, build child_ptr and child_list
   child_ptr(1) = 1
   do i = 1, nnodes+1
      j = child_head(i)
      child_ptr(i+1) = child_ptr(i)
      do while (j .ne. -1)
         child_list(child_ptr(i+1)) = j
         child_ptr(i+1) = child_ptr(i+1) + 1
         j = child_next(j)
      end do
   end do
 end subroutine build_child_pointers

!****************************************************************************
!
! Build a direct mapping (assuming no delays) between a node's rlist and that
! of it's parent such that update can be done as:
! lcol_parent(rlist_direct(i)) = lcol(i)
!
 subroutine build_rlist_direct(n, nnodes, sparent, rptr, rlist, rlist_direct, st)
   implicit none
   integer, intent(in) :: n
   integer, intent(in) :: nnodes
   integer, dimension(nnodes), intent(in) :: sparent
   integer(long), dimension(nnodes+1), intent(in) :: rptr
   integer, dimension(rptr(nnodes+1)-1), intent(in) :: rlist
   integer, dimension(rptr(nnodes+1)-1), intent(out) :: rlist_direct
   integer, intent(out) :: st

   integer :: node, parent
   integer(long) :: ii
   integer, dimension(:), allocatable :: map

   allocate(map(n), stat=st)
   if (st .ne. 0) return

   do node = 1, nnodes
      ! Build a map for parent
      parent = sparent(node)
      if (parent .gt. nnodes) cycle ! root of tree
      do ii = rptr(parent), rptr(parent+1)-1
         map(rlist(ii)) = int(ii-rptr(parent)+1)
      end do

      ! Build rlist_direct
      do ii = rptr(node), rptr(node+1)-1
         rlist_direct(ii) = map(rlist(ii))
      end do
   end do
 end subroutine build_rlist_direct

!****************************************************************************
!
! Copy data generated by analyse phase to GPU
! (Specifically, nlist, rlist and rlist_direct)
!
 subroutine copy_analyse_data_to_device(lnlist, nlist, lrlist, rlist, &
      rlist_direct, gpu_nlist, gpu_rlist, gpu_rlist_direct, cuda_error)
   implicit none
   integer(long), intent(in) :: lnlist
   integer(C_LONG), dimension(lnlist), target, intent(in) :: nlist
   integer(long), intent(in) :: lrlist
   integer(C_INT), dimension(lrlist), target, intent(in) :: rlist
   integer(C_INT), dimension(lrlist), target, intent(in) :: rlist_direct
   type(C_PTR), intent(out) :: gpu_nlist
   type(C_PTR), intent(out) :: gpu_rlist
   type(C_PTR), intent(out) :: gpu_rlist_direct
   integer, intent(out) :: cuda_error ! Non-zero on error

   ! Copy nlist
   cuda_error = cudaMalloc(gpu_nlist, aligned_size(lnlist*C_SIZEOF(nlist(1))))
   if (cuda_error .ne. 0) return
   cuda_error = cudaMemcpy_h2d(gpu_nlist, C_LOC(nlist), &
        lnlist*C_SIZEOF(nlist(1)))
   if (cuda_error .ne. 0) return

   ! Copy rlist
   cuda_error = cudaMalloc(gpu_rlist, aligned_size(lrlist*C_SIZEOF(rlist(1))))
   if (cuda_error .ne. 0) return
   cuda_error = cudaMemcpy_h2d(gpu_rlist, C_LOC(rlist), &
        lrlist*C_SIZEOF(rlist(1)))
   if (cuda_error .ne. 0) return

   ! Copy rlist_direct
   cuda_error = cudaMalloc(gpu_rlist_direct, aligned_size(lrlist*C_SIZEOF(rlist_direct(1))))
   if (cuda_error .ne. 0) return
   cuda_error = cudaMemcpy_h2d(gpu_rlist_direct, C_LOC(rlist_direct), &
        lrlist*C_SIZEOF(rlist_direct(1)))
   if (cuda_error .ne. 0) return

   ! Synchronise the device, see:
   ! http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
   cuda_error = cudaDeviceSynchronize()
 end subroutine copy_analyse_data_to_device
 
 subroutine symbolic_cleanup(this)
   implicit none
   class(gpu_symbolic_subtree), intent(inout) :: this

   integer :: flag
   
   ! Free GPU arrays if needed
   if (C_ASSOCIATED(this%gpu_nlist)) then
      flag = cudaFree(this%gpu_nlist)
      this%gpu_nlist = C_NULL_PTR
      if (flag .ne. 0) &
           print *, "CUDA error freeing this%gpu_nlist flag = ", flag
   end if
   if (C_ASSOCIATED(this%gpu_rlist)) then
      flag = cudaFree(this%gpu_rlist)
      this%gpu_rlist = C_NULL_PTR
      if (flag .ne. 0) &
           print *, "CUDA error freeing this%gpu_rlist flag = ", flag
   end if
   if (C_ASSOCIATED(this%gpu_rlist_direct)) then
      flag = cudaFree(this%gpu_rlist_direct)
      this%gpu_rlist_direct = C_NULL_PTR
      if (flag .ne. 0) &
           print *, "CUDA error freeing this%gpu_rlist_direct flag = ", flag
   end if
 end subroutine symbolic_cleanup

 function factor(this, posdef, aval, child_contrib, options, inform, scaling)
   implicit none
   class(numeric_subtree_base), pointer :: factor
   class(gpu_symbolic_subtree), target, intent(inout) :: this
   logical, intent(in) :: posdef
   real(wp), dimension(*), target, intent(in) :: aval
   type(contrib_type), dimension(:), target, intent(inout) :: child_contrib
   type(ssids_options), intent(in) :: options
   type(ssids_inform), intent(inout) :: inform
   real(wp), dimension(*), target, optional, intent(in) :: scaling

   type(gpu_numeric_subtree), pointer :: gpu_factor
   
   type(C_PTR) :: gpu_val, gpu_scaling
   type(C_PTR), dimension(:), allocatable :: gpu_contribs
   integer(long) :: sz
   integer, dimension(:), allocatable :: map ! work array, one copy per
      ! thread. Size (0:n), with 0 index used to track which node current map
      ! refers to.
   type(thread_stats) :: stats ! accumulates per thread statistics that are
      ! then summed to obtain global stats in inform. FIXME: not needed?

   integer :: cuda_error, st

   ! Leave output as null until successful exit
   nullify(factor)

   ! Specify which device we're using
   cuda_error = cudaSetDevice(this%device)
   if (cuda_error .ne. 0) goto 200

   ! Allocate cpu_factor for output
   allocate(gpu_factor, stat=st)
   if (st .ne. 0) goto 10
   gpu_factor%symbolic => this
   gpu_factor%posdef = posdef
   gpu_factor%device = this%device

   ! Allocate memory
   gpu_factor%host_factors = .false.
   allocate(gpu_factor%nodes(this%nnodes+1), stat=st)
   if (st .ne. 0) goto 10
   gpu_factor%nodes(:)%ndelay = 0

   allocate(map(0:this%n), stat=st)
   if (st .ne. 0) goto 10
   map(0) = -1 ! initally map unassociated with any node

   ! Setup child contribution array
   ! Note: only non-NULL where we're passing contributions between subtrees
   allocate(gpu_contribs(this%nnodes), stat=st)
   if (st .ne. 0) goto 10
   gpu_contribs(:) = C_NULL_PTR

   ! Initialise host allocator
   ! * options%multiplier * n             integers (for nodes(:)%perm)
   ! * options%multiplier * (nfactor+2*n) reals    (for nodes(:)%lcol)
   ! FIXME: do we really need this multiplier memory????
   ! FIXME: In posdef case ints and reals not needed!
   allocate(gpu_factor%alloc, stat=st)
   if (st .ne. 0) goto 10
   call smalloc_setup(gpu_factor%alloc, &
        max(this%n+0_long, int(options%multiplier*this%n,kind=long)), &
        max(this%nfactor+2*this%n, &
        int(options%multiplier*real(this%nfactor,wp)+2*this%n,kind=long)), st)
   if (st .ne. 0) goto 10

   ! Copy A values to GPU
   sz = this%max_a_idx
   cuda_error = cudaMalloc(gpu_val, aligned_size(sz*C_SIZEOF(aval(1))))
   if (cuda_error .ne. 0) goto 200
   cuda_error = cudaMemcpy_h2d(gpu_val, C_LOC(aval), sz*C_SIZEOF(aval(1)))
   if (cuda_error .ne. 0) goto 200
   if (present(scaling)) then
      ! Copy scaling vector to GPU
      cuda_error = cudaMalloc(gpu_scaling, aligned_size(this%n*C_SIZEOF(scaling(1))))
      if (cuda_error .ne. 0) goto 200
      cuda_error = cudaMemcpy_h2d(gpu_scaling, C_LOC(scaling), &
           this%n*C_SIZEOF(scaling(1)))
      if (cuda_error .ne. 0) goto 200
   end if
   ! Synchronise the device, see:
   ! http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
   cuda_error = cudaDeviceSynchronize()
   if (cuda_error .ne. 0) goto 200

   ! Initialize stream and contrib_wait event
   cuda_error = cudaStreamCreate(gpu_factor%stream_handle)
   if (cuda_error .ne. 0) goto 200
   cuda_error = cudaEventCreateWithFlags(gpu_factor%contrib_wait, &
        cudaEventBlockingSync+cudaEventDisableTiming)
   if (cuda_error .ne. 0) goto 200

   ! Call main factorization routine
   if (present(scaling)) then
      ! ! Copy scaling vector to GPU
      ! cuda_error = cudaMalloc(gpu_scaling, aligned_size(this%n*C_SIZEOF(scaling(1))))
      ! if (cuda_error .ne. 0) goto 200
      ! cuda_error = cudaMemcpy_h2d(gpu_scaling, C_LOC(scaling), &
      !      this%n*C_SIZEOF(scaling(1)))
      ! if (cuda_error .ne. 0) goto 200

      ! Perform factorization
      call parfactor(posdef, this%child_ptr, this%child_list, this%n,         &
           this%nptr, this%gpu_nlist, gpu_val, this%nnodes, gpu_factor%nodes, &
           this%sptr, this%sparent, this%rptr, this%rlist,                    &
           this%rlist_direct, this%gpu_rlist, this%gpu_rlist_direct,          &
           gpu_contribs, gpu_factor%stream_handle, gpu_factor%stream_data,    &
           gpu_factor%gpu_rlist_with_delays,                                  &
           gpu_factor%gpu_rlist_direct_with_delays, gpu_factor%gpu_clists,    &
           gpu_factor%gpu_clists_direct, gpu_factor%gpu_clen,                 &
           gpu_factor%contrib, gpu_factor%contrib_wait, gpu_factor%alloc,     &
           options, stats, ptr_scale=gpu_scaling)
      cuda_error = cudaFree(gpu_scaling)
      if (cuda_error .ne. 0) goto 200
   else
      call parfactor(posdef, this%child_ptr, this%child_list, this%n,         &
           this%nptr, this%gpu_nlist, gpu_val, this%nnodes, gpu_factor%nodes, &
           this%sptr, this%sparent, this%rptr, this%rlist,                    &
           this%rlist_direct, this%gpu_rlist, this%gpu_rlist_direct,          &
           gpu_contribs, gpu_factor%stream_handle, gpu_factor%stream_data,    &
           gpu_factor%gpu_rlist_with_delays,                                  &
           gpu_factor%gpu_rlist_direct_with_delays, gpu_factor%gpu_clists,    &
           gpu_factor%gpu_clists_direct, gpu_factor%gpu_clen,                 &
           gpu_factor%contrib, gpu_factor%contrib_wait, gpu_factor%alloc,     &
           options, stats)
   end if

   cuda_error = cudaFree(gpu_val)
   if (cuda_error .ne. 0) goto 200
   
   ! Set inform (in cumulative fashion)
   if ((inform%flag .lt. 0) .or. (stats%flag .lt. 0)) then
      inform%flag = min(inform%flag, stats%flag) ! error
   else
      inform%flag = max(inform%flag, stats%flag) ! warning/success
   end if
   if (stats%st .ne. 0) inform%stat = stats%st
   inform%maxfront = max(inform%maxfront, stats%maxfront)
   inform%num_factor = inform%num_factor+stats%num_factor
   inform%num_flops = inform%num_flops+stats%num_flops
   inform%num_delay = inform%num_delay+stats%num_delay
   inform%num_neg = inform%num_neg+stats%num_neg
   inform%num_two = inform%num_two+stats%num_two
   inform%matrix_rank = inform%matrix_rank - stats%num_zero
   if (stats%cuda_error .ne. 0) inform%cuda_error = stats%cuda_error
   if (stats%cublas_error .ne. 0) inform%cublas_error = stats%cublas_error

   ! Success, set result and return
   factor => gpu_factor
   return

   ! Allocation error handler
10 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   inform%stat = st
   deallocate(gpu_factor, stat=st)
   return

200 continue
   inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
   inform%cuda_error = cuda_error
   deallocate(gpu_factor, stat=st)
   return
 end function factor

 subroutine numeric_cleanup(this)
   implicit none
   class(gpu_numeric_subtree), intent(inout) :: this

   integer :: flag, cuda_error

   ! FIXME: error handling?

   cuda_error = cudaSetDevice(this%device)
   if (cuda_error .ne. 0) then
      print *, "Failed to set device in numeric_cleanup ", cuda_error
      stop
   end if

   ! Skip if nothing intialized
   if (.not. allocated(this%nodes)) return

   !
   ! Now cleanup GPU-specific stuff
   !
   call free_gpu_type(this%stream_data, flag)

   ! CPU-side allocator
   call smfreeall(this%alloc)
   deallocate(this%alloc)
   nullify(this%alloc)

   ! Cleanup top-level presolve info
   if (C_ASSOCIATED(this%gpu_rlist_with_delays)) then
      flag = cudaFree(this%gpu_rlist_with_delays)
      this%gpu_rlist_with_delays = C_NULL_PTR
      if (flag .ne. 0) return
   end if
   if (C_ASSOCIATED(this%gpu_clists)) then
      flag = cudaFree(this%gpu_clists)
      this%gpu_clists = C_NULL_PTR
      if (flag .ne. 0) return
   end if
   if (C_ASSOCIATED(this%gpu_clists_direct)) then
      flag = cudaFree(this%gpu_clists)
      this%gpu_clists = C_NULL_PTR
      if (flag .ne. 0) return
   end if
   if (C_ASSOCIATED(this%gpu_clen)) then
      flag = cudaFree(this%gpu_clen)
      this%gpu_clen = C_NULL_PTR
      if (flag .ne. 0) return
   end if

   ! Release event
   flag = cudaEventDestroy(this%contrib_wait)
   if (flag .ne. 0) return
   this%contrib_wait = C_NULL_PTR

   ! Release stream
   flag = cudaStreamDestroy(this%stream_handle)
   if (flag .ne. 0) return
   this%stream_handle = C_NULL_PTR
 end subroutine numeric_cleanup

 function get_contrib(this)
   implicit none
   type(contrib_type) :: get_contrib
   class(gpu_numeric_subtree), intent(in) :: this

   integer :: cuda_error

   cuda_error = cudaSetDevice(this%symbolic%device)

   get_contrib = this%contrib
   get_contrib%owner = 1 ! gpu

   ! Now wait until data copy has finished before releasing result
   ! FIXME: handle cuda_error?
   cuda_error = cudaEventSynchronize(this%contrib_wait)
   ! Play safe and synchronize the entire stream.
   ! FIXME: handle cuda_error?
   ! FIXME: remove if not needed
   cuda_error = cudaStreamSynchronize(this%stream_handle)
 end function get_contrib

 subroutine gpu_free_contrib(contrib)
   implicit none
   type(contrib_type), intent(inout) :: contrib

   ! FIXME: stat check?
   if (associated(contrib%delay_val)) &
        deallocate(contrib%delay_val)
   nullify(contrib%delay_val)
   if (associated(contrib%val)) &
        deallocate(contrib%val)
   nullify(contrib%val)
 end subroutine gpu_free_contrib

! FIXME: general: push/pop cuda settings at higher level
! FIXME: general: do we need to worry about avoiding unnecessary gpu_x creation?
 subroutine solve_fwd(this, nrhs, x, ldx, inform)
   implicit none
   class(gpu_numeric_subtree), intent(inout) :: this
   integer, intent(in) :: nrhs
   real(wp), dimension(*), intent(inout) :: x
   integer, intent(in) :: ldx
   type(ssids_inform), intent(inout) :: inform

   integer :: r
   integer :: cuda_error

   ! Specify which device we're using
   cuda_error = cudaSetDevice(this%symbolic%device)
   if (cuda_error .ne. 0) goto 200

   ! NB: gpu solve doesn't support nrhs > 1, so just loop instead
   do r = 0, nrhs-1
      call fwd_solve_gpu(this%posdef, this%symbolic%child_ptr,      &
           this%symbolic%child_list, this%symbolic%n,               &
           this%symbolic%nnodes, this%nodes, this%symbolic%rptr,    &
           this%stream_handle, this%stream_data,                    &
           x(r*ldx+1:r*ldx:this%symbolic%n), inform%stat, cuda_error)
      if (inform%stat .ne. 0) goto 100
      if (cuda_error .ne. 0) goto 200
   end do

   return

100 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   return

200 continue ! CUDA error
   inform%cuda_error = cuda_error
   inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
   return
 end subroutine solve_fwd

 ! FIXME: general solve : recover gpu_x memory on error?
 subroutine solve_diag(this, nrhs, x, ldx, inform)
   implicit none
   class(gpu_numeric_subtree), intent(inout) :: this
   integer, intent(in) :: nrhs
   real(wp), dimension(*), intent(inout) :: x
   integer, intent(in) :: ldx
   type(ssids_inform), intent(inout) :: inform

   integer :: r
   integer :: cuda_error

   ! Specify which device we're using
   cuda_error = cudaSetDevice(this%symbolic%device)
   if (cuda_error .ne. 0) goto 200

   ! NB: gpu solve doesn't support nrhs > 1, so just loop instead
   do r = 0, nrhs-1
      call d_solve_gpu(this%symbolic%nnodes, this%symbolic%sptr,    &
           this%stream_handle, this%stream_data, this%symbolic%n,   &
           x(r*ldx+1:r*ldx:this%symbolic%n), inform%stat, cuda_error)
      if (inform%stat .ne. 0) goto 100
      if (cuda_error .ne. 0) goto 200
   end do

   return

100 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   return

200 continue ! CUDA error
   inform%cuda_error = cuda_error
   inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
   return
 end subroutine solve_diag

 subroutine solve_diag_bwd(this, nrhs, x, ldx, inform)
   implicit none
   class(gpu_numeric_subtree), intent(inout) :: this
   integer, intent(in) :: nrhs
   real(wp), dimension(*), intent(inout) :: x
   integer, intent(in) :: ldx
   type(ssids_inform), intent(inout) :: inform
   
   integer :: r
   integer :: cuda_error

   ! Specify which device we're using
   cuda_error = cudaSetDevice(this%symbolic%device)
   if (cuda_error .ne. 0) goto 200

   ! NB: gpu solve doesn't support nrhs > 1, so just loop instead
   do r = 0, nrhs-1
      call bwd_solve_gpu(SSIDS_SOLVE_JOB_DIAG_BWD, this%posdef,     &
           this%symbolic%n, this%stream_handle, this%stream_data,   &
           x(r*ldx+1:r*ldx:this%symbolic%n), inform%stat, cuda_error)
      if (inform%stat .ne. 0) goto 100
      if (cuda_error .ne. 0) goto 200
   end do

   return

100 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   return

200 continue ! CUDA error
   inform%cuda_error = cuda_error
   inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
   return
 end subroutine solve_diag_bwd

 subroutine solve_bwd(this, nrhs, x, ldx, inform)
   implicit none
   class(gpu_numeric_subtree), intent(inout) :: this
   integer, intent(in) :: nrhs
   real(wp), dimension(*), intent(inout) :: x
   integer, intent(in) :: ldx
   type(ssids_inform), intent(inout) :: inform

   integer :: r
   integer :: cuda_error

   ! Specify which device we're using
   cuda_error = cudaSetDevice(this%symbolic%device)
   if (cuda_error .ne. 0) goto 200

   ! NB: gpu solve doesn't support nrhs > 1, so just loop instead
   do r = 0, nrhs-1
      call bwd_solve_gpu(SSIDS_SOLVE_JOB_BWD, this%posdef, this%symbolic%n, &
           this%stream_handle, this%stream_data,                            &
           x(r*ldx+1:r*ldx:this%symbolic%n), inform%stat, cuda_error)
      if (inform%stat .ne. 0) goto 100
      if (cuda_error .ne. 0) goto 200
   end do

   return

100 continue
   inform%flag = SSIDS_ERROR_ALLOCATION
   return

200 continue ! CUDA error
   inform%cuda_error = cuda_error
   inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
   return
 end subroutine solve_bwd

 subroutine enquire_posdef(this, d)
   implicit none
   class(gpu_numeric_subtree), target, intent(in) :: this
   real(wp), dimension(*), target, intent(out) :: d

   integer :: blkn, blkm
   integer(long) :: i
   integer :: j
   integer :: node
   integer :: piv
   integer :: cuda_error

   type(node_type), pointer :: nptr

   ! Specify which device we're using
   cuda_error = cudaSetDevice(this%symbolic%device)
   ! FIXME: error handling

   !if(.not. this%host_factors) then
   !   ! FIXME: Not implemented enquire_posdef for GPU without host factors yet
   !   inform%flag = SSIDS_ERROR_UNIMPLEMENTED
   !   return
   !endif

   piv = 1
   do node = 1, this%symbolic%nnodes
      nptr => this%nodes(node) 
      blkn = this%symbolic%sptr(node+1) - this%symbolic%sptr(node)
      blkm = int(this%symbolic%rptr(node+1) - this%symbolic%rptr(node))
      i = 1
      do j = 1, blkn
         d(piv) = nptr%lcol(i)
         i = i + blkm + 1
         piv = piv + 1
      end do
   end do
 end subroutine enquire_posdef

 subroutine enquire_indef(this, piv_order, d)
   implicit none
   class(gpu_numeric_subtree), target, intent(in) :: this
   integer, dimension(*), target, optional, intent(out) :: piv_order
   real(wp), dimension(2,*), target, optional, intent(out) :: d

   integer :: blkn, blkm
   integer :: j, k
   integer :: n
   integer :: nd
   integer :: node
   integer(long) :: offset
   integer :: piv
   real(C_DOUBLE), dimension(:), allocatable, target :: d2
   type(C_PTR) :: srcptr
   integer, dimension(:), allocatable :: lvllookup
   integer :: st, cuda_error
   real(wp) :: real_dummy

   type(node_type), pointer :: nptr

   cuda_error = cudaSetDevice(this%symbolic%device)
   ! FIXME: error handling

   if (this%host_factors) then
      ! Call CPU version instead
      call enquire_indef_gpu_cpu(this, piv_order=piv_order, d=d)
      return
   end if

   !
   ! Otherwise extract information from GPU memory
   !

   n = this%symbolic%n
   if (present(d)) then
      ! ensure d is not returned undefined
      d(1:2,1:n) = 0.0
   end if
   
   allocate(lvllookup(this%symbolic%nnodes), d2(2*this%symbolic%n), stat=st)
   if (st .ne. 0) goto 100

   piv = 1
   do node = 1, this%symbolic%nnodes
      nptr => this%nodes(node)
      j = 1
      nd = nptr%ndelay
      blkn = this%symbolic%sptr(node+1) - this%symbolic%sptr(node) + nd
      blkm = int(this%symbolic%rptr(node+1) - this%symbolic%rptr(node)) + nd
      offset = blkm*(blkn+0_long)
      srcptr = c_ptr_plus(nptr%gpu_lcol, offset*C_SIZEOF(real_dummy))
      cuda_error = cudaMemcpy_d2h(C_LOC(d2), srcptr, &
           2*nptr%nelim*C_SIZEOF(d2(1)))
      if (cuda_error .ne. 0) goto 200
      do while (j .le. nptr%nelim)
         if (d2(2*j) .ne. 0) then
            ! 2x2 pivot
            if (present(piv_order))  then
               k = nptr%perm(j)
               piv_order(k) = -piv
               k = nptr%perm(j+1)
               piv_order(k) = -(piv+1)
            end if
            if (present(d)) then
               d(1,piv) = d2(2*j-1)
               d(2,piv) = d2(2*j)
               d(1,piv+1) = d2(2*j+1)
               d(2,piv+1) = 0
            end if
            piv = piv + 2
            j = j + 2
         else
            ! 1x1 pivot
            if (present(piv_order)) then
               k = nptr%perm(j)
               piv_order(k) = piv
            end if
            if (present(d)) then
               d(1,piv) = d2(2*j-1)
               d(2,piv) = 0
            end if
            piv = piv + 1
            j = j + 1
         end if
      end do
   end do

   return ! Normal return

100 continue ! Memory allocation error
   ! FIXME
   !inform%stat = st
   !inform%flag = SSIDS_ERROR_ALLOCATION
   return

200 continue ! CUDA error
   ! FIXME
   !select type(inform)
   !type is (ssids_inform_gpu)
   !   inform%cuda_error = cuda_error
   !end select
   !inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
   return
 end subroutine enquire_indef

 ! Provide implementation for when factors are on host
 subroutine enquire_indef_gpu_cpu(this, piv_order, d)
   implicit none
   class(gpu_numeric_subtree), target, intent(in) :: this
   integer, dimension(*), optional, intent(out) :: piv_order
      ! If i is used to index a variable, its position in the pivot sequence
      ! will be placed in piv_order(i), with its sign negative if it is
      ! part of a 2 x 2 pivot; otherwise, piv_order(i) will be set to zero.
   real(wp), dimension(2,*), optional, intent(out) :: d ! The diagonal
      ! entries of D^{-1} will be placed in d(1,:i) and the off-diagonal
      ! entries will be placed in d(2,:). The entries are held in pivot order.

   integer :: blkn, blkm
   integer :: j, k
   integer :: nd
   integer :: node
   integer(long) :: offset
   integer :: piv

   type(node_type), pointer :: nptr

   piv = 1
   do node = 1, this%symbolic%nnodes
      nptr => this%nodes(node)
      j = 1
      nd = nptr%ndelay
      blkn = this%symbolic%sptr(node+1) - this%symbolic%sptr(node) + nd
      blkm = int(this%symbolic%rptr(node+1) - this%symbolic%rptr(node)) + nd
      offset = blkm*(blkn+0_long)
      do while (j .le. nptr%nelim)
         if (nptr%lcol(offset+2*j).ne.0) then
            ! 2x2 pivot
            if (present(piv_order))  then
               k = nptr%perm(j)
               piv_order(k) = -piv
               k = nptr%perm(j+1)
               piv_order(k) = -(piv+1)
            end if
            if (present(d)) then
               d(1,piv) = nptr%lcol(offset+2*j-1)
               d(2,piv) = nptr%lcol(offset+2*j)
               d(1,piv+1) = nptr%lcol(offset+2*j+1)
               d(2,piv+1) = 0
            end if
            piv = piv + 2
            j = j + 2
         else
            ! 1x1 pivot
            if (present(piv_order)) then
               k = nptr%perm(j)
               piv_order(k) = piv
            end if
            if (present(d)) then
               d(1,piv) = nptr%lcol(offset+2*j-1)
               d(2,piv) = 0
            end if
            piv = piv + 1
            j = j + 1
         end if
      end do
   end do
 end subroutine enquire_indef_gpu_cpu

 subroutine alter(this, d)
   implicit none
   class(gpu_numeric_subtree), target, intent(inout) :: this
   real(wp), dimension(2,*), intent(in) :: d

   integer :: blkm, blkn
   integer(long) :: ip
   integer :: j
   integer :: nd
   integer :: node
   integer :: piv
   real(wp), dimension(:), allocatable, target :: d2
   integer, dimension(:), allocatable :: lvllookup
   type(C_PTR) :: srcptr
   integer :: st, cuda_error
   real(wp) :: real_dummy

   type(node_type), pointer :: nptr

   cuda_error = cudaSetDevice(this%symbolic%device)
   ! FIXME: error handling

   if (this%host_factors) &
        call alter_gpu_cpu(this, d)

   !
   ! Also alter GPU factors if they exist
   !

   ! FIXME: Move to gpu_factors a la host_factors?
   ! FIXME: if statement
   !if (options%use_gpu_solve) then
      allocate(lvllookup(this%symbolic%nnodes), d2(2*this%symbolic%n), stat=st)
      if (st .ne. 0) goto 100

      piv = 1
      do node = 1, this%symbolic%nnodes
         nptr => this%nodes(node)
         nd = nptr%ndelay
         blkn = this%symbolic%sptr(node+1) - this%symbolic%sptr(node) + nd
         blkm = int(this%symbolic%rptr(node+1) - this%symbolic%rptr(node)) + nd
         ip = 1
         do j = 1, nptr%nelim
            d2(ip)   = d(1,piv)
            d2(ip+1) = d(2,piv)
            ip = ip + 2
            piv = piv + 1
         end do
         srcptr = c_ptr_plus(this%nodes(node)%gpu_lcol, &
            blkm*(blkn+0_long)*C_SIZEOF(real_dummy))
         cuda_error = cudaMemcpy_h2d(srcptr, C_LOC(d2), &
            2*nptr%nelim*C_SIZEOF(d2(1)))
         if (cuda_error .ne. 0) goto 200
      end do
   !endif

   ! Synchronise the device, see:
   ! http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
   cuda_error = cudaDeviceSynchronize()
   if (cuda_error .ne. 0) goto 200

   return ! Normal return

100 continue ! Memory allocation error
   ! FIXME
   !inform%stat = st
   !inform%flag = SSIDS_ERROR_ALLOCATION
   return

200 continue ! CUDA error
   ! FIXME
   !select type(inform)
   !type is (ssids_inform_gpu)
   !   inform%cuda_error = cuda_error
   !end select
   !inform%flag = SSIDS_ERROR_CUDA_UNKNOWN
   return
 end subroutine alter

 ! Alters D for factors stored on CPU
 subroutine alter_gpu_cpu(this, d)
   implicit none
   class(gpu_numeric_subtree), target, intent(inout) :: this
   real(wp), dimension(2,*), intent(in) :: d  ! The required diagonal entries
     ! of D^{-1} must be placed in d(1,i) (i = 1,...n)
     ! and the off-diagonal entries must be placed in d(2,i) (i = 1,...n-1).

   integer :: blkm, blkn
   integer(long) :: ip
   integer :: j
   integer :: nd
   integer :: node
   integer :: piv

   type(node_type), pointer :: nptr

   piv = 1
   do node = 1, this%symbolic%nnodes
      nptr => this%nodes(node)
      nd = nptr%ndelay
      blkn = this%symbolic%sptr(node+1) - this%symbolic%sptr(node) + nd
      blkm = int(this%symbolic%rptr(node+1) - this%symbolic%rptr(node)) + nd
      ip = blkm*(blkn+0_long) + 1
      do j = 1, nptr%nelim
         nptr%lcol(ip)   = d(1,piv)
         nptr%lcol(ip+1) = d(2,piv)
         ip = ip + 2
         piv = piv + 1
      end do
   end do
 end subroutine alter_gpu_cpu

end module spral_ssids_gpu_subtree
