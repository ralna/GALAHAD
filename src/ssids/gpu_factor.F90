! Copyright (c) 2013 Science and Technology Facilities Council (STFC)
! Authors: Evgueni Ovtchinnikov and Jonathan Hogg
!
! Factorize phase to run on GPU
module spral_ssids_gpu_factor
  use, intrinsic :: iso_c_binding
  use spral_cuda
  use spral_ssids_contrib, only : contrib_type
  use spral_ssids_datatypes
  use spral_ssids_profile, only : profile_set_state
  use spral_ssids_gpu_alloc, only : cuda_stack_alloc_type, custack_alloc, &
       custack_init, custack_finalize, custack_free
  use spral_ssids_gpu_datatypes
  use spral_ssids_gpu_interfaces
  use spral_ssids_gpu_dense_factor, only : &
       node_ldlt, node_llt, multinode_llt, multinode_ldlt
  use spral_ssids_gpu_smalloc, only : smalloc
  use spral_ssids_gpu_solve, only : setup_gpu_solve
  implicit none

  private
  public :: parfactor ! Performs factorization phase using multiple streams
  ! C interfaces
  public :: spral_ssids_assign_nodes_to_levels
  
  type :: ntype
     integer :: level = 1
     integer :: subtree = 0
     integer(long) :: work_here = 0
     integer(long) :: work_below = 0 ! includes work_here
  end type ntype

  type :: asmtype
     integer :: npassed      ! #rows passed to parent total
     integer :: npassl       ! #cols passed to parent's L part
     integer(long) :: offset ! start of rows to pass up in rlist(:)
     ! i.e. rptr(child)+blkn-1
  end type asmtype

contains

  subroutine parfactor(pos_def, child_ptr, child_list, n, nptr, gpu_nlist,    &
       ptr_val, nnodes, nodes, sptr, sparent, rptr, rlist, rlist_direct,      &
       gpu_rlist, gpu_rlist_direct, gpu_contribs, stream_handle, stream_data, &
       gpu_rlist_with_delays, gpu_rlist_direct_with_delays, gpu_clists,       &
       gpu_clists_direct, gpu_clen, contrib, contrib_wait, alloc, options,    &
       stats, ptr_scale)
    implicit none
    logical, intent(in) :: pos_def ! True if problem is supposedly pos-definite
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    integer, intent(in) :: n
    integer(long), dimension(*), intent(in) :: nptr
    type(C_PTR), intent(in) :: gpu_nlist
    type(C_PTR), intent(in) :: ptr_val
    ! Note: gfortran-4.3 bug requires explicit size of nodes array
    integer, intent(in) :: nnodes
    type(node_type), dimension(nnodes), intent(inout) :: nodes
    integer, dimension(*), intent(in) :: sptr
    integer, dimension(*), intent(in) :: sparent
    integer(long), dimension(*), intent(in) :: rptr
    integer, dimension(*), intent(in) :: rlist
    integer, dimension(*), intent(in), target :: rlist_direct
    type(C_PTR), intent(in) :: gpu_rlist
    type(C_PTR), intent(in) :: gpu_rlist_direct
    type(C_PTR), dimension(*), intent(inout) :: gpu_contribs
    type(C_PTR), intent(in) :: stream_handle
    type(gpu_type), intent(out) :: stream_data
    type(C_PTR), intent(out) :: gpu_rlist_with_delays
    type(C_PTR), intent(out) :: gpu_rlist_direct_with_delays
    type(C_PTR), intent(out) :: gpu_clists
    type(C_PTR), intent(out) :: gpu_clists_direct
    type(C_PTR), intent(out) :: gpu_clen
    type(contrib_type), intent(out) :: contrib
    type(C_PTR), intent(inout) :: contrib_wait
    type(smalloc_type), target, intent(inout) :: alloc ! Contains actual memory
      ! allocations for L. Everything else (within the subtree) is just a
      ! pointer to this.
    ! explicit size required on buf to avoid gfortran-4.3 bug
    type(ssids_options), intent(in) :: options
    type(thread_stats), intent(inout) :: stats
    type(C_PTR), optional, intent(in) :: ptr_scale

    type(C_PTR) :: gpu_LDLT
   
    integer :: st

    type(cuda_settings_type) :: user_settings

    ! Start recording profile
    call profile_set_state("C_GPU0", "ST_GPU_TASK", "GT_FACTOR")

    ! Set GPU device settings as we wish
    call push_ssids_cuda_settings(user_settings, stats%cuda_error)
    if (stats%cuda_error .ne. 0) goto 200

    ! Determine level structure
    allocate(stream_data%lvllist(nnodes), &
         stream_data%lvlptr(nnodes + 1), stat=stats%st)
    if (stats%st .ne. 0) goto  100
    call assign_nodes_to_levels(nnodes, sparent, gpu_contribs, &
         stream_data%num_levels, stream_data%lvlptr, stream_data%lvllist, stats%st)
    if (stats%st .ne. 0) goto 100

    ! Perform actual factorization
    call subtree_factor_gpu(stream_handle, pos_def, child_ptr, child_list, &
      n, nptr, gpu_nlist, ptr_val, nnodes, nodes, sptr, sparent, rptr,     &
      rlist_direct, gpu_rlist, gpu_rlist_direct, gpu_contribs, gpu_LDLT,   &
      stream_data, alloc, options, stats, ptr_scale)
    if (stats%flag .lt. 0) return
    if (stats%cuda_error .ne. 0) goto 200
    
    ! Extract contribution to parent of subtree (if any)
    if (C_ASSOCIATED(gpu_LDLT)) then
       call transfer_contrib(nnodes, sptr, rptr, rlist, nodes, gpu_contribs, &
            contrib, contrib_wait, stream_handle, stats%st, stats%cuda_error)
       if (stats%st .ne. 0) goto 100
       if (stats%cuda_error .ne. 0) goto 200
    end if

    ! Free gpu_LDLT as required
    if (C_ASSOCIATED(gpu_LDLT)) then
       stats%cuda_error = cudaFree(gpu_LDLT)
       if (stats%cuda_error .ne. 0) goto 200
    end if

    ! Setup data-structures for normal GPU solve
    call setup_gpu_solve(n, child_ptr, child_list, nnodes, nodes, sparent,     &
         sptr, rptr, rlist, stream_handle, stream_data, gpu_rlist_with_delays, &
         gpu_clists, gpu_clists_direct, gpu_clen, stats%st, stats%cuda_error,  &
         gpu_rlist_direct_with_delays)
    if (stats%st .ne. 0) goto 100
    if (stats%cuda_error .ne. 0) goto 200

    ! Restore user GPU device settings
    call push_ssids_cuda_settings(user_settings, stats%cuda_error)
    if (stats%cuda_error .ne. 0) goto 200

    ! Stop recording profile
    call profile_set_state("C_GPU0", "ST_GPU_TASK", "0")

    return

100 continue ! Fortran Memory allocation error
    stats%flag = SSIDS_ERROR_ALLOCATION
    call push_ssids_cuda_settings(user_settings, st)
    return

200 continue ! CUDA Error
    stats%flag = SSIDS_ERROR_CUDA_UNKNOWN
    call push_ssids_cuda_settings(user_settings, st)
    return
  end subroutine parfactor

  subroutine transfer_contrib(node, sptr, rptr, rlist, nodes, gpu_contribs, &
       contrib, contrib_wait, stream, st, cuda_error)
    implicit none
    integer, intent(in) :: node
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    integer, dimension(*), target, intent(in) :: rlist
    type(node_type), dimension(*), intent(in) :: nodes
    type(C_PTR), dimension(*), intent(in) :: gpu_contribs
    type(contrib_type), intent(out) :: contrib
    type(C_PTR), intent(inout) :: contrib_wait ! event pointer
    type(C_PTR), intent(in) :: stream
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    integer :: blkn, blkm, ndelay_in, nelim
    integer(C_SIZE_T) :: sz
    type(C_PTR) :: gpu_delay_ptr

    cuda_error = 0
    st = 0

    ! Initialise block statistics
    blkn = sptr(node+1) - sptr(node)
    blkm = int(rptr(node+1) - rptr(node))
    ndelay_in = nodes(node)%ndelay
    nelim = nodes(node)%nelim

    ! Handle expected contribution block
    contrib%n = blkm - blkn
    sz = int(contrib%n,c_size_t)*contrib%n
    allocate(contrib%val(sz), stat=st)
    if (st .ne. 0) return
    contrib%ldval = contrib%n
    cuda_error = cudaMemcpyAsync_d2h(C_LOC(contrib%val), gpu_contribs(node), &
         sz*C_SIZEOF(contrib%val(1)), stream)
    if (cuda_error .ne. 0) return
    contrib%rlist => rlist(rptr(node)+blkn:rptr(node)+blkm)

    ! Handle any delays
    contrib%ndelay = blkn+ndelay_in - nelim
    if (contrib%ndelay .gt. 0) then
       ! There are some delays
       contrib%delay_perm => nodes(node)%perm(nelim+1:blkn+ndelay_in)
       contrib%lddelay = blkm-blkn + contrib%ndelay
       allocate(contrib%delay_val(contrib%lddelay*int(contrib%ndelay,long)), &
            stat=st)
       if (st .ne. 0) return
       gpu_delay_ptr = c_ptr_plus( nodes(node)%gpu_lcol, &
            nelim*(blkm+ndelay_in+1_C_SIZE_T)*C_SIZEOF(contrib%delay_val(1)) )
       cuda_error = cudaMemcpy2DAsync(C_LOC(contrib%delay_val), &
            contrib%lddelay*C_SIZEOF(contrib%delay_val(1)), gpu_delay_ptr, &
            (blkm+ndelay_in)*C_SIZEOF(contrib%delay_val(1)), &
            contrib%lddelay*C_SIZEOF(contrib%delay_val(1)), &
            int(contrib%ndelay, C_SIZE_T), cudaMemcpyDeviceToHost, stream)
       if (cuda_error .ne. 0) return
    else
       ! No delays
       nullify(contrib%delay_perm)
       nullify(contrib%delay_val)
       contrib%lddelay = 0
    end if

    ! Record event when we're finished
    cuda_error = cudaEventRecord(contrib_wait, stream)
    if (cuda_error .ne. 0) return
  end subroutine transfer_contrib

!*******************************
!
! This subroutine factorises the subtree(s) that include nodes sa through
! en. Any elements being passed to nodes numbered higher than en are allocated
! using Fortran allocate statemenets rather than stack_alloc.
!
! We maintain the factors seperately from the generated elements to avoid
! copying. Factors are stored in alloc, but pointed to by entries of nodes(:)
! for ease of reference.
!
! Generated elements are stored in a pair of stacks (need two so we can copy
! from child to parent). They are not touched until the factorization has
! been performed on columns that we expect to eliminate.
!
! Entries of A are only added to L just before they are expected to be
! eliminated.
!
  subroutine subtree_factor_gpu(stream, pos_def, child_ptr, child_list, n, &
       nptr, gpu_nlist, ptr_val, nnodes, nodes, sptr, sparent, rptr,      &
       rlist_direct, gpu_rlist, gpu_rlist_direct, gpu_contribs, gpu_LDLT, &
       gpu, alloc, options, stats, ptr_scale)
    implicit none
    type(C_PTR), intent(in) :: stream ! stream handle to execute on
    logical, intent(in) :: pos_def ! True if problem is supposedly pos-definite
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    integer, intent(in) :: n
    integer(long), dimension(*), intent(in) :: nptr
    type(C_PTR), intent(in) :: gpu_nlist
    type(C_PTR), intent(in) :: ptr_val
    ! Note: gfortran-4.3 bug requires explicit size of nodes array
    integer, intent(in) :: nnodes
    type(node_type), dimension(nnodes), intent(inout) :: nodes
    integer, dimension(*), intent(in) :: sptr
    integer, dimension(*), intent(in) :: sparent
    integer(long), dimension(*), intent(in) :: rptr
    integer, dimension(*), intent(in), target :: rlist_direct
    type(C_PTR), intent(in) :: gpu_rlist
    type(C_PTR), intent(in) :: gpu_rlist_direct
    type(C_PTR), dimension(*), intent(inout) :: gpu_contribs
    type(C_PTR), intent(out) :: gpu_LDLT ! ptr to mem that needs freed
    type(gpu_type), intent(inout) :: gpu
    type(smalloc_type), target, intent(inout) :: alloc ! Contains actual memory
      ! allocations for L. Everything else (within the subtree) is just a
      ! pointer to this.
    ! explicit size required on buf to avoid gfortran-4.3 bug
    type(ssids_options), intent(in) :: options
    type(thread_stats), intent(inout) :: stats
    type(C_PTR), optional, intent(in) :: ptr_scale

    integer :: blkm
    integer :: blkn
    integer :: cblkm
    integer :: cblkn
    integer :: cn
    integer :: cnode
    integer :: i
    integer(long) :: ii
    integer :: j
    integer :: k
    integer :: m
    integer :: ndelay

    integer :: node
    integer, dimension(:), pointer :: lperm

    integer(long) :: level_size
    integer :: level_width, level_height
    integer :: total_nch
    integer :: idata_size, max_idata_size
    integer :: LDLT_size, max_LDLT_size
    integer(long) :: pc_size
    integer :: ncb, max_ncb
  
    integer :: nch
    logical :: free_contrib
    integer :: p, q ! general purpose indices
  
    ! per-node assembly info
    type(asmtype), dimension(:), allocatable :: asminf

    real(wp) :: delta, eps
    real(wp), target :: s
    real(wp) :: dummy_real

    ! elimination tree data
    integer :: llist, lev
    integer(long), allocatable :: off_LDLT(:) ! node LDLT contribution offset
      ! in levLDLT

    ! GPU work space (reused for many different things)
    type(cuda_stack_alloc_type) :: gwork
    integer(C_SIZE_T) :: lgpu_work

    ! device pointers
    type(C_PTR), dimension(:), allocatable :: gpu_ldcol
    type(C_PTR) :: ptr_levL, ptr_levLD, ptr_levLDLT
    type(C_PTR) :: ptr_u, ptr_v
    type(C_PTR) :: ptr_cval, ptr_ccval
    type(C_PTR) :: cublas_handle

    ! CUDA-side stats
    type(cuda_stats_type), target :: custats
    type(C_PTR) :: gpu_custats

    gpu_LDLT = C_NULL_PTR
    delta = 0.01_wp
    eps = tiny(1.0_wp)

    if (gpu%num_levels .eq. 0) return ! Shortcut empty streams (v. small matrices)
  
    gpu%n = n
    gpu%nnodes = nnodes

    ! Initialize CUBLAS handle
    stats%cublas_error = cublasCreate(cublas_handle)
    if (stats%cublas_error .ne. 0) goto 300
    stats%cublas_error = cublasSetStream(cublas_handle, stream)
    if (stats%cublas_error .ne. 0) goto 300

    ! Initialize CUDA stats
    stats%cuda_error = cudaMalloc(gpu_custats, aligned_size(C_SIZEOF(custats)))
    if (stats%cuda_error .ne. 0) goto 200
    stats%cuda_error = cudaMemsetAsync(gpu_custats, 0, C_SIZEOF(custats), stream)
    if (stats%cuda_error .ne. 0) goto 200

    ! Precalculate level information
    max_LDLT_size = 0
    max_ncb = 2
    max_idata_size = 0
    do lev = 1, gpu%num_levels
       total_nch = 0
       LDLT_size = 0
       idata_size = 0
       p = 0
       q = 0
       do llist = gpu%lvlptr(lev), gpu%lvlptr(lev + 1) - 1
          node = gpu%lvllist(llist)
          blkn = sptr(node+1) - sptr(node)
          blkm = int(rptr(node+1) - rptr(node))
          nch = child_ptr(node + 1) - child_ptr(node)
          total_nch = total_nch + nch
          m = blkm - blkn
          LDLT_size = LDLT_size + m*m
          k = (m - 1)/BLOCK_SIZE + 1
          p = p + (k*(k + 1))/2
          q = q + (blkm - 1)/BLOCK_SIZE + 2
       end do
       ncb = gpu%lvlptr(lev + 1) - gpu%lvlptr(lev)
       idata_size = max(10*p, 9*q, total_nch)
       max_LDLT_size = max(max_LDLT_size, LDLT_size)
       max_ncb = max(max_ncb, ncb)
       max_idata_size = max(max_idata_size, idata_size)
    end do
  
    ii = nptr(nnodes + 1) - 1
    stats%cuda_error = cudaMalloc(gpu_LDLT, aligned_size(2*max_LDLT_size*C_SIZEOF(dummy_real)))
    if (stats%cuda_error .ne. 0) goto 200
  
    allocate(gpu%values_L(gpu%num_levels), gpu%off_L(nnodes), &
         off_LDLT(nnodes), asminf(nnodes), stat=stats%st)
    if (stats%st .ne. 0) goto 100

    do node = 1, nnodes
       blkm = int(rptr(node + 1) - rptr(node))
       blkn = sptr(node + 1) - sptr(node)
       do cn = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(cn)
          cblkn = sptr(cnode + 1) - sptr(cnode)
          cblkm = int(rptr(cnode + 1) - rptr(cnode))
          m = 0
          do ii = rptr(cnode) + cblkn, rptr(cnode + 1) - 1
             if (rlist_direct(ii) .gt. blkn) exit
             m = m + 1
          end do
          asminf(cnode)%npassed = cblkm - cblkn
          asminf(cnode)%npassl = m
          asminf(cnode)%offset = rptr(cnode) + cblkn - 1
       end do
    end do

    !
    ! Loop over levels doing work
    !
    do lev = 1, gpu%num_levels
       lgpu_work = level_gpu_work_size(lev, gpu%lvlptr, gpu%lvllist, child_ptr, &
            child_list, nodes, sptr, rptr, asminf)
       call custack_init(gwork, lgpu_work, stats%cuda_error)
       if (stats%cuda_error .ne. 0) goto 200

       ncb = gpu%lvlptr(lev + 1) - gpu%lvlptr(lev)

       !
       ! Initialize level information
       !
       level_size = 0
       level_width = 0
       level_height = 0
       total_nch = 0
       pc_size = 0
       do llist = gpu%lvlptr(lev), gpu%lvlptr(lev + 1) - 1
          node = gpu%lvllist(llist)
          ndelay = nodes(node)%ndelay
          blkn = sptr(node+1) - sptr(node) + ndelay
          blkm = int(rptr(node+1) - rptr(node)) + ndelay

          gpu%off_L(node) = level_size
          off_LDLT(node) = pc_size

          nch = child_ptr(node+1) - child_ptr(node)
          level_size = level_size + (blkm + 2_long)*blkn
          level_width = level_width + blkn
          level_height = max(level_height, blkm)
          total_nch = total_nch + nch
          m = blkm - blkn
          pc_size = pc_size + m*(m+0_long)
       end do
      
       !
       ! Generate pointers for this level
       !
       if (mod(lev, 2) .gt. 0) then
          ptr_levLDLT = gpu_LDLT
       else
          ptr_levLDLT = c_ptr_plus(gpu_LDLT, max_LDLT_size*C_SIZEOF(dummy_real))
       end if

       stats%cuda_error = &
            cudaMalloc(gpu%values_L(lev)%ptr_levL, aligned_size(level_size*C_SIZEOF(dummy_real)))
       if (stats%cuda_error .ne. 0) goto 200
       ptr_levL = gpu%values_L(lev)%ptr_levL
       if (.not. pos_def) then
          stats%cuda_error = &
               cudaMalloc(ptr_levLD, aligned_size(level_size*C_SIZEOF(dummy_real)))
          if (stats%cuda_error .ne. 0) goto 200

          ! Initialize pointers to LD storage
          if (allocated(gpu_ldcol)) deallocate(gpu_ldcol, stat=stats%st)
          if (stats%st .ne. 0) goto 100
          allocate(gpu_ldcol(gpu%lvlptr(lev+1)-gpu%lvlptr(lev)), stat=stats%st)
          if (stats%st .ne. 0) goto 100
          do llist = gpu%lvlptr(lev), gpu%lvlptr(lev + 1) - 1
             node = gpu%lvllist(llist)        
             i = llist - gpu%lvlptr(lev) + 1
             gpu_ldcol(i) = &
                  c_ptr_plus(ptr_levLD, gpu%off_L(node)*C_SIZEOF(dummy_real))
          end do
       end if

       ! Set up node pointers
       level_size = 0
       do llist = gpu%lvlptr(lev), gpu%lvlptr(lev + 1) - 1
          node = gpu%lvllist(llist)
          ndelay = nodes(node)%ndelay
          blkn = sptr(node+1) - sptr(node) + ndelay
          blkm = int(rptr(node+1) - rptr(node)) + ndelay
      
          nodes(node)%gpu_lcol = c_ptr_plus(gpu%values_L(lev)%ptr_levL, &
               level_size*C_SIZEOF(dummy_real))

          level_size = level_size + (blkm + 2_long)*blkn
       end do

       ! Allocate+Initialize lperm for fronts on this level
       do llist = gpu%lvlptr(lev), gpu%lvlptr(lev + 1) - 1

          node = gpu%lvllist(llist)
          ndelay = nodes(node)%ndelay
          blkn = sptr(node+1) - sptr(node) + ndelay
          blkm = int(rptr(node+1) - rptr(node)) + ndelay
    
          stats%maxfront = max(stats%maxfront, blkn)

          ! Allocate memory for the local permutation (lperm).
          call smalloc(alloc, nodes(node)%perm, blkn+0_long, &
               nodes(node)%ismptr, nodes(node)%ismsa, stats%st)
          if (stats%st .ne. 0) go to 100
          lperm => nodes(node)%perm

          ! Initialise lperm
          j = ndelay + 1
          do i = sptr(node), sptr(node+1)-1
             lperm(j) = i
             j = j + 1
          end do
    
       end do

       ! Initialize L to 0, and add A to it.
       call init_L_with_A(stream, lev, gpu%lvlptr, gpu%lvllist, nodes, ncb,  &
            level_size, nptr, rptr, gpu_nlist, gpu_rlist, ptr_val, ptr_levL, &
            gwork, stats%st, stats%cuda_error, ptr_scale=ptr_scale)
       if (stats%st .ne. 0) goto 100
       if (stats%cuda_error .ne. 0) goto 200

       ! The pivot is considered to be zero if less than epsilon(ONE) times
       ! the maximum value of A on this level: must use relative threshold
       ! in case the user opts for no scaling
       ! FIXME: Make optional?
       ! FIXME: Use CUBLAS instead? [Check if zeroed]
       k = int( min(65535_long, (level_size - 1)/256 + 1) )
       ptr_u = custack_alloc(gwork, k*C_SIZEOF(dummy_real))
       ptr_v = custack_alloc(gwork, C_SIZEOF(dummy_real))
       call max_abs( stream, k, level_size, ptr_levL, ptr_u, ptr_v )
       stats%cuda_error = cudaMemcpyAsync_D2H(C_LOC(s), ptr_v, C_SIZEOF(s), &
            stream)
       if (stats%cuda_error .ne. 0) goto 200
       stats%cuda_error = cudaStreamSynchronize(stream) ! Wait for ptr_u, ptr_v
       if (stats%cuda_error .ne. 0) goto 200
       call custack_free(gwork, C_SIZEOF(dummy_real)) ! ptr_v
       call custack_free(gwork, k*C_SIZEOF(dummy_real)) ! ptr_u
       eps = s*epsilon(ONE)

       !
       ! Assemble fully summed columns
       !
       call assemble_fully_summed(stream, total_nch, lev, gpu%lvlptr,         &
            gpu%lvllist, nodes, ptr_ccval, gpu_contribs, ptr_levL,            &
            gpu_rlist_direct, child_ptr,  child_list, off_LDLT, asminf, rptr, &
            sptr, gwork, stats%st, stats%cuda_error)
       if (stats%st .ne. 0) goto 100
       if (stats%cuda_error .ne. 0) goto 200
    
       !
       ! Perform factorization (of fully summed columns)
       !
       if (pos_def) then
          call factor_posdef(stream, lev, gpu%lvlptr, nodes, gpu%lvllist, &
               sptr, rptr, ptr_levL, cublas_handle, stats, gwork)
       else
          call factor_indef(stream, lev, gpu%lvlptr, nnodes, nodes, gpu%lvllist,&
               sparent, sptr, rptr, level_height, level_width, delta, eps, &
               gpu_ldcol, gwork, cublas_handle, options, stats, gpu_custats)
       end if
       if (stats%flag .lt. 0) goto 20
       if (stats%st .ne. 0) goto 100
       if (stats%cuda_error .ne. 0) goto 200
       if (stats%cublas_error .ne. 0) goto 300

       !
       ! Form contribution block (of non-fully summed columns)
       !
       if (pc_size .gt. 0) then
          if (pos_def) then
             call form_contrib(stream, lev, gpu%lvlptr, nodes, gpu%lvllist, &
                  off_LDLT, sptr, rptr, ptr_levLDLT, gwork, stats%st,       &
                  stats%cuda_error)
          else
             call form_contrib(stream, lev, gpu%lvlptr, nodes, gpu%lvllist, &
                  off_LDLT, sptr, rptr, ptr_levLDLT, gwork, stats%st,       &
                  stats%cuda_error, gpu_ldcol=gpu_ldcol)
          end if
          if (stats%st .ne. 0) goto 100
          if (stats%cuda_error .ne. 0) goto 200
       end if
       if (stats%flag .lt. 0) goto 20

       !
       ! Assemble children into contribution block
       !
       call assemble_contrib(stream, total_nch, lev, gpu%lvlptr, gpu%lvllist, &
            child_ptr, child_list, sptr, rptr, asminf, pc_size, &
            off_LDLT, ptr_ccval, gpu_contribs, ptr_levLDLT, gpu_rlist_direct, &
            gwork, stats%st, stats%cuda_error)
       if (stats%st .ne. 0) goto 100
       if (stats%cuda_error .ne. 0) goto 200

       ! Free allocs specific to this level
       if (.not. pos_def) then
          stats%cuda_error = cudaFree(ptr_levLD)
          if (stats%cuda_error.ne.0) goto 200
       end if

       ! Store pointers for use on next level
       ptr_cval = ptr_levL
       ptr_ccval = ptr_levLDLT

    end do ! lev

    ! Free stack memory
    call custack_finalize(gwork, stats%cuda_error)
    if (stats%cuda_error .ne. 0) goto 200

    ! Copy GPU stats back to host and free it
    stats%cuda_error = cudaMemcpy_d2h(C_LOC(custats), gpu_custats, &
         C_SIZEOF(custats))
    if (stats%cuda_error .ne. 0) goto 200
    stats%cuda_error = cudaFree(gpu_custats)
    if (stats%cuda_error .ne. 0) goto 200
    stats%num_zero = custats%num_zero
    stats%num_neg = custats%num_neg
    stats%num_two = custats%num_two
  
20  continue ! start of cleanup

    free_contrib = .true.
    do llist = gpu%lvlptr(gpu%num_levels), gpu%lvlptr(gpu%num_levels+1)-1
       node = gpu%lvllist(llist)
       blkm = int(rptr(node+1) - rptr(node))
       blkn = sptr(node+1) - sptr(node)
       if (blkm .gt. blkn) then
          gpu_contribs(node) = &
               c_ptr_plus(ptr_ccval, off_LDLT(node)*C_SIZEOF(dummy_real))
          free_contrib = .false.
       end if
    end do
    if (free_contrib) then
       stats%cuda_error = cudaFree(gpu_LDLT)
       if (stats%cuda_error .ne. 0) goto 200
       gpu_LDLT = C_NULL_PTR
    end if
  
    deallocate(off_LDLT, asminf, stat=stats%st)
    if (stats%st .ne. 0) goto 100
  
    ! Destroy CUBLAS handle
    stats%cublas_error = cublasDestroy(cublas_handle)
    if (stats%cuda_error .ne. 0) goto 300
    
    return ! Normal return

100 continue ! Fortran Memory allocation error
    stats%flag = SSIDS_ERROR_ALLOCATION
    return

200 continue ! CUDA failure
    stats%flag = SSIDS_ERROR_CUDA_UNKNOWN
    return

300 continue ! CUBLAS failure
    stats%flag = SSIDS_ERROR_CUBLAS_UNKNOWN
    return
  end subroutine subtree_factor_gpu

! Following routine calculates size of gwork required for a given level.
! gwork size is made up of maximum of space required for each individual
! work routine
  integer(C_SIZE_T) function level_gpu_work_size(lev, lvlptr, lvllist, &
       child_ptr, child_list, nodes, sptr, rptr, asminf) result(lgpu)
    implicit none
    integer, intent(in) :: lev
    integer, dimension(*), intent(in) :: lvlptr
    integer, dimension(*), intent(in) :: lvllist
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    type(node_type), dimension(*), intent(in) :: nodes
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    type(asmtype), dimension(:), intent(in) :: asminf

    integer(C_SIZE_T) :: sz, sz2, sz3, sz4, sz5, sz6, sz7, sz8
    integer :: li, ci
    integer :: ndelay, blkm, blkn
    integer(long) :: bx, by
    integer :: node, cnode

    ! Dummy datatypes to get sizes
    type(load_nodes_type) :: lnt_dummy
    type(assemble_cp_type) :: acpt_dummy
    type(assemble_blk_type) :: abt_dummy
    type(assemble_delay_type) :: adt_dummy
    type(multisymm_type) :: mst_dummy
    type(multiswap_type) :: mswt_dummy
    type(multisyrk_type) :: msyrt_dymmy
    type(multinode_fact_type) :: mnft_dummy
    type(multiblock_fact_type) :: mbft_dummy
    type(multireorder_data) :: mr_dummy
    type(multielm_data) :: me_dummy
    type(cstat_data_type) :: cdt_dummy
    integer(C_INT) :: int_dummy
    real(C_DOUBLE) :: real_dummy

    ! Initialize space required to 0
    lgpu = 0

    ! Space for lndata in init_L_with_A()
    sz = lvlptr(lev+1)-lvlptr(lev)
    lgpu = max(lgpu, sz*C_SIZEOF(lnt_dummy))

    ! Space for ptr_u, ptr_v with cuda_max_abs()
    sz = 0

    do li = lvlptr(lev), lvlptr(lev+1)-1
       node = lvllist(li)
       ndelay = nodes(node)%ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       sz = sz + (blkm+2_long)*blkn ! calculate 'level_size'
    end do
    sz = int( min(65535_long, (sz - 1)/256 + 1) )
    lgpu = max(lgpu, &
         sz*C_SIZEOF(real_dummy) + & ! ptr_u
         C_SIZEOF(real_dummy))       ! ptr_v

    ! Space for cpdata, blkdata, ddata and sync in assemble_fully_summed()
    sz = 0
    sz2 = 0
    do li = lvlptr(lev), lvlptr(lev+1)-1
       node = lvllist(li)
       sz = sz + child_ptr(node+1)-child_ptr(node)
       do ci = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(ci)
          bx = (asminf(cnode)%npassl-1) / HOGG_ASSEMBLE_TX + 1
          by = (asminf(cnode)%npassed-1) / HOGG_ASSEMBLE_TY + 1
          sz2 = sz2 + bx*by
       end do
    end do
    lgpu = max(lgpu, sz*C_SIZEOF(acpt_dummy) + sz2*C_SIZEOF(abt_dummy) + &
         sz*C_SIZEOF(adt_dummy) + (1+sz)*C_SIZEOF(int_dummy))

    ! Space for msdata in factor_posdef()
    sz = lvlptr(lev+1)-lvlptr(lev) ! number required for msdata
    lgpu = max(lgpu, sz*C_SIZEOF(mst_dummy))

    ! Space for swapdata in factor_indef()
    lgpu = max(lgpu, sz*C_SIZEOF(mswt_dummy))

    ! Space for msdata, ptr_ind and ptr_B in factor_indef()/factor_posdef()
    ! Also gpu_aux, gpu_perm, gpu_rdata, gpu_mdata, gpu_mnfdata, gpu_mbfdata
    !    in multinode_ldlt() / multinode_llt()
    ! Also gpu_r in node_ldlt()
    ! Also gpu_csdata in collect_stats_indef()
    sz = 0 ! number required for ptr_B
    sz2 = 0 ! number required for ptr_ind
    sz4 = 0 ! number required for gpu_perm
    sz5 = 0 ! number required for gpu_rdata
    sz6 = 0 ! number required for gpu_mdata
    sz7 = 0 ! number required for gpu_mbfdata
    sz8 = 0 ! number required for gpu_r
    do li = lvlptr(lev), lvlptr(lev+1)-1
       node = lvllist(li)
       ndelay = nodes(node)%ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       sz = sz + blkm
       sz2 = sz2 + blkn
       sz4 = sz4 + blkn
       sz5 = sz5 + (blkm-1)/(32*BLOCK_SIZE) + 2
       sz6 = sz6 + ((blkm-1)/32+1) * ((blkn-1)/32+1)
       sz7 = sz7 + (blkm - 1)/(BLOCK_SIZE*(MNF_BLOCKS - 1)) + 1
       if ((blkm .eq. blkn) .or. (lvlptr(lev+1)-lvlptr(lev) .eq. 1)) &
            sz8 = max(sz8, blkm*(blkm+0_C_SIZE_T))
    end do
    sz2 = max(int(sz2), (lvlptr(lev+1)-lvlptr(lev))*BLOCK_SIZE)
    sz = sz*BLOCK_SIZE
    sz3 = lvlptr(lev+1)-lvlptr(lev) ! number of blocks for msdata et al
    lgpu = max(lgpu, &
         max( &
         sz*C_SIZEOF(real_dummy)    + & ! ptr_B
         sz2*C_SIZEOF(int_dummy)    + & ! ptr_ind
         8*C_SIZEOF(int_dummy)      + & ! gpu_aux
         sz4*C_SIZEOF(int_dummy)    + & ! gpu_perm
         sz5*4*C_SIZEOF(int_dummy)  + & ! gpu_rdata
         sz5*C_SIZEOF(mr_dummy)     + & ! gpu_rdata
         sz6*2*C_SIZEOF(int_dummy)  + & ! gpu_mdata
         sz6*C_SIZEOF(me_dummy)     + & ! gpu_mdata
         sz3*C_SIZEOF(mnft_dummy)   + & ! gpu_mnfdata
         sz3*C_SIZEOF(int_dummy)    + & ! gpu_stat
         sz7*C_SIZEOF(mbft_dummy)   + & ! gpu_mbfdata
         sz8*C_SIZEOF(real_dummy),    & ! gpu_r
         sz3*C_SIZEOF(cdt_dummy)      & ! gpu_csdata
         ) &
         )

    ! Space for gpu_msdata in form_contrib()
    sz = 0 ! spac for gpu_msdata
    do li = lvlptr(lev), lvlptr(lev+1)-1
       node = lvllist(li)
       ndelay = nodes(node)%ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       bx = (blkm-blkn-1)/32 + 1
       sz = sz + (bx*(bx+1))/2
    end do
    lgpu = max(lgpu, sz*C_SIZEOF(msyrt_dymmy))

    ! Space for cpdata, blkdata and sync in assemble_contrib()
    sz = 0
    sz2 = 0
    do li = lvlptr(lev), lvlptr(lev+1)-1
       node = lvllist(li)
       sz = sz + child_ptr(node+1)-child_ptr(node)
       do ci = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(ci)
          if (asminf(cnode)%npassed-asminf(cnode)%npassl .le. 0) cycle
          bx = (asminf(cnode)%npassed-asminf(cnode)%npassl-1) / &
               HOGG_ASSEMBLE_TX + 1
          by = (asminf(cnode)%npassed-asminf(cnode)%npassl-1) / &
               HOGG_ASSEMBLE_TY + 1
          sz2 = sz2 + bx*by
       end do
    end do
    lgpu = max(lgpu, sz*C_SIZEOF(acpt_dummy) + sz2*C_SIZEOF(abt_dummy) + &
         (1+sz)*C_SIZEOF(int_dummy))

    ! Allow for alignment retification of up to 10 pointers beyond the first
    lgpu = lgpu + 10*256
  end function level_gpu_work_size

  ! Assigns nodes to level lists, with level 1 being the closest to the leaves
  subroutine assign_nodes_to_levels(nnodes, sparent, gpu_contribs, num_levels, &
       lvlptr, lvllist, st)
    integer, intent(in) :: nnodes
    integer, dimension(*), intent(in) :: sparent
    type(C_PTR), dimension(*), intent(in) :: gpu_contribs
    integer, intent(out) :: num_levels
    integer, dimension(*), intent(out) :: lvlptr
    integer, dimension(*), intent(out) :: lvllist
    integer, intent(out) :: st

    integer :: node, lvl, j
    integer, dimension(:), allocatable :: level ! level of node
    integer, dimension(:), allocatable :: lvlcount

    logical, dimension(:), allocatable :: dead

    allocate(level(nnodes+1), lvlcount(nnodes+1), dead(nnodes+1), stat=st)
    if (st .ne. 0) return

    ! Find level of each node, with level 1 being a root
    num_levels = 1
    dead(:) = .false.
    lvlcount(:) = 0
    level(nnodes+1) = 0
    do node = nnodes, 1, -1
       j = min(sparent(node), nnodes+1) ! handle parents outwith subtree
       ! Mark as dead nodes in subtrees rooted at nodes with defined contrib
       if (C_ASSOCIATED(gpu_contribs(node))) dead(node) = .true.
       if (dead(j)) dead(node) = .true.
       if (dead(node)) cycle
       ! Record non-dead nodes
       lvl = level(j) + 1
       level(node) = lvl
       lvlcount(lvl) = lvlcount(lvl) + 1
       num_levels = max(num_levels, lvl)
    end do

    ! Remove any virtual root we have
    dead(nnodes+1) = .true.

    ! Setup pointers, note that the final level we want for each node is
    ! num_levels-level(node)+1 as we number from the bottom, not the top!
    ! We use lvlptr(i+1) as the insert position for level i
    lvlptr(1:2) = 1
    do lvl = 2, num_levels
       lvlptr(lvl+1) = lvlptr(lvl) + lvlcount(num_levels-(lvl-1)+1)
    end do

    ! Finally assign nodes to levels
    do node = 1, nnodes
       if (dead(node)) cycle
       lvl = num_levels - level(node) + 1
       lvllist(lvlptr(lvl+1)) = node
       lvlptr(lvl+1) = lvlptr(lvl+1) + 1
    end do
  end subroutine assign_nodes_to_levels

  subroutine init_L_with_A(stream, lev, lvlptr, lvllist, nodes, ncb, level_size, &
       nptr, rptr, gpu_nlist, gpu_rlist, ptr_val, ptr_levL, &
       gwork, st, cuda_error, ptr_scale)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: lev
    integer, dimension(*), intent(in) :: lvlptr
    integer, dimension(*), intent(in) :: lvllist
    type(node_type), dimension(*), intent(in) :: nodes
    integer, intent(in) :: ncb
    integer(long), intent(in) :: level_size
    integer(long), dimension(*), intent(in) :: nptr
    integer(long), dimension(*), intent(in) :: rptr
    type(C_PTR), intent(in) :: gpu_nlist
    type(C_PTR), intent(in) :: gpu_rlist
    type(C_PTR), intent(in) :: ptr_val
    type(C_PTR), intent(in) :: ptr_levL ! target data is altered
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error
    type(C_PTR), optional, intent(in) :: ptr_scale

    integer :: llist, node, i
    type(load_nodes_type), dimension(:), allocatable, target :: lndata
    type(C_PTR) :: gpu_lndata
    real(wp) :: dummy_real

    st = 0
    cuda_error = 0

    ! Initialize data for cuda_load_nodes and copy to GPU
    allocate(lndata(ncb), stat=st)
    if (st .ne. 0) return
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)
       i = llist - lvlptr(lev) + 1
       lndata(i)%offn = nptr(node) - 1
       lndata(i)%nnz = nptr(node + 1) - nptr(node)
       lndata(i)%lda = int(rptr(node + 1) - rptr(node))
       lndata(i)%offr = rptr(node) - 1
       lndata(i)%ldl = int(rptr(node + 1) - rptr(node)) + nodes(node)%ndelay
       lndata(i)%lcol = c_ptr_plus( nodes(node)%gpu_lcol, &
            nodes(node)%ndelay * (1+lndata(i)%ldl) * C_SIZEOF(dummy_real) )
    end do
    gpu_lndata = custack_alloc(gwork, ncb*C_SIZEOF(lndata(1)))
    cuda_error = cudaMemcpyAsync_H2D(gpu_lndata, C_LOC(lndata), &
         ncb*C_SIZEOF(lndata(1)), stream)
    if (cuda_error .ne. 0) return

    ! Initialize frontal matrices to 0
    cuda_error = &
         cudaMemsetAsync(ptr_levL, 0, level_size*C_SIZEOF(dummy_real), stream)
    if (cuda_error .ne. 0) return
 
    ! Store values of A into front for fully summed variables
    if (present(ptr_scale)) then
       call load_nodes_sc(stream, ncb, gpu_lndata, gpu_nlist, gpu_rlist,&
            ptr_scale, ptr_val)
    else
       call load_nodes( stream, ncb, gpu_lndata, gpu_nlist, ptr_val )
    end if
    call custack_free(gwork, ncb*C_SIZEOF(lndata(1)))
  end subroutine init_L_with_A

  subroutine form_contrib(stream, lev, lvlptr, nodes, lvllist, off_LDLT,&
       sptr, rptr, ptr_levLDLT, gwork, st, cuda_error, gpu_ldcol)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: lev
    integer, dimension(*), intent(in) :: lvlptr
    type(node_type), dimension(*), intent(inout) :: nodes
    integer, dimension(*), intent(in) :: lvllist
    integer(long), dimension(*), intent(in) :: off_LDLT
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    type(C_PTR), intent(in) :: ptr_levLDLT
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error
    type(C_PTR), dimension(*), optional, intent(in) :: gpu_ldcol

    type(multisyrk_type), dimension(:), allocatable, target :: msdata
    type(C_PTR) :: gpu_msdata
   
    integer :: i, j, k, m
    integer :: llist, ncb, nn
    integer :: ndelay, nelim, blkm, blkn, node
    real(wp) :: dummy_real

    cuda_error = 0
    st = 0

    ncb = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)
       ndelay = nodes(node)%ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       m = blkm - blkn
       k = (m - 1)/32 + 1
       ncb = ncb + (k*(k + 1))/2
    end do
    allocate(msdata(ncb), stat=st)
    if (st .ne. 0) return

    ncb = 0
    nn = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       i = llist - lvlptr(lev) + 1
       node = lvllist(llist)
       ndelay = nodes(node)%ndelay
       nelim = nodes(node)%nelim
       blkn = sptr(node + 1) - sptr(node) + ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       m = blkm - blkn
       nn = nn + 1
       k = (m - 1)/32 + 1
       k = (k*(k + 1))/2
       do j = 1, k
          msdata(ncb+j)%first = ncb
          msdata(ncb+j)%lval = c_ptr_plus(nodes(node)%gpu_lcol, &
               blkn*C_SIZEOF(dummy_real))
          if (present(gpu_ldcol)) then
             msdata(ncb+j)%ldval = c_ptr_plus(gpu_ldcol(i), &
                  blkn*C_SIZEOF(dummy_real)) ! LD in indef case
          else
             msdata(ncb+j)%ldval = c_ptr_plus(nodes(node)%gpu_lcol, &
                  blkn*C_SIZEOF(dummy_real)) ! L in posdef case
          end if
          msdata(ncb+j)%offc = off_LDLT(node)
          msdata(ncb+j)%n = m
          msdata(ncb+j)%k = nelim
          msdata(ncb+j)%lda = blkm
          msdata(ncb+j)%ldb = blkm
       end do
       ncb = ncb + k
    end do

    if (ncb .gt. 0) then
       gpu_msdata = custack_alloc(gwork, ncb*C_SIZEOF(msdata(1)))
       cuda_error = cudaMemcpyAsync_H2D(gpu_msdata, C_LOC(msdata), &
            ncb*C_SIZEOF(msdata(1)), stream)
       if (cuda_error .ne. 0) return
       call cuda_multidsyrk_low_col( stream, ncb, gpu_msdata, ptr_levLDLT )
       call custack_free(gwork, ncb*C_SIZEOF(msdata(1))) ! gpu_msdata
    end if
  end subroutine form_contrib

  subroutine factor_posdef(stream, lev, lvlptr, nodes, lvllist, sptr, rptr, &
       ptr_levL, cublas_handle, stats, gwork)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: lev
    integer, dimension(*), intent(in) :: lvlptr
    type(node_type), dimension(*), intent(inout) :: nodes
    integer, dimension(*), intent(in) :: lvllist
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    type(C_PTR), intent(in) :: ptr_levL
    type(C_PTR), intent(in) :: cublas_handle
    type(thread_stats), intent(inout) :: stats
    type(cuda_stack_alloc_type), intent(inout) :: gwork

    type(multisymm_type), dimension(:), allocatable, target :: msdata
    type(C_PTR) :: gpu_msdata

    type(C_PTR) :: ptr_B
    integer :: i, j
    integer :: ncb, llist, rmax, rtot
    integer :: blkm, blkn, node

    type(C_PTR) :: ptr_L

    integer, dimension(:), allocatable :: node_m, node_n
    type(C_PTR), dimension(:), allocatable :: node_lcol
    
    integer, allocatable, target :: nelm(:) ! work arrays
    real(wp) :: dummy_real

    ncb = lvlptr(lev + 1) - lvlptr(lev)
    allocate(nelm(ncb), msdata(ncb), node_m(ncb), node_n(ncb), node_lcol(ncb), &
         stat=stats%st)
    if (stats%st .ne. 0) return

    !
    ! Copy lower triangle into upper triangle so we can use access (i,j) or
    ! (j,i) to get the same number while pivoting.
    !
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)        
       i = llist - lvlptr(lev) + 1
       msdata(i)%lcol = nodes(node)%gpu_lcol
       msdata(i)%ncols = sptr(node + 1) - sptr(node)
       msdata(i)%nrows = int(rptr(node + 1) - rptr(node))
    end do
    gpu_msdata = custack_alloc(gwork, ncb*C_SIZEOF(msdata(1)))
    stats%cuda_error = &
         cudaMemcpyAsync_h2d(gpu_msdata, C_LOC(msdata), ncb*C_SIZEOF(msdata(1)), stream)
    if (stats%cuda_error .ne. 0) return
    call multisymm(stream, ncb, gpu_msdata)
    call custack_free(gwork, ncb*C_SIZEOF(msdata(1))) ! gpu_msdata

    !
    ! Factor several nodes simultaneously
    !
    ! Setup
    ncb = 0
    rmax = 0
    rtot = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)        
       i = llist - lvlptr(lev) + 1
       blkn = sptr(node + 1) - sptr(node)
       blkm = int(rptr(node + 1) - rptr(node))
       if (blkn .lt. blkm) then
          ncb = ncb + 1
          node_m(ncb) = blkm
          node_n(ncb) = blkn
          node_lcol(ncb) = nodes(node)%gpu_lcol
          rtot = rtot + blkm
       else if (blkn .eq. blkm) then
          rmax = max(rmax, blkm)
       end if
    end do
    rmax = max(rmax, rtot)
    ptr_B = custack_alloc(gwork, rmax*BLOCK_SIZE*C_SIZEOF(dummy_real))
    if (ncb .gt. 0) then
       ! Perform simultaneous LLT factorization
       call multinode_llt(stream, ncb, node_m, node_n, node_lcol,                &
            cublas_handle, ptr_levL, ptr_B, BLOCK_SIZE, nelm, stats%flag, gwork, &
            stats%st, stats%cuda_error, stats%cublas_error)
       if (stats%st .ne. 0) return
       if ((stats%cuda_error .ne. 0) .or. (stats%cublas_error .ne. 0)) return
       if (stats%flag .lt. 0) return

       ! Store outcome of factorization
       ncb = 0
       do llist = lvlptr(lev), lvlptr(lev + 1) - 1
          node = lvllist(llist)
          blkn = sptr(node + 1) - sptr(node)
          blkm = int(rptr(node + 1) - rptr(node))
          if (blkn .lt. blkm) then
             ncb = ncb + 1
             nodes(node)%nelim = nelm(ncb)
             do j = blkm, blkm-nelm(ncb)+1, -1
                stats%num_factor = stats%num_factor + j
                stats%num_flops = stats%num_flops + j**2_long
             end do
          end if
       end do
    end if

    !
    ! Factor root nodes
    !
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)
       ptr_L = nodes(node)%gpu_lcol
       node = lvllist(llist)
       blkn = sptr(node + 1) - sptr(node)
       blkm = int(rptr(node + 1) - rptr(node))
       ! Positive-definite, LL^T factorization, no pivoting
       if (blkm .eq. blkn) then
          call node_llt(stream, blkm, blkn, ptr_L, blkm, ptr_B, BLOCK_SIZE, &
               cublas_handle, stats%flag, gwork, stats%cuda_error,          &
               stats%cublas_error)
          if ((stats%cuda_error .ne. 0) .or. (stats%cublas_error .ne. 0)) return
          if (stats%flag .lt. 0) return
          nodes(node)%nelim = blkn
       end if
    end do

    call custack_free(gwork, rmax*BLOCK_SIZE*C_SIZEOF(dummy_real)) ! ptr_B
  end subroutine factor_posdef

  subroutine collect_stats_indef(stream, lev, lvlptr, lvllist, nodes, &
       sptr, rptr, stats, gwork, gpu_custats)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: lev
    integer, dimension(*), intent(in) :: lvlptr
    integer, dimension(*), intent(in) :: lvllist
    type(node_type), dimension(*), intent(inout) :: nodes
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    type(thread_stats), intent(inout) :: stats
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    type(C_PTR), intent(in) :: gpu_custats

    type(cstat_data_type), dimension(:), allocatable, target :: csdata
    type(C_PTR) :: gpu_csdata

    integer :: llist, llvlptr
    integer :: node, blkm, blkn, ndelay
    real(wp) :: dummy_real

    llvlptr = lvlptr(lev+1)-lvlptr(lev)
    allocate(csdata(llvlptr), stat=stats%st)
    if (stats%st .ne. 0) return
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)
       ndelay = nodes(node)%ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       csdata(llist-lvlptr(lev)+1)%nelim = nodes(node)%nelim
       csdata(llist-lvlptr(lev)+1)%dval = c_ptr_plus(nodes(node)%gpu_lcol, &
            blkm*(blkn+0_long)*C_SIZEOF(dummy_real))
    end do

    gpu_csdata = custack_alloc(gwork, llvlptr*C_SIZEOF(csdata(1)))
    stats%cuda_error = cudaMemcpyAsync_H2D(gpu_csdata, C_LOC(csdata), &
         llvlptr*C_SIZEOF(csdata(1)), stream)
    if (stats%cuda_error .ne. 0) return

    call cuda_collect_stats(stream, size(csdata), gpu_csdata, gpu_custats)

    call custack_free(gwork, llvlptr*C_SIZEOF(csdata(1))) ! gpu_csdata
  end subroutine collect_stats_indef

  ! Factorize a nodal matrix (not contrib block)
  subroutine factor_indef( stream, lev, lvlptr, nnodes, nodes, lvllist, sparent, &
       sptr, rptr, level_height, level_width, delta, eps, gpu_ldcol, gwork, &
       cublas_handle, options, stats, gpu_custats)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: lev
    integer, dimension(*), intent(in) :: lvlptr
    integer, intent(in) :: nnodes
    type(node_type), dimension(*), intent(inout) :: nodes
    integer, dimension(*), intent(in) :: lvllist
    integer, dimension(*), intent(in) :: sparent
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    integer, intent(in) :: level_height
    integer, intent(in) :: level_width
    real(wp), intent(inout) :: delta, eps
    type(C_PTR), dimension(:), intent(in) :: gpu_ldcol
    type(C_PTR), intent(in) :: cublas_handle
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    type(ssids_options), intent(in) :: options
    type(thread_stats), intent(inout) :: stats
    type(C_PTR), intent(in) :: gpu_custats

    type(multisymm_type), dimension(:), allocatable, target :: msdata
    type(C_PTR) :: gpu_msdata

    type(multiswap_type), dimension(:), allocatable, target :: swapdata
    type(C_PTR) :: gpu_swapdata

    type(C_PTR) :: ptr_ind, ptr_B
    integer :: ind_len, B_len
    integer :: i, j, k, p
    integer :: ncb, last_ln, llist, maxr
    integer :: ndelay, blkm, blkn, nelim, parent, node, dif
    integer, dimension(:), pointer :: lperm

    type(C_PTR) :: ptr_D, ptr_L, ptr_LD

    integer, dimension(:), allocatable :: node_m, node_n, node_skip
    type(C_PTR), dimension(:), allocatable :: node_lcol, node_ldcol

    integer, allocatable, target :: iwork(:), perm(:), nelm(:) ! work arrays
    real(wp) :: dummy_real
    integer(C_INT) :: dummy_int

    ncb = lvlptr(lev + 1) - lvlptr(lev)
    allocate(perm(level_width), nelm(ncb), msdata(ncb), &
         node_m(ncb), node_n(ncb), node_lcol(ncb), node_ldcol(ncb), &
         node_skip(ncb), swapdata(ncb), stat=stats%st )
    if (stats%st .ne. 0) return

    ! Initialize variables to avoid warnings
    last_ln = 0

    !
    ! Copy lower triangle into upper triangle so we can use access (i,j) or
    ! (j,i) to get the same number while pivoting.
    !
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)        
       i = llist - lvlptr(lev) + 1
       ndelay = nodes(node)%ndelay
       msdata(i)%lcol = nodes(node)%gpu_lcol
       msdata(i)%ncols = sptr(node + 1) - sptr(node) + ndelay
       msdata(i)%nrows = int(rptr(node + 1) - rptr(node)) + ndelay
    end do
    gpu_msdata = custack_alloc(gwork, ncb*C_SIZEOF(msdata(1)))
    stats%cuda_error = &
         cudaMemcpyAsync_H2D(gpu_msdata, C_LOC(msdata), ncb*C_SIZEOF(msdata(1)), &
         stream)
    if (stats%cuda_error .ne. 0) return
    call multisymm(stream, ncb, gpu_msdata)
    call custack_free(gwork, ncb*C_SIZEOF(msdata(1)))
    ! Note: Done with gpu_msdata

    !
    ! Swap delays to end
    !
    ncb = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)
       i = llist - lvlptr(lev) + 1
       ndelay = nodes(node)%ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       if ((ndelay .gt. 0) .and. (blkn .gt. 1)) then
          blkm = int(rptr(node + 1) - rptr(node)) + ndelay
          k = min(ndelay, blkn - ndelay)
          ncb = ncb + 1
          swapdata(ncb)%nrows = blkm
          swapdata(ncb)%ncols = blkn
          swapdata(ncb)%k = k
          swapdata(ncb)%lcol = nodes(node)%gpu_lcol
          swapdata(ncb)%lda = blkm
          swapdata(ncb)%off = blkn - k
          lperm => nodes(node)%perm
          do i = 1, k
             j = blkn - k + i
             p = lperm(i)
             lperm(i) = lperm(j)
             lperm(j) = p
          end do
       end if
    end do
    if (ncb .gt. 0) then
       gpu_swapdata = custack_alloc(gwork, ncb*C_SIZEOF(swapdata(1)))
       stats%cuda_error = cudaMemcpyAsync_H2D(gpu_swapdata, C_LOC(swapdata), &
            ncb*C_SIZEOF(swapdata(1)), stream)
       if (stats%cuda_error .ne. 0) return
       call swap_ni2Dm( stream, ncb, gpu_swapdata )
       call custack_free(gwork, ncb*C_SIZEOF(swapdata(1)))
    end if

    !
    ! Factor several nodes simultaneously
    !
    ! Setup
    p = 0
    k = 0
    ncb = 0
    maxr = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)        
       i = llist - lvlptr(lev) + 1
       ndelay = nodes(node)%ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       if (blkn .lt. blkm) then
          ncb = ncb + 1
          lperm => nodes(node)%perm
          node_m(ncb) = blkm
          node_n(ncb) = blkn
          node_lcol(ncb) = nodes(node)%gpu_lcol
          node_ldcol(ncb) = gpu_ldcol(i)
          node_skip(ncb) = k
          do j = 1, blkn
             perm(k + j) = lperm(j)
          end do
          k = k + blkn
          maxr = max(maxr, blkm)
          p = p + blkm
       end if
    end do
  
    ind_len = max(level_width, ncb*BLOCK_SIZE)
    B_len = max(p, level_height)*BLOCK_SIZE

    ptr_ind = custack_alloc(gwork, ind_len*C_SIZEOF(dummy_int))
    ptr_B = custack_alloc(gwork, B_len*C_SIZEOF(dummy_real))

    last_ln = ncb

    if (ncb .gt. 1) then

       ! Perform simultaneous factorization of several nodes
       call multinode_ldlt(stream, ncb, node_m, node_n, node_lcol, node_ldcol,    &
            node_skip, ptr_B, ptr_ind, delta, eps, BLOCK_SIZE, perm, nelm, gwork, &
            cublas_handle, stats%st, stats%cuda_error, stats%cublas_error)
       if ((stats%st .ne. 0) .or. (stats%cuda_error .ne. 0) .or. &
            (stats%cublas_error .ne. 0)) return

       ! Store outcome of factorization
       ncb = 0
       k = 0
       do llist = lvlptr(lev), lvlptr(lev + 1) - 1
          node = lvllist(llist)
          ndelay = nodes(node)%ndelay
          blkn = sptr(node + 1) - sptr(node) + ndelay
          blkm = int(rptr(node + 1) - rptr(node)) + ndelay
          parent = min(sparent(node), nnodes+1)
          lperm => nodes(node)%perm
          if (blkn .lt. blkm) then
             ncb = ncb + 1
             nelim = nelm(ncb)
             do j = 1, blkn
                lperm(j) = perm(k + j)
             end do
             k = k + blkn
             nodes(node)%nelim = nelim
             dif = blkn - nelim
             !$omp atomic update
             nodes(parent)%ndelay = nodes(parent)%ndelay + dif
             !$omp end atomic
             stats%num_delay &
                  = stats%num_delay + blkn - nelim
             do j = blkm, blkm-nelim+1, -1
                stats%num_factor = stats%num_factor + j
                stats%num_flops = stats%num_flops + j**2_long
             end do
          end if
       end do
    end if
   
    allocate(iwork(2*level_width), stat=stats%st)
    if (stats%st .ne. 0) return
   
    !
    ! Factor remaining nodes one by one
    !
    ncb = 0
    k = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       i = llist - lvlptr(lev) + 1
       node = lvllist(llist)
       ptr_L = nodes(node)%gpu_lcol
       ptr_LD = gpu_ldcol(i)

       node = lvllist(llist)
       ndelay = nodes(node)%ndelay
       blkn = sptr(node + 1) - sptr(node) + ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       parent = min(sparent(node), nnodes+1)
       lperm => nodes(node)%perm
   
       ptr_D = c_ptr_plus( ptr_L, blkm*blkn*C_SIZEOF(dummy_real) )

       ! Indefinite, LDL^T factorization, with pivoting
       if ((blkm .eq. blkn) .or. (last_ln .eq. 1)) then

          call node_ldlt(stream, blkm, blkn, ptr_L, ptr_LD, blkm, ptr_D, ptr_B, &
               ptr_ind, delta, eps, BLOCK_SIZE, lperm, iwork, nelim, gwork, &
               cublas_handle, stats%cuda_error, stats%cublas_error)
          if ((stats%cuda_error .ne. 0) .or. (stats%cublas_error .ne. 0)) return

          if ((blkm .eq. blkn) .and. (nelim .lt. blkn)) then
             if (options%action) then
                stats%flag = SSIDS_WARNING_FACT_SINGULAR
             else
                stats%flag = SSIDS_ERROR_SINGULAR
                return
             end if
          end if

          ! Record delays
          nodes(node)%nelim = nelim
          if (blkn .lt. blkm) then
             dif = blkn - nelim
             !$omp atomic update
             nodes(parent)%ndelay = nodes(parent)%ndelay + dif
             !$omp end atomic
          end if
          stats%num_delay = stats%num_delay + blkn - nelim
          do j = blkm, blkm-nelim+1, -1
             stats%num_factor = stats%num_factor + j
             stats%num_flops = stats%num_flops + j**2_long
          end do

       end if

    end do

    call custack_free(gwork, B_len*C_SIZEOF(dummy_real)) ! ptr_B
    call custack_free(gwork, ind_len*C_SIZEOF(dummy_int)) ! ptr_ind

    call collect_stats_indef(stream, lev, lvlptr, lvllist, nodes, &
         sptr, rptr, stats, gwork, gpu_custats)
  end subroutine factor_indef

  subroutine setup_assemble_contrib(stream, lev, lvlptr, lvllist, child_ptr, &
       child_list, sptr, rptr, asminf, gpu_ccval, gpu_contribs, ptr_levLDLT, &
       off_LDLT, gpu_rlist_direct, gwork, ncp, gpu_cpdata, nblk, &
       gpu_blkdata, gpu_sync, st, cuda_error)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: lev
    integer, intent(in) :: lvlptr(*) ! Pointers into lvllist for level
    integer, intent(in) :: lvllist(*) ! Nodes at level lev are given by:
      ! lvllist(lvlptr(lev):lvlptr(lev+1)-1)
    integer, intent(in) :: child_ptr(*) ! Pointers into child_list for node
    integer, intent(in) :: child_list(*) ! Children of node node are given by:
      ! child_list(child_ptr(node):child_ptr(node+1)-1)
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    type(asmtype), dimension(:), intent(in) :: asminf ! Assembly info
    type(C_PTR), intent(in) :: gpu_ccval ! GPU (*gpu_ccval) points to previous
      ! level's contribution blocks
    type(C_PTR), dimension(*), intent(in) :: gpu_contribs ! For each node, is
      ! either NULL or points to a contribution block arriving from a subtree
    type(C_PTR), intent(in) :: ptr_levLDLT
    integer(long), intent(in) :: off_LDLT(*) ! Offsets for children
    type(C_PTR), intent(in) :: gpu_rlist_direct ! GPU pointer to rlist_direct
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    integer, intent(out) :: ncp ! Number of child-parent pairs
    type(C_PTR), intent(out) :: gpu_cpdata ! Ouput child-parent info
    integer, intent(out) :: nblk ! Number of blocks
    type(C_PTR), intent(out) :: gpu_blkdata ! Output block-by-block info
    type(C_PTR), intent(out) :: gpu_sync
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    type(assemble_cp_type), dimension(:), allocatable, target :: cpdata
    type(assemble_blk_type), dimension(:), allocatable, target :: blkdata

    integer :: child, maxchild
    integer :: cpi, bi, blki, blkj
    integer :: j, k, m, npassl, blkm, blkn
    integer :: llist, node, cnode, npassed, bx, by
    integer :: blk
    real(wp) :: dummy_real
    integer(C_INT) :: dummy_int

    ! Ensure all return values initialized (mainly to prevent warnings)
    ncp = 0
    gpu_cpdata = C_NULL_PTR
    nblk = 0
    gpu_blkdata = C_NULL_PTR
    gpu_sync = C_NULL_PTR
    st = 0
    cuda_error = 0

    ! Count number of children with work to do
    ncp = 0
    nblk = 0
    maxchild = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)
       blkm = int(rptr(node + 1) - rptr(node))
       blkn = sptr(node + 1) - sptr(node)
       m = blkm - blkn
       if (m .gt. 0) then
          do child = child_ptr(node), child_ptr(node+1)-1
             cnode = child_list(child)
             npassed = asminf(cnode)%npassed
             npassl = asminf(cnode)%npassl
             if ((npassed-npassl) .gt. 0) then
                ncp = ncp + 1
                bx = (npassed-npassl-1) / HOGG_ASSEMBLE_TX + 1
                by = (npassed-npassl-1) / HOGG_ASSEMBLE_TY + 1
                nblk = nblk + &
                     calc_blks_lwr(bx, by, HOGG_ASSEMBLE_TX, HOGG_ASSEMBLE_TY)
             end if
          end do
       end if
       maxchild = max(maxchild, child_ptr(node+1)-child_ptr(node))
    end do

    ! Fill in child-parent information
    allocate(cpdata(ncp), stat=st)
    if (st .ne. 0) return
    cpi = 1
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       k = llist - lvlptr(lev) + 1
       node = lvllist(llist)
       blkm = int(rptr(node + 1) - rptr(node))
       blkn = sptr(node + 1) - sptr(node)
       blk = 0
       do child = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(child)
          npassed = asminf(cnode)%npassed
          npassl = asminf(cnode)%npassl
          ! We are only doing the contribution block of the parent
          if ((npassed-npassl) .le. 0) cycle
          cpdata(cpi)%cm = npassed - npassl
          cpdata(cpi)%cn = npassed - npassl
          cpdata(cpi)%ldp = blkm - blkn
          ! Note: row rlist(i) of parent is row rlist(i)-blkn of contribution blk
          ! so we alter pval to account for this
          cpdata(cpi)%pval = c_ptr_plus(ptr_levLDLT, &
               (off_LDLT(node) - blkn*(1+cpdata(cpi)%ldp))*C_SIZEOF(dummy_real) )
          cpdata(cpi)%ldc = asminf(cnode)%npassed
          cpdata(cpi)%cvoffset = off_LDLT(cnode) + &
               npassl * (1+cpdata(cpi)%ldc)
          if (C_ASSOCIATED(gpu_contribs(cnode))) then
             cpdata(cpi)%cv = c_ptr_plus(gpu_contribs(cnode), &
                  (npassl * (1+cpdata(cpi)%ldc)) * C_SIZEOF(dummy_real))
          else
             cpdata(cpi)%cv = c_ptr_plus(gpu_ccval, &
                  (off_LDLT(cnode) + npassl * (1+cpdata(cpi)%ldc)) * &
                  C_SIZEOF(dummy_real))
          end if
          cpdata(cpi)%rlist_direct = c_ptr_plus(gpu_rlist_direct, &
               (asminf(cnode)%offset + npassl)*C_SIZEOF(dummy_int) &
               )
          cpdata(cpi)%sync_offset = cpi-1 - 1 ! 0-indexed
          cpdata(cpi)%sync_wait_for = blk
          ! Calulate how many blocks next iteration needs to wait for
          bx = (cpdata(cpi)%cm-1) / HOGG_ASSEMBLE_TX + 1
          by = (cpdata(cpi)%cn-1) / HOGG_ASSEMBLE_TY + 1
          blk = calc_blks_lwr(bx, by, HOGG_ASSEMBLE_TX, HOGG_ASSEMBLE_TY)
          cpi = cpi + 1
       end do
    end do

    ! Setup block information: do all first children, then all second, etc.
    ! This ensures maximum parallelism can be exploited
    allocate(blkdata(nblk), stat=st)
    if (st .ne. 0) return
    bi = 1
    do child = 1, maxchild
       cpi = 1
       do llist = lvlptr(lev), lvlptr(lev + 1) - 1
          node = lvllist(llist)
          do j = child_ptr(node), child_ptr(node+1)-1
             cnode = child_list(j)
             ! We are only doing the contribution block of the parent
             if (asminf(cnode)%npassed-asminf(cnode)%npassl .le. 0) cycle
             if (j-child_ptr(node)+1 .eq. child) then
                bx = (cpdata(cpi)%cm-1) / HOGG_ASSEMBLE_TX + 1
                by = (cpdata(cpi)%cn-1) / HOGG_ASSEMBLE_TY + 1
                do blkj = 0, by-1
                   do blki = 0, bx-1
                      if ((blki+1)*HOGG_ASSEMBLE_TX .lt. (blkj+1)*HOGG_ASSEMBLE_TY) &
                           cycle ! Entirely in upper triangle
                      blkdata(bi)%cp = cpi-1
                      blkdata(bi)%blk = blkj*bx + blki
                      bi = bi + 1
                   end do
                end do
             end if
             cpi = cpi + 1
          end do
       end do
    end do

    gpu_cpdata = custack_alloc(gwork, ncp*C_SIZEOF(cpdata(1)))
    gpu_blkdata = custack_alloc(gwork, nblk*C_SIZEOF(blkdata(1)))
    gpu_sync = custack_alloc(gwork, (1+ncp)*C_SIZEOF(dummy_int))

    cuda_error = cudaMemcpyAsync_H2D(gpu_cpdata, C_LOC(cpdata), &
         ncp*C_SIZEOF(cpdata(1)), stream)
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpyAsync_H2D(gpu_blkdata, C_LOC(blkdata), &
         nblk*C_SIZEOF(blkdata(1)), stream)
    if (cuda_error .ne. 0) return
  end subroutine setup_assemble_contrib

  ! Return number of blocks not entirely in upper triangle
  ! For nx x ny block matrix with block size nbx x nby
  integer function calc_blks_lwr(nx, ny, nbx, nby)
    implicit none
    integer, intent(in) :: nx
    integer, intent(in) :: ny
    integer, intent(in) :: nbx
    integer, intent(in) :: nby

    integer :: i

    calc_blks_lwr = 0
    do i = 1, nx
       calc_blks_lwr = calc_blks_lwr + min(ny,(i*nbx)/nby)
    end do

  end function calc_blks_lwr

  subroutine assemble_contrib(stream, total_nch, lev, lvlptr, lvllist, &
       child_ptr, child_list, sptr, rptr, asminf, pc_size, off_LDLT,   &
       gpu_ccval, gpu_contribs, ptr_levLDLT, gpu_rlist_direct, gwork,  &
       st, cuda_error)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: total_nch
    integer, intent(in) :: lev
    integer, intent(in) :: lvlptr(*) ! Pointers into lvllist for level
    integer, intent(in) :: lvllist(*) ! Nodes at level lev are given by:
      ! lvllist(lvlptr(lev):lvlptr(lev+1)-1)
    integer, intent(in) :: child_ptr(*) ! Pointers into child_list for node
    integer, intent(in) :: child_list(*) ! Children of node node are given by:
      ! child_list(child_ptr(node):child_ptr(node+1)-1)
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    type(asmtype), dimension(:), intent(in) :: asminf ! Assembly info
    integer(long), intent(in) :: pc_size
    integer(long), dimension(*), intent(in) :: off_LDLT ! Offsets for children
    type(C_PTR), intent(in) :: gpu_ccval ! GPU (*gpu_ccval) points to previous
      ! level's contribution blocks
    type(C_PTR), dimension(*), intent(in) :: gpu_contribs ! For each node, is
      ! either NULL or points to a contribution block arriving from a subtree
    type(C_PTR), intent(in) :: ptr_levLDLT ! GPU pointer to contribution blocks
    type(C_PTR), intent(in) :: gpu_rlist_direct! GPU pointer to rlist_direct copy
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    integer :: ncp, nblk
    type(C_PTR) :: gpu_cpdata, gpu_blkdata, gpu_sync
    type(assemble_cp_type) :: act_dummy
    type(assemble_blk_type) :: abt_dummy
    integer(C_INT) :: dummy_int

    if ((total_nch .eq. 0) .or. (pc_size .eq. 0)) return ! Nothing to do

    call setup_assemble_contrib(stream, lev, lvlptr, lvllist, child_ptr, &
         child_list, sptr, rptr, asminf, gpu_ccval, gpu_contribs, ptr_levLDLT, &
         off_LDLT, gpu_rlist_direct, gwork, ncp, gpu_cpdata, nblk, &
         gpu_blkdata, gpu_sync, st, cuda_error)
    if ((st .ne. 0) .or. (cuda_error .ne. 0)) return

    call assemble(stream, nblk, 0, gpu_blkdata, ncp, gpu_cpdata, &
         gpu_ccval, ptr_levLDLT, gpu_sync)

    ! Free in reverse alloc order
    call custack_free(gwork, (1+ncp)*C_SIZEOF(dummy_int)) ! gpu_sync
    call custack_free(gwork, nblk*C_SIZEOF(abt_dummy)) ! gpu_blkdata
    call custack_free(gwork, ncp*C_SIZEOF(act_dummy)) ! gpu_cpdata
  end subroutine assemble_contrib

  subroutine setup_assemble_fully_summed(stream, total_nch, lev, lvlptr, lvllist,&
       child_ptr, child_list, nodes, asminf, sptr, rptr, gpu_ccval,             &
       gpu_contribs, off_LDLT, gpu_rlist_direct, gwork, ncp, gpu_cpdata, nblk,  &
       gpu_blkdata, ndblk, gpu_ddata, gpu_sync, st, cuda_error)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: total_nch ! Total number of children for nodes in
      ! this level
    integer, intent(in) :: lev ! Current level
    integer, intent(in) :: lvlptr(*) ! Pointers into lvllist for level
    integer, intent(in) :: lvllist(*) ! Nodes at level lev are given by:
      ! lvllist(lvlptr(lev):lvlptr(lev+1)-1)
    integer, intent(in) :: child_ptr(*) ! Pointers into child_list for node
    integer, intent(in) :: child_list(*) ! Children of node node are given by:
      ! child_list(child_ptr(node):child_ptr(node+1)-1)
    type(node_type), dimension(*), intent(in) :: nodes ! node data
    type(asmtype), dimension(*), intent(in) :: asminf
    integer, intent(in) :: sptr(*)
    integer(long), intent(in) :: rptr(*)
    type(C_PTR), intent(in) :: gpu_ccval ! GPU (*ptr_ccval) points to previous
      ! level's contribution blocks
    type(C_PTR), dimension(*), intent(in) :: gpu_contribs ! For each node, is
      ! either NULL or points to a contribution block arriving from a subtree
    integer(long), intent(in) :: off_LDLT(*) ! Offsets for children
    type(C_PTR), intent(in) :: gpu_rlist_direct ! GPU pointer to rlist_direct
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    integer, intent(out) :: ncp ! Number of child-parent pairs
    type(C_PTR), intent(out) :: gpu_cpdata ! Ouput child-parent info
    integer, intent(out) :: nblk ! Number of blocks
    type(C_PTR), intent(out) :: gpu_blkdata ! Output block-by-block info
    integer, intent(out) :: ndblk
    type(C_PTR), intent(out) :: gpu_ddata
    type(C_PTR), intent(out) :: gpu_sync
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    integer :: i, bi, ni, blk, bx, by, child, ci, cnode, cpi, node, blki, blkj
    integer :: ndelay, ldp, maxchild, blkm
    integer :: cndelay, cnelim, cblkm, cblkn, llist, nd
    type(C_PTR) :: pval
    type(assemble_cp_type), dimension(:), allocatable, target :: cpdata
      ! child-parent data to be copied to GPU
    type(assemble_blk_type), dimension(:), allocatable, target :: blkdata
      ! block-level data to be copied to GPU
    type(assemble_delay_type), dimension(:), allocatable, target :: ddata
      ! delay data to be copied to GPU
    integer(C_INT) :: dummy_int
    real(wp) :: dummy_real
   
    cuda_error = 0
    st = 0

    ! Ensure all output data is initialized
    ncp = 0
    gpu_cpdata = C_NULL_PTR
    nblk = 0
    gpu_blkdata = C_NULL_PTR
    ndblk = 0
    gpu_ddata = C_NULL_PTR
    gpu_sync = C_NULL_PTR
    cuda_error = 0

    ! Check for trivial return
    if (total_nch .le. 0) return ! No children to worry about

    ! Count maximum number of children
    maxchild = 0
    do ni = lvlptr(lev), lvlptr(lev+1)-1
       node = lvllist(ni)
       maxchild = max(maxchild, child_ptr(node+1)-child_ptr(node))
    end do

    ! Initialize child-parent data, count number of blocks at each level
    allocate(cpdata(total_nch), stat=st)
    if (st .ne. 0) return
    cpi = 1
    nblk = 0
    do ni = lvlptr(lev), lvlptr(lev+1)-1
       node = lvllist(ni)
       i = ni - lvlptr(lev) + 1
       ndelay = nodes(node)%ndelay
       ldp = int(rptr(node+1)-rptr(node)) + ndelay ! adjusted by #delays
       pval = c_ptr_plus(nodes(node)%gpu_lcol, &
            ndelay*(1+ldp)*C_SIZEOF(dummy_real)) ! adjusted past delays
       blk = 0
       do ci = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(ci)
          cpdata(cpi)%pval = pval
          cpdata(cpi)%ldp = ldp
          cpdata(cpi)%cm = asminf(cnode)%npassed
          cpdata(cpi)%cn = asminf(cnode)%npassl
          cpdata(cpi)%ldc = asminf(cnode)%npassed
          cpdata(cpi)%cvoffset = off_LDLT(cnode)
          if (C_ASSOCIATED(gpu_contribs(cnode))) then
             cpdata(cpi)%cv = gpu_contribs(cnode)
          else
             cpdata(cpi)%cv = &
                  c_ptr_plus(gpu_ccval, off_LDLT(cnode)*C_SIZEOF(dummy_real))
          end if
          cpdata(cpi)%rlist_direct = c_ptr_plus(gpu_rlist_direct, &
               asminf(cnode)%offset*C_SIZEOF(dummy_int))
          cpdata(cpi)%sync_offset = max(0, cpi-1 - 1)
          cpdata(cpi)%sync_wait_for = blk
          bx = (cpdata(cpi)%cm-1) / HOGG_ASSEMBLE_TX + 1
          by = (cpdata(cpi)%cn-1) / HOGG_ASSEMBLE_TY + 1
          i = ci-child_ptr(node)+1
          blk = calc_blks_lwr(bx, by, HOGG_ASSEMBLE_TX, HOGG_ASSEMBLE_TY)
          nblk = nblk + blk
          cpi = cpi + 1
       end do
    end do
    ncp = size(cpdata)

    ! Initialize blkdata
    allocate(blkdata(nblk), stat=st)
    if (st .ne. 0) return
    bi = 1
    do child = 1, maxchild
       cpi = 1
       do ni = lvlptr(lev), lvlptr(lev+1)-1
          node = lvllist(ni)
          if ((child_ptr(node)+child-1) .lt. child_ptr(node+1)) then
             bx = (cpdata(cpi+child-1)%cm-1) / HOGG_ASSEMBLE_TX + 1
             by = (cpdata(cpi+child-1)%cn-1) / HOGG_ASSEMBLE_TY + 1
             do blkj = 0, by-1
                do blki = 0, bx-1
                   if (((blki+1)*HOGG_ASSEMBLE_TX) .lt. ((blkj+1)*HOGG_ASSEMBLE_TY)) &
                        cycle ! Entirely in upper triangle
                   blkdata(bi)%cp = cpi + (child-1) - 1 ! 0 indexed
                   blkdata(bi)%blk = blkj*bx + blki
                   bi = bi + 1
                end do
             end do
          end if
          cpi = cpi + child_ptr(node+1)-child_ptr(node)
       end do
    end do

    ! Initialize ddata (for copying in any delays)
    allocate(ddata(total_nch), stat=st)
    if (st .ne. 0) return
    ndblk = 0
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1
       node = lvllist(llist)
       ndelay = nodes(node)%ndelay
       blkm = int(rptr(node + 1) - rptr(node)) + ndelay
       i = llist - lvlptr(lev) + 1
       nd = 0
       do child = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(child)
          cblkm = int(rptr(cnode + 1) - rptr(cnode))
          cblkn = sptr(cnode + 1) - sptr(cnode)
          cndelay = nodes(cnode)%ndelay
          cnelim = nodes(cnode)%nelim
          if ((cblkn+cndelay) .le. cnelim) cycle ! No delays from this child
          ndblk = ndblk + 1
          ddata(ndblk)%ldd = blkm
          ddata(ndblk)%ndelay = ndelay - nd
          ddata(ndblk)%m = cblkm + cndelay - cnelim
          ddata(ndblk)%n = cblkn + cndelay - cnelim
          ddata(ndblk)%lds = cblkm + cndelay
          ddata(ndblk)%dval = c_ptr_plus(nodes(node)%gpu_lcol, &
               nd*(1_C_SIZE_T+blkm)*C_SIZEOF(dummy_real))
          ddata(ndblk)%sval = c_ptr_plus(nodes(cnode)%gpu_lcol, &
               cnelim*(1_C_SIZE_T+ddata(ndblk)%lds)*C_SIZEOF(dummy_real))
          ddata(ndblk)%roffset = rptr(cnode) + cblkn - 1
          nd = nd + ddata(ndblk)%n
       end do
    end do

    ! Copy data to GPU
    gpu_cpdata = custack_alloc(gwork, ncp*C_SIZEOF(cpdata(1)))
    gpu_blkdata = custack_alloc(gwork, nblk*C_SIZEOF(blkdata(1)))
    gpu_ddata = custack_alloc(gwork, ndblk*C_SIZEOF(ddata(1)))
    gpu_sync = custack_alloc(gwork, (ncp + 1)*C_SIZEOF(dummy_int))

    cuda_error = cudaMemcpyAsync_H2D(gpu_cpdata, C_LOC(cpdata), &
         ncp*C_SIZEOF(cpdata(1)), stream)
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpyAsync_H2D(gpu_blkdata, C_LOC(blkdata), &
         nblk*C_SIZEOF(blkdata(1)), stream)
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpyAsync_H2D(gpu_ddata, C_LOC(ddata), &
         ndblk*C_SIZEOF(ddata(1)), stream)
    if (cuda_error .ne. 0) return
  end subroutine setup_assemble_fully_summed

!
! Perform assembly for fully summed columns
!
! At a given level, we launch one kernel for each set of ith children:
!    1) Kernel that does assembly for all 1st children
!    2) Kernel that does assembly for all 2nd children
!    3) ...
! Information about a particular child-parent assembly is stored in cpdata
! For each block that gets launched we store an offset into cpdata and a
!   the subblock of that assembly this block is to perform.
  subroutine assemble_fully_summed(stream, total_nch, lev, lvlptr, lvllist, &
       nodes, gpu_ccval, gpu_contribs, ptr_levL, gpu_rlist_direct, &
       child_ptr, child_list, off_LDLT, asminf, rptr, sptr, gwork, &
       st, cuda_error)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: total_nch ! Total number of children for nodes in
      ! this level
    integer, intent(in) :: lev ! Current level
    integer, intent(in) :: lvlptr(*) ! Pointers into lvllist for level
    integer, intent(in) :: lvllist(*) ! Nodes at level lev are given by:
      ! lvllist(lvlptr(lev):lvlptr(lev+1)-1)
    integer, intent(in) :: child_ptr(*) ! Pointers into child_list for node
    integer, intent(in) :: child_list(*) ! Children of node node are given by:
      ! child_list(child_ptr(node):child_ptr(node+1)-1)
    type(asmtype), dimension(*), intent(in) :: asminf ! Assembly info
    type(C_PTR), intent(in) :: gpu_ccval ! GPU (*gpu_ccval) points to previous
      ! level's contribution blocks
    type(C_PTR), dimension(*), intent(in) :: gpu_contribs ! For each node, is
      ! either NULL or points to a contribution block arriving from a subtree
    type(C_PTR), intent(in) :: ptr_levL ! GPU (*ptr_levL) points to L storage
      ! for current level
    type(C_PTR), intent(in) :: gpu_rlist_direct! GPU pointer to rlist_direct copy
    type(node_type), intent(in) :: nodes(*)
    integer(long), intent(in) :: off_LDLT(*)
    integer(long), intent(in) :: rptr(*)
    integer, intent(in) :: sptr(*)
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    integer, dimension(:), pointer :: lperm
    integer :: cnd
    integer :: k
    integer :: cblkn
    integer :: cn
    integer :: cnode
    integer :: i, j
    integer :: llist
    integer :: nd
    integer :: node

    integer :: ncp, nblk, ndblk
    type(C_PTR) :: gpu_cpdata, gpu_blkdata, gpu_ddata, gpu_sync
    type(assemble_cp_type) :: act_dummy
    type(assemble_blk_type) :: abt_dummy
    type(assemble_delay_type) :: adt_dummy
    integer(C_INT) :: dummy_int

    if (total_nch .le. 0) return

    ! Setup data structures (allocates gpu_cpdata, gpu_blkdata)
    call setup_assemble_fully_summed(stream, total_nch, lev, lvlptr, lvllist, &
         child_ptr, child_list, nodes, asminf, sptr, rptr, &
         gpu_ccval, gpu_contribs, off_LDLT, gpu_rlist_direct, gwork, &
         ncp, gpu_cpdata, nblk, gpu_blkdata, ndblk, gpu_ddata, gpu_sync, &
         st, cuda_error)
    if ((st .ne. 0) .or. (cuda_error .ne. 0)) return

    ! Perform assembly of child contributions to fully summed columns
    call assemble(stream, nblk, 0, gpu_blkdata, ncp, gpu_cpdata, &
         gpu_ccval, ptr_levL, gpu_sync)

    ! Copy any delayed columns
    call add_delays(stream, ndblk, gpu_ddata, gpu_rlist_direct)

    ! Release memory (in reverse order of alloc)
    call custack_free(gwork, (ncp + 1)*C_SIZEOF(dummy_int)) ! gpu_sync
    call custack_free(gwork, ndblk*C_SIZEOF(adt_dummy)) ! gpu_ddata
    call custack_free(gwork, nblk*C_SIZEOF(abt_dummy)) ! gpu_blkdata
    call custack_free(gwork, ncp*C_SIZEOF(act_dummy)) ! gpu_cpdata

    ! Set lperm for delayed columns from children
    do llist = lvlptr(lev), lvlptr(lev + 1) - 1

       node = lvllist(llist)
       lperm => nodes(node)%perm
 
       nd = 0
       do cn = child_ptr(node), child_ptr(node + 1) - 1
          cnode = child_list(cn)
          cnd = nodes(cnode)%ndelay
          cblkn = sptr(cnode + 1) - sptr(cnode)
          k = nd
          do i = nodes(cnode)%nelim + 1, cblkn + cnd
             j = nodes(cnode)%perm(i)
             k = k + 1
             lperm(k) = j
          end do
          nd = nd + cblkn + cnd - nodes(cnode)%nelim
       end do
    end do
  end subroutine assemble_fully_summed

  subroutine asminf_init(nnodes, child_ptr, child_list, sptr, rptr, rlist_direct, asminf)
    implicit none

    integer, intent(in) :: nnodes
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    integer, dimension(*), intent(in) :: rlist_direct
    type(asmtype), dimension(*), intent(out) :: asminf

    integer :: node, cnode
    integer :: blkm
    integer :: blkn
    integer :: cblkm
    integer :: cblkn
    integer :: m
    integer :: cn
    integer :: ii

    do node = 1, nnodes
       blkm = int(rptr(node + 1) - rptr(node))
       blkn = sptr(node + 1) - sptr(node)
       do cn = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(cn)
          cblkn = sptr(cnode + 1) - sptr(cnode)
          cblkm = int(rptr(cnode + 1) - rptr(cnode))
          m = 0
          do ii = rptr(cnode) + cblkn, rptr(cnode + 1) - 1
             if (rlist_direct(ii) .gt. blkn) exit
             m = m + 1
          end do
          asminf(cnode)%npassed = cblkm - cblkn
          asminf(cnode)%npassl = m
          asminf(cnode)%offset = rptr(cnode) + cblkn - 1
       end do
    end do

  end subroutine asminf_init

  ! C interfaces

  !> @brief Assemble contribution block
  subroutine spral_ssids_assemble_contrib(stream, nnodes, total_nch, c_lev, &
       c_lvlptr, c_lvllist, c_child_ptr, c_child_list, c_rptr, c_sptr, &
       c_asminf, pc_size, off_LDLT, ptr_ccval, c_gpu_contribs, ptr_levLDLT, &
       gpu_rlist_direct, c_gwork) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr), value, intent(in) :: stream ! CUDA stream
    integer(c_int), value :: nnodes ! Number of nodes in the atree
    integer(c_int), value :: total_nch ! Number of child node
    integer(c_int), value :: c_lev ! 0-based level index 
    type(c_ptr), value, intent(in) :: c_lvlptr
    type(c_ptr), value, intent(in) :: c_lvllist
    type(c_ptr), value, intent(in) :: c_child_ptr 
    type(c_ptr), value, intent(in) :: c_child_list 
    type(c_ptr), value, intent(in) :: c_rptr
    type(c_ptr), value, intent(in) :: c_sptr
    type(c_ptr), value, intent(in) :: c_asminf
    integer(c_long), value, intent(in) :: pc_size
    ! type(c_ptr), value, intent(in) :: c_off_LDLT
    integer(c_long), dimension(*), intent(in) :: off_LDLT
    type(c_ptr), value, intent(in) :: ptr_ccval ! GPU (*ptr_ccval) points to previous
      ! level's contribution blocks
    type(c_ptr), value, intent(in) :: c_gpu_contribs
    type(c_ptr), value             :: ptr_levLDLT ! GPU pointer to contribution blocks
    type(c_ptr), value, intent(in) :: gpu_rlist_direct
    type(c_ptr), value             :: c_gwork ! GPU workspace allocator 

    integer :: lev ! Fortran-indexed level index 
    integer, dimension(:), pointer :: lvlptr
    integer, dimension(:), pointer :: lvllist
    integer, dimension(:), pointer :: child_ptr
    integer, dimension(:), pointer :: child_list
    integer, dimension(:), pointer :: sptr
    integer(long), dimension(:), pointer :: rptr
    type(asmtype), dimension(:), pointer :: asminf ! Assembly info
    ! integer(long), dimension(:), pointer :: off_LDLT
    type(c_ptr), dimension(:), pointer :: gpu_contribs
    type(cuda_stack_alloc_type), pointer :: gwork ! GPU memory workspace allocator
    type(thread_stats) :: stats

    lev = c_lev+1

    ! lvlptr
    if (C_ASSOCIATED(c_lvlptr)) then
       call C_F_POINTER(c_lvlptr, lvlptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: lvlptr not associated"
       return
    end if

    ! lvllist
    if (C_ASSOCIATED(c_lvllist)) then
       call C_F_POINTER(c_lvllist, lvllist, shape=(/ nnodes /))
    else
       print *, "Error: lvllist not associated"
       return
    end if

    ! child_ptr
    if (C_ASSOCIATED(c_child_ptr)) then
       call C_F_POINTER(c_child_ptr, child_ptr, shape=(/ nnodes+2 /))
    else
       print *, "Error: child_ptr not associated"
       return
    end if

    ! child_list
    if (C_ASSOCIATED(c_child_list)) then
       call C_F_POINTER(c_child_list, child_list, shape=(/ nnodes /))
    else
       print *, "Error: child_list not associated"
       return  
    end if

    ! sptr
    if (C_ASSOCIATED(c_sptr)) then
       call C_F_POINTER(c_sptr, sptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: sptr not associated"
       return        
    end if

    ! rptr
    if (C_ASSOCIATED(c_rptr)) then
       call C_F_POINTER(c_rptr, rptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    ! asminf
    if (C_ASSOCIATED(c_asminf)) then
       call C_F_POINTER(c_asminf, asminf, shape=(/ nnodes /))
    else
       print *, "Error: asminf not associated"
       return
    end if

    ! ! off_LDLT
    ! if (C_ASSOCIATED(c_off_LDLT)) then
    !    call C_F_POINTER(c_off_LDLT, off_LDLT, shape=(/ nnodes /))
    ! else
    !    print *, "Error: off_LDLT not associated"
    !    return
    ! end if

    ! gpu_contribs
    if (C_ASSOCIATED(c_gpu_contribs)) then
       call C_F_POINTER(c_gpu_contribs, gpu_contribs, shape=(/ nnodes /))
    else
       print *, "Error: gpu_contribs not associated"
       return
    end if

    ! gwork
    if (C_ASSOCIATED(c_gwork)) then
       call C_F_POINTER(c_gwork, gwork)
    else
       print *, "Error: gwork not associated"
       return        
    end if

    call assemble_contrib(stream, total_nch, lev, lvlptr, lvllist, &
            child_ptr, child_list, sptr, rptr, asminf, pc_size, &
            off_LDLT, ptr_ccval, gpu_contribs, ptr_levLDLT, gpu_rlist_direct, &
            gwork, stats%st, stats%cuda_error)
      if (stats%st .ne. 0) goto 100
      if (stats%cuda_error .ne. 0) goto 200

300 continue
    return
200 continue
    print *, "[Error][spral_ssids_assemble_contrib] CUDA error"
    goto 300
100 continue
    print *, "[Error][spral_ssids_assemble_contrib] Allocation error"
    goto 300  
  end subroutine spral_ssids_assemble_contrib
    
  !> @brief Form contribution block
  subroutine spral_ssids_form_contrib(stream, nnodes, c_lev, c_lvlptr, c_lvllist, &
       c_nodes, c_rptr, c_sptr, off_LDLT, ptr_levLDLT, c_gwork) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr), value, intent(in) :: stream ! CUDA stream
    integer(c_int), value :: nnodes ! Number of nodes in the atree
    integer(c_int), value :: c_lev ! 0-based level index 
    type(c_ptr), value, intent(in) :: c_lvlptr
    type(c_ptr), value, intent(in) :: c_lvllist
    type(c_ptr), value, intent(in) :: c_nodes
    type(c_ptr), value, intent(in) :: c_rptr
    type(c_ptr), value, intent(in) :: c_sptr
    integer(c_long), dimension(*), intent(in) :: off_LDLT
    ! type(c_ptr), value, intent(in) :: c_off_LDLT
    type(c_ptr), value             :: ptr_levLDLT 
    type(c_ptr), value             :: c_gwork ! GPU workspace allocator 

    integer :: lev ! Fortran-indexed level index 
    integer, dimension(:), pointer :: lvlptr
    integer, dimension(:), pointer :: lvllist
    type(node_type), dimension(:), pointer :: nodes ! Nodes collection
    integer, dimension(:), pointer :: sptr
    integer(long), dimension(:), pointer :: rptr
    ! integer(long), dimension(:), pointer :: off_LDLT
    type(cuda_stack_alloc_type), pointer :: gwork ! GPU memory workspace allocator
    type(thread_stats) :: stats

    lev = c_lev+1

    ! lvlptr
    if (C_ASSOCIATED(c_lvlptr)) then
       call C_F_POINTER(c_lvlptr, lvlptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: lvlptr not associated"
       return
    end if

    ! lvllist
    if (C_ASSOCIATED(c_lvllist)) then
       call C_F_POINTER(c_lvllist, lvllist, shape=(/ nnodes /))
    else
       print *, "Error: lvllist not associated"
       return
    end if

    ! nodes
    if (C_ASSOCIATED(c_nodes)) then
       call C_F_POINTER(c_nodes, nodes, shape=(/ nnodes+1 /))
    else
       print *, "Error: nodes not associated"
       return
    end if

    ! sptr
    if (C_ASSOCIATED(c_sptr)) then
       call C_F_POINTER(c_sptr, sptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: sptr not associated"
       return        
    end if

    ! rptr
    if (C_ASSOCIATED(c_rptr)) then
       call C_F_POINTER(c_rptr, rptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    ! ! off_LDLT
    ! if (C_ASSOCIATED(c_off_LDLT)) then
    !    call C_F_POINTER(c_off_LDLT, off_LDLT, shape=(/ nnodes /))
    ! else
    !    print *, "Error: off_LDLT not associated"
    !    return
    ! end if

    ! gwork
    if (C_ASSOCIATED(c_gwork)) then
       call C_F_POINTER(c_gwork, gwork)
    else
       print *, "Error: gwork not associated"
       return        
    end if

    call form_contrib(stream, lev, lvlptr, nodes, lvllist, &
         off_LDLT, sptr, rptr, ptr_levLDLT, gwork, stats%st,       &
         stats%cuda_error)
    if (stats%st .ne. 0) goto 100
    if (stats%flag .ne. SSIDS_SUCCESS) goto 200

300 continue
    return
200 continue
    print *, "[Error][spral_ssids_assemble_form_contrib] Unknown error"
    goto 300
100 continue
    print *, "[Error][spral_ssids_assemble_form_contrib] Allocation error"
    goto 300
  end subroutine spral_ssids_form_contrib

  !> @brief Perform Cholesky factorizatino on fully-summed colmuns
  subroutine spral_ssids_factor_posdef(stream, nnodes, c_lev, c_lvlptr, c_lvllist, &
       c_nodes, c_rptr, c_sptr, ptr_levL, cublas_handle, c_gwork) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr), value, intent(in) :: stream ! CUDA stream
    integer(c_int), value :: nnodes ! Number of nodes in the atree
    integer(c_int), value :: c_lev ! 0-based level index 
    type(c_ptr), value, intent(in) :: c_lvlptr 
    type(c_ptr), value, intent(in) :: c_lvllist
    type(c_ptr), value, intent(in) :: c_nodes
    type(c_ptr), value, intent(in) :: c_rptr
    type(c_ptr), value, intent(in) :: c_sptr
    type(c_ptr), value :: ptr_levL ! GPU (*ptr_levL) points to L storage
    ! for current level
    type(c_ptr), value, intent(in) :: cublas_handle ! cuBLAS handle
    type(c_ptr), value             :: c_gwork ! GPU workspace allocator 

    integer :: lev ! Fortran-indexed level index 
    integer, dimension(:), pointer :: lvlptr
    integer, dimension(:), pointer :: lvllist
    type(node_type), dimension(:), pointer :: nodes ! Nodes collection
    integer, dimension(:), pointer :: sptr
    integer(long), dimension(:), pointer :: rptr
    type(cuda_stack_alloc_type), pointer :: gwork ! GPU memory workspace allocator
    type(thread_stats) :: stats
    integer :: p, n
    
    lev = c_lev+1

    ! lvlptr
    if (C_ASSOCIATED(c_lvlptr)) then
       call C_F_POINTER(c_lvlptr, lvlptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: lvlptr not associated"
       return
    end if

    ! lvllist
    if (C_ASSOCIATED(c_lvllist)) then
       call C_F_POINTER(c_lvllist, lvllist, shape=(/ nnodes /))
    else
       print *, "Error: lvllist not associated"
       return
    end if

    ! nodes
    if (C_ASSOCIATED(c_nodes)) then
       call C_F_POINTER(c_nodes, nodes, shape=(/ nnodes+1 /))
    else
       print *, "Error: nodes not associated"
       return
    end if

    ! sptr
    if (C_ASSOCIATED(c_sptr)) then
       call C_F_POINTER(c_sptr, sptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: sptr not associated"
       return        
    end if

    ! rptr
    if (C_ASSOCIATED(c_rptr)) then
       call C_F_POINTER(c_rptr, rptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    ! gwork
    if (C_ASSOCIATED(c_gwork)) then
       call C_F_POINTER(c_gwork, gwork)
    else
       print *, "Error: gwork not associated"
       return        
    end if

    call factor_posdef(stream, lev, lvlptr, nodes, lvllist, &
         sptr, rptr, ptr_levL, cublas_handle, stats, gwork)
    if (stats%st .ne. 0) goto 100
    if (stats%flag .ne. SSIDS_SUCCESS) then
       if (stats%flag .eq. SSIDS_ERROR_NOT_POS_DEF) then
          goto 300
       else
          goto 200
       end if
    end if
    if (stats%cuda_error .ne. 0) goto 400

    ! do p = lvlptr(lev), lvlptr(lev + 1) - 1
    !    n = lvllist(p)
    !    print *, "node = ", n, ", nelim = ", nodes(n)%nelim 
    ! end do
    
1000 continue
    return
400 continue
    print *, "[Error][spral_ssids_factor_posdef] CUDA error: ", cudaGetErrorString(stats%cuda_error)
    goto 1000
300 continue
    print *, "[Error][spral_ssids_factor_posdef] Matrix not posdef"
    goto 1000
200 continue
    print *, "[Error][spral_ssids_factor_posdef] Unknown error"
    goto 1000
100 continue
    print *, "[Error][spral_ssids_factor_posdef] Allocation error"
    goto 1000
  end subroutine spral_ssids_factor_posdef
    
  !> @brief Assemble fully-summed colmuns
  subroutine spral_ssids_assemble_fully_summed(stream, nnodes, total_nch, &
       c_lev, c_lvlptr, c_lvllist, c_nodes, gpu_ccval, c_gpu_contribs, &
       ptr_levL, gpu_rlist_direct, c_child_ptr, c_child_list, off_LDLT, &
       c_asminf, c_rptr, c_sptr, c_gwork, cuda_error) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr), value, intent(in) :: stream ! CUDA stream
    integer(c_int), value :: nnodes ! Number of nodes in the atree
    integer(c_int), value :: total_nch ! Total number of children for
      ! nodes in this level
    integer(c_int), value :: c_lev ! 0-based level index 
    type(c_ptr), value, intent(in) :: c_lvlptr 
    type(c_ptr), value, intent(in) :: c_lvllist
    type(c_ptr), value, intent(in) :: c_nodes
    type(c_ptr), value, intent(in) :: gpu_ccval ! GPU (*gpu_ccval)
    ! points to previous level's contribution blocks
    type(c_ptr), value, intent(in) :: c_gpu_contribs
    type(c_ptr), value             :: ptr_levL ! GPU (*ptr_levL) points to L storage
    ! for current level
    type(c_ptr), value, intent(in) :: gpu_rlist_direct
    type(c_ptr), value, intent(in) :: c_child_ptr 
    type(c_ptr), value, intent(in) :: c_child_list 
    ! type(c_ptr), value, intent(in) :: c_off_LDLT
    integer(c_long), dimension(*) :: off_LDLT
    type(c_ptr), value, intent(in) :: c_asminf ! Assembly info 
    type(c_ptr), value, intent(in) :: c_rptr
    type(c_ptr), value, intent(in) :: c_sptr
    type(c_ptr), value, intent(in) :: c_gwork ! GPU workspace allocator 
    integer(c_int) :: cuda_error ! CUDA error
    
    integer :: lev ! Fortran-indexed level index 
    integer, dimension(:), pointer :: lvlptr
    integer, dimension(:), pointer :: lvllist
    type(node_type), dimension(:), pointer :: nodes ! Nodes collection
    type(c_ptr), dimension(:), pointer :: gpu_contribs
    integer, dimension(:), pointer :: child_ptr
    integer, dimension(:), pointer :: child_list
    ! integer(long), dimension(:), pointer :: off_LDLT
    type(asmtype), dimension(:), pointer :: asminf ! Assembly info
    integer, dimension(:), pointer :: sptr
    integer(long), dimension(:), pointer :: rptr
    type(cuda_stack_alloc_type), pointer :: gwork ! GPU memory workspace allocator
    integer :: st ! Allocation error

    st = 0
    lev = c_lev+1

    ! lvlptr
    if (C_ASSOCIATED(c_lvlptr)) then
       call C_F_POINTER(c_lvlptr, lvlptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: lvlptr not associated"
       return
    end if

    ! lvllist
    if (C_ASSOCIATED(c_lvllist)) then
       call C_F_POINTER(c_lvllist, lvllist, shape=(/ nnodes /))
    else
       print *, "Error: lvllist not associated"
       return
    end if

    ! nodes
    if (C_ASSOCIATED(c_nodes)) then
       call C_F_POINTER(c_nodes, nodes, shape=(/ nnodes+1 /))
    else
       print *, "Error: nodes not associated"
       return
    end if

    ! gpu_contribs
    if (C_ASSOCIATED(c_gpu_contribs)) then
       call C_F_POINTER(c_gpu_contribs, gpu_contribs, shape=(/ nnodes /))
    else
       print *, "Error: gpu_contribs not associated"
       return
    end if

    ! child_ptr
    if (C_ASSOCIATED(c_child_ptr)) then
       call C_F_POINTER(c_child_ptr, child_ptr, shape=(/ nnodes+2 /))
    else
       print *, "Error: child_ptr not associated"
       return
    end if

    ! child_list
    if (C_ASSOCIATED(c_child_list)) then
       call C_F_POINTER(c_child_list, child_list, shape=(/ nnodes /))
    else
       print *, "Error: child_list not associated"
       return  
    end if

    ! ! off_LDLT
    ! if (C_ASSOCIATED(c_off_LDLT)) then
    !    call C_F_POINTER(c_off_LDLT, off_LDLT, shape=(/ nnodes /))
    ! else
    !    print *, "Error: off_LDLT not associated"
    !    return
    ! end if
    
    ! asminf
    if (C_ASSOCIATED(c_asminf)) then
       call C_F_POINTER(c_asminf, asminf, shape=(/ nnodes /))
    else
       print *, "Error: asminf not associated"
       return
    end if

    ! sptr
    if (C_ASSOCIATED(c_sptr)) then
       call C_F_POINTER(c_sptr, sptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: sptr not associated"
       return        
    end if

    ! rptr
    if (C_ASSOCIATED(c_rptr)) then
       call C_F_POINTER(c_rptr, rptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    ! gwork
    if (C_ASSOCIATED(c_gwork)) then
       call C_F_POINTER(c_gwork, gwork)
    else
       print *, "Error: gwork not associated"
       return        
    end if

    call assemble_fully_summed(stream, total_nch, lev, lvlptr, lvllist, &
         nodes, gpu_ccval, gpu_contribs, ptr_levL, gpu_rlist_direct, &
         child_ptr, child_list, off_LDLT, asminf, rptr, sptr, gwork, &
         st, cuda_error)
    if (st .ne. 0) goto 100
    if (cuda_error .ne. 0) goto 200

1000 continue
    return
200 continue
    print *, "[Error][spral_ssids_assemble_fully_summed] CUDA error"
    goto 1000
100 continue
    print *, "[Error][spral_ssids_assemble_fully_summed] Memory allocation"
    goto 1000
  end subroutine spral_ssids_assemble_fully_summed
  
  !> @brief Init factor L with original entries from matrix A
  subroutine spral_ssids_init_l_with_a(stream, nnodes, c_lev, c_lvlptr, &
       c_lvllist, c_nodes, ncb, level_size, c_nptr, c_rptr, gpu_nlist, &
       gpu_rlist, ptr_val, ptr_levL, c_gwork, cuda_error, ptr_scale) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr), value :: stream ! CUDA stream
    integer(c_int), value :: nnodes ! Number of nodes in the atree
    integer(c_int), value :: c_lev ! 0-based level index 
    type(c_ptr), value, intent(in) :: c_lvlptr 
    type(c_ptr), value, intent(in) :: c_lvllist
    type(c_ptr), value, intent(in) :: c_nodes
    integer(c_int), value :: ncb ! Number of nodes in level
    integer(c_long), value :: level_size ! Number of (fully-summed) entries in level  
    type(c_ptr), value, intent(in) :: c_nptr
    type(c_ptr), value, intent(in) :: c_rptr
    type(c_ptr), value, intent(in) :: gpu_nlist ! nlist array on the GPU
    type(c_ptr), value, intent(in) :: gpu_rlist ! rlist array on the GPU
    type(c_ptr), value, intent(in) :: ptr_val ! A matrix entries on the GPU
    type(c_ptr), value :: ptr_levL ! fully-summed entries on the GPU in level 
    type(c_ptr), value, intent(in) :: c_gwork
    integer(c_int) :: cuda_error ! CUDA error
    type(c_ptr), value, intent(in) :: ptr_scale
    
    integer :: lev ! Fortran-indexed level index 
    integer, dimension(:), pointer :: lvlptr
    integer, dimension(:), pointer :: lvllist
    type(node_type), dimension(:), pointer :: nodes ! Nodes collection
    integer(long), dimension(:), pointer :: nptr
    integer(long), dimension(:), pointer :: rptr
    
    type(cuda_stack_alloc_type), pointer :: gwork ! GPU memory workspace allocator
    integer :: st ! Allocation error

    lev = c_lev+1

    ! lvlptr
    if (C_ASSOCIATED(c_lvlptr)) then
       call C_F_POINTER(c_lvlptr, lvlptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: lvlptr not associated"
       return        
    end if

    ! lvllist
    if (C_ASSOCIATED(c_lvllist)) then
       call C_F_POINTER(c_lvllist, lvllist, shape=(/ nnodes /))
    else
       print *, "Error: lvllist not associated"
       return
    end if

    ! nodes
    if (C_ASSOCIATED(c_nodes)) then
       call C_F_POINTER(c_nodes, nodes, shape=(/ nnodes+1 /))
    else
       print *, "Error: nodes not associated"
       return        
    end if

    ! gwork
    if (C_ASSOCIATED(c_gwork)) then
       call C_F_POINTER(c_gwork, gwork)
    else
       print *, "Error: gwork not associated"
       return        
    end if

    ! nptr
    if (C_ASSOCIATED(c_nptr)) then
       call C_F_POINTER(c_nptr, nptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: nptr not associated"
       return        
    end if

    ! rptr
    if (C_ASSOCIATED(c_rptr)) then
       call C_F_POINTER(c_rptr, rptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    if (c_associated(ptr_scale)) then
       call init_L_with_A(stream, lev, lvlptr, lvllist, nodes, ncb, level_size, &
            nptr, rptr, gpu_nlist, gpu_rlist, ptr_val, ptr_levL, &
            gwork, st, cuda_error, ptr_scale)
    else
       call init_L_with_A(stream, lev, lvlptr, lvllist, nodes, ncb, level_size, &
            nptr, rptr, gpu_nlist, gpu_rlist, ptr_val, ptr_levL, &
            gwork, st, cuda_error)
    end if
    if (st .ne. 0) goto 100
    
200 continue
    return
100 continue
    print *, "Error: Memory allocation"
    goto 200
  end subroutine spral_ssids_init_l_with_a
  
  !> @brief Set gpu_lcol to c_gpu_lcol for given node
  subroutine spral_ssids_node_set_gpu_lcol(nnodes, c_nodes, c_node, &
       c_gpu_lcol) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), value :: nnodes ! Number of nodes
    type(c_ptr), value :: c_nodes ! C pointer to (Fortran) nodes structure
    integer(c_int), value :: c_node ! Node index, C-indexed
    type(c_ptr), value :: c_gpu_lcol ! C pointer to GPU memory for lcol 
    
    type(node_type), dimension(:), pointer :: nodes
    integer :: node

    if (C_ASSOCIATED(c_nodes)) then
       call C_F_POINTER(c_nodes, nodes, shape=(/ nnodes+1 /))
    else
       print *, "Error: nodes not associated"
       return        
    end if

    node = c_node+1

    nodes(node)%gpu_lcol = c_gpu_lcol
    
  end subroutine spral_ssids_node_set_gpu_lcol

    !> @brief Set gpu_lcol to c_gpu_lcol for given node
  subroutine spral_ssids_node_init_lperm(nnodes, c_nodes, c_node, c_sptr) &
       bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), value :: nnodes ! Number of nodes
    type(c_ptr), value :: c_nodes ! C pointer to (Fortran) nodes structure
    integer(c_int), value :: c_node ! Node index, C-indexed
    type(c_ptr), value, intent(in) :: c_sptr

    integer :: i, j
    integer :: blkn ! Number of (fully-summed) columns
    integer :: ndelay ! Number of incoming delays
    type(node_type), dimension(:), pointer :: nodes
    integer :: node ! Node index, Fortran-indexed
    integer, dimension(:), pointer :: sptr
    integer, dimension(:), pointer :: lperm

    ! nodes
    if (C_ASSOCIATED(c_nodes)) then
       call C_F_POINTER(c_nodes, nodes, shape=(/ nnodes+1 /))
    else
       print *, "Error: nodes not associated"
       return        
    end if

    ! sptr
    if (C_ASSOCIATED(c_sptr)) then
       call C_F_POINTER(c_sptr, sptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: sptr not associated"
       return        
    end if

    node = c_node+1 ! Fortran-index
    
    ndelay = nodes(node)%ndelay
    blkn = sptr(node+1) - sptr(node) + ndelay
    ! nodes(node)%nelim = blkn FIXME only valid for posdef
    allocate(nodes(node)%perm(blkn))
    lperm => nodes(node)%perm
    
    j = ndelay + 1
    do i = sptr(node), sptr(node+1)-1
       lperm(j) = i
       j = j + 1
    end do    
    
  end subroutine spral_ssids_node_init_lperm
    
  !> @brief Init asminf array
  subroutine spral_ssids_asminf_init(nnodes, c_child_ptr, c_child_list, &
       c_sptr, c_rptr, c_rlist_direct, c_asminf) bind (C) 
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), value :: nnodes
    type(c_ptr), value, intent(in) :: c_child_ptr 
    type(c_ptr), value, intent(in) :: c_child_list 
    type(c_ptr), value, intent(in) :: c_sptr
    type(c_ptr), value, intent(in) :: c_rptr
    type(c_ptr), value :: c_rlist_direct
    type(c_ptr) :: c_asminf

    integer, dimension(:), pointer :: child_ptr
    integer, dimension(:), pointer :: child_list
    integer, dimension(:), pointer :: sptr
    integer(long), dimension(:), pointer :: rptr
    integer, dimension(:), pointer :: rlist_direct
    type(asmtype), dimension(:), pointer :: asminf
    integer :: st

    ! child_ptr
    if (C_ASSOCIATED(c_child_ptr)) then
       call C_F_POINTER(c_child_ptr, child_ptr, shape=(/ nnodes+2 /))
    else
       print *, "Error: child_ptr not associated"
       return        
    end if

    ! child_list
    if (C_ASSOCIATED(c_child_list)) then
       call C_F_POINTER(c_child_list, child_list, shape=(/ nnodes /))
    else
       print *, "Error: child_list not associated"
       return        
    end if
    
    ! sptr
    if (C_ASSOCIATED(c_sptr)) then
       call C_F_POINTER(c_sptr, sptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: sptr not associated"
       return        
    end if

    ! rptr
    if (C_ASSOCIATED(c_rptr)) then
       call C_F_POINTER(c_rptr, rptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    ! rlist_direct
    if (C_ASSOCIATED(c_rlist_direct)) then
       call C_F_POINTER(c_rlist_direct, rlist_direct, shape=(/ rptr(nnodes+1)-1 /))
    else
       print *, "Error: c_rlist_direct is NULL"
       return
    end if

    c_asminf = c_null_ptr
    allocate(asminf(nnodes), stat=st)
    if (st .ne. 0) goto 100
    
    call asminf_init(nnodes, child_ptr, child_list, sptr, rptr, rlist_direct, asminf)

    ! Update C pointer with location of asminf array
    c_asminf = c_loc(asminf(1))
    
200 continue
    return
100 continue
    print *, "Error: Memory allocation"
    goto 200
  end subroutine spral_ssids_asminf_init

  !> @brief Init custack and set C pointer to it  
  subroutine spral_ssids_custack_init(c_gwork) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none

    type(c_ptr) :: c_gwork

    type(cuda_stack_alloc_type), pointer :: gwork
    integer :: st

    nullify(gwork)
    c_gwork = c_null_ptr    
    allocate(gwork, stat=st)
    if (st .ne. 0) goto 100

    c_gwork = c_loc(gwork)
    
200 continue
    return
100 continue
    print *, "Error: Allocation"
    goto 200
  end subroutine spral_ssids_custack_init

  !> @brief Init custack for level lev
  !> @param lev Level index that is C-indexed i.e. 0-based
  !> @param[out] c_gwork C pointer to the cuda stack initialized for level lev  
  subroutine spral_ssids_level_custack_init(c_lev, nnodes, c_lvlptr, c_lvllist, &
       c_child_ptr, c_child_list, c_nodes, c_sptr, c_rptr, c_asminf, c_gwork, &
       cuerr) bind(C)
    use, intrinsic :: iso_c_binding
    implicit none
    
    integer(c_int), value :: c_lev
    integer(c_int), value :: nnodes
    type(c_ptr), value, intent(in) :: c_lvlptr 
    type(c_ptr), value, intent(in) :: c_lvllist
    type(c_ptr), value, intent(in) :: c_child_ptr 
    type(c_ptr), value, intent(in) :: c_child_list 
    type(c_ptr), value, intent(in) :: c_nodes 
    type(c_ptr), value, intent(in) :: c_sptr 
    type(c_ptr), value, intent(in) :: c_rptr 
    type(c_ptr), value, intent(in) :: c_asminf 
    type(c_ptr), value             :: c_gwork    
    integer(c_int)                 :: cuerr

    integer :: lev
    integer, dimension(:), pointer :: lvlptr
    integer, dimension(:), pointer :: lvllist    
    integer, dimension(:), pointer :: child_ptr
    integer, dimension(:), pointer :: child_list
    type(node_type), dimension(:), pointer :: nodes
    integer, dimension(:), pointer :: sptr
    integer(long), dimension(:), pointer :: rptr
    type(asmtype), dimension(:), pointer :: asminf
    type(cuda_stack_alloc_type), pointer :: gwork
    integer(C_SIZE_T) :: lgpu_work

    if (C_ASSOCIATED(c_lvlptr)) then
       call C_F_POINTER(c_lvlptr, lvlptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: lvlptr not associated"
       return        
    end if

    if (C_ASSOCIATED(c_lvllist)) then
       call C_F_POINTER(c_lvllist, lvllist, shape=(/ nnodes /))
    else
       print *, "Error: lvllist not associated"
       return
    end if

    if (C_ASSOCIATED(c_child_ptr)) then
       call C_F_POINTER(c_child_ptr, child_ptr, shape=(/ nnodes+2 /))
    else
       print *, "Error: child_ptr not associated"
       return        
    end if

    if (C_ASSOCIATED(c_child_list)) then
       call C_F_POINTER(c_child_list, child_list, shape=(/ nnodes /))
    else
       print *, "Error: child_list not associated"
       return        
    end if

    if (C_ASSOCIATED(c_nodes)) then
       call C_F_POINTER(c_nodes, nodes, shape=(/ nnodes+1 /))
    else
       print *, "Error: nodes not associated"
       return        
    end if

    if (C_ASSOCIATED(c_sptr)) then
       call C_F_POINTER(c_sptr, sptr, shape=(/ nnodes+1 /))
    else
       print *, "Error sptr not associated"
       return        
    end if

    if (C_ASSOCIATED(c_rptr)) then
       call C_F_POINTER(c_rptr, rptr, shape=(/ nnodes+1 /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    if (C_ASSOCIATED(c_asminf)) then
       call C_F_POINTER(c_asminf, asminf, shape=(/ nnodes /))
    else
       print *, "Error: rptr not associated"
       return        
    end if

    ! gwork
    if (C_ASSOCIATED(c_gwork)) then
       call C_F_POINTER(c_gwork, gwork)
    else
       print *, "Error: gwork not associated"
       return        
    end if

    lev = c_lev + 1 ! c_lev is 0-based
    
    lgpu_work = level_gpu_work_size(lev, lvlptr, lvllist, child_ptr, &
         child_list, nodes, sptr, rptr, asminf)
    call custack_init(gwork, lgpu_work, cuerr)
    if (cuerr .ne. 0) goto 100

    ! print *, "lgpu_work = ", lgpu_work
    
200 continue
    return
100 continue
    print *, "[Error][spral_ssids_level_custack_init] CUDA error"
    goto 200
  end subroutine spral_ssids_level_custack_init
  
  subroutine spral_ssids_assign_nodes_to_levels(nnodes, c_sparent, c_gpu_contribs, c_num_levels, &
       c_lvlptr, c_lvllist) bind (C)
    use, intrinsic :: iso_c_binding
    implicit none
    
    integer(c_int), value :: nnodes
    type(c_ptr), intent(in), value :: c_sparent
    type(c_ptr), value :: c_gpu_contribs
    integer(c_int) :: c_num_levels
    type(c_ptr), value :: c_lvlptr 
    type(c_ptr), value :: c_lvllist

    integer, dimension(:), pointer :: sparent
    type(C_PTR), dimension(:), pointer :: gpu_contribs
    integer :: num_levels
    integer, dimension(:), pointer :: lvlptr
    integer, dimension(:), pointer :: lvllist
    integer :: st

    if (C_ASSOCIATED(c_sparent)) then
       call C_F_POINTER(c_sparent, sparent, shape=(/ nnodes /))
    else
       print *, "Error sparent not associated"
       return
    end if

    if (C_ASSOCIATED(c_gpu_contribs)) then
       call C_F_POINTER(c_gpu_contribs, gpu_contribs, shape=(/ nnodes /))
    else
       allocate(gpu_contribs(nnodes), stat=st)
       gpu_contribs(:) = c_null_ptr
    end if

    if (C_ASSOCIATED(c_lvlptr)) then
       call C_F_POINTER(c_lvlptr, lvlptr, shape=(/ nnodes+1 /))
    else
       print *, "Error lvlptr not associated"
       return        
    end if

    if (C_ASSOCIATED(c_lvllist)) then
       call C_F_POINTER(c_lvllist, lvllist, shape=(/ nnodes /))
    else
       print *, "Error lvllist not associated"
       return
    end if
    ! allocate(lvllist(nnodes), lvlptr(nnodes+1), stat=st)
    ! if (st .ne. 0) goto 100

    call assign_nodes_to_levels(nnodes, sparent, gpu_contribs, num_levels, &
         lvlptr, lvllist, st)
    if (st .ne. 0) goto 100

    c_num_levels = num_levels 
    ! c_lvlptr = c_loc(lvlptr(1))
    ! c_lvllist = c_loc(lvllist(1))

200 continue
    if (.not. C_ASSOCIATED(c_gpu_contribs) .and. &
         associated(gpu_contribs)) deallocate(gpu_contribs)
    return
100 continue
    print *, "Allocation error"
    goto 200
  end subroutine spral_ssids_assign_nodes_to_levels

end module spral_ssids_gpu_factor
