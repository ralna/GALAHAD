module spral_ssids_gpu_solve
  use iso_c_binding
  use spral_cuda
  use spral_ssids_gpu_datatypes
  use spral_ssids_gpu_interfaces
  use spral_ssids_datatypes
  implicit none

  private
  public :: bwd_solve_gpu, & ! Backwards solve on GPU
       fwd_solve_gpu,      & ! Forwards solve on GPU
       d_solve_gpu,        & ! D solve on GPU
       setup_gpu_solve       ! Setup data strucutres prior to solve

contains

  subroutine bwd_solve_gpu(job, posdef, n, stream_handle, stream_data, x, &
       st, cuda_error)
    implicit none
    integer, intent(in) :: job
    logical, intent(in) :: posdef
    integer, intent(in) :: n
    type(C_PTR), intent(in) :: stream_handle
    type(gpu_type), intent(in) :: stream_data
    real(wp), dimension(*), target, intent(inout) :: x
    integer, intent(out) :: cuda_error
    integer, intent(out) :: st  ! stat parameter

    type(C_PTR) :: gpu_x

    st = 0
    cuda_error = 0

    ! Push x on to GPU
    cuda_error = cudaMalloc(gpu_x, aligned_size(n*C_SIZEOF(x(1))))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpy_h2d(gpu_x, C_LOC(x), n*C_SIZEOF(x(1)))
    if (cuda_error .ne. 0) return
    ! Synchronise the device, see:
    ! http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
    cuda_error = cudaDeviceSynchronize()
    if (cuda_error .ne. 0) return

    ! Backwards solve
    call subtree_bwd_solve_gpu(job, posdef, stream_data%num_levels, &
         stream_data%bwd_slv_lookup, stream_data%bwd_slv_lwork,     &
         stream_data%bwd_slv_nsync, gpu_x, st, cuda_error,          &
         stream_handle)
    if (cuda_error .ne. 0) return

    ! Bring x back from GPU
    cuda_error = cudaMemcpy_d2h(C_LOC(x), gpu_x, n*C_SIZEOF(x(1)))
    if (cuda_error .ne. 0) return
    cuda_error = cudaFree(gpu_x)
    if (cuda_error .ne. 0) return
  end subroutine bwd_solve_gpu

!*************************************************************************
!
! This subroutine performs a backwards solve on the chunk of nodes specified
! by sa:en.
!
  subroutine subtree_bwd_solve_gpu(job, posdef, num_levels, bwd_slv_lookup, &
       lwork, nsync, gpu_x, st, cuda_error, stream)
    implicit none
    integer, intent(in) :: job
    logical, intent(in) :: posdef
    integer, intent(in) :: num_levels
    type(lookups_gpu_bwd), dimension(:) :: bwd_slv_lookup
    integer, intent(in) :: lwork
    integer, intent(in) :: nsync
    type(C_PTR), intent(inout) :: gpu_x
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error
    type(C_PTR), intent(in) :: stream

    integer(C_INT) :: dummy_int
    real(C_DOUBLE), target :: dummy_real
    type(C_PTR) :: gpu_work, gpu_sync
    integer(long) :: nrect, ndiag
    integer :: lvl

    logical(C_BOOL) :: dsolve, unit_diagonal

    nrect = 0; ndiag = 0
    cuda_error = 0
    st = 0

    if (posdef) then
       dsolve = .false. ! Never solve with D if we have an LL^T factorization
       unit_diagonal = .false.
    else ! indef
       dsolve = (job.ne.SSIDS_SOLVE_JOB_BWD) ! Do we solve L^T or (DL^T)?
       unit_diagonal = .true.
    end if

    ! Allocate workspace
    cuda_error = cudaMalloc(gpu_work, aligned_size(lwork*C_SIZEOF(dummy_real)))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMalloc(gpu_sync, aligned_size(2*nsync*C_SIZEOF(dummy_int)))
    if (cuda_error .ne. 0) return

    ! Backwards solve DL^Tx = z or L^Tx = z
    do lvl = num_levels, 1, -1
       call run_bwd_solve_kernels(dsolve, unit_diagonal, gpu_x, gpu_work, &
            nsync, gpu_sync, bwd_slv_lookup(lvl), stream)
    end do

    ! Free workspace
    cuda_error = cudaFree(gpu_work)
    if (cuda_error .ne. 0) return
    cuda_error = cudaFree(gpu_sync)
    if (cuda_error .ne. 0) return
  end subroutine subtree_bwd_solve_gpu

  subroutine d_solve_gpu(nnodes, sptr, stream_handle, &
       stream_data, n, x, st, cuda_error)
    implicit none
    integer, intent(in) :: nnodes
    integer, dimension(nnodes+1), intent(in) :: sptr
    type(C_PTR), intent(in) :: stream_handle
    type(gpu_type), intent(in) :: stream_data
    integer, intent(in) :: n
    real(wp), dimension(*), target, intent(inout) :: x
    integer, intent(out) :: cuda_error
    integer, intent(out) :: st  ! stat parameter

    type(C_PTR) :: gpu_x, gpu_y

    st = 0
    cuda_error = 0

    ! Allocate workspace on GPU (code doesn't work in place, so need in and out)
    cuda_error = cudaMalloc(gpu_x, 2*aligned_size(n*C_SIZEOF(x(n))))
    if (cuda_error .ne. 0) return
    gpu_y = c_ptr_plus_aligned(gpu_x, n*C_SIZEOF(x(n)))

    ! Push x on to GPU
    cuda_error = cudaMemcpy_h2d(gpu_x, C_LOC(x), n*C_SIZEOF(x(1)))
    if (cuda_error .ne. 0) return
    ! Synchronise the device, see:
    ! http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
    cuda_error = cudaDeviceSynchronize()
    if (cuda_error .ne. 0) return

    call subtree_d_solve_gpu(stream_data%num_levels, &
         stream_data%bwd_slv_lookup, gpu_x, gpu_y, stream_handle)

    ! Bring x back from GPU
    cuda_error = cudaMemcpy_d2h(C_LOC(x), gpu_y, n*C_SIZEOF(x(1)))
    if (cuda_error .ne. 0) return
    ! Free GPU memory
    cuda_error = cudaFree(gpu_x)
    if (cuda_error .ne. 0) return
  end subroutine d_solve_gpu

!*************************************************************************
!
! This subroutine performs a D solve on the specified subtree
!
  subroutine subtree_d_solve_gpu(num_levels, bwd_slv_lookup, gpu_x, gpu_y, stream)
    implicit none
    integer, intent(in) :: num_levels
    type(lookups_gpu_bwd), dimension(:) :: bwd_slv_lookup
    type(C_PTR), intent(inout) :: gpu_x
    type(C_PTR), intent(inout) :: gpu_y
    type(C_PTR), intent(in) :: stream

    integer :: lvl

    ! Diagonal solve Dy = x
    do lvl = num_levels, 1, -1
       call run_d_solve_kernel(gpu_x, gpu_y, bwd_slv_lookup(lvl), stream)
    end do
  end subroutine subtree_d_solve_gpu

  subroutine fwd_solve_gpu(posdef, child_ptr, child_list, n, nnodes, nodes, &
       rptr, stream_handle, stream_data, x, st, cuda_error)
    implicit none
    logical, intent(in) :: posdef
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    integer, intent(in) :: n
    integer, intent(in) :: nnodes
    type(node_type), dimension(nnodes), intent(in) :: nodes
    integer(long), dimension(*), intent(in) :: rptr
    type(C_PTR), intent(in) :: stream_handle
    type(gpu_type), intent(in) :: stream_data
    real(wp), dimension(*), target, intent(inout) :: x
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    type(C_PTR) :: gpu_x

    integer, dimension(:), allocatable :: cvmap
    type(C_PTR) :: gpu_cvalues

    integer :: blkm
    integer :: cn
    integer :: cnode
    integer :: ndelay
    integer :: nelim

    integer :: node

    type(C_PTR), dimension(:), allocatable, target :: cvalues

    integer(long) :: stack_ptr
    type(C_PTR) :: gpu_stack

    real(C_DOUBLE) :: dummy_real

    st = 0
    cuda_error = 0

    ! Push x on to GPU
    cuda_error = cudaMalloc(gpu_x, aligned_size(n*C_SIZEOF(x(1))))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpy_h2d(gpu_x, C_LOC(x), n*C_SIZEOF(x(1)))
    if (cuda_error .ne. 0) return

    ! Build map from nodes to position in child lists
    allocate(cvmap(nnodes), stat=st)
    if (st .ne. 0) return
    cvmap(:) = child_ptr(nnodes+1)-1 ! default for no parent
    do node = 1, nnodes
       do cn = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(cn)
          cvmap(cnode) = cn-1
       end do
    end do

    ! Determine size of "stack" and allocate it
    ! (Not an actual stack?)
    stack_ptr = 0
    do node = 1, nnodes
       ndelay = nodes(node)%ndelay
       nelim = nodes(node)%nelim
       blkm = int(rptr(node+1) - rptr(node)) + ndelay
       stack_ptr = stack_ptr + blkm-nelim
    end do
    cuda_error = cudaMalloc(gpu_stack, aligned_size(stack_ptr*C_SIZEOF(dummy_real)))
    if (cuda_error .ne. 0) return

    ! Build index map for use of "stack" and copy to GPU
    allocate(cvalues(nnodes), stat=st)
    if (st .ne. 0) return
    cuda_error = cudaMalloc(gpu_cvalues, aligned_size(nnodes*C_SIZEOF(cvalues(1))))
    if (cuda_error .ne. 0) return
    stack_ptr = 0
    do node = 1, nnodes
       ndelay = nodes(node)%ndelay
       nelim = nodes(node)%nelim
       blkm = int(rptr(node+1) - rptr(node)) + ndelay
       if ((blkm-nelim) .gt. 0) cvalues(cvmap(node)+1) = &
            c_ptr_plus(gpu_stack, stack_ptr*C_SIZEOF(dummy_real))
       stack_ptr = stack_ptr + blkm-nelim
    end do
    cuda_error = cudaMemcpy_h2d(gpu_cvalues, C_LOC(cvalues), &
         nnodes*C_SIZEOF(cvalues(1)))
    if (cuda_error .ne. 0) return
    ! Synchronise the device, see:
    ! http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
    cuda_error = cudaDeviceSynchronize()
    if (cuda_error .ne. 0) return

    ! Forwards solve
    call subtree_fwd_solve_gpu_lvl(posdef, stream_data%num_levels, gpu_x,    &
         stream_data%fwd_slv_lookup, stream_data%fwd_slv_lwork,              &
         stream_data%fwd_slv_nlocal, stream_data%fwd_slv_nsync,              &
         stream_data%fwd_slv_nasync, gpu_cvalues, cuda_error, stream_handle)
    if (cuda_error .ne. 0) return

    ! Apply any contribution above subtree to x directly
    if (stream_data%fwd_slv_contrib_lookup%nscatter .gt. 0) then
       call run_slv_contrib_fwd(stream_data%fwd_slv_contrib_lookup, gpu_x, &
            cvalues(cvmap(nnodes)+1), stream_handle)
    end if

    ! Free memory
    cuda_error = cudaFree(gpu_cvalues)
    if (cuda_error .ne. 0) return
    cuda_error = cudaFree(gpu_stack)
    if (cuda_error .ne. 0) return

    ! Bring x back from GPU
    cuda_error = cudaMemcpy_d2h(C_LOC(x), gpu_x, n*C_SIZEOF(dummy_real))
    if (cuda_error .ne. 0) return
    cuda_error = cudaFree(gpu_x)
    if (cuda_error .ne. 0) return
  end subroutine fwd_solve_gpu

  subroutine subtree_fwd_solve_gpu_lvl(posdef, num_levels, gpu_x, fwd_slv_lookup,&
       lwork, nlocal, nsync, nasm_sync, gpu_cvalues, cuda_error, stream)
    implicit none
    logical, intent(in) :: posdef
    integer, intent(in) :: num_levels
    type(C_PTR), intent(inout) :: gpu_x
    type(lookups_gpu_fwd), dimension(*) :: fwd_slv_lookup
    integer, intent(in) :: lwork
    integer, intent(in) :: nlocal
    integer, intent(in) :: nsync
    integer, intent(in) :: nasm_sync
    type(C_PTR), intent(in) :: gpu_cvalues
    integer, intent(out) :: cuda_error
    type(C_PTR), intent(in) :: stream

    integer :: lvl

    type(C_PTR) :: gpu_work, gpu_sync, gpu_asm_sync, gpu_xlocal
    logical(C_BOOL) :: cposdef

    integer(C_INT) :: dummy_int
    real(C_DOUBLE) :: dummy_real

    cposdef = posdef ! convert from Fortran to C logical size

    ! Do actual work
    cuda_error = cudaMalloc(gpu_sync, aligned_size(2*nsync*C_SIZEOF(dummy_int)))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMalloc(gpu_asm_sync, aligned_size((1+nasm_sync)*C_SIZEOF(dummy_int)))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMalloc(gpu_xlocal, aligned_size(nlocal*C_SIZEOF(dummy_real)))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMalloc(gpu_work, aligned_size(lwork*C_SIZEOF(dummy_real)))
    if (cuda_error .ne. 0) return
    do lvl = 1, num_levels
       call run_fwd_solve_kernels(cposdef, fwd_slv_lookup(lvl), gpu_xlocal, &
            gpu_cvalues, gpu_x, gpu_cvalues, gpu_work, nsync, gpu_sync,     &
            nasm_sync, gpu_asm_sync, stream)
    end do
    cuda_error = cudaFree(gpu_work)
    if (cuda_error .ne. 0) return
    cuda_error = cudaFree(gpu_xlocal)
    if (cuda_error .ne. 0) return
    cuda_error = cudaFree(gpu_asm_sync)
    if (cuda_error .ne. 0) return
    cuda_error = cudaFree(gpu_sync)
    if (cuda_error .ne. 0) return
  end subroutine subtree_fwd_solve_gpu_lvl

  subroutine create_gpu_lookup_fwd(nlvl, lvllist, nodes, child_ptr, child_list, &
       cvmap, sptr, rptr, rptr_with_delays, gpu_rlist_with_delays, gpu_clen,   &
       gpu_clists, gpu_clists_direct, gpul, nsync, nlocal, lwork, nasm_sync,   &
       stream, st, cuda_error)
    implicit none
    integer, intent(in) :: nlvl
    integer, dimension(*), intent(in) :: lvllist
    type(node_type), dimension(*), intent(in) :: nodes
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    integer, dimension(*), intent(in) :: cvmap
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    integer(C_SIZE_T), dimension(*), intent(in) :: rptr_with_delays
    type(C_PTR), intent(in) :: gpu_rlist_with_delays
    type(C_PTR), intent(in) :: gpu_clen
    type(C_PTR), intent(in) :: gpu_clists
    type(C_PTR), intent(in) :: gpu_clists_direct
    type(lookups_gpu_fwd), intent(out) :: gpul
    integer, intent(out) :: nsync
    integer, intent(out) :: nlocal
    integer, intent(out) :: lwork
    integer, intent(out) :: nasm_sync
    type(C_PTR), intent(in) :: stream
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    integer :: i, j, ci, ni
    integer :: node, blkm, blkn, nelim, gldl, nchild, ndelay, syncblk
    integer :: child, cnelim, cblkm, cblkn, cndelay

    integer :: nx, ny
    integer :: nassemble, nassemble2, nasmblk, ntrsv, ngemv, nreduce, nscatter
    type(assemble_lookup_type), dimension(:), allocatable, target :: &
         assemble_lookup
    type(assemble_lookup2_type), dimension(:), allocatable, target :: &
         assemble_lookup2
    type(assemble_blk_type), dimension(:), allocatable, target :: asmblkdata
    type(trsv_lookup_type), dimension(:), allocatable, target :: trsv_lookup
    type(gemv_notrans_lookup), dimension(:), allocatable, target :: gemv_lookup
    type(reduce_notrans_lookup), dimension(:), allocatable, target :: &
         reduce_lookup
    type(scatter_lookup_type), dimension(:), allocatable, target :: &
         scatter_lookup

    integer(C_SIZE_T) :: sz
    type(C_PTR) :: pmem

    type(C_PTR) :: ptdummy_int
    integer(C_INT) :: dummy_int
    real(C_DOUBLE) :: dummy_real

    ! Initialize outputs
    st = 0; cuda_error = 0
    nsync = 0
    nlocal = 0
    lwork = 0

    ! Calculate size of lookups and allocate
    nassemble = 0
    nasm_sync = 0
    nassemble2 = 0
    nasmblk = 0
    ntrsv = 0
    ngemv = 0
    nreduce = 0
    nscatter = 0
    do ni = 1, nlvl
       node = lvllist(ni)
       ! Setup basic data about node
       ndelay = nodes(node)%ndelay
       nelim = nodes(node)%nelim
       blkn = sptr(node+1) - sptr(node) + ndelay
       blkm = int(rptr(node+1) - rptr(node)) + ndelay
       nchild = child_ptr(node+1) - child_ptr(node)

       nassemble = nassemble + (blkm-1) / SLV_ASSEMBLE_NB + 1
       nasm_sync = nasm_sync + 1
       nassemble2 = nassemble2 + nchild
       do ci = child_ptr(node), child_ptr(node+1)-1
          child = child_list(ci)
          cndelay = nodes(child)%ndelay
          cnelim = nodes(child)%nelim
          cblkm = int(rptr(child+1) - rptr(child)) + cndelay

          nasmblk = nasmblk + (cblkm-cnelim-1) / SLV_ASSEMBLE_NB + 1
       end do

       if (nelim .gt. 0) &
            ntrsv = ntrsv + (nelim-1)/SLV_TRSV_NB_TASK + 1

       if (((blkm-nelim) .gt. 0) .and. (nelim .ne. 0)) then
          nx = (blkm-nelim-1)/SLV_GEMV_NX + 1
          ny = (nelim-1)/SLV_GEMV_NY + 1
          ngemv = ngemv + nx*ny
          nreduce = nreduce + nx
       end if

       nscatter = nscatter + (nelim-1)/SLV_SCATTER_NB + 1
    end do
    allocate(assemble_lookup(nassemble), trsv_lookup(ntrsv), gemv_lookup(ngemv),&
         reduce_lookup(nreduce), scatter_lookup(nscatter), &
         assemble_lookup2(nassemble2), asmblkdata(nasmblk), stat=st)
    if (st .ne. 0) return

    sz = aligned_size(nassemble*C_SIZEOF(assemble_lookup(1))) + &
         aligned_size(ntrsv*C_SIZEOF(trsv_lookup(1))) + &
         aligned_size(ngemv*C_SIZEOF(gemv_lookup(1))) + &
         aligned_size(nreduce*C_SIZEOF(reduce_lookup(1))) + &
         aligned_size(nscatter*C_SIZEOF(scatter_lookup(1))) + &
         aligned_size(nassemble2*C_SIZEOF(assemble_lookup2(1))) + &
         aligned_size(nasmblk*C_SIZEOF(asmblkdata(1)))
    cuda_error = cudaMalloc(pmem, sz)
    if (cuda_error .ne. 0) return

    ! Setup lookups
    nsync = 0
    nlocal = 0
    lwork = 0
    nassemble = 0
    nassemble2 = 0
    nasmblk = 0
    ntrsv = 0
    ngemv = 0
    nreduce = 0
    nscatter = 0
    do ni = 1, nlvl
       node = lvllist(ni)
       ! Setup basic data about node
       ndelay = nodes(node)%ndelay
       nelim = nodes(node)%nelim
       blkn = sptr(node+1) - sptr(node) + ndelay
       blkm = int(rptr(node+1) - rptr(node)) + ndelay
       gldl = blkm

       ! Add contributions
       nx = (blkm-1) / SLV_ASSEMBLE_NB + 1
       nchild = child_ptr(node+1) - child_ptr(node)
       do i = 0, nx-1
          nassemble = nassemble + 1
          assemble_lookup(nassemble)%m = &
               min(SLV_ASSEMBLE_NB, blkm-i*SLV_ASSEMBLE_NB)
          assemble_lookup(nassemble)%xend = &
               max(0, min(SLV_ASSEMBLE_NB, nelim-i*SLV_ASSEMBLE_NB))
          assemble_lookup(nassemble)%list = c_ptr_plus( gpu_rlist_with_delays, &
               rptr_with_delays(node) + i*SLV_ASSEMBLE_NB*C_SIZEOF(dummy_int) )
          assemble_lookup(nassemble)%x_offset = nlocal + i*SLV_ASSEMBLE_NB
          assemble_lookup(nassemble)%contrib_idx = cvmap(node)
          assemble_lookup(nassemble)%contrib_offset = i*SLV_ASSEMBLE_NB-nelim
          assemble_lookup(nassemble)%nchild = nchild
          assemble_lookup(nassemble)%clen = &
               c_ptr_plus(gpu_clen, (child_ptr(node)-1)*C_SIZEOF(dummy_int))
          assemble_lookup(nassemble)%clists = &
               c_ptr_plus(gpu_clists, (child_ptr(node)-1)*C_SIZEOF(ptdummy_int))
          assemble_lookup(nassemble)%clists_direct = &
               c_ptr_plus(gpu_clists_direct, (child_ptr(node)-1)*C_SIZEOF(ptdummy_int))
          assemble_lookup(nassemble)%cvalues_offset = child_ptr(node)-1
       end do

       ! Add contributions (new)
       syncblk = 0
       do ci = child_ptr(node), child_ptr(node+1)-1
          child = child_list(ci)
          cndelay = nodes(child)%ndelay
          cnelim = nodes(child)%nelim
          cblkn = sptr(child+1) - sptr(child) + cndelay
          cblkm = int(rptr(child+1) - rptr(child)) + cndelay

          nassemble2 = nassemble2 + 1
          assemble_lookup2(nassemble2)%m = cblkm-cnelim
          assemble_lookup2(nassemble2)%nelim = nelim
          assemble_lookup2(nassemble2)%list = &
               c_ptr_plus(gpu_clists_direct, (ci-1)*C_SIZEOF(ptdummy_int))
          if (blkm .gt. nelim) then
             assemble_lookup2(nassemble2)%cvparent = cvmap(node)
          else
             assemble_lookup2(nassemble2)%cvparent = 0 ! Avoid OOB error
          end if
          assemble_lookup2(nassemble2)%cvchild = cvmap(child)
          assemble_lookup2(nassemble2)%sync_offset = ni - 1
          assemble_lookup2(nassemble2)%sync_waitfor = syncblk
          assemble_lookup2(nassemble2)%x_offset = nlocal

          syncblk = syncblk + (cblkm-cnelim-1)/SLV_ASSEMBLE_NB + 1
          do i = 1, (cblkm-cnelim-1)/SLV_ASSEMBLE_NB + 1
             nasmblk = nasmblk + 1
             asmblkdata(nasmblk)%cp = nassemble2-1
             asmblkdata(nasmblk)%blk = i-1
          end do
       end do

       ! Solve with diagonal block
       if (nelim .gt. 0) then
          nx = (nelim-1)/SLV_TRSV_NB_TASK + 1
          do i = 0, nx-1
             ntrsv = ntrsv + 1
             trsv_lookup(ntrsv)%n = nelim
             trsv_lookup(ntrsv)%a = nodes(node)%gpu_lcol
             trsv_lookup(ntrsv)%lda = gldl
             trsv_lookup(ntrsv)%x_offset = nlocal
             trsv_lookup(ntrsv)%sync_offset = 2*nsync
          end do
       end if

       ! Update with off-diagonal block
       if (((blkm-nelim) .gt. 0) .and. (nelim .ne. 0)) then
          nx = (blkm-nelim-1)/SLV_GEMV_NX + 1
          ny = (nelim-1)/SLV_GEMV_NY + 1
          do j = 0, ny-1
             do i = 0, nx-1
                ngemv = ngemv + 1
                gemv_lookup(ngemv)%m = &
                     min(SLV_GEMV_NX, (blkm-nelim)-i*SLV_GEMV_NX)
                gemv_lookup(ngemv)%n = min(SLV_GEMV_NY, nelim-j*SLV_GEMV_NY)
                gemv_lookup(ngemv)%a = c_ptr_plus( nodes(node)%gpu_lcol, &
                     nelim*C_SIZEOF(dummy_real) + &
                     (i*SLV_GEMV_NX + j*SLV_GEMV_NY*gldl)*C_SIZEOF(dummy_real) )
                gemv_lookup(ngemv)%lda = gldl
                gemv_lookup(ngemv)%x_offset = nlocal+j*SLV_GEMV_NY
                gemv_lookup(ngemv)%y_offset = lwork+(blkm-nelim)*j+SLV_GEMV_NX*i
             end do
          end do
          do i = 0, nx-1
             nreduce = nreduce + 1
             reduce_lookup(nreduce)%m = min(SLV_GEMV_NX,blkm-nelim-i*SLV_GEMV_NX)
             reduce_lookup(nreduce)%n = ny
             reduce_lookup(nreduce)%src_offset = lwork + i*SLV_GEMV_NX
             reduce_lookup(nreduce)%ldsrc = blkm-nelim
             reduce_lookup(nreduce)%dest_idx = cvmap(node)
             reduce_lookup(nreduce)%dest_offset = i*SLV_GEMV_NX
          end do
          lwork = lwork + (blkm-nelim)*ny
       end if

       ! Copy eliminated (and delayed) variables back to x
       nx = (nelim-1)/SLV_SCATTER_NB + 1
       j = nscatter+1
       do i=0, nx-1
          nscatter = nscatter + 1
          scatter_lookup(nscatter)%n = min(SLV_SCATTER_NB,nelim-i*SLV_SCATTER_NB)
          scatter_lookup(nscatter)%src_offset = nlocal+i*SLV_SCATTER_NB
          scatter_lookup(nscatter)%index = c_ptr_plus( gpu_rlist_with_delays, &
               rptr_with_delays(node) + i*SLV_SCATTER_NB*C_SIZEOF(dummy_int) )
          scatter_lookup(nscatter)%dest_offset = 0
       end do

       nsync = nsync + 1
       nlocal = nlocal + nelim
    end do

    gpul%nassemble = nassemble
    sz = gpul%nassemble*C_SIZEOF(assemble_lookup(1))
    gpul%assemble = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%assemble, C_LOC(assemble_lookup), sz, &
         stream)
    if (cuda_error .ne. 0) return

    gpul%ntrsv = ntrsv
    sz = gpul%ntrsv*C_SIZEOF(trsv_lookup(1))
    gpul%trsv = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%trsv, C_LOC(trsv_lookup), sz, stream)
    if (cuda_error .ne. 0) return

    gpul%ngemv = ngemv
    sz = gpul%ngemv*C_SIZEOF(gemv_lookup(1))
    gpul%gemv = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%gemv, C_LOC(gemv_lookup), sz, stream)
    if (cuda_error .ne. 0) return

    gpul%nreduce = nreduce
    sz = gpul%nreduce*C_SIZEOF(reduce_lookup(1))
    gpul%reduce = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%reduce, C_LOC(reduce_lookup), sz, &
         stream)
    if (cuda_error .ne. 0) return

    gpul%nscatter = nscatter
    sz = gpul%nscatter*C_SIZEOF(scatter_lookup(1))
    gpul%scatter = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%scatter, C_LOC(scatter_lookup), sz, &
         stream)
    if (cuda_error .ne. 0) return

    gpul%nassemble2 = nassemble2
    gpul%nasm_sync = nasm_sync
    sz = gpul%nassemble2*C_SIZEOF(assemble_lookup2(1))
    gpul%assemble2 = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%assemble2, C_LOC(assemble_lookup2), &
         sz, stream)
    if (cuda_error .ne. 0) return

    gpul%nasmblk = nasmblk
    sz = gpul%nasmblk*C_SIZEOF(asmblkdata(1))
    gpul%asmblk = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%asmblk, C_LOC(asmblkdata), &
         sz, stream)
    if (cuda_error .ne. 0) return
  end subroutine create_gpu_lookup_fwd

  subroutine create_gpu_lookup_bwd(nlvl, lvllist, nodes, sptr, rptr,         &
       rptr_with_delays, gpu_rlist_with_delays, gpul, lwork, nsync, stream, &
       st, cuda_error)
    implicit none
    integer, intent(in) :: nlvl
    integer, dimension(*), intent(in) :: lvllist
    type(node_type), dimension(*), intent(in) :: nodes
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    integer(C_SIZE_T), dimension(*), intent(in) :: rptr_with_delays
    type(C_PTR), intent(in) :: gpu_rlist_with_delays
    type(lookups_gpu_bwd), intent(out) :: gpul
    integer, intent(out) :: lwork
    integer, intent(out) :: nsync
    type(C_PTR), intent(in) :: stream
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error

    integer :: i, j, xi, yi, li
    integer :: node, nelim, nd, blkm, blkn
    integer :: nx, ny, nrds, ntrsv, nscatter
    integer :: blk_gemv, blk_rds, blk_trsv, blk_scatter
    integer :: woffset, lupd
    type(gemv_transpose_lookup), dimension(:), allocatable, target :: gemv_lookup
    type(reducing_d_solve_lookup), dimension(:), allocatable, target :: &
         rds_lookup
    type(trsv_lookup_type), dimension(:), allocatable, target :: trsv_lookup
    type(scatter_lookup_type), dimension(:), allocatable, target :: &
         scatter_lookup

    integer(C_INT) :: dummy_int
    real(C_DOUBLE) :: dummy_real

    type(C_PTR) :: pmem
    integer(C_SIZE_T) :: sz

    st = 0; cuda_error = 0

    blk_gemv = 0
    blk_rds = 0
    blk_trsv = 0
    blk_scatter = 0
    lwork = 0
    nsync = 0
    do li = 1, nlvl
       node = lvllist(li)
       nsync = nsync + 1

       nelim = nodes(node)%nelim
       nd = nodes(node)%ndelay
       blkm = int(rptr(node+1) - rptr(node)) + nd

       nx = (blkm-nelim-1)/SLV_TRSM_TR_NBX + 1
       ny = (nelim-1)/SLV_TRSM_TR_NBY + 1
       blk_gemv = blk_gemv + nx*ny

       nrds = (nelim-1)/SLV_REDUCING_D_SOLVE_THREADS_PER_BLOCK + 1
       blk_rds = blk_rds + nrds
       lwork = lwork + nx*ny*SLV_TRSM_TR_NBY

       ntrsv = (nelim-1)/SLV_TRSV_NB_TASK + 1
       blk_trsv = blk_trsv + ntrsv

       nscatter = (nelim-1)/SLV_SCATTER_NB + 1
       blk_scatter = blk_scatter + nscatter
    end do
    allocate(gemv_lookup(blk_gemv), rds_lookup(blk_rds), trsv_lookup(blk_trsv), &
         scatter_lookup(blk_scatter), stat=st)
    if (st .ne. 0) return

    sz = aligned_size(blk_gemv*C_SIZEOF(gemv_lookup(1))) + &
         aligned_size(blk_rds*C_SIZEOF(rds_lookup(1))) + &
         aligned_size(blk_trsv*C_SIZEOF(trsv_lookup(1))) + &
         aligned_size(blk_scatter*C_SIZEOF(scatter_lookup(1)))
    cuda_error = cudaMalloc(pmem, sz)
    if (cuda_error .ne. 0) return

    woffset = 0
    blk_gemv = 1
    blk_rds = 1
    blk_trsv = 1
    blk_scatter = 1
    j = 0 ! Count of nodes at this level used for determing sync
    do li = 1, nlvl
       node = lvllist(li)
       nelim = nodes(node)%nelim
       if (nelim .eq. 0) cycle
       nd = nodes(node)%ndelay
       blkn = sptr(node+1) - sptr(node) + nd
       blkm = int(rptr(node+1) - rptr(node)) + nd

       ! Setup tasks for gemv
       nx = (blkm-nelim-1)/SLV_TRSM_TR_NBX + 1
       ny = (nelim-1)/SLV_TRSM_TR_NBY + 1
       lupd = ny * SLV_TRSM_TR_NBY
       do yi = 0, ny-1
          do xi = 0, nx-1
             gemv_lookup(blk_gemv)%m = &
                  min(SLV_TRSM_TR_NBX, blkm-nelim-xi*SLV_TRSM_TR_NBX)
             gemv_lookup(blk_gemv)%n = &
                  min(SLV_TRSM_TR_NBY, nelim-yi*SLV_TRSM_TR_NBY)
             gemv_lookup(blk_gemv)%lda = blkm
             gemv_lookup(blk_gemv)%a = &
                  c_ptr_plus(nodes(node)%gpu_lcol, &
                  C_SIZEOF(dummy_real)*( &
                  gemv_lookup(blk_gemv)%lda*yi*SLV_TRSM_TR_NBY + &
                  nelim+xi*SLV_TRSM_TR_NBX) &
                  )
             gemv_lookup(blk_gemv)%rlist = &
                  c_ptr_plus( gpu_rlist_with_delays, rptr_with_delays(node) + &
                  C_SIZEOF(dummy_int)*(nelim+xi*SLV_TRSM_TR_NBX) )
             gemv_lookup(blk_gemv)%yoffset = &
                  woffset + xi*lupd + yi*SLV_TRSM_TR_NBY;
             blk_gemv = blk_gemv + 1
          end do
       end do

       ! Setup tasks for reducing_d_solve()
       nrds = (nelim-1)/SLV_REDUCING_D_SOLVE_THREADS_PER_BLOCK + 1
       do xi = 0, nrds-1
          rds_lookup(blk_rds)%first_idx = &
               xi*SLV_REDUCING_D_SOLVE_THREADS_PER_BLOCK
          rds_lookup(blk_rds)%m = min(SLV_REDUCING_D_SOLVE_THREADS_PER_BLOCK, &
               nelim-xi*SLV_REDUCING_D_SOLVE_THREADS_PER_BLOCK)
          rds_lookup(blk_rds)%n = nx
          rds_lookup(blk_rds)%updoffset = woffset
          rds_lookup(blk_rds)%d = c_ptr_plus(nodes(node)%gpu_lcol, &
               blkm*blkn*C_SIZEOF(dummy_real))
          rds_lookup(blk_rds)%perm = &
               c_ptr_plus(gpu_rlist_with_delays, rptr_with_delays(node))
          rds_lookup(blk_rds)%ldupd = lupd
          blk_rds = blk_rds+1
       end do

       ! Setup tasks for trsv
       ntrsv = (nelim-1)/SLV_TRSV_NB_TASK + 1
       do i = 1, ntrsv
          trsv_lookup(blk_trsv)%n = nelim
          trsv_lookup(blk_trsv)%a = nodes(node)%gpu_lcol
          trsv_lookup(blk_trsv)%lda = blkm
          trsv_lookup(blk_trsv)%x_offset = woffset
          trsv_lookup(blk_trsv)%sync_offset = 2*j
          blk_trsv = blk_trsv + 1
       end do
       j = j + 1

       nscatter = (nelim-1)/SLV_SCATTER_NB + 1
       do i=0, nscatter-1
          scatter_lookup(blk_scatter)%n = &
               min(SLV_SCATTER_NB, nelim-i*SLV_SCATTER_NB)
          scatter_lookup(blk_scatter)%src_offset = &
               woffset+i*SLV_SCATTER_NB
          scatter_lookup(blk_scatter)%index = c_ptr_plus(gpu_rlist_with_delays, &
               rptr_with_delays(node) + i*SLV_SCATTER_NB*C_SIZEOF(dummy_int))
          scatter_lookup(blk_scatter)%dest_offset = 0
          blk_scatter = blk_scatter + 1
       end do

       woffset = woffset + nx*ny*SLV_TRSM_TR_NBY
    end do

    gpul%ngemv = blk_gemv-1
    gpul%nrds = blk_rds-1
    gpul%ntrsv = blk_trsv-1
    gpul%nscatter = blk_scatter-1

    sz = gpul%ngemv*C_SIZEOF(gemv_lookup(1))
    gpul%gemv = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%gemv, C_LOC(gemv_lookup), sz, stream)
    if (cuda_error .ne. 0) return

    sz = gpul%nrds*C_SIZEOF(rds_lookup(1))
    gpul%rds = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%rds, C_LOC(rds_lookup), sz, stream)
    if (cuda_error .ne. 0) return

    sz = gpul%ntrsv*C_SIZEOF(trsv_lookup(1))
    gpul%trsv = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%trsv, C_LOC(trsv_lookup), sz, stream)
    if (cuda_error .ne. 0) return

    sz = gpul%nscatter*C_SIZEOF(scatter_lookup(1))
    gpul%scatter = pmem
    pmem = c_ptr_plus_aligned(pmem, sz)
    cuda_error = cudaMemcpyAsync_h2d(gpul%scatter, C_LOC(scatter_lookup), sz, &
         stream)
    if (cuda_error .ne. 0) return
  end subroutine create_gpu_lookup_bwd

  subroutine setup_gpu_solve(n, child_ptr, child_list, nnodes, nodes, sparent, &
       sptr, rptr, rlist, stream_handle, stream_data, gpu_rlist_with_delays,   &
       gpu_clists, gpu_clists_direct, gpu_clen, st, cuda_error,                &
       gpu_rlist_direct_with_delays)
    implicit none
    integer, intent(in) :: n
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    integer, intent(in) :: nnodes
    type(node_type), dimension(*), intent(in) :: nodes
    integer, dimension(nnodes), intent(in) :: sparent
    integer, dimension(nnodes+1), intent(in) :: sptr
    integer(long), dimension(nnodes+1), intent(in) :: rptr
    integer, dimension(rptr(nnodes+1)-1), target, intent(in) :: rlist
    type(C_PTR), intent(in) :: stream_handle
    type(gpu_type), intent(inout) :: stream_data
    type(C_PTR), intent(out) :: gpu_rlist_with_delays
    type(C_PTR), intent(out) :: gpu_clists
    type(C_PTR), intent(out) :: gpu_clists_direct
    type(C_PTR), intent(out) :: gpu_clen
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error
    type(C_PTR), intent(out) :: gpu_rlist_direct_with_delays

    integer :: node, nd, nelim
    integer(long) :: i, k
    integer(long) :: blkm, blkn, ip

    integer(C_INT) :: dummy_int ! used for size of element of rlist
    integer, dimension(:), allocatable, target :: rlist2, rlist_direct2
    integer, dimension(:), pointer :: lperm

    integer(long) :: int_data

    integer :: cn, cnelim, cnode, parent
    integer, dimension(:), allocatable :: cvmap, rmap
    integer(C_INT), dimension(:), allocatable, target :: clen
    type(C_PTR), dimension(:), allocatable, target :: clists
    type(C_PTR), dimension(:), allocatable, target :: clists_direct
    integer(C_SIZE_T), dimension(:), allocatable :: rptr_with_delays

    st = 0; cuda_error = 0

    !
    ! Copy rlist, but delays while doing so
    !
    ! Count space
    allocate(rptr_with_delays(nnodes+1), stat=st)
    if (st .ne. 0) return
    rptr_with_delays(1) = 0
    do node = 1, nnodes
       nd = nodes(node)%ndelay
       blkm = int(rptr(node+1) - rptr(node)) + nd
       rptr_with_delays(node+1) = rptr_with_delays(node) + &
            C_SIZEOF(dummy_int)*blkm
    end do
    cuda_error = cudaMalloc(gpu_rlist_with_delays, aligned_size(rptr_with_delays(nnodes+1)))
    if (cuda_error .ne. 0) return
    int_data = rptr_with_delays(nnodes+1)
    allocate(rlist2(rptr_with_delays(nnodes+1)/C_SIZEOF(dummy_int)), stat=st)
    if (st .ne. 0) return
    ip = 0
    do node = 1, nnodes
       nelim = nodes(node)%nelim
       nd = nodes(node)%ndelay
       blkn = sptr(node+1) - sptr(node) + nd
       blkm = int(rptr(node+1) - rptr(node)) + nd
       lperm => nodes(node)%perm
       do i = 1, blkn
          rlist2(ip+i) = lperm(i) - 1
       end do
       k = rptr(node)+blkn-nd
       do i = blkn+1, blkm
          rlist2(ip+i) = rlist(k) - 1
          k = k + 1
       end do
       ip = ip + blkm
    end do
    cuda_error = cudaMemcpy_h2d(gpu_rlist_with_delays, C_LOC(rlist2), &
         rptr_with_delays(nnodes+1))
    if (cuda_error .ne. 0) return

    !
    ! Setup rlist_direct_with_delays
    !
    allocate(rlist_direct2(rptr_with_delays(nnodes+1)/C_SIZEOF(dummy_int)), &
         rmap(0:n-1), stat=st)
    if (st .ne. 0) return
    cuda_error = cudaMalloc(gpu_rlist_direct_with_delays, &
         aligned_size(rptr_with_delays(nnodes+1)))
    if (cuda_error .ne. 0) return
    do node = 1, nnodes
       nelim = nodes(node)%nelim
       parent = sparent(node)
       if ((parent .lt. 0) .or. (parent .gt. nnodes)) cycle ! root
       ! drop parent locs into map
       do i = rptr_with_delays(parent)/C_SIZEOF(dummy_int), rptr_with_delays(parent+1)/C_SIZEOF(dummy_int) -1
          rmap(rlist2(i+1)) = int(i - rptr_with_delays(parent)/C_SIZEOF(dummy_int))
       end do
       ! build rlist_direct2
       do i = rptr_with_delays(node)/C_SIZEOF(dummy_int)+nelim, rptr_with_delays(node+1)/C_SIZEOF(dummy_int) -1
          rlist_direct2(i+1) = rmap(rlist2(i+1))
       end do
    end do
    cuda_error = cudaMemcpy_h2d(gpu_rlist_direct_with_delays, &
         C_LOC(rlist_direct2), rptr_with_delays(nnodes+1))
    if (cuda_error .ne. 0) return

    !
    ! Setup solve info
    !
    allocate(clists(nnodes), clists_direct(nnodes), cvmap(nnodes), &
         clen(nnodes), stat=st)
    if (st .ne. 0) return
    cuda_error = cudaMalloc(gpu_clists, aligned_size(nnodes*C_SIZEOF(clists(1))))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMalloc(gpu_clists_direct, aligned_size(nnodes*C_SIZEOF(clists_direct(1))))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMalloc(gpu_clen, aligned_size(nnodes*C_SIZEOF(dummy_int)))
    if (cuda_error .ne. 0) return
    cvmap(:) = child_ptr(nnodes+1)-1 ! default for no parent
    do node = 1, nnodes
       do cn = child_ptr(node), child_ptr(node+1)-1
          cnode = child_list(cn)
          cnelim = nodes(cnode)%nelim

          cvmap(cnode) = cn-1
          clists(cn) = c_ptr_plus( gpu_rlist_with_delays, &
               rptr_with_delays(cnode) + cnelim*C_SIZEOF(dummy_int) )
          clists_direct(cn) = c_ptr_plus( gpu_rlist_direct_with_delays, &
               rptr_with_delays(cnode) + cnelim*C_SIZEOF(dummy_int) )
          clen(cn) = int((rptr_with_delays(cnode+1) - rptr_with_delays(cnode)) /&
               C_SIZEOF(dummy_int)) - cnelim
       end do
    end do
    cuda_error = cudaMemcpy_h2d(gpu_clists, C_LOC(clists), &
         nnodes*C_SIZEOF(clists(1)))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpy_h2d(gpu_clists_direct, C_LOC(clists_direct), &
         nnodes*C_SIZEOF(clists_direct(1)))
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpy_h2d(gpu_clen, C_LOC(clen), nnodes*C_SIZEOF(clen(1)))
    if (cuda_error .ne. 0) return
    ! Synchronise the device, see:
    ! http://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
    cuda_error = cudaDeviceSynchronize()
    if (cuda_error .ne. 0) return

    !
    ! Setup solve lookups
    !
    call setup_bwd_slv(nodes, sptr, rptr, stream_data%num_levels,   &
         stream_data%lvlptr, stream_data%lvllist, rptr_with_delays, &
         gpu_rlist_with_delays, stream_data%bwd_slv_lookup,         &
         stream_data%bwd_slv_lwork, stream_data%bwd_slv_nsync, st,  &
         cuda_error, stream_handle)
    if (st .ne. 0) return
    if (cuda_error .ne. 0) return
    call setup_fwd_slv(child_ptr, child_list, nnodes, nodes, sptr, rptr,  &
         stream_data%num_levels, stream_data%lvlptr, stream_data%lvllist, &
         rptr_with_delays, gpu_rlist_with_delays, gpu_clen, gpu_clists,   &
         gpu_clists_direct, cvmap, stream_data%fwd_slv_lookup,            &
         stream_data%fwd_slv_contrib_lookup, stream_data%fwd_slv_lwork,   &
         stream_data%fwd_slv_nlocal, stream_data%fwd_slv_nsync,           &
         stream_data%fwd_slv_nasync, st, cuda_error, stream_handle)
    if (st .ne. 0) return
    if (cuda_error .ne. 0) return
  end subroutine setup_gpu_solve

  subroutine setup_fwd_slv(child_ptr, child_list, nnodes, nodes, sptr, rptr, &
       num_levels, lvlptr, lvllist, rptr_with_delays, gpu_rlist_with_delays, &
       gpu_clen, gpu_clists, gpu_clists_direct, cvmap, fwd_slv_lookup,       &
       fwd_slv_contrib_lookup, lwork, nlocal, nsync, nasm_sync, st,          &
       cuda_error, stream)
    implicit none
    integer, dimension(*), intent(in) :: child_ptr
    integer, dimension(*), intent(in) :: child_list
    integer, intent(in) :: nnodes
    type(node_type), dimension(*), intent(in) :: nodes
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    integer, intent(in) :: num_levels
    integer, dimension(*), intent(in) :: lvlptr
    integer, dimension(*), intent(in) :: lvllist
    integer(C_SIZE_T), dimension(*), intent(in) :: rptr_with_delays
    type(C_PTR), intent(in) :: gpu_rlist_with_delays
    type(C_PTR), intent(in) :: gpu_clen
    type(C_PTR), intent(in) :: gpu_clists
    type(C_PTR), intent(in) :: gpu_clists_direct
    integer, dimension(*), intent(in) :: cvmap
    type(lookups_gpu_fwd), dimension(:), allocatable, intent(out) :: &
         fwd_slv_lookup
    type(lookup_contrib_fwd), intent(out) :: fwd_slv_contrib_lookup
    integer, intent(out) :: lwork
    integer, intent(out) :: nlocal
    integer, intent(out) :: nsync
    integer, intent(out) :: nasm_sync
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error
    type(C_PTR), intent(in) :: stream

    integer :: lvl, i, j, k, p
    integer :: blkm, blkn, ndelay, nelim
    type(scatter_lookup_type), dimension(:), allocatable, target :: &
         scatter_lookup
    integer(C_INT) :: dummy_int

    st = 0
    cuda_error = 0

    allocate(fwd_slv_lookup(num_levels), stat=st)
    if (st .ne. 0) return

    nsync = 0
    nlocal = 0
    lwork = 0
    nasm_sync = 0
    do lvl = 1, num_levels
       call create_gpu_lookup_fwd(lvlptr(lvl+1)-lvlptr(lvl),                     &
            lvllist(lvlptr(lvl):lvlptr(lvl+1)-1), nodes, child_ptr, child_list,  &
            cvmap, sptr, rptr, rptr_with_delays, gpu_rlist_with_delays, gpu_clen,&
            gpu_clists, gpu_clists_direct, fwd_slv_lookup(lvl), i, j, k, p,      &
            stream, st, cuda_error)
       if (st .ne. 0) return
       if (cuda_error .ne. 0) return
       nsync = max(nsync, i)
       nlocal = max(nlocal, j)
       lwork = max(lwork, k)
       nasm_sync = max(nasm_sync, p)
    end do

    ! If we contribute to a parent subtree, we need to do a scatter at end of slv
    ! If there is a contribution, it must be from the last node
    blkm = int(rptr(nnodes+1)-rptr(nnodes))
    blkn = sptr(nnodes+1)-sptr(nnodes)
    if (blkm .gt. blkn) then
       ndelay = nodes(nnodes)%ndelay
       nelim = nodes(nnodes)%nelim
       fwd_slv_contrib_lookup%nscatter = (blkm+ndelay-nelim-1)/SLV_SCATTER_NB + 1
       allocate(scatter_lookup(fwd_slv_contrib_lookup%nscatter), stat=st)
       if (st .ne. 0) return
       cuda_error = cudaMalloc(fwd_slv_contrib_lookup%scatter, &
            aligned_size(fwd_slv_contrib_lookup%nscatter*C_SIZEOF(scatter_lookup(1))))
       if (cuda_error .ne. 0) return
       do i = 0, fwd_slv_contrib_lookup%nscatter-1
          scatter_lookup(i+1)%n = &
               min(SLV_SCATTER_NB,blkm+ndelay-nelim-i*SLV_SCATTER_NB)
          scatter_lookup(i+1)%src_offset = i*SLV_SCATTER_NB
          scatter_lookup(i+1)%index = &
               c_ptr_plus(gpu_rlist_with_delays, rptr_with_delays(nnodes) + &
               (nelim+i*SLV_SCATTER_NB)*C_SIZEOF(dummy_int))
          scatter_lookup(i+1)%dest_offset = 0
       end do
       cuda_error = cudaMemcpyAsync_h2d(fwd_slv_contrib_lookup%scatter, &
            C_LOC(scatter_lookup), &
            fwd_slv_contrib_lookup%nscatter*C_SIZEOF(scatter_lookup(1)), stream)
       if (cuda_error .ne. 0) return
    else
       fwd_slv_contrib_lookup%nscatter = 0
       fwd_slv_contrib_lookup%scatter = C_NULL_PTR
    end if
  end subroutine setup_fwd_slv

  subroutine setup_bwd_slv(nodes, sptr, rptr, num_levels, lvlptr, lvllist,   &
      rptr_with_delays, gpu_rlist_with_delays, bwd_slv_lookup, lwork, nsync, &
      st, cuda_error, stream)
    implicit none
    type(node_type), dimension(*), intent(in) :: nodes
    integer, dimension(*), intent(in) :: sptr
    integer(long), dimension(*), intent(in) :: rptr
    integer, intent(in) :: num_levels
    integer, dimension(*), intent(in) :: lvlptr
    integer, dimension(*), intent(in) :: lvllist
    integer(C_SIZE_T), dimension(*), intent(in) :: rptr_with_delays
    type(C_PTR), intent(in) :: gpu_rlist_with_delays
    type(lookups_gpu_bwd), dimension(:), allocatable, intent(out) :: &
         bwd_slv_lookup
    integer, intent(out) :: lwork
    integer, intent(out) :: nsync
    integer, intent(out) :: cuda_error
    integer, intent(out) :: st
    type(C_PTR), intent(in) :: stream

    integer :: i, j, lvl

    st = 0

    allocate(bwd_slv_lookup(num_levels), stat=st)
    if (st .ne. 0) return

    lwork = 0
    nsync = 0
    do lvl = 1, num_levels
       call create_gpu_lookup_bwd(lvlptr(lvl+1)-lvlptr(lvl),           &
         lvllist(lvlptr(lvl):lvlptr(lvl+1)-1), nodes, sptr, rptr,      &
         rptr_with_delays, gpu_rlist_with_delays, bwd_slv_lookup(lvl), &
         i, j, stream, cuda_error, st)
       if (cuda_error .ne. 0) return
       if (st .ne. 0) return
       lwork = max(lwork, i)
       nsync = max(nsync, j)
    end do
  end subroutine setup_bwd_slv

end module spral_ssids_gpu_solve
