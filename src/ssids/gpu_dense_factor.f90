module spral_ssids_gpu_dense_factor
  use, intrinsic :: iso_c_binding
  use spral_cuda
  use spral_ssids_gpu_alloc, only : cuda_stack_alloc_type, custack_alloc, &
       custack_free
  use spral_ssids_gpu_datatypes, only : multinode_fact_type, &
       multiblock_fact_type, multireorder_data, multielm_data
  use spral_ssids_gpu_interfaces
  use spral_ssids_datatypes, only : wp, one, &
       MNF_BLOCKS, SSIDS_ERROR_NOT_POS_DEF
  implicit none

  private
  public :: node_llt,      & ! Perform LL^T factorization of a single node
            node_ldlt,     & ! Perform LDL^T factorization of a single node
            multinode_llt, & ! Perform LL^T factorizations of multiple nodes
            multinode_ldlt   ! Perform LDL^T factorizations of multiple nodes

contains

!
! ldlt factorization A = L*D*L^T for one node
!
! repeatedly calls CUDA kernel block_ldlt to factorize
! a block of <block_size> columns ("pending block")
! and, if needed, reordering and forward elimination kernels
!
  subroutine node_ldlt(stream, nrows, ncols, gpu_L, gpu_LD, ldL, gpu_D, gpu_B,  &
       gpu_ind, delta, eps, block_size, perm, ind, done, gwork, cublas_handle, &
       cuda_error, cublas_error)
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: nrows ! A's n.o. rows
    integer, intent(in) :: ncols ! A's n.o. cols
    integer, intent(in) :: ldL   ! A/L's leading dimension
    integer, intent(in) :: block_size ! no cols factorized by block_ldlt kernel
    type(C_PTR) :: gpu_L  ! dev. pointer to L-factor (nrows x ncols, A on input)
    type(C_PTR) :: gpu_LD ! same for L*D (nrows x ncols)
    type(C_PTR) :: gpu_D  ! same for D-factor (2 x ncols)
    type(C_PTR) :: gpu_B  ! same for a buffer (nrows x block_size)
    type(C_PTR) :: gpu_ind ! same for pivot indices (block_size)
    real(kind = wp), intent(in) :: delta ! admissible pivot threshold
    real(kind = wp), intent(in) :: eps ! zero pivot threshold
    integer, intent(inout), target :: perm(nrows) ! row ordering
    integer(C_INT), intent(inout), target :: ind(2*ncols) ! work array for reorder
    integer, intent(out) :: done ! no successfully pivoted cols
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    type(C_PTR), intent(in) :: cublas_handle ! CUBLAS handle
    integer, intent(out) :: cuda_error
    integer, intent(out) :: cublas_error

    integer :: ib, jb ! first and last column of the pending column block
    integer :: rb, cb ! n.o. rows and cols in the pending block
    integer :: pivoted ! n.o. successful pivots after recent block_ldlt call
    integer :: delayed ! current number of delayed columns
    integer :: recent ! n.o. successful pivots after last step back
    integer :: step ! forward elimination step, see (1) below
    integer :: right ! the last col to which forward elimination reaches
    integer :: left ! the last fully eliminated col

    integer(C_SIZE_T) :: lpitch  ! leading dimension of A/L in bytes
    integer(C_SIZE_T) :: bpitch  ! same for the buffer

    integer :: i, j, l, p, q ! aux integers
    integer(C_SIZE_T) :: sz ! offset for c_ptr_plus

    integer(C_INT), target :: pstat ! block_ldlt return status, see (5)

    type(C_PTR) :: gpu_u, gpu_v, gpu_w, gpu_p, gpu_q, gpu_r ! aux device pointers
    type(C_PTR) :: gpu_stat ! same for pivoting kernel status (scalar)

    integer(C_SIZE_T) :: r_size

    real(wp) :: dummy_real

    cuda_error = 0

    gpu_stat = custack_alloc(gwork, C_SIZEOF(pstat))

    lpitch = ldL*C_SIZEOF(dummy_real)
    bpitch = nrows*C_SIZEOF(dummy_real)

    !
    !(1) in order to maximize the efficiency of cublasDgemm, forward elimination
    !    of pivoted columns is performed in two stages: partial and full
    !
    ! - at the first stage, each successfully pivoted column is eliminated only
    !   from non-pivoted columns among the leftmost <step> columns
    !
    ! - once all these <step> columns are processed  (ie each one is either
    !   pivoted or delayed) all pivoted columns that have not yet been fully
    !   eliminated are eliminated from all columns to the right of the processed
    !   (see (3))
    !
    ! - the described two-stage procedure is applied recursively to the
    !   remaining non-processed columns
    !
    step = 24*block_size ! seems to be the optimal choice
    !
    ! the two-stage elimination procedure just described is controlled by
    ! two parameters:
    ! <left> is the number of fully eliminated columns
    ! <right> is the last column to which elimination reaches at the first stage
    !
    left = 0 ! initially, there are no fully eliminated columns
    if (ncols .gt. 2*step) then
       right = step  ! initial value for <right>
    else
       right = ncols ! if <ncols> is too small, eliminate all the way to the right
    end if

    done = 0 ! last successfully pivoted column

    !
    ! the main step of the algorithm is the factorization of
    ! a rectangular matrix formed by columns ib, ib + 1, ..., jb
    ! and rows done + 1, ..., nrows where
    !
    !   jb = min(ib + block_size, right)
    !
    jb = 0 ! no columns processed yet
    recent = 0

    do while (jb .lt. ncols)

       ib = jb + 1 ! the first column to be processed next
       if (ib .gt. right) then ! the current partial elimination stage is over
          if ((right .lt. ncols) .and. (left .lt. done)) then
             !
             ! (3) fully eliminate all partially eliminated columns
             !
             ! - partially eliminated columns are left + 1, ..., done
             ! - they need to be eliminated from columns right + 1, ..., ncols
             ! - <done> columns are fully processed, so respective rows
             !   will no longer participate in this node's factorization
             ! - thus, we need to update
             !     A(done + 1 : nrows, right + 1 : ncols) -=
             !       L(done + 1 : nrows, left + 1 : done) *
             !         LD(right + 1 : nrows, left + 1 : done)^T
             !
             sz = (done + left*ldL)*C_SIZEOF(dummy_real)
             gpu_u = c_ptr_plus( gpu_L, sz )
             sz = (right + left*ldL)*C_SIZEOF(dummy_real)
             gpu_v = c_ptr_plus( gpu_LD, sz )
             sz = (done + right*ldL)*C_SIZEOF(dummy_real)
             gpu_w = c_ptr_plus( gpu_L, sz )
             cublas_error = cublasDgemm(cublas_handle, 'N', 'T', nrows-done, &
                  ncols-right, done-left, -ONE, gpu_u, ldL, gpu_v, ldL, ONE, gpu_w, ldL)
          end if
          left = done ! update the number of fully eliminated columns
          right = min(ncols, right + step) ! move the partial elimination margin
       end if

       rb = nrows - done ! n.o. rows in the pending block
       cb = min(block_size, right - ib + 1) ! n.o. of cols thereof
       jb = jb + cb ! last column of the pending block

       delayed = ib - done - 1 ! columns between <done> and <ib> failed

       !
       ! obtain pointers to the pending block of A/L
       ! and respective pointers for LD and D
       !
       sz = (done + (ib - 1)*ldL)*C_SIZEOF(dummy_real)
       gpu_u = c_ptr_plus( gpu_L, sz )
       sz = (done + (ib - 1)*ldL)*C_SIZEOF(dummy_real)
       gpu_v = c_ptr_plus( gpu_LD, sz )
       gpu_w = c_ptr_plus( gpu_D, 2*done*C_SIZEOF(dummy_real) )

       !
       ! try to factorize the pending block
       !
       ! since the number and location of successfully pivoted columns
       ! is not known a priori and may be different for different
       ! CUDA blocks, the computed part of L factor is placed in a buffer
       ! so that parts of unsuccesfull columns of A are not overwritten
       !
       ! LD and D are ok as their columns right of <done> do not
       ! contain any useful information prior to this call
       !
       call block_ldlt(stream, rb, cb, delayed, gpu_u, ldL, gpu_B, rb, gpu_v, ldL,&
            gpu_w, delta, eps, gpu_ind, gpu_stat)
       !
       ! retrieve the execution status (n.o. successful pivots) and pivot order
       !
       cuda_error = &
            cudaMemcpyAsync_D2H(C_LOC(pstat), gpu_stat, C_SIZEOF(pstat), stream)
       if (cuda_error .ne. 0) return
       cuda_error = &
            cudaMemcpyAsync_D2H(C_LOC(ind), gpu_ind, cb*C_SIZEOF(ind(1)), stream)
       if (cuda_error .ne. 0) return

       cuda_error = cudaStreamSynchronize(stream) ! Wait for pstat, ind
       if (cuda_error .ne. 0) return

       pivoted = pstat ! (5) n.o. successfully pivoted columns
       !
       ! copy successfully pivoted columns to L
       !
       if (pivoted .gt. 0) &
            call copy_mc( stream, rb, cb, gpu_B, rb, gpu_u, ldL, gpu_ind )

       if (pivoted .lt. 1) then ! complete failure

          ! step back after hitting the partial elimination margin
          ! if there have been delays, and try to pivot them again
          if ((jb .eq. right) .and. (recent .gt. 0)) then
             jb = done
             recent = 0
          end if

          ! delay all columns in the pending block
          cycle

       else if (pivoted .le. delayed) then

          !
          ! newly pivoted columns are swapped with those that failed earlier
          ! and are next to the succeeded ones, ie have indices
          ! done + 1, ..., done + pivoted
          !
          ! nothing to do with D as the successful pivots have been
          ! placed where they should be by block_ldlt

          !
          ! global row/col ordering must be changed accordingly
          !
          do j = 1, cb
             if (ind(j) .lt. 1) cycle ! this column failed, leave it
             p = done + ind(j)
             q = ib + j - 1
             l = perm(q)
             perm(q) = perm(p)
             perm(p) = l
          end do

          !
          ! swap newly pivoted columns of L with delayed columns
          ! done + 1, ..., done + pivoted
          !
          gpu_u = c_ptr_plus( gpu_L, done*lpitch )
          gpu_v = c_ptr_plus( gpu_L, (ib - 1)*lpitch )
          call swap_ni2D_ic(stream, nrows, cb, gpu_u, ldL, gpu_v, ldL, gpu_ind)
          !
          ! swap respective rows
          !
          gpu_u = c_ptr_plus( gpu_L, done*C_SIZEOF(dummy_real) )
          gpu_v = c_ptr_plus( gpu_L, (ib - 1)*C_SIZEOF(dummy_real) )
          call swap_ni2D_ir(stream, cb, ncols, gpu_u, ldL, gpu_v, ldL, gpu_ind)

          !
          ! same for LD
          !
          gpu_u = c_ptr_plus( gpu_LD, done*lpitch )
          gpu_v = c_ptr_plus( gpu_LD, (ib - 1)*lpitch )
          call copy_ic(stream, nrows, cb, gpu_v, ldL, gpu_u, ldL, gpu_ind)
          gpu_u = c_ptr_plus( gpu_LD, done*C_SIZEOF(dummy_real) )
          gpu_v = c_ptr_plus( gpu_LD, (ib - 1)*C_SIZEOF(dummy_real) )
          call swap_ni2D_ir(stream, cb, ncols, gpu_u, ldL, gpu_v, ldL, gpu_ind)

       else ! not enough space for swap, has to reorder all columns between
            ! <done> and <jb + 1> placing the newly pivoted first

          !
          ! compute permutation index based on pivot order returned by block_ldlt
          !
          if (delayed .gt. 0) then
             do i = cb, 1, -1
                ind(i + delayed) = ind(i)
             end do
             do i = 1, delayed
                ind(i) = 0
             end do
          end if
          j = pivoted
          do i = 1, cb + delayed
             if (ind(i) .eq. 0) then
                j = j + 1
                ind(i) = j
             end if
          end do

          !
          ! reorder row/col indices
          !
          do i = 1, cb + delayed
             j = ind(i)
             ind(ncols + j) = perm(done + i)
          end do
          do i = 1, cb + delayed
             perm(done + i) = ind(ncols + i)
          end do

          !
          ! copy to GPU
          !
          cuda_error = cudaMemcpyAsync_H2D(gpu_ind, C_LOC(ind), &
               (cb+delayed)*C_SIZEOF(ind(1)), stream)
          if (cuda_error .ne. 0) return

          !
          ! reorder columns of L and LD in focus
          !
          gpu_u = c_ptr_plus( gpu_L, done*lpitch )
          gpu_v = c_ptr_plus( gpu_LD, done*lpitch )
          call reorder_cols2(stream, nrows, cb + delayed, gpu_u, ldL, gpu_v, ldL, &
               gpu_ind, 1)

          !
          ! reorder respective rows
          !
          gpu_u = c_ptr_plus( gpu_L, done*C_SIZEOF(dummy_real) )
          gpu_v = c_ptr_plus( gpu_LD, done*C_SIZEOF(dummy_real) )
          call reorder_rows2(stream, cb + delayed, ncols, gpu_u, ldL, gpu_v, ldL, &
               gpu_ind, 1)

       end if

       !
       ! the newly pivoted columns are now in columns <ib> to <jb>, where
       !
       ib = done + 1
       j = done + pivoted

       recent = recent + pivoted

       if ((j .lt. nrows) .and. (j .lt. right)) then
          !
          ! do partial forward elimination
          !
          ! L(jb + 1 : nrows, jb + 1 : right) -=
          !   L(jb + 1 : nrows, ib : jb) * LD(jb + 1 : right, ib : jb)^T

          sz = (j + done*ldL)*C_SIZEOF(dummy_real)
          gpu_u = c_ptr_plus( gpu_L, sz ) ! L(jb + 1, ib)
          gpu_v = c_ptr_plus( gpu_LD, sz ) ! LD(jb + 1, ib)
          sz = (j + j*ldL)*C_SIZEOF(dummy_real)
          gpu_w = c_ptr_plus( gpu_L, sz ) ! L(jb + 1, jb + 1)
          call cuda_dsyrk(stream, nrows - j, right - j, pivoted, -ONE, gpu_u, ldL, &
               gpu_v, ldL, ONE, gpu_w, ldL )
       end if

       done = j ! update the number of successfully processed columns

       ! step back after hitting the partial elimination margin
       ! if there have been delays, and try to pivot them again
       if ((jb .eq. right) .and. (recent .gt. 0)) then
          jb = done
          recent = 0
       end if

    end do

    if ((ncols .lt. nrows) .or. (done .eq. ncols)) then
       call custack_free(gwork, C_SIZEOF(pstat)) ! gpu_stat
       return
    end if

    sz = (done + done*ldL)*C_SIZEOF(dummy_real)
    gpu_u = c_ptr_plus( gpu_L, sz )
    gpu_v = c_ptr_plus( gpu_LD, sz )
    sz = 2*done*C_SIZEOF(dummy_real)
    gpu_w = c_ptr_plus( gpu_D, sz )
    sz = done*C_SIZEOF(dummy_real)
    gpu_p = c_ptr_plus( gpu_L, sz )
    gpu_q = c_ptr_plus( gpu_LD, sz )
    if ( done > nrows/2 ) then
       sz = done*ldL*C_SIZEOF(dummy_real)
       gpu_r = c_ptr_plus( gpu_LD, sz )
    else
       r_size = ldL*(nrows - done)*C_SIZEOF(dummy_real)
       gpu_r = custack_alloc(gwork, r_size)
    end if

    call square_ldlt(stream, nrows - done, gpu_u, gpu_v, gpu_r, gpu_w, ldL, &
         delta, eps, gpu_ind, gpu_stat)

    cuda_error = cudaMemcpyAsync_D2H(C_LOC(pstat), gpu_stat, &
         C_SIZEOF(pstat), stream)
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemcpyAsync_D2H(C_LOC(ind), gpu_ind, &
         (nrows-done)*C_SIZEOF(ind(1)), stream)
    if (cuda_error .ne. 0) return

    cuda_error = cudaStreamSynchronize(stream) ! Wait for pstat, ind
    if (cuda_error .ne. 0) return

    if (done .le. (nrows/2)) then
       call custack_free(gwork, r_size) ! gpu_r
    endif

    if (pstat .lt. (nrows - done)) return

    call reorder_rows(stream, nrows - done, done, gpu_p, ldL, gpu_q, ldL, &
         gpu_ind)

    do i = 1, nrows - done
       j = ind(i)
       ind(nrows + j) = perm(done + i)
    end do
    do i = 1, nrows - done
       perm(done + i) = ind(nrows + i)
    end do

    done = nrows

    call custack_free(gwork, C_SIZEOF(pstat)) ! gpu_stat
  end subroutine node_ldlt

!
! simultaneous ldlt factorization of several nodes
! by the same algorithm as in node_ldlt
!
! in this subroutine's comments L, LD and D refer to
! arrays containing all L-factors, all LD-products and all
! D-factors respectively stored one after another
! in the order of the corresponding elimination tree nodes
!
! as with node_ldlt, on input L = A
!
  subroutine multinode_ldlt(stream, nlvlnodes, node_m, node_n, node_lcol, &
       node_ldcol, node_skip, gpu_B, gpu_ind, delta, eps, block_size,      &
       perm, done, gwork, cublas_handle, st, cuda_error, cublas_error)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: nlvlnodes ! number of nodes at this level
    integer, intent(in) :: node_m(nlvlnodes)
    integer, intent(in) :: node_n(nlvlnodes)
    type(C_PTR), intent(in) :: node_lcol(nlvlnodes)
    type(C_PTR), intent(in) :: node_ldcol(nlvlnodes)
    integer, intent(in) :: node_skip(nlvlnodes)
    integer, intent(in) :: block_size ! same as in node_ldlt
    type(C_PTR) :: gpu_B   ! dev. pointer to a buffer
    type(C_PTR) :: gpu_ind  ! dev. pointer to the array of all pivot indices
    real(kind = wp), intent(in) :: delta, eps ! same as in node_ldlt
    integer(C_INT), intent(inout), target :: perm(*) ! array of all column indices
    integer, intent(out) :: done(nlvlnodes) ! same as done in node_ldlt per node
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    type(C_PTR), intent(in) :: cublas_handle
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error
    integer, intent(out) :: cublas_error

    integer :: nrows, ncols, rb, cb ! same as in node_ldlt
    integer :: ncb  ! n.o. CUDA blocks for block factorization
    integer :: ncbr ! same for the data reordering
    integer :: ncbe ! same for the partial elimination
    integer :: width ! front width
    integer :: node ! node number
    integer :: pivoted ! same as in node_ldlt
    integer :: step ! same as in node_ldlt

    integer, allocatable :: ib(:), jb(:), left(:), right(:) ! same as in node_ldlt
    integer(C_INT), allocatable, target :: pstat(:) ! same as in node_ldlt

    !
    ! structure for multinode factorization data
    !
    type(multinode_fact_type), dimension(:), allocatable, target :: mnfdata

    !
    ! structure for multinode reordering data
    !
    type(multireorder_data), dimension(:), allocatable, target :: mrdata

    !
    ! structure for multinode elimination data
    !
    type(multielm_data), dimension(:), allocatable, target :: medata

    integer :: j, k, l ! aux integers
    integer(C_SIZE_T) :: sz ! aux size

    type(C_PTR) :: gpu_mnfdata ! dev. pointer to a copy of mnfdata on device
    type(C_PTR) :: gpu_mrdata ! dev. pointer to reordering data
    type(C_PTR) :: gpu_medata ! dev. pointer to elimination data
    type(C_PTR) :: gpu_perm ! dev. pointer to a copy of perm on device
    type(C_PTR) :: gpu_aux ! dev. pointer to an auxiliary array
    type(C_PTR) :: gpu_u, gpu_v, gpu_w ! aux pointers
    type(C_PTR) :: gpu_stat ! dev. pointer to the array of successful pivots' no

    type(multiblock_fact_type) :: mbft_dummy ! for memory allocation only
    type(C_PTR) :: gpu_mbfdata ! dev. pointer to multi-block factorization data

    integer(C_SIZE_T) :: mrdata_size, medata_size, mbfdata_size

    type(C_PTR) :: pstat_event

    integer(C_INT) :: dummy_int
    real(wp) :: dummy_real

    st = 0; cuda_error = 0; cublas_error = 0

    gpu_aux = custack_alloc(gwork, 8*C_SIZEOF(dummy_int))
    allocate(ib(nlvlnodes), jb(nlvlnodes), pstat(nlvlnodes), left(nlvlnodes), &
         right(nlvlnodes), stat=st)
    if (st .ne. 0) return

    !
    ! compute width and ncbr
    !
    width = 0
    ncbr = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       width = width + ncols
       k = (nrows - 1)/(32*block_size) + 2 ! CUDA blocks per node, must be > 1
       ncbr = ncbr + k
    end do
    !
    ! copy perm to GPU
    !
    gpu_perm = custack_alloc(gwork, width*C_SIZEOF(perm(1)))
    cuda_error = cudaMemcpyAsync_H2D(gpu_perm, C_LOC(perm), &
         width*C_SIZEOF(perm(1)), stream);
    if (cuda_error .ne. 0) return
    !
    ! prepare data for reordering
    !
    allocate(mrdata(ncbr), stat=st)
    if (st .ne. 0) return
    mrdata_size = ncbr*C_SIZEOF(mrdata(1))
    gpu_mrdata = custack_alloc(gwork, mrdata_size)

    !  Put heavier workload blocks earlier
    ncbr = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       k = (nrows - 1)/(32*block_size) + 2
       ! Note: j=1 iterations handled in next loop, as tend to have lighter
       ! workload so want to schedule last (to better fill any "gaps")
       do j = 2, k
          mrdata(ncbr + j - 1)%node    = node - 1 ! (C) index of the processed node
          mrdata(ncbr + j - 1)%block   = j - 1 ! relative CUDA block number
          mrdata(ncbr + j - 1)%nblocks = k ! CUDA blocks for the this node
       end do
       ncbr = ncbr + k - 1
    end do
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       k = (nrows - 1)/(32*block_size) + 2
       ! Note: j=2,k handled in previous loop
       do j = 1, 1
          mrdata(ncbr + j)%node    = node - 1 ! (C) index of the processed node
          mrdata(ncbr + j)%block   = j - 1 ! relative CUDA block number
          mrdata(ncbr + j)%nblocks = k ! CUDA blocks for the this node
       end do
       ncbr = ncbr + 1
    end do
    cuda_error = cudaMemcpyAsync_H2D(gpu_mrdata, C_LOC(mrdata), mrdata_size, &
         stream)
    if (cuda_error .ne. 0) return

    !
    ! compute ncbm
    !
    ncbe = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       k = (nrows - 1)/32 + 1 ! CUDA blocks rows per this node
       l = (ncols - 1)/32 + 1 ! CUDA blocks cols per this node
       ncbe = ncbe + k*l
    end do
    !
    ! prepare data for partial elimination
    !
    allocate(medata(ncbe), stat=st)
    if (st .ne. 0) return
    medata_size = ncbe*C_SIZEOF(medata(1))
    gpu_medata = custack_alloc(gwork, medata_size)
    ncbe = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       k = (nrows - 1)/32 + 1
       l = (ncols - 1)/32 + 1
       do j = 1, k*l
          medata(ncbe + j)%node = node - 1 ! (C) index of the processed node
          medata(ncbe + j)%offb = ncbe ! the first CUDA block to process this node
       end do
       ncbe = ncbe + k*l
    end do
    cuda_error = &
         cudaMemcpyAsync_H2D(gpu_medata, C_LOC(medata), medata_size, stream)
    if (cuda_error .ne. 0) return

    !
    ! allocate data for multi-node factorization
    !
    allocate(mnfdata(nlvlnodes), stat=st)
    if (st .ne. 0) return
    gpu_mnfdata = custack_alloc(gwork, nlvlnodes*C_SIZEOF(mnfdata(1)))

    step = 24*block_size

    done = 0  ! last processed column
    ib = 1    ! first column to be processed
    jb = 0    ! last visited column

    !
    ! compute ncb and setup multi-node factorization
    !
    ncb = 0
    l = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       left(node) = 0 ! no fully eliminated nodes yet
       if (ncols .gt. (2*step)) then
          right(node) = step ! last col to be partially eliminated from
       else
          right(node) = ncols ! last col to be partially eliminated from
       end if
       nrows = node_m(node)
       mnfdata(node)%ncols = ncols
       mnfdata(node)%nrows = nrows
       mnfdata(node)%lval = node_lcol(node)
       mnfdata(node)%ldval = node_ldcol(node)
       mnfdata(node)%dval = &
            c_ptr_plus(node_lcol(node), nrows*ncols*C_SIZEOF(dummy_real))
       mnfdata(node)%offp = node_skip(node)
       mnfdata(node)%ib = 0
       mnfdata(node)%jb = 0
       mnfdata(node)%done = 0
       mnfdata(node)%rght = right(node)
       mnfdata(node)%lbuf = l
       l = l + nrows*block_size
    end do
    cuda_error = cudaMemcpyAsync_H2D(gpu_mnfdata, C_LOC(mnfdata), &
         nlvlnodes*C_SIZEOF(mnfdata(1)), stream)
    if (cuda_error .ne. 0) return
    gpu_stat = custack_alloc(gwork, nlvlnodes*C_SIZEOF(dummy_int))
    cuda_error = cudaMemsetAsync(gpu_stat, 0, nlvlnodes*C_SIZEOF(dummy_int), &
       stream)
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemsetAsync(gpu_aux, 0, 2*C_SIZEOF(dummy_int), stream)
    if (cuda_error .ne. 0) return

    ! Find max size of gpu_mbfdata and allocate it
    ncb = 0
    do node = 1, nlvlnodes
       nrows = node_m(node)
       ncb = ncb + (nrows - 1)/(block_size*(MNF_BLOCKS - 1)) + 1
    end do
    mbfdata_size = ncb*C_SIZEOF(mbft_dummy)
    gpu_mbfdata = custack_alloc(gwork, mbfdata_size)

    ! Create an event to use for synchronization
    cuda_error = cudaEventCreateWithFlags(pstat_event, cudaEventDisableTiming)
    if (cuda_error .ne. 0) return

    main_loop: do

       call multiblock_ldlt_setup(stream, nlvlnodes, gpu_mnfdata, gpu_mbfdata, &
            step, block_size, MNF_BLOCKS, gpu_stat, gpu_ind, gpu_aux)
       ncb = 0
       do node = 1, nlvlnodes
          ncols = node_n(node)
          nrows = node_m(node)
          if (jb(node) .le. ncols) ib(node) = jb(node) + 1
          if (ib(node) .gt. ncols) cycle
          if (ib(node) .gt. right(node)) then
             if ((right(node) .lt. ncols) .and. (left(node) .lt. done(node))) then
                sz = (done(node) + left(node)*nrows)*C_SIZEOF(dummy_real)
                gpu_u = c_ptr_plus(node_lcol(node), sz)
                sz = (right(node) + left(node)*nrows)*C_SIZEOF(dummy_real)
                gpu_v = c_ptr_plus(node_ldcol(node), sz)
                sz = (done(node) + right(node)*nrows)*C_SIZEOF(dummy_real)
                gpu_w = c_ptr_plus(node_lcol(node), sz)
                cublas_error = cublasDgemm(cublas_handle, 'N', 'T', nrows-done(node),&
                     ncols-right(node), done(node)-left(node), -ONE, gpu_u, nrows,     &
                     gpu_v, nrows, ONE, gpu_w, nrows)
             end if
             left(node) = done(node)
             right(node) = min(ncols, right(node) + step)
          end if
          cb = min(block_size, right(node) - ib(node) + 1)
          rb = nrows - done(node)
          jb(node) = jb(node) + cb
          ncb = ncb + (rb - cb - 1)/(block_size*(MNF_BLOCKS - 1)) + 1
       end do
       if (ncb .eq. 0) exit main_loop

       call multiblock_ldlt(stream, ncb, gpu_mbfdata, gpu_B, delta, eps, gpu_ind, &
            gpu_stat)

       cuda_error = cudaMemcpyAsync_D2H(C_LOC(pstat), gpu_stat, &
            nlvlnodes*C_SIZEOF(pstat(1)), stream)
       if (cuda_error .ne. 0) return
       cuda_error = cudaEventRecord(pstat_event, stream)
       if (cuda_error .ne. 0) return

       call multireorder(stream, ncbr, gpu_mnfdata, gpu_mrdata, gpu_B, gpu_stat, &
            gpu_ind, gpu_perm, gpu_aux)

       call cuda_multidsyrk(stream, logical(.false.,C_BOOL), ncbe, gpu_stat, &
            gpu_medata, gpu_mnfdata)

       cuda_error = cudaEventSynchronize(pstat_event)
       if (cuda_error .ne. 0) return
       do node = 1, nlvlnodes
          cb = jb(node) - ib(node) + 1
          if (cb .lt. 1) cycle
          pivoted = pstat(node)
          if (pivoted .lt. 1) cycle
          done(node) = done(node) + pivoted
          if (jb(node) .eq. right(node)) jb(node) = done(node)
       end do

    end do main_loop

    ! Destroy event now we're done with it
    cuda_error = cudaEventDestroy(pstat_event)
    if (cuda_error .ne. 0) return

    cuda_error = cudaMemcpyAsync_D2H(C_LOC(perm), gpu_perm, &
       width*C_SIZEOF(perm(1)), stream);
    if (cuda_error .ne. 0) return
    cuda_error = cudaStreamSynchronize(stream) ! Wait for perm before custack_free
    if (cuda_error .ne. 0) return

    call custack_free(gwork, mbfdata_size) ! gpu_mbfdata
    call custack_free(gwork, nlvlnodes*C_SIZEOF(dummy_int)) ! gpu_stat
    call custack_free(gwork, nlvlnodes*C_SIZEOF(mnfdata(1))) ! gpu_mnfdata
    call custack_free(gwork, medata_size) ! gpu_mdata
    call custack_free(gwork, mrdata_size) ! gpu_rdata
    call custack_free(gwork, width*C_SIZEOF(dummy_int)) ! gpu_perm
    call custack_free(gwork, 8*C_SIZEOF(dummy_int)) ! gpu_aux
  end subroutine multinode_ldlt

  ! llt factorization of one node
  subroutine node_llt(stream, nrows, ncols, gpu_L, ldL, gpu_B, block_size, &
       cublas_handle, flag, gwork, cuda_error, cublas_error)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: nrows, ncols, ldL, block_size
    type(C_PTR) :: gpu_L ! L-factor (nrows x ncols, A on input)
    type(C_PTR) :: gpu_B ! buffer
    type(C_PTR) :: cublas_handle
    integer, intent(inout) :: flag
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    integer, intent(out) :: cuda_error
    integer, intent(out) :: cublas_error

    integer :: left, right, step

    integer :: ib, jb, rb, cb

    integer(C_SIZE_T) :: sz

    integer(C_INT), target :: pstat

    type(C_PTR) :: gpu_u, gpu_v, gpu_w, gpu_stat
    real(wp) :: dummy_real

    cuda_error = 0; cublas_error = 0

    gpu_stat = custack_alloc(gwork, C_SIZEOF(pstat))

    step = 24*block_size

    left = 0
    if (ncols .gt. (2*step)) then
       right = step
    else
       right = ncols
    end if

    jb = 0 ! last visited column
    do while (jb .lt. ncols)

       ib = jb + 1 ! first columnt to be processed
       if (ib .gt. right) then

          if ((right .lt. ncols) .and. (left .lt. jb)) then
             sz = (jb + left*ldL)*C_SIZEOF(dummy_real)
             gpu_u = c_ptr_plus(gpu_L, sz)
             sz = (right + left*ldL)*C_SIZEOF(dummy_real)
             gpu_v = c_ptr_plus(gpu_L, sz)
             sz = (jb + right*ldL)*C_SIZEOF(dummy_real)
             gpu_w = c_ptr_plus(gpu_L, sz)

             cublas_error = cublasDgemm(cublas_handle, 'N', 'T', nrows-jb, &
                  ncols-right, jb-left, -ONE, gpu_u, ldL, gpu_v, ldL, ONE, gpu_w, ldL)
          end if

          left = jb
          right = min(ncols, right + step)
       end if

       rb = nrows - jb
       cb = min(block_size, ncols - ib + 1)
       jb = jb + cb                      ! last column to be processed

       ! try to pivot
       sz = (ib - 1 + (ib - 1)*ldL)*C_SIZEOF(dummy_real)
       gpu_u = c_ptr_plus( gpu_L, sz )

       call block_llt(stream, rb, cb, gpu_u, ldL, gpu_B, nrows, gpu_stat)

       cuda_error = cudaMemcpyAsync_d2h(C_LOC(pstat), gpu_stat, C_SIZEOF(pstat), &
            stream)
       if (cuda_error .ne. 0) return

       if (pstat .ne. cb) then
          flag = SSIDS_ERROR_NOT_POS_DEF ! (numerically) not positive definite
          return
       end if

       cuda_error = cudaMemcpyAsync_d2d(gpu_u, gpu_B, &
            (rb + nrows*(cb - 1))*C_SIZEOF(dummy_real), stream)
       if (cuda_error .ne. 0) return
       if ((jb .lt. nrows) .and. (jb .lt. right)) then ! do forward elimination

          sz = (jb + (ib - 1)*ldL)*C_SIZEOF(dummy_real)
          gpu_u = c_ptr_plus( gpu_L, sz ) ! L(jb + 1, ib)
          sz = (jb + jb*ldL)*C_SIZEOF(dummy_real)
          gpu_w = c_ptr_plus( gpu_L, sz ) ! L(jb + 1, jb + 1)
          call cuda_dsyrk( stream, nrows - jb, right - jb, cb, &
               -ONE, gpu_u, ldL, gpu_u, ldL, &
               ONE, gpu_w, ldL )
       end if

    end do

    call custack_free(gwork, C_SIZEOF(pstat)) ! gpu_stat
  end subroutine node_llt

! simultaneous llt factorization of several nodes
  subroutine multinode_llt(stream, nlvlnodes, node_m, node_n, node_lcol, cublas, &
       gpu_L, gpu_B, block_size, done, flag, gwork, st, cuda_error, cublas_error)
    implicit none
    type(C_PTR), intent(in) :: stream
    integer, intent(in) :: nlvlnodes
    integer, intent(in) :: node_m(nlvlnodes)
    integer, intent(in) :: node_n(nlvlnodes)
    type(C_PTR), intent(in) :: node_lcol(nlvlnodes)
    type(C_PTR) :: gpu_L ! L-factors (A on input)
    type(C_PTR) :: gpu_B
    integer, intent(in) :: block_size
    integer, intent(out) :: done(nlvlnodes)
    integer, intent(inout) :: flag
    type(cuda_stack_alloc_type), intent(inout) :: gwork
    integer, intent(out) :: st
    integer, intent(out) :: cuda_error
    integer, intent(out) :: cublas_error

    ! integer, parameter :: LD_NDATA = 10
    integer, parameter :: BLOCKS = 8

    integer :: nrows, ncols, rb, cb
    integer :: ncb, ncbr, ncbe
    integer :: width
    integer :: node, pivoted
    integer :: step

    integer, allocatable :: ib(:), jb(:), left(:), right(:)

    integer :: j, k, l

    integer(C_INT), allocatable, target :: pstat(:)

    type(multinode_fact_type), dimension(:), allocatable, target :: mnfdata

    !
    ! structure for multinode reordering data
    !
    type(multireorder_data), dimension(:), allocatable, target :: mrdata

    !
    ! structure for multinode elimination data
    !
    type(multielm_data), dimension(:), allocatable, target :: medata

    integer(C_SIZE_T) :: sz

    type(C_PTR) :: gpu_mnfdata
    type(C_PTR) :: gpu_mrdata
    type(C_PTR) :: gpu_medata
    type(C_PTR) :: gpu_aux
    type(C_PTR) :: gpu_stat
    type(C_PTR) :: gpu_u, gpu_v, gpu_w

    type(C_PTR) :: cublas

    type(multiblock_fact_type) :: mbft_dummy
    type(C_PTR) :: gpu_mbfdata

    integer(C_SIZE_T) :: mrdata_size, medata_size, mbfdata_size

    integer(C_INT) :: dummy_int
    real(wp) :: dummy_real

    cuda_error = 0; cublas_error = 0

    step = 24*block_size

    gpu_aux = custack_alloc(gwork, 8*C_SIZEOF(dummy_int))
    allocate(ib(nlvlnodes), jb(nlvlnodes), pstat(nlvlnodes), left(nlvlnodes), &
         right(nlvlnodes), stat=st)
    if (st .ne. 0) return

    width = 0
    ncbr = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       width = width + ncols
       k = (nrows - 1)/(32*block_size) + 2
       ncbr = ncbr + k
    end do
    allocate(mrdata(ncbr), stat=st)
    if (st .ne. 0) return
    mrdata_size = ncbr*C_SIZEOF(mrdata(1))
    gpu_mrdata = custack_alloc(gwork, mrdata_size)
    ncbr = 0
    l = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       k = (nrows - 1)/(32*block_size) + 2
       do j = 1, k
          mrdata(ncbr + j)%node = node - 1
          mrdata(ncbr + j)%block = j - 1
          mrdata(ncbr + j)%nblocks = k
       end do
       ncbr = ncbr + k
       l = l + nrows*block_size
    end do
    cuda_error = &
         cudaMemcpyAsync_h2d(gpu_mrdata, C_LOC(mrdata), mrdata_size, stream)
    if (cuda_error .ne. 0) return

    ncbe = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       k = (nrows - 1)/32 + 1
       l = (ncols - 1)/32 + 1
       ncbe = ncbe + k*l
    end do
    allocate(medata(ncbe), stat=st)
    if (st .ne. 0) return
    medata_size = ncbe*C_SIZEOF(medata(1))
    gpu_medata = custack_alloc(gwork, medata_size)
    ncbe = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       nrows = node_m(node)
       k = (nrows - 1)/32 + 1
       l = (ncols - 1)/32 + 1
       do j = 1, k*l
          medata(ncbe + j)%node = node - 1
          medata(ncbe + j)%offb = ncbe
       end do
       ncbe = ncbe + k*l
    end do
    cuda_error = cudaMemcpyAsync_h2d(gpu_medata, C_LOC(medata), medata_size, &
         stream)
    if (cuda_error .ne. 0) return

    allocate(mnfdata(nlvlnodes), stat=st)
    if (st .ne. 0) return
    gpu_mnfdata = custack_alloc(gwork, nlvlnodes*C_SIZEOF(mnfdata(1)))

    done = 0  ! last processed column
    ib = 1    ! first column to be processed
    jb = 0    ! last visited column

    ncb = 0
    l = 0
    do node = 1, nlvlnodes
       ncols = node_n(node)
       left(node) = 0 ! no fully eliminated nodes yet
       if (ncols .gt. (2*step)) then
          right(node) = step ! last col to be partially eliminated from
       else
          right(node) = ncols ! last col to be partially eliminated from
       end if
       nrows = node_m(node)
       mnfdata(node)%ncols = ncols
       mnfdata(node)%nrows = nrows
       mnfdata(node)%lval = node_lcol(node)
       mnfdata(node)%dval = &
            c_ptr_plus(node_lcol(node), nrows*ncols*C_SIZEOF(dummy_real))
       mnfdata(node)%offp = 0 ! never have failed pivots in posdef case
       mnfdata(node)%ib = 0
       mnfdata(node)%jb = 0
       mnfdata(node)%done = 0
       mnfdata(node)%rght = right(node)
       mnfdata(node)%lbuf = l
       l = l + nrows*block_size
    end do
    cuda_error = cudaMemcpyAsync_h2d(gpu_mnfdata, C_LOC(mnfdata), &
         nlvlnodes*C_SIZEOF(mnfdata(1)), stream)
    if (cuda_error .ne. 0) return
    gpu_stat = custack_alloc(gwork, nlvlnodes*C_SIZEOF(dummy_int))
    cuda_error = cudaMemsetAsync(gpu_stat, 0, nlvlnodes*C_SIZEOF(dummy_int), &
         stream)
    if (cuda_error .ne. 0) return
    cuda_error = cudaMemsetAsync(gpu_aux, 0, 8*C_SIZEOF(dummy_int), stream)
    if (cuda_error .ne. 0) return

    ! Find max size of gpu_mbfdata and allocate it
    ncb = 0
    do node = 1, nlvlnodes
       nrows = node_m(node)
       ncb = ncb + (nrows - 1)/(block_size*(BLOCKS - 1)) + 1
    end do
    mbfdata_size = ncb*C_SIZEOF(mbft_dummy)
    gpu_mbfdata = custack_alloc(gwork, mbfdata_size)

    main_loop: do
       call multiblock_llt_setup(stream, nlvlnodes, gpu_mnfdata, gpu_mbfdata, &
            step, block_size, BLOCKS, gpu_stat, gpu_aux)
       ncb = 0
       do node = 1, nlvlnodes
          ncols = node_n(node)
          nrows = node_m(node)
          if (jb(node) .le. ncols) ib(node) = jb(node) + 1
          if (ib(node) .gt. ncols) cycle
          if (ib(node) .gt. right(node)) then
             if ((right(node) .lt. ncols) .and. (left(node) .lt. done(node))) then
                sz = (done(node) + left(node)*nrows)*C_SIZEOF(dummy_real)
                gpu_u = c_ptr_plus(node_lcol(node), sz)
                sz = (right(node) + left(node)*nrows)*C_SIZEOF(dummy_real)
                gpu_v = c_ptr_plus(node_lcol(node), sz)
                sz = (done(node) + right(node)*nrows)*C_SIZEOF(dummy_real)
                gpu_w = c_ptr_plus(node_lcol(node), sz)
                cublas_error = cublasDgemm(cublas, 'N', 'T', nrows-done(node),    &
                     ncols-right(node), done(node)-left(node), -ONE, gpu_u, nrows,   &
                     gpu_v, nrows, ONE, gpu_w, nrows)
             end if
             left(node) = done(node)
             right(node) = min(ncols, right(node) + step)
          end if
          cb = min(block_size, right(node) - ib(node) + 1)
          rb = nrows - done(node)
          jb(node) = jb(node) + cb
          ncb = ncb + (rb - cb - 1)/(block_size*(BLOCKS - 1)) + 1
       end do
       if (ncb .eq. 0) exit main_loop

       call multiblock_llt(stream, ncb, gpu_mbfdata, gpu_B, gpu_stat)
       cuda_error = cudaMemcpyAsync_d2h(C_LOC(pstat), gpu_stat, &
            nlvlnodes*C_SIZEOF(pstat(1)), stream)
       if (cuda_error .ne. 0) return

       do node = 1, nlvlnodes
          cb = jb(node) - ib(node) + 1
          if (cb .lt. 1) cycle
          pivoted = pstat(node)
          if (pivoted .lt. cb) then
             flag = SSIDS_ERROR_NOT_POS_DEF ! (numerically) not positive definite
             return
          end if
          ib(node) = done(node) + 1
          jb(node) = done(node) + pivoted
          done(node) = jb(node)
       end do

       call multicopy(stream, ncbr, gpu_mnfdata, gpu_mrdata, gpu_L, gpu_B, &
            gpu_stat, gpu_aux)

       call cuda_multidsyrk(stream, logical(.true., C_BOOL), ncbe, gpu_stat, &
            gpu_medata, gpu_mnfdata)

    end do main_loop

    call custack_free(gwork, mbfdata_size) ! gpu_mbfdata
    call custack_free(gwork, nlvlnodes*C_SIZEOF(dummy_int)) ! gpu_stat
    call custack_free(gwork, nlvlnodes*C_SIZEOF(mnfdata(1))) ! gpu_mnfdata
    call custack_free(gwork, medata_size) ! gpu_medata
    call custack_free(gwork, mrdata_size) ! gpu_mrdata
    call custack_free(gwork, 8*C_SIZEOF(dummy_int)) ! gpu_aux
  end subroutine multinode_llt

end module spral_ssids_gpu_dense_factor
