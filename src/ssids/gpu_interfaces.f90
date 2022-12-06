! Copyright (c) 2013 Science and Technology Facilities Council (STFC)
! Authors: Evgueni Ovtchinnikov and Jonathan Hogg
!
! Interface definitions for CUDA kernels
module spral_ssids_gpu_interfaces
   use, intrinsic :: iso_c_binding
   use spral_cuda, only : cudaDeviceGetSharedMemConfig, &
      cudaDeviceSetSharedMemConfig, cudaSharedMemBankSizeEightByte
   implicit none

   private

   !
   ! assemble_kernels.cu
   !
   public :: &
             add_delays,    & ! Copies/expands any delayed pivots into node
             assemble,      & ! Performs assembly of non-delayed part
             load_nodes,    & ! Copies A into L (no scaling)
             load_nodes_sc, & ! Copies A into L (with scaling)
             max_abs          ! Find max absolute value in node
   interface ! assemble_kernels.cu
      subroutine add_delays(stream, ndblk, gpu_dinfo, rlist_direct) &
            bind(C, name="spral_ssids_add_delays")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), value :: ndblk
         type(C_PTR), value :: gpu_dinfo
         type(C_PTR), value :: rlist_direct
      end subroutine add_delays
      subroutine assemble(stream, nblk, blkoffset, blkdata, ncp, cpdata, &
            children, parents, gpu_next_sync) &
            bind(C, name="spral_ssids_assemble")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), value :: nblk
         integer(C_INT), value :: blkoffset
         type(C_PTR), value :: blkdata
         integer(C_INT), value :: ncp
         type(C_PTR), value :: cpdata
         type(C_PTR), value :: children
         type(C_PTR), value :: parents
         type(C_PTR), value :: gpu_next_sync ! >= (1+ncp)*sizeof(unsigned int)
      end subroutine assemble
      subroutine load_nodes(stream, nb, lndata, list, mval) &
            bind(C, name="spral_ssids_load_nodes")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: lndata
         type(C_PTR), value :: list
      type(C_PTR), value :: mval
      end subroutine load_nodes
      subroutine load_nodes_sc(stream, nb, lndata, list, rlist, scl, mval)&
            bind(C, name="spral_ssids_load_nodes_sc")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: lndata
         type(C_PTR), value :: list
         type(C_PTR), value :: rlist
         type(C_PTR), value :: scl
         type(C_PTR), value :: mval
      end subroutine load_nodes_sc
      subroutine max_abs(stream, nb, n, array, buff, maxabs) &
            bind(C, name="spral_ssids_max_abs")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         integer(C_INT64_T), intent(in), value :: n
         type(C_PTR), value :: array
         type(C_PTR), value :: buff
         type(C_PTR), value :: maxabs
      end subroutine max_abs
   end interface ! assemble_kernels.cu

   !
   ! syrk_kernels.cu
   !
   public :: cuda_dsyrk,               & ! Form Schur complement, single block
             cuda_multidsyrk,          & ! Form Schur complement, multiple blks
             cuda_multidsyrk_low_col     ! As above, high aspect ratio version
   interface ! syrk_kernels.cu
      subroutine cuda_dsyrk(stream, n, m, k, alpha, a, lda, b, ldb, beta, c, &
            ldc) bind(C, name="spral_ssids_dsyrk")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, k, lda, ldb, ldc
         real(C_DOUBLE), intent(in), value :: alpha, beta
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: c
      end subroutine cuda_dsyrk
      subroutine cuda_multidsyrk(stream, posdef, nb, stat, mdata, ndata) &
            bind(C, name="spral_ssids_multidsyrk")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         logical(C_BOOL), value :: posdef
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: stat
         type(C_PTR), value :: mdata
         type(C_PTR), value :: ndata
      end subroutine cuda_multidsyrk
      subroutine cuda_multidsyrk_low_col(stream, nb, msdata, c) &
            bind(C, name="spral_ssids_multidsyrk_low_col")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: msdata
         type(C_PTR), value :: c
      end subroutine cuda_multidsyrk_low_col
   end interface ! syrk_kernels.cu

   !
   ! dense_factor_kernels.cu
   !
   public :: block_ldlt,            & ! LDL^T kernel for single block
             block_llt,             & ! LL^T kernel for single block
             cuda_collect_stats,    & ! Accumulates statistics for a level
             multiblock_ldlt,       & ! LDL^T kernel for multiple blocks
             multiblock_ldlt_setup, & ! Sets up data for next multiblock_ldlt
             multiblock_llt,        & ! LL^T kernel for multiple blocks
             multiblock_llt_setup,  & ! Sets up data for next multiblock_llt
             square_ldlt              ! LDL^T kernel for root delays block
   interface
      subroutine block_ldlt(stream, n, m, p, a, lda, f, ldf, fd, ldfd, d, &
            delta, eps, ind, stat) bind(C, name="spral_ssids_block_ldlt")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, p, lda, ldf, ldfd
         type(C_PTR), value :: a
         type(C_PTR), value :: f
         type(C_PTR), value :: fd
         type(C_PTR), value :: d
         real(C_DOUBLE), intent(in), value :: delta, eps
         type(C_PTR), value :: ind
         type(C_PTR), value :: stat
      end subroutine block_ldlt
      subroutine block_llt(stream, n, m, a, lda, f, ldf, stat) &
            bind(C, name="spral_ssids_block_llt")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldf
         type(C_PTR), value :: a
         type(C_PTR), value :: f
         type(C_PTR), value :: stat
      end subroutine block_llt
      subroutine cuda_collect_stats(stream, nblk, csdata, custats) &
            bind(C, name="spral_ssids_collect_stats")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), value :: nblk
         type(C_PTR), value :: csdata
         type(C_PTR), value :: custats
      end subroutine cuda_collect_stats
      subroutine multiblock_ldlt(stream, nn, mbfdata, f, delta, eps, ind, &
            stat) bind(C, name="spral_ssids_multiblock_ldlt")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nn
         type(C_PTR), value :: mbfdata
         type(C_PTR), value :: f
         real(C_DOUBLE), intent(in), value :: delta, eps
         type(C_PTR), value :: ind
         type(C_PTR), value :: stat
      end subroutine multiblock_ldlt
      subroutine multiblock_ldlt_setup(stream, nb, ndata, mbfdata, &
            step, block_size, blocks, stat, ind, ncb) &
            bind(C, name="spral_ssids_multiblock_ldlt_setup")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb, step, block_size, blocks
         type(C_PTR), value :: ndata
         type(C_PTR), value :: mbfdata
         type(C_PTR), value :: stat
         type(C_PTR), value :: ind
         type(C_PTR), value :: ncb
      end subroutine multiblock_ldlt_setup
      subroutine multiblock_llt(stream, nn, mbfdata, f, stat) &
            bind(C, name="spral_ssids_multiblock_llt")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nn
         type(C_PTR), value :: mbfdata
         type(C_PTR), value :: f
         type(C_PTR), value :: stat
      end subroutine multiblock_llt
      subroutine multiblock_llt_setup(stream, nb, ndata, mbfdata, step, &
            block_size, blocks, stat, ncb) &
            bind(C, name="spral_ssids_multiblock_llt_setup")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb, step, block_size, blocks
         type(C_PTR), value :: ndata
         type(C_PTR), value :: mbfdata
         type(C_PTR), value :: stat
         type(C_PTR), value :: ncb
      end subroutine multiblock_llt_setup
      subroutine square_ldlt(stream, n, a, f, w, d, ld, delta, eps, ind, stat) &
            bind(C, name="spral_ssids_square_ldlt")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, ld
         type(C_PTR), value :: a
         type(C_PTR), value :: f
         type(C_PTR), value :: w
         type(C_PTR), value :: d
         real(C_DOUBLE), intent(in), value :: delta, eps
         type(C_PTR), value :: ind
         type(C_PTR), value :: stat
      end subroutine square_ldlt
   end interface ! dense_factor_kernels.cu

   !
   ! reorder_kernels.cu
   !
   public :: copy_ic,         & ! 2D copy with column permutation
             copy_mc,         & ! Straight forward 2d copy with mask on column
             multisymm,       & ! symmetrically fill in upper triangle
             multicopy,       & ! copies column blocks in multiple nodes
             multireorder,    & ! copies L and LD about with permutation
             reorder_cols2,   & ! in place col perm via workspace (2 arrays)
             reorder_rows,    & ! in place row permutation via workspace
             reorder_rows2,   & ! in place row perm via workspace (2 arrays)
             swap_ni2Dm,      & ! 2D swap with row and col perm (non-intersect)
             swap_ni2D_ic,    & ! 2D swap with column perm (non-intersecting)
             swap_ni2D_ir       ! 2D swap with row perm (non-intersection)
   interface ! reorder_kernels.cu
      subroutine copy_ic(stream, n, m, a, lda, b, ldb, mask) &
            bind(C, name="spral_ssids_copy_ic")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldb
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: mask
      end subroutine copy_ic
      subroutine copy_mc(stream, n, m, a, lda, b, ldb, mask) &
            bind(C, name="spral_ssids_copy_mc")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldb
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: mask
      end subroutine copy_mc
      subroutine multisymm(stream, nb, msdata) &
            bind(C, name="spral_ssids_multisymm")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: msdata
      end subroutine multisymm
      subroutine multicopy(stream, nb, ndata, idata, a, b, stat, ncb) &
             bind(C, name="spral_ssids_multicopy")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: ndata
         type(C_PTR), value :: idata
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: stat
         type(C_PTR), value :: ncb
      end subroutine multicopy
      subroutine multireorder(stream, nb, ndata, rdata, c, stat, &
           indf, indr, ncb) bind(C, name="spral_ssids_multireorder")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: ndata
         type(C_PTR), value :: rdata
         type(C_PTR), value :: c
         type(C_PTR), value :: stat
         type(C_PTR), value :: indf
         type(C_PTR), value :: indr
         type(C_PTR), value :: ncb
      end subroutine multireorder
      subroutine reorder_cols2(stream, n, m, a, lda, b, ldb, ind, mode) &
            bind(C, name="spral_ssids_reorder_cols2")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldb, mode
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: ind
      end subroutine reorder_cols2
      subroutine reorder_rows(stream, n, m, a, lda, b, ldb, ind) &
            bind(C, name="spral_ssids_reorder_rows")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldb
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: ind
      end subroutine reorder_rows
      subroutine reorder_rows2(stream, n, m, a, lda, b, ldb, ind, mode) &
            bind(C, name="spral_ssids_reorder_rows2")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldb, mode
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: ind
      end subroutine reorder_rows2
      subroutine swap_ni2Dm(stream, nb, msdata) &
            bind(C, name = "spral_ssids_swap_ni2Dm")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: nb
         type(C_PTR), value :: msdata
      end subroutine swap_ni2Dm
      subroutine swap_ni2D_ic(stream, n, m, a, lda, b, ldb, ind) &
            bind(C, name = "spral_ssids_swap_ni2D_ic")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldb
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: ind
      end subroutine swap_ni2D_ic
      subroutine swap_ni2D_ir(stream, n, m, a, lda, b, ldb, ind) &
            bind(C, name = "spral_ssids_swap_ni2D_ir")
         use, intrinsic :: iso_c_binding
         implicit none
         type(C_PTR), value :: stream
         integer(C_INT), intent(in), value :: n, m, lda, ldb
         type(C_PTR), value :: a
         type(C_PTR), value :: b
         type(C_PTR), value :: ind
      end subroutine swap_ni2D_ir
   end interface ! reorder_kernels.cu

   !
   ! solve_kernels.cu
   !
   public :: run_bwd_solve_kernels,    & ! execute prepared bwd solve
             run_d_solve_kernel,       & ! execute prepared d solve
             run_fwd_solve_kernels,    & ! execute prepared fwd solve
             run_slv_contrib_fwd         ! execute prepared scatter of contrib
   interface ! solve_kernels.cu
      subroutine run_bwd_solve_kernels(dsolve, unit_diagonal, x_gpu, work_gpu, &
            nsync, sync_gpu, gpu, stream) &
            bind(C, name="spral_ssids_run_bwd_solve_kernels")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         logical(C_BOOL), value :: dsolve
         logical(C_BOOL), value :: unit_diagonal
         type(C_PTR), value :: x_gpu
         type(C_PTR), value :: work_gpu
         integer(C_INT), value :: nsync
         type(C_PTR), value :: sync_gpu
         type(lookups_gpu_bwd), intent(in) :: gpu
         type(C_PTR), value :: stream
      end subroutine run_bwd_solve_kernels
      subroutine run_d_solve_kernel(x_gpu, y_gpu, gpu, stream) &
            bind(C, name="spral_ssids_run_d_solve_kernel")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         type(C_PTR), value :: x_gpu
         type(C_PTR), value :: y_gpu
         type(lookups_gpu_bwd), intent(in) :: gpu
         type(C_PTR), value :: stream
      end subroutine run_d_solve_kernel
      subroutine run_fwd_solve_kernels(posdef, gpu, xlocal_gpu, xstack_gpu, &
            x_gpu, cvalues_gpu, work_gpu, nsync, sync_gpu, nasm_sync,       &
            asm_sync_gpu, stream) &
            bind(C, name="spral_ssids_run_fwd_solve_kernels")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         logical(C_BOOL), value :: posdef
         type(lookups_gpu_fwd), intent(in) :: gpu
         type(C_PTR), value :: xlocal_gpu
         type(C_PTR), value :: xstack_gpu
         type(C_PTR), value :: x_gpu
         type(C_PTR), value :: cvalues_gpu
         type(C_PTR), value :: work_gpu
         integer(C_INT), value :: nsync
         type(C_PTR), value :: sync_gpu
         integer(C_INT), value :: nasm_sync
         type(C_PTR), value :: asm_sync_gpu
         type(C_PTR), value :: stream
      end subroutine run_fwd_solve_kernels
      subroutine run_slv_contrib_fwd(gpu, x_gpu, xstack_gpu, stream) &
            bind(C, name="spral_ssids_run_slv_contrib_fwd")
         use, intrinsic :: iso_c_binding
         use spral_ssids_gpu_datatypes
         type(lookup_contrib_fwd), intent(in) :: gpu
         type(C_PTR), value :: x_gpu
         type(C_PTR), value :: xstack_gpu
         type(C_PTR), value :: stream
      end subroutine run_slv_contrib_fwd
   end interface ! solve_kernels.cu

   public :: cuda_settings_type
   public :: push_ssids_cuda_settings, pop_ssids_cuda_settings

   type cuda_settings_type
      integer(C_INT) :: SharedMemConfig = -999
   end type cuda_settings_type

contains
   ! Sets device settings to desired mode for SSIDS, and stores old settings
   ! in the settings variable that can be passed to pop_ssids_cuda_settings()
   ! before returning to user code.
   subroutine push_ssids_cuda_settings(settings, cuda_error)
      type(cuda_settings_type), intent(out) :: settings
      integer, intent(out) :: cuda_error

      ! Store current settings for later restore
      cuda_error = cudaDeviceGetSharedMemConfig(settings%SharedMemConfig)
      if(cuda_error.ne.0) return

      ! Set SSIDS specific values
      !
      ! Note: The following option has no effect on Pascal and newer
      ! architectures
      cuda_error = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)
      if(cuda_error.ne.0) return

   end subroutine push_ssids_cuda_settings

   ! Restores user settings that have been stored in the settings variable by
   ! a previous call to pop_ssids_cuda_settings().
   subroutine pop_ssids_cuda_settings(settings, cuda_error)
      type(cuda_settings_type), intent(in) :: settings
      integer, intent(out) :: cuda_error

      cuda_error = cudaDeviceSetSharedMemConfig(settings%SharedMemConfig)
      if(cuda_error.ne.0) return

   end subroutine pop_ssids_cuda_settings

 end module spral_ssids_gpu_interfaces
