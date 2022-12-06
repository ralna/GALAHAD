! Copyright (c) 2013 Science and Technology Facilities Council (STFC)
! Authors: Evgueni Ovtchinnikov and Jonathan Hogg
!
! Interoperable datatypes for passing structured data to CUDA
! (Done as separate module from spral_ssids_cuda_interfaces so we can USE it
!  in interface blocks)
module spral_ssids_gpu_datatypes
   use spral_cuda, only : cudaFree
   use, intrinsic :: iso_c_binding
   implicit none

   private
   ! Data types
   public :: load_nodes_type, assemble_cp_type, assemble_delay_type, &
      assemble_blk_type, smblk, gemv_transpose_lookup, reducing_d_solve_lookup,&
      trsv_lookup_type, scatter_lookup_type, gemv_notrans_lookup, &
      reduce_notrans_lookup, assemble_lookup_type, lookups_gpu_fwd, &
      lookups_gpu_bwd, multiblock_fact_type, multinode_fact_type, &
      cuda_stats_type, cstat_data_type, multisymm_type, &
      multiswap_type, multisyrk_type, &
      node_data, node_solve_data, multireorder_data, multielm_data, &
      assemble_lookup2_type, gpu_type, eltree_level, lookup_contrib_fwd
   ! Procedures
   public :: free_gpu_type,    &
             free_lookup_gpu       ! Cleanup data structures
   ! Constants
   public :: SLV_ASSEMBLE_NB, SLV_GEMV_NX, SLV_GEMV_NY, SLV_TRSM_TR_NBX, &
      SLV_TRSM_TR_NBY, SLV_REDUCING_D_SOLVE_THREADS_PER_BLOCK, &
      SLV_TRSV_NB_TASK, SLV_SCATTER_NB, GPU_ALIGN


   type, bind(C) :: load_nodes_type
      integer(C_INT64_T) :: nnz   ! Number of entries to map
      integer(C_INT) :: lda   ! Leading dimension of A
      integer(C_INT) :: ldl   ! Leading dimension of L
      type(C_PTR) :: lcol     ! Pointer to non-delay part of L
      integer(C_INT64_T) :: offn  ! Offset into nlist
      integer(C_INT64_T) :: offr ! Offset into rlist
   end type load_nodes_type

   type, bind(C) :: assemble_cp_type
      integer(C_INT) :: pvoffset
      type(C_PTR) :: pval
      integer(C_INT) :: ldp
      integer(C_INT) :: cm
      integer(C_INT) :: cn
      integer(C_INT) :: ldc
      integer(C_INT64_T) :: cvoffset
      type(C_PTR) :: cv
      type(C_PTR) :: rlist_direct
      type(C_PTR) :: ind
      integer(C_INT) :: sync_offset
      integer(C_INT) :: sync_wait_for
   end type assemble_cp_type

   type, bind(C) :: assemble_delay_type
      integer(C_INT) :: ndelay
      integer(C_INT) :: m
      integer(C_INT) :: n
      integer(C_INT) :: ldd
      integer(C_INT) :: lds
      type(C_PTR) :: dval
      type(C_PTR) :: sval
      integer(C_INT64_T) :: roffset
   end type assemble_delay_type

   type, bind(C) :: assemble_blk_type
      integer(C_INT) :: cp
      integer(C_INT) :: blk
   end type assemble_blk_type

   type, bind(C) :: smblk
      integer(C_INT) :: bcol
      integer(C_INT) :: blkm
      integer(C_INT) :: nelim
      type(C_PTR) :: lcol
      integer(C_SIZE_T) :: lpitch
      type(C_PTR) :: d
      type(C_PTR) :: rlist
      integer(C_SIZE_T) :: ndone
      integer(C_SIZE_T) :: upd
   end type smblk

   type, bind(C) :: gemv_transpose_lookup
      integer(C_INT) :: m
      integer(C_INT) :: n
      type(C_PTR) :: a
      integer(C_INT) :: lda
      type(C_PTR) :: rlist
      integer(C_INT) :: yoffset
   end type gemv_transpose_lookup

   type, bind(C) :: reducing_d_solve_lookup
      integer(C_INT) :: first_idx
      integer(C_INT) :: m
      integer(C_INT) :: n
      integer(C_INT) :: ldupd
      integer(C_INT) :: updoffset
      type(C_PTR) :: d
      type(C_PTR) :: perm
   end type reducing_d_solve_lookup

   type, bind(C) :: trsv_lookup_type
      integer(C_INT) :: n
      type(C_PTR) :: a
      integer(C_INT) :: lda
      integer(C_INT) :: x_offset
      integer(C_INT) :: sync_offset
   end type trsv_lookup_type

   type, bind(C) :: scatter_lookup_type
      integer(C_INT) :: n
      integer(C_INT) :: src_offset
      type(C_PTR) :: index
      integer(C_INT) :: dest_offset
   end type scatter_lookup_type

   type, bind(C) :: gemv_notrans_lookup
      integer(C_INT) :: m
      integer(C_INT) :: n
      type(C_PTR) :: a
      integer(C_INT) :: lda
      integer(C_INT) :: x_offset
      integer(C_INT) :: y_offset
   end type gemv_notrans_lookup

   type, bind(C) :: reduce_notrans_lookup
      integer(C_INT) :: m
      integer(C_INT) :: n
      integer(C_INT) :: src_offset
      integer(C_INT) :: ldsrc
      integer(C_INT) :: dest_idx
      integer(C_INT) :: dest_offset
   end type reduce_notrans_lookup

   type, bind(C) :: assemble_lookup_type
      integer(C_INT) :: m
      integer(C_INT) :: xend
      type(C_PTR) :: list
      integer(C_INT) :: x_offset
      integer(C_INT) :: contrib_idx
      integer(C_INT) :: contrib_offset
      integer(C_INT) :: nchild
      type(C_PTR) :: clen
      type(C_PTR) :: clists
      type(C_PTR) :: clists_direct
      integer(C_INT) :: cvalues_offset
      integer(C_INT) :: first
   end type assemble_lookup_type

   type, bind(C) :: assemble_lookup2_type
      integer(C_INT) :: m
      integer(C_INT) :: nelim
      integer(C_INT) :: x_offset
      type(C_PTR) :: list
      integer(C_INT) :: cvparent
      integer(C_INT) :: cvchild
      integer(C_INT) :: sync_offset
      integer(C_INT) :: sync_waitfor
   end type assemble_lookup2_type

   type, bind(C) :: lookups_gpu_fwd ! for fwd slv
      integer(C_INT) :: nassemble
      integer(C_INT) :: nasm_sync
      integer(C_INT) :: nassemble2
      integer(C_INT) :: nasmblk
      integer(C_INT) :: ntrsv
      integer(C_INT) :: ngemv
      integer(C_INT) :: nreduce
      integer(C_INT) :: nscatter
      type(C_PTR) :: assemble
      type(C_PTR) :: assemble2
      type(C_PTR) :: asmblk
      type(C_PTR) :: trsv
      type(C_PTR) :: gemv
      type(C_PTR) :: reduce
      type(C_PTR) :: scatter
   end type lookups_gpu_fwd

   type, bind(C) :: lookup_contrib_fwd ! for fwd slv, contrib to parent subtree
      integer(C_INT) :: nscatter
      type(C_PTR) :: scatter
   end type lookup_contrib_fwd

   type, bind(C) :: lookups_gpu_bwd ! for bwd slv
      integer(C_INT) :: ngemv
      integer(C_INT) :: nrds
      integer(C_INT) :: ntrsv
      integer(C_INT) :: nscatter
      type(C_PTR) :: gemv
      type(C_PTR) :: rds
      type(C_PTR) :: trsv
      type(C_PTR) :: scatter
      type(C_PTR) :: gemv_times
      type(C_PTR) :: rds_times
      type(C_PTR) :: trsv_times
      type(C_PTR) :: scatter_times
   end type lookups_gpu_bwd

   type eltree_level
      type(C_PTR) :: ptr_levL ! device pointer to L-factor level data
      integer :: lx_size
      integer :: lc_size
      integer :: ln_size
      integer :: lcc_size
      integer :: total_nch
      integer :: off_col_ind
      integer :: off_row_ind
      integer :: ncp_pre = 0
      integer :: ncb_asm_pre
      integer :: ncp_post = 0
      integer :: ncb_asm_post
      integer :: ncb_slv_n = 0
      integer :: ncb_slv_t = 0
      integer :: nimp = 0
      integer :: nexp = 0
      integer, allocatable :: import(:)
      integer, allocatable :: export(:)
      type(C_PTR) :: gpu_cpdata_pre
      type(C_PTR) :: gpu_blkdata_pre
      type(C_PTR) :: gpu_cpdata_post
      type(C_PTR) :: gpu_blkdata_post
      type(C_PTR) :: gpu_solve_n_data
      type(C_PTR) :: gpu_solve_t_data
   end type eltree_level

   type gpu_type
      integer :: n = 0
      integer :: nnodes = 0
      integer :: num_levels = 0 ! number of levels
      integer :: presolve = 0
      integer, dimension(:), allocatable :: lvlptr ! pointers into lvllist
      integer, dimension(:), allocatable :: lvllist ! list of nodes at level
      integer(C_INT64_T), dimension(:), allocatable :: off_L ! offsets for each node
      ! the following three are row offsets for independence from nrhs
      integer, dimension(:), allocatable :: off_lx ! node offsets for fwd solve
      integer, dimension(:), allocatable :: off_lc ! offsets for node contrib.
      integer, dimension(:), allocatable :: off_ln ! node offsets for bwd solve
      integer(C_INT64_T) :: rd_size = 0
      integer :: max_lx_size = 0
      integer :: max_lc_size = 0
      type(eltree_level), dimension(:), allocatable :: values_L(:) ! data
      type(C_PTR) :: gpu_rlist_direct = C_NULL_PTR
      type(C_PTR) :: gpu_col_ind = C_NULL_PTR
      type(C_PTR) :: gpu_row_ind = C_NULL_PTR
      type(C_PTR) :: gpu_diag = C_NULL_PTR
      type(C_PTR) :: gpu_sync = C_NULL_PTR

      ! Solve data (non-presolve)
      type(lookups_gpu_bwd), dimension(:), allocatable :: bwd_slv_lookup
      integer :: bwd_slv_lwork
      integer :: bwd_slv_nsync
      type(lookups_gpu_fwd), dimension(:), allocatable :: fwd_slv_lookup
      type(lookup_contrib_fwd) :: fwd_slv_contrib_lookup
      integer :: fwd_slv_lwork
      integer :: fwd_slv_nlocal
      integer :: fwd_slv_nsync
      integer :: fwd_slv_nasync
   end type gpu_type

   type, bind(C) :: multiblock_fact_type
      integer(C_INT) :: nrows
      integer(C_INT) :: ncols
      integer(C_INT) :: ld
      integer(C_INT) :: p
      type(C_PTR) :: aptr
      type(C_PTR) :: ldptr
      integer(C_INT) :: offf
      type(C_PTR) :: dptr
      integer(C_INT) :: node
      integer(C_INT) :: offb
   end type multiblock_fact_type

   type, bind(C) :: multinode_fact_type
      integer(C_INT) :: nrows
      integer(C_INT) :: ncols
      type(C_PTR) :: lval
      type(C_PTR) :: ldval
      type(C_PTR) :: dval
      integer(C_INT) :: offp
      integer(C_INT) :: ib
      integer(C_INT) :: jb
      integer(C_INT) :: done
      integer(C_INT) :: rght
      integer(C_INT) :: lbuf
   end type multinode_fact_type

   type, bind(C) :: cuda_stats_type
      integer(C_INT) :: num_two
      integer(C_INT) :: num_neg
      integer(C_INT) :: num_zero
   end type cuda_stats_type

   type, bind(C) :: cstat_data_type
      integer(C_INT) :: nelim
      type(C_PTR) :: dval
   end type cstat_data_type

   type, bind(C) :: multisymm_type
      type(C_PTR) :: lcol
      integer(C_INT) :: ncols
      integer(C_INT) :: nrows
   end type multisymm_type

   type, bind(C) :: multiswap_type
      integer(C_INT) :: nrows
      integer(C_INT) :: ncols
      integer(C_INT) :: k
      type(C_PTR) :: lcol
      integer(C_INT) :: lda
      integer(C_INT) :: off
   end type multiswap_type

   type, bind(C) :: multisyrk_type
      integer(C_INT) :: first
      type(C_PTR) :: lval
      type(C_PTR) :: ldval
      integer(C_INT64_T) :: offc
      integer(C_INT) :: n
      integer(C_INT) :: k
      integer(C_INT) :: lda
      integer(C_INT) :: ldb
   end type multisyrk_type

   type, bind(C) :: node_data
    type(C_PTR) :: ptr_v
    integer(C_INT) :: ld
    integer(C_INT) :: nrows
    integer(C_INT) :: ncols
   end type node_data

   type, bind(C) :: node_solve_data
    type(C_PTR) :: ptr_a
    type(C_PTR) :: ptr_b
    type(C_PTR) :: ptr_u
    type(C_PTR) :: ptr_v
    integer(C_INT) :: lda
    integer(C_INT) :: ldb
    integer(C_INT) :: ldu
    integer(C_INT) :: ldv
    integer(C_INT) :: nrows
    integer(C_INT) :: ncols
    integer(C_INT) :: nrhs
    integer(C_INT) :: offb
    integer(C_INT64_T) :: off_a
    integer(C_INT) :: off_b
    integer(C_INT) :: off_u
    integer(C_INT) :: off_v
   end type node_solve_data

   type, bind(C) :: multireorder_data
    integer(C_INT) :: node
    integer(C_INT) :: block
    integer(C_INT) :: nblocks
   end type multireorder_data

   type, bind(C) :: multielm_data
    integer(C_INT) :: node
    integer(C_INT) :: offb
   end type multielm_data

   ! Preprocessor constants
   integer, parameter :: SLV_ASSEMBLE_NB = 128 ! MUST be same as C #define
   integer, parameter :: SLV_GEMV_NX = 32 ! MUST be same as C #define
   integer, parameter :: SLV_GEMV_NY = 32 ! MUST be same as C #define
   integer, parameter :: SLV_TRSM_TR_NBX = 256 ! MUST be same as C #define
   integer, parameter :: SLV_TRSM_TR_NBY = 32 ! MUST be same as C #define
   integer, parameter :: SLV_REDUCING_D_SOLVE_THREADS_PER_BLOCK = 256
      ! MUST be same as C #define
   integer, parameter :: SLV_TRSV_NB_TASK = 32 ! MUST be same as C #define
   integer, parameter :: SLV_SCATTER_NB = 256 ! MUST be same as C #define

   integer, parameter :: GPU_ALIGN = 256 ! Align on this byte boundary

   interface free_lookup_gpu
      module procedure free_lookup_gpu_fwd, free_lookup_gpu_bwd
   end interface free_lookup_gpu

contains

subroutine free_gpu_type(sdata, cuda_error)
   implicit none
   type(gpu_type), intent(inout) :: sdata
   integer, intent(out) :: cuda_error

   integer :: lev
   integer :: st

   if(allocated(sdata%values_L)) then
      do lev = 1, sdata%num_levels
         cuda_error = cudaFree(sdata%values_L(lev)%ptr_levL)
         if(cuda_error.ne.0) return
         if (sdata%values_L(lev)%ncp_pre .gt. 0) then
            sdata%values_L(lev)%ncp_pre = 0
            cuda_error = cudaFree(sdata%values_L(lev)%gpu_cpdata_pre)
            if(cuda_error.ne.0) return
            cuda_error = cudaFree(sdata%values_L(lev)%gpu_blkdata_pre)
            if(cuda_error.ne.0) return
         end if
         if (sdata%values_L(lev)%ncp_post .gt. 0) then
            sdata%values_L(lev)%ncp_post = 0
            cuda_error = cudaFree(sdata%values_L(lev)%gpu_cpdata_post)
            if(cuda_error.ne.0) return
            cuda_error = cudaFree(sdata%values_L(lev)%gpu_blkdata_post)
            if(cuda_error.ne.0) return
         end if
         if (sdata%values_L(lev)%ncb_slv_n .gt. 0) then
            sdata%values_L(lev)%ncb_slv_n = 0
            cuda_error = cudaFree(sdata%values_L(lev)%gpu_solve_n_data)
            if(cuda_error.ne.0) return
         end if
         if (sdata%values_L(lev)%ncb_slv_t .gt. 0) then
            sdata%values_L(lev)%ncb_slv_t = 0
            cuda_error = cudaFree(sdata%values_L(lev)%gpu_solve_t_data)
            if(cuda_error.ne.0) return
         end if
         if (sdata%values_L(lev)%nexp .gt. 0) then
            sdata%values_L(lev)%nexp = 0
            deallocate( sdata%values_L(lev)%export )
         end if
         if (sdata%values_L(lev)%nimp .gt. 0) then
            sdata%values_L(lev)%nimp = 0
            deallocate( sdata%values_L(lev)%import )
         end if
      end do
      deallocate( sdata%values_L, stat=st )
   endif
   deallocate( sdata%lvlptr, stat=st )
   deallocate( sdata%lvllist, stat=st )
   deallocate( sdata%off_L, stat=st )
   if(allocated(sdata%bwd_slv_lookup)) then
      do lev = 1, sdata%num_levels
         call free_lookup_gpu(sdata%bwd_slv_lookup(lev), cuda_error)
         if(cuda_error.ne.0) return
         call free_lookup_gpu(sdata%fwd_slv_lookup(lev), cuda_error)
         if(cuda_error.ne.0) return
      end do
      deallocate(sdata%bwd_slv_lookup, stat=st)
      if(C_ASSOCIATED(sdata%fwd_slv_contrib_lookup%scatter)) then
         cuda_error = cudaFree(sdata%fwd_slv_contrib_lookup%scatter)
         sdata%fwd_slv_contrib_lookup%scatter = C_NULL_PTR
         if(cuda_error.ne.0) return
      endif
   endif
   if(sdata%presolve.ne.0) then
      deallocate( sdata%off_lx, stat=st )
      deallocate( sdata%off_lc, stat=st )
      deallocate( sdata%off_ln, stat=st )
      cuda_error = cudaFree(sdata%gpu_rlist_direct)
      sdata%gpu_rlist_direct = C_NULL_PTR
      if(cuda_error.ne.0) return
      cuda_error = cudaFree(sdata%gpu_sync)
      sdata%gpu_sync = C_NULL_PTR
      if(cuda_error.ne.0) return
      cuda_error = cudaFree(sdata%gpu_row_ind)
      sdata%gpu_row_ind = C_NULL_PTR
      if(cuda_error.ne.0) return
      cuda_error = cudaFree(sdata%gpu_col_ind)
      sdata%gpu_col_ind = C_NULL_PTR
      if(cuda_error.ne.0) return
      cuda_error = cudaFree(sdata%gpu_diag)
      sdata%gpu_diag = C_NULL_PTR
      if(cuda_error.ne.0) return
   end if

end subroutine free_gpu_type

subroutine free_lookup_gpu_bwd(gpul, cuda_error)
   implicit none
   type(lookups_gpu_bwd), intent(inout) :: gpul
   integer, intent(out) :: cuda_error

   ! Note: only gpul%gemv is a cudaMalloc'd address, all others are just
   ! pointer addition from that location
   if(C_ASSOCIATED(gpul%gemv)) then
      cuda_error = cudaFree(gpul%gemv); gpul%gemv = C_NULL_PTR
      if(cuda_error.ne.0) return
   endif
end subroutine free_lookup_gpu_bwd

subroutine free_lookup_gpu_fwd(gpu, cuda_error)
   implicit none
   type(lookups_gpu_fwd), intent(inout) :: gpu
   integer, intent(out) :: cuda_error

   ! Note: only gpu%assemble is a cudaMalloc'd location, others are all
   ! just pointer addition from that address
   cuda_error = cudaFree(gpu%assemble)
   if(cuda_error.ne.0) return
end subroutine free_lookup_gpu_fwd

end module spral_ssids_gpu_datatypes
