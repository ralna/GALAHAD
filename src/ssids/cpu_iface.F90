! THIS VERSION: GALAHAD 4.3 - 2024-01-27 AT 09:10 GMT.

#ifdef SPRAL_SINGLE
#ifdef SPRAL_64BIT_INTEGER
#define spral_kinds_precision spral_kinds_single_64
#define spral_ssids_cpu_iface_precision spral_ssids_cpu_iface_single_64
#define spral_ssids_inform_precision spral_ssids_inform_single_64
#define spral_ssids_types_precision spral_ssids_types_single_64
#define spral_c_gemv  spral_c_sgemv_64
#define spral_c_trsv  spral_c_strsv_64
#define spral_c_syrk  spral_c_ssyrk_64
#define spral_c_trsm  spral_c_strsm_64
#define spral_c_sytrf spral_c_ssytrf_64
#define spral_c_potrf spral_c_spotrf_64
#define spral_c_gemm  spral_c_sgemm_64
#define gemv  sgemv_64
#define trsv  strsv_64
#define syrk  ssyrk_64
#define trsm  strsm_64
#define sytrf ssytrf_64
#define potrf spotrf_64
#define gemm  sgemm_64
#else
#define spral_kinds_precision spral_kinds_single
#define spral_ssids_cpu_iface_precision spral_ssids_cpu_iface_single
#define spral_ssids_inform_precision spral_ssids_inform_single
#define spral_ssids_types_precision spral_ssids_types_single
#define spral_c_gemv  spral_c_sgemv
#define spral_c_trsv  spral_c_strsv
#define spral_c_syrk  spral_c_ssyrk
#define spral_c_trsm  spral_c_strsm
#define spral_c_sytrf spral_c_ssytrf
#define spral_c_potrf spral_c_spotrf
#define spral_c_gemm  spral_c_sgemm
#define gemv  sgemv
#define trsv  strsv
#define syrk  ssyrk
#define trsm  strsm
#define sytrf ssytrf
#define potrf spotrf
#define gemm  sgemm
#endif
#else
#ifdef SPRAL_64BIT_INTEGER
#define spral_kinds_precision spral_kinds_double_64
#define spral_ssids_cpu_iface_precision spral_ssids_cpu_iface_double_64
#define spral_ssids_inform_precision spral_ssids_inform_double_64
#define spral_ssids_types_precision spral_ssids_types_double_64
#define spral_c_gemv  spral_c_dgemv_64
#define spral_c_trsv  spral_c_dtrsv_64
#define spral_c_syrk  spral_c_dsyrk_64
#define spral_c_trsm  spral_c_dtrsm_64
#define spral_c_sytrf spral_c_dsytrf_64
#define spral_c_potrf spral_c_dpotrf_64
#define spral_c_gemm  spral_c_dgemm_64
#define gemv  dgemv_64
#define trsv  dtrsv_64
#define syrk  dsyrk_64
#define trsm  dtrsm_64
#define sytrf dsytrf_64
#define potrf dpotrf_64
#define gemm  dgemm_64
#else
#define spral_kinds_precision spral_kinds_double
#define spral_ssids_cpu_iface_precision spral_ssids_cpu_iface_double
#define spral_ssids_inform_precision spral_ssids_inform_double
#define spral_ssids_types_precision spral_ssids_types_double
#define spral_c_gemv  spral_c_dgemv
#define spral_c_trsv  spral_c_dtrsv
#define spral_c_syrk  spral_c_dsyrk
#define spral_c_trsm  spral_c_dtrsm
#define spral_c_sytrf spral_c_dsytrf
#define spral_c_potrf spral_c_dpotrf
#define spral_c_gemm  spral_c_dgemm
#define gemv  dgemv
#define trsv  dtrsv
#define syrk  dsyrk
#define trsm  dtrsm
#define sytrf dsytrf
#define potrf dpotrf
#define gemm  dgemm
#endif
#endif

!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_cpu_iface_precision
   use spral_kinds_precision
   use, intrinsic :: iso_c_binding
   use spral_ssids_types_precision, only : ssids_options
   use spral_ssids_inform_precision, only : ssids_inform
   implicit none

   private
   public :: cpu_factor_options, cpu_factor_stats
   public :: cpu_copy_options_in, cpu_copy_stats_out

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> @brief Interoperable subset of ssids_options
   !> @details Interoperates with cpu_factor_options C++ type
   !> @sa spral_ssids_types_precision::ssids_options
   !> @sa spral::ssids::cpu::cpu_factor_options
   type, bind(C) :: cpu_factor_options
      integer(C_IP_) :: print_level
      logical(C_BOOL) :: action
      real(C_RP_) :: small
      real(C_RP_) :: u
      real(C_RP_) :: multiplier
      integer(C_INT64_T) :: small_subtree_threshold
      integer(C_IP_) :: cpu_block_size
      integer(C_IP_) :: pivot_method
      integer(C_IP_) :: failed_pivot_method
   end type cpu_factor_options

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> @brief Interoperable subset of ssids_inform
   !> @details Interoperates with ThreadStats C++ type
   !> @sa spral_ssids_inform_precision::ssids_inform
   !> @sa spral::ssids::cpu::ThreadStats
   type, bind(C) :: cpu_factor_stats
      integer(C_IP_) :: flag
      integer(C_IP_) :: num_delay
      integer(C_INT64_T) :: num_factor
      integer(C_INT64_T) :: num_flops
      integer(C_IP_) :: num_neg
      integer(C_IP_) :: num_two
      integer(C_IP_) :: num_zero
      integer(C_IP_) :: maxfront
      integer(C_IP_) :: maxsupernode
      integer(C_IP_) :: not_first_pass
      integer(C_IP_) :: not_second_pass
   end type cpu_factor_stats

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Copy subset of ssids_options to interoperable type
subroutine cpu_copy_options_in(foptions, coptions)
   type(ssids_options), intent(in) :: foptions
   type(cpu_factor_options), intent(out) :: coptions

   coptions%print_level    = foptions%print_level
   coptions%action         = foptions%action
   coptions%small          = foptions%small
   coptions%u              = foptions%u
   coptions%multiplier     = foptions%multiplier
   coptions%small_subtree_threshold = foptions%small_subtree_threshold
   coptions%cpu_block_size = foptions%cpu_block_size
   coptions%pivot_method   = min(3, max(1, foptions%pivot_method))
   coptions%failed_pivot_method = min(2, max(1, foptions%failed_pivot_method))
end subroutine cpu_copy_options_in

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Copy subset of ssids_inform from interoperable type
subroutine cpu_copy_stats_out(cstats, finform)
   type(cpu_factor_stats), intent(in) :: cstats
   type(ssids_inform), intent(inout) :: finform

   ! Combine stats
   if(cstats%flag < 0) then
      finform%flag = min(finform%flag, cstats%flag) ! error
   else
      finform%flag = max(finform%flag, cstats%flag) ! success or warning
   endif
   finform%num_delay    = finform%num_delay + cstats%num_delay
   finform%num_factor   = finform%num_factor + cstats%num_factor
   finform%num_flops    = finform%num_flops + cstats%num_flops
   finform%num_neg      = finform%num_neg + cstats%num_neg
   finform%num_two      = finform%num_two + cstats%num_two
   finform%maxfront     = max(finform%maxfront, cstats%maxfront)
   finform%maxsupernode = max(finform%maxsupernode, cstats%maxsupernode)
   finform%not_first_pass = finform%not_first_pass + cstats%not_first_pass
   finform%not_second_pass = finform%not_second_pass + cstats%not_second_pass
   finform%matrix_rank  = finform%matrix_rank - cstats%num_zero
end subroutine cpu_copy_stats_out

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Wrapper functions for BLAS/LAPACK routines for standard conforming
!> interop calls from C.
subroutine spral_c_gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
bind(C)
   use spral_kinds_precision, only: C_IP_, C_RP_
   use spral_ssids_blas_iface, only : dgemm
   character(C_CHAR), intent(in) :: ta, tb
   integer(C_IP_), intent(in) :: m, n, k
   integer(C_IP_), intent(in) :: lda, ldb, ldc
   real(C_RP_), intent(in) :: alpha, beta
   real(C_RP_), intent(in   ), dimension(lda, *) :: a
   real(C_RP_), intent(in   ), dimension(ldb, *) :: b
   real(C_RP_), intent(inout), dimension(ldc, *) :: c
   call gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
end subroutine spral_c_gemm

subroutine spral_c_potrf(uplo, n, a, lda, info) bind(C)
   use spral_kinds_precision, only: C_IP_, C_RP_
   use spral_ssids_lapack_iface, only : dpotrf
   character(C_CHAR), intent(in) :: uplo
   integer(C_IP_), intent(in) :: n, lda
   integer(C_IP_), intent(out) :: info
   real(C_RP_), intent(inout), dimension(lda, *) :: a
   call potrf(uplo, n, a, lda, info)
end subroutine spral_c_potrf

subroutine spral_c_sytrf(uplo, n, a, lda, ipiv, work, lwork, info) bind(C)
   use spral_kinds_precision, only: C_IP_, C_RP_
   use spral_ssids_lapack_iface, only : dsytrf
   character(C_CHAR), intent(in) :: uplo
   integer(C_IP_), intent(in) :: n, lda, lwork
   integer(C_IP_), intent(out), dimension(n) :: ipiv
   integer(C_IP_), intent(out) :: info
   real(C_RP_), intent(inout), dimension(lda, *) :: a
   real(C_RP_), intent(out  ), dimension(*) :: work
   call sytrf(uplo, n, a, lda, ipiv, work, lwork, info)
end subroutine spral_c_sytrf

subroutine spral_c_trsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
                         ldb) bind(C)
   use spral_kinds_precision, only: C_IP_, C_RP_
   use spral_ssids_blas_iface, only : dtrsm
   character(C_CHAR), intent(in) :: side, uplo, transa, diag
   integer(C_IP_), intent(in) :: m, n, lda, ldb
   real(C_RP_), intent(in   ) :: alpha
   real(C_RP_), intent(in   ) :: a(lda, *)
   real(C_RP_), intent(inout) :: b(ldb, n)
   call trsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
end subroutine spral_c_trsm

subroutine spral_c_syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc) bind(C)
   use spral_kinds_precision, only: C_IP_, C_RP_
   use spral_ssids_blas_iface, only : dsyrk
   character(C_CHAR), intent(in) :: uplo, trans
   integer(C_IP_), intent(in) :: n, k, lda, ldc
   real(C_RP_), intent(in) :: alpha, beta
   real(C_RP_), intent(in   ), dimension(lda, *) :: a
   real(C_RP_), intent(inout), dimension(ldc, n) :: c
   call syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
end subroutine spral_c_syrk

subroutine spral_c_trsv(uplo, trans, diag, n, a, lda, x, incx) bind(C)
   use spral_kinds_precision, only: C_IP_, C_RP_
   use spral_ssids_blas_iface, only : dtrsv
   character(C_CHAR), intent(in) :: uplo, trans, diag
   integer(C_IP_), intent(in) :: n, lda, incx
   real(C_RP_), intent(in   ), dimension(lda, n) :: a
   real(C_RP_), intent(inout), dimension(*) :: x
   call trsv(uplo, trans, diag, n, a, lda, x, incx)
end subroutine spral_c_trsv

subroutine spral_c_gemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy) &
bind(C)
   use spral_kinds_precision, only: C_IP_, C_RP_
   use spral_ssids_blas_iface, only : dgemv
   character(C_CHAR), intent(in) :: trans
   integer(C_IP_), intent(in) :: m, n, lda, incx, incy
   real(C_RP_), intent(in) :: alpha, beta
   real(C_RP_), intent(in   ), dimension(lda, n) :: a
   real(C_RP_), intent(in   ), dimension(*) :: x
   real(C_RP_), intent(inout), dimension(*) :: y
   call gemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
end subroutine spral_c_gemv

end module spral_ssids_cpu_iface_precision

