! THIS VERSION: GALAHAD 4.1 - 2022-12-31 AT 09:55 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_cpu_iface
   use :: GALAHAD_KINDS
   use spral_ssids_datatypes, only : ssids_options
   use spral_ssids_inform, only : ssids_inform
   implicit none

   private
   public :: cpu_factor_options, cpu_factor_stats
   public :: cpu_copy_options_in, cpu_copy_stats_out

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> @brief Interoperable subset of ssids_options
   !> @details Interoperates with cpu_factor_options C++ type
   !> @sa spral_ssids_datatypes::ssids_options
   !> @sa spral::ssids::cpu::cpu_factor_options
   type, bind(C) :: cpu_factor_options
      integer(ipc_) :: print_level
      logical(C_BOOL) :: action
      real(rpc_) :: small
      real(rpc_) :: u
      real(rpc_) :: multiplier
      integer(long_) :: small_subtree_threshold
      integer(ipc_) :: cpu_block_size
      integer(ipc_) :: pivot_method
      integer(ipc_) :: failed_pivot_method
   end type cpu_factor_options

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !> @brief Interoperable subset of ssids_inform
   !> @details Interoperates with ThreadStats C++ type
   !> @sa spral_ssids_inform::ssids_inform
   !> @sa spral::ssids::cpu::ThreadStats
   type, bind(C) :: cpu_factor_stats
      integer(ipc_) :: flag
      integer(ipc_) :: num_delay
      integer(ipc_) :: num_neg
      integer(ipc_) :: num_two
      integer(ipc_) :: num_zero
      integer(ipc_) :: maxfront
      integer(ipc_) :: not_first_pass
      integer(ipc_) :: not_second_pass
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
   finform%num_neg      = finform%num_neg + cstats%num_neg
   finform%num_two      = finform%num_two + cstats%num_two
   finform%maxfront     = max(finform%maxfront, cstats%maxfront)
   finform%not_first_pass = finform%not_first_pass + cstats%not_first_pass
   finform%not_second_pass = finform%not_second_pass + cstats%not_second_pass
   finform%matrix_rank  = finform%matrix_rank - cstats%num_zero
end subroutine cpu_copy_stats_out

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Wrapper functions for BLAS/LAPACK routines for standard conforming
!> interop calls from C.
subroutine spral_c_dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) &
bind(C)
   use spral_blas_iface, only : dgemm
   character(C_CHAR), intent(in) :: ta, tb
   integer(ipc_), intent(in) :: m, n, k
   integer(ipc_), intent(in) :: lda, ldb, ldc
   real(rpc_), intent(in) :: alpha, beta
   real(rpc_), intent(in   ), dimension(lda, *) :: a
   real(rpc_), intent(in   ), dimension(ldb, *) :: b
   real(rpc_), intent(inout), dimension(ldc, *) :: c
   call dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
end subroutine spral_c_dgemm

subroutine spral_c_dpotrf(uplo, n, a, lda, info) bind(C)
   use spral_lapack_iface, only : dpotrf
   character(C_CHAR), intent(in) :: uplo
   integer(ipc_), intent(in) :: n, lda
   integer(ipc_), intent(out) :: info
   real(rpc_), intent(inout), dimension(lda, *) :: a
   call dpotrf(uplo, n, a, lda, info)
end subroutine spral_c_dpotrf

subroutine spral_c_dsytrf(uplo, n, a, lda, ipiv, work, lwork, info) bind(C)
   use spral_lapack_iface, only : dsytrf
   character(C_CHAR), intent(in) :: uplo
   integer(ipc_), intent(in) :: n, lda, lwork
   integer(ipc_), intent(out), dimension(n) :: ipiv
   integer(ipc_), intent(out) :: info
   real(rpc_), intent(inout), dimension(lda, *) :: a
   real(rpc_), intent(out  ), dimension(*) :: work
   call dsytrf(uplo, n, a, lda, ipiv, work, lwork, info)
end subroutine spral_c_dsytrf

subroutine spral_c_dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, &
                         ldb) bind(C)
   use spral_blas_iface, only : dtrsm
   character(C_CHAR), intent(in) :: side, uplo, transa, diag
   integer(ipc_), intent(in) :: m, n, lda, ldb
   real(rpc_), intent(in   ) :: alpha
   real(rpc_), intent(in   ) :: a(lda, *)
   real(rpc_), intent(inout) :: b(ldb, n)
   call dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
end subroutine spral_c_dtrsm

subroutine spral_c_dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc) bind(C)
   use spral_blas_iface, only : dsyrk
   character(C_CHAR), intent(in) :: uplo, trans
   integer(ipc_), intent(in) :: n, k, lda, ldc
   real(rpc_), intent(in) :: alpha, beta
   real(rpc_), intent(in   ), dimension(lda, *) :: a
   real(rpc_), intent(inout), dimension(ldc, n) :: c
   call dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
end subroutine spral_c_dsyrk

subroutine spral_c_dtrsv(uplo, trans, diag, n, a, lda, x, incx) bind(C)
   use spral_blas_iface, only : dtrsv
   character(C_CHAR), intent(in) :: uplo, trans, diag
   integer(ipc_), intent(in) :: n, lda, incx
   real(rpc_), intent(in   ), dimension(lda, n) :: a
   real(rpc_), intent(inout), dimension(*) :: x
   call dtrsv(uplo, trans, diag, n, a, lda, x, incx)
end subroutine spral_c_dtrsv

subroutine spral_c_dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy) &
bind(C)
   use spral_blas_iface, only : dgemv
   character(C_CHAR), intent(in) :: trans
   integer(ipc_), intent(in) :: m, n, lda, incx, incy
   real(rpc_), intent(in) :: alpha, beta
   real(rpc_), intent(in   ), dimension(lda, n) :: a
   real(rpc_), intent(in   ), dimension(*) :: x
   real(rpc_), intent(inout), dimension(*) :: y
   call dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
end subroutine spral_c_dgemv

end module spral_ssids_cpu_iface
