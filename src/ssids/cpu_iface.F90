! THIS VERSION: GALAHAD 5.3 - 2025-08-31 AT 10:00 GMT

!  COPYRIGHT (c) 2016 The Science and Technology Facilities Council (STFC)
!  author: Jonathan Hogg
!  licence: BSD licence, see LICENCE file for details
!  Forked and extended for GALAHAD, Nick Gould, version 3.1, 2016

#include "galahad_lapack.h"
#include "galahad_modules.h"

#ifdef REAL_32
#ifdef INTEGER_64
#define spral_c_gemv  spral_c_sgemv_64
#define spral_c_trsv  spral_c_strsv_64
#define spral_c_syrk  spral_c_ssyrk_64
#define spral_c_trsm  spral_c_strsm_64
#define spral_c_sytrf spral_c_ssytrf_64
#define spral_c_potrf spral_c_spotrf_64
#define spral_c_gemm  spral_c_sgemm_64
#define GALAHAD_BLAS_inter_precision GALAHAD_BLAS_inter_single_64
#define GALAHAD_LAPACK_inter_precision GALAHAD_LAPACK_inter_single_64
#else
#define spral_c_gemv  spral_c_sgemv
#define spral_c_trsv  spral_c_strsv
#define spral_c_syrk  spral_c_ssyrk
#define spral_c_trsm  spral_c_strsm
#define spral_c_sytrf spral_c_ssytrf
#define spral_c_potrf spral_c_spotrf
#define spral_c_gemm  spral_c_sgemm
#define GALAHAD_BLAS_inter_precision GALAHAD_BLAS_inter_single
#define GALAHAD_LAPACK_inter_precision GALAHAD_LAPACK_inter_single
#endif
#elif REAL_128
#ifdef INTEGER_64
#define spral_c_gemv  spral_c_qgemv_64
#define spral_c_trsv  spral_c_qtrsv_64
#define spral_c_syrk  spral_c_qsyrk_64
#define spral_c_trsm  spral_c_qtrsm_64
#define spral_c_sytrf spral_c_qsytrf_64
#define spral_c_potrf spral_c_qpotrf_64
#define spral_c_gemm  spral_c_qgemm_64
#define GALAHAD_BLAS_inter_precision GALAHAD_BLAS_inter_quadruple_64
#define GALAHAD_LAPACK_inter_precision GALAHAD_LAPACK_inter_quadruple_64
#else
#define spral_c_gemv  spral_c_qgemv
#define spral_c_trsv  spral_c_qtrsv
#define spral_c_syrk  spral_c_qsyrk
#define spral_c_trsm  spral_c_qtrsm
#define spral_c_sytrf spral_c_qsytrf
#define spral_c_potrf spral_c_qpotrf
#define spral_c_gemm  spral_c_qgemm
#define GALAHAD_BLAS_inter_precision GALAHAD_BLAS_inter_quadruple
#define GALAHAD_LAPACK_inter_precision GALAHAD_LAPACK_inter_quadruple
#endif
#else
#ifdef INTEGER_64
#define spral_c_gemv  spral_c_dgemv_64
#define spral_c_trsv  spral_c_dtrsv_64
#define spral_c_syrk  spral_c_dsyrk_64
#define spral_c_trsm  spral_c_dtrsm_64
#define spral_c_sytrf spral_c_dsytrf_64
#define spral_c_potrf spral_c_dpotrf_64
#define spral_c_gemm  spral_c_dgemm_64
#define GALAHAD_BLAS_inter_precision GALAHAD_BLAS_inter_double_64
#define GALAHAD_LAPACK_inter_precision GALAHAD_LAPACK_inter_double_64
#else
#define spral_c_gemv  spral_c_dgemv
#define spral_c_trsv  spral_c_dtrsv
#define spral_c_syrk  spral_c_dsyrk
#define spral_c_trsm  spral_c_dtrsm
#define spral_c_sytrf spral_c_dsytrf
#define spral_c_potrf spral_c_dpotrf
#define spral_c_gemm  spral_c_dgemm
#define GALAHAD_BLAS_inter_precision GALAHAD_BLAS_inter_double
#define GALAHAD_LAPACK_inter_precision GALAHAD_LAPACK_inter_double
#endif
#endif

 MODULE GALAHAD_SSIDS_cpu_iface_precision
   USE GALAHAD_KINDS_precision
   USE, INTRINSIC :: iso_c_binding
   USE GALAHAD_SSIDS_types_precision, ONLY: SSIDS_control_type,                &
                                            SSIDS_inform_type
   USE GALAHAD_BLAS_inter_precision, ONLY: GEMV, GEMM, TRSV, TRSM, SYRK
   USE GALAHAD_LAPACK_inter_precision, ONLY: SYTRF, POTRF
   IMPLICIT none

   PRIVATE
   PUBLIC :: cpu_factor_control, cpu_factor_stats
   PUBLIC :: cpu_copy_control_in, cpu_copy_stats_out

!  interoperable subset of ssids_control
!  Interoperates with cpu_factor_control C++ type
!  see also galahad_ssids_types_precision::ssids_control
!           galahad::ssids::cpu::cpu_factor_control

   TYPE, BIND( C ) :: cpu_factor_control
     INTEGER( KIND = C_IP_ ) :: print_level
     LOGICAL(C_BOOL) :: action
     REAL( KIND = C_RP_ ) :: small
     REAL( KIND = C_RP_ ) :: u
     REAL( KIND = C_RP_ ) :: multiplier
     INTEGER( KIND = C_INT64_T ) :: small_subtree_threshold
     INTEGER( KIND = C_IP_ ) :: cpu_block_size
     INTEGER( KIND = C_IP_ ) :: pivot_method
     INTEGER( KIND = C_IP_ ) :: failed_pivot_method
   END TYPE cpu_factor_control

!  interoperable subset of ssids_inform
!  interoperates with ThreadStats C++ type
!  see also galahad_ssids_inform_precision::ssids_inform
!           galahad::ssids::cpu::ThreadStats

   TYPE, BIND( C ) :: cpu_factor_stats
     INTEGER( KIND = C_IP_ ) :: flag
     INTEGER( KIND = C_IP_ ) :: num_delay
     INTEGER( KIND = C_INT64_T ) :: num_factor
     INTEGER( KIND = C_INT64_T ) :: num_flops
     INTEGER( KIND = C_IP_ ) :: num_neg
     INTEGER( KIND = C_IP_ ) :: num_two
     INTEGER( KIND = C_IP_ ) :: num_zero
     INTEGER( KIND = C_IP_ ) :: maxfront
     INTEGER( KIND = C_IP_ ) :: maxsupernode
     INTEGER( KIND = C_IP_ ) :: not_first_pass
     INTEGER( KIND = C_IP_ ) :: not_second_pass
   END TYPE cpu_factor_stats

 CONTAINS

   SUBROUTINE cpu_copy_control_in( fcontrol, ccontrol )

!  copy subset of ssids_control to interoperable type

   TYPE( SSIDS_control_type ), INTENT( IN ) :: fcontrol
   TYPE( cpu_factor_control ), INTENT( OUT ) :: ccontrol

   ccontrol%print_level = fcontrol%print_level
   ccontrol%action = fcontrol%action
   ccontrol%small = fcontrol%small
   ccontrol%u = fcontrol%u
   ccontrol%multiplier = fcontrol%multiplier
   ccontrol%small_subtree_threshold = fcontrol%small_subtree_threshold
   ccontrol%cpu_block_size = fcontrol%cpu_block_size
   ccontrol%pivot_method   = MIN( 3, MAX( 1, fcontrol%pivot_method ) )
   ccontrol%failed_pivot_method = MIN( 2, MAX(1, fcontrol%failed_pivot_method ))
   RETURN

   END SUBROUTINE cpu_copy_control_in

   SUBROUTINE cpu_copy_stats_out( cstats, finform )

!  copy subset of ssids_inform from interoperable type

   TYPE( cpu_factor_stats ), INTENT( IN ) :: cstats
   TYPE( SSIDS_inform_type ), INTENT( INOUT ) :: finform

   ! Combine stats
   IF ( cstats%flag < 0 ) THEN
     finform%flag = MIN( finform%flag, cstats%flag ) ! error
   ELSE
     finform%flag = MAX( finform%flag, cstats%flag ) ! success or warning
   END IF
   finform%num_delay = finform%num_delay + cstats%num_delay
   finform%num_factor = finform%num_factor + cstats%num_factor
   finform%num_flops = finform%num_flops + cstats%num_flops
   finform%num_neg = finform%num_neg + cstats%num_neg
   finform%num_two = finform%num_two + cstats%num_two
   finform%maxfront = MAX( finform%maxfront, cstats%maxfront )
   finform%maxsupernode = MAX( finform%maxsupernode, cstats%maxsupernode )
   finform%not_first_pass = finform%not_first_pass + cstats%not_first_pass
   finform%not_second_pass = finform%not_second_pass + cstats%not_second_pass
   finform%matrix_rank  = finform%matrix_rank - cstats%num_zero
   RETURN

   END SUBROUTINE cpu_copy_stats_out

!  wrapper functions for BLAS/LAPACK routines for standard conforming

   SUBROUTINE spral_c_gemm( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta,      &
                            c, ldc ) BIND( C )

!  interopability calls from C

   USE GALAHAD_KINDS_precision, only: C_IP_, C_RP_
   CHARACTER( C_CHAR ), INTENT( IN ) :: ta, tb
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: m, n, k
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: lda, ldb, ldc
   REAL( KIND = C_RP_ ), INTENT( IN ) :: alpha, beta
   REAL( KIND = C_RP_ ), INTENT( IN ), DIMENSION(lda, *) :: a
   REAL( KIND = C_RP_ ), INTENT( IN ), DIMENSION(ldb, *) :: b
   REAL( KIND = C_RP_ ), INTENT( INOUT ), DIMENSION(ldc, *) :: c
   CALL DGEMM( ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
   END SUBROUTINE spral_c_gemm

   SUBROUTINE spral_c_potrf( uplo, n, a, lda, info ) BIND( C )
   USE GALAHAD_KINDS_precision, only: C_IP_, C_RP_
   CHARACTER( C_CHAR ), INTENT( IN ) :: uplo
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: n, lda
   INTEGER( KIND = C_IP_ ), INTENT( OUT ) :: info
   REAL( KIND = C_RP_ ), INTENT( INOUT ), DIMENSION(lda, *) :: a
   CALL DPOTRF(uplo, n, a, lda, info)
   END SUBROUTINE spral_c_potrf

   SUBROUTINE spral_c_sytrf( uplo, n, a, lda, ipiv, work,                      &
                             lwork, info ) BIND( C )
   USE GALAHAD_KINDS_precision, ONLY: C_IP_, C_RP_
   CHARACTER( C_CHAR ), INTENT( IN ) :: uplo
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: n, lda, lwork
   INTEGER( KIND = C_IP_ ), INTENT( OUT ), DIMENSION(n) :: ipiv
   INTEGER( KIND = C_IP_ ), INTENT( OUT ) :: info
   REAL( KIND = C_RP_ ), INTENT( INOUT ), DIMENSION(lda, *) :: a
   REAL( KIND = C_RP_ ), INTENT( OUT ), DIMENSION(*) :: work
   CALL DSYTRF( uplo, n, a, lda, ipiv, work, lwork, info )
   END SUBROUTINE spral_c_sytrf

   SUBROUTINE spral_c_trsm( side, uplo, transa, diag, m, n, alpha, a, lda, b,  &
                            ldb ) BIND( C )
   USE GALAHAD_KINDS_precision, ONLY: C_IP_, C_RP_
   CHARACTER( C_CHAR ), INTENT( IN ) :: side, uplo, transa, diag
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: m, n, lda, ldb
   REAL( KIND = C_RP_ ), INTENT( IN ) :: alpha
   REAL( KIND = C_RP_ ), INTENT( IN ) :: a(lda, *)
   REAL( KIND = C_RP_ ), INTENT( INOUT ) :: b(ldb, n)
   CALL DTRSM( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb )
   END SUBROUTINE spral_c_trsm

   SUBROUTINE spral_c_syrk( uplo, trans, n, k, alpha, a, lda, beta,            &
                            c, ldc ) BIND(C)
   USE GALAHAD_KINDS_precision, ONLY: C_IP_, C_RP_
   CHARACTER( C_CHAR ), INTENT( IN ) :: uplo, trans
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: n, k, lda, ldc
   REAL( KIND = C_RP_ ), INTENT( IN ) :: alpha, beta
   REAL( KIND = C_RP_ ), INTENT( IN ), DIMENSION( lda, * ) :: a
   REAL( KIND = C_RP_ ), INTENT( INOUT ), DIMENSION( ldc, n ) :: c
   CALL DSYRK( uplo, trans, n, k, alpha, a, lda, beta, c, ldc )
   END SUBROUTINE spral_c_syrk

   SUBROUTINE spral_c_trsv( uplo, trans, diag, n, a, lda, x, incx ) BIND( C )
   USE GALAHAD_KINDS_precision, ONLY: C_IP_, C_RP_
   character(C_CHAR), INTENT( IN ) :: uplo, trans, diag
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: n, lda, incx
   REAL( KIND = C_RP_ ), INTENT( IN ), DIMENSION(lda, n) :: a
   REAL( KIND = C_RP_ ), INTENT( INOUT ), DIMENSION(*) :: x
   call DTRSV( uplo, trans, diag, n, a, lda, x, incx )
   END SUBROUTINE spral_c_trsv

   subroutine spral_c_gemv( trans, m, n, alpha, a, lda, x, incx, beta,        &
                            y, incy ) BIND( C )
   use GALAHAD_KINDS_precision, only: C_IP_, C_RP_
   CHARACTER( C_CHAR ), INTENT( IN ) :: trans
   INTEGER( KIND = C_IP_ ), INTENT( IN ) :: m, n, lda, incx, incy
   REAL( KIND = C_RP_ ), INTENT( IN ) :: alpha, beta
   REAL( KIND = C_RP_ ), INTENT( IN ), DIMENSION(lda, n) :: a
   REAL( KIND = C_RP_ ), INTENT( IN ), DIMENSION(*) :: x
   REAL( KIND = C_RP_ ), INTENT( INOUT ), DIMENSION(*) :: y
   CALL DGEMV( trans, m, n, alpha, a, lda, x, incx, beta, y, incy )
   END SUBROUTINE spral_c_gemv

  END MODULE GALAHAD_SSIDS_cpu_iface_precision

