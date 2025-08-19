/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 08:00 GMT
 */

#pragma once

#include <stdint.h>

#include "ssids_routines.h"
#include "ssids_rip.hxx"
#include "ssids_cpu_kernels_common.hxx"

namespace galahad { namespace ssids { namespace cpu {

/* _GEMM */
template <typename T>
void host_gemm(enum galahad::ssids::cpu::operation transa, 
               enum galahad::ssids::cpu::operation transb, 
               int m, int n, int k, T alpha, const T* a, 
               int lda, const T* b, int ldb, T beta, 
               T* c, int ldc);

/* _GEMV */
template <typename T>
void gemv(enum galahad::ssids::cpu::operation trans, 
         int m, int n, T alpha, const T* a, int lda, 
         const T* x, int incx, T beta, T* y, int incy);

/* _POTRF */
template <typename T>
int lapack_potrf(enum galahad::ssids::cpu::fillmode uplo, int n, 
                 T* a, int lda);

/* _SYTRF - Bunch-Kaufman factorization */
template <typename T>
int lapack_sytrf(enum galahad::ssids::cpu::fillmode uplo, 
                 int n, T* a, int lda, int* ipiv, 
                 T* work, int lwork);

/* _SYRK */
template <typename T>
void host_syrk(enum galahad::ssids::cpu::fillmode uplo, 
              enum galahad::ssids::cpu::operation trans, 
              int n, int k, T alpha, const T* a, int lda, 
              T beta, T* c, int ldc);

/* _TRSV */
template <typename T>
void host_trsv(enum galahad::ssids::cpu::fillmode uplo, 
               enum galahad::ssids::cpu::operation trans, 
               enum galahad::ssids::cpu::diagonal diag, 
               int n, const T* a, int lda, T* x, int incx);

/* _TRSM */
template <typename T>
void host_trsm(enum galahad::ssids::cpu::side side, 
               enum galahad::ssids::cpu::fillmode uplo, 
               enum galahad::ssids::cpu::operation transa, 
               enum galahad::ssids::cpu::diagonal diag, 
               int m, int n, T alpha, const T* a, int lda, 
               T* b, int ldb);

/* _GEMM_64 */
template <typename T>
void host_gemm_64(enum galahad::ssids::cpu::operation transa, 
               enum galahad::ssids::cpu::operation transb, 
               longc_ m, longc_ n, longc_ k, T alpha, const T* a, 
               longc_ lda, const T* b, longc_ ldb, T beta, 
               T* c, longc_ ldc);

/* _GEMV_64 */
template <typename T>
void gemv_64(enum galahad::ssids::cpu::operation trans, 
         longc_ m, longc_ n, T alpha, const T* a, longc_ lda, 
         const T* x, longc_ incx, T beta, T* y, longc_ incy);

/* _POTRF_64 */
template <typename T>
longc_ lapack_potrf_64(enum galahad::ssids::cpu::fillmode uplo, longc_ n, 
                 T* a, longc_ lda);

/* _SYTRF_64 - Bunch-Kaufman factorization */
template <typename T>
longc_ lapack_sytrf_64(enum galahad::ssids::cpu::fillmode uplo, 
                 longc_ n, T* a, longc_ lda, longc_* ipiv, 
                 T* work, longc_ lwork);

/* _SYRK_64 */
template <typename T>
void host_syrk_64(enum galahad::ssids::cpu::fillmode uplo, 
              enum galahad::ssids::cpu::operation trans, 
              longc_ n, longc_ k, T alpha, const T* a, longc_ lda, 
              T beta, T* c, longc_ ldc);

/* _TRSV_64 */
template <typename T>
void host_trsv_64(enum galahad::ssids::cpu::fillmode uplo, 
               enum galahad::ssids::cpu::operation trans, 
               enum galahad::ssids::cpu::diagonal diag, 
               longc_ n, const T* a, longc_ lda, T* x, longc_ incx);

/* _TRSM_64 */
template <typename T>
void host_trsm_64(enum galahad::ssids::cpu::side side, 
               enum galahad::ssids::cpu::fillmode uplo, 
               enum galahad::ssids::cpu::operation transa, 
               enum galahad::ssids::cpu::diagonal diag, 
               longc_ m, longc_ n, T alpha, const T* a, longc_ lda, 
               T* b, longc_ ldb);

}}} /* namespaces galahad::ssids::cpu */

