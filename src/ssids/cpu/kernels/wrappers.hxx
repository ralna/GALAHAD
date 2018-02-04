/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include "ssids/cpu/kernels/common.hxx"

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <typename T>
void host_gemm(enum spral::ssids::cpu::operation transa, enum spral::ssids::cpu::operation transb, int m, int n, int k, T alpha, const T* a, int lda, const T* b, int ldb, T beta, T* c, int ldc);

/* _GEMV */
template <typename T>
void gemv(enum spral::ssids::cpu::operation trans, int m, int n, T alpha, const T* a, int lda, const T* x, int incx, T beta, T* y, int incy);

/* _POTRF */
template <typename T>
int lapack_potrf(enum spral::ssids::cpu::fillmode uplo, int n, T* a, int lda);

/* _SYTRF - Bunch-Kaufman factorization */
template <typename T>
int lapack_sytrf(enum spral::ssids::cpu::fillmode uplo, int n, T* a, int lda, int* ipiv, T* work, int lwork);

/* _SYRK */
template <typename T>
void host_syrk(enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation trans, int n, int k, T alpha, const T* a, int lda, T beta, T* c, int ldc);

/* _TRSV */
template <typename T>
void host_trsv(enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation trans, enum spral::ssids::cpu::diagonal diag, int n, const T* a, int lda, T* x, int incx);

/* _TRSM */
template <typename T>
void host_trsm(enum spral::ssids::cpu::side side, enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation transa, enum spral::ssids::cpu::diagonal diag, int m, int n, T alpha, const T* a, int lda, T* b, int ldb);

}}} /* namespaces spral::ssids::cpu */
