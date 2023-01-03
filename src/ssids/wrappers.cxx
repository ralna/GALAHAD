/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#include "ssids_cpu_kernels_wrappers.hxx"

#include <stdexcept>

#ifdef SPRAL_SINGLE
extern "C" {
   void spral_c_sgemm(char* transa, char* transb, int* m, int* n, int* k, float* alpha, const float* a, int* lda, const float* b, int* ldb, float *beta, float* c, int* ldc);
   void spral_c_spotrf(char *uplo, int *n, float *a, int *lda, int *info);
   void spral_c_ssytrf(char *uplo, int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);
   void spral_c_strsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const float *alpha, const float *a, int *lda, float *b, int *ldb);
   void spral_c_ssyrk(char *uplo, char *trans, int *n, int *k, float *alpha, const float *a, int *lda, float *beta, float *c, int *ldc);
   void spral_c_strsv(char *uplo, char *trans, char *diag, int *n, const float *a, int *lda, float *x, int *incx);
   void spral_c_sgemv(char *trans, int *m, int *n, const float* alpha, const float* a, int *lda, const float* x, int* incx, const float* beta, float* y, int* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm<float>(enum spral::ssids::cpu::operation transa, enum spral::ssids::cpu::operation transb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_sgemm(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv<float>(enum spral::ssids::cpu::operation trans, int m, int n, float alpha, const float* a, int lda, const float* x, int incx, float beta, float* y, int incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_sgemv(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

/* _POTRF */
template<>
int lapack_potrf<float>(enum spral::ssids::cpu::fillmode uplo, int n, float* a, int lda) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int info;
   spral_c_spotrf(&fuplo, &n, a, &lda, &info);
   return info;
}

/* _SYTRF - Bunch-Kaufman factorization */
template<>
int lapack_sytrf<float>(enum spral::ssids::cpu::fillmode uplo, int n, float* a, int lda, int *ipiv, float* work, int lwork) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int info;
   spral_c_ssytrf(&fuplo, &n, a, &lda, ipiv, work, &lwork, &info);
   return info;
}

/* _SYRK */
template <>
void host_syrk<float>(enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation trans, int n, int k, float alpha, const float* a, int lda, float beta, float* c, int ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_ssyrk(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv<float>(enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation trans, enum spral::ssids::cpu::diagonal diag, int n, const float* a, int lda, float* x, int incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_strsv(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm<float>(enum spral::ssids::cpu::side side, enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation transa, enum spral::ssids::cpu::diagonal diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_strsm(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */

#else

extern "C" {
   void spral_c_dgemm(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double *beta, double* c, int* ldc);
   void spral_c_dpotrf(char *uplo, int *n, double *a, int *lda, int *info);
   void spral_c_dsytrf(char *uplo, int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
   void spral_c_dtrsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const double *alpha, const double *a, int *lda, double *b, int *ldb);
   void spral_c_dsyrk(char *uplo, char *trans, int *n, int *k, double *alpha, const double *a, int *lda, double *beta, double *c, int *ldc);
   void spral_c_dtrsv(char *uplo, char *trans, char *diag, int *n, const double *a, int *lda, double *x, int *incx);
   void spral_c_dgemv(char *trans, int *m, int *n, const double* alpha, const double* a, int *lda, const double* x, int* incx, const double* beta, double* y, int* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm<double>(enum spral::ssids::cpu::operation transa, enum spral::ssids::cpu::operation transb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dgemm(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv<double>(enum spral::ssids::cpu::operation trans, int m, int n, double alpha, const double* a, int lda, const double* x, int incx, double beta, double* y, int incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dgemv(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

/* _POTRF */
template<>
int lapack_potrf<double>(enum spral::ssids::cpu::fillmode uplo, int n, double* a, int lda) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int info;
   spral_c_dpotrf(&fuplo, &n, a, &lda, &info);
   return info;
}

/* _SYTRF - Bunch-Kaufman factorization */
template<>
int lapack_sytrf<double>(enum spral::ssids::cpu::fillmode uplo, int n, double* a, int lda, int *ipiv, double* work, int lwork) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int info;
   spral_c_dsytrf(&fuplo, &n, a, &lda, ipiv, work, &lwork, &info);
   return info;
}

/* _SYRK */
template <>
void host_syrk<double>(enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation trans, int n, int k, double alpha, const double* a, int lda, double beta, double* c, int ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dsyrk(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv<double>(enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation trans, enum spral::ssids::cpu::diagonal diag, int n, const double* a, int lda, double* x, int incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_dtrsv(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm<double>(enum spral::ssids::cpu::side side, enum spral::ssids::cpu::fillmode uplo, enum spral::ssids::cpu::operation transa, enum spral::ssids::cpu::diagonal diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_dtrsm(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */
#endif

