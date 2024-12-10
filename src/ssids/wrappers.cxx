/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.0 - 2024-06-11 AT 09:50 GMT
 */
#include "ssids_cpu_kernels_wrappers.hxx"

#include <stdexcept>

#ifdef REAL_32

/* ================ SINGLE PRECISION WITH 64 BIT INTEGERS =================== */

#ifdef INTEGER_64

extern "C" {
   void spral_c_sgemm_64(char* transa, char* transb, 
                         int64_t* m, int64_t* n, int64_t* k, 
                         float* alpha, const float* a, int64_t* lda, 
                         const float* b, int64_t* ldb, float *beta, 
                         float* c, int64_t* ldc);
   void spral_c_spotrf_64(char *uplo, int64_t *n, float *a, 
                          int64_t *lda, int64_t *info);
   void spral_c_ssytrf_64(char *uplo, int64_t *n, float *a, 
                          int64_t *lda, int64_t *ipiv, float *work, 
                          int64_t *lwork, int64_t *info);
   void spral_c_strsm_64(char *side, char *uplo, char *transa, 
                         char *diag, int64_t *m, int64_t *n, 
                         const float *alpha, const float *a, 
                         int64_t *lda, float *b, int64_t *ldb);
   void spral_c_ssyrk_64(char *uplo, char *trans, 
                         int64_t *n, int64_t *k, float *alpha, 
                         const float *a, int64_t *lda, float *beta, 
                         float *c, int64_t *ldc);
   void spral_c_strsv_64(char *uplo, char *trans, char *diag, 
                         int64_t *n, const float *a, int64_t *lda, 
                         float *x, int64_t *incx);
   void spral_c_sgemv_64(char *trans, int64_t *m, int64_t *n, 
                         const float* alpha, const float* a, 
                         int64_t *lda, const float* x, int64_t* incx, 
                         const float* beta, float* y, int64_t* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm_64<float>(enum spral::ssids::cpu::operation transa, 
                         enum spral::ssids::cpu::operation transb, 
                         int64_t m, int64_t n, int64_t k, float alpha, 
                         const float* a, int64_t lda, const float* b, 
                         int64_t ldb, float beta, float* c, int64_t ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_sgemm_64(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, 
                 b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv_64<float>(enum spral::ssids::cpu::operation trans, 
                    int64_t m, int64_t n, float alpha, const float* a, 
                    int64_t lda, const float* x, int64_t incx, 
                    float beta, float* y, int64_t incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_sgemv_64(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, 
                 &beta, y, &incy);
}

/* _POTRF */
template<>
int64_t lapack_potrf_64<float>(enum spral::ssids::cpu::fillmode uplo, 
                               int64_t n, float* a, int64_t lda) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int64_t info;
   spral_c_spotrf_64(&fuplo, &n, a, &lda, &info);
   return info;
}

/* _SYTRF - Bunch-Kaufman factorization */
template<>
int64_t lapack_sytrf_64<float>(enum spral::ssids::cpu::fillmode uplo, 
                               int64_t n, float* a, int64_t lda, 
                               int64_t *ipiv, float* work, int64_t lwork) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int64_t info;
   spral_c_ssytrf_64(&fuplo, &n, a, &lda, ipiv, work, &lwork, &info);
   return info;
}

/* _SYRK */
template <>
void host_syrk_64<float>(enum spral::ssids::cpu::fillmode uplo, 
                         enum spral::ssids::cpu::operation trans, 
                         int64_t n, int64_t k, float alpha, const float* a, 
                         int64_t lda, float beta, float* c, int64_t ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_ssyrk_64(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv_64<float>(enum spral::ssids::cpu::fillmode uplo, 
                         enum spral::ssids::cpu::operation trans, 
                         enum spral::ssids::cpu::diagonal diag, 
                         int64_t n, const float* a, int64_t lda, 
                         float* x, int64_t incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_strsv_64(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm_64<float>(enum spral::ssids::cpu::side side, 
                         enum spral::ssids::cpu::fillmode uplo, 
                         enum spral::ssids::cpu::operation transa, 
                         enum spral::ssids::cpu::diagonal diag, 
                         int64_t m, int64_t n, float alpha, const float* a, 
                         int64_t lda, float* b, int64_t ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_strsm_64(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, 
                 a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */

/* =========================== SINGLE PRECISION ============================= */

#else

extern "C" {
   void spral_c_sgemm(char* transa, char* transb, 
                      int* m, int* n, int* k, 
                      float* alpha, const float* a, int* lda, 
                      const float* b, int* ldb, float *beta, 
                      float* c, int* ldc);
   void spral_c_spotrf(char *uplo, int *n, float *a, 
                       int *lda, int *info);
   void spral_c_ssytrf(char *uplo, int *n, float *a, 
                       int *lda, int *ipiv, float *work, 
                       int *lwork, int *info);
   void spral_c_strsm(char *side, char *uplo, char *transa, 
                      char *diag, int *m, int *n, 
                      const float *alpha, const float *a, 
                      int *lda, float *b, int *ldb);
   void spral_c_ssyrk(char *uplo, char *trans, 
                      int *n, int *k, float *alpha, 
                      const float *a, int *lda, float *beta, 
                      float *c, int *ldc);
   void spral_c_strsv(char *uplo, char *trans, char *diag, 
                      int *n, const float *a, int *lda, 
                      float *x, int *incx);
   void spral_c_sgemv(char *trans, int *m, int *n, 
                      const float* alpha, const float* a, 
                      int *lda, const float* x, int* incx, 
                      const float* beta, float* y, int* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm<float>(enum spral::ssids::cpu::operation transa, 
                      enum spral::ssids::cpu::operation transb, 
                      int m, int n, int k, float alpha, 
                      const float* a, int lda, const float* b, 
                      int ldb, float beta, float* c, int ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_sgemm(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, 
                 b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv<float>(enum spral::ssids::cpu::operation trans, 
                 int m, int n, float alpha, const float* a, 
                 int lda, const float* x, int incx, 
                 float beta, float* y, int incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_sgemv(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, 
                 &beta, y, &incy);
}

/* _POTRF */
template<>
int lapack_potrf<float>(enum spral::ssids::cpu::fillmode uplo, 
                        int n, float* a, int lda) {
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
int lapack_sytrf<float>(enum spral::ssids::cpu::fillmode uplo, 
                        int n, float* a, int lda, 
                        int *ipiv, float* work, int lwork) {
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
void host_syrk<float>(enum spral::ssids::cpu::fillmode uplo, 
                      enum spral::ssids::cpu::operation trans, 
                      int n, int k, float alpha, const float* a, 
                      int lda, float beta, float* c, int ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_ssyrk(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv<float>(enum spral::ssids::cpu::fillmode uplo, 
                      enum spral::ssids::cpu::operation trans, 
                      enum spral::ssids::cpu::diagonal diag, 
                      int n, const float* a, int lda, 
                      float* x, int incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_strsv(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm<float>(enum spral::ssids::cpu::side side, 
                      enum spral::ssids::cpu::fillmode uplo, 
                      enum spral::ssids::cpu::operation transa, 
                      enum spral::ssids::cpu::diagonal diag, 
                      int m, int n, float alpha, const float* a, 
                      int lda, float* b, int ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_strsm(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, 
                 a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */

#endif

#elif REAL_128

/* ============== QUADRUPLE PRECISION WITH 64 BIT INTEGERS ================= */

#ifdef INTEGER_64

extern "C" {
   void spral_c_qgemm_64(char* transa, char* transb, 
                         int64_t* m, int64_t* n, int64_t* k, 
                         __float128* alpha, const __float128* a, int64_t* lda, 
                         const __float128* b, int64_t* ldb, __float128 *beta, 
                         __float128* c, int64_t* ldc);
   void spral_c_qpotrf_64(char *uplo, int64_t *n, __float128 *a, 
                          int64_t *lda, int64_t *info);
   void spral_c_qsytrf_64(char *uplo, int64_t *n, __float128 *a, 
                          int64_t *lda, int64_t *ipiv, __float128 *work, 
                          int64_t *lwork, int64_t *info);
   void spral_c_qtrsm_64(char *side, char *uplo, char *transa, 
                         char *diag, int64_t *m, int64_t *n, 
                         const __float128 *alpha, const __float128 *a, 
                         int64_t *lda, __float128 *b, int64_t *ldb);
   void spral_c_qsyrk_64(char *uplo, char *trans, 
                         int64_t *n, int64_t *k, __float128 *alpha, 
                         const __float128 *a, int64_t *lda, __float128 *beta, 
                         __float128 *c, int64_t *ldc);
   void spral_c_qtrsv_64(char *uplo, char *trans, char *diag, 
                         int64_t *n, const __float128 *a, int64_t *lda, 
                         __float128 *x, int64_t *incx);
   void spral_c_qgemv_64(char *trans, int64_t *m, int64_t *n, 
                         const __float128* alpha, const __float128* a, 
                         int64_t *lda, const __float128* x, int64_t* incx, 
                         const __float128* beta, __float128* y, int64_t* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm_64<__float128>(enum spral::ssids::cpu::operation transa, 
                         enum spral::ssids::cpu::operation transb, 
                         int64_t m, int64_t n, int64_t k, __float128 alpha, 
                         const __float128* a, int64_t lda, const __float128* b, 
                         int64_t ldb, __float128 beta, __float128* c, 
                         int64_t ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_qgemm_64(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, 
                 b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv_64<__float128>(enum spral::ssids::cpu::operation trans, 
                    int64_t m, int64_t n, __float128 alpha, 
                    const __float128* a, 
                    int64_t lda, const __float128* x, int64_t incx, 
                    __float128 beta, __float128* y, int64_t incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_qgemv_64(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, 
                 &beta, y, &incy);
}

/* _POTRF */
template<>
int64_t lapack_potrf_64<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                               int64_t n, __float128* a, int64_t lda) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int64_t info;
   spral_c_qpotrf_64(&fuplo, &n, a, &lda, &info);
   return info;
}

/* _SYTRF - Bunch-Kaufman factorization */
template<>
int64_t lapack_sytrf_64<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                               int64_t n, __float128* a, int64_t lda, 
                               int64_t *ipiv, __float128* work, int64_t lwork) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int64_t info;
   spral_c_qsytrf_64(&fuplo, &n, a, &lda, ipiv, work, &lwork, &info);
   return info;
}

/* _SYRK */
template <>
void host_syrk_64<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                         enum spral::ssids::cpu::operation trans, 
                         int64_t n, int64_t k, __float128 alpha, 
                         const __float128* a, int64_t lda, __float128 beta, 
                         __float128* c, int64_t ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_qsyrk_64(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv_64<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                         enum spral::ssids::cpu::operation trans, 
                         enum spral::ssids::cpu::diagonal diag, 
                         int64_t n, const __float128* a, int64_t lda, 
                         __float128* x, int64_t incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_qtrsv_64(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm_64<__float128>(enum spral::ssids::cpu::side side, 
                         enum spral::ssids::cpu::fillmode uplo, 
                         enum spral::ssids::cpu::operation transa, 
                         enum spral::ssids::cpu::diagonal diag, 
                         int64_t m, int64_t n, __float128 alpha, 
                         const __float128* a, int64_t lda, __float128* b, 
                         int64_t ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_qtrsm_64(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, 
                 a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */

/* ========================= QUADRUPLE PRECISION =========================== */

#else

extern "C" {
   void spral_c_qgemm(char* transa, char* transb, 
                      int* m, int* n, int* k, 
                      __float128* alpha, const __float128* a, int* lda, 
                      const __float128* b, int* ldb, __float128 *beta, 
                      __float128* c, int* ldc);
   void spral_c_qpotrf(char *uplo, int *n, __float128 *a, 
                       int *lda, int *info);
   void spral_c_qsytrf(char *uplo, int *n, __float128 *a, 
                       int *lda, int *ipiv, __float128 *work, 
                       int *lwork, int *info);
   void spral_c_qtrsm(char *side, char *uplo, char *transa, 
                      char *diag, int *m, int *n, 
                      const __float128 *alpha, const __float128 *a, 
                      int *lda, __float128 *b, int *ldb);
   void spral_c_qsyrk(char *uplo, char *trans, 
                      int *n, int *k, __float128 *alpha, 
                      const __float128 *a, int *lda, __float128 *beta, 
                      __float128 *c, int *ldc);
   void spral_c_qtrsv(char *uplo, char *trans, char *diag, 
                      int *n, const __float128 *a, int *lda, 
                      __float128 *x, int *incx);
   void spral_c_qgemv(char *trans, int *m, int *n, 
                      const __float128* alpha, const __float128* a, 
                      int *lda, const __float128* x, int* incx, 
                      const __float128* beta, __float128* y, int* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm<__float128>(enum spral::ssids::cpu::operation transa, 
                      enum spral::ssids::cpu::operation transb, 
                      int m, int n, int k, __float128 alpha, 
                      const __float128* a, int lda, const __float128* b, 
                      int ldb, __float128 beta, __float128* c, int ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_qgemm(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, 
                 b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv<__float128>(enum spral::ssids::cpu::operation trans, 
                 int m, int n, __float128 alpha, const __float128* a, 
                 int lda, const __float128* x, int incx, 
                 __float128 beta, __float128* y, int incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_qgemv(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, 
                 &beta, y, &incy);
}

/* _POTRF */
template<>
int lapack_potrf<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                        int n, __float128* a, int lda) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int info;
   spral_c_qpotrf(&fuplo, &n, a, &lda, &info);
   return info;
}

/* _SYTRF - Bunch-Kaufman factorization */
template<>
int lapack_sytrf<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                        int n, __float128* a, int lda, 
                        int *ipiv, __float128* work, int lwork) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int info;
   spral_c_qsytrf(&fuplo, &n, a, &lda, ipiv, work, &lwork, &info);
   return info;
}

/* _SYRK */
template <>
void host_syrk<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                      enum spral::ssids::cpu::operation trans, 
                      int n, int k, __float128 alpha, const __float128* a, 
                      int lda, __float128 beta, __float128* c, int ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_qsyrk(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv<__float128>(enum spral::ssids::cpu::fillmode uplo, 
                      enum spral::ssids::cpu::operation trans, 
                      enum spral::ssids::cpu::diagonal diag, 
                      int n, const __float128* a, int lda, 
                      __float128* x, int incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_qtrsv(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm<__float128>(enum spral::ssids::cpu::side side, 
                      enum spral::ssids::cpu::fillmode uplo, 
                      enum spral::ssids::cpu::operation transa, 
                      enum spral::ssids::cpu::diagonal diag, 
                      int m, int n, __float128 alpha, const __float128* a, 
                      int lda, __float128* b, int ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_qtrsm(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, 
                 a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */

#endif



















#else

/* ================ DOUBLE PRECISION WITH 64 BIT INTEGERS =================== */

#ifdef INTEGER_64

extern "C" {
   void spral_c_dgemm_64(char* transa, char* transb, 
                         int64_t* m, int64_t* n, int64_t* k, 
                         double* alpha, const double* a, int64_t* lda, 
                         const double* b, int64_t* ldb, double *beta, 
                         double* c, int64_t* ldc);
   void spral_c_dpotrf_64(char *uplo, int64_t *n, double *a, 
                          int64_t *lda, int64_t *info);
   void spral_c_dsytrf_64(char *uplo, int64_t *n, double *a, 
                          int64_t *lda, int64_t *ipiv, double *work, 
                          int64_t *lwork, int64_t *info);
   void spral_c_dtrsm_64(char *side, char *uplo, char *transa, 
                         char *diag, int64_t *m, int64_t *n, 
                         const double *alpha, const double *a, 
                         int64_t *lda, double *b, int64_t *ldb);
   void spral_c_dsyrk_64(char *uplo, char *trans, 
                         int64_t *n, int64_t *k, double *alpha, 
                         const double *a, int64_t *lda, double *beta, 
                         double *c, int64_t *ldc);
   void spral_c_dtrsv_64(char *uplo, char *trans, char *diag, 
                         int64_t *n, const double *a, int64_t *lda, 
                         double *x, int64_t *incx);
   void spral_c_dgemv_64(char *trans, int64_t *m, int64_t *n, 
                         const double* alpha, const double* a, 
                         int64_t *lda, const double* x, int64_t* incx, 
                         const double* beta, double* y, int64_t* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm_64<double>(enum spral::ssids::cpu::operation transa, 
                          enum spral::ssids::cpu::operation transb, 
                          int64_t m, int64_t n, int64_t k, double alpha, 
                          const double* a, int64_t lda, const double* b, 
                          int64_t ldb, double beta, double* c, int64_t ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dgemm_64(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, 
                 b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv_64<double>(enum spral::ssids::cpu::operation trans, 
                     int64_t m, int64_t n, double alpha, const double* a, 
                     int64_t lda, const double* x, int64_t incx, 
                     double beta, double* y, int64_t incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dgemv_64(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, 
                 &beta, y, &incy);
}

/* _POTRF */
template<>
int64_t lapack_potrf_64<double>(enum spral::ssids::cpu::fillmode uplo, 
                                int64_t n, double* a, int64_t lda) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int64_t info;
   spral_c_dpotrf_64(&fuplo, &n, a, &lda, &info);
   return info;
}

/* _SYTRF - Bunch-Kaufman factorization */
template<>
int64_t lapack_sytrf_64<double>(enum spral::ssids::cpu::fillmode uplo, 
                                int64_t n, double* a, 
                                int64_t lda, int64_t *ipiv, 
                                double* work, int64_t lwork) {
   char fuplo;
   switch(uplo) {
      case spral::ssids::cpu::FILL_MODE_LWR: fuplo = 'L'; break;
      case spral::ssids::cpu::FILL_MODE_UPR: fuplo = 'U'; break;
      default: throw std::runtime_error("Unknown fill mode");
   }
   int64_t info;
   spral_c_dsytrf_64(&fuplo, &n, a, &lda, ipiv, work, &lwork, &info);
   return info;
}

/* _SYRK */
template <>
void host_syrk_64<double>(enum spral::ssids::cpu::fillmode uplo, 
                          enum spral::ssids::cpu::operation trans, 
                          int64_t n, int64_t k, double alpha, const double* a, 
                          int64_t lda, double beta, double* c, int64_t ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dsyrk_64(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv_64<double>(enum spral::ssids::cpu::fillmode uplo, 
                          enum spral::ssids::cpu::operation trans, 
                          enum spral::ssids::cpu::diagonal diag, 
                          int64_t n, const double* a, int64_t lda, 
                          double* x, int64_t incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_dtrsv_64(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm_64<double>(enum spral::ssids::cpu::side side, 
                          enum spral::ssids::cpu::fillmode uplo, 
                          enum spral::ssids::cpu::operation transa, 
                          enum spral::ssids::cpu::diagonal diag, 
                          int64_t m, int64_t n, double alpha, const double* a, 
                          int64_t lda, double* b, int64_t ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_dtrsm_64(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, 
                 a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */

/* =========================== DOUBLE PRECISION ============================= */

#else

extern "C" {
   void spral_c_dgemm(char* transa, char* transb, 
                      int* m, int* n, int* k, 
                      double* alpha, const double* a, int* lda, 
                      const double* b, int* ldb, double *beta, 
                      double* c, int* ldc);
   void spral_c_dpotrf(char *uplo, int *n, double *a, 
                       int *lda, int *info);
   void spral_c_dsytrf(char *uplo, int *n, double *a, 
                       int *lda, int *ipiv, double *work, 
                       int *lwork, int *info);
   void spral_c_dtrsm(char *side, char *uplo, char *transa, 
                      char *diag, int *m, int *n, 
                      const double *alpha, const double *a, 
                      int *lda, double *b, int *ldb);
   void spral_c_dsyrk(char *uplo, char *trans, 
                      int *n, int *k, double *alpha, 
                      const double *a, int *lda, double *beta, 
                      double *c, int *ldc);
   void spral_c_dtrsv(char *uplo, char *trans, char *diag, 
                      int *n, const double *a, int *lda, 
                      double *x, int *incx);
   void spral_c_dgemv(char *trans, int *m, int *n, 
                      const double* alpha, const double* a, 
                      int *lda, const double* x, int* incx, 
                      const double* beta, double* y, int* incy);
}

namespace spral { namespace ssids { namespace cpu {

/* _GEMM */
template <>
void host_gemm<double>(enum spral::ssids::cpu::operation transa, 
                       enum spral::ssids::cpu::operation transb, 
                       int m, int n, int k, double alpha, 
                       const double* a, int lda, const double* b, 
                       int ldb, double beta, double* c, int ldc) {
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char ftransb = (transb==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dgemm(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, 
                 b, &ldb, &beta, c, &ldc);
}

/* _GEMV */
template <>
void gemv<double>(enum spral::ssids::cpu::operation trans, 
                 int m, int n, double alpha, const double* a, 
                 int lda, const double* x, int incx, 
                 double beta, double* y, int incy) {
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dgemv(&ftrans, &m, &n, &alpha, a, &lda, x, &incx, 
                 &beta, y, &incy);
}

/* _POTRF */
template<>
int lapack_potrf<double>(enum spral::ssids::cpu::fillmode uplo, 
                         int n, double* a, int lda) {
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
int lapack_sytrf<double>(enum spral::ssids::cpu::fillmode uplo, 
                         int n, double* a, int lda, int *ipiv, 
                         double* work, int lwork) {
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
void host_syrk<double>(enum spral::ssids::cpu::fillmode uplo, 
                       enum spral::ssids::cpu::operation trans, 
                       int n, int k, double alpha, const double* a, 
                       int lda, double beta, double* c, int ldc) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   spral_c_dsyrk(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

/* _TRSV */
template <>
void host_trsv<double>(enum spral::ssids::cpu::fillmode uplo, 
                       enum spral::ssids::cpu::operation trans, 
                       enum spral::ssids::cpu::diagonal diag, 
                       int n, const double* a, int lda, 
                       double* x, int incx) {
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftrans = (trans==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_dtrsv(&fuplo, &ftrans, &fdiag, &n, a, &lda, x, &incx);
}

/* _TRSM */
template <>
void host_trsm<double>(enum spral::ssids::cpu::side side, 
                       enum spral::ssids::cpu::fillmode uplo, 
                       enum spral::ssids::cpu::operation transa, 
                       enum spral::ssids::cpu::diagonal diag, 
                       int m, int n, double alpha, const double* a, 
                       int lda, double* b, int ldb) {
   char fside = (side==spral::ssids::cpu::SIDE_LEFT) ? 'L' : 'R';
   char fuplo = (uplo==spral::ssids::cpu::FILL_MODE_LWR) ? 'L' : 'U';
   char ftransa = (transa==spral::ssids::cpu::OP_N) ? 'N' : 'T';
   char fdiag = (diag==spral::ssids::cpu::DIAG_UNIT) ? 'U' : 'N';
   spral_c_dtrsm(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, 
                 a, &lda, b, &ldb);
}

}}} /* namespaces spral::ssids::cpu */
#endif
#endif











