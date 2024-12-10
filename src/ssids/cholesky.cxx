/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.0 - 2024-11-21 AT 10:20 GMT
 */

#include "ssids_cpu_kernels_cholesky.hxx"

#include <algorithm>
#include <cstdio> // FIXME: remove as only used for debug

#include "ssids_rip.hxx"
#include "ssids_profile.hxx"
#include "ssids_cpu_kernels_wrappers.hxx"

#ifdef REAL_32
#define cholesky_factor cholesky_factor_sgl
#define cholesky_solve_fwd cholesky_solve_fwd_sgl
#define cholesky_solve_bwd cholesky_solve_bwd_sgl
#elif REAL_128
#define cholesky_factor cholesky_factor_qul
#define cholesky_solve_fwd cholesky_solve_fwd_qul
#define cholesky_solve_bwd cholesky_solve_bwd_qul
#else
#define cholesky_factor cholesky_factor_dbl
#define cholesky_solve_fwd cholesky_solve_fwd_dbl
#define cholesky_solve_bwd cholesky_solve_bwd_dbl
#endif

#ifdef INTEGER_64
#define host_gemm host_gemm_64
#define lapack_potrf lapack_potrf_64
#define host_syrk host_syrk_64
#define host_trsv host_trsv_64
#define host_trsm host_trsm_64
#define gemv gemv_64
#endif

namespace spral { namespace ssids { namespace cpu {

/** Perform Cholesky factorization of lower triangular matrix a[] in place.
 * Optionally calculates the contribution block (beta*C) - LL^T.
 *
 * \param m the number of rows
 * \param n the number of columns
 * \param a the matrix to be factorized, only lower triangle is used, however
 *    upper triangle may get overwritten with rubbish
 * \param lda the leading dimension of a
 * \param beta the coefficient to multiply C by (normally 0.0 or 1.0)
 * \param upd the (m-n) x (m-n) contribution block C (may be null)
 * \param ldup the leading dimension of upd
 * \param blksz the block size to use for parallelization. Blocks are aimed to
 *    contain at most blksz**2 entries.
 * \param info is initialized to -1, and will be changed to the index of any
 *    column where a non-zero column is encountered.
 */
void cholesky_factor(ipc_ m, ipc_ n, rpc_* a, ipc_ lda, rpc_ beta,
                     rpc_* upd, ipc_ ldupd, ipc_ blksz, ipc_ *info) {
   if(n < blksz) {
      // Adjust so blocks have blksz**2 entries
      blksz = ipc_((long(blksz)*blksz) / n);
   }

   #pragma omp atomic write
   *info = -1;

   /* FIXME: Would this be better row-wise to ensure critical path, rather than
    * its current col-wise implementation ensuring maximum work available??? */
   #pragma omp taskgroup

   for(ipc_ j = 0; j < n; j += blksz) {
     ipc_ blkn = std::min(blksz, n-j);
     /* Diagonal Block Factorization Task */
     #pragma omp task default(none)                      \
        firstprivate(j, blkn)                            \
        shared(m, a, lda, blksz, info, beta, upd, ldupd) \
        depend(inout: a[j*(lda+1):1])
     {
       ipc_ my_info;
       #pragma omp atomic read
       my_info = *info;
       if (my_info == -1) {
#ifdef PROFILE
         Profile::Task task("TA_CHOL_DIAG");
#endif
         ipc_ blkm = std::min(blksz, m-j);
         ipc_ flag = lapack_potrf(FILL_MODE_LWR, blkn, &a[j*(lda+1)], lda);
         if (flag > 0) {
           // Matrix was not positive definite
           #pragma omp atomic write
           *info = flag-1; // flag uses Fortran indexing
         } else if (blkm > blkn) {
           // Diagonal block factored OK, handle some rectangular part of block
           rpc_ one_val = 1.0;
           rpc_ minus_one_val = - 1.0;
           host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT,
                     blkm-blkn, blkn, one_val, &a[j*(lda+1)], lda,
                     &a[j*(lda+1)+blkn], lda);
           if (upd) {
             rpc_ rbeta = (j==0) ? beta : 1.0;
             host_syrk(FILL_MODE_LWR, OP_N, blkm-blkn, blkn, minus_one_val,
                       &a[j*(lda+1)+blkn], lda, rbeta, upd, ldupd);
           }
         }
#ifdef PROFILE
         task.done();
#endif
       }
     }
     /* Column Solve Tasks */
     for (ipc_ i = j+blksz; i < m; i += blksz) {
       ipc_ blkm = std::min(blksz, m-i);
       #pragma omp task default(none)                        \
         firstprivate(i, j, blkn, blkm)                      \
         shared(a, lda, info, beta, upd, ldupd, blksz, n)    \
         depend(in: a[j*(lda+1):1])                          \
         depend(inout: a[j*lda + i:1])
       {
         ipc_ my_info;
         #pragma omp atomic read
         my_info = *info;
         if (my_info == -1) {
#ifdef PROFILE
           Profile::Task task("TA_CHOL_TRSM");
#endif
           rpc_ one_val = 1.0;
           rpc_ minus_one_val = - 1.0;
           host_trsm(SIDE_RIGHT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT, blkm,
                     blkn, one_val, &a[j*(lda+1)], lda, &a[j*lda+i], lda);
           if ((blkn < blksz) && upd) {
             rpc_ rbeta = (j==0) ? beta : 1.0;
             host_gemm(OP_N, OP_T, blkm, blksz-blkn, blkn, minus_one_val,
                       &a[j*lda+i], lda, &a[j*(lda+1)+blkn], lda,
                       rbeta, &upd[i-n], ldupd);
           }
#ifdef PROFILE
           task.done();
#endif
         }
       }
     }
     /* Schur Update Tasks: mostly internal */
     for (ipc_ k = j+blksz; k < n; k += blksz) {
       ipc_ blkk = std::min(blksz, n-k);
       for (ipc_ i = k; i < m; i += blksz) {
         #pragma omp task default(none)                            \
           firstprivate(i, j, k, blkn, blkk)                       \
           shared(m, a, lda, blksz, info, beta, upd, ldupd, n)     \
           depend(in: a[j*lda+k:1])                                \
           depend(in: a[j*lda+i:1])                                \
           depend(inout: a[k*lda+i:1])
         {
           ipc_ my_info;
           #pragma omp atomic read
           my_info = *info;
           if (my_info == -1) {
#ifdef PROFILE
             Profile::Task task("TA_CHOL_UPD");
#endif
             ipc_ blkm = std::min(blksz, m-i);
             rpc_ one_val = 1.0;
             rpc_ minus_one_val = - 1.0;
             host_gemm(OP_N, OP_T, blkm, blkk, blkn, minus_one_val,
                       &a[j*lda+i], lda, &a[j*lda+k], lda, one_val,
                       &a[k*lda+i], lda);
             if ((blkk < blksz) && upd) {
               rpc_ rbeta = (j==0) ? beta : 1.0;
               ipc_ upd_width = (m<k+blksz) ? blkm - blkk : blksz - blkk;
               if ((i-n) < 0) {
                 // Special case for first block of contrib
                 host_gemm(OP_N, OP_T, blkm+i-n, upd_width, blkn, minus_one_val,
                           &a[j*lda+n], lda, &a[j*lda+k+blkk], lda, rbeta,
                           upd, ldupd);
               } else {
                 host_gemm(OP_N, OP_T, blkm, upd_width, blkn, minus_one_val,
                           &a[j*lda+i], lda, &a[j*lda+k+blkk], lda, rbeta,
                           &upd[i-n], ldupd);
               }
             }
#ifdef PROFILE
             task.done();
#endif
           }
         }
       }
     }
     /* Contrib Schur complement update: external */
     if (upd) {
       for (ipc_ k = blksz*((n-1)/blksz+1); k < m; k += blksz) {
         ipc_ blkk = std::min(blksz, m-k);
         for (ipc_ i = k; i < m; i += blksz) {
           #pragma omp task default(none)                        \
             firstprivate(i, j, k, blkn, blkk)                   \
             shared(m, n, a, lda, blksz, info, beta, upd, ldupd) \
             depend(in: a[j*lda+k:1])                            \
             depend(in: a[j*lda+i:1])                            \
             depend(inout: upd[(k-n)*lda+(i-n):1])
           {
             ipc_ my_info;
             #pragma omp atomic read
             my_info = *info;
             if (my_info == -1) {
#ifdef PROFILE
               Profile::Task task("TA_CHOL_UPD");
#endif
               ipc_ blkm = std::min(blksz, m-i);
               rpc_ rbeta = (j==0) ? beta : 1.0;
               rpc_ minus_one_val = - 1.0;
               host_gemm(OP_N, OP_T, blkm, blkk, blkn, minus_one_val,
                         &a[j*lda+i], lda, &a[j*lda+k], lda,
                         rbeta, &upd[(k-n)*ldupd+(i-n)], ldupd);
#ifdef PROFILE
               task.done();
#endif
             }
           }
         }
       }
     }
   }
}

/* Forwards solve corresponding to cholesky_factor() */
void cholesky_solve_fwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
                        ipc_ nrhs, rpc_* x, ipc_ ldx) {
   rpc_ one_val = 1.0;
   rpc_ minus_one_val = - 1.0;
   if(nrhs==1) {
      host_trsv(FILL_MODE_LWR, OP_N, DIAG_NON_UNIT, n, a, lda, x, 1);
      if(m > n)
         gemv(OP_N, m-n, n, minus_one_val, &a[n], lda, x, 1, one_val, &x[n], 1);
   } else {
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_NON_UNIT, n, nrhs,
                one_val, a, lda, x, ldx);
      if(m > n)
         host_gemm(OP_N, OP_N, m-n, nrhs, n, minus_one_val, &a[n], lda, x,
                   ldx, one_val, &x[n], ldx);
   }
}

/* Backwards solve corresponding to cholesky_factor() */
void cholesky_solve_bwd(ipc_ m, ipc_ n, rpc_ const* a, ipc_ lda,
                        ipc_ nrhs, rpc_* x, ipc_ ldx) {
   rpc_ one_val = 1.0;
   rpc_ minus_one_val = - 1.0;
   if(nrhs==1) {
      if(m > n)
         gemv(OP_T, m-n, n, minus_one_val, &a[n], lda, &x[n], 1, one_val, x, 1);
      host_trsv(FILL_MODE_LWR, OP_T, DIAG_NON_UNIT, n, a, lda, x, 1);
   } else {
      if(m > n)
         host_gemm(OP_T, OP_N, n, nrhs, m-n, minus_one_val, &a[n], lda, &x[n],
                   ldx, one_val, x, ldx);
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_T, DIAG_NON_UNIT, n, nrhs,
                one_val, a, lda, x, ldx);
   }
}

}}} /* namespaces spral::ssids::cpu */
