/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <vector>

#include "ssids/cpu/kernels/wrappers.hxx"

namespace spral { namespace ssids { namespace cpu {

namespace verify_internal {

template <typename T>
void calcLD(int m, int n, T const* lcol, int ldl, T const* d, T* ld) {
   for(int j=0; j<n;) {
      if(j+1==n || std::isfinite(d[2*j+2])) {
         // 1x1 pivot
         // (Actually stored as D^-1 so need to invert it again)
         if(d[2*j] == 0.0) {
            // Handle zero pivots with care
            for(int i=0; i<m; i++) {
               ld[j*m+i] = 0.0;
            }
         } else {
            // Standard 1x1 pivot
            T d11 = 1/d[2*j];
            // And calulate ld
            for(int i=0; i<m; i++) {
               ld[j*m+i] = d11*lcol[j*ldl+i];
            }
         }
         // Increment j
         j++;
      } else {
         // 2x2 pivot
         // (Actually stored as D^-1 so need to invert it again)
         T di11 = d[2*j]; T di21 = d[2*j+1]; T di22 = d[2*j+3];
         T det = di11*di22 - di21*di21;
         T d11 = di22 / det; T d21 = -di21 / det; T d22 = di11 / det;
         // And calulate ld
         for(int i=0; i<m; i++) {
            ld[j*m+i]     = d11*lcol[j*ldl+i] + d21*lcol[(j+1)*ldl+i];
            ld[(j+1)*m+i] = d21*lcol[j*ldl+i] + d22*lcol[(j+1)*ldl+i];
         }
         // Increment j
         j += 2;
      }
   }
}

} /* namespace verify_internal */

template<typename T>
class Verify {
public:
   Verify(int m, int n, int const* perm, T const* a, int lda)
      : m_(m), n_(n), lda_(m), a_(m*n), perm_(n)
   {
      // Take a copy
      for(int j=0; j<n; ++j)
      for(int i=j; i<m; ++i)
         a_[j*lda_+i] = a[j*lda+i];
      for(int i=0; i<n; ++i)
         perm_[i] = perm[i];
   }

   void verify(int nelim, int const* perm, T const* l, int ldl, T const* d) const {
      printf("Verifying %d %d %d\n", m_, n_, nelim);
      if(nelim==0) return;

      // Construct lperm
      int *lperm = new int[n_];
      for(int i=0; i<n_; ++i)
         for(int j=0; j<n_; ++j)
            if(perm_[i] == perm[j]) {
               lperm[j] = i;
               break;
            }

      // Take copy of l and explicitly zero upper triangle
      T *lcopy = new T[m_*n_];
      for(int j=0; j<nelim; ++j) {
         for(int i=0; i<j; ++i)
            lcopy[j*m_+i] = 0.0;
         for(int i=j; i<m_; ++i)
            lcopy[j*m_+i] = l[j*ldl+i];
      }

      // Verify diagonal is indeed LDL^T
      T *ld = new T[nelim*nelim];
      verify_internal::calcLD(nelim, nelim, lcopy, m_, d, ld);
      T *ldlt = new T[nelim*nelim];
      host_gemm(
            OP_N, OP_T, nelim, nelim, nelim, 1.0, lcopy, m_, ld, nelim,
            0.0, ldlt, nelim
            );
      for(int j=0; j<nelim; ++j) {
         int c = lperm[j];
         for(int i=j; i<nelim; ++i) {
            int r = lperm[i];
            if(r >= c) {
               if(std::abs(a_[c*lda_+r] - ldlt[j*nelim+i]) > 1e-10) {
                  printf("Mismatch1 [%d,%d]=%e  != [%d,%d]=%e diff %e\n", r, c,
                        a_[c*lda_+r], i, j, ldlt[j*nelim+i],
                        std::abs(a_[c*lda_+r] - ldlt[j*nelim+i]));
                  exit(1);
               }
            } else {
               if(std::abs(a_[r*lda_+c] - ldlt[j*nelim+i]) > 1e-12) {
                  printf("Mismatch1 [%d,%d]=%e  != [%d,%d]=%e diff %e\n", c, r,
                        a_[r*lda_+c], i, j, ldlt[j*nelim+i],
                        std::abs(a_[r*lda_+c] - ldlt[j*nelim+i]));
                  exit(1);
               }
            }
         }
      }
      delete[] ldlt;

      // Apply pivots to block below
      if(m_ > nelim) {
         T *below = new T[(m_-nelim)*nelim];
         host_gemm(
               OP_N, OP_T, m_-nelim, nelim, nelim, 1.0, &lcopy[nelim], m_,
               ld, nelim, 0.0, below, m_-nelim
               );
         // rows nelim:n may be permuted
         for(int j=0; j<nelim; ++j) {
            int c = lperm[j];
            for(int i=nelim; i<n_; ++i) {
               int r = lperm[i];
               if(r >= c) {
                  if(std::abs(a_[c*lda_+r] - below[j*(m_-nelim)+i-nelim]) > 1e-10) {
                     printf("Mismatch2 [%d,%d]=%e  != [%d,%d]=%e diff %e\n", r, c,
                           a_[c*lda_+r], i, j, below[j*(m_-nelim)+i-nelim],
                           std::abs(a_[c*lda_+r] - below[j*(m_-nelim)+i-nelim]));
                     exit(1);
                  }
               } else {
                  if(std::abs(a_[r*lda_+c] - below[j*(m_-nelim)+i-nelim]) > 1e-12) {
                     printf("Mismatch2 [%d,%d]=%e  != [%d,%d]=%e diff %e\n", c, r,
                           a_[r*lda_+c], i, j, below[j*(m_-nelim)+i-nelim],
                           std::abs(a_[r*lda_+c] - below[j*(m_-nelim)+i-nelim]));
                     exit(1);
                  }
               }
            }
         }
         // rows nelim:n are only column permuted
         for(int j=0; j<nelim; ++j) {
            int c = lperm[j];
            for(int i=n_; i<m_; ++i) {
               int r = i;
               if(std::abs(a_[c*lda_+r] - below[j*(m_-nelim)+i-nelim]) > 1e-10) {
                  printf("Mismatch3 [%d,%d]=%e  != [%d,%d]=%e diff %e\n", r, c,
                        a_[c*lda_+r], i, j, below[j*(m_-nelim)+i-nelim],
                        std::abs(a_[c*lda_+r] - below[j*(m_-nelim)+i-nelim]));
                  exit(1);
               }
            }
         }
         delete[] below;
      }

      // release memory
      delete[] ld;
      delete[] lperm;
      delete[] lcopy;
   }

private:
   int m_;
   int n_;
   int lda_;
   std::vector<T> a_;
   std::vector<int> perm_;
};


}}} /* end of namespace spral::ssids::cpu */
