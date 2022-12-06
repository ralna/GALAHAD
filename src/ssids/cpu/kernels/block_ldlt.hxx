/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <cstdlib> // FIXME: remove debug?
#include <limits>

#include "ssids/cpu/ThreadStats.hxx"
#include "ssids/cpu/kernels/SimdVec.hxx"

namespace spral { namespace ssids { namespace cpu {
namespace block_ldlt_internal {

/** Swaps two columns of A */
/* NB: ldwork only well defined for c<idx1 */
template<typename T, int BLOCK_SIZE>
void swap_cols(int idx1, int idx2, int n, T *a, int lda, T *ldwork, int *perm) {
   if(idx1==idx2) return; // noop

   /* Ensure wlog idx1 < idx2 */
   if(idx1 > idx2) {
      int temp = idx1;
      idx1 = idx2;
      idx2 = temp;
   }

   /* Swap perm */
   if(perm) {
      int temp = perm[idx1];
      perm[idx1] = perm[idx2];
      perm[idx2] = temp;
   }

   /* Swap ldwork */
   if(ldwork) {
      for(int c=0; c<idx1; c++) {
         T temp = ldwork[c*BLOCK_SIZE+idx1];
         ldwork[c*BLOCK_SIZE+idx1] = ldwork[c*BLOCK_SIZE+idx2];
         ldwork[c*BLOCK_SIZE+idx2] = temp;
      }
   }

   /* Swap row portions */
   for(int c=0; c<idx1; c++) {
      T temp = a[c*lda+idx1];
      a[c*lda+idx1] = a[c*lda+idx2];
      a[c*lda+idx2] = temp;
   }

   /* Swap row of idx2 with col of idx1 */
   for(int i=idx1+1; i<idx2; i++) {
      T temp = a[idx1*lda+i];
      a[idx1*lda+i] = a[i*lda+idx2];
      a[i*lda+idx2] = temp;
   }

   /* Swap diagonals */
   {
      T temp = a[idx1*(lda+1)];
      a[idx1*(lda+1)] = a[idx2*(lda+1)];
      a[idx2*(lda+1)] = temp;
   }

   /* Swap col portions */
   for(int r=idx2+1; r<n; r++) {
      T temp = a[idx1*lda+r];
      a[idx1*lda+r] = a[idx2*lda+r];
      a[idx2*lda+r] = temp;
   }
}


template <typename T, int BLOCK_SIZE>
void find_maxloc(const int from, const T *a, int lda, T &bestv_out, int &rloc, int &cloc) {
   typedef SimdVec<T> SimdVecT;

   /* Handle special cases:
    *    1) block size less than vector length
    *    2) block size not multiple of twice vector length
    */
   if(   BLOCK_SIZE < SimdVecT::vector_length ||
         BLOCK_SIZE % (2*SimdVecT::vector_length) != 0) {
      T bestv = -1.0;
      rloc = BLOCK_SIZE; cloc = BLOCK_SIZE;
      for(int c=from; c<BLOCK_SIZE; c++) {
         for(int r=c; r<BLOCK_SIZE; r++) {
            double v = a[c*lda+r];
            if(fabs(v) > bestv) {
               bestv = fabs(v);
               rloc = r;
               cloc = c;
            }
         }
      }
      bestv_out =
         (cloc < BLOCK_SIZE && rloc < BLOCK_SIZE) ? a[cloc*lda+rloc]
                                                  : 0.0;
      return;
   }

   // Define a union that lets us abuse T to store ints and still use
   // avx blend.
   union intT {
      int i;
      T d;
   };

   // Initialize best in lane vars to value 0.0 and position INT_MAX,INT_MAX
   SimdVecT bestv(-1.0);
   SimdVecT bestv2(-1.0);
   intT imax;
   imax.i = std::numeric_limits<int>::max();
   SimdVecT bestr(imax.d);
   SimdVecT bestr2(imax.d);
   SimdVecT bestc(imax.d);
   SimdVecT bestc2(imax.d);
   // Loop over array at stride equal to vector length
   for(int c=from; c<BLOCK_SIZE; c++) {
      // Coerce c to be treated as a T then scatter it
      intT c_d;
      c_d.i = c;
      SimdVecT c_vec(c_d.d);
      // First iteration must be careful as we only want the lower triangle
      const int vlen = SimdVecT::vector_length;
      {
         intT r_d;
         r_d.i = vlen *(c / vlen);
         SimdVecT r_vec(r_d.d);
         // Load vector of values, taking absolute value
         SimdVecT v = fabs(SimdVecT::load_aligned(&a[c*lda+r_d.i]));
         // Compare against best in lane
         SimdVecT v_gt_bestv = (v > bestv);
         v_gt_bestv = v_gt_bestv & SimdVecT::gt_mask(c%vlen);
         // If better, update best in lane
         bestv = blend(bestv, v, v_gt_bestv);
         bestr = blend(bestr, r_vec, v_gt_bestv);
         bestc = blend(bestc, c_vec, v_gt_bestv);
      }
      // Handle any second part of the first 2*vlen chunk
      if(vlen*(c/vlen + 1) < 2*vlen*(c/(2*vlen) + 1)) {
         intT r_d;
         r_d.i = vlen *(c/vlen + 1);
         SimdVecT r_vec(r_d.d);
         // Load vector of values, taking absolute value
         SimdVecT v = fabs(SimdVecT::load_aligned(&a[c*lda+r_d.i]));
         // If better, update best in lane
         SimdVecT v_gt_bestv = (v > bestv);
         bestv = blend(bestv, v, v_gt_bestv);
         bestr = blend(bestr, r_vec, v_gt_bestv);
         bestc = blend(bestc, c_vec, v_gt_bestv);
      }
      // Remaining iterations can use full vector with unroll of 2
      intT r_d, r_d2;
      for(r_d.i=2*vlen*(c/(2*vlen) + 1); r_d.i<BLOCK_SIZE; r_d.i+=2*vlen) {
         r_d2.i = r_d.i + vlen;
         // Coerce r to be treated as a T then scatter it
         SimdVecT r_vec(r_d.d);
         SimdVecT r_vec2(r_d2.d);
         // Load vector of values, taking absolute value
         SimdVecT v = fabs(SimdVecT::load_aligned(&a[c*lda+r_d.i]));
         SimdVecT v2 = fabs(SimdVecT::load_aligned(&a[c*lda+r_d2.i]));
         // If better, update best in lane
         SimdVecT v_gt_bestv = (v > bestv);
         bestv = blend(bestv, v, v_gt_bestv);
         bestr = blend(bestr, r_vec, v_gt_bestv);
         bestc = blend(bestc, c_vec, v_gt_bestv);
         SimdVecT v_gt_bestv2 = (v2 > bestv2);
         bestv2 = blend(bestv2, v2, v_gt_bestv2);
         bestr2 = blend(bestr2, r_vec2, v_gt_bestv2);
         bestc2 = blend(bestc2, c_vec, v_gt_bestv2);
      }
   }
   // Merge bestv and bestv2
   SimdVecT v_gt_bestv = (bestv2 > bestv);
   bestv = blend(bestv, bestv2, v_gt_bestv);
   bestr = blend(bestr, bestr2, v_gt_bestv);
   bestc = blend(bestc, bestc2, v_gt_bestv);
   // Extract results
#if defined(__AVX512F__)
   T __attribute__((aligned(64))) bv2[SimdVecT::vector_length];
   intT __attribute__((aligned(64))) br2[SimdVecT::vector_length], bc2[SimdVecT::vector_length];
#elif defined(__AVX__)
   T __attribute__((aligned(32))) bv2[SimdVecT::vector_length];
   intT __attribute__((aligned(32))) br2[SimdVecT::vector_length], bc2[SimdVecT::vector_length];
#else
   T __attribute__((aligned(16))) bv2[SimdVecT::vector_length];
   intT __attribute__((aligned(16))) br2[SimdVecT::vector_length], bc2[SimdVecT::vector_length];
#endif
   bestv.store_aligned(bv2);
   bestr.store_aligned(&br2[0].d);
   bestc.store_aligned(&bc2[0].d);
   bestv_out = bv2[0];
   rloc = br2[0].i;
   cloc = bc2[0].i;
   for(int i=1; i<SimdVecT::vector_length; i++) {
      if(bv2[i] > bestv_out) {
         bestv_out = bv2[i];
         rloc = br2[i].i + i; // NB rloc only stores base of vector, so need +i
         cloc = bc2[i].i;
      }
   }
   bestv_out = a[cloc*lda+rloc];
}

/** Returns true if a 2x2 pivot can be stably inverted.
 *
 * We assume that a21 is the maximum entry in the matrix, at which stage
 * the proofs in doc/LDLT.tex apply.
 */
template <typename T>
bool test_2x2(T a11, T a21, T a22, T &detpiv, T &detscale) {
   detscale = 1.0/fabs(a21); // |a21|=max(|a11|, |a21|, |a22|) by construction
   detpiv = (a11*detscale) * a22 - fabs(a21);
   return (fabs(detpiv) >= fabs(a21)/2);
}

/** Updates the trailing submatrix (2x2 case) */
template <typename T, int BLOCK_SIZE>
void update_2x2(int p, T *a, int lda, const T *ld) {
   for(int c=p+2; c<BLOCK_SIZE; c++) {
      #pragma omp simd
      for(int r=c; r<BLOCK_SIZE; r++) {
         a[c*lda+r] -= ld[c]*a[p*lda+r] + ld[BLOCK_SIZE+c]*a[(p+1)*lda+r];
      }
   }
}

/** Updates the trailing submatrix (1x1 case) */
template <typename T, int BLOCK_SIZE>
void update_1x1(int p, T *a, int lda, const T *ld) {
#if 0
   for(int c=p+1; c<BLOCK_SIZE; c++)
      #pragma omp simd
      for(int r=c; r<BLOCK_SIZE; r++)
         a[c*lda+r] -= ld[c]*a[p*lda+r];
#else
   const int vlen = SimdVec<T>::vector_length;
   const int unroll=4; // How many iteration of loop we're doing

   // Handle case of small BLOCK_SIZE safely
   if(BLOCK_SIZE < vlen || BLOCK_SIZE%vlen != 0 || BLOCK_SIZE < unroll) {
      for(int c=p+1; c<BLOCK_SIZE; c++)
         for(int r=c; r<BLOCK_SIZE; r++)
            a[c*lda+r] -= ld[c]*a[p*lda+r];
      return;
   }
   for(int c=p+1; c<unroll*((p+1-1)/unroll+1); c++) {
      SimdVec<T> ldvec( -ld[c] ); // NB minus so we can use fma below
      for(int r=vlen*(c/vlen); r<BLOCK_SIZE; r+=vlen) {
         SimdVec<T> lvec = SimdVec<T>::load_aligned(&a[p*lda+r]);
         SimdVec<T> avec = SimdVec<T>::load_aligned(&a[c*lda+r]);
         avec = fmadd(avec, lvec, ldvec);
         avec.store_aligned(&a[c*lda+r]);
      }
   }
   for(int c=unroll*((p+1-1)/unroll+1); c<BLOCK_SIZE; c+=unroll) {
      // NB we use minus ld[c] below to allow fma afterwards
      SimdVec<T> ldvec0( -ld[c] ); // NB minus so we can use fma below
      SimdVec<T> ldvec1( -ld[c+1] ); // NB minus so we can use fma below
      SimdVec<T> ldvec2( -ld[c+2] ); // NB minus so we can use fma below
      SimdVec<T> ldvec3( -ld[c+3] ); // NB minus so we can use fma below
      for(int r=vlen*(c/vlen); r<BLOCK_SIZE; r+=vlen) {
         SimdVec<T> lvec = SimdVec<T>::load_aligned(&a[p*lda+r]);
         SimdVec<T> avec0 = SimdVec<T>::load_aligned(&a[(c+0)*lda+r]);
         SimdVec<T> avec1 = SimdVec<T>::load_aligned(&a[(c+1)*lda+r]);
         SimdVec<T> avec2 = SimdVec<T>::load_aligned(&a[(c+2)*lda+r]);
         SimdVec<T> avec3 = SimdVec<T>::load_aligned(&a[(c+3)*lda+r]);
         avec0 = fmadd(avec0, lvec, ldvec0);
         avec1 = fmadd(avec1, lvec, ldvec1);
         avec2 = fmadd(avec2, lvec, ldvec2);
         avec3 = fmadd(avec3, lvec, ldvec3);
         avec0.store_aligned(&a[(c+0)*lda+r]);
         avec1.store_aligned(&a[(c+1)*lda+r]);
         avec2.store_aligned(&a[(c+2)*lda+r]);
         avec3.store_aligned(&a[(c+3)*lda+r]);
      }
   }
#endif
}

} // namespace block_ldlt_internal

/** Factorize a square block without restricting pivots
 *  Expects to be given a square block of size BLOCK_SIZE with numbers of
 *  interest in bottom right part. */
template<typename T, int BLOCK_SIZE>
void block_ldlt(int from, int *perm, T *a, int lda, T *d, T *ldwork,
      bool action, const T u, const T small, int *lperm=nullptr) {
   using namespace block_ldlt_internal;

   /* Main loop */
   for(int p=from; p<BLOCK_SIZE; ) {
      // Find largest uneliminated entry
      T bestv; // Value of maximum entry
      int t, m; // row and col location of maximum entry
      find_maxloc<T,BLOCK_SIZE>(p, a, lda, bestv, t, m);

      // Handle case where everything remaining is small
      // NB: There might be delayed columns!
      if(fabs(bestv) < small) {
         if(!action) throw SingularError(p);
         // Loop over remaining columns
         for(; p<BLOCK_SIZE; ) {
            // Zero out col
            d[2*p] = 0.0; d[2*p+1] = 0.0;
            for(int r=p; r<BLOCK_SIZE; r++)
               a[p*lda+r] = 0.0;
            for(int r=p; r<BLOCK_SIZE; r++)
               ldwork[p*BLOCK_SIZE+r] = 0.0;
            // NB: lperm remains unchanged
            p++;
         }
         break; // All done
      }

      // Figure out pivot size
      int pivsiz = 0;
      int m2=m, t2=t; // FIXME: debug remove
      T a11, a21, a22, detscale, detpiv;
      if(t==m) {
         a11 = a[t*lda+t];
         pivsiz = 1;
      } else {
         a11 = a[m*lda+m];
         a22 = a[t*lda+t];
         a21 = a[m*lda+t];
         if( test_2x2(a11, a21, a22, detpiv, detscale) ) {
            pivsiz = 2;
         } else {
            if(fabs(a11) > fabs(a22)) {
               // Go for a11 as 1x1 pivot
               pivsiz = 1;
               t = m;
               if(fabs(a11 / a21) < u) pivsiz = 0; // Fail pivot
            } else {
               // Go for a22 as 1x1 pivot
               pivsiz = 1;
               a11 = a22;
               m = t;
               if(fabs(a22 / a21) < u) pivsiz = 0; // Fail pivot
            }
         }
      }

      // Apply pivot, swapping columns as required
      if(pivsiz == 0) {
         // FIXME: debug remove
         printf("broken!\n");
         printf("t = %d m = %d\n", t2, m2);
         a11 = a[m2*lda+m2];
         printf("[%d] = %e\n", m2*BLOCK_SIZE+m2, a[m2*lda+m2]);
         a22 = a[t2*lda+t2];
         a21 = a[m2*lda+t2];
         printf("a11 = %e a21 = %e a22 = %e\n", a11, a21, a22);
         exit(1);
      }
      if(pivsiz == 1) {
         /* 1x1 pivot */
         T d11 = 1.0/a11;
         swap_cols<T, BLOCK_SIZE>
            (p, t, BLOCK_SIZE, a, lda, ldwork, perm);
         if(lperm) { int temp=lperm[p]; lperm[p]=lperm[t]; lperm[t]=temp; }
         /* Divide through, preserving a copy */
         T *work = &ldwork[p*BLOCK_SIZE];
         for(int r=p+1; r<BLOCK_SIZE; r++) {
            work[r] = a[p*lda+r];
            a[p*lda+r] *= d11;
         }
         /* Perform update */
         update_1x1<T, BLOCK_SIZE>(p, a, lda, work);
         /* Store d */
         d[2*p] = d11;
         d[2*p+1] = 0.0;
         /* Set diagonal to I */
         a[p*lda+p] = 1.0;
      } else {
         /* 2x2 pivot */
         /* NB t > m by construction. Hence m>=p, t>=p+1 and swaps are safe */
         swap_cols<T, BLOCK_SIZE>
            (p,   m, BLOCK_SIZE, a, lda, ldwork, perm);
         if(lperm) { int temp=lperm[p]; lperm[p]=lperm[m]; lperm[m]=temp; }
         swap_cols<T, BLOCK_SIZE>
            (p+1, t, BLOCK_SIZE, a, lda, ldwork, perm);
         if(lperm) { int temp=lperm[p+1]; lperm[p+1]=lperm[t]; lperm[t]=temp; }
         /* Calculate 2x2 inverse */
         T d11 = (a22*detscale)/detpiv;
         T d22 = (a11*detscale)/detpiv;
         T d21 = (-a21*detscale)/detpiv;
         /* Divide through, preserving a copy */
         T *work = &ldwork[p*BLOCK_SIZE];
         for(int r=p+2; r<BLOCK_SIZE; r++) {
            work[r]   = a[p*lda+r];
            work[BLOCK_SIZE+r] = a[(p+1)*lda+r];
            a[p*lda+r]     = d11*work[r] + d21*work[BLOCK_SIZE+r];
            a[(p+1)*lda+r] = d21*work[r] + d22*work[BLOCK_SIZE+r];
         }
         /* Perform update */
         update_2x2<T,BLOCK_SIZE>(p, a, lda, work);
         /* Store d */
         d[2*p  ] = d11;
         d[2*p+1] = d21;
         d[2*p+2] = std::numeric_limits<T>::infinity();
         d[2*p+3] = d22;
         /* Set diagonal to I */
         a[p*(lda+1)] = 1.0;
         a[p*(lda+1)+1] = 0.0;
         a[(p+1)*(lda+1)] = 1.0;
      }
      p += pivsiz;
   }
}
}}} /* namespaces spral::ssids::cpu */
