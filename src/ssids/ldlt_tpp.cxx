/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-26 AT 11:40 GMT
 */

#include "ssids_cpu_kernels_ldlt_tpp.hxx"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "ssids_cpu_ThreadStats.hxx"
#include "ssids_cpu_kernels_wrappers.hxx"

#ifdef INTEGER_64
#define host_gemm host_gemm_64
#define host_trsv host_trsv_64
#define host_trsm host_trsm_64
#define gemv gemv_64
#endif

namespace spral { namespace ssids { namespace cpu {

namespace {

/** overload fabs for floats and doubles */
rpc_ fabs_(rpc_ x) {
#ifdef REAL_32
     double fabsd = fabs(double(x));
     float fabss;
     fabss = fabsd;
     return fabss;
#elif REAL_128
     double fabsd = fabs(double(x));
     __float128 fabsq;
     fabsq = fabsd;
     return fabsq;
#else
     return fabs(x);
#endif
}

/** Returns true if all entries in col are less than small in abs value */
bool check_col_small(ipc_ idx, ipc_ from, ipc_ to, rpc_ const* a,
                     ipc_ lda, rpc_ small) {
   bool check = true;
   for(ipc_ c=from; c<idx; ++c)
      check = check && (fabs(a[c*lda+idx]) < small);
   for(ipc_ r=idx; r<to; ++r)
      check = check && (fabs(a[idx*lda+r]) < small);
   return check;
}

/** Returns col index of largest entry in row starting at a */
ipc_ find_row_abs_max(ipc_ from, ipc_ to, rpc_ const* a, ipc_ lda) {
   if(from>=to) return -1;
   ipc_ best_idx=from; rpc_ best_val=fabs(a[from*lda]);
   for(ipc_ idx=from+1; idx<to; ++idx)
      if(fabs(a[idx*lda]) > best_val) {
         best_idx = idx;
         best_val = fabs(a[idx*lda]);
      }
   return best_idx;
}

/** Performs symmetric swap of col1 and col2 in lower triangle */
// FIXME: remove n only here for debug
void swap_cols(ipc_ col1, ipc_ col2, ipc_ m, ipc_ n, ipc_* perm, rpc_* a,
               ipc_ lda, ipc_ nleft, rpc_* aleft, ipc_ ldleft) {
   if(col1 == col2) return; // No-op

   // Ensure col1 < col2
   if(col2<col1)
      std::swap(col1, col2);

   // Swap perm entries
   std::swap( perm[col1], perm[col2] );

   // Swap aleft(col1, :) and aleft(col2, :)
   for(ipc_ c=0; c<nleft; ++c)
      std::swap( aleft[c*ldleft+col1], aleft[c*ldleft+col2] );

   // Swap a(col1, 0:col1-1) and a(col2, 0:col1-1)
   for(ipc_ c=0; c<col1; ++c)
      std::swap( a[c*lda+col1], a[c*lda+col2] );

   // Swap a(col1+1:col2-1, col1) and a(col2, col1+1:col2-1)
   for(ipc_ i=col1+1; i<col2; ++i)
      std::swap( a[col1*lda+i], a[i*lda+col2] );

   // Swap a(col2+1:m, col1) and a(col2+1:m, col2)
   for(ipc_ r=col2+1; r<m; ++r)
      std::swap( a[col1*lda+r], a[col2*lda+r] );

   // Swap a(col1, col1) and a(col2, col2)
   std::swap( a[col1*lda+col1], a[col2*lda+col2] );
}

/** Returns abs value of largest unelim entry in row/col not in posn exclude or on diagonal */
rpc_ find_rc_abs_max_exclude(ipc_ col, ipc_ nelim, ipc_ m,
                                   rpc_ const* a, ipc_ lda, ipc_ exclude) {
   rpc_ best = 0.0;
   for(ipc_ c=nelim; c<col; ++c) {
      if(c==exclude) continue;
      best = std::max(best, fabs_(a[c*lda+col]));
   }
   for(ipc_ r=col+1; r<m; ++r) {
      if(r==exclude) continue;
      best = std::max(best, fabs_(a[col*lda+r]));
   }
   return best;
}

/** Return true if (t,p) is a good 2x2 pivot, false otherwise */
bool test_2x2(ipc_ t, ipc_ p, rpc_ maxt, rpc_ maxp, rpc_ const* a, ipc_ lda, rpc_ u, rpc_ small, rpc_* d) {
   // NB: We know t < p

   // Check there is a non-zero in the pivot block
   rpc_ a11 = a[t*lda+t];
   rpc_ a21 = a[t*lda+p];
   rpc_ a22 = a[p*lda+p];
   //pripc_f("Testing 2x2 pivot (%d, %d) %e %e %e vs %e %e\n", t, p, a11, a21, a22, maxt, maxp);
   rpc_ maxpiv = std::max(fabs(a11), std::max(fabs(a21), fabs(a22)));
   if(maxpiv < small) return false;

   // Ensure non-singular and not afflicted by cancellation
   rpc_ detscale = 1/maxpiv;
   rpc_ detpiv0 = (a11*detscale)*a22;
   rpc_ detpiv1 = (a21*detscale)*a21;
   rpc_ detpiv = detpiv0 - detpiv1;
   //printf("t1 %e < %e %e %e?\n", fabs_(detpiv), small, fabs_(detpiv0/2), fabs_(detpiv1/2));
   if(fabs_(detpiv) < std::max(small, std::max(fabs_(detpiv0/2),
                                               fabs_(detpiv1/2)))) return false;

   // Finally apply threshold pivot check
   d[0] = (a22*detscale)/detpiv;
   d[1] = (-a21*detscale)/detpiv;
#ifdef REAL_128
   d[2] = 1.0/0.0;
#else
   d[2] = std::numeric_limits<rpc_>::infinity();
#endif
   d[3] = (a11*detscale)/detpiv;
   //printf("t2 %e < %e?\n", std::max(maxp, maxt), small);
   if(std::max(maxp, maxt) < small) return true; // Rest of col small
   rpc_ x1 = fabs(d[0])*maxt + fabs(d[1])*maxp;
   rpc_ x2 = fabs(d[1])*maxt + fabs(d[3])*maxp;
   //printf("t3 %e < %e?\n", std::max(x1, x2), 1.0/u);
   return ( u*std::max(x1, x2) < 1.0 );
}

/** Applies the 2x2 pivot to rest of block column */
void apply_2x2(ipc_ nelim, ipc_ m, rpc_* a, ipc_ lda, rpc_* ld,
               ipc_ ldld, rpc_* d) {
   /* Set diagonal block to identity */
   rpc_* a1 = &a[nelim*lda];
   rpc_* a2 = &a[(nelim+1)*lda];
   a1[nelim] = 1.0;
   a1[nelim+1] = 0.0;
   a2[nelim+1] = 1.0;
   /* Extract D^-1 values */
   rpc_ d11 = d[2*nelim];
   rpc_ d21 = d[2*nelim+1];
   rpc_ d22 = d[2*nelim+3];
   /* Divide through, preserving copy in ld */
   for(ipc_ r=nelim+2; r<m; ++r) {
      ld[r] = a1[r]; ld[ldld+r] = a2[r];
      a1[r] = d11*ld[r] + d21*ld[ldld+r];
      a2[r] = d21*ld[r] + d22*ld[ldld+r];
   }
}

/** Applies the 1x1 pivot to rest of block column */
void apply_1x1(ipc_ nelim, ipc_ m, rpc_* a, ipc_ lda, rpc_* ld,
               ipc_ ldld, rpc_* d) {
   /* Set diagonal block to identity */
   rpc_* a1 = &a[nelim*lda];
   a1[nelim] = 1.0;
   /* Extract D^-1 values */
   rpc_ d11 = d[2*nelim];
   /* Divide through, preserving copy in ld */
   for(ipc_ r=nelim+1; r<m; ++r) {
      ld[r] = a1[r];
      a1[r] *= d11;
   }
}

/** Sets column to zero */
void zero_col(ipc_ col, ipc_ m, rpc_* a, ipc_ lda) {
   for(ipc_ r=col; r<m; ++r) {
      a[col*lda+r] = 0.0;
   }
}

} /* anon namespace */

/** Simple LDL^T with threshold partial pivoting.
 * Intended for finishing off small matrices, not for performance */
ipc_ ldlt_tpp_factor(ipc_ m, ipc_ n, ipc_* perm, rpc_* a, ipc_ lda,
                    rpc_* d,  rpc_* ld, ipc_ ldld, bool action,
                    rpc_ u, rpc_ small, ipc_ nleft,
      rpc_* aleft, ipc_ ldleft) {
   //printf("=== ENTRY %d %d ===\n", m, n);
   ipc_ nelim = 0; // Number of eliminated variables
   rpc_ one_val = 1.0;
   rpc_ minus_one_val = - 1.0;
   while(nelim<n) {
      /*printf("nelim = %d\n", nelim);
      for(ipc_ r=0; r<m; ++r) {
         printf("%d: ", perm[r]);
         for(ipc_ c=0; c<=std::min(r,n-1); ++c) printf(" %e", a[c*lda+r]);
         printf("\n");
      }*/
      // Need to check if col nelim is zero now or it gets missed
      if(check_col_small(nelim, nelim, m, a, lda, small)) {
         // Record zero pivot
         //printf("Zero pivot %d\n", nelim);
         if(!action) throw SingularError(nelim);
         swap_cols(nelim, nelim, m, n, perm, a, lda, nleft, aleft, ldleft);
         zero_col(nelim, m, a, lda);
         d[2*nelim] = 0.0;
         d[2*nelim+1] = 0.0;
         nelim++;
         continue;
      }
      ipc_ p; // Index of current candidate pivot [starts at col 2]
      for(p=nelim+1; p<n; ++p) {
         //printf("Consider p=%d\n", p);
         // Check if column p is effectively zero
         if(check_col_small(p, nelim, m, a, lda, small)) {
            // Record zero pivot
            //printf("Zero pivot\n");
            if(!action) throw SingularError(nelim);
            swap_cols(p, nelim, m, n, perm, a, lda, nleft, aleft, ldleft);
            zero_col(nelim, m, a, lda);
            d[2*nelim] = 0.0;
            d[2*nelim+1] = 0.0;
            nelim++;
            break;
         }

         // Find column index of largest entry in |a(p, nelim+1:p-1)|
         ipc_ t = find_row_abs_max(nelim, p, &a[p], lda);

         // Try (t,p) as 2x2 pivot
         rpc_ maxt = find_rc_abs_max_exclude(t, nelim, m, a, lda, p);
         rpc_ maxp = find_rc_abs_max_exclude(p, nelim, m, a, lda, t);
         if( test_2x2(t, p, maxt, maxp, a, lda, u, small, &d[2*nelim]) ) {
            //printf("2x2 pivot\n");
            swap_cols(t, nelim, m, n, perm, a, lda, nleft, aleft, ldleft);
            swap_cols(p, nelim+1, m, n, perm, a, lda, nleft, aleft, ldleft);
            apply_2x2(nelim, m, a, lda, ld, ldld, d);
            host_gemm(OP_N, OP_T, m-nelim-2, n-nelim-2, 2, minus_one_val,
                  &a[nelim*lda+nelim+2], lda, &ld[nelim+2], ldld, one_val,
                  &a[(nelim+2)*lda+nelim+2], lda); // update trailing mat
            nelim += 2;
            break;
         }

         // Try p as 1x1 pivot
         maxp = std::max(maxp, fabs_(a[t*lda+p]));
         if( fabs_(a[p*lda+p]) >= u*maxp ) {
            //printf("1x1 pivot\n");
            swap_cols(p, nelim, m, n, perm, a, lda, nleft, aleft, ldleft);
            d[2*nelim] = 1 / a[nelim*lda+nelim];
            d[2*nelim+1] = 0.0;
            apply_1x1(nelim, m, a, lda, ld, ldld, d);
            host_gemm(OP_N, OP_T, m-nelim-1, n-nelim-1, 1, minus_one_val,
                  &a[nelim*lda+nelim+1], lda, &ld[nelim+1], ldld, one_val,
                  &a[(nelim+1)*lda+nelim+1], lda); // update trailing mat
            nelim += 1;
            break;
         }
      }
      if(p>=n) {
         // Pivot search failed

         // Try 1x1 pivot on p=nelim as last resort (we started at p=nelim+1)
         p = nelim;
         rpc_ maxp = find_rc_abs_max_exclude(p, nelim, m, a, lda, -1);
         if( fabs_(a[p*lda+p]) >= u*maxp ) {
            //printf("1x1 pivot %d\n", p);
            swap_cols(p, nelim, m, n, perm, a, lda, nleft, aleft, ldleft);
            d[2*nelim] = 1 / a[nelim*lda+nelim];
            d[2*nelim+1] = 0.0;
            apply_1x1(nelim, m, a, lda, ld, ldld, d);
            host_gemm(OP_N, OP_T, m-nelim-1, n-nelim-1, 1, minus_one_val,
                  &a[nelim*lda+nelim+1], lda, &ld[nelim+1], ldld, one_val,
                  &a[(nelim+1)*lda+nelim+1], lda); // update trailing mat
            nelim += 1;
         } else {
            // That didn't work either. No more pivots to be found
            //printf("Out of pivots\n");
            break;
         }
      }
   }
   /*printf("==== EXIT ====\n");
   for(int r=0; r<m; ++r) {
      printf("%d: ", perm[r]);
      for(int c=0; c<=std::min(r,n-1); ++c) printf(" %e", a[c*lda+r]);
      printf("\n");
   }
   printf("==== EXIT ====\n");*/
   return nelim;
}

void ldlt_tpp_solve_fwd(ipc_ m, ipc_ n, rpc_ const* l, ipc_ ldl, ipc_ nrhs,
                        rpc_* x, ipc_ ldx) {
   rpc_ one_val = 1.0;
   rpc_ minus_one_val = - 1.0;
   if(nrhs==1) {
      host_trsv(FILL_MODE_LWR, OP_N, DIAG_UNIT, n, l, ldl, x, 1);
      if(m > n)
         gemv(OP_N, m-n, n, minus_one_val, &l[n], ldl, x, 1, one_val, &x[n], 1);
   } else {
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_N, DIAG_UNIT, n, nrhs,
                one_val, l, ldl, x, ldx);
      if(m > n)
         host_gemm(OP_N, OP_N, m-n, nrhs, n, minus_one_val, &l[n],
                   ldl, x, ldx, one_val, &x[n], ldx);
   }
}

void ldlt_tpp_solve_diag(ipc_ n, rpc_ const* d, rpc_* x) {
   for(ipc_ i=0; i<n; ) {
#ifdef REAL_128
      if(i+1<n && std::isinf(static_cast<double>(d[2*i+2]))) {
#else
      if(i+1<n && std::isinf(d[2*i+2])) {
#endif
         // 2x2 pivot
         rpc_ d11 = d[2*i];
         rpc_ d21 = d[2*i+1];
         rpc_ d22 = d[2*i+3];
         rpc_ x1 = x[i];
         rpc_ x2 = x[i+1];
         x[i]   = d11*x1 + d21*x2;
         x[i+1] = d21*x1 + d22*x2;
         i += 2;
      } else {
         // 1x1 pivot
         rpc_ d11 = d[2*i];
         x[i] *= d11;
         i++;
      }
   }
}

void ldlt_tpp_solve_bwd(ipc_ m, ipc_ n, rpc_ const* l, ipc_ ldl, ipc_ nrhs,
                        rpc_* x, ipc_ ldx) {
   rpc_ one_val = 1.0;
   rpc_ minus_one_val = - 1.0;
   if(nrhs==1) {
      if(m > n)
         gemv(OP_T, m-n, n, minus_one_val, &l[n], ldl, &x[n], 1, one_val, x, 1);
      host_trsv(FILL_MODE_LWR, OP_T, DIAG_UNIT, n, l, ldl, x, 1);
   } else {
      if(m > n)
         host_gemm(OP_T, OP_N, n, nrhs, m-n, minus_one_val, &l[n], ldl,
                   &x[n], ldx, one_val, x, ldx);
      host_trsm(SIDE_LEFT, FILL_MODE_LWR, OP_T, DIAG_UNIT, n, nrhs,
                one_val, l, ldl, x, ldx);
   }
}

}}} /* end of namespace spral::ssids::cpu */
