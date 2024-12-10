/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-04 AT 08:30 GMT
 */

#pragma once

#include <cmath>
#include <limits>

#include "ssids_rip.hxx"
#include "ssids_cpu_kernels_common.hxx"
#include "ssids_cpu_kernels_SimdVec.hxx"

namespace spral { namespace ssids { namespace cpu {

/** Return number of elements to skip at beginning to get an aligned element,
 *  or max int if alignment is not (trivally) possible.
 *
 *  Note this will mostly just fail if sizeof(T) doesn't divide into alignment.
 */
template <typename T>
ipc_ offset_to_align(T* ptr) {
#if defined(__AVX512F__)
  ipc_ const align = 64;
#elif defined(__AVX__)
  ipc_ const align = 32;
#else
  ipc_ const align = 16;
#endif
   uintptr_t offset = align - (reinterpret_cast<uintptr_t>(ptr) % align);
   offset /= sizeof(T);
   if((reinterpret_cast<uintptr_t>(ptr+offset) % align) == 0) return offset;
   else return std::numeric_limits<ipc_>::max();
}

/** Calculates LD from L and D.
 *
 * We assume that both l and ld are 32-bytes aligned, and ldl and ldld are
 * multiples of 32 bytes, so we can use AVX.
 */
template <enum operation op, typename T>
void calcLD(ipc_ m, ipc_ n, T const* l, ipc_ ldl, T const* d, T* ld, 
            ipc_ ldld) {
   typedef SimdVec<T> SimdVecT;

   for(ipc_ col=0; col<n; ) {


#ifdef REAL_128
      if(col+1==n || std::isfinite(static_cast<double>(d[2*col+2]))) {
#else
      if(col+1==n || std::isfinite(d[2*col+2])) {
#endif
         // 1x1 pivot
         T d11 = d[2*col];
         if(d11 != 0.0) d11 = 1/d11; // Zero pivots just cause zeroes
         if(op==OP_N) {
            ipc_ const vlen = SimdVecT::vector_length;
            ipc_ const unroll = 4;
            ipc_ offset = offset_to_align(l);
            if(offset_to_align(ld) != offset) offset = m; // give up on vectors
            ipc_ i0_ = 0;
            ipc_ nvec = std::max(i0_, (m-offset) / vlen);
            for(ipc_ row=0; row<std::min(offset,m); ++row)
               ld[col*ldld+row] = d11 * l[col*ldl+row];
            SimdVecT d11v(d11);
            if(nvec < unroll) {
               for(ipc_ row=offset; row<offset+nvec*vlen; row+=vlen) {
                  SimdVecT lv = SimdVecT::load_aligned(&l[col*ldl+row]);
                  lv = lv * d11;
                  lv.store_aligned(&ld[col*ldld+row]);
               }
            } else {
               ipc_ nunroll = nvec / unroll;
               for(ipc_ row=offset; row<offset+nunroll*unroll*vlen; 
                        row+=unroll*vlen) {
                  SimdVecT lv0 = SimdVecT::load_aligned(&l[col*ldl+row+0*vlen]);
                  SimdVecT lv1 = SimdVecT::load_aligned(&l[col*ldl+row+1*vlen]);
                  SimdVecT lv2 = SimdVecT::load_aligned(&l[col*ldl+row+2*vlen]);
                  SimdVecT lv3 = SimdVecT::load_aligned(&l[col*ldl+row+3*vlen]);
                  lv0 *= d11;
                  lv1 *= d11;
                  lv2 *= d11;
                  lv3 *= d11;
                  lv0.store_aligned(&ld[col*ldld+row+0*vlen]);
                  lv1.store_aligned(&ld[col*ldld+row+1*vlen]);
                  lv2.store_aligned(&ld[col*ldld+row+2*vlen]);
                  lv3.store_aligned(&ld[col*ldld+row+3*vlen]);
               }
               for(ipc_ row=offset+nunroll*unroll*vlen; 
                        row<offset+nvec*vlen; row+=vlen) {
                  SimdVecT lv = SimdVecT::load_aligned(&l[col*ldl+row]);
                  lv = lv * d11;
                  lv.store_aligned(&ld[col*ldld+row]);
               }
            }
            for(ipc_ row=offset+nvec*vlen; row<m; row++)
               ld[col*ldld+row] = d11 * l[col*ldl+row];
         } else { /* op==OP_T */
            for(ipc_ row=0; row<m; row++)
               ld[col*ldld+row] = d11 * l[row*ldl+col];
         }
         col++;
      } else {
         // 2x2 pivot
         T d11 = d[2*col];
         T d21 = d[2*col+1];
         T d22 = d[2*col+3];
         T det = d11*d22 - d21*d21;
         d11 = d11/det;
         d21 = d21/det;
         d22 = d22/det;
         for(ipc_ row=0; row<m; row++) {
            T a1, a2;
            if(op==OP_N) {
               a1 = l[col*ldl+row];
               a2 = l[(col+1)*ldl+row];
            } else { /* op==OP_T */
               a1 = l[row*ldl+col];
               a2 = l[row*ldl+(col+1)];
            }
            ld[col*ldld+row]     =  d22*a1 - d21*a2;
            ld[(col+1)*ldld+row] = -d21*a1 + d11*a2;
         }
         col += 2;
      }
   }
}

}}} /* namespaces spral::ssids::cpu */
