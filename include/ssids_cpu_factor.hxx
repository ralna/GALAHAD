/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 09:00 GMT
 */
#pragma once

/* Standard headers */
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

/* SPRAL headers */

#include "ssids_routines.h"
#include "galahad_precision.h"
#include "ssids_cpu_cpu_iface.hxx"
#include "ssids_cpu_SymbolicNode.hxx"
#include "ssids_cpu_ThreadStats.hxx"
#include "ssids_cpu_Workspace.hxx"
#include "ssids_cpu_kernels_assemble.hxx"
#include "ssids_cpu_kernels_calc_ld.hxx"
#include "ssids_cpu_kernels_cholesky.hxx"
#include "ssids_cpu_kernels_ldlt_app.hxx"
#include "ssids_cpu_kernels_ldlt_tpp.hxx"
#include "ssids_cpu_kernels_wrappers.hxx"

//#include "ssids_cpu_kernels_verify.hxx" // FIXME: remove debug

namespace galahad { namespace ssids { namespace cpu {

/* Factorize a node (indef) */
template <typename T, typename PoolAlloc>
void factor_node_indef(
      SymbolicNode const& snode,
      NumericNode<T, PoolAlloc> &node,
      struct cpu_factor_control const& control,
      ThreadStats& stats,
      std::vector<Workspace>& work,
      PoolAlloc& pool_alloc
      ) {
   /* Extract useful information about node */
   ipc_ m = snode.nrow + node.ndelay_in;
   ipc_ n = snode.ncol + node.ndelay_in;
   size_t ldl = align_lda<T>(m);
   T *lcol = node.lcol;
   T *d = &node.lcol[ n*ldl ];
   ipc_ *perm = node.perm;
   T *contrib = node.contrib;

   /* Perform factorization */
   if(control.pivot_method != PivotMethod::tpp) {
      // Use an APP based pivot method
      T zero_val = 0.0;
      node.nelim = ldlt_app_factor(
            m, n, perm, lcol, ldl, d, zero_val, contrib, m-n, control, work,
            pool_alloc
            );
      if(node.nelim < 0) {
         stats.flag = static_cast<Flag>(node.nelim);
         return;
      }
   } else {
      // Otherwise, force use of TPP
      node.nelim = 0;
   }
//printf("past\n");
   /* Finish factorization worth simplistic code */
   if(node.nelim < n) {
      ipc_ nelim = node.nelim;
      if(control.pivot_method!=PivotMethod::tpp)
         stats.not_first_pass += n-nelim;
      // Only use TPP to finish off if we're a root node, it's not finishing
      // off but actually doing it, or failed_pivot_method says to do so
      if(m==n || control.pivot_method==PivotMethod::tpp ||
            control.failed_pivot_method==FailedPivotMethod::tpp) {
         T *ld = work[omp_get_thread_num()].get_ptr<T>(2*(m-nelim));
         node.nelim += ldlt_tpp_factor(
               m-nelim, n-nelim, &perm[nelim], &lcol[nelim*(ldl+1)], ldl,
               &d[2*nelim], ld, m-nelim, control.action, control.u,
               control.small, nelim, &lcol[nelim], ldl
               );
         if(m-n>0 && node.nelim>nelim) {
            ipc_ nelim2 = node.nelim - nelim;
            ipc_ ldld = align_lda<T>(m-n);
            T *ld = work[omp_get_thread_num()].get_ptr<T>(nelim2*ldld);
            calcLD<OP_N>(
                  m-n, nelim2, &lcol[nelim*ldl+n], ldl, &d[2*nelim], ld, ldld
                  );
            T rbeta = (nelim==0) ? 0.0 : 1.0;
            host_gemm<T>(OP_N, OP_T, m-n, m-n, nelim2,
                  -1.0, &lcol[nelim*ldl+n], ldl, ld, ldld,
                  rbeta, node.contrib, m-n);
         }
         if(control.pivot_method==PivotMethod::tpp) {
            stats.not_first_pass += n - node.nelim;
         } else {
            stats.not_second_pass += n - node.nelim;
         }
      }
   }

   /* Record information */
   node.ndelay_out = n - node.nelim;
   stats.num_delay += node.ndelay_out;
   for (longc_ j = m; j >= m-(node.nelim)+1; --j) {
       stats.num_factor += j;
       stats.num_flops += j*j;
   }

   /* Mark as no contribution if we make no contribution */
   if(node.nelim==0 && !node.first_child && snode.contrib.size()==0) {
      // FIXME: Actually loop over children and check one exists with contrib
      //        rather than current approach of just looking for children.
      node.free_contrib();
   } else if(node.nelim==0) {
      // FIXME: If we fix the above, we don't need this explict zeroing
      longc_ contrib_size = m-n;
      memset(node.contrib, 0, contrib_size*contrib_size*sizeof(T));
   }
}
/* Factorize a node (posdef) */
template <typename T, typename PoolAlloc>
void factor_node_posdef(
      T beta,
      SymbolicNode const& snode,
      NumericNode<T, PoolAlloc> &node,
      struct cpu_factor_control const& control,
      ThreadStats& stats
      ) {
   /* Extract useful information about node */
   ipc_ m = snode.nrow;
   ipc_ n = snode.ncol;
   ipc_ ldl = align_lda<T>(m);
   T *lcol = node.lcol;
   T *contrib = node.contrib;

   /* Perform factorization */
   ipc_ flag;
   cholesky_factor(
         m, n, lcol, ldl, beta, contrib, m-n, control.cpu_block_size, &flag
         );
   if(flag!=-1) {
      node.nelim = flag+1;
      stats.flag = Flag::ERROR_NOT_POS_DEF;
      return;
   }
   node.nelim = n;

   /* Record information */
   node.ndelay_out = 0;
   for (longc_ j = m; j >= m-(node.nelim)+1; --j) {
       stats.num_factor += j;
       stats.num_flops += j*j;
   }
}
/* Factorize a node (wrapper) */
template <bool posdef, typename T, typename PoolAlloc>
void factor_node(
      SymbolicNode const& snode,
      NumericNode<T, PoolAlloc> &node,
      struct cpu_factor_control const& control,
      ThreadStats& stats,
      std::vector<Workspace>& work,
      PoolAlloc& pool_alloc
      ) {
   T zero_val = 0.0;
   if(posdef) factor_node_posdef(zero_val, snode, node, control, stats);
   else       factor_node_indef(snode, node, control, stats, work, pool_alloc);
}

}}} /* end of namespace galahad::ssids::cpu */
