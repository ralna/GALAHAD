/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-21 AT 11:00 GMT
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

#include "ssids_rip.hxx"
#include "ssids_profile.hxx"
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

#ifdef REAL_32
#define cholesky_factor cholesky_factor_sgl
#define ldlt_app_factor ldlt_app_factor_sgl
#define ldlt_tpp_factor ldlt_tpp_factor_sgl
#elif REAL_128
#define cholesky_factor cholesky_factor_qul
#define ldlt_app_factor ldlt_app_factor_qul
#define ldlt_tpp_factor ldlt_tpp_factor_qul
#else
#define cholesky_factor cholesky_factor_dbl
#define ldlt_app_factor ldlt_app_factor_dbl
#define ldlt_tpp_factor ldlt_tpp_factor_dbl
#endif

#ifdef INTEGER_64
#define host_gemm host_gemm_64
#endif

namespace spral { namespace ssids { namespace cpu {

/* Factorize a node (indef) */
template <typename T, typename PoolAlloc>
void factor_node_indef(
      ipc_ ni, // FIXME: remove post debug
      SymbolicNode const& snode,
      NumericNode<T, PoolAlloc> &node,
      struct cpu_factor_options const& options,
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
   //Verify<T> verifier(m, n, perm, lcol, ldl);
   if(options.pivot_method != PivotMethod::tpp) {
      // Use an APP based pivot method
      T zero_val = 0.0;
      node.nelim = ldlt_app_factor(
            m, n, perm, lcol, ldl, d, zero_val, contrib, m-n, options, work,
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
   //verifier.verify(node.nelim, perm, lcol, ldl, d);

   /* Finish factorization worth simplistic code */
   if(node.nelim < n) {
      ipc_ nelim = node.nelim;
      if(options.pivot_method!=PivotMethod::tpp)
         stats.not_first_pass += n-nelim;
      // Only use TPP to finish off if we're a root node, it's not finishing
      // off but actually doing it, or failed_pivot_method says to do so
      if(m==n || options.pivot_method==PivotMethod::tpp ||
            options.failed_pivot_method==FailedPivotMethod::tpp) {
#ifdef PROFILE
         Profile::Task task_tpp("TA_LDLT_TPP");
#endif
         T *ld = work[omp_get_thread_num()].get_ptr<T>(2*(m-nelim));
         node.nelim += ldlt_tpp_factor(
               m-nelim, n-nelim, &perm[nelim], &lcol[nelim*(ldl+1)], ldl,
               &d[2*nelim], ld, m-nelim, options.action, options.u,
               options.small, nelim, &lcol[nelim], ldl
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
         if(options.pivot_method==PivotMethod::tpp) {
            stats.not_first_pass += n - node.nelim;
         } else {
            stats.not_second_pass += n - node.nelim;
         }
#ifdef profile
         task_tpp.done();
#endif
      }
   }

#ifdef PROFILE
      Profile::setState("TA_MISC1");
#endif
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
      struct cpu_factor_options const& options,
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
         m, n, lcol, ldl, beta, contrib, m-n, options.cpu_block_size, &flag
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
      ipc_ ni,
      SymbolicNode const& snode,
      NumericNode<T, PoolAlloc> &node,
      struct cpu_factor_options const& options,
      ThreadStats& stats,
      std::vector<Workspace>& work,
      PoolAlloc& pool_alloc
      ) {
   T zero_val = 0.0;
   if(posdef) factor_node_posdef(zero_val, snode, node, options, stats);
   else       factor_node_indef(ni, snode, node, options, stats, work, pool_alloc);
}

}}} /* end of namespace spral::ssids::cpu */
