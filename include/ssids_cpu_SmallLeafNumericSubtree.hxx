/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-06-11 AT 10:30 GMT
 */

#pragma once

#include <memory>

#include "ssids_rip.hxx"
#include "ssids_cpu_cpu_iface.hxx"
#include "ssids_cpu_factor.hxx"
#include "ssids_cpu_NumericNode.hxx"
#include "ssids_cpu_SmallLeafSymbolicSubtree.hxx"
#include "ssids_cpu_ThreadStats.hxx"

#ifdef REAL_32
#define FAPrecisionTraits FASingleTraits
#define factor_alloc_precision factor_alloc_single
#define ldlt_tpp_factor ldlt_tpp_factor_sgl
#elif REAL_128
#define FAPrecisionTraits FAQuadrupleTraits
#define factor_alloc_precision factor_alloc_quadruple
#define ldlt_tpp_factor ldlt_tpp_factor_qul
#else
#define FAPrecisionTraits FADoubleTraits
#define factor_alloc_precision factor_alloc_double
#define ldlt_tpp_factor ldlt_tpp_factor_dbl
#endif

#ifdef INTEGER_64
#define host_gemm host_gemm_64
#endif

/* SPRAL headers */

namespace spral { namespace ssids { namespace cpu {

template <bool posdef,
          typename T,
          typename FactorAllocator, // Allocator to use for factor storage
          typename PoolAllocator // Allocator for pool memory usage
          >
class SmallLeafNumericSubtree;

/// Positive-definite specialization
template <typename T,
          typename FactorAllocator, // Allocator to use for factor storage
          typename PoolAllocator // Allocator for pool memory usage
          >
class SmallLeafNumericSubtree<true, T, FactorAllocator, PoolAllocator> {
   typedef typename std::allocator_traits<FactorAllocator>::template rebind_traits<rpc_> FAPrecisionTraits;
   typedef typename std::allocator_traits<FactorAllocator>::template rebind_traits<ipc_> FAIntTraits;
   typedef std::allocator_traits<PoolAllocator> PATraits;
public:
   SmallLeafNumericSubtree(SmallLeafSymbolicSubtree const& symb, std::vector<NumericNode<T,PoolAllocator>>& old_nodes, T const* aval, T const* scaling, FactorAllocator& factor_alloc, PoolAllocator& pool_alloc, std::vector<Workspace>& work_vec, struct cpu_factor_options const& options, ThreadStats& stats)
      : old_nodes_(old_nodes), symb_(symb), lcol_(FAPrecisionTraits::allocate(factor_alloc, symb.nfactor_))
   {
      Workspace& work = work_vec[omp_get_thread_num()];
      /* Initialize nodes */
      for(ipc_ ni=symb_.sa_; ni<=symb_.en_; ++ni) {
         old_nodes_[ni].ndelay_in = 0;
         old_nodes_[ni].lcol = lcol_ + symb_[ni-symb_.sa_].lcol_offset;
      }
      memset(lcol_, 0, symb_.nfactor_*sizeof(T));

      /* Add aval entries */
      for(ipc_ ni=symb_.sa_; ni<=symb_.en_; ++ni)
         add_a(ni-symb_.sa_, symb_.symb_[ni], aval, scaling);

      /* Perform factorization */
      for(ipc_ ni=symb_.sa_; ni<=symb_.en_; ++ni) {
         // Assembly
         ipc_* map = work.get_ptr<ipc_>(symb_.symb_.n+1);
         assemble
            (ni-symb_.sa_, symb_.symb_[ni], &old_nodes_[ni], factor_alloc,
             pool_alloc, map, aval, scaling);
         // Update stats
         ipc_ nrow = symb_.symb_[ni].nrow;
         stats.maxfront = std::max(stats.maxfront, nrow);
         ipc_ ncol = symb_.symb_[ni].ncol;
         stats.maxsupernode = std::max(stats.maxsupernode, ncol);
         // Factorization
         rpc_ one_val = 1.0;
         factor_node_posdef
            (one_val, symb_.symb_[ni], old_nodes_[ni], options, stats);
         if(stats.flag<Flag::SUCCESS) return;
      }
   }

private:
void add_a(
      ipc_ si,
      SymbolicNode const& snode,
      T const* aval,
      T const* scaling
      ) {
   rpc_ *lcol = lcol_ + symb_[si].lcol_offset;
   size_t ldl = align_lda<rpc_>(snode.nrow);
   if(scaling) {
      /* Scaling to apply */
      for(ipc_ i=0; i<snode.num_a; i++) {
         longc_ src  = snode.amap[2*i+0] - 1; // amap contains 1-based values
         longc_ dest = snode.amap[2*i+1] - 1; // amap contains 1-based values
         ipc_ c = dest / snode.nrow;
         ipc_ r = dest % snode.nrow;
         T rscale = scaling[ snode.rlist[r]-1 ];
         T cscale = scaling[ snode.rlist[c]-1 ];
         size_t k = c*ldl + r;
         lcol[k] = rscale * aval[src] * cscale;
      }
   } else {
      /* No scaling to apply */
      for(ipc_ i=0; i<snode.num_a; i++) {
         longc_ src  = snode.amap[2*i+0] - 1; // amap contains 1-based values
         longc_ dest = snode.amap[2*i+1] - 1; // amap contains 1-based values
         ipc_ c = dest / snode.nrow;
         ipc_ r = dest % snode.nrow;
         size_t k = c*ldl + r;
         lcol[k] = aval[src];
      }
   }
}

void assemble(
      ipc_ si,
      SymbolicNode const& snode,
      NumericNode<T,PoolAllocator>* node,
      FactorAllocator& factor_alloc,
      PoolAllocator& pool_alloc,
      ipc_* map,
      T const* aval,
      T const* scaling
      ) {
   /* Rebind allocators */
   typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);

   /* Count incoming delays and determine size of node */
   ipc_ nrow = snode.nrow;
   ipc_ ncol = snode.ncol;

   /* Get space for contribution block + zero it */
   longc_ contrib_dimn = snode.nrow - snode.ncol;
   node->contrib = (contrib_dimn > 0) ? PATraits::allocate(pool_alloc, 
      contrib_dimn*contrib_dimn) : nullptr;
   if(node->contrib)
      memset(node->contrib, 0, contrib_dimn*contrib_dimn*sizeof(T));

   /* Alloc + set perm */
   node->perm = FAIntTraits::allocate(factor_alloc_int, ncol); // ncol fully summed variables
   for(ipc_ i=0; i<snode.ncol; i++)
      node->perm[i] = snode.rlist[i];

   /* Add children */
   if(node->first_child != NULL) {
      /* Build lookup vector, allowing for insertion of delayed vars */
      /* Note that while rlist[] is 1-indexed this is fine so long as lookup
       * is also 1-indexed (which it is as it is another node's rlist[] */
      for(ipc_ i=0; i<snode.nrow; i++)
         map[ snode.rlist[i] ] = i;
      /* Loop over children adding contributions */
      for(auto* child=node->first_child; child!=NULL; child=child->next_child) {
         SymbolicNode const& csnode = child->symb;
         /* Handle expected contributions (only if something there) */
         if(child->contrib) {
            ipc_ cm = csnode.nrow - csnode.ncol;
            for(ipc_ i=0; i<cm; i++) {
               ipc_ c = map[ csnode.rlist[csnode.ncol+i] ];
               T *src = &child->contrib[i*cm];
               if(c < snode.ncol) {
                  // Contribution added to lcol
                  ipc_ ldd = align_lda<rpc_>(nrow);
                  T *dest = &node->lcol[c*ldd];
                  for(ipc_ j=i; j<cm; j++) {
                     ipc_ r = map[ csnode.rlist[csnode.ncol+j] ];
                     dest[r] += src[j];
                  }
               } else {
                  // Contribution added to contrib
                  // FIXME: Add after contribution block established?
                  ipc_ ldd = snode.nrow - snode.ncol;
                  T *dest = &node->contrib[(c-ncol)*ldd];
                  for(ipc_ j=i; j<cm; j++) {
                     ipc_ r = map[ csnode.rlist[csnode.ncol+j] ] - ncol;
                     dest[r] += src[j];
                  }
               }
            }
            /* Free memory from child contribution block */
            child->free_contrib();
         }
      }
   }
}

private:
   std::vector<NumericNode<T,PoolAllocator>>& old_nodes_;
   SmallLeafSymbolicSubtree const& symb_;
   T* lcol_;
};

// Indefinite specialization
template <typename T,
          typename FactorAllocator, // Allocator to use for factor storage
          typename PoolAllocator // Allocator for pool memory usage
          >
class SmallLeafNumericSubtree<false, T, FactorAllocator, PoolAllocator> {
   typedef typename std::allocator_traits<FactorAllocator>::template rebind_traits<rpc_> FAPrecisionTraits;
   typedef typename std::allocator_traits<FactorAllocator>::template rebind_traits<ipc_> FAIntTraits;
   typedef std::allocator_traits<PoolAllocator> PATraits;
public:
   SmallLeafNumericSubtree(SmallLeafSymbolicSubtree const& symb, std::vector<NumericNode<T,PoolAllocator>>& old_nodes, T const* aval, T const* scaling, FactorAllocator& factor_alloc, PoolAllocator& pool_alloc, std::vector<Workspace>& work_vec, struct cpu_factor_options const& options, ThreadStats& stats)
   : old_nodes_(old_nodes), symb_(symb)
   {
      Workspace& work = work_vec[omp_get_thread_num()];
      for(ipc_ ni=symb_.sa_; ni<=symb_.en_; ++ni) {
         /*printf("%d: Node %d parent %d (of %d) size %d x %d\n",
               omp_get_thread_num(), ni, symb_[ni].parent, symb_.nnodes_,
               symb_[ni].nrow, symb_[ni].ncol);*/
         // Assembly of node (not of contribution block)
         ipc_* map = work.get_ptr<ipc_>(symb_.symb_.n+1);
         assemble_pre
            (symb_.symb_[ni], old_nodes_[ni], factor_alloc,
             pool_alloc, map, aval, scaling);
         // Update stats
         ipc_ nrow = symb_.symb_[ni].nrow + old_nodes_[ni].ndelay_in;
         stats.maxfront = std::max(stats.maxfront, nrow);
         ipc_ ncol = symb_.symb_[ni].ncol + old_nodes_[ni].ndelay_in;
         stats.maxsupernode = std::max(stats.maxsupernode, ncol);

         // Factorization
         factor_node
            (symb_.symb_[ni], &old_nodes_[ni], options,
             stats, work, pool_alloc);
         if(stats.flag<Flag::SUCCESS) return; // something is wrong

         // Assemble children into contribution block
         assemble_post(symb_.symb_[ni], old_nodes_[ni], pool_alloc, map);
      }
   }

private:
   void assemble_pre(
         SymbolicNode const& snode,
         NumericNode<T,PoolAllocator>& node,
         FactorAllocator& factor_alloc,
         PoolAllocator& pool_alloc,
         ipc_* map,
         T const* aval,
         T const* scaling
         ) {
      /* Rebind allocators */
      typename FAPrecisionTraits::allocator_type factor_alloc_precision(factor_alloc);
      typename FAIntTraits::allocator_type factor_alloc_int(factor_alloc);

      /* Count incoming delays and determine size of node */
      node.ndelay_in = 0;
      for(auto* child=node.first_child; child!=NULL; child=child->next_child) {
         node.ndelay_in += child->ndelay_out;
      }
      ipc_ nrow = snode.nrow + node.ndelay_in;
      ipc_ ncol = snode.ncol + node.ndelay_in;

      /* Get space for node now we know it size using Fortran allocator + zero it*/
      // NB L is  nrow x ncol and D is 2 x ncol (but no D if posdef)
      size_t ldl = align_lda<rpc_>(nrow);
      size_t len = (ldl+2) * ncol; // +2 is for D
      node.lcol = FAPrecisionTraits::allocate(factor_alloc_precision, len);
      memset(node.lcol, 0, len*sizeof(T));

      /* Get space for contribution block + (explicitly do not zero it!) */
      longc_ contrib_dimn = snode.nrow - snode.ncol;
      node.contrib = (contrib_dimn > 0) ? PATraits::allocate(pool_alloc, contrib_dimn*contrib_dimn) : nullptr;

      /* Alloc + set perm for expected eliminations at this node (delays are set
       * when they are imported from children) */
      node.perm = FAIntTraits::allocate(factor_alloc_int, ncol); // ncol fully summed variables
      for(ipc_ i=0; i<snode.ncol; i++)
         node.perm[i] = snode.rlist[i];

      /* Add A */
      if(scaling) {
         /* Scaling to apply */
         for(ipc_ i=0; i<snode.num_a; i++) {
            longc_ src  = snode.amap[2*i+0] - 1; // amap contains 1-based values
            longc_ dest = snode.amap[2*i+1] - 1; // amap contains 1-based values
            ipc_ c = dest / snode.nrow;
            ipc_ r = dest % snode.nrow;
            longc_ k = c*ldl + r;
            if(r >= snode.ncol) k += node.ndelay_in;
            T rscale = scaling[ snode.rlist[r]-1 ];
            T cscale = scaling[ snode.rlist[c]-1 ];
            node.lcol[k] = rscale * aval[src] * cscale;
         }
      } else {
         /* No scaling to apply */
         for(ipc_ i=0; i<snode.num_a; i++) {
            longc_ src  = snode.amap[2*i+0] - 1; // amap contains 1-based values
            longc_ dest = snode.amap[2*i+1] - 1; // amap contains 1-based values
            ipc_ c = dest / snode.nrow;
            ipc_ r = dest % snode.nrow;
            longc_ k = c*ldl + r;
            if(r >= snode.ncol) k += node.ndelay_in;
            node.lcol[k] = aval[src];
         }
      }

      /* Add children */
      if(node.first_child != NULL) {
         /* Build lookup vector, allowing for insertion of delayed vars */
         /* Note that while rlist[] is 1-indexed this is fine so long as lookup
          * is also 1-indexed (which it is as it is another node's rlist[] */
         for(ipc_ i=0; i<snode.ncol; i++)
            map[ snode.rlist[i] ] = i;
         for(ipc_ i=snode.ncol; i<snode.nrow; i++)
            map[ snode.rlist[i] ] = i + node.ndelay_in;
         /* Loop over children adding contributions */
         ipc_ delay_col = snode.ncol;
         for(auto* child=node.first_child; child!=NULL; child=child->next_child) {
            SymbolicNode const& csnode = child->symb;
            /* Handle delays - go to back of node
             * (i.e. become the last rows as in lower triangular format) */
            for(ipc_ i=0; i<child->ndelay_out; i++) {
               // Add delayed rows (from delayed cols)
               T *dest = &node.lcol[delay_col*(ldl+1)];
               ipc_ lds = align_lda<T>(csnode.nrow + child->ndelay_in);
               T *src = &child->lcol[(child->nelim+i)*(lds+1)];
               node.perm[delay_col] = child->perm[child->nelim+i];
               for(ipc_ j=0; j<child->ndelay_out-i; j++) {
                  dest[j] = src[j];
               }
               // Add child's non-fully summed rows (from delayed cols)
               dest = node.lcol;
               src = &child->lcol[child->nelim*lds + child->ndelay_in +i*lds];
               for(ipc_ j=csnode.ncol; j<csnode.nrow; j++) {
                  ipc_ r = map[ csnode.rlist[j] ];
                  if(r < ncol) dest[r*ldl+delay_col] = src[j];
                  else         dest[delay_col*ldl+r] = src[j];
               }
               delay_col++;
            }

            /* Handle expected contributions (only if something there) */
            if(child->contrib) {
               ipc_ cm = csnode.nrow - csnode.ncol;
               for(ipc_ i=0; i<cm; i++) {
                  ipc_ c = map[ csnode.rlist[csnode.ncol+i] ];
                  T *src = &child->contrib[i*cm];
                  // NB: we handle contribution to contrib in assemble_post()
                  if(c < snode.ncol) {
                     // Contribution added to lcol
                     ipc_ ldd = align_lda<T>(nrow);
                     T *dest = &node.lcol[c*ldd];
                     for(ipc_ j=i; j<cm; j++) {
                        ipc_ r = map[ csnode.rlist[csnode.ncol+j] ];
                        dest[r] += src[j];
                     }
                  }
               }
            }
         }
      }
   }

   /* Factorize a node (indef) */
   void factor_node(
         SymbolicNode const& snode,
         NumericNode<T,PoolAllocator>* node,
         struct cpu_factor_options const& options,
         ThreadStats& stats,
         Workspace& work,
         PoolAllocator& pool_alloc
         ) {
      /* Extract useful information about node */
      ipc_ m = snode.nrow + node->ndelay_in;
      ipc_ n = snode.ncol + node->ndelay_in;
      size_t ldl = align_lda<T>(m);
      T *lcol = node->lcol;
      T *d = &node->lcol[ n*ldl ];
      ipc_ *perm = node->perm;

      /* Perform factorization */
      //Verify<T> verifier(m, n, perm, lcol, ldl);
      T *ld = work.get_ptr<T>(2*m);
      node->nelim = ldlt_tpp_factor(
            m, n, perm, lcol, ldl, d, ld, m, options.action, options.u,
            options.small
            );
      //verifier.verify(node->nelim, perm, lcol, ldl, d);

      if(m-n>0 && node->nelim>0) {
         ipc_ nelim = node->nelim;
         ipc_ ldld = align_lda<T>(m-n);
         T *ld = work.get_ptr<T>(nelim*ldld);
         calcLD<OP_N>(m-n, nelim, &lcol[n], ldl, d, ld, ldld);
         host_gemm<T>(OP_N, OP_T, m-n, m-n, nelim,
               -1.0, &lcol[n], ldl, ld, ldld,
               0.0, node->contrib, m-n);
      }

      /* Record information */
      node->ndelay_out = n - node->nelim;
      stats.num_delay += node->ndelay_out;
      for (longc_ j = m; j >= m-(node->nelim)+1; --j) {
          stats.num_factor += j;
          stats.num_flops += j*j;
      }

      /* Mark as no contribution if we make no contribution */
      if(node->nelim==0 && !node->first_child) {
         // FIXME: Actually loop over children and check one exists with contrib
         //        rather than current approach of just looking for children.
         node->free_contrib();
      } else if(node->nelim==0) {
         // FIXME: If we fix the above, we don't need this explict zeroing
         longc_ contrib_size = m-n;
         memset(node->contrib, 0, contrib_size*contrib_size*sizeof(T));
      }
   }

   void assemble_post(
         SymbolicNode const& snode,
         NumericNode<T,PoolAllocator>& node,
         PoolAllocator& pool_alloc,
         ipc_* map
         ) {
      /* Initialise variables */
      ipc_ ncol = snode.ncol + node.ndelay_in;

      /* Add children */
      if(node.first_child != NULL) {
         /* Build lookup vector, allowing for insertion of delayed vars */
         /* Note that while rlist[] is 1-indexed this is fine so long as lookup
          * is also 1-indexed (which it is as it is another node's rlist[] */
         for(ipc_ i=0; i<snode.ncol; i++)
            map[ snode.rlist[i] ] = i;
         for(ipc_ i=snode.ncol; i<snode.nrow; i++)
            map[ snode.rlist[i] ] = i + node.ndelay_in;
         /* Loop over children adding contributions */
         for(auto* child=node.first_child; child!=NULL; child=child->next_child) {
            SymbolicNode const& csnode = child->symb;
            if(!child->contrib) continue;
            ipc_ cm = csnode.nrow - csnode.ncol;
            for(ipc_ i=0; i<cm; i++) {
               ipc_ c = map[ csnode.rlist[csnode.ncol+i] ];
               T *src = &child->contrib[i*cm];
               // NB: only interested in contribution to generated element
               if(c >= snode.ncol) {
                  // Contribution added to contrib
                  ipc_ ldd = snode.nrow - snode.ncol;
                  T *dest = &node.contrib[(c-ncol)*ldd];
                  for(ipc_ j=i; j<cm; j++) {
                     ipc_ r = map[ csnode.rlist[csnode.ncol+j] ] - ncol;
                     dest[r] += src[j];
                  }
               }
            }
            /* Free memory from child contribution block */
            child->free_contrib();
         }
      }
   }

   std::vector<NumericNode<T,PoolAllocator>>& old_nodes_;
   SmallLeafSymbolicSubtree const& symb_;
};

}}} /* namespaces spral::ssids::cpu */
