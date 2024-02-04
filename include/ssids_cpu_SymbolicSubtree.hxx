/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-03 AT 15:00 GMT
 */
#pragma once

#include <cstddef>
#include <vector>
#include <stdio.h>

#include "ssids_rip.hxx"
#include "ssids_cpu_SmallLeafSymbolicSubtree.hxx"
#include "ssids_cpu_SymbolicNode.hxx"

namespace spral { namespace ssids { namespace cpu {

/** Symbolic factorization of a subtree to be factored on the CPU */
class SymbolicSubtree {
public:
   SymbolicSubtree(ipc_ n, ipc_ sa, ipc_ en, ipc_ const* sptr, 
                   ipc_ const* sparent, longc_ const* rptr, 
                   ipc_ const* rlist, longc_ const* nptr, 
                   longc_ const* nlist, ipc_ ncontrib, 
                   ipc_ const* contrib_idx, 
                   struct cpu_factor_options const& options)
   : n(n), nnodes_(en-sa), nodes_(nnodes_+1)
   {
      // Adjust sa to C indexing (en is not used except in nnodes_ init above)
      sa--;
      // FIXME: don't process nodes that are in small leaf subtrees
      /* Fill out basic details */
      maxfront_ = 0;
      for(ipc_ ni=0; ni<nnodes_; ++ni) {
         nodes_[ni].idx = ni;
         nodes_[ni].nrow = static_cast<ipc_>(rptr[sa+ni+1] - rptr[sa+ni]);
         nodes_[ni].ncol = sptr[sa+ni+1] - sptr[sa+ni];
         nodes_[ni].first_child = nullptr;
         nodes_[ni].next_child = nullptr;
         nodes_[ni].rlist = &rlist[rptr[sa+ni]-1]; // rptr is Fortran indexed
         nodes_[ni].num_a = nptr[sa+ni+1] - nptr[sa+ni];
         nodes_[ni].amap = &nlist[2*(nptr[sa+ni]-1)]; // nptr is Fortran indexed
         nodes_[ni].parent = sparent[sa+ni]-sa-1; // sparent is Fortran indexed
         nodes_[ni].insmallleaf = false; // default to not in small leaf subtree
         maxfront_ = std::max(maxfront_, (size_t) nodes_[ni].nrow);
      }
      nodes_[nnodes_].first_child = nullptr; // List of roots
      /* Build child linked lists */
      for(ipc_ ni=0; ni<nnodes_; ++ni) {
         SymbolicNode *parent = &nodes_[ std::min(nodes_[ni].parent, nnodes_) ];
         nodes_[ni].next_child = parent->first_child;
         parent->first_child = &nodes_[ni];
      }
      /* Record contribution block inputs */
      for(ipc_ ci=0; ci<ncontrib; ++ci) {
         ipc_ idx = contrib_idx[ci]-1 - sa; // contrib_idx is Fortran indexed
         nodes_[idx].contrib.push_back(ci);
      }
      /* Count size of factors */
      nfactor_ = 0;
      for(ipc_ ni=0; ni<nnodes_; ++ni)
         nfactor_ += static_cast<size_t>(nodes_[ni].nrow)*nodes_[ni].ncol;
      /* Find small leaf subtrees */
      // Count flops below each node
      std::vector<longc_> flops(nnodes_+1, 0);
      for(ipc_ ni=0; ni<nnodes_; ++ni) {
         for(ipc_ k=0; k<nodes_[ni].ncol; ++k)
            flops[ni] += (nodes_[ni].nrow - k)*(nodes_[ni].nrow - k);
         if(nodes_[ni].contrib.size() > 0) // not a leaf!
            flops[ni] += options.small_subtree_threshold;
         ipc_ parent = std::min(nodes_[ni].parent, nnodes_);
         flops[parent] += flops[ni];
      }
      // Start at least node and work way up using parents until too large
      for(ipc_ ni=0; ni<nnodes_; ) {
         if(nodes_[ni].first_child) { ++ni; continue; } // Not a leaf
         ipc_ last = ni;
         for(ipc_ current=ni; current<nnodes_; current=nodes_[current].parent) {
            if(flops[current] >= options.small_subtree_threshold) break;
            last = current;
         }
         if(last==ni) { ++ni; continue; } // No point for a single node
         // Nodes ni:last are in subtree
         small_leafs_.emplace_back(
               ni, last, sa, sptr, sparent, rptr, rlist, nptr, nlist, *this
               );
         for(ipc_ i=ni; i<=last; ++i)
            nodes_[i].insmallleaf = true;
         ni = last+1; // Skip to next node not in this subtree
      }
   }

   SymbolicNode const& operator[](ipc_ idx) const {
      return nodes_[idx];
   }
   size_t get_factor_mem_est(rpc_ multiplier) const {
      size_t mem = n*sizeof(ipc_) + (2*n+nfactor_)*sizeof(rpc_);
      return std::max(mem, static_cast<size_t>(mem*multiplier));
   }
   template <typename T>
   size_t get_pool_size() const {
      return maxfront_*align_lda<rpc_>(maxfront_);
   }
public:
   ipc_ const n; //< Maximum row index
private:
   ipc_ nnodes_;
   size_t nfactor_;
   size_t maxfront_;
   std::vector<SymbolicNode> nodes_;
   std::vector<SmallLeafSymbolicSubtree> small_leafs_;

   template <bool posdef, typename T, size_t PAGE_SIZE, typename FactorAlloc>
   friend class NumericSubtree;
};

}}} /* end of namespace spral::ssids::cpu */
