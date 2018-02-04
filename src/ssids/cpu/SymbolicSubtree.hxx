/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <cstddef>
#include <vector>

#include "ssids/cpu/SmallLeafSymbolicSubtree.hxx"
#include "ssids/cpu/SymbolicNode.hxx"

namespace spral { namespace ssids { namespace cpu {

/** Symbolic factorization of a subtree to be factored on the CPU */
class SymbolicSubtree {
public:
   SymbolicSubtree(int n, int sa, int en, int const* sptr, int const* sparent, long const* rptr, int const* rlist, long const* nptr, long const* nlist, int ncontrib, int const* contrib_idx, struct cpu_factor_options const& options)
   : n(n), nnodes_(en-sa), nodes_(nnodes_+1)
   {
      // Adjust sa to C indexing (en is not used except in nnodes_ init above)
      sa--;
      // FIXME: don't process nodes that are in small leaf subtrees
      /* Fill out basic details */
      maxfront_ = 0;
      for(int ni=0; ni<nnodes_; ++ni) {
         nodes_[ni].idx = ni;
         nodes_[ni].nrow = static_cast<int>(rptr[sa+ni+1] - rptr[sa+ni]);
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
      for(int ni=0; ni<nnodes_; ++ni) {
         SymbolicNode *parent = &nodes_[ std::min(nodes_[ni].parent, nnodes_) ];
         nodes_[ni].next_child = parent->first_child;
         parent->first_child = &nodes_[ni];
      }
      /* Record contribution block inputs */
      for(int ci=0; ci<ncontrib; ++ci) {
         int idx = contrib_idx[ci]-1 - sa; // contrib_idx is Fortran indexed
         nodes_[idx].contrib.push_back(ci);
      }
      /* Count size of factors */
      nfactor_ = 0;
      for(int ni=0; ni<nnodes_; ++ni)
         nfactor_ += static_cast<size_t>(nodes_[ni].nrow)*nodes_[ni].ncol;
      /* Find small leaf subtrees */
      // Count flops below each node
      std::vector<long> flops(nnodes_+1, 0);
      for(int ni=0; ni<nnodes_; ++ni) {
         for(int k=0; k<nodes_[ni].ncol; ++k)
            flops[ni] += (nodes_[ni].nrow - k)*(nodes_[ni].nrow - k);
         if(nodes_[ni].contrib.size() > 0) // not a leaf!
            flops[ni] += options.small_subtree_threshold;
         int parent = std::min(nodes_[ni].parent, nnodes_);
         flops[parent] += flops[ni];
      }
      // Start at least node and work way up using parents until too large
      for(int ni=0; ni<nnodes_; ) {
         if(nodes_[ni].first_child) { ++ni; continue; } // Not a leaf
         int last = ni;
         for(int current=ni; current<nnodes_; current=nodes_[current].parent) {
            if(flops[current] >= options.small_subtree_threshold) break;
            last = current;
         }
         if(last==ni) { ++ni; continue; } // No point for a single node
         // Nodes ni:last are in subtree
         small_leafs_.emplace_back(
               ni, last, sa, sptr, sparent, rptr, rlist, nptr, nlist, *this
               );
         for(int i=ni; i<=last; ++i)
            nodes_[i].insmallleaf = true;
         ni = last+1; // Skip to next node not in this subtree
      }
   }

   SymbolicNode const& operator[](int idx) const {
      return nodes_[idx];
   }
   size_t get_factor_mem_est(double multiplier) const {
      size_t mem = n*sizeof(int) + (2*n+nfactor_)*sizeof(double);
      return std::max(mem, static_cast<size_t>(mem*multiplier));
   }
   template <typename T>
   size_t get_pool_size() const {
      return maxfront_*align_lda<double>(maxfront_);
   }
public:
   int const n; //< Maximum row index
private:
   int nnodes_;
   size_t nfactor_;
   size_t maxfront_;
   std::vector<SymbolicNode> nodes_;
   std::vector<SmallLeafSymbolicSubtree> small_leafs_;

   template <bool posdef, typename T, size_t PAGE_SIZE, typename FactorAlloc>
   friend class NumericSubtree;
};

}}} /* end of namespace spral::ssids::cpu */
