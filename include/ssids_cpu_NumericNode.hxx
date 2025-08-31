/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 08:00 GMT
 */

#pragma once

#include "ssids_routines.h"
#include "galahad_precision.h"

namespace galahad { namespace ssids { namespace cpu {

class SymbolicNode;

template <typename T, typename PoolAllocator>
class NumericNode {
   typedef std::allocator_traits<PoolAllocator> PATraits;
public:
   /**
    * \brief Constructor
    * \param symb Associated symbolic node.
    * \param pool_alloc Pool Allocator to use for contrib allocation.
    */
   NumericNode(SymbolicNode const& symb, PoolAllocator const& pool_alloc)
   : symb(symb), contrib(nullptr), pool_alloc_(pool_alloc)
   {}
   /**
    * \brief Destructor
    */
   ~NumericNode() {
      free_contrib();
   }

   /**
    * \brief Allocate space for contribution block.
    *
    * Note done at construction time, as a major memory commitment that is
    * transitory.
    */
   void alloc_contrib() {
      size_t contrib_dimn = symb.nrow - symb.ncol;
      contrib_dimn = contrib_dimn*contrib_dimn;
      contrib = (contrib_dimn>0) ? PATraits::allocate(pool_alloc_, contrib_dimn)
                                 : nullptr;
   }

   /** \brief Free space for contribution block (if allocated) */
   void free_contrib() {
      if(!contrib) return;
      size_t contrib_dimn = symb.nrow - symb.ncol;
      contrib_dimn = contrib_dimn*contrib_dimn;
      PATraits::deallocate(pool_alloc_, contrib, contrib_dimn);
      contrib = nullptr;
   }

   /** \brief Return leading dimension of node's lcol member. */
   size_t get_ldl() {
      return align_lda<T>(symb.nrow + ndelay_in);
   }

public:
   /* Symbolic node associate with this one */
   SymbolicNode const& symb;

   /* Fixed data from analyse */
   NumericNode<T, PoolAllocator>* first_child; // Pointer to our first child
   NumericNode<T, PoolAllocator>* next_child; // Pointer to parent's next child

   /* Data that changes during factorize */
   ipc_ ndelay_in; // Number of delays arising from children
   ipc_ ndelay_out; // Number of delays arising to push into parent
   ipc_ nelim; // Number of columns succesfully eliminated
   T *lcol; // Pointer to start of factor data
   ipc_ *perm; // Pointer to permutation
   T *contrib; // Pointer to contribution block
private:
   PoolAllocator pool_alloc_; // Our own version of pool allocator for freeing
                              // contrib
};

}}} /* namespaces galahad::ssids::cpu */
