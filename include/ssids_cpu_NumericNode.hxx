/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

namespace spral { namespace ssids { namespace cpu {

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
   int ndelay_in; // Number of delays arising from children
   int ndelay_out; // Number of delays arising to push into parent
   int nelim; // Number of columns succesfully eliminated
   T *lcol; // Pointer to start of factor data
   int *perm; // Pointer to permutation
   T *contrib; // Pointer to contribution block
private:
   PoolAllocator pool_alloc_; // Our own version of pool allocator for freeing
                              // contrib
};

}}} /* namespaces spral::ssids::cpu */
