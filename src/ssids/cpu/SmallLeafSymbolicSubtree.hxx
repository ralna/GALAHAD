/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <memory>

#include "ssids/cpu/cpu_iface.hxx"
#include "ssids/cpu/SymbolicNode.hxx"

namespace spral { namespace ssids { namespace cpu {

class SymbolicSubtree;

/** Handles the factorization of a small leaf subtree on a single core.
 *
 * This code uses supernodal working within the tree, and generates a
 * multifrontal-style contribution block above the tree. The analyse phase
 * generates internal data structures that guide the assembly process in an
 * efficient fashion, aiming to maximize vectorization.
 *
 * It is expected that the subtree will fit within L2 cache exclusively owned
 * by the executing thread.
 */
class SmallLeafSymbolicSubtree {
private:
   class Node {
   public:
      int nrow;
      int ncol;
      int sparent;
      int* rlist;
      int lcol_offset;
   };

public:
   /** 
    * \brief Constructor
    *
    * Perform work in the analyse phase of the solver. Set up data structures
    * for fast numerical factorization. We are passed datastructures describing
    * the whole tree, and consider the subtree composed of contigously numbered
    * nodes sa:en.
    *
    * NB: The global tree is split into parts (parttrees), which then futher
    * split themselves into small leaf subtrees like this.
    *
    * \param sa First (start) supernode in subtree, from start of containing
    *        parttree.
    * \param en Last (end) supernode in subtree, from start of containing
    *        parttree.
    * \param part_offset Offset of containing parttree into global tree.
    * \param sptr Supernode pointers. Supernode i consists of columns
    *        sptr[i]:sptr[i+1]-1. Entries of sptr are numbered from 1 not 0.
    * \param sparent Supernode parent list. Supernode i has parent sparent[i].
    *        If sparent[i]>part_offset+en then it belongs to a parent subtree.
    *        Or is a virtual root if node i is square.
    * \param rptr Row list pointers. Supernode i consists of rows
    *        row_list[rptr[i]-1:rptr[i+1]-1-1]. Note entries are numbered from 1
    *        not 0.
    * \param rlist Row lists. Supernode i consists of rows
    *        row_list[rptr[i]-1:rptr[i+1]-1-1]. Note entries are numbered from 1
    *        not 0.
    * \param nptr Node pointers for map from \f$ A \f$ to \f$ L \f$. Node i
    *        has map entries nlist[2*(nptr[i]-1):2*(nptr[i+1]-1-1)+1].
    *        Note entries are numbered from 1 not 0.
    * \param nlist Mapping from \f$ A \f$ to \f$ L \f$. Each map entry is a
    *        pair such that entry nlist[2*i+0] of \f$ A \f$ maps to entry
    *        nlist[2*i+1] of the relevant supernode (as per nptr) of \f$ L \f$.
    * \param symb Underlying SymbolicSubtree for containing parttree.
    */
   SmallLeafSymbolicSubtree(int sa, int en, int part_offset, int const* sptr, int const* sparent, long const* rptr, int const* rlist, long const* nptr, long const* nlist, SymbolicSubtree const& symb)
   : sa_(sa), en_(en), nnodes_(en-sa+1), parent_(sparent[part_offset+en]-1-part_offset),
     nodes_(nnodes_),
     rlist_(new int[rptr[part_offset+en+1]-rptr[part_offset+sa]], std::default_delete<int[]>()),
     nptr_(nptr), nlist_(nlist), symb_(symb)
   {
      /* Setup basic node information */
      nfactor_ = 0;
      int* newrlist = rlist_.get();
      for(int ni=sa; ni<=en; ++ni) {
         nodes_[ni-sa].nrow = rptr[part_offset+ni+1] - rptr[part_offset+ni];
         nodes_[ni-sa].ncol = sptr[part_offset+ni+1] - sptr[part_offset+ni];
         nodes_[ni-sa].sparent = sparent[part_offset+ni]-sa-1; // sparent is Fortran indexed
         // FIXME: subtract ncol off rlist for elim'd vars
         nodes_[ni-sa].rlist = &newrlist[rptr[part_offset+ni]-rptr[part_offset+sa]];
         nodes_[ni-sa].lcol_offset = nfactor_;
         size_t ldl = align_lda<double>(nodes_[ni-sa].nrow);
         nfactor_ += nodes_[ni-sa].ncol*ldl;
      }
      /* Construct rlist_ being offsets into parent node */
      for(int ni=sa; ni<=en; ++ni) {
         if(nodes_[ni-sa].ncol == nodes_[ni-sa].nrow) continue; // is root
         int const* ilist = &rlist[rptr[part_offset+ni]-1]; // rptr is Fortran indexed
         ilist += nodes_[ni-sa].ncol; // Skip eliminated vars
         int pnode = sparent[part_offset+ni]-1; //Fortran indexed
         int const* jlist = &rlist[rptr[pnode]-1]; // rptr is Fortran indexed
         int const* jstart = jlist;
         int *outlist = nodes_[ni-sa].rlist;
         for(int i=nodes_[ni-sa].ncol; i<nodes_[ni-sa].nrow; ++i) {
            for(; *ilist != *jlist; ++jlist); // Finds match in jlist
            *(outlist++) = jlist - jstart;
            ++ilist;
         }
      }
   }

   /** \brief Return parent node of subtree in parttree indexing. */
   int get_parent() const { return parent_; }
   /** \brief Return given node of this tree. */
   Node const& operator[](int idx) const { return nodes_[idx]; }
protected:
   int sa_; //< First node in subtree.
   int en_; //< Last node in subtree.
   int nnodes_; //< Number of nodes in subtree.
   int nfactor_; //< Number of entries in factor for subtree.
   int parent_; //< Parent of subtree in parttree.
   std::vector<Node> nodes_; //< Nodes of this subtree.
   std::shared_ptr<int> rlist_; //< Row entries of this subtree.
   long const* nptr_; //< Node mapping into nlist_.
   long const* nlist_; //< Mapping from \f$ A \f$ to \f$ L \f$.
   SymbolicSubtree const& symb_; //< Underlying parttree
   
   template <bool posdef, typename T, typename FactorAllocator,
             typename PoolAllocator>
   friend class SmallLeafNumericSubtree;
};


}}} /* namespaces spral::ssids::cpu */
