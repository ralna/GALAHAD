/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#include "ssids/cpu/SymbolicSubtree.hxx"

using namespace spral::ssids::cpu;

extern "C"
void* spral_ssids_cpu_create_symbolic_subtree(
      int n, int sa, int en, int const* sptr, int const* sparent,
      long const* rptr, int const* rlist, long const* nptr, long const* nlist,
      int ncontrib, int const* contrib_idx,
      struct cpu_factor_options const* options) {
   return (void*) new SymbolicSubtree(
         n, sa, en, sptr, sparent, rptr, rlist, nptr, nlist, ncontrib,
         contrib_idx, *options
         );
}

extern "C"
void spral_ssids_cpu_destroy_symbolic_subtree(void* target) {
   if(!target) return;

   auto *subtree = static_cast<SymbolicSubtree*>(target);
   delete subtree;
}
