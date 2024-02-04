/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 4.3 - 2024-02-03 AT 16:00 GMT
 */

#include "ssids_rip.hxx"
#include "ssids_cpu_SymbolicSubtree.hxx"

using namespace spral::ssids::cpu;

extern "C"
void* spral_ssids_cpu_create_symbolic_subtree(
      ipc_ n, ipc_ sa, ipc_ en, ipc_ const* sptr, ipc_ const* sparent,
      longc_ const* rptr, ipc_ const* rlist, longc_ const* nptr, 
      longc_ const* nlist, ipc_ ncontrib, ipc_ const* contrib_idx,
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
