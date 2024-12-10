/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   GALAHAD 5.1 - 2024-11-26 AT 12:45 GMT
 */

#include "ssids_cpu_NumericSubtree.hxx"

#include <cassert>
#include <cstdio>
#include <memory>

#include "spral_omp.hxx"
#include "ssids_cpu_AppendAlloc.hxx"
#include "ssids_rip.hxx"

#ifdef REAL_32
#define spral_ssids_cpu_create_num_subtree \
        spral_ssids_cpu_create_num_subtree_sgl
#define spral_ssids_cpu_destroy_num_subtree \
        spral_ssids_cpu_destroy_num_subtree_sgl
#define spral_ssids_cpu_subtree_solve_fwd \
        spral_ssids_cpu_subtree_solve_fwd_sgl
#define spral_ssids_cpu_subtree_solve_diag \
        spral_ssids_cpu_subtree_solve_diag_sgl
#define spral_ssids_cpu_subtree_solve_diag_bwd \
        spral_ssids_cpu_subtree_solve_diag_bwd_sgl
#define spral_ssids_cpu_subtree_solve_bwd \
        spral_ssids_cpu_subtree_solve_bwd_sgl
#define spral_ssids_cpu_subtree_enquire \
        spral_ssids_cpu_subtree_enquire_sgl
#define spral_ssids_cpu_subtree_alter \
        spral_ssids_cpu_subtree_alter_sgl
#define spral_ssids_cpu_subtree_get_contrib \
        spral_ssids_cpu_subtree_get_contrib_sgl
#define spral_ssids_cpu_subtree_free_contrib \
        spral_ssids_cpu_subtree_free_contrib_sgl
#elif REAL_128
#define spral_ssids_cpu_create_num_subtree \
        spral_ssids_cpu_create_num_subtree_qul
#define spral_ssids_cpu_destroy_num_subtree \
        spral_ssids_cpu_destroy_num_subtree_qul
#define spral_ssids_cpu_subtree_solve_fwd \
        spral_ssids_cpu_subtree_solve_fwd_qul
#define spral_ssids_cpu_subtree_solve_diag \
        spral_ssids_cpu_subtree_solve_diag_qul
#define spral_ssids_cpu_subtree_solve_diag_bwd \
        spral_ssids_cpu_subtree_solve_diag_bwd_qul
#define spral_ssids_cpu_subtree_solve_bwd \
        spral_ssids_cpu_subtree_solve_bwd_qul
#define spral_ssids_cpu_subtree_enquire \
        spral_ssids_cpu_subtree_enquire_qul
#define spral_ssids_cpu_subtree_alter \
        spral_ssids_cpu_subtree_alter_qul
#define spral_ssids_cpu_subtree_get_contrib \
        spral_ssids_cpu_subtree_get_contrib_qul
#define spral_ssids_cpu_subtree_free_contrib \
        spral_ssids_cpu_subtree_free_contrib_qul
#else
#define spral_ssids_cpu_create_num_subtree \
        spral_ssids_cpu_create_num_subtree_dbl
#define spral_ssids_cpu_destroy_num_subtree \
        spral_ssids_cpu_destroy_num_subtree_dbl
#define spral_ssids_cpu_subtree_solve_fwd \
        spral_ssids_cpu_subtree_solve_fwd_dbl
#define spral_ssids_cpu_subtree_solve_diag \
        spral_ssids_cpu_subtree_solve_diag_dbl
#define spral_ssids_cpu_subtree_solve_diag_bwd \
        spral_ssids_cpu_subtree_solve_diag_bwd_dbl
#define spral_ssids_cpu_subtree_solve_bwd \
        spral_ssids_cpu_subtree_solve_bwd_dbl
#define spral_ssids_cpu_subtree_enquire \
        spral_ssids_cpu_subtree_enquire_dbl
#define spral_ssids_cpu_subtree_alter \
        spral_ssids_cpu_subtree_alter_dbl
#define spral_ssids_cpu_subtree_get_contrib \
        spral_ssids_cpu_subtree_get_contrib_dbl
#define spral_ssids_cpu_subtree_free_contrib \
        spral_ssids_cpu_subtree_free_contrib_dbl
#endif

using namespace spral::ssids::cpu;

/////////////////////////////////////////////////////////////////////////////
// anonymous namespace
namespace {

#ifdef REAL_32
typedef float T;
#elif REAL_128
typedef __float128 T;
#else
typedef double T;
#endif
const ipc_ PAGE_SIZE = 8*1024*1024; // 8MB
typedef NumericSubtree<true, T, PAGE_SIZE, AppendAlloc<T>> NumericSubtreePosdef;
typedef NumericSubtree<false, T, PAGE_SIZE, AppendAlloc<T>> NumericSubtreeIndef;

} /* end of anon namespace */
//////////////////////////////////////////////////////////////////////////

extern "C"
void* spral_ssids_cpu_create_num_subtree(
      bool posdef,
      void const* symbolic_subtree_ptr,
      const rpc_ *const aval, // Values of A
      const rpc_ *const scaling, // Scaling vector (NULL if none)
      void** child_contrib, // Contributions from child subtrees
      struct cpu_factor_options const* options, // Options in
      ThreadStats* stats // Info out
      ) {
   auto const& symbolic_subtree = 
      *static_cast<SymbolicSubtree const*>(symbolic_subtree_ptr);

   // Perform factorization
   if(posdef) {
      auto* subtree = new NumericSubtreePosdef
         (symbolic_subtree, aval, scaling, child_contrib, *options, *stats);
      if(options->print_level > 9999) {
         printf("Final factors:\n");
         subtree->print();
      }
      return (void*) subtree;
   } else { /* indef */
      auto* subtree = new NumericSubtreeIndef
         (symbolic_subtree, aval, scaling, child_contrib, *options, *stats);
      if(options->print_level > 9999) {
         printf("Final factors:\n");
         subtree->print();
      }
      return (void*) subtree;
   }
}

extern "C"
void spral_ssids_cpu_destroy_num_subtree(bool posdef, void* target) {
   if(!target) return;

   if(posdef) {
      auto *subtree = static_cast<NumericSubtreePosdef*>(target);
      delete subtree;
   } else {
      auto *subtree = static_cast<NumericSubtreeIndef*>(target);
      delete subtree;
   }
}

/* wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_fwd(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      ipc_ nrhs,         // number of right-hand sides
      rpc_* x,        // ldx x nrhs array of right-hand sides
      ipc_ ldx           // leading dimension of x
      ) {

   // Call method
   try {
      if(posdef) { // Converting from runtime to compile time posdef value
         auto &subtree =
            *static_cast<NumericSubtreePosdef const*>(subtree_ptr);
         subtree.solve_fwd(nrhs, x, ldx);
      } else {
         auto &subtree =
            *static_cast<NumericSubtreeIndef const*>(subtree_ptr);
         subtree.solve_fwd(nrhs, x, ldx);
      }
   } catch(std::bad_alloc const&) {
      return Flag::ERROR_ALLOCATION;
   }
   return Flag::SUCCESS;
}

/* wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_diag(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      ipc_ nrhs,         // number of right-hand sides
      rpc_* x,        // ldx x nrhs array of right-hand sides
      ipc_ ldx           // leading dimension of x
      ) {

   // Call method
   try {
      if(posdef) { // Converting from runtime to compile time posdef value
         auto &subtree = *static_cast<NumericSubtreePosdef const*>(subtree_ptr);
         subtree.solve_diag(nrhs, x, ldx);
      } else {
         auto &subtree = *static_cast<NumericSubtreeIndef const*>(subtree_ptr);
         subtree.solve_diag(nrhs, x, ldx);
      }
   } catch(std::bad_alloc const&) {
      return Flag::ERROR_ALLOCATION;
   }
   return Flag::SUCCESS;
}

/* wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_diag_bwd(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      ipc_ nrhs,         // number of right-hand sides
      rpc_* x,        // ldx x nrhs array of right-hand sides
      ipc_ ldx           // leading dimension of x
      ) {

   // Call method
   try {
      if(posdef) { // Converting from runtime to compile time posdef value
         auto &subtree =
            *static_cast<NumericSubtreePosdef const*>(subtree_ptr);
         subtree.solve_diag_bwd(nrhs, x, ldx);
      } else {
         auto &subtree =
            *static_cast<NumericSubtreeIndef const*>(subtree_ptr);
         subtree.solve_diag_bwd(nrhs, x, ldx);
      }
   } catch(std::bad_alloc const&) {
      return Flag::ERROR_ALLOCATION;
   }
   return Flag::SUCCESS;
}

/* wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_bwd(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      ipc_ nrhs,         // number of right-hand sides
      rpc_* x,        // ldx x nrhs array of right-hand sides
      ipc_ ldx           // leading dimension of x
      ) {

   // Call method
   try {
      if(posdef) { // Converting from runtime to compile time posdef value
         auto &subtree =
            *static_cast<NumericSubtreePosdef const*>(subtree_ptr);
         subtree.solve_bwd(nrhs, x, ldx);
      } else {
         auto &subtree =
            *static_cast<NumericSubtreeIndef const*>(subtree_ptr);
         subtree.solve_bwd(nrhs, x, ldx);
      }
   } catch(std::bad_alloc const&) {
      return Flag::ERROR_ALLOCATION;
   }
   return Flag::SUCCESS;
}

/* wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_enquire(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      ipc_* piv_order,   // pivot order, may be null, only used if indef
      rpc_* d         // diagonal entries, may be null
      ) {

   // Call method
   if(posdef) { // Converting from runtime to compile time posdef value
      auto &subtree =
         *static_cast<NumericSubtreePosdef const*>(subtree_ptr);
      subtree.enquire(piv_order, d);
   } else {
      auto &subtree =
         *static_cast<NumericSubtreeIndef const*>(subtree_ptr);
      subtree.enquire(piv_order, d);
   }
}

/* wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_alter(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void* subtree_ptr,// pointer to relevant type of NumericSubtree
      rpc_ const* d   // new diagonal entries
      ) {

   assert(!posdef); // Should never be called on positive definite matrices.

   // Call method
   auto &subtree = *static_cast<NumericSubtreeIndef*>(subtree_ptr);
   subtree.alter(d);
}

/* wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_get_contrib(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void* subtree_ptr,// pointer to relevant type of NumericSubtree
      ipc_* n,           // returned dimension of contribution block
      rpc_ const** val,     // returned pointer to contribution block
      ipc_* ldval,       // leading dimension of val
      ipc_ const** rlist,      // returned pointer to row list
      ipc_* ndelay,      // returned number of delays
      ipc_ const** delay_perm,  // returned pointer to delay values
      rpc_ const** delay_val,  // returned pointer to delay values
      ipc_* lddelay      // leading dimension of delay_val
      ) {
   // Call method
   if(posdef) { // Converting from runtime to compile time posdef value
      auto &subtree =
         *static_cast<NumericSubtreePosdef*>(subtree_ptr);
      subtree.get_contrib(
            *n, *val, *ldval, *rlist, *ndelay, *delay_perm, *delay_val, *lddelay
            );
   } else {
      auto &subtree =
         *static_cast<NumericSubtreeIndef*>(subtree_ptr);
      subtree.get_contrib(
            *n, *val, *ldval, *rlist, *ndelay, *delay_perm, *delay_val, *lddelay
            );
   }
}

/* wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_free_contrib(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void* subtree_ptr // pointer to relevant type of NumericSubtree
      ) {
   // Call method
   if(posdef) { // Converting from runtime to compile time posdef value
      auto &subtree =
         *static_cast<NumericSubtreePosdef*>(subtree_ptr);
      subtree.free_contrib();
   } else {
      auto &subtree =
         *static_cast<NumericSubtreeIndef*>(subtree_ptr);
      subtree.free_contrib();
   }
}
