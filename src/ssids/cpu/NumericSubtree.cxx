/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#include "ssids/cpu/NumericSubtree.hxx"

#include <cassert>
#include <cstdio>
#include <memory>

#include "omp.hxx"
#include "ssids/cpu/AppendAlloc.hxx"

using namespace spral::ssids::cpu;

/////////////////////////////////////////////////////////////////////////////
// anonymous namespace
namespace {

typedef double T;
const int PAGE_SIZE = 8*1024*1024; // 8MB
typedef NumericSubtree<true, T, PAGE_SIZE, AppendAlloc<T>> NumericSubtreePosdef;
typedef NumericSubtree<false, T, PAGE_SIZE, AppendAlloc<T>> NumericSubtreeIndef;

} /* end of anon namespace */
//////////////////////////////////////////////////////////////////////////

extern "C"
void* spral_ssids_cpu_create_num_subtree_dbl(
      bool posdef,
      void const* symbolic_subtree_ptr,
      const double *const aval, // Values of A
      const double *const scaling, // Scaling vector (NULL if none)
      void** child_contrib, // Contributions from child subtrees
      struct cpu_factor_options const* options, // Options in
      ThreadStats* stats // Info out
      ) {
   auto const& symbolic_subtree = *static_cast<SymbolicSubtree const*>(symbolic_subtree_ptr);

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
void spral_ssids_cpu_destroy_num_subtree_dbl(bool posdef, void* target) {
   if(!target) return;

   if(posdef) {
      auto *subtree = static_cast<NumericSubtreePosdef*>(target);
      delete subtree;
   } else {
      auto *subtree = static_cast<NumericSubtreeIndef*>(target);
      delete subtree;
   }
}

/* Double precision wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_fwd_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
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

/* Double precision wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_diag_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
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

/* Double precision wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_diag_bwd_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
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

/* Double precision wrapper around templated routines */
extern "C"
Flag spral_ssids_cpu_subtree_solve_bwd_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
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

/* Double precision wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_enquire_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void const* subtree_ptr,// pointer to relevant type of NumericSubtree
      int* piv_order,   // pivot order, may be null, only used if indef
      double* d         // diagonal entries, may be null
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

/* Double precision wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_alter_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void* subtree_ptr,// pointer to relevant type of NumericSubtree
      double const* d   // new diagonal entries
      ) {

   assert(!posdef); // Should never be called on positive definite matrices.

   // Call method
   auto &subtree = *static_cast<NumericSubtreeIndef*>(subtree_ptr);
   subtree.alter(d);
}

/* Double precision wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_get_contrib_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void* subtree_ptr,// pointer to relevant type of NumericSubtree
      int* n,           // returned dimension of contribution block
      double const** val,     // returned pointer to contribution block
      int* ldval,       // leading dimension of val
      int const** rlist,      // returned pointer to row list
      int* ndelay,      // returned number of delays
      int const** delay_perm,  // returned pointer to delay values
      double const** delay_val,  // returned pointer to delay values
      int* lddelay      // leading dimension of delay_val
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

/* Double precision wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_free_contrib_dbl(
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
