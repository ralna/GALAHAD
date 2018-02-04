/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 */
#pragma once

#include <cstddef>

namespace spral { namespace ssids { namespace cpu {

enum struct PivotMethod : int {
   app_aggressive = 1,
   app_block      = 2,
   tpp            = 3
};

enum struct FailedPivotMethod : int {
   tpp            = 1,
   pass           = 2
};

struct cpu_factor_options {
   int print_level;
   bool action;
   double small;
   double u;
   double multiplier;
   long small_subtree_threshold;
   int cpu_block_size;
   PivotMethod pivot_method;
   FailedPivotMethod failed_pivot_method;
};

/** Return nearest value greater than supplied lda that is multiple of alignment */
template<typename T>
size_t align_lda(size_t lda) {
#if defined(__AVX512F__)
  int const align = 64;
#elif defined(__AVX__)
  int const align = 32;
#else
  int const align = 16;
#endif
   static_assert(align % sizeof(T) == 0, "Can only align if T divides align");
   int const Talign = align / sizeof(T);
   return Talign*((lda-1)/Talign + 1);
}

}}} /* namespaces spral::ssids::cpu */
