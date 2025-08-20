/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  \version   Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 09:00 GMT
 */

#pragma once

#include <cstddef>

#include "ssids_routines.h"
#include "galahad_precision.h"

namespace galahad { namespace ssids { namespace cpu {

enum struct PivotMethod : ipc_ {
   app_aggressive = 1,
   app_block      = 2,
   tpp            = 3
};

enum struct FailedPivotMethod : ipc_ {
   tpp            = 1,
   pass           = 2
};

struct cpu_factor_control {
   ipc_ print_level;
   bool action;
   rpc_ small;
   rpc_ u;
   rpc_ multiplier;
   longc_ small_subtree_threshold;
   ipc_ cpu_block_size;
   PivotMethod pivot_method;
   FailedPivotMethod failed_pivot_method;
};

/** Return nearest value greater than supplied lda that is multiple of alignment */
template<typename T>
size_t align_lda(size_t lda) {
#if defined(__AVX512F__)
  ipc_ const align = 64;
#elif defined(__AVX__)
  ipc_ const align = 32;
#else
  ipc_ const align = 16;
#endif
   static_assert(align % sizeof(T) == 0, "Can only align if T divides align");
   ipc_ const Talign = align / sizeof(T);
   return Talign*((lda-1)/Talign + 1);
}

}}} /* namespaces galahad::ssids::cpu */
