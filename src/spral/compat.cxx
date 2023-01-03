/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *
 * \brief
 * Implements compatability functions depending on the value of autoconf macros.
 */
#include "spral_compat.hxx"

#include "spral_config.h"

#ifdef GALAHAD_HAVE_HWLOC
#define HAVE_HWLOC 1
#else
#undef HAVE_HWLOC
#endif

#ifdef GALAHAD_HAVE_SCHED_GETCPU
#define HAVE_SCHED_GETCPU 1
#else
#undef HAVE_SCHED_GETCPU
#endif

// #ifndef HAVE_STD_ALIGN
// Older versions of g++ (and intel that relies on equivalent -lstdc++) don't
// define std::align, so we do it ourselves.
// If there is insufficient space, return nullptr
// namespace std {
// void* align(std::size_t alignment, std::size_t size, void*& ptr, std::size_t& space) noexcept {
//    auto cptr = reinterpret_cast<uintptr_t>(ptr);
//    auto pad = cptr % alignment;
//    if(pad == 0) return (size>space) ? nullptr : ptr;
//    pad = alignment - pad;
//    if(size+pad > space) return nullptr;
//    cptr += pad;
//    space -= pad;
//    ptr = reinterpret_cast<void*>(cptr);
//    return ptr;
// }
// } /* namespace std */
// #endif

