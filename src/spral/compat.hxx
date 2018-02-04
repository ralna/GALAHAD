/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *
 * \brief
 * Implements compatability functions depending on the value of autoconf macros.
 */
#pragma once

#include <cstdint>
#include <cstdio>

#include "config.h"

#ifndef HAVE_STD_ALIGN
// Older versions of g++ (and intel that relies on equivalent -lstdc++) don't
// define std::align, so we do it ourselves.
namespace std {
void* align(std::size_t alignment, std::size_t size, void*& ptr, std::size_t& space) noexcept;
} /* namespace std */
#endif /* HAVE_STD_ALIGN */

#ifndef _OPENMP
inline int omp_get_thread_num(void) { return 0; }
inline int omp_get_num_threads(void) { return 1; }
inline int omp_get_max_threads(void) { return 1; }
#endif /* _OPENMP */
