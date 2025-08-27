/** \file
 *  \copyright 2016 The Science and Technology Facilities Council (STFC)
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Jonathan Hogg
 *  Nick Gould, fork for GALAHAD 5.3 - 2025-08-17 AT 09:00 GMT
 *
 * \brief
 * Implements compatability functions depending on the value of autoconf macros.
 *  current version: 2025-08-27
 */
#include "ssids_compat.hxx"
#include "ssids_config.h"

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
