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
