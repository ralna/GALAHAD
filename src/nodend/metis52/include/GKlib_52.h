/*
 * GKlib.h
 * 
 * George's library of most frequently used routines
 *
 * $Id: GKlib.h 14866 2013-08-03 16:40:04Z karypis $
 *
 * modified by Nick Gould for GALAHAD version 2024-04-12
 */

#ifndef _GKLIB_H_
#define _GKLIB_H_ 1

#define GKMSPACE

#if defined(_MSC_VER)
#define __MSC__
#endif
#if defined(__ICC)
#define __ICC__
#endif

#include "gk_arch_52.h" /*!< This should be here, prior to the includes */

/*************************************************************************
* Header file inclusion section
**************************************************************************/
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <memory.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>
#include <sys/stat.h>

#include <gk_rename_52.h>

#if defined(USE_GKREGEX)
#include "gk_regex_52.h"
#else
#include <regex.h>
#endif

#if defined(__OPENMP__) 
#include <omp.h>
#endif

#include <gk_types_52.h>
#include <gk_struct_52.h>
#include <gk_externs_52.h>
#include <gk_defs_52.h>
#include <gk_macros_52.h>
#include <gk_getopt_52.h>
#include <gk_mksort_52.h>
#include <gk_mkblas_52.h>
#include <gk_mkmemory_52.h>
#include <gk_mkpqueue_52.h>
#include <gk_mkpqueue2_52.h>
#include <gk_mkrandom_52.h>
#include <gk_mkutils_52.h>
#include <gk_proto_52.h>

#endif  /* GKlib.h */
