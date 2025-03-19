/*
 * GKlib.h
 * 
 * George's library of most frequently used routines
 *
 * $Id: GKlib.h 13005 2012-10-23 22:34:36Z karypis $
 *
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


#include "gk_arch_51.h" /*!< This should be here, prior to the includes */


/*************************************************************************
* Header file inclusion section
**************************************************************************/
#include <stddef.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
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

#if defined(__WITHPCRE__)
  #include <pcreposix.h>
#else
  #if defined(USE_GKREGEX)
    #include "gk_regex_51.h"
  #else
    #include <regex.h>
  #endif /* defined(USE_GKREGEX) */
#endif /* defined(__WITHPCRE__) */

#if defined(__OPENMP__) 
#include <omp.h>
#endif

#define gk_cur_jbufs gk_cur_jbufs_51
#define gk_jbufs gk_jbufs_51
#define gk_jbuf gk_jbuf_51

#include <gk_types_51.h>
#include <gk_struct_51.h>
#include <gk_externs_51.h>
#include <gk_defs_51.h>
#include <gk_macros_51.h>
#include <gk_getopt_51.h>
#include <gk_mksort_51.h>
#include <gk_mkblas_51.h>
#include <gk_mkmemory_51.h>
#include <gk_mkpqueue_51.h>
#include <gk_mkpqueue2_51.h>
#include <gk_mkrandom_51.h>
#include <gk_mkutils_51.h>
#include <gk_proto_51.h>

#endif  /* GKlib.h */


