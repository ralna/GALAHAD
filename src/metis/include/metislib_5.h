/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * metis.h (very slightly modified by Nick Gould, STFC-RAL, 2024-03-21, below)
 *
 * This file includes all necessary header files
 *
 * Started 8/27/94
 * George
 *
 * $Id: metislib.h 10655 2011-08-02 17:38:11Z benjamin $
 */

#ifndef _LIBMETIS_METISLIB_H_
#define _LIBMETIS_METISLIB_H_

#include <GKlib_5.h>

#if defined(ENABLE_OPENMP)
  #include <omp.h>
#endif

#include <hsl_metis_5.h> /* changed from metis.h to allow extra hsl includes */
#include "rename_5.h" /* modified to provide unique 64-bit procedure names*/
#include "gklib_defs_5.h"

#include "defs_5.h"
#include "struct_5.h"
#include "macros_5.h"
#include "proto_5.h"


#if defined(COMPILER_MSC)
#if defined(rint)
  #undef rint
#endif
#define rint(x) ((idx_t)((x)+0.5))  /* MSC does not have rint() function */
#endif

#endif
