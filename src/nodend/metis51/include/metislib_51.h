/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * metis.h
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

#include <GKlib_51.h>

#if defined(ENABLE_OPENMP)
  #include <omp.h>
#endif


#include <metis_51.h>
#include <rename_51.h>
#include <gklib_defs_51.h>

#include <defs_51.h>
#include <struct_51.h>
#include <macros_51.h>
#include <proto_51.h>


#if defined(COMPILER_MSC)
#if defined(rint)
  #undef rint
#endif
#define rint(x) ((idx_t)((x)+0.5))  /* MSC does not have rint() function */
#endif

#endif
