/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rename.h (modified by Nick Gould, STFC-RAL, 2025-03-01, to provide 
 *   64-bit integer support, additional prototypes for METIS_nodeND, 
 *   METIS_free and METIS_SetDefaultOptions, and to add _51 suffixes to 
 *   avoid possible conflicts)
 *
 * This file contains header files
 *
 * Started 10/2/97
 * George
 *
 * $Id: rename.h 13933 2013-03-29 22:20:46Z karypis $
 *
 */


#ifndef _LIBMETIS_RENAME_H_
#define _LIBMETIS_RENAME_H_

#ifdef REAL_32
#include "rename_51s.h"
#elif REAL_128
#include "rename_51q.h"
#else
#include "rename_51d.h"
#endif

#endif



