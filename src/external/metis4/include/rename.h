/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * rename.h
 *
 * This file contains header files
 *
 * Started 10/2/97
 * George
 *
 * modified by Nick Gould, STFC-RAL, 2025-03-0 to remove non metis_nd 
 * components, _4 suffix and 64-bit integer-support added to all procedures
 *
 * $Id: rename.h,v 1.1 1998/11/27 17:59:29 karypis Exp $
 *
 */

#ifndef _LIBMETIS_RENAME_H_
#define _LIBMETIS_RENAME_H_

#ifdef REAL_32
#include "rename_s.h"
#elif REAL_128
#include "rename_q.h"
#else
#include "rename_d.h"
#endif

#endif


