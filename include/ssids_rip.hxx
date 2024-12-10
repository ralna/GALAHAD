/** \file
 *  \copyright 2024 GALAHAD productions
 *  \licence   BSD licence, see LICENCE file for details
 *  \author    Nick Gould
 *  \version   GALAHAD 5.1 - 2024-11-21 AT 10:30 GMT
 */

#include <stdint.h>

/* include guard */
#ifndef SSIDS_RIP_H
#define SSIDS_RIP_H

/* real precision employed */

#ifdef REAL_32
typedef float rpc_;
#elif REAL_128
typedef __float128 rpc_;
#else
typedef double rpc_;
#endif

/* integer storage employed */

#ifdef INTEGER_64
typedef int64_t ipc_;
typedef uint64_t uipc_;
#else
typedef int ipc_;
typedef unsigned int uipc_;
#endif

/* C long integer */

typedef int64_t longc_;

/* end include guard */
#endif
