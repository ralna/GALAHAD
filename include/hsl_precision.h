/*
 * THIS VERSION: HSL SUBSET 1.1 - 2024-11-21 AT 08:35 GMT
 *
 *-*-*-*-*-*-*-*-*-  HSL SUBSET C INTERFACE PRECISION  *-*-*-*-*-*-*-*-*-*-
 *
 */

#include <stdint.h>

// include guard
#ifndef HSL_PRECISION_H
#define HSL_PRECISION_H

// real precision

#ifdef REAL_32
typedef float rpc_;
#define f_rpc_ "f" 
#elif REAL_128
typedef __float128 rpc_;
#define f_rpc_ "Qf"
#else
typedef double rpc_;
#define f_rpc_ "lf"
#endif

// integer length

#ifdef INTEGER_64
typedef int64_t ipc_;  // integer precision
#define d_ipc_ "ld"
#define i_ipc_ "li"
#else
typedef int ipc_;  // integer precision
#define d_ipc_ "d"
#define i_ipc_ "i"
#endif

// C long integer 

typedef int64_t hsl_longc_;

// end include guard
#endif
