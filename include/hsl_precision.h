/*
 * THIS VERSION: HSL SUBSET 1.0 - 2024-06-11 AT 08:25 GMT
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

// end include guard
#endif
