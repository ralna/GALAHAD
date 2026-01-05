/*
 * THIS VERSION: HSL SUBSET 1.1 - 2024-11-21 AT 08:35 GMT
 *
 *-*-*-*-*-*-*-*-*-  HSL SUBSET C INTERFACE PRECISION  *-*-*-*-*-*-*-*-*-*-
 *
 */

#include <stdint.h>
#include <inttypes.h>

// include guard
#ifndef HSL_PRECISION_H
#define HSL_PRECISION_H

// real precision

#ifdef REAL_32
typedef float rpc_;
#define f_rpc_ "f"
#elif REAL_128
typedef _Float128 rpc_;
#define f_rpc_ "Qf"
#else
typedef double rpc_;
#define f_rpc_ "lf"
#endif

// integer length

#ifdef INTEGER_64
typedef int64_t ipc_;  // integer precision
#define d_ipc_ PRId64
#else
typedef int32_t ipc_;  // integer precision
#define d_ipc_ PRId32
#endif

// C long integer

typedef int64_t hsl_longc_;

// end include guard
#endif
