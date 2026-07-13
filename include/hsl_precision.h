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

/* rpc_ is also defined by galahad_precision.h */
#ifndef GALAHAD_RPC_DEFINED
#define GALAHAD_RPC_DEFINED
#ifdef REAL_32
typedef float rpc_;
#elif REAL_128
typedef __float128 rpc_;
#else
typedef double rpc_;
#endif
#endif

#ifdef REAL_32
#define f_rpc_ "f"
#elif REAL_128
#define f_rpc_ "Qf"
#else
#define f_rpc_ "lf"
#endif

// integer length

/* ipc_ / d_ipc_ are also defined by galahad_precision.h */
#ifndef GALAHAD_IPC_DEFINED
#define GALAHAD_IPC_DEFINED
#ifdef INTEGER_64
typedef int64_t ipc_;  // integer precision
#define d_ipc_ PRId64
#else
typedef int32_t ipc_;  // integer precision
#define d_ipc_ PRId32
#endif
#endif

// C long integer

typedef int64_t hsl_longc_;

// end include guard
#endif
