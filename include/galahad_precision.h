/*
 * THIS VERSION: GALAHAD 5.0 - 2024-06-11 AT 08:30 GMT
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD C INTERFACE PRECISION  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes
 *
 *  History -
 *   originally released GALAHAD Version 4.1. December 9th 2022
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include <stdint.h>
#include <inttypes.h>

// include guard
#ifndef GALAHAD_PRECISION_H
#define GALAHAD_PRECISION_H
/** `real_sp_` is real single precision */
typedef float real_sp_;   // single precision

#ifdef REAL_32
/** `real_wp_` is real working precision */
typedef float real_wp_;  // working precision
typedef float rpc_;  // working precision
#elif REAL_128
/** `real_wp_` is real working precision */
typedef __float128 real_wp_;  // working precision
typedef __float128 rpc_;  // working precision
#else
/** `real_wp_` is the real working precision used */
typedef double real_wp_;  // working precision
typedef double rpc_;  // working precision
#endif

typedef int64_t longc_;  // long integers

#ifdef INTEGER_64
typedef int64_t ipc_;  // integer type
typedef uint64_t uipc_;
#ifdef __cplusplus
#define d_ipc_ "lld"
#else
#define d_ipc_ PRId64
#endif
#else
typedef int32_t ipc_;  // integer type
typedef uint32_t uipc_;
#ifdef __cplusplus
#define d_ipc_ "d"
#else
#define d_ipc_ PRId32
#endif
#endif

// end include guard
#endif
