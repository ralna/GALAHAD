/*
 * THIS VERSION: GALAHAD 4.1 - 2023-06-04 AT 12:05 GMT
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

// include guard
#ifndef GALAHAD_PRECISION_H
#define GALAHAD_PRECISION_H
/** `real_sp_` is real single precision */
typedef float real_sp_;   // single precision

#ifdef GALAHAD_SINGLE
/** `real_wp_` is real working precision */
typedef float real_wp_;  // working precision
#else
/** `real_wp_` is the real working precision used */
typedef double real_wp_;  // working precision
#endif

#ifdef INTEGER_64
typedef int ipc_;  // integer precision
#else
typedef int64_t ipc_;  // integer precision
#endif

// end include guard
#endif
