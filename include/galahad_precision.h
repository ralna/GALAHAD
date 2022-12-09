/*
 * THIS VERSION: GALAHAD 4.1 - 2022-12-08 AT 07:05 GMT
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

// include guard
#ifndef GALAHAD_PRECISION_H
#define GALAHAD_PRECISION_H
#ifdef GALAHAD_SINGLE
typedef float real_wp_;  // working precision
typedef float real_sp_;   // single precision
#else
typedef double real_wp_;  // working precision
typedef float real_sp_;   // single precision
#endif

// end include guard
#endif
