//* \file hsl_mi28.h */

/*
 * THIS VERSION: GALAHAD 4.0 - 2022-01-27 AT 09:00 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD MI28 C INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 3.3. November 27th 2021
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef HSL_MI28_H
#define HSL_MI28_H

// precision
#include "galahad_precision.h"

/* Derived type to hold control parameters for hsl_mi28 */
struct mi28_control {
   ipc_ f_arrays;
   real_wp_ alpha;
   bool check ;
   ipc_ iorder;
   ipc_ iscale;
   real_wp_ lowalpha;
   ipc_ maxshift;
   bool rrt;
   real_wp_ shift_factor;
   real_wp_ shift_factor2;
   real_wp_ small;
   real_wp_ tau1;
   real_wp_ tau2;
   ipc_ unit_error;
   ipc_ unit_warning;
};

/* Communucates errors and information to the user. */
struct mi28_info {
   ipc_ band_after;
   ipc_ band_before;
   ipc_ dup;
   ipc_ flag;
   ipc_ flag61;
   ipc_ flag64;
   ipc_ flag68;
   ipc_ flag77;
   ipc_ nrestart;
   ipc_ nshift;
   ipc_ oor;
   real_wp_ profile_before;
   real_wp_ profile_after;
   long size_r;
   ipc_ stat;
   real_wp_ alpha;
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
