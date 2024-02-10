//* \file hsl_mi28.h */

/*
 * THIS VERSION: GALAHAD 4.3 - 2024-02-10 AT 14:45 GMT.
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
   rpc_ alpha;
   bool check ;
   ipc_ iorder;
   ipc_ iscale;
   rpc_ lowalpha;
   ipc_ maxshift;
   bool rrt;
   rpc_ shift_factor;
   rpc_ shift_factor2;
   rpc_ small;
   rpc_ tau1;
   rpc_ tau2;
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
   rpc_ profile_before;
   rpc_ profile_after;
   long size_r;
   ipc_ stat;
   rpc_ alpha;
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
