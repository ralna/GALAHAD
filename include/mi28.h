 * Nick Gould for GALAHAD use, 2022-01-15

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
   real_wp_ alpha;
   bool check ;
   int iorder;
   int iscale;
   real_wp_ lowalpha;
   int maxshift;
   bool rrt;
   real_wp_ shift_factor;
   real_wp_ shift_factor2;
   real_wp_ small;
   real_wp_ tau1;
   real_wp_ tau2;
   int unit_error;
   int unit_warning;
};

/* Communucates errors and information to the user. */
struct mi28_info {
   int band_after;
   int band_before;
   int dup;
   int flag;
   int flag61;
   int flag64;
   int flag68;
   int flag77;
   int nrestart;
   int nshift;
   int oor;
   real_wp_ profile_before;
   real_wp_ profile_after;
   long size_r;
   int stat;
   real_wp_ alpha;
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
